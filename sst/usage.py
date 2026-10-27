"""What Gemini costs: an estimate, today and this month, by feature (AI & models' cost card).

No API key can read the real spend or balance (only AI Studio's Billing page shows it; the Cloud Billing API needs
OAuth), so Rflow counts what it sends and gets back and prices it at Google's list prices (sst.costs):
  generateContent (speech on a Flash model, Gemini Transcribe)  usageMetadata: the prompt by modality (audio, text),
                                                                the answer, and the thinking, billed as output
  the OpenAI-compatible endpoint (cleanup, Text Transform,      usage: prompt_tokens in; total_tokens - prompt_tokens
  Translate)                                                    out (Gemini leaves the thinking out of completion_tokens)
  the Live API (Transcribe Live, Live Translate)                seconds of audio sent and received, 25 tokens a second
                                                                (sst.costs.LIVE_AUDIO); its usageMetadata is logged only
Live translation's cost saver also counts the seconds it didn't send ("unsent", never billed): the card shows them and
what they would have cost each way (sst.costs.live_rates), since Gemini's voice comes back 1:1 with what's sent.
Tokens and seconds are kept, not dollars, so a day is priced as it was billed (a price changing on a known day:
sst.costs.TEXT_FROM) and a corrected price reprices everything. Only Gemini is counted: own servers and this PC's
models cost nothing. Nothing said or written is kept, only numbers.

Each call is recorded once where it is made (a retry, a backup model and the hint-echo check are calls of their own),
with the feature its caller names (`feature()`), else the call site's own. The profile's file (Profile.usage_file):
    {"2026-10-23": {"dictation|gemini-3.5-transcribe": {"audio_in": 812, "text_in": 0, "out": 61, "thoughts": 0,
                                                        "audio_out": 0, "seconds": 32.5, "calls": 4, "unsent": 0},
                   ...}, ...}  (a file from before the cost saver has no "unsent": it loads as 0)
Saved off the UI thread, a few seconds after a change (save_after), and at once by flush(). Standard library and
sst.costs only: live translation (sst.live) records here too, and never imports dictation's modules.
"""
import contextlib
import contextvars
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sst import costs

FEATURES = {  # key -> what the card calls it, and whether its amount is minutes of audio (else calls)
    "dictation": ("Dictation", True),
    "live_dictation": ("Dictation as you speak", True),
    "cleanup": ("AI cleanup", False),
    "transform": ("Text Transform", False),
    "translate": ("Translate", False),
    "live_translation": ("Live translation", True),
    "tests": ("Tests and the Reading test", False),
}
COUNTS = ("audio_in", "text_in", "out", "thoughts", "audio_out", "seconds", "calls", "unsent")  # unsent: s, not billed
SAVE_AFTER = 5.0  # seconds after a change: a dictation's few calls are one write
KEEP_DAYS = 400  # older days are dropped when the file is saved: this month and last year's are what matter
BILLING = "https://aistudio.google.com/billing"  # Google's own figures (AI Studio's Billing page, the Gemini API docs)
CHARS_PER_TOKEN = 4  # an answer without usage is counted by its length

log = logging.getLogger(__name__)
_feature: contextvars.ContextVar[str] = contextvars.ContextVar("rflow_usage_feature", default="")


@contextlib.contextmanager
def feature(name: str):
    """The calls made within (on this thread) are for `name`. The outermost caller says: Text Transform run by a Test
    button is a test."""
    if _feature.get():
        yield
        return
    token = _feature.set(name)
    try:
        yield
    finally:
        _feature.reset(token)


def within(name: str, work: Callable, *args):
    """work(*args) with its calls counted for `name` (for a function run on another thread)."""
    with feature(name):
        return work(*args)


# ---------------------------------------------------------------- the price of a day's tokens

@dataclass(frozen=True)
class Rates:
    """USD per 1M tokens."""

    audio_in: float
    text_in: float
    out: float  # thinking included
    audio_out: float = 0.0


def rates(model: str, day: str) -> Rates | None:
    """A Gemini model's prices on a day (None: Rflow has none). A speech price first: it has the audio price."""
    if speech := costs.speech_price_on("gemini", model, day):
        if speech.per_minute:
            return None
        return Rates(speech.audio, speech.text or speech.audio, speech.output, speech.audio_out)
    if text := costs.text_price_on("gemini", model, day):
        return Rates(text.input, text.input, text.output)
    return None


def cost(model: str, day: str, counts: dict) -> float | None:
    """What these counts cost on that day, in USD (None: no price for the model)."""
    price = rates(model, day)
    if price is None:
        return None
    tokens = (counts.get("audio_in", 0) * price.audio_in + counts.get("text_in", 0) * price.text_in
              + (counts.get("out", 0) + counts.get("thoughts", 0)) * price.out + counts.get("audio_out", 0) * price.audio_out)
    return tokens / 1e6


# ---------------------------------------------------------------- what an answer says it used

def gemini_counts(answer: dict, seconds: float, transcribe: bool) -> dict:
    """A generateContent answer's usageMetadata as counts; without it, the audio by its length (Google's rates)."""
    meta = answer.get("usageMetadata") if isinstance(answer, dict) else None
    if not isinstance(meta, dict):
        rate = costs.TRANSCRIBE_AUDIO if transcribe else costs.FLASH_AUDIO
        return {"audio_in": round(seconds * rate), "seconds": seconds}
    prompt = _int(meta.get("promptTokenCount"))
    audio = sum(_int(d.get("tokenCount")) for d in meta.get("promptTokensDetails") or ()
                if isinstance(d, dict) and str(d.get("modality", "")).upper() == "AUDIO")
    return {"audio_in": audio, "text_in": max(0, prompt - audio), "out": _int(meta.get("candidatesTokenCount")),
            "thoughts": _int(meta.get("thoughtsTokenCount")), "seconds": seconds}


def chat_counts(answer: dict, sent: str, written: str) -> dict:
    """An OpenAI-compatible answer's usage as counts: billed output is total - prompt, since Gemini's endpoint leaves the
    thinking out of completion_tokens. Without usage, the lengths of what was sent and written."""
    use = answer.get("usage") if isinstance(answer, dict) else None
    if not isinstance(use, dict):
        return {"text_in": len(sent) // CHARS_PER_TOKEN, "out": len(written) // CHARS_PER_TOKEN}
    prompt, written_tokens = _int(use.get("prompt_tokens")), _int(use.get("completion_tokens"))
    total = _int(use.get("total_tokens")) or prompt + written_tokens
    return {"text_in": prompt, "out": written_tokens, "thoughts": max(0, total - prompt - written_tokens)}


def _int(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------- the ledger

@dataclass
class Line:
    """A feature's month: what it cost, its minutes of audio or its calls, and what the cost saver didn't send."""

    feature: str
    cost: float = 0.0
    seconds: float = 0.0
    calls: int = 0
    unsent: float = 0.0  # seconds live translation's cost saver didn't send
    saved: float = 0.0  # USD: those seconds at the Live API's prices in and out

    @property
    def title(self) -> str:
        return FEATURES.get(self.feature, (self.feature, False))[0]

    @property
    def amount(self) -> str:
        """"12 min" for audio ("72 min · 30 min not sent" with the cost saver), "34 calls" for the rest."""
        if FEATURES.get(self.feature, ("", False))[1]:
            heard = _minutes(self.seconds)
            return f"{heard} · {_minutes(self.unsent)} not sent" if self.unsent > 0 else heard
        return f"{self.calls:,} call" + ("" if self.calls == 1 else "s")


def _minutes(seconds: float) -> str:
    minutes = seconds / 60
    return "under 1 min" if 0 < minutes < 1 else f"{minutes:,.0f} min"


@dataclass
class Summary:
    today: float = 0.0
    month: float = 0.0
    lines: list[Line] = field(default_factory=list)  # this month's, in FEATURES' order; only those used
    unpriced: int = 0  # calls this month to a model Rflow has no price for (left out of the sums)
    saved: float = 0.0  # USD this month: what live translation's cost saver didn't send

    @property
    def used(self) -> bool:
        return bool(self.lines or self.unpriced)


class Ledger:
    """The profile's Gemini use, a day at a time. Thread-safe: calls are recorded from the threads that make them."""

    def __init__(self, path: Path | None, on_change: Callable[[], None] | None = None,
                 today: Callable[[], date] = date.today, save_after: float = SAVE_AFTER):
        self.path, self.on_change, self._today, self.save_after = path, on_change, today, save_after
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._dirty = False
        self.days: dict[str, dict[str, dict[str, float]]] = self._load()

    def _load(self) -> dict:
        if self.path is None:
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            log.warning("The cost tracker's %s is damaged (%s): kept as %s, counting again from now", self.path.name, e,
                        self.path.with_suffix(".damaged.json").name)
            with contextlib.suppress(OSError):
                self.path.replace(self.path.with_suffix(".damaged.json"))
            return {}
        days = {}
        for day, entries in (data.items() if isinstance(data, dict) else ()):
            if not (isinstance(day, str) and len(day) == 10 and isinstance(entries, dict)):
                continue
            for key, counts in entries.items():
                if isinstance(key, str) and "|" in key and isinstance(counts, dict):
                    days.setdefault(day, {})[key] = {name: _number(counts.get(name)) for name in COUNTS}
        return days

    def record(self, feature_: str, model: str, *, audio_in: float = 0, text_in: float = 0, out: float = 0,
               thoughts: float = 0, audio_out: float = 0, seconds: float = 0.0, calls: int = 1, unsent: float = 0) -> None:
        """One call (or a stretch of a live session: calls=0) to a Gemini model, for a feature; unsent: seconds the cost
        saver didn't send in it."""
        day = self._today().isoformat()
        key = f"{feature_}|{model.strip().removeprefix('models/')}"
        with self._lock:
            entry = self.days.setdefault(day, {}).setdefault(key, dict.fromkeys(COUNTS, 0.0))
            for name, value in (("audio_in", audio_in), ("text_in", text_in), ("out", out), ("thoughts", thoughts),
                                ("audio_out", audio_out), ("seconds", seconds), ("calls", calls), ("unsent", unsent)):
                entry[name] = entry.get(name, 0.0) + max(0.0, float(value or 0))
            self._dirty = True
            if self.path is not None and self._timer is None:
                self._timer = threading.Timer(self.save_after, self.flush)
                self._timer.daemon = True
                self._timer.start()
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:  # the window being rebuilt, say: never the call's problem
                log.exception("The cost card couldn't be told")

    def summary(self, day: date | None = None) -> Summary:
        """Today's and this month's estimate, and the month by feature."""
        today = (day or self._today()).isoformat()
        month = today[:7]
        with self._lock:
            days = {d: {k: dict(v) for k, v in entries.items()} for d, entries in self.days.items() if d[:7] == month}
        out, lines = Summary(), {}
        for d, entries in days.items():
            for key, counts in entries.items():
                name, model = key.split("|", 1)
                line = lines.setdefault(name, Line(name))
                line.seconds += counts.get("seconds", 0.0)
                line.calls += int(counts.get("calls", 0))
                if unsent := counts.get("unsent", 0.0):  # priced as it would have been billed that day, in and out
                    saved = unsent * sum(costs.live_rates(model, d))
                    line.unsent += unsent
                    line.saved += saved
                    out.saved += saved
                price = cost(model, d, counts)
                if price is None:
                    out.unpriced += int(counts.get("calls", 0)) or 1
                    continue
                line.cost += price
                out.month += price
                if d == today:
                    out.today += price
        order = list(FEATURES)
        out.lines = sorted(lines.values(), key=lambda line: order.index(line.feature) if line.feature in order else 99)
        return out

    def flush(self) -> None:
        """Write the file now (the timer's thread, or quitting and switching profiles)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if not self._dirty or self.path is None:
                return
            keep = sorted(self.days)[-KEEP_DAYS:]
            self.days = {d: self.days[d] for d in keep}
            data = json.dumps(self.days, separators=(",", ":"))
            self._dirty = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(self.path)
        except OSError as e:  # never a dictation's problem; tried again with the next change
            log.warning("Couldn't save the cost tracker: %s", e)
            with self._lock:
                self._dirty = True


def _number(value) -> float:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 and number == number and number != float("inf") else 0.0


# ---------------------------------------------------------------- the profile's ledger, for the call sites

_ledger: Ledger | None = None  # the app sets it for the profile in use; none (the CLI, tests) counts nothing


def use(ledger: Ledger | None) -> None:
    global _ledger
    old, _ledger = _ledger, ledger
    if old is not None and old is not ledger:
        old.flush()


def active() -> Ledger | None:
    return _ledger


def record(default: str, model: str, **counts) -> None:
    """A call to a Gemini model: for the feature its caller named (feature()), else `default`."""
    ledger = _ledger
    if ledger is None:
        return
    try:
        ledger.record(_feature.get() or default, model, **counts)
    except Exception:  # counting never fails a dictation
        log.exception("Couldn't count a call")
