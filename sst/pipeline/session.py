"""The orchestrator: one Session per key press, from the first audio sample to the FinalText that is typed.

    Session.feed(block)      while the key is held: the chunker cuts chunks and the ASR workers start on them at once;
                             finished sentences are cleaned up (dictionary -> formatting -> LLM) as their parts come back
    Session.finish()         key released (after the tail): the last chunk goes out
    Session.result()         waits for the ordered results, then merge -> dictionary -> formatting -> LLM -> guard (the
                             LLM only for the rest after the sentences cleaned already; the guard on the whole text)

Sessions are isolated: every chunk and result carries its session id, so a slow answer for session A can never land
in session B (sst.pipeline.asr). If a chunk can't be transcribed, the session fails closed: nothing is typed rather than
a text with a hole in it, and the caller keeps the recording.
"""
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from sst import snippets
from sst.pipeline.asr import ASRScheduler, SessionFailed
from sst.pipeline.audio_stream import Chunker, trim_preroll
from sst.pipeline.contracts import (
    AudioChunk,
    ChunkingConfig,
    ChunkResult,
    FinalText,
    GuardResult,
    MergedTranscript,
    Provenance,
    SessionState,
    VoiceConfig,
    new_session_id,
)
from sst.pipeline.merge import Token, TranscriptMerger
from sst.snippets import Snippet

log = logging.getLogger(__name__)

RESULT_TIMEOUT = 120.0  # seconds to wait for the last ASR results after the key is released
BEEP_WINDOW = 0.6  # seconds after the key press in which the start beep is filtered out
WARM_ON = ("asr_retrying", "asr_fallback", "asr_failed")  # a cloud model failing: the local stand-in gets ready


class _ToneNotch:
    """A narrow notch (a biquad, Q 8) at one frequency over one span of a session's samples, kept continuous across
    blocks. It takes out a pure tone (the start beep) and leaves speech, whose energy is spread over many frequencies."""

    def __init__(self, rate: int, hz: float, start: int, end: int):
        w = 2 * np.pi * hz / rate
        alpha, cos = np.sin(w) / 16, np.cos(w)
        self.b = np.array([1, -2 * cos, 1]) / (1 + alpha)
        self.a = np.array([-2 * cos, 1 - alpha]) / (1 + alpha)
        self.start, self.end = start, end
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def __call__(self, block: np.ndarray, position: int) -> np.ndarray:
        lo, hi = max(self.start, position), min(self.end, position + len(block))
        if lo >= hi:
            return block
        block = block.copy()
        (b0, b1, b2), (a1, a2) = self.b, self.a
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        for i in range(lo - position, hi - position):  # 0.6 s once per dictation: a plain loop is fast enough
            x = float(block[i])
            y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, x, y1, y
            block[i] = y
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return block


class LazyBackend:
    """A speech backend made on first use: Parakeet as a cloud model's fallback, so a cloud-only user never loads it.
    If it can't be made (Parakeet isn't downloaded), the chunk fails and the session fails closed."""

    def __init__(self, name: str, make: Callable[[], object]):
        self.name, self._make, self._backend = name, make, None
        self._lock = threading.Lock()
        self._warming = False

    def transcribe(self, chunk: AudioChunk):
        with self._lock:
            if self._backend is None:
                self._backend = self._make()
        return self._backend.transcribe(chunk)

    def warm(self) -> None:
        """Make it now, in the background: the cloud model has just failed, so the next part may need the stand-in, and
        making it takes 7-12 s that a part waiting for it would add to the dictation. Never blocks. If it can't be made
        (Parakeet isn't downloaded: `make` says so without loading anything), nothing happens."""
        with self._lock:
            if self._backend is not None or self._warming:
                return
            self._warming = True

        def make() -> None:
            try:
                with self._lock:
                    if self._backend is None:
                        self._backend = self._make()
                log.info("%s is ready to stand in", self.name)
            except Exception as e:
                log.info("%s can't stand in: %s", self.name, e)
            finally:
                self._warming = False
        threading.Thread(target=make, name=f"{self.name}-warm", daemon=True).start()


SENTENCE_ENDS = ".!?…。！？"
_CLOSERS = "\"')]»”’」』"
_ABBREVIATIONS = frozenset("mr mrs ms dr prof st vs etc no approx dept est inc ltd jr sr fig".split())
SETTLED_AFTER = 12  # words after a sentence end before it counts as settled: the next part's merge (an overlap, a short
# last part sent with the end of this one) reaches only into the last few seconds of the text


def _sentence_end(token: str) -> bool:
    """A token that ends a sentence: "Tuesday.", "right?", "「はい。」"; not "Dr." or "e.g." (an abbreviation)."""
    word = token.rstrip(_CLOSERS)
    if not word or word[-1] not in SENTENCE_ENDS:
        return False
    stem = word.rstrip(SENTENCE_ENDS)
    return bool(stem) and stem.lower() not in _ABBREVIATIONS and not re.fullmatch(r"(?:\w\.)+\w", stem)


def settled_text(tokens: Sequence[Token]) -> str:
    """The merged text up to its last settled sentence end: one inside a part (a part's last word may carry a period
    only because the part ended there) with SETTLED_AFTER words after it. "" when there is none."""
    for i in range(len(tokens) - 1 - SETTLED_AFTER, -1, -1):
        if _sentence_end(tokens[i].text) and tokens[i + 1].chunk_end == tokens[i].chunk_end:
            return " ".join(t.text for t in tokens[:i + 1])
    return ""


class EarlyCleanup:
    """A dictation's finished sentences cleaned up while the user is still speaking (LLMConfig.early), so that after the
    key release only the rest waits for the AI.

    Each new stretch of settled text (settled_text) goes through the same stages as at the end, on its own thread:
    snippets checked, dictionary, formatting, then the LLM with the cleaned text before it as context
    (polish_continued), and the guard. Anything unexpected (an error, the guard refusing, a snippet's placeholder, text
    already cleaned changed by a later merge) stops it, and the session cleans the whole text at the end, as before.
    Session.result() uses the parts only if their text is exactly the beginning of the whole trusted text."""

    def __init__(self, stages: "Stages"):
        self.stages = stages
        self.parts: list[tuple[str, str]] = []  # (trusted text, cleaned text) of each stretch, in order
        self.ms = 0.0  # LLM time spent while the user was speaking
        self.off = False
        self._raw = self._target = ""  # merged text cleaned so far / settled so far
        self._busy = self._closed = False
        self._since = 0.0  # when the current stretch went to the LLM
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()

    def offer(self, tokens: Sequence[Token]) -> None:
        """New merged text (from an ASR worker): starts cleaning what is newly settled. Never waits."""
        target = settled_text(tokens)
        with self._lock:
            if self.off or self._closed or len(target) <= len(self._target):
                return
            self._target = target
            if self._busy:
                return
            self._busy = True
            self._idle.clear()
        threading.Thread(target=self._run, name="early-cleanup", daemon=True).start()

    def close(self, released: float, timeout: float) -> list[tuple[str, str]]:
        """No more stretches: the parts cleaned, once the one under way is done. One that went out after the key
        release is not waited for (the whole text is cleaned instead)."""
        with self._lock:
            self._closed = True
            late = self._busy and released and self._since > released
        if late or not self._idle.wait(timeout):
            self.off = True
        return [] if self.off else list(self.parts)

    def _run(self) -> None:
        while True:
            with self._lock:
                if self.off or self._closed or len(self._target) <= len(self._raw):
                    self._busy = False
                    self._idle.set()
                    return
                target, raw = self._target, self._raw
                self._since = time.monotonic()
            ok = target.startswith(raw)  # else a later merge changed text already cleaned
            try:
                ok = ok and self._clean(target[len(raw):].strip())
            except Exception as e:  # the AI unreachable, a bug in a stage: the end of the dictation does it all
                log.info("Early cleanup stopped: %s", e)
                ok = False
            with self._lock:
                self._raw, self.off = (target, self.off) if ok else (raw, True)

    def _clean(self, text: str) -> bool:
        s = self.stages
        mine = s.snippets()
        if mine and snippets.protect(text, mine)[1]:
            return False  # placeholders are numbered over the whole text: it is cleaned at the end
        if s.dictionary is not None:
            text = s.dictionary.correct(text).text
        if s.formatter is not None:
            text = s.formatter.format(text).text
        terms = [t for t in s.terms() if t.lower() in text.lower()]
        before = " ".join(cleaned for _, cleaned in self.parts)
        t0 = time.perf_counter()
        cleaned = s.llm.polish_continued(text, before, terms) if before else s.llm.polish(text, terms)
        self.ms += (time.perf_counter() - t0) * 1000
        if not cleaned.strip() or (s.guard is not None and not s.guard.validate(text, cleaned, terms).accepted):
            return False
        self.parts.append((text, cleaned))
        return True


def _covered(trusted: str, parts: list[tuple[str, str]]) -> str | None:
    """The rest of the trusted text after the parts cleaned early, or None when they aren't exactly its beginning."""
    whole, head = " ".join(trusted.split()), " ".join(" ".join(t for t, _ in parts).split())
    if not head or not (whole == head or whole.startswith(head + " ")):
        return None
    return whole[len(head):].strip()


def _merge_config(config: ChunkingConfig) -> ChunkingConfig:
    """The merge's text window reaches as far back as a short last part's lead-in may (Chunker._lead_in)."""
    return replace(config, overlap_ms=max(config.overlap_ms, 2 * getattr(config, "join_context_ms", 0)))


@dataclass
class Stages:
    """The optional text stages; None skips one. All are plain objects with one method (see each module)."""
    dictionary: object | None = None  # sst.pipeline.dictionary.DictionaryEngine: correct(text) -> DictionaryResult
    formatter: object | None = None  # sst.pipeline.formatting.Formatter: format(text) -> FormattedTranscript
    llm: object | None = None  # sst.pipeline.polish.LLMPolisher: polish(text, protected_terms) -> str
    guard: object | None = None  # sst.pipeline.guard.Guard: validate(before, after, terms) -> GuardResult
    terms: Callable[[], list[str]] = list  # the user's dictionary terms, protected through the LLM step
    command: Callable[[str], str | None] | None = None  # a voice command in the text heard (sst.commands), or None
    snippets: Callable[[], Sequence[Snippet]] = tuple  # the user's snippets (sst.snippets): "my email" -> their email


@dataclass
class Metrics:
    started: float = field(default_factory=time.monotonic)
    released: float = 0.0  # the key was let go
    first_speech_audio: float = 0.0
    first_asr_submit: float = 0.0
    stage_ms: dict[str, float] = field(default_factory=dict)
    chunks: int = 0
    retries: int = 0
    fallbacks: int = 0

    def as_dict(self) -> dict:
        data = {"stage_ms": {k: round(v, 1) for k, v in self.stage_ms.items()}, "chunks": self.chunks,
                "retries": self.retries, "fallbacks": self.fallbacks}
        if self.released:
            data["session_s"] = round(self.released - self.started, 2)
        if self.first_asr_submit:
            data["first_asr_after_s"] = round(self.first_asr_submit - self.started, 2)
        return data


class Session:
    """One dictation. feed() is called from one thread (the audio feeder), result() from another (the dictation
    worker); the ASR results arrive on the scheduler's worker threads."""

    def __init__(self, pipeline: "VoicePipeline", rate: int, preroll: np.ndarray | None = None,
                 beep_hz: float | None = None):
        self.pipeline, self.rate = pipeline, rate
        self.session_id = new_session_id()
        self.state = SessionState.RECORDING
        self.metrics = Metrics()
        config = pipeline.config
        self._chunker = Chunker(self.session_id, rate, config.chunking, config.vad, config.audio.frame_ms)
        self._merger = TranscriptMerger(_merge_config(config.chunking))
        self._merged: MergedTranscript | None = None
        self._timed = True  # every part so far came back with word times
        llm = pipeline.stages.llm
        self._early = EarlyCleanup(pipeline.stages) if getattr(config.llm, "early", False) and llm is not None \
            and callable(getattr(llm, "polish_continued", None)) else None
        self._asr = pipeline.scheduler.open_session(self.session_id, on_result=self._on_result,
                                                    keep_audio=pipeline.keep_audio)
        self._submitted = 0
        self._chunks: list[AudioChunk] = []  # for diagnostics only (their audio is released once accepted)
        self._finished = False
        self._fed = 0
        self._notch: _ToneNotch | None = None
        preroll = preroll if preroll is not None and len(preroll) else None
        # Keep the speech that runs into the key press, not a sentence said to someone two seconds earlier. `trimmed`:
        # samples of the take's pre-roll left out; the whole-recording paths leave them out too.
        self.trimmed = trim_preroll(preroll, rate, len(preroll), config.vad) if preroll is not None else 0
        if beep_hz:
            # Rflow's start beep reaches the microphone about 0.15 s after the key press, louder than the voice (39 of
            # the owner's 40 recordings): it counted as speech and set the level the audio is raised to.
            press = len(preroll) - self.trimmed if preroll is not None else 0
            self._notch = _ToneNotch(rate, beep_hz, press, press + int(BEEP_WINDOW * rate))
        if preroll is not None:
            self.feed(preroll[self.trimmed:])

    # -- audio in

    def feed(self, block: np.ndarray) -> None:
        if self._finished or not len(block):
            return
        block = np.asarray(block, dtype=np.float32)
        if self._notch is not None:
            block = self._notch(block, self._fed)
        self._fed += len(block)
        for chunk in self._chunker.push(block):
            self._submit(chunk)

    def finish(self) -> None:
        """The key was let go and its tail recorded: send what is left and expect no more chunks."""
        if self._finished:
            return
        self._finished = True
        self.metrics.released = time.monotonic()
        self.state = SessionState.FINALIZING_AUDIO
        # A short last part goes with only the end of the part before it when the engine's word times can take the
        # repeated words out: known from this engine's last answers (the pipeline's) and this session's.
        for chunk in self._chunker.finish(timed=bool(self._timed and self.pipeline.word_times)):
            self._submit(chunk)
        self._asr.finish(self._submitted)
        self.state = SessionState.WAITING_FOR_ASR

    def cancel(self) -> None:
        self._finished = True
        self.state = SessionState.CANCELLED
        self._asr.cancel()
        if self._early is not None:
            self._early.off = True

    def _submit(self, chunk: AudioChunk) -> None:
        self._submitted += 1
        self._chunks.append(chunk)
        if not self.metrics.first_asr_submit:
            self.metrics.first_asr_submit = time.monotonic()
        log.info("%s: chunk %d %.1f-%.1f s (%s, %.1f s of speech)", self.session_id, chunk.sequence, chunk.start,
                 chunk.end, chunk.boundary, chunk.speech_seconds)
        self._asr.submit(chunk)

    def _on_result(self, result: ChunkResult) -> None:
        # Called in sequence order by the scheduler: the provisional transcript grows while the user is still talking.
        chunk = self._chunks[result.sequence - 1]
        before = (chunk.replaces or result.sequence) - 1  # a replacement follows what the part it replaces followed
        previous = self._chunks[before - 1] if before >= 1 else None
        if previous is not None and previous.boundary == "pause" and chunk.start_sample >= previous.end_sample:
            # Cut in a pause: the overlap holds only silence, so nothing in it can be a duplicate. Without word times
            # the merger couldn't tell, and a phrase the user really said twice across the pause would lose a copy.
            # (A short last part that starts inside the part before it repeats its end: that overlap is real.)
            result.overlap_end = result.start
        if result.text.strip():
            self._timed = self._timed and bool(result.words)
            if not result.note:  # the engine's own answer, not a stand-in's
                self.pipeline.word_times = bool(result.words)
        self._merged = self._merger.add(result)
        if self._early is not None and not self._finished:  # after the release the whole rest is cleaned at once
            self._early.offer(self._merger.tokens)

    # -- text out

    def result(self, timeout: float = RESULT_TIMEOUT) -> FinalText:
        """Wait for every chunk, then the text stages. Never raises: a failure becomes a FAILED FinalText."""
        stages: dict[str, str] = {}
        notes: list[str] = []
        try:
            t0 = time.perf_counter()
            try:
                results, failure = self._asr.wait(timeout), None
            except SessionFailed as e:
                results, failure = e.results, e
            # Read only now: the app asks for the result the moment the recording ends, while its feeder thread may still
            # be cutting the last part, the one that replaces the part before it. wait() returns after finish(), so every
            # chunk is known here; read before it, both parts were typed (the owner's dictation, 2026-10-04).
            replaced = {c.replaces for c in self._chunks if c.replaces}  # a short rest sent again with the part before it
            if failure is not None and (not replaced or any(r.sequence not in replaced for r in failure.failed)):
                raise failure  # else only a superseded part failed: its replacement holds the same words
            results = [r for r in results if r.sequence not in replaced]
            self._stage("asr_wait", t0)
            notes += [r.note for r in results if r.note]
            self.metrics.chunks = len(results)
            self.metrics.retries = sum(r.retries for r in results)
            self.metrics.fallbacks = sum(1 for r in results if r.note)
            stages["raw"] = " | ".join(r.text for r in results)
            self.state = SessionState.MERGING
            t0 = time.perf_counter()
            merged = self._merger.final() if not replaced else self._merge(results)
            self._stage("merge", t0)
            stages["merged"] = merged.text
            return self._text_stages(merged.text, stages, notes)
        except SessionFailed as e:
            failed = ", ".join(f"part {r.sequence}: {r.error or 'no usable text'}" for r in e.failed)
            return self._fail(f"part of the dictation couldn't be transcribed ({failed})", stages)
        except TimeoutError:
            self.cancel()
            return self._fail("the speech service took too long", stages)
        except Exception as e:  # a bug in a stage must not lose the recording: the caller keeps it
            log.exception("%s failed", self.session_id)
            return self._fail(f"{type(e).__name__}: {e}", stages)

    def _merge(self, results: list[ChunkResult]) -> MergedTranscript:
        """The final transcript from these results alone (the progressive one also holds a superseded part)."""
        merger = TranscriptMerger(_merge_config(self.pipeline.config.chunking))
        for result in results:
            merger.add(result)
        return merger.final()

    def text_result(self, text: str) -> FinalText:
        """The text stages for a text transcribed another way: the whole recording at once, after a part couldn't be
        transcribed (the speech model decoding a part to nothing). Complete, so not a text with a hole in it."""
        return self._text_stages(text, {"raw": text, "merged": text, "whole_recording": text}, [])

    def _text_stages(self, text: str, stages: dict[str, str], notes: list[str]) -> FinalText:
        s = self.pipeline.stages
        heard, mine = text, s.snippets() if text else ()
        if mine and (snippet := snippets.alone(text, mine)) is not None:
            # "my email": the user's own text, exactly as entered; it wins over a voice command with the same words
            log.info("%s: snippet '%s'", self.session_id, snippet.cue)
            self.state = SessionState.READY
            final = FinalText(snippet.text, self.session_id, Provenance.FORMATTED, stages=stages, notes=notes,
                              metrics=self._metrics())
            self.pipeline.record(final, self._chunks)
            return final
        if s.command is not None and text and (command := s.command(text)):
            # "make it concise": a command for Text Transform, checked on the words heard, before any cleanup
            self.state = SessionState.READY
            final = FinalText("", self.session_id, Provenance.FORMATTED, stages=stages, notes=notes,
                              metrics=self._metrics(), command=command)
            self.pipeline.record(final, self._chunks)
            return final
        text, slots = snippets.protect(text, mine) if mine else (text, {})
        if slots:  # a cue inside the sentence: a placeholder until the very end, kept by the AI like code
            stages["snippets"] = text
        if s.dictionary is not None and text:
            self.state = SessionState.DICTIONARY
            t0 = time.perf_counter()
            text = s.dictionary.correct(text).text
            self._stage("dictionary", t0)
        stages["dictionary"] = text
        if s.formatter is not None and text:
            self.state = SessionState.FORMATTING
            t0 = time.perf_counter()
            text = s.formatter.format(text).text
            self._stage("formatting", t0)
        stages["formatted"] = trusted = text
        provenance, guard_result, error = Provenance.FORMATTED, None, ""
        if s.llm is not None and trusted.strip():
            terms = [t for t in s.terms() if t.lower() in trusted.lower()] + list(slots)
            early = self._early_parts(s, trusted, slots)
            polished, guard_result, error = self._polish(s, trusted, terms, early, stages)
            if early and polished is None and guard_result is not None:
                # Cleaned in parts, refused as a whole: the whole text in one go, as without the early cleanup
                log.info("%s: the guard refused the text cleaned in parts; cleaning it whole", self.session_id)
                polished, guard_result, error = self._polish(s, trusted, terms, None, stages)
            if polished is not None:
                text, provenance = polished, Provenance.LLM_POLISHED
            else:
                provenance = Provenance.FORMATTED_FALLBACK
        text = " ".join(text.split())
        if slots:  # the snippets' own text, last: no stage touches it, line breaks included
            text = snippets.expand(text, slots) or snippets.expand(" ".join(trusted.split()), slots) \
                or " ".join(heard.split())
        stages["final"] = text
        self.state = SessionState.READY
        final = FinalText(text, self.session_id, provenance, stages=stages, guard=guard_result, error=error, notes=notes,
                          metrics=self._metrics())
        self.pipeline.record(final, self._chunks)
        return final

    def _early_parts(self, s: Stages, trusted: str, slots: dict) -> list[tuple[str, str]] | None:
        """The parts cleaned while the user spoke, when they are exactly the beginning of this trusted text; else None.
        Only for a dictation of two parts or more: one part is cleaned as a whole, as before."""
        early = self._early
        if early is None:
            return None
        parts = early.close(self.metrics.released, self.pipeline.config.llm.timeout_ms / 1000)
        if early.ms:
            self.metrics.stage_ms["llm_early"] = early.ms
        if not parts or slots or s is not early.stages or self.metrics.chunks < 2 or _covered(trusted, parts) is None:
            return None
        return parts

    def _polish(self, s: Stages, trusted: str, terms: list[str], early: list[tuple[str, str]] | None,
                stages: dict[str, str]) -> tuple[str | None, GuardResult | None, str]:
        """The LLM's text and the guard's verdict on it: (the text if the guard took it, the verdict, why not).
        `early`: the parts cleaned already; only the rest goes to the LLM, after them."""
        self.state = SessionState.LLM_POLISH
        t0 = time.perf_counter()
        try:
            if early:
                before = " ".join(cleaned for _, cleaned in early)
                rest = _covered(trusted, early)
                rest_terms = [t for t in terms if t.lower() in rest.lower()]
                polished = before + " " + s.llm.polish_continued(rest, before, rest_terms) if rest else before
                stages["early"] = before
            else:
                polished = s.llm.polish(trusted, terms)
        except Exception as e:  # unreachable, slow, refused: the trusted text is typed
            error = str(e) or type(e).__name__
            log.warning("%s: LLM polish failed: %s", self.session_id, error)
            self._stage("llm", t0)
            return None, None, error
        self._stage("llm", t0)
        stages["polished"] = polished
        self.state = SessionState.GUARD
        t0 = time.perf_counter()
        guard_result = s.guard.validate(trusted, polished, terms) if s.guard is not None else None
        self._stage("guard", t0)
        if guard_result is None or guard_result.accepted:
            return polished, guard_result, ""
        log.warning("%s: guard kept the text before the LLM: %s", self.session_id, "; ".join(guard_result.reasons))
        return None, guard_result, ""

    def _fail(self, error: str, stages: dict[str, str]) -> FinalText:
        self.state = SessionState.FAILED
        log.warning("%s failed: %s", self.session_id, error)
        final = FinalText("", self.session_id, Provenance.FAILED, stages=stages, error=error, metrics=self._metrics())
        self.pipeline.record(final, self._chunks)
        return final

    def _stage(self, name: str, t0: float) -> None:
        self.metrics.stage_ms[name] = (time.perf_counter() - t0) * 1000

    def _metrics(self) -> dict:
        data = self.metrics.as_dict()
        if self.metrics.released:
            data["release_to_final_s"] = round(time.monotonic() - self.metrics.released, 2)  # what the user waits
        return data


class VoicePipeline:
    """One per app (rebuilt when the speech model, the cleanup or the dictionary changes). Thread-safe: sessions may
    overlap (a new dictation while the previous one's ASR is still finishing)."""

    def __init__(self, scheduler: ASRScheduler, config: VoiceConfig | None = None, stages: Stages | None = None,
                 debug_dir: Path | None = None):
        self.scheduler, self.config = scheduler, config or VoiceConfig()
        self.stages = stages or Stages()
        self.debug_dir = debug_dir  # set: every session's stages (and chunk audio) are kept there for debugging
        self.keep_audio = debug_dir is not None
        self.word_times: bool | None = None  # whether the speech engine's last answer had word times (None: no answer yet)
        fallback = getattr(scheduler, "fallback", None)
        if callable(getattr(fallback, "warm", None)):
            # The local stand-in (LazyBackend) starts loading at the cloud model's first failure, not when a part
            # needs it: loading takes 7-12 s, which a part waiting for it would add to the dictation.
            inner = scheduler.on_event

            def on_event(name: str, data: dict) -> None:
                if name in WARM_ON and data.get("kind") != "suspicious":  # "suspicious": the cloud did answer
                    fallback.warm()
                if inner is not None:
                    inner(name, data)
            scheduler.on_event = on_event

    def start(self, rate: int, preroll: np.ndarray | None = None, beep_hz: float | None = None) -> Session:
        return Session(self, rate, preroll, beep_hz)

    def process_audio(self, audio: np.ndarray, rate: int, timeout: float = RESULT_TIMEOUT) -> FinalText:
        """A whole recording at once (a retry, the tests, the reading test): the same stages as live dictation."""
        session = self.start(rate)
        step = max(1, rate // 10)
        for i in range(0, len(audio), step):
            session.feed(audio[i:i + step])
        session.finish()
        return session.result(timeout)

    def close(self) -> None:
        self.scheduler.close()

    def record(self, final: FinalText, chunks: list[AudioChunk]) -> None:
        stages = " -> ".join(f"{k}: {v!r}" for k, v in final.stages.items() if k in ("merged", "final"))
        log.info("%s %s in %s; %s", final.session_id, final.provenance.value, final.metrics, stages)
        if self.debug_dir is None:
            return
        folder = self.debug_dir / final.session_id
        try:
            folder.mkdir(parents=True, exist_ok=True)
            data = asdict(final)
            data["provenance"] = final.provenance.value
            data["chunks"] = [{"sequence": c.sequence, "start": c.start, "end": c.end, "overlap_end": c.overlap_end,
                               "boundary": c.boundary, "speech_seconds": c.speech_seconds, "status": c.status.value}
                              for c in chunks]
            (folder / "session.json").write_text(json.dumps(data, indent=1, default=str, ensure_ascii=False),
                                                 encoding="utf-8")
            from sst.audio import save_wav
            for c in chunks:
                if c.audio is not None and len(c.audio):
                    save_wav(folder / f"chunk_{c.sequence:03d}.wav", c.audio, c.rate)
        except OSError as e:
            log.warning("Could not keep the debug files of %s: %s", final.session_id, e)
