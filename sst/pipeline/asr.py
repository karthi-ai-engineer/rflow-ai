"""Speech recognition for the voice pipeline (the plan's sections 17 and 22-32; failure policy 89-91; sessions 75-77).

    AudioChunk -> ASRScheduler: a bounded worker pool shared by every session
               -> backend.transcribe (retried by error kind) -> ASRValidator (one verification retry when suspicious)
               -> fallback backend (when the primary one gave nothing usable) -> ChunkResult in session time
               -> SessionASR: results released in sequence order; wait() fails closed

A successful API call is not a good transcript: every answer is checked (empty, too much, a loop, a sentence out of
silence, broken word times) before it is trusted, and a chunk that can't be transcribed fails its whole session
instead of quietly leaving a hole in the text. Nothing here knows about a particular provider: a backend is anything
with `name` and `transcribe(chunk) -> RawTranscript`, and EngineBackend adapts the existing sst engines.
"""
import http.client
import json
import logging
import math
import random
import re
import threading
import time
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from sst.pipeline.contracts import (
    ASRConfig,
    AudioChunk,
    ChunkResult,
    ChunkStatus,
    ErrorKind,
    RawTranscript,
    ValidationResult,
    WordInfo,
)

log = logging.getLogger(__name__)

MAX_RETRY_AFTER = 5.0  # seconds: a server's Retry-After is honoured up to this; the user is waiting for the text


class ASRError(Exception):
    """A failure the backend has already classified, e.g. unsupported audio (PERMANENT) or an answer that is
    obviously not a transcript (SUSPICIOUS)."""

    def __init__(self, message: str, kind: ErrorKind, status: int = 0):
        super().__init__(message)
        self.kind, self.status = kind, status


def classify(exc: BaseException) -> ErrorKind:
    """Whether a failed attempt is worth repeating. Unknown errors count as transient: attempts are bounded, and
    giving up on a chunk fails the whole dictation. A backend marks an error it knows is final with `permanent = True`
    (sst.engines.cloud: no credit, a quota used up, a key not accepted, unreachable): the fallback stands in at once."""
    if isinstance(exc, ASRError):
        return exc.kind
    if not isinstance(exc, Exception):  # KeyboardInterrupt, SystemExit: never retried
        return ErrorKind.PERMANENT
    if getattr(exc, "permanent", None) is True:  # the backend knows asking again won't help (no credit, offline...)
        return ErrorKind.PERMANENT
    for attr in ("status", "status_code"):  # sst.engines.cloud.CloudError, urllib's HTTPError, other HTTP clients
        status = getattr(exc, attr, None)
        if isinstance(status, int) and not isinstance(status, bool) and (status == 0 or 400 <= status <= 599):
            # 0: no answer at all. 408 timeout, 425 too early, 429 rate limited, 5xx the server's trouble.
            return ErrorKind.TRANSIENT if status in (0, 408, 425, 429) or status >= 500 else ErrorKind.PERMANENT
    if isinstance(exc, json.JSONDecodeError | UnicodeDecodeError):  # a garbled answer (a proxy's error page), not a bad request
        return ErrorKind.TRANSIENT
    if isinstance(exc, OSError | http.client.HTTPException):  # TimeoutError and ConnectionError are OSErrors
        return ErrorKind.TRANSIENT
    if isinstance(exc, ValueError | TypeError | NotImplementedError):
        return ErrorKind.PERMANENT
    return ErrorKind.TRANSIENT


class ASRBackend(Protocol):
    """A speech recogniser for whole chunks. transcribe() returns word times relative to the chunk's first sample and
    raises on failure (ASRError, or anything classify() understands). It must keep to its own time limit
    (ASRConfig.timeout_ms): a worker thread can't be stopped from outside."""
    name: str

    def transcribe(self, chunk: AudioChunk) -> RawTranscript: ...


class EngineBackend:
    """An sst engine (sst.engines) as a backend. Engines with transcribe_chunk(audio, rate) -> RawTranscript (the
    cloud ones) give word times and raise on failure; the others give plain text, without word times."""

    def __init__(self, engine):
        self.engine = engine

    @property
    def name(self) -> str:
        return self.engine.name

    @property
    def title(self) -> str:
        return getattr(self.engine, "title", "") or self.engine.name

    def transcribe(self, chunk: AudioChunk) -> RawTranscript:
        if chunk.audio is None or not getattr(chunk.audio, "size", 0):
            raise ASRError(f"{chunk.chunk_id} has no audio to transcribe", ErrorKind.PERMANENT)
        by_chunk = getattr(self.engine, "transcribe_chunk", None)
        if callable(by_chunk):
            raw = by_chunk(chunk.audio, chunk.rate)
            if isinstance(raw, RawTranscript) and not raw.backend:  # anything else is refused by the scheduler
                raw.backend = self.name
            return raw
        return RawTranscript(self.engine.transcribe(chunk.audio, chunk.rate) or "", backend=self.name)


# ---------------------------------------------------------------- validation (plan §27-28, §90-91)

WEIGHTS = {  # what each signal adds to the suspicion score (compared with ASRConfig.suspicion_threshold)
    "empty": 1.5,  # speech was heard, nothing was recognised
    "silence_hallucination": 1.5,  # a sentence out of near silence ("Thank you for watching.")
    "excessive_rate": 1.0,  # more words or characters than anyone says in that much speech
    "extreme_rate": 0.0,  # far beyond that even over the chunk's whole length: decides failing, adds no suspicion
    "tiny": 0.4,  # one word for a long stretch of speech: possible ("yes", then a long pause), never enough alone
    "repetition": 1.5,  # a decoding loop
    "timestamps": 0.6,  # word times out of order or outside the chunk
}
FATAL = ("empty", "silence_hallucination", "repetition")  # at 1.0 or more after the verification retry: the chunk fails
MAX_WORDS_PER_SECOND = 6.0  # of speech; fast talkers manage 4-5
MAX_CHARS_PER_SECOND = 30.0
EXTREME_RATE = 3.0  # times those rates over the chunk's whole length: no VAD mistake explains that
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")  # written without spaces between words


class ASRValidator:
    """Checks an answer against the chunk it came from. Conservative: a short legitimate phrase ("yes", "OK,
    thanks.") always passes, and weak signals only count when others agree. A chunk whose speech_seconds is unknown
    (negative or not a number) is judged by its length alone, without the speech-based checks."""

    def __init__(self, config: ASRConfig):
        self.config = config

    def validate(self, chunk: AudioChunk, raw: RawTranscript) -> ValidationResult:
        text = (raw.text or "").strip() or " ".join(w.text for w in raw.words)
        tokens = [t for t in text.split() if any(c.isalnum() for c in t)]  # "...", "-" are not words
        words = sum(max(1.0, len(_CJK.findall(t)) / 2) for t in tokens)  # about two CJK characters a word
        chars = sum(len(t) for t in tokens)
        duration = max(chunk.duration, 0.0)
        speech = chunk.speech_seconds
        known = isinstance(speech, int | float) and math.isfinite(speech) and speech >= 0
        if known and duration:
            speech = min(speech, duration)
        signals, reasons = dict.fromkeys(WEIGHTS, 0.0), []
        if known and speech >= 0.5 and not tokens:
            signals["empty"] = 1.0 + min(1.0, (speech - 0.5) / 4)
            reasons.append(f"nothing recognised in {speech:.1f} s of speech")
        if known and speech < 0.25 and words >= 4:
            signals["silence_hallucination"] = 1.0 + min(1.0, (words - 4) / 10)
            reasons.append(f"{words:.0f} words from {speech:.2f} s of speech")
        excess = _excess(words, chars, speech if known else duration)
        if excess > 1:
            signals["excessive_rate"] = min(10.0, 2 * (excess - 1))
            reasons.append(f"{excess:.1f} times the plausible speaking rate")
            # The VAD may hear only part of a quiet voice; over the whole chunk this can only be made up text.
            signals["extreme_rate"] = float(_excess(words, chars, duration) >= EXTREME_RATE)
        if known and speech >= 8 and words <= 1:
            signals["tiny"] = min(1.0, speech / 20)
            reasons.append(f"{words:.0f} word(s) for {speech:.0f} s of speech")
        signals["repetition"], why = _repetition(text, tokens)
        if why:
            reasons.append(why)
        signals["timestamps"], why = _timestamp_problems(raw.words, duration)
        if why:
            reasons.append(why)
        suspicion = round(sum(WEIGHTS[k] * v for k, v in signals.items()), 3)
        return ValidationResult(suspicion < self.config.suspicion_threshold, suspicion, signals, "; ".join(reasons))

    @staticmethod
    def fatal(result: ValidationResult) -> bool:
        """Suspicion that must never be typed: nothing for real speech, a sentence out of silence, a loop, or text no
        one could have said in that time. Other suspicion is tolerated (and flagged) when a retry didn't clear it."""
        return bool(result.signals.get("extreme_rate")) or any(result.signals.get(k, 0.0) >= 1.0 for k in FATAL)


def _excess(words: float, chars: int, seconds: float) -> float:
    """How many times the plausible rate this is (1 = the limit); at least a second, so a quick "yes" never counts."""
    seconds = max(seconds, 1.0)
    return max(words / seconds / MAX_WORDS_PER_SECOND, chars / seconds / MAX_CHARS_PER_SECOND)


def _repetition(text: str, tokens: list[str]) -> tuple[float, str]:
    """A word or phrase (1-4 words) repeated back to back, or long text that compresses like a loop. A single word
    four or five times is only a medium signal: people do say "no, no, no, no"."""
    norm = [re.sub(r"[^\w']", "", t.lower()) for t in tokens]
    best, why = 0.0, ""
    for n in range(1, 5):
        r = _longest_run(norm, n)
        if n == 1:
            score = 1.0 + min(1.0, (r - 6) / 10) if r >= 6 else 0.5 if r >= 4 else 0.0
        else:
            score = 1.0 + min(1.0, (r - 4) / 6) if r >= 4 else 0.0
        if score > best:
            best, why = score, f"a {n}-word phrase repeated {r} times in a row"
    if len(text) >= 100:  # short text compresses badly whatever it says
        data = text.encode("utf-8")
        ratio = len(data) / len(zlib.compress(data))
        score = min(2.0, max(0.0, (ratio - 2.4) / 0.6))  # natural speech stays well under 2; loops go far above
        if score > best:
            best, why = score, f"the text compresses {ratio:.1f} times (repetitive)"
    return best, why


def _longest_run(tokens: list[str], n: int) -> int:
    """The most times an n-gram occurs back to back."""
    run, best = [1] * len(tokens), 1 if len(tokens) >= n else 0
    for i in range(len(tokens) - 2 * n, -1, -1):
        if tokens[i:i + n] == tokens[i + n:i + 2 * n]:
            run[i] = run[i + n] + 1
            best = max(best, run[i])
    return best


def _timestamp_problems(words: list[WordInfo], duration: float) -> tuple[float, str]:
    bad, previous_end = 0, None
    for w in words:
        start, end = w.start, w.end
        if not (_finite(start) and _finite(end)):
            bad += 1
            continue
        if (start > end + 0.01 or start < -0.5 or (duration and end > duration + 0.5)
                or (previous_end is not None and start < previous_end - 0.2)):
            bad += 1
        previous_end = end if previous_end is None else max(previous_end, end)
    if not bad:
        return 0.0, ""
    return min(1.0, 0.5 + bad / len(words)), f"{bad} of {len(words)} word times out of order or outside the chunk"


def _finite(x) -> bool:
    return isinstance(x, int | float) and math.isfinite(x)


# ---------------------------------------------------------------- ordering and sessions (plan §24, §32, §75-77, §89)

class OrderedResultBuffer:
    """Releases results in sequence order (1, 2, 3...), whatever order they finish in. Results already released or
    already held, and (when session_id is given) another session's, are ignored. Thread-safe."""

    def __init__(self, session_id: str | None = None, first: int = 1):
        self.session_id, self.next_expected = session_id, first
        self._held: dict[int, ChunkResult] = {}
        self._lock = threading.Lock()

    def push(self, result: ChunkResult) -> list[ChunkResult]:
        """The results this one makes releasable, in order (often none, sometimes several)."""
        with self._lock:
            if self.session_id is not None and result.session_id != self.session_id:
                return []
            if result.sequence < self.next_expected or result.sequence in self._held:
                return []
            self._held[result.sequence] = result
            out = []
            while self.next_expected in self._held:
                out.append(self._held.pop(self.next_expected))
                self.next_expected += 1
            return out

    def pending(self) -> int:
        return len(self._held)


class SessionFailed(Exception):
    """A chunk could not be transcribed, so the session has no complete text (fail closed: never chunk 1 + chunk 3).
    `results`: the accepted chunks; `failed`: the failed ones (their chunks keep their audio for a retry)."""

    def __init__(self, session_id: str, results: list[ChunkResult], failed: list[ChunkResult]):
        first = failed[0]
        super().__init__(f"{session_id}: {len(failed)} chunk(s) could not be transcribed "
                         f"(chunk {first.sequence}: {first.error})")
        self.session_id, self.results, self.failed = session_id, results, failed


class SessionCancelled(Exception):
    def __init__(self, session_id: str):
        super().__init__(f"{session_id} was cancelled")
        self.session_id = session_id


class SessionASR:
    """One session's chunks in the shared scheduler (open it with ASRScheduler.open_session).

    on_result(result) is called once per final result, in sequence order, from whichever worker thread completes the
    chunk that makes it releasable (one call at a time, never concurrently). A FAILED result is passed on too, and
    nothing after it: progressive text must not jump a gap. It must not call wait(). Results of a cancelled session,
    of another session, and repeated ones are dropped. The session keeps the backends it was opened with, so a model
    switch never mixes two recognisers in one text."""

    def __init__(self, scheduler: "ASRScheduler", session_id: str,
                 on_result: Callable[[ChunkResult], None] | None = None, keep_audio: bool = False):
        self.session_id, self.keep_audio = session_id, keep_audio
        self.backend, self.fallback = scheduler.backend, scheduler.fallback
        self._scheduler, self._on_result = scheduler, on_result
        self._buffer = OrderedResultBuffer(session_id)
        self._results: dict[int, ChunkResult] = {}
        self._submitted: set[int] = set()
        self._total: int | None = None
        self._done = 0
        self._cancelled = self._stopped = False
        self._cond = threading.Condition()
        self._deliver = threading.RLock()  # serialises on_result; cancel() takes it, so no call comes after it

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def submit(self, chunk: AudioChunk) -> None:
        """Queues a chunk without waiting. A repeated sequence number is ignored; after cancel() everything is."""
        if chunk.session_id != self.session_id:
            raise ValueError(f"{chunk.chunk_id} belongs to another session than {self.session_id}")
        if not isinstance(chunk.sequence, int) or chunk.sequence < 1:
            raise ValueError(f"{chunk.chunk_id}: sequence numbers start at 1")
        with self._cond:
            if self._cancelled:
                return
            if chunk.sequence in self._submitted:
                log.warning("%s was submitted twice; the second is ignored", chunk.chunk_id)
                return
            if self._total is not None and chunk.sequence > self._total:
                raise ValueError(f"{chunk.chunk_id}: the session has only {self._total} chunks")
            self._submitted.add(chunk.sequence)
            chunk.status = ChunkStatus.QUEUED  # before a worker can mark it SENT
        self._scheduler._enqueue(self, chunk)

    def finish(self, total: int) -> None:
        """No chunk after `total` will come (chunks 1..total may still be submitted)."""
        with self._cond:
            if total < 0 or (self._total is not None and total != self._total) or max(self._submitted, default=0) > total:
                raise ValueError(f"{self.session_id}: {total} chunks in all doesn't match what was submitted")
            self._total = total
            complete = self._done >= total
            self._cond.notify_all()
        if complete:
            self._scheduler._forget(self)

    def wait(self, timeout: float | None = None) -> list[ChunkResult]:
        """Every chunk's result in sequence order, once all are final (finish() must be called). Raises SessionFailed
        if any chunk failed, SessionCancelled after cancel(), TimeoutError when `timeout` seconds pass first."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._cancelled or (self._total is not None and self._done >= self._total),
                                       timeout):
                done, total = self._progress()
                raise TimeoutError(f"{self.session_id}: {done} of {total} chunks transcribed in time")
            if self._cancelled:
                raise SessionCancelled(self.session_id)
            results = [self._results[i] for i in range(1, self._total + 1)]
        failed = [r for r in results if r.status is ChunkStatus.FAILED]
        if failed:
            raise SessionFailed(self.session_id, [r for r in results if r.status is not ChunkStatus.FAILED], failed)
        return results

    def cancel(self) -> None:
        """Chunks not started are skipped, results still coming are dropped, wait() raises SessionCancelled."""
        with self._deliver, self._cond:
            if self._cancelled:
                return
            self._cancelled = True
            self._cond.notify_all()
        self._scheduler._forget(self)

    def progress(self) -> tuple[int, int]:
        """(final results, chunks): the second is the submitted count until finish() gives the total."""
        with self._cond:
            return self._progress()

    def _progress(self) -> tuple[int, int]:
        return self._done, self._total if self._total is not None else len(self._submitted)

    def _finish_chunk(self, result: ChunkResult) -> None:
        """A worker's final result for one chunk (plan §77: does the session still exist, is this chunk expected,
        is it already processed?)."""
        with self._deliver:
            with self._cond:
                if (self._cancelled or result.session_id != self.session_id or result.sequence not in self._submitted
                        or result.sequence in self._results):
                    return
                self._results[result.sequence] = result
            for ready in self._buffer.push(result):
                if self._cancelled or self._stopped:
                    break
                self._stopped = ready.status is ChunkStatus.FAILED
                if self._on_result is not None:
                    try:
                        self._on_result(ready)
                    except Exception:
                        log.exception("on_result failed for %s#%03d", ready.session_id, ready.sequence)
            with self._cond:
                self._done += 1
                complete = self._total is not None and self._done >= self._total
                self._cond.notify_all()
        if complete:
            self._scheduler._forget(self)


# ---------------------------------------------------------------- the scheduler (plan §22-23, §29-31)

class ASRScheduler:
    """Transcribes chunks in one bounded worker pool shared by every session (never 30 requests at once), so speech
    recognition overlaps with recording. Per chunk: transient errors are retried with backoff and jitter (through the
    injected `sleep`, which runs in the worker), permanent ones are not; a suspicious answer gets one verification
    retry; when the primary backend gives nothing usable, the fallback (e.g. Parakeet) gets one try, with a note.

    on_event(name, data) is called from worker threads: "asr_started", "asr_retrying", "asr_fallback",
    "asr_succeeded", "asr_failed"; data always has session_id, sequence and chunk_id."""

    def __init__(self, backend: ASRBackend, config: ASRConfig | None = None, fallback: ASRBackend | None = None,
                 on_event: Callable[[str, dict], None] | None = None, sleep: Callable[[float], None] = time.sleep,
                 rng: Callable[[], float] = random.random):
        self.backend, self.fallback, self.config = backend, fallback, config or ASRConfig()
        self.validator = ASRValidator(self.config)
        self.on_event, self._sleep, self._rng = on_event, sleep, rng
        self._pool = ThreadPoolExecutor(max_workers=max(1, self.config.max_in_flight), thread_name_prefix="asr")
        self._sessions: dict[str, SessionASR] = {}
        self._lock = threading.Lock()
        self._closed = False

    def open_session(self, session_id: str, on_result: Callable[[ChunkResult], None] | None = None,
                     keep_audio: bool = False) -> SessionASR:
        """keep_audio: chunks keep their audio after acceptance (debugging, the reading test); else it is released
        as soon as the chunk's text is safe (plan §31). A failed chunk always keeps it."""
        with self._lock:
            if self._closed:
                raise RuntimeError("the ASR scheduler is closed")
            if session_id in self._sessions:
                raise ValueError(f"{session_id} is already open")
            session = self._sessions[session_id] = SessionASR(self, session_id, on_result, keep_audio)
        return session

    def close(self, wait: bool = False) -> None:
        """Cancels every open session and stops the workers (`wait`: until the running chunks are done)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sessions = list(self._sessions.values())
        for session in sessions:
            session.cancel()
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "ASRScheduler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _enqueue(self, session: SessionASR, chunk: AudioChunk) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("the ASR scheduler is closed")
            self._pool.submit(self._run, session, chunk)

    def _forget(self, session: SessionASR) -> None:
        with self._lock:
            if self._sessions.get(session.session_id) is session:
                del self._sessions[session.session_id]

    def _run(self, session: SessionASR, chunk: AudioChunk) -> None:
        if session.cancelled:
            return  # skipped: cancelled while it waited for a worker
        try:
            result = self._process(session, chunk)
        except Exception as e:  # a bug must not leave wait() hanging: the chunk fails, and so does its session
            log.exception("Transcribing %s failed unexpectedly", chunk.chunk_id)
            chunk.status = ChunkStatus.FAILED
            result = self._failed(chunk, session.backend, {"attempts": 0, "durations_ms": [], "errors": []},
                                  f"internal error: {e!r}")
        if result is not None:
            session._finish_chunk(result)

    def _process(self, session: SessionASR, chunk: AudioChunk) -> ChunkResult | None:
        primary, fallback = session.backend, session.fallback
        diag = {"backend": primary.name, "attempts": 0, "durations_ms": [], "errors": []}
        self._emit("asr_started", chunk, backend=primary.name)
        missing = _missing_audio(chunk)
        if missing:  # nothing any backend could do: no attempt, no fallback
            return self._failed(chunk, primary, diag, f"{chunk.chunk_id} can't be transcribed: {missing}")
        used, note, verdict = primary, "", None
        raw, error = self._attempts(session, primary, chunk, diag, self.config.max_attempts)
        if raw is not None:
            raw, verdict, error = self._verified(session, primary, chunk, raw, diag)
        if raw is None and fallback is not None and not session.cancelled:
            log.info("%s: %s; %s stands in", chunk.chunk_id, error, _label(fallback))
            self._emit("asr_fallback", chunk, backend=fallback.name, reason=error)
            stand_in, fallback_error = self._attempts(session, fallback, chunk, diag, 1)
            if stand_in is not None:
                checked = self.validator.validate(chunk, stand_in)
                if checked.accepted or not self.validator.fatal(checked):
                    raw, verdict, used, note = stand_in, checked, fallback, f"{_label(fallback)} stood in: {error}"
                else:
                    error += f"; {_label(fallback)}: unusable result ({checked.reason})"
            else:
                error += f"; {fallback_error}"
        if session.cancelled:
            return None
        if raw is None:
            return self._failed(chunk, primary, diag, error)
        return self._accepted(session, chunk, raw, verdict, used, note, diag)

    def _attempts(self, session: SessionASR, backend: ASRBackend, chunk: AudioChunk, diag: dict,
                  attempts: int) -> tuple[RawTranscript | None, str]:
        """Up to `attempts` calls: transient errors are retried after a backoff, permanent ones end it at once.
        Returns (the answer, "") or (None, why there is none)."""
        error = "cancelled"
        for attempt in range(1, attempts + 1):
            if session.cancelled:
                return None, "cancelled"
            chunk.status = ChunkStatus.SENT
            t0 = time.perf_counter()
            try:
                raw = backend.transcribe(chunk)
                if isinstance(raw, str):  # a simple backend may answer with the text alone
                    raw = RawTranscript(raw, backend=backend.name)
                elif not isinstance(raw, RawTranscript) or not isinstance(raw.text, str):
                    what = type(raw.text if isinstance(raw, RawTranscript) else raw).__name__
                    raise ASRError(f"answered with {what}, not a transcript", ErrorKind.PERMANENT)
            except Exception as e:
                kind = classify(e)
                error = f"{_label(backend)}: {_describe(e)}"
                _record(diag, t0, f"{kind.value}: {error}")
                if kind is ErrorKind.PERMANENT or attempt == attempts:
                    log.warning("%s: %s (attempt %d, %s)", chunk.chunk_id, error, attempt, kind.value)
                    return None, error
                delay = self._backoff(attempt - 1, e) if kind is ErrorKind.TRANSIENT else 0.0
                self._emit("asr_retrying", chunk, backend=backend.name, attempt=attempt + 1, kind=kind.value,
                           reason=error, delay_ms=round(delay * 1000))
                if delay:
                    self._sleep(delay)
                continue
            _record(diag, t0)
            chunk.status = ChunkStatus.RESULT_RECEIVED
            return raw, ""
        return None, error

    def _verified(self, session: SessionASR, backend: ASRBackend, chunk: AudioChunk, raw: RawTranscript,
                  diag: dict) -> tuple[RawTranscript | None, ValidationResult, str]:
        """Validates an answer; a suspicious one gets exactly one verification retry, and the better of the two is
        kept. Returns (the answer to use or None, its validation, why it can't be used)."""
        verdict = self.validator.validate(chunk, raw)
        if not verdict.accepted:
            chunk.status = ChunkStatus.REJECTED
            self._emit("asr_retrying", chunk, backend=backend.name, attempt=diag["attempts"] + 1,
                       kind=ErrorKind.SUSPICIOUS.value, reason=verdict.reason, delay_ms=0)
            again, _ = self._attempts(session, backend, chunk, diag, 1)
            if again is not None:
                second = self.validator.validate(chunk, again)
                if self._rank(second) < self._rank(verdict):
                    raw, verdict = again, second
        if not verdict.accepted and self.validator.fatal(verdict):
            return None, verdict, f"{_label(backend)}: unusable result ({verdict.reason})"
        chunk.status = ChunkStatus.VALIDATED
        return raw, verdict, ""

    def _rank(self, verdict: ValidationResult) -> tuple:
        """Which of two answers is better: an accepted one, then a tolerable one, then the less suspicious."""
        return not verdict.accepted, self.validator.fatal(verdict), verdict.suspicion

    def _accepted(self, session: SessionASR, chunk: AudioChunk, raw: RawTranscript, verdict: ValidationResult,
                  backend: ASRBackend, note: str, diag: dict) -> ChunkResult:
        words = raw.words
        if all(_finite(w.start) and _finite(w.end) for w in words):
            words = [WordInfo(w.text, chunk.start + w.start, chunk.start + w.end, w.confidence) for w in words]
        else:  # broken times would mislead the overlap merge; it falls back on the text
            words, diag["words_dropped"] = [], "word times that aren't numbers"
        diag.update(backend=raw.backend or backend.name, validation=dict(verdict.signals), suspicion=verdict.suspicion)
        if not verdict.accepted:
            diag["suspicious"], diag["suspicion_reason"] = True, verdict.reason
        if raw.diagnostics:
            diag["backend_diagnostics"] = dict(raw.diagnostics)
        chunk.status = ChunkStatus.ACCEPTED
        if not session.keep_audio:
            chunk.audio = None  # the text is safe now: the audio isn't needed any more (plan §31)
        text = raw.text.strip() or " ".join(w.text for w in raw.words)  # what the validator judged
        result = ChunkResult(chunk.session_id, chunk.sequence, chunk.start, chunk.end, chunk.overlap_end,
                             text, words, raw.language, ChunkStatus.ACCEPTED,
                             max(0, diag["attempts"] - 1), raw.backend or backend.name, note, diagnostics=diag)
        self._emit("asr_succeeded", chunk, backend=result.backend, attempts=diag["attempts"],
                   ms=round(sum(diag["durations_ms"]), 1), suspicious=not verdict.accepted, note=note)
        return result

    def _failed(self, chunk: AudioChunk, backend: ASRBackend, diag: dict, error: str) -> ChunkResult:
        chunk.status = ChunkStatus.FAILED  # its audio stays, for a retry or recovery (plan §31, §89)
        self._emit("asr_failed", chunk, backend=backend.name, attempts=diag["attempts"], error=error)
        return ChunkResult(chunk.session_id, chunk.sequence, chunk.start, chunk.end, chunk.overlap_end, "",
                           status=ChunkStatus.FAILED, retries=max(0, diag["attempts"] - 1), backend=backend.name,
                           error=error, diagnostics=diag)

    def _backoff(self, retry: int, exc: BaseException) -> float:
        """Seconds before retry number `retry` (0-based): backoff_ms with +-jitter, or the server's Retry-After."""
        steps = self.config.backoff_ms or (0,)
        delay = steps[min(retry, len(steps) - 1)] / 1000 * (1 + self.config.jitter * (2 * self._rng() - 1))
        after = getattr(exc, "retry_after", None)  # seconds, when a backend passes the server's Retry-After on
        if _finite(after) and not isinstance(after, bool) and after > delay:
            delay = min(float(after), MAX_RETRY_AFTER)
        return max(0.0, delay)

    def _emit(self, name: str, chunk: AudioChunk, **data) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(name, {"session_id": chunk.session_id, "sequence": chunk.sequence,
                                 "chunk_id": chunk.chunk_id, **data})
        except Exception:
            log.exception("The ASR event handler failed on %s", name)


def _missing_audio(chunk: AudioChunk) -> str:
    if chunk.audio is None:
        return "its audio was already released"
    if not getattr(chunk.audio, "size", 0):
        return "its audio is empty"
    if not chunk.rate or chunk.rate <= 0:
        return f"invalid sample rate {chunk.rate}"
    return ""


def _record(diag: dict, t0: float, error: str = "") -> None:
    diag["attempts"] += 1
    diag["durations_ms"].append(round((time.perf_counter() - t0) * 1000, 1))
    if error:
        diag["errors"].append(error)


def _label(backend) -> str:
    return getattr(backend, "title", "") or backend.name


def _describe(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError) and not str(exc):
        return "no answer in time"
    return str(exc) or type(exc).__name__
