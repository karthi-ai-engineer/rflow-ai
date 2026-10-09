"""The orchestrator: one Session per key press, from the first audio sample to the FinalText that is typed.

    Session.feed(block)      while the key is held: the chunker cuts chunks and the ASR workers start on them at once
    Session.finish()         key released (after the tail): the last chunk goes out
    Session.result()         waits for the ordered results, then merge -> dictionary -> formatting -> LLM -> guard

Sessions are isolated: every chunk and result carries its session id, so a slow answer for session A can never land
in session B (sst.pipeline.asr). If a chunk can't be transcribed, the session fails closed: nothing is typed rather than
a text with a hole in it, and the caller keeps the recording.
"""
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sst import snippets
from sst.pipeline.asr import ASRScheduler, SessionFailed
from sst.pipeline.audio_stream import Chunker, trim_preroll
from sst.pipeline.contracts import (
    AudioChunk,
    ChunkResult,
    FinalText,
    MergedTranscript,
    Provenance,
    SessionState,
    VoiceConfig,
    new_session_id,
)
from sst.pipeline.merge import TranscriptMerger
from sst.snippets import Snippet

log = logging.getLogger(__name__)

RESULT_TIMEOUT = 120.0  # seconds to wait for the last ASR results after the key is released
BEEP_WINDOW = 0.6  # seconds after the key press in which the start beep is filtered out


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

    def transcribe(self, chunk: AudioChunk):
        with self._lock:
            if self._backend is None:
                self._backend = self._make()
        return self._backend.transcribe(chunk)


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
        self._merger = TranscriptMerger(config.chunking)
        self._merged: MergedTranscript | None = None
        self._asr = pipeline.scheduler.open_session(self.session_id, on_result=self._on_result,
                                                    keep_audio=pipeline.keep_audio)
        self._submitted = 0
        self._chunks: list[AudioChunk] = []  # for diagnostics only (their audio is released once accepted)
        self._finished = False
        self._fed = 0
        self._notch: _ToneNotch | None = None
        # The words heard, trimmed before everything else (the wake word's keyword and end phrase): set by the caller
        self.trim: Callable[[str], str] | None = None
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
        for chunk in self._chunker.finish():
            self._submit(chunk)
        self._asr.finish(self._submitted)
        self.state = SessionState.WAITING_FOR_ASR

    def cancel(self) -> None:
        self._finished = True
        self.state = SessionState.CANCELLED
        self._asr.cancel()

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
        before = (self._chunks[result.sequence - 1].replaces or result.sequence) - 1  # a replacement follows what the
        previous = self._chunks[before - 1] if before >= 1 else None  # part it replaces followed
        if previous is not None and previous.boundary == "pause":
            # Cut in a pause: the overlap holds only silence, so nothing in it can be a duplicate. Without word times
            # the merger couldn't tell, and a phrase the user really said twice across the pause would lose a copy.
            result.overlap_end = result.start
        self._merged = self._merger.add(result)

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
        merger = TranscriptMerger(self.pipeline.config.chunking)
        for result in results:
            merger.add(result)
        return merger.final()

    def text_result(self, text: str) -> FinalText:
        """The text stages for a text transcribed another way: the whole recording at once, after a part couldn't be
        transcribed (the speech model decoding a part to nothing). Complete, so not a text with a hole in it."""
        return self._text_stages(text, {"raw": text, "merged": text, "whole_recording": text}, [])

    def _text_stages(self, text: str, stages: dict[str, str], notes: list[str]) -> FinalText:
        s = self.pipeline.stages
        if self.trim is not None and text:
            try:  # before the snippets and the voice command: a keyword said first mustn't hide either
                text = self.trim(text)
            except Exception:  # typing a keyword is better than losing the dictation
                log.exception("%s: trimming the words heard failed; kept as heard", self.session_id)
            stages["trimmed"] = text
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
            self.state = SessionState.LLM_POLISH
            t0 = time.perf_counter()
            try:
                polished = s.llm.polish(trusted, terms)
            except Exception as e:  # unreachable, slow, refused: the trusted text is typed
                polished, error = None, str(e) or type(e).__name__
                log.warning("%s: LLM polish failed: %s", self.session_id, error)
            self._stage("llm", t0)
            if polished is None:
                provenance = Provenance.FORMATTED_FALLBACK
            else:
                stages["polished"] = polished
                self.state = SessionState.GUARD
                t0 = time.perf_counter()
                guard_result = s.guard.validate(trusted, polished, terms) if s.guard is not None else None
                self._stage("guard", t0)
                if guard_result is None or guard_result.accepted:
                    text, provenance = polished, Provenance.LLM_POLISHED
                else:
                    provenance = Provenance.FORMATTED_FALLBACK
                    log.warning("%s: guard kept the text before the LLM: %s", self.session_id,
                                "; ".join(guard_result.reasons))
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
