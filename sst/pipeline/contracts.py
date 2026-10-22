"""The contracts between the stages of the voice pipeline (the owner's plan of 2026-10-01, "appropriate_plan").

    microphone -> ring buffer (pre-roll) -> session -> VAD -> chunker (pause / max, overlap)
    -> ASR workers (bounded, retried, validated) -> ordered results -> merge
    -> dictionary -> formatting -> LLM polish -> guard -> FinalText -> the existing paste

Each stage takes one of these objects and returns a new one, so every stage can be tested alone, a speech backend can be
swapped without touching the rest, and a wrong final text can be traced to the stage that made it (FinalText.stages).

Audio positions are sample indexes at the session's own rate (the microphone's: 16, 44.1 or 48 kHz), counted from the
session's first sample and never going backwards. Chunks are resampled to 16 kHz only when they go to a speech engine,
so no streaming resampler can smear a chunk boundary. Word times are seconds from the session's start.
"""
import enum
import itertools
import time
from dataclasses import dataclass, field

import numpy as np

ASR_RATE = 16_000  # what speech engines get


class SessionState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    FINALIZING_AUDIO = "finalizing_audio"  # the key is released; the tail and the last chunk are being cut
    WAITING_FOR_ASR = "waiting_for_asr"
    MERGING = "merging"
    DICTIONARY = "dictionary"
    FORMATTING = "formatting"
    LLM_POLISH = "llm_polish"
    GUARD = "guard"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChunkStatus(enum.Enum):
    CREATED = "created"
    QUEUED = "queued"
    SENT = "sent"
    RESULT_RECEIVED = "result_received"
    VALIDATED = "validated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"  # failed validation: retried
    FAILED = "failed"  # no usable result after every attempt (and no fallback): the session fails closed


class ErrorKind(enum.Enum):
    TRANSIENT = "transient"  # timeout, connection, HTTP 408/429/5xx: retried with backoff
    PERMANENT = "permanent"  # invalid request, bad key, unsupported audio: not retried
    SUSPICIOUS = "suspicious"  # the API answered, but the text looks wrong: one verification retry


class Provenance(enum.Enum):
    LLM_POLISHED = "llm_polished"  # the LLM's text passed the guard
    FORMATTED = "formatted"  # no LLM step (cleanup off)
    FORMATTED_FALLBACK = "formatted_fallback"  # the LLM failed or the guard rejected its change
    FAILED = "failed"  # the session could not be transcribed completely: nothing is typed


_ids = itertools.count(1)


def new_session_id() -> str:
    return f"sess_{time.strftime('%Y%m%d_%H%M%S')}_{next(_ids):03d}"


@dataclass
class AudioChunk:
    """A piece of one session's audio, cut at a pause or at the maximum length, sent to the speech engine as a whole.
    [start_sample, overlap_end_sample) repeats the end of the previous chunk (only to protect words cut at a forced
    boundary; it never appears twice in the final text)."""
    session_id: str
    sequence: int  # 1, 2, 3... in audio order: results are ordered by it, never by completion time
    audio: np.ndarray | None  # float32 mono at `rate`; released (None) once the result is accepted
    rate: int
    start_sample: int  # session-relative, at `rate`
    end_sample: int
    overlap_end_sample: int  # == start_sample when there is no overlap (the first chunk)
    speech_seconds: float  # how much of it the VAD heard as speech
    boundary: str = "end"  # why it ends: "pause", "max" (the maximum length was reached) or "end" (key released)
    status: ChunkStatus = ChunkStatus.CREATED
    replaces: int = 0  # the sequence of an earlier chunk this one repeats and supersedes (0: none); see Chunker.finish

    @property
    def chunk_id(self) -> str:
        return f"{self.session_id}#{self.sequence:03d}"

    @property
    def start(self) -> float:
        return self.start_sample / self.rate

    @property
    def end(self) -> float:
        return self.end_sample / self.rate

    @property
    def overlap_end(self) -> float:
        return self.overlap_end_sample / self.rate

    @property
    def duration(self) -> float:
        return (self.end_sample - self.start_sample) / self.rate


@dataclass
class WordInfo:
    text: str
    start: float  # seconds; chunk-relative from a backend, session-relative in a ChunkResult
    end: float
    confidence: float | None = None


@dataclass
class RawTranscript:
    """What a speech backend returns for one chunk: word times are relative to the chunk's first sample."""
    text: str
    words: list[WordInfo] = field(default_factory=list)  # empty when the engine gives no word times
    language: str = ""
    backend: str = ""  # e.g. "gemini-3.5-transcribe", "parakeet"
    diagnostics: dict = field(default_factory=dict)


@dataclass
class ChunkResult:
    """A chunk's accepted (or failed) transcript, in session time."""
    session_id: str
    sequence: int
    start: float  # the chunk's audio span, session seconds
    end: float
    overlap_end: float  # [start, overlap_end) is the overlap with the previous chunk
    text: str
    words: list[WordInfo] = field(default_factory=list)  # session-relative
    language: str = ""
    status: ChunkStatus = ChunkStatus.ACCEPTED
    retries: int = 0
    backend: str = ""
    note: str = ""  # e.g. "Parakeet stood in: Google Gemini: HTTP 503"
    error: str = ""  # why it failed (status FAILED)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    accepted: bool
    suspicion: float  # weighted sum of the signals; compared with the validator's threshold
    signals: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class MergedTranscript:
    text: str
    words: list[WordInfo] = field(default_factory=list)
    chunks: int = 0
    dedup: list[str] = field(default_factory=list)  # what each boundary removed, for diagnostics
    final: bool = False  # False: provisional (more chunks may come)


@dataclass
class DictionaryReplacement:
    start: int  # character span in the text before this stage
    end: int
    original: str
    replacement: str
    term: str  # the preferred term
    kind: str  # "case" (only the spelling/casing), "alias", "fuzzy"
    score: float = 1.0


@dataclass
class DictionaryResult:
    text: str
    replacements: list[DictionaryReplacement] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # fuzzy candidates seen but not trusted enough (diagnostics)


@dataclass
class FormatChange:
    kind: str  # "number", "date", "time", "currency", "percent", "unit", "ordinal", "email", "url"
    original: str
    replacement: str


@dataclass
class FormattedTranscript:
    text: str
    changes: list[FormatChange] = field(default_factory=list)


@dataclass
class ProtectedEntity:
    kind: str  # "NUMBER", "DATE", "TIME", "CURRENCY", "PERCENT", "UNIT", "EMAIL", "URL", "TERM", "WEEKDAY", "RELDATE"
    value: str  # as written in the text
    normalized: str  # what is compared ("5:30", "2026-10-01", "$25.50", "john@example.com", "postgresql"...)
    span: tuple[int, int] = (0, 0)


@dataclass
class GuardResult:
    accepted: bool  # True: the LLM's text may be used
    reasons: list[str] = field(default_factory=list)  # why it was rejected (or notes when accepted)
    change_ratio: float = 0.0
    diagnostics: dict = field(default_factory=dict)


@dataclass
class FinalText:
    text: str  # what is typed ("" when the session failed or nothing was said)
    session_id: str
    provenance: Provenance
    created_at: float = field(default_factory=time.time)
    stages: dict[str, str] = field(default_factory=dict)  # raw, merged, dictionary, formatted, polished, final
    guard: GuardResult | None = None
    error: str = ""  # why the session failed, or why the LLM's text wasn't used
    command: str = ""  # the dictation was a voice command (sst.commands): nothing is typed; the app carries it out
    notes: list[str] = field(default_factory=list)  # e.g. a chunk Parakeet transcribed instead of the cloud model
    metrics: dict = field(default_factory=dict)  # durations (ms) per stage, chunk counts, retries...


# ---------------------------------------------------------------- configuration: every tunable value lives here

@dataclass
class AudioConfig:
    pre_roll_ms: int = 2000  # kept from before the key press (the ring buffer's length)
    frame_ms: int = 20  # the VAD's frame


@dataclass
class VadConfig:
    speech_threshold: float = 0.6  # probability at which speech starts (hysteresis: on above this...)
    silence_threshold: float = 0.35  # ...and off below this
    min_speech_ms: int = 60  # shorter blips (clicks, a key press) don't start speech
    hangover_ms: int = 200  # speech continues this long after the level drops (word endings, short gaps)
    noise_floor_margin_db: float = 9.0  # how far above the adaptive noise floor speech sits
    absolute_floor_db: float = -62.0  # never call anything quieter than this speech


@dataclass
class ChunkingConfig:
    pause_boundary_ms: int = 1200  # a natural pause this long ends a chunk
    max_duration_ms: int = 20000  # a chunk is cut near this length even without a pause
    overlap_ms: int = 1000  # each chunk after the first starts this much before the previous one's cut
    boundary_grace_ms: int = 2000  # at the maximum, the cut goes to the quietest point within this window before it
    boundary_lookahead_ms: int = 400  # ... or slightly after it (a chunk is never longer than max + this)
    min_speech_ms: int = 150  # a chunk with less speech than this is not sent (silence, a cough)
    min_alone_speech_ms: int = 2000  # a last part with less of its own speech than this isn't sent alone: on the
    # owner's recordings every such part (1.4 s or less) came back from Gemini with text never said, often the Your-words
    # list (5 of 5; none of 5 with 2.1 s or more, 2026-10-02). It goes again with the part before it, as one chunk.
    min_cut_speech_ms: int = 4000  # a pause ends a part only after this much of its own speech. A 2.5 s fragment cut at
    # a mid-sentence pause came back rewritten ("Under some protocol..."): a part needs enough words for its context.
    # (Before: 400, which kept a breath alone, heard as "Yeah.", from making a part; this covers that too.)
    keep_silence_ms: int = 300  # silence kept at the end of a chunk cut at a pause
    join_context_ms: int = 3000  # a short last part goes with only the end of the part before it, enough for this much
    # speech in all, when the engine gives word times (they take the repeated words out); the rest of that part is already
    # transcribed. 0, or an engine without word times: with the whole part before it, as before phase 44.


@dataclass
class ASRConfig:
    max_in_flight: int = 2  # chunks transcribed at the same time (cloud); local engines run one at a time anyway
    timeout_ms: int = 30000  # one attempt
    max_attempts: int = 3  # transient errors: 1 + 2 retries
    backoff_ms: tuple[int, ...] = (250, 750, 1500)
    jitter: float = 0.2  # +-20% of each backoff
    suspicion_threshold: float = 1.0  # a result scoring this or more is retried once
    timestamp_mode: str = "timestamps"  # Gemini Transcribe: "timestamps" (word times) or "vocabulary" (custom words)


@dataclass
class DictionaryConfig:
    enabled: bool = True
    fuzzy_threshold: float = 0.88  # AUTOMATIC terms: similarity needed for a fuzzy replacement
    careful_threshold: float = 0.92  # CAREFUL terms: needed without context support
    context_threshold: float = 0.84  # CAREFUL terms: enough when a context word is near
    context_window: int = 4  # words on each side looked at for context


@dataclass
class FormattingConfig:
    enabled: bool = True
    number_style: str = "automatic"  # "automatic": structured values and numbers from ten up as digits; "off"
    date_style: str = "long"  # "October 1, 2026"
    time_style: str = "12h"  # "3:30 PM"; "24h": "15:30"
    currency_style: str = "symbol"  # "$25.50"
    unit_style: str = "abbreviated"  # "5 km"


@dataclass
class LLMConfig:
    enabled: bool = True  # only when AI cleanup has a model
    timeout_ms: int = 8000
    early: bool = True  # clean up the finished sentences while the user is still speaking, so that after the key release
    # only the rest waits for the AI (with the cleaned beginning as context). The guard still checks the whole text.


@dataclass
class GuardConfig:
    max_change_ratio: float = 0.6  # above this the change is looked at harder (never a rejection alone)
    max_added_words: int = 2  # new content words the LLM may add (e.g. a missing "the")
    semantic_check_enabled: bool = True  # question/command kept, nothing new asserted


@dataclass
class VoiceConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
