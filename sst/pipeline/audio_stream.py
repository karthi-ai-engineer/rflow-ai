"""The audio stages of the voice pipeline: the microphone's ring buffer (the pre-roll), a local voice activity detector,
the chunker that cuts a session's audio at pauses or at the maximum length (with overlap), and the pre-roll trim.

Positions are sample indexes at the session's own rate, counted from its first sample, and every decision is taken at a
VAD frame boundary on that fixed grid: a cut lands on the same sample whether the audio came in one block or a thousand.
Nothing here touches the microphone, the network or the disk. The audio callback only writes to the RingBuffer; the
Chunker runs on a worker thread.
"""
import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from sst.pipeline.contracts import AudioChunk, ChunkingConfig, VadConfig


def _mono(samples) -> np.ndarray:
    """1-D float32. A sounddevice block is (frames, channels): one channel is taken as is, several are averaged."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 0:
        return x.reshape(1)
    x = x.reshape(len(x), -1)
    return x[:, 0] if x.shape[1] == 1 else x.mean(axis=1, dtype=np.float32)


# ---------------------------------------------------------------- ring buffer

class RingBuffer:
    """The newest `capacity` samples (float32 mono), for the pre-roll: the moment before the key press.

    Positions are monotonic: write_index counts every sample ever written and never goes back, so a reader names audio
    by absolute index (e.g. "from the key press on") however often the storage has wrapped. One writer (the audio
    callback) and any number of readers on other threads: the lock is held only while samples are copied, so the
    callback never waits long (plan section 73)."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("a ring buffer needs room for at least one sample")
        self.capacity = int(capacity)
        self._data = np.zeros(self.capacity, dtype=np.float32)  # allocated once: the callback never allocates storage
        self._written = 0
        self._lock = threading.Lock()

    @property
    def write_index(self) -> int:
        """Samples written so far: the absolute index the next sample will get."""
        return self._written

    @property
    def oldest_index(self) -> int:
        """The absolute index of the oldest sample still held."""
        return max(0, self._written - self.capacity)

    def __len__(self) -> int:
        return min(self._written, self.capacity)

    def write(self, samples) -> None:
        samples = _mono(samples)
        n = len(samples)
        if not n:
            return
        tail = samples[-self.capacity:]  # a write longer than the buffer: only its end can be kept
        with self._lock:
            end = self._written + n
            pos = (end - len(tail)) % self.capacity
            first = min(len(tail), self.capacity - pos)
            self._data[pos:pos + first] = tail[:first]
            self._data[:len(tail) - first] = tail[first:]  # the rest wraps to the front
            self._written = end

    def snapshot_last(self, n: int) -> np.ndarray:
        """A copy of the newest min(n, held) samples."""
        with self._lock:
            n = max(0, min(int(n), self._written, self.capacity))
            return self._copy(self._written - n, self._written)

    def get_range(self, start: int, end: int) -> np.ndarray:
        """A copy of absolute samples [start, end), clipped to what is still held (overwritten or future samples are
        simply not there: compare the length, or oldest_index, when it matters)."""
        with self._lock:
            lo, hi = max(int(start), self._written - self.capacity, 0), min(int(end), self._written)
            return self._copy(lo, hi) if hi > lo else np.zeros(0, dtype=np.float32)

    def _copy(self, lo: int, hi: int) -> np.ndarray:
        i, n = lo % self.capacity, hi - lo
        if i + n <= self.capacity:
            return self._data[i:i + n].copy()
        return np.concatenate((self._data[i:], self._data[:i + n - self.capacity]))


# ---------------------------------------------------------------- voice activity detection

@dataclass
class VadResult:
    probability: float  # how much this frame alone sounds like speech, 0..1
    speech: bool  # the smoothed state (hysteresis, min_speech_ms to start, hangover_ms to stop)
    level_db: float  # RMS without DC, dBFS


MIN_DB = -120.0  # the level of digital silence, and of a broken (NaN) frame
DIGITAL_SILENCE_DB = -100.0  # a device sending zeros (starting up, muted): no room is this quiet, so it teaches the floor nothing
INITIAL_FLOOR_DB = -60.0  # a laptop microphone's usual noise floor, until the audio shows the real one
FLOOR_MAX_DB = -30.0  # the floor never rises above this, so loud speech is always heard
FLOOR_DOWN_S = 0.04  # the floor follows the level down at once (a pause shows the room)...
FLOOR_UP_S = 3.0  # ...and up slowly while the level moves, never above the quietest moment of the last FLOOR_MEMORY_S,
FLOOR_MEMORY_S = 5.0  # so a long sentence without a real gap isn't learnt as noise...
FLOOR_STEADY_S = 0.3  # ...but quickly while the level is steady: a fan turned on, a noisy room at the session's start
STEADY_MS = 200  # speech always swings within this long (syllables, gaps between words); a steady level is noise
STEADY_DB = 6.0
STEADY_MIN_FRAMES = 3  # fewer frames than this can't tell steady from not
SLOPE_DB = 2.0  # the logistic's width: probability 0.5 at the margin, 0.88 at 4 dB above it, 0.12 at 4 dB below


class VAD:
    """A local, energy-based voice activity detector with temporal smoothing (plan section 10).

    Each frame's level (RMS dBFS after removing DC) is compared with an adaptive noise floor: the probability is the
    logistic of (level - floor - noise_floor_margin_db), and nothing below absolute_floor_db is speech. The floor drops
    with the level at once, rises quickly while the level is steady (noise) and slowly while it swings (speech), then
    never above the quietest moment of the last few seconds; it stays within [absolute_floor_db - margin,
    FLOOR_MAX_DB]. Speech starts once the probability stays at speech_threshold or above for min_speech_ms of frames
    that aren't steady (a click is too short; a fan's hum is steady), and stops hangover_ms after it falls below
    silence_threshold.

    After each frame, `voiced` says whether it was heard as speech itself (not only held on by the hangover), and
    `segment_start` is the sample where the current speech began (backdated to the first frame of its onset): a
    segment runs from there to the end of its last voiced frame. Positions count the samples given to process()."""

    def __init__(self, rate: int, config: VadConfig, frame_ms: int = 20, *, noise_floor_db: float | None = None):
        if rate < 1:
            raise ValueError("the sample rate must be positive")
        self.rate, self.config = rate, config
        self.frame = max(1, round(rate * frame_ms / 1000))
        seconds = self.frame / rate

        def frames(ms: float) -> int:
            return max(0, math.ceil(ms / 1000 / seconds - 1e-9))

        self._start_frames = max(1, frames(config.min_speech_ms))
        self._hang_frames = frames(config.hangover_ms)
        self._down = 1 - math.exp(-seconds / FLOOR_DOWN_S)
        self._up = 1 - math.exp(-seconds / FLOOR_UP_S)
        self._steady_rate = 1 - math.exp(-seconds / FLOOR_STEADY_S)
        self._floor_range = (config.absolute_floor_db - config.noise_floor_margin_db, FLOOR_MAX_DB)
        # A floor known from before (the always-on microphone's VAD) spares the first frames' guess.
        self.noise_floor_db = self._clamp(INITIAL_FLOOR_DB if noise_floor_db is None else noise_floor_db)
        self._levels: deque[float] = deque(maxlen=max(STEADY_MIN_FRAMES, frames(STEADY_MS)))
        # The quietest level of each half second, for the last FLOOR_MEMORY_S (a sliding minimum at O(1) per frame).
        self._block_frames = max(1, frames(500))
        self._block_min, self._block_count = math.inf, 0
        self._minima: deque[float] = deque(maxlen=max(1, round(FLOOR_MEMORY_S / 0.5)))
        self.speech = False
        self.voiced = False
        self.position = 0  # samples processed
        self.segment_start = 0
        self._run = 0  # frames in a row loud enough to start speech
        self._run_start = 0
        self._quiet = 0  # frames in a row too quiet to continue it
        self._pending = np.zeros(0, dtype=np.float32)  # process_block's partial frame

    def _clamp(self, db: float) -> float:
        lo, hi = self._floor_range
        return max(lo, min(hi, db))

    def _recent_min(self, level: float) -> float:
        self._block_min, self._block_count = min(self._block_min, level), self._block_count + 1
        recent = min(self._block_min, min(self._minima, default=math.inf))
        if self._block_count == self._block_frames:
            self._minima.append(self._block_min)
            self._block_min, self._block_count = math.inf, 0
        return recent

    def process(self, frame) -> VadResult:
        """One frame (normally `self.frame` samples) -> its probability, the smoothed speech state and its level."""
        x = np.asarray(frame, dtype=np.float64).reshape(-1)
        level = MIN_DB
        if len(x):
            x = x - x.mean()  # a microphone's DC offset is no sound
            power = float(np.dot(x, x)) / len(x)
            if math.isfinite(power) and power > 0:  # NaN or inf from a broken driver counts as silence
                level = max(MIN_DB, 10 * math.log10(power))
        cfg, floor = self.config, self.noise_floor_db
        excess = (level - floor - cfg.noise_floor_margin_db) / SLOPE_DB
        p = 0.0 if level < cfg.absolute_floor_db else 1 / (1 + math.exp(-max(-60.0, min(60.0, excess))))

        steady = False
        if level > DIGITAL_SILENCE_DB:
            self._levels.append(level)
            steady = len(self._levels) >= STEADY_MIN_FRAMES and max(self._levels) - min(self._levels) <= STEADY_DB
            recent = self._recent_min(level)
            if level < floor:
                floor += (level - floor) * self._down
            elif steady:
                floor += (level - floor) * self._steady_rate
            else:
                floor = min(floor + (level - floor) * self._up, max(floor, recent))
            self.noise_floor_db = self._clamp(floor)

        self.voiced = False
        if not self.speech:
            # A steady sound doesn't start speech: it is the noise the floor hasn't caught up with yet.
            if p >= cfg.speech_threshold and not steady:
                if not self._run:
                    self._run_start = self.position
                self._run += 1
            else:
                self._run = 0
            if self._run >= self._start_frames:
                self.speech = self.voiced = True
                self.segment_start, self._run, self._quiet = self._run_start, 0, 0
        elif p >= cfg.silence_threshold:
            self.voiced, self._quiet = True, 0
        else:
            self._quiet += 1
            if self._quiet > self._hang_frames:
                self.speech = False
        self.position += len(x)
        return VadResult(p, self.speech, level)

    def process_block(self, block) -> list[bool]:
        """Any number of samples -> the speech state of each frame they complete. A partial frame waits for the next
        block, so the decisions don't depend on how the audio was split."""
        x = _mono(block)
        if len(self._pending):
            x = np.concatenate((self._pending, x))
        n, f = len(x) // self.frame, self.frame
        out = [self.process(x[i * f:(i + 1) * f]).speech for i in range(n)]
        self._pending = x[n * f:].copy()
        return out


def _note_voiced(segments: list[list[int]], vad: VAD) -> None:
    """After a voiced frame: extend the current speech segment, or open a new one."""
    if segments and segments[-1][0] == vad.segment_start:
        segments[-1][1] = vad.position
    else:
        segments.append([vad.segment_start, vad.position])


def speech_segments(audio, rate: int, config: VadConfig, frame_ms: int = 20, *,
                    noise_floor_db: float | None = None) -> list[tuple[int, int]]:
    """The speech in a whole recording, as [start, end) sample spans: from the onset of each speech to its last voiced
    frame. Gaps the hangover bridges stay inside a span; the hangover after the last word doesn't extend it. A last
    partial frame is not judged."""
    x = _mono(audio)
    vad = VAD(rate, config, frame_ms, noise_floor_db=noise_floor_db)
    segments: list[list[int]] = []
    f = vad.frame
    for i in range(len(x) // f):
        vad.process(x[i * f:(i + 1) * f])
        if vad.voiced:
            _note_voiced(segments, vad)
    return [(s, e) for s, e in segments]


# ---------------------------------------------------------------- chunker

QUIET_WINDOW_MS = 100  # a forced cut goes to the middle of the quietest window this long
MIN_LEAD_MS = 500  # the least silence a chunk without speech yet keeps before the present


class Chunker:
    """Cuts one session's audio into AudioChunks while it is being recorded (plan sections 11-16).

    - Pause: once the chunk has speech worth sending of its own (min_speech_ms), a silence of pause_boundary_ms ends it,
      keep_silence_ms after the last speech.
    - Maximum: at max_duration_ms the cut goes to the middle of the quietest 100 ms within [max - boundary_grace_ms,
      max + boundary_lookahead_ms], decided once that audio is here: no chunk is longer than max + lookahead, and none
      is cut in a loud place when a quieter one is in reach.
    - Overlap: the next chunk starts overlap_ms before the cut (never before the previous chunk's start), at once, so
      nothing is lost; overlap_end_sample is the previous emitted chunk's end. Its own speech is only what comes after
      the cut: the repeated words neither count towards sending it nor end it at a pause.
    - While a chunk has no speech worth sending, its start follows the present at pause_boundary_ms, though never into
      the last speech or the keep_silence_ms after it (overlap_end_sample then equals start_sample): a long pause
      doesn't use up the next chunk's length, and a cough followed by silence fades out of it.
    - A chunk without speech worth sending (silence, a cough) is not emitted; sequence numbers count emitted chunks.

    speech_seconds is the speech the VAD heard in the whole chunk, overlap included (what the speech engine is given).
    Memory: only the audio from the current chunk's start is kept, at most max + lookahead plus a frame."""

    def __init__(self, session_id: str, rate: int, config: ChunkingConfig, vad_config: VadConfig, frame_ms: int = 20, *,
                 noise_floor_db: float | None = None):
        self.session_id, self.rate, self.config = session_id, rate, config
        self.vad = VAD(rate, vad_config, frame_ms, noise_floor_db=noise_floor_db)
        f = self._frame = self.vad.frame

        def n(ms: float) -> int:
            return max(0, round(ms * rate / 1000))

        # Clamped so every cut moves the chunk's start forward, whatever the configuration.
        self._max = max(2 * f, n(config.max_duration_ms))
        self._overlap = min(n(config.overlap_ms), self._max // 2)
        self._grace = min(n(config.boundary_grace_ms), self._max - self._overlap - 1)
        self._lookahead = n(config.boundary_lookahead_ms)
        self._pause = max(f, n(config.pause_boundary_ms))
        self._keep = n(config.keep_silence_ms)
        self._lead = max(self._pause, n(MIN_LEAD_MS))
        self._min_speech = n(config.min_speech_ms)
        self._min_cut = max(self._min_speech, n(getattr(config, "min_cut_speech_ms", config.min_speech_ms)))
        self._window = max(1, n(QUIET_WINDOW_MS))

        self._buf = np.zeros(2 * rate, dtype=np.float32)  # audio from _buf_start on; grows only up to ~2x a chunk
        self._buf_start = 0
        self._seen = 0
        self._partial = 0  # samples of the frame being filled
        self._start = 0  # the current chunk's first sample
        self._own = 0  # its own audio starts here (after the previous cut): speech before it belongs to the last chunk
        self._prev_end: int | None = None  # the last emitted chunk's end
        self._held: AudioChunk | None = None  # the last chunk sent: a short rest at the release joins it
        self._held_audio: np.ndarray | None = None  # its audio (the scheduler releases the chunk's own once accepted)
        self._held_segments: list[tuple[int, int]] = []  # its speech spans
        self._min_alone = n(getattr(config, "min_alone_speech_ms", 0))
        self._context = n(getattr(config, "join_context_ms", 0))
        self._sequence = 0
        self._segments: list[list[int]] = []  # voiced speech spans that may still fall in a chunk
        self._last_voiced: int | None = None
        self._finished = False

    @property
    def samples_seen(self) -> int:
        return self._seen

    @property
    def retained(self) -> int:
        """Samples kept for the current chunk."""
        return self._seen - self._start

    def push(self, block) -> list[AudioChunk]:
        """Add audio (any length, even one sample) -> the chunks it completed, in order."""
        if self._finished:
            raise RuntimeError("the session's audio is finished")
        x = _mono(block)
        out: list[AudioChunk] = []
        pos = 0
        while pos < len(x):  # frame by frame, so decisions fall on the same samples however the audio was split
            take = min(len(x) - pos, self._frame - self._partial)
            self._append(x[pos:pos + take])
            pos += take
            self._partial += take
            if self._partial == self._frame:
                self._partial = 0
                out += self._on_frame()
        return out

    def finish(self, timed: bool = False) -> list[AudioChunk]:
        """The key is released and the tail is in: the rest of the audio, as the last chunk(s) (boundary "end").
        `timed`: the engine gives word times, so a short rest can go with only the end of the part before it."""
        if self._finished:
            return []
        self._finished = True
        out: list[AudioChunk] = []
        while self._seen - self._start > self._max + self._lookahead:  # the last partial frame went past the maximum
            out += self._cut(self._quietest(), "max")
        if self._seen > self._start:
            joined = self._join_rest(timed)
            out += [joined] if joined is not None else self._cut(self._seen, "end")
        self._buf, self._segments = np.zeros(0, dtype=np.float32), []
        return out

    def _append(self, x: np.ndarray) -> None:
        if self._seen + len(x) - self._buf_start > len(self._buf):
            # Drop what no chunk needs any more; grow only when that doesn't leave half the room free, so audio is
            # moved rarely (amortised O(1)) and the buffer stays within about twice the longest chunk.
            used = self._seen - self._start
            size = len(self._buf) if 2 * (used + len(x)) <= len(self._buf) else 2 * (used + len(x))
            buf = self._buf if size == len(self._buf) else np.empty(size, dtype=np.float32)
            buf[:used] = self._buf[self._start - self._buf_start:self._seen - self._buf_start]
            self._buf, self._buf_start = buf, self._start
        i = self._seen - self._buf_start
        self._buf[i:i + len(x)] = x
        self._seen += len(x)

    def _audio(self, start: int, end: int) -> np.ndarray:
        return self._buf[start - self._buf_start:end - self._buf_start]

    def _on_frame(self) -> list[AudioChunk]:
        now = self._seen
        self.vad.process(self._audio(now - self._frame, now))
        if self.vad.voiced:
            _note_voiced(self._segments, self.vad)
            self._last_voiced = now
        out: list[AudioChunk] = []
        while True:
            lv = self._last_voiced
            if lv is not None and now - lv >= self._pause and self._speech(self._own, now) >= self._min_cut:
                out += self._cut(min(lv + self._keep, now), "pause")
            elif now - self._start >= self._max + self._lookahead:
                out += self._cut(self._quietest(), "max")
            else:
                break
        lv = self._last_voiced
        if not self._worth(self._own, now):
            target = now - self._lead
            if target > self._start and (lv is None or target >= lv + self._keep):
                self._start, self._own = target, max(self._own, target)
                self._prune()
        return out

    def _speech(self, start: int, end: int) -> int:
        return sum(max(0, min(e, end) - max(s, start)) for s, e in self._segments)

    def _worth(self, start: int, end: int) -> bool:
        speech = self._speech(start, end)
        return speech > 0 and speech >= self._min_speech

    def _quietest(self) -> int:
        lo, hi = self._start + self._max - self._grace, self._start + self._max + self._lookahead
        x = self._audio(lo, hi).astype(np.float64)
        w = min(self._window, len(x))
        if not w:
            return lo
        x -= x.mean()
        energy = np.concatenate(([0.0], np.cumsum(x * x)))
        return lo + int(np.argmin(energy[w:] - energy[:-w])) + w // 2

    def _join_rest(self, timed: bool = False) -> AudioChunk | None:
        """The rest at the key release with little speech of its own (min_alone_speech_ms) isn't sent alone: after a
        cut at the maximum it is mostly the repeated overlap, after a pause often only the release click, and a cloud
        model given that wrote text never said. The part before it goes again with the rest, as one chunk that replaces
        it; the session merges that chunk instead of the two. With word times (`timed`), only the end of that part goes
        with the rest, as an overlap the merge takes out like any other: the part itself is already being transcribed, and
        sending it all again made the last answer, the one the user waits for, the slowest (a 19 s part, 2026-10-22).
        None: send the rest as usual."""
        held, audio, end = self._held, self._held_audio, self._seen
        if held is None or audio is None or not self._worth(self._own, end):
            return None
        rest = self._speech(self._own, end)
        if rest >= self._min_alone or end - held.start_sample > 2 * self._max:
            return None  # enough speech of its own, or a long wait before the release: the rest is sent as it is
        lead = self._lead_in(held, self._speech(held.end_sample, end)) if timed else None
        start = held.start_sample if lead is None else lead
        self._sequence += 1
        speech = (held.speech_seconds if lead is None else self._held_speech(lead, held.end_sample) / self.rate) \
            + self._speech(held.end_sample, end) / self.rate
        # Silence after a pause cut may already be dropped (the next part's start follows the present): it goes back as
        # zeros, so the chunk's audio still spans its samples.
        kept = max(held.end_sample, self._buf_start)
        audio = np.concatenate([audio[start - held.start_sample:], np.zeros(kept - held.end_sample, dtype=np.float32),
                                self._audio(kept, end)])
        if lead is not None:
            return AudioChunk(self.session_id, self._sequence, audio, self.rate, lead, end, held.end_sample, speech, "end")
        return AudioChunk(self.session_id, self._sequence, audio, self.rate, held.start_sample, end,
                          held.overlap_end_sample, speech, "end", replaces=held.sequence)

    def _held_speech(self, start: int, end: int) -> int:
        return sum(max(0, min(e, end) - max(s, start)) for s, e in self._held_segments)

    def _lead_in(self, held: AudioChunk, rest: int) -> int | None:
        """Where the rest's chunk starts in the part before it: back far enough for join_context_ms of speech in all,
        then to the quietest moment just before (not inside a word), never into the part before that one (its words are
        settled). None: the whole part goes again (no word times, not enough speech in reach, or too little saved)."""
        need = self._context - rest
        if self._context <= 0:
            return None
        floor = max(held.start_sample, held.overlap_end_sample)
        point, got = held.end_sample, 0
        for s, e in reversed(self._held_segments):
            if need <= got:
                break
            s, e = max(s, floor), min(e, held.end_sample)
            if e <= s:
                continue
            point = e - min(e - s, need - got)
            got += e - point
        if got < need or held.end_sample - point > 2 * self._context:
            return None
        point = max(floor, min(point, self._start))  # at least the overlap the rest has anyway (after a cut at the maximum)
        lo = max(floor, point - round(0.5 * self.rate))  # up to half a second earlier: a gap between words
        if point - lo >= self._window:
            x = self._held_audio[lo - held.start_sample:point - held.start_sample].astype(np.float64)
            x -= x.mean()
            energy = np.concatenate(([0.0], np.cumsum(x * x)))
            point = lo + int(np.argmin(energy[self._window:] - energy[:-self._window])) + self._window // 2
        if point - held.start_sample < 2 * self.rate:
            return None  # less than two seconds saved: not worth a second kind of chunk
        return point

    def _cut(self, cut: int, boundary: str) -> list[AudioChunk]:
        out = []
        if self._worth(self._own, cut):
            self._sequence += 1
            overlap_end = self._start if self._prev_end is None else min(max(self._prev_end, self._start), cut)
            chunk = AudioChunk(self.session_id, self._sequence, self._audio(self._start, cut).copy(), self.rate,
                               self._start, cut, overlap_end, self._speech(self._start, cut) / self.rate, boundary)
            out.append(chunk)
            self._held, self._held_audio = chunk, chunk.audio
            self._held_segments = [(max(s, self._start), min(e, cut)) for s, e in self._segments
                                   if e > self._start and s < cut]
            self._prev_end = cut
        # The overlap protects words cut at the maximum length. A pause cut sits after the last word plus keep_silence_ms,
        # so a 1 s overlap would reach back into that word: on the owner's recordings Parakeet then decoded the next part
        # to nothing (2026-10-01). After a pause the next part starts at the cut, with silence.
        overlap = self._overlap if boundary == "max" else 0
        self._start, self._own = max(cut - overlap, self._start), cut
        self._prune()
        return out

    def _prune(self) -> None:
        self._segments = [s for s in self._segments if s[1] > self._start]


# ---------------------------------------------------------------- pre-roll

PREROLL_KEEP_MS = 300  # silence before the key press: this much of it is kept
PREROLL_RECENT_MS = 300  # speech that ended this close before the press still belongs to the dictation
PREROLL_MARGIN_MS = 200  # kept before that speech's onset (soft first sounds the VAD hears late)
UTTERANCE_GAP_MS = 500  # speech pieces closer than this are one utterance (a pause between words or phrases)


def trim_preroll(audio, rate: int, preroll_samples: int, vad_config: VadConfig, *,
                 noise_floor_db: float | None = None) -> int:
    """How many samples to drop from the start of a session whose first `preroll_samples` came from before the key
    press (plan section 94). Speech that runs into the press (or ended less than 300 ms before it) is kept from its
    start, 200 ms early, with the pieces of the same utterance before it: the user began a moment before pressing.
    Otherwise only the pre-roll's last 300 ms are kept: earlier talk isn't dictation, and silence makes no giant
    session. Nothing after the press is ever dropped."""
    x = _mono(audio)
    press = max(0, min(int(preroll_samples), len(x)))
    if not press:
        return 0

    def n(ms: float) -> int:
        return round(ms * rate / 1000)

    # A little audio after the press confirms an onset that began just before it.
    segments = [s for s in speech_segments(x[:press + n(PREROLL_KEEP_MS)], rate, vad_config, noise_floor_db=noise_floor_db)
                if s[0] < press]
    keep_from = press - n(PREROLL_KEEP_MS)
    if segments and segments[-1][1] > press - n(PREROLL_RECENT_MS):
        start = segments[-1][0]
        for s, e in reversed(segments[:-1]):
            if start - e >= n(UTTERANCE_GAP_MS):
                break
            start = s
        keep_from = min(keep_from, start - n(PREROLL_MARGIN_MS))
    return max(0, min(press, keep_from))
