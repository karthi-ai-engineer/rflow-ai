"""The cost saver's speech gate: which 100 ms frames of live audio are worth sending to Google.

Gemini Live Translate bills every second streamed, both ways, and its voice comes back 1:1 with what's sent, silence
and room noise included. A meeting is about half talk, so sending only the talk (and the pause after it, so the model
hears the sentence end) roughly halves the bill. The digital-silence rule in gemini.py catches a muted microphone or a
quiet video; this gate also catches a room, a fan, a keyboard:

    the level of each 10 ms in the speech band (BAND_LO-BAND_HI Hz)
    the noise floor: the quietest of the last FLOOR_MEM_S (minimum statistics), learnt fast when the sound is steady
    voiced: a 10 ms OPEN_DB over the floor (HOLD_DB once open); speech: >= MIN_VOICED of them in the last 200 ms, in
            runs of RUN or more (a key click is 1-3 of them, a syllable 10 or more)
    sent: speech, the HANG_S after it, the first START_S of a session, and everything once the floor is above
          FAIL_OPEN_FLOOR (a loud room or music: it can't tell, so it never risks losing words)

When unsure it sends: any exception sends the frame. All-zero frames (a muted microphone, Rflow's own voice playing)
are never speech and teach the floor nothing. The frames skipped just before an opening are the engine's to replay
(PREROLL_FRAMES); the gate only decides. Pure numpy, under 0.1 ms a frame.
"""
import logging
from collections import deque

import numpy as np

from sst.live.contracts import RATE

log = logging.getLogger(__name__)

SUB_S = 0.010  # what's judged at a time: 10 ms, shorter than a key click
BAND_LO, BAND_HI = 200.0, 3800.0  # Hz: where speech is, above hum and rumble
OPEN_DB = 7.0  # a 10 ms this far over the noise floor is voiced
HOLD_DB = 6.0  # once open, a little less (hysteresis: a soft word at a sentence end keeps it open)
VOICED_WINDOW = 20  # 10 ms pieces judged together: the last 200 ms
RUN = 4  # voiced pieces in a row that count: 40 ms, longer than a key click
MIN_VOICED = 8  # 80 ms in such runs opens the gate
KEEP_VOICED = 4  # once open, 40 ms keeps it open (typing alone doesn't)
MIN_LEVEL = -75.0  # dB: quieter than this is never speech, whatever the floor
HANG_S = 2.0  # still sent after the last speech: the model hears the pause and ends its line
START_S = 1.0  # the first second is always sent (the floor isn't known yet, and the model hears the start)
FLOOR_START = -70.0  # dB: the floor before anything is heard
FLOOR_MEM_S = 8.0  # minimum statistics over this long: longer than a sentence, so its pauses are in it
FLOOR_BLOCK = 50  # 10 ms pieces per minimum kept (0.5 s)
FLOOR_UP_DB_S = 1.0  # the floor creeps up at most this fast when nothing quieter is heard
STEADY_S = 1.0  # sound this long that stays within STEADY_DB (a fan, a hum) is the floor: learnt in about 0.5 s
STEADY_DB = 6.0
STEADY_RATE = 0.2  # of the way to the steady level, each frame
SMOOTH = 3  # 10 ms pieces averaged for the floor (30 ms): noise varies less
FAIL_OPEN_FLOOR = -25.0  # dB: a floor this loud (loud room, music): everything is sent
ZERO_DB = -120.0


class SpeechGate:
    """16-bit mono frames in, whether to send each out; state kept across calls. `gated_s` (seconds not sent) and
    `openings` (closed to open) are for the log."""

    PREROLL_FRAMES = 10  # 1.0 s of 100 ms frames the engine keeps while closed, sent first when it opens

    def __init__(self, rate: int = RATE):
        self.rate = rate
        self.sub = max(1, round(rate * SUB_S))
        self.nfft = 1 << (self.sub - 1).bit_length()
        freq = np.fft.rfftfreq(self.nfft, 1 / rate)
        self._band = (freq >= BAND_LO) & (freq <= BAND_HI)
        self._win = np.hanning(self.sub).astype(np.float32)
        self._norm = self.sub * 0.375 * self.nfft / 2  # a Hann window's power, so the level is the band's mean power
        self.floor = FLOOR_START
        self._mins: deque[float] = deque(maxlen=round(FLOOR_MEM_S / (FLOOR_BLOCK * SUB_S)))
        self._block_min, self._block_n = np.inf, 0
        self._recent: deque[float] = deque(maxlen=round(STEADY_S / SUB_S))
        self._voiced: deque[bool] = deque(maxlen=VOICED_WINDOW)
        self._tail = np.zeros(0)  # the last SMOOTH - 1 powers, so the smoothing runs across frames
        self.open = False
        self._hang = 0.0
        self._heard_s = 0.0
        self.gated_s = 0.0
        self.openings = 0
        self._failed = False

    def process(self, frame: bytes) -> bool:
        """True: send this frame (speech, its hangover, the first START_S, or failing open)."""
        try:
            return self._process(frame)
        except Exception:
            if not self._failed:  # once: a broken gate sends everything, it never loses words
                self._failed = True
                log.exception("Speech gate failed: sending everything")
            return True

    def _process(self, frame: bytes) -> bool:
        x = np.frombuffer(frame, np.int16)
        dur = len(x) / self.rate
        sound = bool(len(x)) and bool(np.any(x))
        if sound:
            raw = self._levels(x)
            self._learn(raw)
        else:  # digital silence: never speech, and the floor stays where it was
            raw = np.full(max(1, len(x) // self.sub), ZERO_DB)
        thr = self.floor + (HOLD_DB if self.open else OPEN_DB)
        self._voiced.extend(bool(v > thr and v > MIN_LEVEL) for v in raw)  # unsmoothed: a click stays short
        sustained = run = 0
        for v in self._voiced:
            run = run + 1 if v else 0
            sustained += RUN if run == RUN else run > RUN
        speech = sustained >= (KEEP_VOICED if self.open else MIN_VOICED) or (sound and self.floor > FAIL_OPEN_FLOOR)
        starting = self._heard_s < START_S - 1e-9
        self._heard_s += dur
        if speech:
            self._hang = HANG_S
            if not self.open:
                self.open = True
                self.openings += 1
            return True
        if self.open:
            self._hang -= dur
            if self._hang <= 1e-9:
                self.open = False
            return True
        if starting:
            return True
        self.gated_s += dur
        return False

    def _levels(self, x: np.ndarray) -> np.ndarray:
        """The speech band's level (dB) of each 10 ms."""
        x = x.astype(np.float32) / 32768
        x = x - x.mean()
        if len(x) < self.sub:  # a short frame: judged as one 10 ms
            x = np.pad(x, (0, self.sub - len(x)))
        subs = x[: len(x) // self.sub * self.sub].reshape(-1, self.sub)
        spec = np.abs(np.fft.rfft(subs * self._win, self.nfft, axis=1)) ** 2
        return 10 * np.log10(spec[:, self._band].sum(1) / self._norm + 1e-13)

    def _learn(self, raw: np.ndarray) -> None:
        """The noise floor: minimum statistics over smoothed levels, learnt fast when the sound is steady."""
        p = np.concatenate([self._tail, 10 ** (raw / 10)])
        self._tail = p[-(SMOOTH - 1):]
        if len(p) < SMOOTH:
            return
        smooth = 10 * np.log10(np.convolve(p, np.ones(SMOOTH) / SMOOTH, "valid"))
        for v in smooth.tolist():
            self._block_min = min(self._block_min, v)
            self._block_n += 1
            if self._block_n == FLOOR_BLOCK:
                self._mins.append(self._block_min)
                self._block_min, self._block_n = np.inf, 0
        self._recent.extend(smooth.tolist())
        lowest = min(min(self._mins, default=np.inf), self._block_min)
        quiet, loud = min(self._recent), max(self._recent)
        if lowest < self.floor:
            self.floor = lowest
        elif loud - quiet <= STEADY_DB and len(self._recent) == self._recent.maxlen:
            self.floor += (quiet - self.floor) * STEADY_RATE
        else:
            self.floor = min(lowest, self.floor + FLOOR_UP_DB_S * len(raw) * SUB_S)
