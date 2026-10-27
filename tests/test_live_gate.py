"""The cost saver's speech gate, on synthetic sound only (no recordings): a room, a fan, typing, speech."""
import time

import numpy as np
import pytest

from sst.live import gate as G
from sst.live.gate import SpeechGate

RATE, FRAME = 16_000, 1600
rng = np.random.default_rng(7)


def at_dbfs(x: np.ndarray, db: float) -> np.ndarray:
    return x * (10 ** (db / 20) * 32768 / (np.sqrt(np.mean(x ** 2)) + 1e-12))


def pink(seconds: float) -> np.ndarray:
    n = int(seconds * RATE)
    w = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(w), dtype=float)
    f[0] = 1
    return np.fft.irfft(w / np.sqrt(f), n)


def room(seconds: float, db: float = -62.0) -> np.ndarray:
    return at_dbfs(pink(seconds), db)


def fan(seconds: float, db: float = -45.0) -> np.ndarray:  # broadband with mains hum harmonics
    t = np.arange(int(seconds * RATE)) / RATE
    return at_dbfs(pink(seconds) + 0.5 * sum(np.sin(2 * np.pi * h * 120 * t) for h in (1, 2, 3)), db)


def speech(seconds: float, db: float = -26.0) -> tuple[np.ndarray, np.ndarray]:
    """Syllables: 180 ms of a 140 Hz voice with harmonics to 3.5 kHz, 70 ms apart; and each sample's 'is talking'."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    voice = sum(np.sin(2 * np.pi * k * 140 * t + k) / k for k in range(1, 26))
    on = (t % 0.25) < 0.18
    return at_dbfs(voice * on, db) * 1.0, on


def clicks(seconds: float, db: float = -20.0) -> np.ndarray:
    """Typing: a sharp decaying click every 120-250 ms."""
    n = int(seconds * RATE)
    y = np.zeros(n)
    i = 0
    while (i := i + int(rng.uniform(0.12, 0.25) * RATE)) + 400 < n:
        y[i:i + 400] += rng.standard_normal(400) * np.exp(-np.arange(400) / 60)
    return y * 10 ** (db / 20) * 32768


def frames(x: np.ndarray) -> list[bytes]:
    x = np.clip(np.round(x), -32768, 32767).astype(np.int16)
    return [x[i:i + FRAME].tobytes() for i in range(0, len(x) - FRAME + 1, FRAME)]


def run(gate: SpeechGate, x: np.ndarray) -> np.ndarray:
    return np.array([gate.process(f) for f in frames(x)])


def test_the_first_second_is_sent():
    sent = run(SpeechGate(), np.zeros(5 * RATE))
    assert sent[:10].all() and not sent[10:].any()


@pytest.mark.parametrize("noise", [room, fan], ids=["room", "fan"])
def test_noise_closes_the_gate_after_the_hangover(noise):
    gate = SpeechGate()
    sent = run(gate, noise(20))
    assert not sent[60:].any()  # learnt and closed well within 6 s, including the 2 s hangover
    assert not gate.open and gate.gated_s >= 14.0


@pytest.mark.parametrize("noise", [room, fan], ids=["room", "fan"])
def test_speech_opens_it_and_no_speech_frame_is_lost(noise):
    gate = SpeechGate()
    quiet, talk = 10, 4
    voice, on = speech(talk)
    x = noise(quiet + talk + 5)
    x[quiet * RATE:(quiet + talk) * RATE] += voice
    talking = np.zeros(len(x), bool)
    talking[quiet * RATE:(quiet + talk) * RATE] = on
    sent = run(gate, x[:quiet * RATE])
    before = gate.openings  # a fan may open it once while its floor is learnt
    sent = np.concatenate([sent, run(gate, x[quiet * RATE:])])
    assert not sent[quiet * 10 - 10:quiet * 10].any()  # closed before
    assert gate.openings == before + 1
    replayed = sent.copy()  # the engine sends the PREROLL_FRAMES skipped before an opening first
    for i in np.flatnonzero(sent[1:] & ~sent[:-1]) + 1:
        replayed[max(0, i - SpeechGate.PREROLL_FRAMES):i] = True
    speaking = talking[:len(sent) * FRAME].reshape(-1, FRAME).any(1)
    assert replayed[speaking].all()
    assert sent[(quiet + talk) * 10:(quiet + talk) * 10 + 19].all()  # the 2 s pause after it too
    assert not sent[(quiet + talk + 3) * 10:].any()  # then closed again


def test_key_clicks_do_not_open_it():
    gate = SpeechGate()
    run(gate, room(5))
    before = gate.openings
    sent = run(gate, room(30) + clicks(30))
    assert gate.openings == before and not sent.any()


def test_all_zero_frames_are_not_speech_and_leave_the_floor():
    gate = SpeechGate()
    run(gate, room(10))
    floor = gate.floor
    sent = run(gate, np.zeros(10 * RATE))
    assert not sent.any() and gate.floor == floor and gate.openings == 0


def test_a_floor_too_loud_fails_open():
    gate = SpeechGate()
    sent = run(gate, fan(20, db=-12))
    assert gate.floor > G.FAIL_OPEN_FLOOR and sent.all() and gate.gated_s == 0


def test_an_exception_inside_fails_open(monkeypatch, caplog):
    gate = SpeechGate()
    run(gate, room(10))
    monkeypatch.setattr(gate, "_levels", lambda x: 1 / 0)
    assert all(gate.process(f) for f in frames(room(2)))
    assert gate.process(b"\x01")  # half a sample: not a frame at all
    assert caplog.text.count("Speech gate failed") == 1


def test_gated_seconds_and_openings_are_counted():
    gate = SpeechGate()
    x = room(40)
    voice, _ = speech(3)
    for start in (10, 25):
        x[start * RATE:(start + 3) * RATE] += voice
    sent = run(gate, x)
    assert gate.gated_s == pytest.approx((~sent).sum() * 0.1)
    assert gate.openings == 2 == (sent[1:] & ~sent[:-1]).sum()


def test_a_frame_takes_well_under_two_ms():
    gate = SpeechGate()
    fs = frames(fan(30))
    t = time.perf_counter()
    for f in fs:
        gate.process(f)
    assert (time.perf_counter() - t) / len(fs) < 0.002
