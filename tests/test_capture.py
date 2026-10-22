"""Capture: the warm microphone's lead-in, the tail after the key release, closing when idle, Bluetooth never kept open,
old microphone names, following microphones plugged in and out, a microphone that goes quiet, and the audio's
preparation for the engine. No microphone: the audio callback is fed by hand, Windows' and PortAudio's device lists
are fakes."""
import time
import weakref

import numpy as np
import pytest

from sst import audio, devices
from sst.audio import PREROLL_SECONDS, Recorder, Take, condition

RATE = 16_000
BLOCK = 160  # 10 ms


class FakeStream:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _recorder(monkeypatch, call_quality=False, **options) -> Recorder:
    monkeypatch.setattr(audio, "_live", weakref.WeakSet())  # put back after the test: warm fakes stay "open"
    monkeypatch.setattr(audio, "refresh_devices", lambda force=False: False)  # Windows' microphones never change here
    r = Recorder(**options)
    opened = []

    def fake_open(force_refresh=False):
        r.rate, r.info = RATE, {"call_quality": call_quality, "mode": "raw" if r.raw else "windows"}
        r._stream, r._opened_for = FakeStream(), (r.device, r.raw)
        r._last_audio = time.monotonic()
        audio._live.add(r)
        opened.append(r._stream)
    monkeypatch.setattr(r, "_open", fake_open)
    r.opened = opened
    return r


def _feed(r: Recorder, seconds: float, value: float = 0.1) -> None:
    for _ in range(int(seconds * RATE / BLOCK)):
        r._on_audio(np.full((BLOCK, 1), value, dtype=np.float32), BLOCK, None, None)


def test_a_cold_start_has_no_lead_in_and_closes_after_the_recording(monkeypatch):
    r = _recorder(monkeypatch)
    r.start()
    _feed(r, 1.0)
    out = r.stop()
    assert len(out) == RATE and not r.warm and r.opened[0].closed


def test_a_warm_microphone_keeps_the_moment_before_the_key_press(monkeypatch):
    r = _recorder(monkeypatch, warm_seconds=300)
    r.start()
    _feed(r, 0.5)
    r.stop()
    assert r.warm  # kept open
    _feed(r, 2.0, value=0.2)  # waiting: only the last PREROLL_SECONDS are kept
    r.start()
    _feed(r, 1.0, value=0.3)
    take = r.stop_later()
    out = take.audio()
    assert len(r.opened) == 1  # no second open: instant start
    assert len(out) == pytest.approx((PREROLL_SECONDS + 1.0) * RATE, abs=BLOCK)
    assert np.all(out[:BLOCK] == np.float32(0.2)) and take.seconds == pytest.approx(1.0)  # the lead-in isn't counted


def test_long_blocks_keep_the_whole_pre_roll_and_tail(monkeypatch):
    """The always-on microphone's blocks are BLOCK_SECONDS long (fewer trips into Python): the 2 s pre-roll of the voice
    pipeline and the tail are still there in full, sample for sample."""
    block = int(RATE * audio.BLOCK_SECONDS)
    r = _recorder(monkeypatch, warm_seconds=float("inf"), tail=0.3)
    r.preroll_seconds = 2.0
    r.keep_open()

    def feed(seconds, value):
        for _ in range(int(seconds * RATE / block)):
            r._on_audio(np.full((block, 1), value, dtype=np.float32), block, None, None)
    feed(5.0, 0.1)  # waiting for the key
    r.start()
    feed(1.0, 0.3)
    take = r.stop_later()
    feed(0.5, 0.5)
    out = take.audio()
    assert take.done.is_set() and np.sum(out == np.float32(0.1)) >= 2.0 * RATE  # the pre-roll, whole
    assert np.sum(out == np.float32(0.3)) == RATE and np.sum(out == np.float32(0.5)) == int(0.3 * RATE)
    assert r._blocksize() == block and audio.BLOCK_SECONDS >= 0.05


def test_the_level_is_worked_out_only_when_something_shows_it(monkeypatch):
    r = _recorder(monkeypatch, warm_seconds=float("inf"))
    r.keep_open()
    calls = []
    real_mean = np.mean
    monkeypatch.setattr(audio.np, "mean", lambda *a, **k: calls.append(1) or real_mean(*a, **k))
    _feed(r, 1.0, value=0.25)  # the always-on microphone keeping its pre-roll: no level per block
    assert calls == []
    assert r.level == pytest.approx(0.25) and len(calls) == 1  # the pill asked
    r.start()
    assert r.level == 0.0  # a new recording starts from silence, as before
    _feed(r, 0.1, value=0.5)
    assert r.level == pytest.approx(0.5)
    r.stop()
    assert r.level == 0.0


def test_a_level_meter_shows_the_latest_block():
    meter = audio.LevelMeter()
    meter._on_audio(np.full((BLOCK, 1), 0.2, dtype=np.float32), BLOCK, None, None)
    assert meter.level == pytest.approx(0.2)


def test_the_tail_after_the_release_is_recorded_without_waiting_for_it(monkeypatch):
    r = _recorder(monkeypatch, warm_seconds=300, tail=0.3)
    r.start()
    _feed(r, 1.0)
    take = r.stop_later()
    assert not take.done.is_set()  # returned at once; the tail comes in on the audio thread
    _feed(r, 0.5, value=0.5)
    assert take.done.is_set()
    out = take.audio()
    assert len(out) == int(1.3 * RATE) and np.all(out[-BLOCK:] == np.float32(0.5))


def test_an_idle_warm_microphone_closes_after_its_time(monkeypatch):
    r = _recorder(monkeypatch, warm_seconds=300)
    r.start()
    _feed(r, 0.5)
    r.stop()
    r.tick(r._idle_since + 299)
    assert r.warm
    r.tick(r._idle_since + 301)
    assert not r.warm and r.opened[0].closed


def test_a_bluetooth_microphone_is_never_kept_open(monkeypatch):
    r = _recorder(monkeypatch, call_quality=True, warm_seconds=300)
    r.start()
    _feed(r, 0.5)
    r.stop()
    assert not r.warm  # it would keep the headset playing in call quality


def test_choosing_another_microphone_or_mode_reopens(monkeypatch):
    r = _recorder(monkeypatch, warm_seconds=300)
    r.start()
    r.stop()
    r.raw = True
    r.start()
    assert len(r.opened) == 2 and r.opened[0].closed and r.info["mode"] == "raw"


def test_closing_ends_a_tail_still_being_recorded(monkeypatch):
    r = _recorder(monkeypatch, warm_seconds=300, tail=0.3)
    r.start()
    _feed(r, 0.5)
    take = r.stop_later()
    r.close()
    assert take.done.is_set() and len(take.audio()) == RATE // 2


def test_a_finished_recording_counts_in_full():
    take = Take.ready(np.zeros(RATE, dtype=np.float32), RATE)
    assert take.seconds == 1.0 and take.done.is_set()


def test_old_mme_names_find_their_wasapi_microphone(monkeypatch):
    devices = [{"name": "Microphone Array (Intel® Smart ", "max_input_channels": 2, "hostapi": 0},
               {"name": "Microphone Array (Intel® Smart Sound Technology)", "max_input_channels": 4, "hostapi": 1},
               {"name": "Headset (Buds)", "max_input_channels": 1, "hostapi": 1}]
    monkeypatch.setattr(audio.sd, "query_devices", lambda *a, **k: devices)
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda *a, **k: [{"name": "MME", "default_input_device": 0},
                                                                      {"name": "Windows WASAPI", "default_input_device": 1}])
    assert audio._resolve("Microphone Array (Intel® Smart ") == 1  # saved by an older Rflow (MME cuts names at 31)
    assert audio._resolve("Headset (Buds)") == 2 and audio._resolve("Unplugged mic") is None
    assert audio.input_device_names(refresh=False) == [d["name"] for d in devices[1:]]


def test_condition_removes_dc_and_raises_the_peak_to_minus_1_dbfs():
    t = np.arange(RATE) / RATE
    quiet = (0.02 * np.sin(2 * np.pi * 200 * t) + 0.01).astype(np.float32)  # speech around -37 dBFS, with an offset
    out = condition(quiet)
    assert abs(float(out.mean())) < 1e-4 and float(np.abs(out).max()) == pytest.approx(10 ** (-1 / 20), rel=1e-3)
    hiss = np.full(RATE, 1e-5, dtype=np.float32)
    assert float(np.abs(condition(hiss)).max()) < 1e-4  # near-silence isn't blown up into loud noise
    assert len(condition(np.zeros(0, dtype=np.float32))) == 0


def test_audible_audio_that_decodes_to_nothing_is_tried_again_in_halves():
    from sst.engines.parakeet import ParakeetEngine
    engine = object.__new__(ParakeetEngine)  # no model: the decoder is a fake
    engine.conditioned = True
    calls = []

    def decode(piece, rate):
        calls.append(len(piece))
        return "" if len(calls) == 1 else f"part {len(calls) - 1}"
    engine._decode = decode
    speech = np.random.default_rng(0).uniform(-0.2, 0.2, 4 * RATE).astype(np.float32)
    assert engine._transcribe(speech, RATE) == "part 1 part 2" and calls == [4 * RATE, 2 * RATE, 2 * RATE]
    calls.clear()
    assert engine._transcribe(np.zeros(4 * RATE, dtype=np.float32), RATE) == "" and len(calls) == 1  # silence: no retry


# ---- following microphones plugged in and out (refresh_devices), and a microphone that goes quiet

REAL_REFRESH = audio.refresh_devices
LAPTOP, HEADSET = devices.Device("1", "Microphone (Realtek(R) Audio)"), devices.Device("2", "Headset (Buds)")


def _windows(monkeypatch, *present, default=None):
    """Windows' microphones (sst.devices), and PortAudio's restarts counted instead of done."""
    state = {"snapshot": devices.Snapshot(tuple(present), default or (present[0] if present else None)), "restarts": 0}
    monkeypatch.setattr(audio.devices, "snapshot", lambda: state["snapshot"])
    monkeypatch.setattr(audio.sd, "_terminate", lambda: None)
    monkeypatch.setattr(audio.sd, "_initialize", lambda: state.__setitem__("restarts", state["restarts"] + 1))
    monkeypatch.setattr(audio, "_seen", state["snapshot"])
    monkeypatch.setattr(audio, "_seen_ready", True)
    return state


def test_nothing_is_restarted_while_windows_microphones_stay_the_same(monkeypatch):
    windows = _windows(monkeypatch, LAPTOP)
    r = _recorder(monkeypatch, warm_seconds=float("inf"))
    r.keep_open()
    assert not REAL_REFRESH() and windows["restarts"] == 0 and len(r.opened) == 1


def test_a_headset_plugged_in_reopens_the_open_microphones_on_the_new_list(monkeypatch):
    windows = _windows(monkeypatch, LAPTOP)
    r = _recorder(monkeypatch, warm_seconds=float("inf"))  # the always-on microphone
    r.keep_open()
    windows["snapshot"] = devices.Snapshot((HEADSET, LAPTOP), HEADSET)  # plugged in: Windows makes it the default
    assert REAL_REFRESH() and windows["restarts"] == 1
    assert len(r.opened) == 2 and r.opened[0].closed and r.warm  # closed, PortAudio re-read, opened again
    assert not REAL_REFRESH() and windows["restarts"] == 1  # only once per change


def test_never_while_a_dictation_is_recorded(monkeypatch):
    windows = _windows(monkeypatch, LAPTOP)
    r = _recorder(monkeypatch, warm_seconds=float("inf"))
    r.start()  # recording
    _feed(r, 0.5)
    windows["snapshot"] = devices.Snapshot((HEADSET, LAPTOP), HEADSET)
    assert not REAL_REFRESH() and windows["restarts"] == 0 and len(r.opened) == 1  # the dictation goes on, untouched
    assert len(r.stop()) == RATE // 2
    assert REAL_REFRESH() and windows["restarts"] == 1  # then it happens


def test_a_level_meter_never_holds_a_refresh_back(monkeypatch):
    windows = _windows(monkeypatch, LAPTOP)
    meter = audio.LevelMeter()
    monkeypatch.setattr(audio, "_live", weakref.WeakSet())
    opened = []

    def fake_open(force_refresh=False):
        meter._stream, meter._last_audio = FakeStream(), time.monotonic()
        audio._live.add(meter)
        opened.append(meter._stream)
    monkeypatch.setattr(meter, "_open", fake_open)
    meter.start()
    windows["snapshot"] = devices.Snapshot((HEADSET, LAPTOP), HEADSET)
    assert REAL_REFRESH() and len(opened) == 2 and opened[0].closed  # the meter follows to the new list


def test_without_windows_list_portaudio_is_reread_only_when_nothing_is_open(monkeypatch):
    windows = _windows(monkeypatch)
    windows["snapshot"] = None
    monkeypatch.setattr(audio.devices, "snapshot", lambda: None)
    r = _recorder(monkeypatch, warm_seconds=float("inf"))
    assert REAL_REFRESH() and windows["restarts"] == 1  # as before: nothing open
    r.keep_open()
    assert not REAL_REFRESH() and windows["restarts"] == 1


def test_a_microphone_that_goes_quiet_is_opened_again(monkeypatch):
    r = _recorder(monkeypatch, warm_seconds=float("inf"))
    r.keep_open()
    _feed(r, 0.5)
    assert r.healthy
    r._last_audio -= audio.STALL_SECONDS + 0.1  # unplugged, or switched off when a headset was plugged in: no blocks
    assert not r.healthy
    r.tick(time.monotonic())  # between dictations: noticed and opened again
    assert len(r.opened) == 2 and r.opened[0].closed and r.healthy


def test_a_quiet_microphone_is_opened_again_at_the_key_press(monkeypatch):
    r = _recorder(monkeypatch, warm_seconds=float("inf"))
    r.keep_open()
    r._last_audio -= audio.STALL_SECONDS + 0.1
    r.start()
    assert len(r.opened) == 2 and r.opened[0].closed  # the dictation records on a freshly opened microphone


def test_a_chosen_microphone_not_connected_is_said_and_the_default_records(monkeypatch):
    r = Recorder("Headset (Buds)")
    r.info = {"device": "Microphone (Realtek(R) Audio)", "mode": "windows", "rate": 48000}
    r._note(None)  # the name isn't in the list: Windows' default was opened
    assert r.notice == "Headset (Buds) isn't connected: Rflow uses Microphone (Realtek(R) Audio) until it is."
    assert r.info["wanted"] == "Headset (Buds)"
    r.info = {"device": "Headset (Buds)", "mode": "windows", "rate": 16000}
    r._note(2)  # connected again
    assert r.notice == ""


def test_the_window_lists_windows_microphones_without_restarting_anything(monkeypatch):
    windows = _windows(monkeypatch, LAPTOP, HEADSET, default=HEADSET)
    assert audio.input_device_names() == [LAPTOP.name, HEADSET.name] and windows["restarts"] == 0
    assert audio.default_microphone() == HEADSET.name
