"""What the laptop plays, as live captions get it: the device's format to 16 kHz mono frames, silence filled in, a new
output device followed. Windows is faked; no sound is played or captured."""
import threading
import time

import numpy as np
import pytest

from sst.live import wasapi
from sst.live.level import Leveler
from sst.live.wasapi import FRAME, Capture, Downmix, Framer, HighPass, Mixer, Resampler, choose_microphone, decode


def sine(freq, seconds, rate):
    return np.sin(2 * np.pi * freq * np.arange(int(seconds * rate)) / rate).astype(np.float32)


def pitch(x, rate):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / rate)[spectrum.argmax()]


@pytest.mark.parametrize("rate", [48_000, 44_100, 32_000, 16_000])
def test_any_device_rate_becomes_16_khz_without_seams(rate):
    audio = sine(440, 2.0, rate)
    resample, out, i = Resampler(rate), [], 0
    rng = np.random.default_rng(0)
    while i < len(audio):  # packets of uneven sizes, as WASAPI hands them over
        n = int(rng.integers(100, 2000))
        out.append(resample(audio[i:i + n]))
        i += n
    y = np.concatenate(out)
    assert abs(len(y) - 32_000) <= 2  # 2 s at 16 kHz
    assert abs(pitch(y, 16_000) - 440) < 2
    assert np.max(np.abs(np.diff(y))) < 0.2  # no jumps where packets meet (a 440 Hz sine moves ~0.17 a sample)


def test_formats_decode_to_mono():
    stereo = np.array([[0.5, -0.5], [0.25, 0.75]], dtype="<f4")
    assert np.allclose(decode(stereo.tobytes(), 2, 32, True), [0.0, 0.5])
    pcm16 = np.array([16384, -16384], dtype="<i2")
    assert np.allclose(decode(pcm16.tobytes(), 1, 16, False), [0.5, -0.5])
    pcm24 = bytes([0x00, 0x00, 0x40, 0x00, 0x00, 0xC0])  # +0.5 and -0.5 as 24-bit little-endian
    assert np.allclose(decode(pcm24, 1, 24, False), [0.5, -0.5])
    with pytest.raises(ValueError):
        decode(b"\0" * 8, 1, 8, False)


def test_frames_are_100_ms_of_pcm16():
    framer = Framer()
    assert framer(np.zeros(1000, np.float32)) == []
    frames = framer(np.full(2500, 0.5, np.float32))  # 3,500 samples so far: two frames, 300 left
    assert len(frames) == 2 and all(len(f) == FRAME * 2 for f in frames)
    assert np.frombuffer(frames[1], "<i2")[-1] == 16383
    assert len(framer(np.zeros(1300, np.float32))) == 1  # 300 + 1,300 = one more frame


class FakeStream:
    """An opened output device: hands out the packets queued in it, then nothing (silence)."""
    rate, channels, bits, is_float = 48_000, 2, 32, True

    def __init__(self, packets=(), changed_after=None):
        self.packets, self.changed_after, self.reads, self.closed = list(packets), changed_after, 0, False

    def read(self):
        self.reads += 1
        out, self.packets = self.packets[:1], self.packets[1:]
        return out

    def default_changed(self):
        return self.changed_after is not None and self.reads >= self.changed_after

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.01  # each look at the clock is 10 ms later: the capture loop runs every 10 ms
        return self.now


def run_capture(streams, seconds, until=None, **named):
    frames, opened = [], list(streams)
    capture = Capture(opener=lambda: opened.pop(0), clock=Clock(), **named)
    capture._stop.wait = lambda s: time.sleep(0.0005)  # the loop's 10 ms pauses, much faster
    capture.start(frames.append)
    end = time.monotonic() + (5.0 if until else seconds)
    while time.monotonic() < end and not (until and until()):
        time.sleep(0.01)
    capture.stop()
    return frames, capture


def test_sound_comes_through_as_frames():
    loud = [sine(440, 0.1, 48_000)] * 10  # 1 s of sound, in 100 ms packets
    frames, capture = run_capture([FakeStream(loud)], 0.3)
    audio = np.concatenate([np.frombuffer(f, "<i2") for f in frames]).astype(np.float32)
    assert capture.device_rate == 48_000 and len(frames) >= 9
    assert abs(pitch(audio[:16_000], 16_000) - 440) < 3


def test_silence_keeps_the_clock_going_when_nothing_plays():
    frames, _ = run_capture([FakeStream()], 0.3)
    assert len(frames) > 5 and all(f == bytes(FRAME * 2) for f in frames)  # a frame of silence every 100 ms


def test_a_new_default_output_is_followed():
    first, second = FakeStream(changed_after=50), FakeStream()  # headphones become the default meanwhile
    frames, _ = run_capture([first, second], 0, until=lambda: second.reads > 0)
    assert first.closed and second.reads > 0  # noticed at the next look (every 2 s), and the new one opened


def test_an_output_that_cant_be_opened_is_reported_to_the_caller():
    def broken():
        raise OSError("no output device")
    capture = Capture(opener=broken)
    with pytest.raises(OSError):
        capture.start(lambda frame: None)
    assert not any(t.name == "live-output" and t.is_alive() for t in threading.enumerate())


def test_the_speakers_are_heard_without_rflows_own_sound_and_the_chosen_microphone(monkeypatch):
    opened = []
    monkeypatch.setattr(wasapi, "_ProcessLoopback", lambda: opened.append("process loopback") or FakeStream())
    monkeypatch.setattr(wasapi, "_Stream", lambda flow, role: Whole())  # the whole mix it's checked against
    monkeypatch.setattr(wasapi, "_Meters", FakeMeters)
    monkeypatch.setattr(wasapi, "_Microphone", lambda wanted: opened.append(("microphone", wanted)) or "a microphone")
    speakers = Capture.speakers()._opener()
    assert isinstance(speakers, wasapi._Speakers) and isinstance(speakers.main, wasapi._Checked)
    assert isinstance(speakers.main.main, FakeStream) and isinstance(speakers.main.whole, Whole)
    assert Capture.microphone("Headset Microphone (Jabra)")._opener() == "a microphone"
    assert opened == ["process loopback", ("microphone", "Headset Microphone (Jabra)")]
    assert (Capture.speakers().what, Capture.microphone().what) == ("output device", "microphone")


def test_without_process_loopback_the_whole_output_is_captured(monkeypatch):
    def unavailable():
        raise OSError("process loopback refused (0x80070057)")  # before Windows 11
    monkeypatch.setattr(wasapi, "_ProcessLoopback", unavailable)
    monkeypatch.setattr(wasapi, "_Stream", lambda flow, role: (flow, role))
    assert wasapi._speakers() == (wasapi.E_RENDER, wasapi.E_CONSOLE)


def test_the_capture_says_whether_rflows_own_voice_is_in_it():
    class Mixed(FakeStream):
        hears_self = True
    _, capture = run_capture([Mixed()], 0.1)
    assert capture.hears_self
    _, capture = run_capture([FakeStream()], 0.1)
    assert not capture.hears_self


def test_the_real_mix_format_parts_are_the_documented_sizes():
    assert wasapi.ctypes.sizeof(wasapi._WAVEFORMATEX) == 18 and wasapi.ctypes.sizeof(wasapi._WAVEFORMATEXTENSIBLE) == 40
    f = wasapi._WAVEFORMATEX.float32(16_000)  # what process loopback is asked for first
    assert (f.wFormatTag, f.nChannels, f.nSamplesPerSec, f.nAvgBytesPerSec, f.nBlockAlign, f.wBitsPerSample) == (
        3, 1, 16_000, 64_000, 4, 32)


def level_db(frames):
    audio = np.concatenate([np.frombuffer(f, "<i2") for f in frames]).astype(np.float32) / 32768
    return 20 * np.log10(np.sqrt(np.mean(audio ** 2)))


def test_the_computers_sound_turned_far_down_still_reaches_the_model_at_a_normal_level():
    def heard(**named):
        stream = FakeStream([sine(440, 0.1, 48_000) * 10 ** (-70 / 20)] * 60)  # 6 s at -73 dBFS: a few 16-bit steps
        frames, _ = run_capture([stream], 0, until=lambda: not stream.packets, **named)
        return [f for f in frames if any(f)][-10:]  # its last second (not the silence after it)
    plain, raised = heard(), heard(level=True)
    assert level_db(plain) < -70 and -26 < level_db(raised) < -20  # its loudest 20 ms at -20 dBFS: a normal level
    assert abs(pitch(np.concatenate([np.frombuffer(f, "<i2") for f in raised]).astype(np.float32), 16_000) - 440) < 3
    assert Capture.speakers().level is True and not Capture.speakers().highpass  # the computer's sound: as it is
    mic = Capture.microphone()
    assert mic.level == Leveler.for_microphone and mic.highpass  # the room: with a noise floor, without its rumble


def test_nothing_heard_with_windows_sound_muted_is_said_once():
    class Muted(FakeStream):
        def muted(self):
            return True
    problems = []
    capture = Capture(opener=lambda: Muted(), clock=Clock(), level=True)
    capture.on_problem = problems.append
    capture._stop.wait = lambda s: time.sleep(0.0002)
    capture.start(lambda frame: None)
    end = time.monotonic() + 5
    while time.monotonic() < end and capture._clock.now < 12:  # 12 s of the capture's own time
        time.sleep(0.01)
    capture.stop()
    assert problems == [wasapi.MUTED]


def test_sound_coming_through_a_muted_output_is_no_problem():
    class Muted(FakeStream):  # a PC whose loopback is taken before Windows' volume: muted, and still heard
        def muted(self):
            return True
    problems = []
    capture = Capture(opener=lambda: Muted([sine(440, 0.1, 48_000) * 0.01] * 2000), clock=Clock(), level=True)
    capture.on_problem = problems.append
    capture._stop.wait = lambda s: time.sleep(0.0002)
    capture.start(lambda frame: None)
    end = time.monotonic() + 5
    while time.monotonic() < end and capture._clock.now < 8:
        time.sleep(0.01)
    capture.stop()
    assert problems == []


def db(x):
    x = np.asarray(x, dtype=np.float64)
    return 10 * np.log10(np.mean(x ** 2) + 1e-30)


# ---- the microphone (#144): the one chosen in Rflow, its channels, its rumble

MICS = [("{mic-array}", "Microphone Array (Realtek(R) Audio)"), ("{jabra}", "Headset Microphone (Jabra Evolve2 65)")]


def test_the_microphone_chosen_in_rflow_is_heard_else_windows_default():
    assert choose_microphone(MICS, "Headset Microphone (Jabra Evolve2 65)", "{mic-array}", "{jabra}") == (
        "{jabra}", "chosen in Rflow")
    assert choose_microphone(MICS, "Headset Microphone (Jabra Evol", "{mic-array}", "") == (  # an older Rflow's cut name
        "{jabra}", "chosen in Rflow")
    device, why = choose_microphone(MICS, "USB Microphone (Blue Yeti)", "{mic-array}", "{jabra}")
    assert device == "{mic-array}" and why == "Windows' default (USB Microphone (Blue Yeti) isn't connected)"
    assert choose_microphone(MICS, "", "{mic-array}", "{jabra}") == ("{mic-array}", "Windows' default")
    assert choose_microphone(MICS, "", "", "{jabra}") == ("{jabra}", "Windows' default for calls")
    with pytest.raises(OSError):
        choose_microphone([], "", "", "")


def test_windows_output_muted_isnt_the_microphones_problem():
    stream = wasapi._Stream.__new__(wasapi._Stream)
    stream.flow = wasapi.E_CAPTURE
    assert not stream.muted()  # nothing asked of Windows: a microphone isn't silenced by the output's mute


def test_the_raw_client_properties_are_the_documented_size():
    assert wasapi.ctypes.sizeof(wasapi._ClientProperties) == 16  # AudioClientProperties since Windows 8.1


def packets_of(x, size):
    return [x[i:i + size] for i in range(0, len(x), size)]


def test_a_stereo_microphone_wired_out_of_phase_isnt_cancelled_out():
    voice = sine(300, 1.0, 48_000) * 0.1  # -23 dBFS
    stereo = np.stack([voice, -0.9 * voice], axis=1)  # the second capsule wired the other way round
    downmix = Downmix()
    out = np.concatenate([downmix(p) for p in packets_of(stereo, 480)])  # 10 ms packets, as WASAPI hands them over
    assert db(stereo.mean(axis=1)) < db(voice) - 18  # their average: the voice nearly gone
    assert abs(db(out[24_000:]) - db(voice)) < 0.5 and downmix.loudest == 0  # the louder channel: the voice whole
    raw = stereo.astype("<f4").tobytes()
    assert abs(db(decode(raw, 2, 32, True, Downmix())[24_000:]) - db(voice)) < 0.5


def test_a_stereo_microphone_in_phase_or_just_noisy_is_averaged():
    voice = sine(300, 1.0, 48_000) * 0.1
    rng = np.random.default_rng(0)
    for stereo in (np.stack([voice, 0.9 * voice], axis=1),  # two capsules hearing the same voice
                   rng.standard_normal((48_000, 2)).astype(np.float32) * 0.01):  # a quiet room: uncorrelated noise
        downmix = Downmix()
        out = np.concatenate([downmix(p) for p in packets_of(stereo, 480)])
        assert np.allclose(out, stereo.mean(axis=1)) and downmix.loudest == -1
    mono = voice[:, None]
    assert np.array_equal(Downmix()(mono), voice)


def test_a_raw_microphones_rumble_is_taken_out_and_speech_left():
    def through(freq):
        x = sine(freq, 1.0, 16_000)
        high = HighPass()
        y = np.concatenate([high(p) for p in packets_of(x, 160)])  # 10 ms at a time: no seams
        assert len(y) == len(x)
        return db(y[4000:]) - db(x[4000:])  # past the filter's start
    assert through(50) < -30 and through(30) < -40  # mains hum, a fan, the desk
    assert abs(through(300)) < 0.5 and abs(through(1000)) < 0.5 and abs(through(150)) < 1.0  # voices
    x = np.random.default_rng(1).standard_normal(16_000).astype(np.float32)
    high = HighPass()
    assert np.allclose(HighPass()(x), np.concatenate([high(p) for p in packets_of(x, 237)]), atol=1e-5)


# ---- what the laptop plays (#145): a call on another output device, and saying where the sound is

NAMES = {"speakers": "Speakers (Realtek)", "handsfree": "Headset (WH-1000XM4 Hands-Free)", "hdmi": "Monitor (HDMI)"}


class FakeMeters:
    """Windows' output devices: which are the defaults, and how loud each plays now (the test sets them)."""

    def __init__(self, console="speakers", calls="speakers", peaks=None):
        self.console, self.calls = console, calls
        self.peaks = dict(peaks if peaks is not None else {"speakers": 0.0})
        self.closed = False

    def read(self):
        return self.console, self.calls, dict(self.peaks)

    def name(self, device_id):
        return NAMES.get(device_id, device_id)

    def close(self):
        self.closed = True


class Loopback(FakeStream):
    """Process loopback: already 16 kHz mono."""
    rate, channels = 16_000, 1


def call_device(opened, packets=()):
    def open_call():
        stream = Loopback(packets)
        stream.device_id = "handsfree"
        opened.append(stream)
        return stream
    return open_call


def test_a_call_on_the_headsets_hands_free_device_is_heard_too():
    meters = FakeMeters(console="speakers", calls="handsfree", peaks={"speakers": 0.0, "handsfree": 0.0})
    opened, main = [], Loopback()
    speakers = wasapi._Speakers(main, meters=lambda: meters, open_call=call_device(opened, [sine(300, 0.01, 16_000)] * 100))
    assert speakers.poll(0.0) is None and speakers.call is None  # nothing plays there: not opened (no call mode for it)
    meters.peaks["handsfree"] = 0.3  # the call starts on the headset
    speakers.poll(1.0)
    assert speakers.call is opened[0] and not speakers.hears_self  # Rflow's voice plays on the default output only
    audio = np.concatenate([np.concatenate(speakers.read() or [np.zeros(0, np.float32)]) for _ in range(200)])
    assert len(audio) >= 15_000 and abs(pitch(audio, 16_000) - 300) < 3  # the call's sound, mixed in
    meters.peaks["handsfree"] = 0.0  # a pause in the call: it stays open a while
    speakers.poll(1.0 + wasapi.CALL_LINGER - 1)
    assert speakers.call is not None
    speakers.poll(1.0 + wasapi.CALL_LINGER + 0.5)  # the call is over
    assert speakers.call is None and opened[0].closed
    speakers.close()
    assert main.closed and meters.closed


def test_the_call_device_isnt_captured_twice_when_its_the_default_output():
    meters = FakeMeters(console="speakers", calls="speakers", peaks={"speakers": 0.5})
    opened = []
    speakers = wasapi._Speakers(Loopback(), meters=lambda: meters, open_call=call_device(opened))
    for t in range(5):
        speakers.poll(float(t))
    assert opened == [] and speakers.call is None


def test_the_whole_mix_still_says_rflows_voice_is_in_it():
    class Mixed(FakeStream):
        hears_self = True
    assert wasapi._Speakers(Mixed(), meters=FakeMeters).hears_self


def test_without_meters_the_sound_is_still_captured():
    def broken():
        raise OSError("no meters")
    speakers = wasapi._Speakers(Loopback([np.ones(160, np.float32)]), meters=broken)
    assert speakers.poll(0.0) is None and len(speakers.read()[0]) == 160


def test_two_streams_are_mixed_in_step_and_neither_waits_for_a_silent_one():
    mix, a, b = Mixer(16_000), np.full(160, 0.1, np.float32), np.full(160, 0.2, np.float32)
    assert np.allclose(mix(a, b), 0.3)  # added sample by sample
    assert len(mix(a, None)) == 0  # the other's packet isn't here yet: in step, so it waits
    assert np.allclose(mix(None, b), 0.3)
    out = [mix(a, None) for _ in range(Mixer.IDLE)]  # the other device went quiet (Windows sends nothing)
    assert sum(map(len, out[:-1])) == 0 and np.allclose(out[-1], 0.1) and len(out[-1]) == 160 * Mixer.IDLE
    assert np.allclose(mix(a, None), 0.1)  # from then on at once
    mix = Mixer(16_000)
    assert len(mix(np.zeros(3000, np.float32), np.zeros(10, np.float32))) == 10  # in step
    assert len(mix(np.zeros(1000, np.float32), None)) == 3990 - 3200  # but never more than LAG (0.2 s) behind
    assert len(mix.flush()) == 3200


def test_nothing_heard_while_another_device_plays_says_where_the_sound_may_be():
    meters = FakeMeters(peaks={"speakers": 0.0, "hdmi": 0.0})
    speakers = wasapi._Speakers(Loopback(), meters=lambda: meters)
    assert [speakers.poll(float(t)) for t in range(4)] == [None] * 4
    meters.peaks["hdmi"] = 0.2  # the video plays on the monitor's speakers
    told = {t: speakers.poll(float(t)) for t in range(4, 12)}
    message = wasapi.ELSEWHERE.format(heard="Speakers (Realtek)", other="Monitor (HDMI)")
    assert [m for m in told.values() if m is not None] == [message] and told[8] == message  # once, after a few seconds
    assert "Monitor (HDMI)" in message and "default output" in message
    meters.peaks["speakers"] = 0.1  # heard again
    assert speakers.poll(12.0) == "" and speakers.poll(13.0) is None


def test_the_capture_passes_the_hint_on_and_takes_it_back_when_it_closes():
    class Telling(FakeStream):
        def __init__(self):
            super().__init__()
            self.told = 0

        def poll(self, now):
            self.told += 1
            return "Nothing heard on Speakers" if self.told == 1 else None
    notices = []
    capture = Capture(opener=Telling, clock=Clock())
    capture.on_note = notices.append
    capture._stop.wait = lambda s: time.sleep(0.0005)
    capture.start(lambda frame: None)
    end = time.monotonic() + 5
    while time.monotonic() < end and capture._clock.now < 3:
        time.sleep(0.01)
    capture.stop()
    assert notices == ["Nothing heard on Speakers", ""]


# ---- process loopback that hears nothing (#176): a Teams call on Bluetooth earbuds reached only the whole mix

class Whole(FakeStream):
    """The default output's whole mix, at the device's rate (mono after decoding)."""
    rate, channels = 48_000, 1


def read_seconds(stream, seconds):
    return np.concatenate([np.concatenate(stream.read() or [np.zeros(0, np.float32)]) for _ in range(int(seconds * 10))])


class Ticks:
    """A clock that moves 100 ms each time it's read: _Checked reads it once per read, and each fake packet is 100 ms."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        self.now += 0.1
        return self.now


def checked_with(main_packets, whole_packets):
    main, whole = Loopback(main_packets), Whole(whole_packets)
    return wasapi._Checked(main, whole, clock=Ticks()), main, whole


def read_all(checked, reads):
    """Each read's output, and when (in reads) it switched each way."""
    out, switched, turns = [], checked.switched, []
    for i in range(reads):
        got = checked.read()
        out.append(np.concatenate(got) if got else np.zeros(0, np.float32))
        if checked.switched != switched:
            switched = checked.switched
            turns.append((i, switched))
    return out, turns


def test_the_whole_mix_is_used_when_process_loopback_hears_nothing_while_the_output_plays():
    checked, main, whole = checked_with([], [sine(300, 0.1, 48_000)] * 40)  # 4 s of a call; process loopback: silence
    assert not checked.hears_self
    out, turns = read_all(checked, 45)
    assert turns == [(14, True)] and checked.hears_self and not main.closed  # 1.5 s; Rflow's voice is in the mix now
    audio = np.concatenate(out)
    assert abs(len(audio) - 4 * 16_000) < 100  # all of the call: its first 1.5 s were kept and sent on the switch
    assert len(out[14]) >= 1.4 * 16_000 and abs(pitch(audio[-16_000:], 16_000) - 300) < 3  # at 16 kHz
    checked.close()
    assert whole.closed and main.closed


def test_an_early_sound_through_process_loopback_no_longer_ends_the_check():
    ding = [sine(880, 0.1, 16_000) * 0.5] * 3  # a notification, heard by both
    checked, main, whole = checked_with(ding, [sine(880, 0.1, 48_000) * 0.5] * 3 + [sine(300, 0.1, 48_000)] * 60)
    out, turns = read_all(checked, 63)
    assert [on for _, on in turns] == [True] and turns[0][0] <= 3 + 30  # by the time the ding left the 3 s window
    sent = np.concatenate(out[turns[0][0]:])  # from the switch: the call kept since process loopback last heard
    assert abs(len(sent) - (63 - 3) * 1600) < 100  # the whole call from its start, once; the ding isn't sent twice
    assert abs(pitch(sent[:16_000], 16_000) - 300) < 3


def test_it_switches_back_when_process_loopback_hears_what_plays_again():
    call = [sine(300, 0.1, 48_000)] * 80
    checked, main, whole = checked_with([], call)
    read_all(checked, 20)
    assert checked.switched
    main.packets = [sine(300, 0.1, 16_000)] * 60  # the app plays where process loopback hears it again
    out, turns = read_all(checked, 30)
    assert turns and turns[0][1] is False and turns[0][0] <= 12 and not checked.hears_self
    assert checked.switches == 2 and abs(pitch(np.concatenate(out[turns[0][0]:]), 16_000) - 300) < 3


def test_a_partial_share_or_a_short_burst_doesnt_make_it_flap():
    checked, main, whole = checked_with([sine(300, 0.1, 16_000) * 0.5] * 100, [sine(300, 0.1, 48_000)] * 100)
    _, turns = read_all(checked, 100)  # process loopback at a quarter of the energy: between MISSING and CARRIES
    assert turns == [] and not checked.switched
    checked, main, whole = checked_with([], [sine(300, 0.1, 48_000)] * 200)
    read_all(checked, 20)
    for at in range(20, 200, 15):  # bursts of 0.5 s through process loopback, every 1.5 s
        main.packets = [sine(300, 0.1, 16_000)] * 5
        _, turns = read_all(checked, 15)
        assert turns == [], at
    assert checked.switched and checked.switches == 1


def test_the_check_holds_when_the_apps_are_lowered_to_inaudible():
    from sst.live.ducking import INAUDIBLE
    low = [sine(300, 0.1, 48_000) * INAUDIBLE] * 40  # Teams lowered to 0% while the voice speaks
    checked, *_ = checked_with([], low)
    read_all(checked, 20)
    assert checked.switched


def test_rflows_own_voice_in_the_mix_doesnt_count_as_a_sound_process_loopback_missed(monkeypatch):
    monkeypatch.setattr(wasapi, "_voice_until", 2000.0)  # the voice is playing (process loopback leaves it out)
    checked, main, whole = checked_with([], [sine(300, 0.1, 48_000)] * 40)
    read_seconds(checked, 4)
    assert not checked.switched and not any(np.any(kept) for kept in checked._kept)  # nor kept to be sent: silence


def test_on_the_mix_the_voice_is_silenced_only_when_it_plays_on_that_output(monkeypatch):
    checked, main, whole = checked_with([], [sine(300, 0.1, 48_000)] * 40)
    whole.device_id = "headphones"
    read_all(checked, 20)
    monkeypatch.setattr(wasapi, "_voice_device", "headphones")
    assert checked.hears_self
    monkeypatch.setattr(wasapi, "_voice_device", "speakers")  # the voice plays elsewhere: the mix can't hear it
    assert not checked.hears_self


def test_with_nothing_playing_nothing_changes():
    main, whole = Loopback(), Whole([np.zeros(4800, np.float32)] * 40)
    checked = wasapi._Checked(main, whole, clock=Ticks())
    read_seconds(checked, 4)
    assert not checked.switched and checked.whole is whole and not whole.closed


def test_the_capture_follows_the_switch_for_the_voices_silencing():
    main, whole = Loopback(), Whole([sine(300, 0.1, 48_000)] * 40)
    checked = wasapi._Checked(main, whole, clock=Ticks())
    speakers = wasapi._Speakers(checked, meters=FakeMeters)
    frames, capture = run_capture([speakers], 0, until=lambda: checked.switched)
    time.sleep(0.05)
    assert checked.switched and speakers.hears_self


def test_without_a_whole_mix_process_loopback_is_used_alone(monkeypatch):
    main = Loopback()
    monkeypatch.setattr(wasapi, "_ProcessLoopback", lambda: main)

    def no_mix(*a):
        raise OSError("no output")
    monkeypatch.setattr(wasapi, "_Stream", no_mix)
    assert wasapi._speakers() is main


# ---- Both on speakers: the session needs to know whether the computer plays sound one would hear, and where

def test_the_capture_says_whether_the_output_is_private_and_when_it_last_had_sound_one_would_hear():
    class Speakers(FakeStream):
        private = False
    _, capture = run_capture([Speakers([sine(300, 0.1, 48_000) * 0.3] * 5)], 0.2)
    assert capture.private is False and capture.heard_at > 0 and capture.sounding(1e9)
    _, capture = run_capture([FakeStream([sine(300, 0.1, 48_000) * 1e-4] * 5)], 0.2)  # a call's hiss at -80 dBFS
    assert capture.private is None and capture.heard_at < 0 and not capture.sounding(1.0)


def test_the_speakers_capture_reads_whether_the_output_is_headphones(monkeypatch):
    monkeypatch.setattr(wasapi, "_speakers", lambda: Loopback())
    monkeypatch.setattr(wasapi, "_Meters", FakeMeters)
    monkeypatch.setattr(wasapi, "_output_private", lambda: True)
    stream = Capture.speakers()._opener()
    assert stream.private is True
