"""Live translation's paths end to end, in virtual time: a capture's frames through LiveSession to an engine that
answers like Gemini Live Translate (its voice LAG later, as long as what it heard, silence for silence; a sentence's
words once its voice has come), its voice through the Speaker to a player that only counts. Each source (the
computer, the microphone, both), speaking on and off, the cost saver on and off, a reconnect, the whole mix captured
when process loopback misses a call (Bluetooth), and long meetings of sentence after sentence (#227's stall). Nothing is
played, recorded or sent: the captures, Gemini and the player are fakes."""
import base64
import json
import time
from collections import deque

import numpy as np
import pytest

from sst.live import speaker as speakers
from sst.live.contracts import MIC, SYSTEM, Kind, LiveConfig, LiveEvent
from sst.live.gemini import GeminiLiveTranslate
from sst.live.session import LiveSession
from sst.live.speaker import Speaker
from tests.test_live_gemini import KEY, FakeServer

LAG = 10  # frames (1 s) from what's heard to Gemini's voice for it
MIN_RUN = 3  # frames of speech the interpreter needs to make a sentence of it
VOICE_RATE = 24_000
LOUD = 500  # a frame with samples beyond this has speech in it (the room's noise stays well under)
UNDER_VOICE = round(speakers.TAIL * 10) + 1  # frames of the next sentence a way that hears the voice may lose
SENTENCES = 40


@pytest.fixture(autouse=True)
def _no_voice_left_playing(monkeypatch):
    from sst.live import wasapi
    monkeypatch.setattr(wasapi, "_voice_until", 0.0)
    monkeypatch.setattr(wasapi, "_voice_device", "")


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Air:
    """The output, in virtual time: what's played is heard for as long as it lasts after what's queued before it."""

    def __init__(self, clock, private=False):
        self.clock, self.private, self.until, self.closed = clock, private, 0.0, False
        self.heard_s = 0.0  # seconds of voice played (its pauses aside)

    def play(self, samples, rate):
        now = self.clock()
        self.until = max(self.until, now) + len(samples) / rate
        self.heard_s += float(np.count_nonzero(np.abs(samples) > 0.01)) / rate  # the voice, not its pauses
        return self.until - now

    def interrupt(self):
        self.until = self.clock()

    def close(self):
        self.closed = True


class FakeVoice:
    """The laptop's voice: what it was asked to say."""
    rate = 16_000

    def __init__(self):
        self.said = []

    def synthesize(self, text, speed=1.0):
        self.said.append(text)
        return np.full(int(0.5 * self.rate), 0.2, dtype=np.float32)


class Voice(Speaker):
    """The Speaker as it is, counting what it was handed, so a test knows when it has played or dropped all of it."""

    def __init__(self, *args, **named):
        super().__init__(*args, **named)
        self.handed = 0.0

    def hear_audio(self, lane, pcm, rate):
        if lane in self.own and not self.muted and not self._stop.is_set():
            self.handed += len(pcm) // 2 / rate
        super().hear_audio(lane, pcm, rate)

    def caught_up(self) -> bool:
        return not self._queue and abs(self.sound_s + self.sound_skipped_s + getattr(self, "quiet_s", 0.0) - self.handed) < 1e-6


def pcm(level: int, seconds: float = 0.1, rate: int = VOICE_RATE) -> bytes:
    return np.full(round(seconds * rate), level, dtype="<i2").tobytes()


def loud(frame: bytes) -> bool:
    return bool(len(frame)) and int(np.abs(np.frombuffer(frame, "<i2")).max()) > LOUD


def said(n: int, lane: str = SYSTEM) -> str:
    return f"Sentence {n} is here." if lane == SYSTEM else f"文 {n} です。"


class Answers:
    """What Gemini makes of what it hears: LAG frames later as much of its voice (speech for speech, silence for
    silence), and each run of speech a sentence, its words once its voice has come. `numbers` is shared by the
    connections of one session (a reconnect goes on counting)."""

    def __init__(self, numbers=None):
        self._line, self._run = deque(), 0
        self.numbers = numbers if numbers is not None else [0]

    def hear(self, frame: bytes) -> list[tuple]:
        """("audio", pcm) and ("line", n) answers to one frame heard."""
        self._line.append(loud(frame))
        if len(self._line) <= LAG:
            return []
        out = self._line.popleft()
        answers = [("audio", pcm(3000 if out else 0))]
        if out:
            self._run += 1
        elif self._run:
            if self._run >= MIN_RUN:
                self.numbers[0] += 1
                answers.append(("line", self.numbers[0]))
            self._run = 0
        return answers


class Interpreter:
    """Gemini Live Translate as LiveSession sees it, answering at once on the capture's thread (Answers)."""

    def __init__(self, on_event, lane, config=None, clock=None):
        self.on_event, self.lane, self.on_audio = on_event, lane, None
        self.answers, self.fed = Answers(), []

    def start(self):
        pass

    def stop(self):
        pass

    def feed(self, frame):
        self.fed.append(frame)
        for kind, what in self.answers.hear(frame):
            if kind == "audio" and self.on_audio is not None:
                self.on_audio(what, VOICE_RATE)
            elif kind == "line":
                text = said(what, self.lane)
                self.on_event(LiveEvent(Kind.TRANSLATION, text, lane=self.lane))
                self.on_event(LiveEvent(Kind.LINE, text, source=f"heard {what}", lane=self.lane))


class Capture:
    """A capture LiveSession feeds from: the test hands it frames. `hears_self`: the whole mix (Rflow's voice in it);
    `private`: the output is headphones; sounding(): the computer played sound just now (Both on speakers)."""

    def __init__(self, clock, hears_self=False, private=None):
        self.clock, self.hears_self, self.private, self.on_frame, self.loud_at = clock, hears_self, private, None, -1e9
        self.heard = []

    def start(self, on_frame):
        self.on_frame = on_frame

    def stop(self):
        self.on_frame = None

    def hear(self, frame):
        self.heard.append(frame)
        if loud(frame):
            self.loud_at = self.clock()
        self.on_frame(frame)

    def sounding(self, seconds):
        return self.clock() - self.loud_at < seconds


def talk():
    return np.full(1600, 3000, dtype="<i2").tobytes()


def quiet():
    return bytes(3200)


def meeting(sentences: int, talk_frames: int = 10, pause_frames: int = 10, frame=talk, end=quiet) -> list[bytes]:
    """Sentence after sentence: talk_frames of speech, then pause_frames of quiet, and a pause at the end for the last."""
    frames = []
    for _ in range(sentences):
        frames += [frame() for _ in range(talk_frames)] + [end() for _ in range(pause_frames)]
    return frames + [end() for _ in range(LAG + 2)]


def wait_until(condition, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.0005)
    return condition()


class Run:
    """One session in virtual time: its ways added, the voice while speaking, frames played through step by step.
    `engine(emit, lane, config, clock)` makes each way's engine."""

    def __init__(self, source="computer", speak=True, hears_self=False, private=False, engine=Interpreter, **config):
        self.clock = Clock()
        self.config = LiveConfig(source=source, speak=speak, target="en", mic_target="ja", **config)
        self.shown = []
        self.session = LiveSession(self.config, on_event=self.shown.append)
        self.captures = {}
        for lane in self.config.lanes:
            capture = Capture(self.clock, hears_self=hears_self if lane == SYSTEM else False,
                              private=private if lane == SYSTEM else None)
            self.session.add(lane, capture, lambda emit, lane=lane: engine(emit, lane, self.config, self.clock))
            self.captures[lane] = capture
        self.engines = {lane: engine for lane, (_, engine) in self.session.lanes.items()}
        self.air, self.laptop, self.voice = Air(self.clock, private=private), FakeVoice(), None
        if speak:
            self.speak()

    def speak(self):
        self.voice = Voice(lambda: self.laptop, self.air, self.config.spoken_lanes("en"), "en", clock=self.clock,
                           own=self.config.spoken_lanes())
        self.session.set_speaker(self.voice)

    def silent(self):
        self.session.set_speaker(None)
        self.voice = None

    def play(self, frames: dict[str, list[bytes]] | list[bytes], settle=lambda: True):
        """Each 100 ms: every way hears its frame, the engines answer, the voice plays what came; then time moves on."""
        if not isinstance(frames, dict):
            frames = {next(iter(self.captures)): frames}
        for step in range(max(map(len, frames.values()))):
            for lane, lane_frames in frames.items():
                if step < len(lane_frames):
                    self.captures[lane].hear(lane_frames[step])
            assert wait_until(lambda: settle() and (self.voice is None or self.voice.caught_up())), "the voice stalled"
            self.clock.now += 0.1

    def lines(self, lane=SYSTEM):
        return [e.text for e in self.shown if e.kind is Kind.LINE and e.lane == lane]

    def lost(self, lane=SYSTEM) -> list[int]:
        """The frames of speech that reached the engine as silence (the way was silenced), by when they were heard."""
        return [i for i, (heard, fed) in enumerate(zip(self.captures[lane].heard, self.engines[lane].fed, strict=True))
                if loud(heard) and not any(fed)]

    def lost_per_sentence(self, lane=SYSTEM, period=20) -> int:
        """The most frames of one sentence lost (sentences `period` frames apart)."""
        lost = self.lost(lane)
        return max((sum(1 for i in lost if i // period == n) for n in {i // period for i in lost}), default=0)

    def stop(self):
        self.session.stop()


# ---- the computer's sound

@pytest.mark.parametrize("hears_self", [False, True], ids=["process loopback", "the whole mix"])
def test_a_long_meeting_is_translated_and_spoken_to_its_last_sentence(hears_self):
    """#227: the whole mix (process loopback missed the call) is silenced while the voice speaks, and Gemini answers
    that silence with silence; played as voice, it kept the way silenced, and nothing after the first sentence came."""
    run = Run(hears_self=hears_self)
    run.play(meeting(SENTENCES))
    assert run.lines() == [said(n) for n in range(1, SENTENCES + 1)]  # every sentence, in order, to the last
    voice = run.voice
    assert voice.voiced == SENTENCES and voice.said == 0 and voice.unvoiced == 0  # Gemini's voice said them all
    if hears_self:  # what's said under the voice and its TAIL is lost (it can't be told from it): no more
        assert run.lost() and run.lost_per_sentence() <= UNDER_VOICE and 0 not in run.lost()
        assert run.air.heard_s >= SENTENCES * (1 - UNDER_VOICE / 10) - 1e-6
    else:
        assert run.lost() == [] and run.air.heard_s == pytest.approx(SENTENCES * 1.0)
    assert voice.quiet_s > 0  # Gemini's answer to the pauses: never played
    run.stop()
    assert run.air.closed and run.session.speaker is None


def test_without_speaking_nothing_is_silenced_and_every_line_shows():
    run = Run(speak=False, hears_self=True)
    frames = meeting(SENTENCES)
    run.play(frames)
    assert run.lines() == [said(n) for n in range(1, SENTENCES + 1)] and run.session.speaker is None
    assert run.engines[SYSTEM].fed == frames  # the whole mix as heard: no voice in it
    run.stop()


def test_speaking_turned_off_and_on_in_a_meeting_goes_on_where_it_is():
    run = Run(hears_self=True)
    run.play(meeting(10)[:200])
    run.silent()
    run.play(meeting(10)[:200])
    assert run.lost()[-1] < 200  # silent: nothing lost under a voice
    run.speak()
    run.play(meeting(10))
    assert run.lines() == [said(n) for n in range(1, 31)]
    assert run.voice.voiced == 11 and run.voice.said == 0  # the new voice: the line still coming, and the 10 after it
    run.stop()


def test_the_whole_mix_from_the_middle_of_a_meeting_and_back():
    """_Checked turns to the whole mix when process loopback misses the call (Bluetooth), and back when it hears it
    again: only while on the mix is the way silenced under the voice."""
    run = Run()
    capture = run.captures[SYSTEM]
    run.play(meeting(10)[:200])
    assert run.lost() == []
    capture.hears_self = True
    run.play(meeting(10)[:200])
    assert run.lost() and min(run.lost()) >= 200 and run.lost_per_sentence() <= UNDER_VOICE
    capture.hears_self = False
    lost = run.lost()
    run.play(meeting(10))
    assert run.lines() == [said(n) for n in range(1, 31)] and run.lost() == lost  # back: nothing more lost
    run.stop()


def test_a_meeting_that_never_pauses_still_gets_translated_through_the_whole_mix():
    """Talk with no pause long enough for the voice: the whole mix loses what's said under it, never the rest."""
    run = Run(hears_self=True)
    run.play(meeting(SENTENCES, talk_frames=20, pause_frames=3))
    lines = run.lines()
    assert len(lines) >= SENTENCES // 2 and lines == [said(n) for n in range(1, len(lines) + 1)]
    assert max(i for i, frame in enumerate(run.engines[SYSTEM].fed) if any(frame)) > len(run.captures[SYSTEM].heard) - 40
    run.stop()


# ---- the microphone (the room, or the user)

@pytest.mark.parametrize("private", [False, True], ids=["speakers", "headphones"])
def test_the_microphone_alone_is_silenced_under_the_voice_only_on_speakers(private):
    run = Run(source="microphone", private=private)
    run.play(meeting(20))
    assert run.lines(MIC) == [said(n, MIC) for n in range(1, 21)]
    assert run.laptop.said == []  # into Japanese: Gemini's voice says it, the laptop's (English) never
    if private:
        assert run.lost(MIC) == [] and run.air.heard_s == pytest.approx(20.0)
    else:  # out loud in the room: what's said under the voice and its echo is lost
        assert run.lost(MIC) and run.lost_per_sentence(MIC) <= UNDER_VOICE
    run.stop()


# ---- both: the others from the computer, the user from the microphone

def test_both_on_headphones_speaks_the_others_and_never_the_user():
    run = Run(source="both", private=True)
    others, me = meeting(10), [quiet()] * 10 + meeting(10)[:-10]  # the user answers in the others' pauses
    run.play({SYSTEM: others, MIC: me})
    assert run.lines(SYSTEM) == [said(n) for n in range(1, 11)] and run.lines(MIC) == [said(n, MIC) for n in range(1, 11)]
    assert run.voice.own == {SYSTEM} and run.voice.voiced == 10  # the user's words are for the others: not spoken
    assert run.air.heard_s == pytest.approx(10.0) and run.lost(MIC) == []  # headphones: nothing gated
    run.stop()


def test_both_on_speakers_gates_the_microphone_while_the_computer_or_the_voice_plays():
    run = Run(source="both", private=False)
    others = meeting(1, pause_frames=40) * 5  # 62 frames each: the others speak 1 s, then wait
    me = ([quiet()] * 35 + [talk()] * 10 + [quiet()] * 17) * 5  # the user answers once the voice and its echo are over
    run.play({SYSTEM: others, MIC: me})
    assert run.lines(SYSTEM) == [said(n) for n in range(1, 6)] and run.lines(MIC) == [said(n, MIC) for n in range(1, 6)]
    assert run.lost(MIC) == []
    run.stop()
    run = Run(source="both", private=False)
    run.play({SYSTEM: meeting(5), MIC: [talk()] * len(meeting(5))})  # the user talks over the others all along
    lost = set(run.lost(MIC))
    assert set(range(10)) <= lost  # the others' words from the speakers: never taken for the user's
    assert len(lost) < len(meeting(5))  # between them and the voice, the user is heard
    run.stop()


# ---- Gemini itself (GeminiLiveTranslate against a fake Live API answering like the model)

class EchoServer(FakeServer):
    """One Live API connection that answers the audio sent as the model does (Answers): its voice, and the words."""

    def __init__(self, numbers):
        super().__init__()
        self.answers, self.received, self.voice_s = Answers(numbers), 0, 0.0

    def send(self, text):
        super().send(text)
        audio = json.loads(text).get("realtimeInput", {}).get("audio")
        if audio is None:
            return
        self.received += 1
        for kind, what in self.answers.hear(base64.b64decode(audio["data"])):
            if kind == "audio":
                self.voice_s += len(what) // 2 / VOICE_RATE
                self.say(modelTurn={"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000",
                                                              "data": base64.b64encode(what).decode()}}]})
            else:
                self.say(inputTranscription={"text": f"heard {what}"}, outputTranscription={"text": said(what)})
                self.say(turnComplete=True)


class Gemini(GeminiLiveTranslate):
    """GeminiLiveTranslate as it is, keeping what it was fed (Run.lost)."""

    def __init__(self, *args, **named):
        super().__init__(*args, **named)
        self.fed = []

    def feed(self, frame):
        self.fed.append(frame)
        super().feed(frame)


def gemini(servers: list):
    """An engine factory for Run: GeminiLiveTranslate on `servers`, one per connection, in virtual time."""
    def make(emit, lane, config, clock):
        return Gemini(KEY, config.for_lane(lane), emit, connect=lambda url: servers.pop(0), clock=clock, lane=lane)
    return make


def handled(engine) -> int:
    """Frames the engine took from its queue: sent, or held back (silence, the cost saver; a pre-roll sent later moves
    from one to the other)."""
    return engine.sent + round(engine.silence_s / 0.1)


def settled(run, servers):
    """Every frame heard taken by the engine, every answer read, every piece of voice handed to the speaker."""
    engine = run.engines[SYSTEM]
    return (handled(engine) >= len(run.captures[SYSTEM].heard) and all(s.incoming.empty() for s in servers)
            and wait_until(lambda: run.voice.handed >= sum(s.voice_s for s in servers) - 1e-6, 1.0))


def room(level, seed):
    """100 ms of a room: noise at `level` (speech is loud to the speech gate, the room's hum quiet but not silence)."""
    rng = np.random.default_rng(seed)
    return lambda: (rng.standard_normal(1600) * level).clip(-32000, 32000).astype("<i2").tobytes()


@pytest.mark.parametrize("save", [False, True], ids=["everything sent", "the cost saver"])
@pytest.mark.parametrize("hears_self", [False, True], ids=["process loopback", "the whole mix"])
def test_geminis_answers_are_shown_and_spoken_with_and_without_the_cost_saver(save, hears_self):
    servers = [EchoServer([0])]
    run = Run(hears_self=hears_self, save=save, engine=gemini(list(servers)))
    engine = run.engines[SYSTEM]
    assert wait_until(lambda: servers[0].sent)
    frames = meeting(12, pause_frames=40, frame=room(3000, 1), end=room(20, 2))
    run.play(frames, lambda: settled(run, servers))
    assert wait_until(lambda: len(run.lines()) == 12)
    assert run.lines() == [said(n) for n in range(1, 13)]
    assert run.voice.voiced == 12 and run.voice.said == 0
    if hears_self:
        assert run.lost_per_sentence(period=50) <= UNDER_VOICE
    else:
        assert run.lost() == []
    spend = run.session.spend()
    if save:  # the room between sentences isn't sent: what was saved is counted, and every word is still there
        assert engine.silence_s > 10 and spend.unsent_s == pytest.approx(engine.silence_s) and spend.saved > 0
    else:
        assert engine.sent == len(frames) and spend.saved == 0
    assert spend.cost > 0 and spend.seconds == pytest.approx(len(frames) * 0.1)
    run.stop()


def test_a_reconnect_in_a_long_meeting_loses_no_frame_and_the_voice_goes_on():
    numbers = [0]
    servers = [EchoServer(numbers), EchoServer(numbers)]
    first, second = servers
    run = Run(hears_self=True, engine=gemini(list(servers)))
    engine = run.engines[SYSTEM]
    assert wait_until(lambda: first.sent)
    frames = meeting(20)
    run.play(frames[:205], lambda: settled(run, servers))
    first.incoming.put({"goAway": {"timeLeft": "10s"}})  # Google ends the connection soon: a new one, the same session
    assert wait_until(lambda: second.sent)
    run.play(frames[205:], lambda: settled(run, servers))
    assert wait_until(lambda: numbers[0] >= 19 and len(run.lines()) == numbers[0])
    assert first.closed and engine.reconnects == 1 and first.received + second.received == engine.sent
    lines = run.lines()
    assert lines == [said(n) for n in range(1, len(lines) + 1)] and len(lines) >= 19  # the one cut in two at most
    assert run.voice.voiced == len(lines) and run.voice.said == 0
    run.stop()
