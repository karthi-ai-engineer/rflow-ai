"""Meeting mode's recorder and session, with fakes only: no real audio, window, screen or network."""
import ast
import json
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

import sst.meeting
from sst.live.contracts import RATE
from sst.meeting.recorder import OTHERS, YOU, Recorder
from sst.meeting.session import DONE, FAILED, GONE_S, META, RECORDING, WRITING, MeetingOptions, MeetingSession

FRAME = b"\x01\x00" * (RATE // 10)  # 100 ms of PCM16


class FakeCapture:
    def __init__(self, fail=False):
        self.fail, self.on_frame, self.stopped = fail, None, False

    def start(self, on_frame):
        if self.fail:
            raise OSError("no device")
        self.on_frame = on_frame

    def stop(self):
        self.stopped = True


@dataclass(frozen=True)
class Window:
    app: str = "Microsoft Teams"
    title: str = "Weekly sync | Microsoft Teams"
    hwnd: int = 1
    pid: int = 2


@dataclass(frozen=True)
class Spend:
    cost: float


class FakeLive:
    def __init__(self):
        self.started = self.stopped = False
        self.cost = 0.0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def spend(self):
        return Spend(self.cost) if self.started and not self.stopped else None  # LiveCaptions: None once stopped


class FakeRecorder:
    def __init__(self, folder):
        self.folder, self.seconds, self.files, self.started, self.stopped = folder, 0.0, [], False, False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeFrames:
    def __init__(self, window, folder):
        self.window, self.folder, self.frames, self.started, self.stopped = window, folder, [], False, False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


@dataclass
class Result:
    title: str = "Weekly sync"
    transcript: tuple = (("00:01", "You", "the secret launch date is Friday"),)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make(tmp_path, options=None, *, summarise=None, live=None, alive=None, clock=None):
    made = {}

    def recorder(folder):
        made["recorder"] = FakeRecorder(folder)
        return made["recorder"]

    def frames(window, folder):
        made["frames"] = FakeFrames(window, folder)
        return made["frames"]

    session = MeetingSession(Window(), options or MeetingOptions(), tmp_path / "meeting", recorder_factory=recorder,
                             frames_factory=frames, live=live, summarise=summarise or (lambda folder: Result()),
                             clock=clock or Clock(), alive=alive or (lambda w: True),
                             estimate=lambda seconds, frames: seconds / 3600 * 0.25 + frames * 0.001, watch=0)
    return session, made


# ---- the recorder: two WAVs written as frames come

def test_recorder_writes_both_tracks_and_they_play_after_a_crash(tmp_path):
    others, you = FakeCapture(), FakeCapture()
    rec = Recorder(tmp_path, speakers=lambda: others, microphone=lambda: you)
    rec.start()
    for _ in range(10):
        others.on_frame(FRAME)
    for _ in range(4):
        you.on_frame(FRAME)
    assert rec.files == [tmp_path / OTHERS, tmp_path / YOU]
    assert rec.seconds == pytest.approx(1.0)
    # no stop(): as after a crash, the headers already say how much is there
    for name, frames in ((OTHERS, 10), (YOU, 4)):
        with wave.open(str(tmp_path / name), "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
            assert w.getnframes() == frames * len(FRAME) // 2
            assert w.readframes(w.getnframes()) == FRAME * frames
    rec.stop()
    assert others.stopped and you.stopped
    others.on_frame(FRAME)  # a late frame after stop is dropped, not an error
    with wave.open(str(tmp_path / OTHERS), "rb") as w:
        assert w.getnframes() == 10 * len(FRAME) // 2


def test_recorder_keeps_the_track_that_opens(tmp_path):
    others = FakeCapture()
    rec = Recorder(tmp_path, speakers=lambda: others, microphone=lambda: FakeCapture(fail=True))
    rec.start()
    assert rec.files == [tmp_path / OTHERS] and not (tmp_path / YOU).exists() and "you.wav" in rec.problems[0]
    rec.stop()


def test_recorder_raises_when_nothing_opens(tmp_path):
    rec = Recorder(tmp_path, speakers=lambda: FakeCapture(fail=True), microphone=lambda: FakeCapture(fail=True))
    with pytest.raises(OSError):
        rec.start()
    assert not list(tmp_path.glob("*.wav"))


# ---- the switches' rules

@pytest.mark.parametrize(("given", "works"), [
    (MeetingOptions(), MeetingOptions()),
    (MeetingOptions(record=False), MeetingOptions(record=False, minutes=False, screen=False)),  # no sound, no minutes
    (MeetingOptions(minutes=False), MeetingOptions(minutes=False, screen=False)),  # frames are only for the minutes
    (MeetingOptions(screen=False), MeetingOptions(screen=False)),
])
def test_options_rules(given, works):
    assert given.effective() == works


def test_nothing_switched_on_cannot_start(tmp_path):
    session, _ = make(tmp_path, MeetingOptions(translate=False, record=False))
    with pytest.raises(ValueError):
        session.start()


def test_only_the_parts_switched_on_run(tmp_path):
    live = FakeLive()
    session, made = make(tmp_path, MeetingOptions(translate=False, minutes=False), live=live)
    session.start()
    assert made["recorder"].started and "frames" not in made and not live.started
    session.stop()
    assert session.state == DONE and session.result is None


# ---- the states

def test_recording_then_minutes_then_done(tmp_path):
    gate, states = threading.Event(), []
    live = FakeLive()

    def summarise(folder):
        gate.wait(5)
        return Result()

    session, made = make(tmp_path, summarise=summarise, live=live)
    session.on_change = lambda: states.append(session.state)
    session.start()
    assert session.state == RECORDING and live.started and made["frames"].started
    assert made["frames"].folder == session.folder and made["recorder"].folder == session.folder
    session.stop()
    assert session.state == WRITING and made["recorder"].stopped and made["frames"].stopped and live.stopped
    gate.set()
    session.wait(5)
    assert session.state == DONE and session.result == Result()
    assert states == [RECORDING, WRITING, DONE]
    session.stop()  # a second stop changes nothing
    assert states == [RECORDING, WRITING, DONE]


def test_minutes_failing_keep_the_recording(tmp_path):
    def summarise(folder):
        raise RuntimeError("Gemini said no")

    session, _ = make(tmp_path, summarise=summarise)
    session.start()
    (session.folder / OTHERS).write_bytes(b"RIFF")
    session.stop()
    session.wait(5)
    assert session.state == FAILED and session.error == "Gemini said no" and session.result is None
    assert (session.folder / OTHERS).exists()


def test_a_part_failing_to_start_stops_the_others(tmp_path):
    class BrokenLive(FakeLive):
        def start(self):
            raise OSError("no key")

    session, made = make(tmp_path, live=BrokenLive())
    with pytest.raises(OSError):
        session.start()
    assert session.state == FAILED and made["recorder"].stopped and made["frames"].stopped


# ---- the cost

def test_cost_adds_up_and_keeps_the_translation_after_stop(tmp_path):
    live, clock = FakeLive(), Clock()
    session, made = make(tmp_path, live=live, clock=clock)
    session.start()
    made["recorder"].seconds = 1800.0
    made["frames"].frames = [(5.0, Path("a.jpg")), (60.0, Path("b.jpg"))]
    live.cost = 0.4
    cost = session.cost()
    assert (cost.seconds, cost.frames, cost.translation) == (1800.0, 2, 0.4)
    assert cost.minutes_estimate == pytest.approx(0.125 + 0.002)
    assert cost.total == pytest.approx(cost.translation + cost.minutes_estimate)
    session.stop()
    session.wait(5)
    assert session.cost().translation == 0.4  # the stopped bar has no spend, the meeting remembers it


def test_cost_without_recording_counts_the_clock_and_no_minutes(tmp_path):
    live, clock = FakeLive(), Clock()
    session, _ = make(tmp_path, MeetingOptions(record=False), live=live, clock=clock)
    session.start()
    clock.now += 90
    live.cost = 0.05
    cost = session.cost()
    assert (cost.seconds, cost.minutes_estimate, cost.total, cost.frames) == (90.0, 0.0, 0.05, 0)


# ---- the meeting window closing ends the meeting

def test_the_window_gone_for_a_while_stops_the_meeting(tmp_path):
    clock, here = Clock(), [True]
    session, made = make(tmp_path, alive=lambda w: here[0], clock=clock, summarise=lambda f: Result())
    session.start()
    session.check()
    here[0] = False
    session.check()
    clock.now += GONE_S / 2
    here[0] = True  # back (Teams swapping windows): the count starts again
    session.check()
    here[0] = False
    session.check()
    clock.now += GONE_S - 1
    session.check()
    assert session.state == RECORDING
    clock.now += 1
    session.check()
    session.wait(5)
    assert made["recorder"].stopped and session.state == DONE


def test_the_watcher_thread_looks_by_itself(tmp_path):
    clock, made = Clock(), {}

    def recorder(folder):
        made["r"] = FakeRecorder(folder)
        return made["r"]

    session = MeetingSession(Window(), MeetingOptions(minutes=False, translate=False), tmp_path, recorder_factory=recorder,
                             frames_factory=FakeFrames, live=None, summarise=lambda f: Result(), clock=clock,
                             alive=lambda w: False, estimate=lambda s, f: 0.0, watch=0.01)
    stopped = threading.Event()
    session.on_change = lambda: session.state == DONE and stopped.set()
    session.start()
    session.check()  # the first look sees it gone
    clock.now += GONE_S
    assert stopped.wait(5) and made["r"].stopped


# ---- meeting.json: numbers and names only

def test_meeting_json_holds_no_words(tmp_path):
    session, made = make(tmp_path)
    session.start()
    made["recorder"].seconds = 125.04
    made["frames"].frames = [(35.0, Path("frames/0001 00m35s.jpg"))]
    session.stop()
    session.wait(5)
    text = (session.folder / META).read_text(encoding="utf-8")
    meta = json.loads(text)
    assert set(meta) == {"app", "title", "started", "seconds", "options", "frames"}
    assert meta["app"] == "Microsoft Teams" and meta["seconds"] == 125.0 and meta["frames"] == 1
    assert meta["options"] == {"translate": True, "record": True, "minutes": True, "screen": True}
    assert "secret" not in text and "Friday" not in text  # the minutes' words stay in the minutes


# ---- the rule: a meeting is not dictation

FORBIDDEN = ("sst.pipeline", "sst.dictate", "sst.audio")


def test_meeting_imports_nothing_from_dictation():
    folder = Path(sst.meeting.__file__).parent
    for path in folder.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                assert not name.startswith(FORBIDDEN), f"{path.name} imports {name}: keep meeting mode apart"
