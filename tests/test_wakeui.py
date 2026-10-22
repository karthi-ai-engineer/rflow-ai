"""The wake word in the app (sst.wakeui): the listener's thread and frames, where it listens (never a Bluetooth headset's
microphone), the keyword trimmed, and the 10-try setup. sst.wake is the real one with its models faked (conftest's `wake`);
no microphone, no real keys: the setup's recorder is a fake and the dialog is built off-screen."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst import wakeui  # noqa: E402
from sst.settings import Settings  # noqa: E402
from sst.wakeui import BLUETOOTH, FRAME, WakeController, WakeFeeder, WakeSetupDialog, keyword_trim  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qt():
    return QApplication.instance() or QApplication([])


def _wait(condition, ms: int = 3000) -> bool:
    for _ in range(ms // 10):
        if condition():
            return True
        QTest.qWait(10)
    return condition()


# ---- the listener's thread: 16 kHz frames of 100 ms, paused, ended between frames

class Listener:
    def __init__(self):
        self.frames, self.ended, self.fail = [], 0, False

    def feed(self, frame, now):
        if self.fail:
            raise RuntimeError("the spotter failed")
        self.frames.append((frame, now))

    def end(self):
        self.ended += 1


def test_the_microphones_blocks_become_100_ms_frames_at_16_khz_on_the_audios_clock():
    listener = Listener()
    feeder = WakeFeeder(listener, clock=lambda: 50.0)
    for _ in range(100):  # one second at 48 kHz, in 10 ms blocks
        feeder.handle(np.full(480, 0.1, dtype=np.float32), 48_000)
    feeder.handle(np.full(240, 0.1, dtype=np.float32), 48_000)  # and 5 ms more: not a whole frame, it waits
    assert len(listener.frames) == 10
    assert all(len(frame) == FRAME and frame.dtype == np.float32 for frame, _ in listener.frames)
    times = [round(now, 2) for _, now in listener.frames]
    assert times == [round(50.1 + i / 10, 2) for i in range(10)]  # spaced as the audio is, however it arrived


def test_paused_frames_are_dropped_and_the_listener_starts_afresh_after():
    listener = Listener()
    feeder = WakeFeeder(listener)
    feeder.paused = True
    for _ in range(20):
        feeder.handle(np.zeros(1600, dtype=np.float32), 16_000)
    assert listener.frames == [] and listener.ended == 0
    feeder.paused = False
    feeder.handle(np.zeros(1600, dtype=np.float32), 16_000)
    assert listener.ended == 1 and len(listener.frames) == 1  # what was said meanwhile wasn't heard


def test_the_app_ends_a_turn_between_two_frames_and_a_failing_listener_keeps_listening():
    listener = Listener()
    feeder = WakeFeeder(listener)
    feeder.end()  # asked on the UI thread
    assert listener.ended == 0
    feeder.handle(np.zeros(1600, dtype=np.float32), 16_000)
    assert listener.ended == 1 and len(listener.frames) == 1
    listener.fail = True
    feeder.handle(np.zeros(1600, dtype=np.float32), 16_000)
    listener.fail = False
    feeder.handle(np.zeros(1600, dtype=np.float32), 16_000)
    assert len(listener.frames) == 2


def test_the_thread_takes_blocks_from_the_audio_callback():
    listener = Listener()
    feeder = WakeFeeder(listener)
    feeder.start()
    try:
        for _ in range(10):
            feeder.put(np.zeros(1600, dtype=np.float32), 16_000)  # Recorder.tap, on the audio thread
        assert _wait(lambda: len(listener.frames) == 10)
        feeder.end()  # with no audio coming, the end still reaches the listener
        assert _wait(lambda: listener.ended == 1)
    finally:
        feeder.stop()


# ---- where it listens: a Bluetooth headset's microphone is never kept open

def _controller(microphone="", microphones=("Microphone (Realtek)",), bluetooth=(), default="Microphone (Realtek)"):
    app = SimpleNamespace(settings=Settings(microphone=microphone, wake_enabled=True))
    return WakeController(app, microphones=lambda: list(microphones), default_microphone=lambda: default,
                          call_quality=lambda device: (device or default) in bluetooth)


def test_dictations_own_microphone_listens_when_it_isnt_bluetooth():
    assert _controller().choose_microphone() == (True, None, "Microphone (Realtek)", "")
    assert _controller("Mic B").choose_microphone() == (True, "Mic B", "Mic B", "")


def test_with_a_bluetooth_headset_chosen_another_microphone_listens():
    headset = "Headset (Buds)"
    c = _controller(headset, [headset, "USB mic", "Microphone (Realtek)"], bluetooth=[headset])
    assert c.choose_microphone() == (False, "Microphone (Realtek)", "Microphone (Realtek)", "")  # the default first
    c = _controller("", [headset, "USB mic"], bluetooth=[headset], default=headset)  # the headset is the default
    assert c.choose_microphone() == (False, "USB mic", "USB mic", "")


def test_with_only_a_bluetooth_headset_it_doesnt_listen_and_says_why():
    headset = "Headset (Buds)"
    assert _controller(headset, [headset], bluetooth=[headset]).choose_microphone() == (False, None, "", BLUETOOTH)


def test_a_turns_trim_takes_the_keyword_and_the_end_phrase_out(wake):
    setup = wake.WakeSetup("okay type", "okay type", None, 9)
    trim = keyword_trim(setup, ["stop it", "end it"])
    assert trim("Okay type, send the report tomorrow. Stop it.") == "send the report tomorrow."
    assert trim("Okay, type the report. End it") == "the report."
    assert trim("Thanks, see you.") == "Thanks, see you."  # neither said: the words as heard


# ---- the 10-try setup

class FakeMic:
    """The setup's microphone: each try records what the test says next (a level above 0.05 is heard: conftest)."""

    def __init__(self, rate=48_000):
        self.rate, self.level, self.warm_seconds = rate, 0.0, 0.0
        self.next = 0.2
        self.opened = self.closed = False
        self.starts = 0

    def keep_open(self):
        self.opened = True

    def start(self):
        self.starts += 1

    def stop(self):
        return np.full(self.rate, self.next, dtype=np.float32)  # a second

    def close(self):
        self.closed = True


def _dialog(wake, problem="", saved=None, mic=None):
    mic = mic or FakeMic()
    enrollments = []

    def enroll(keyword):
        enrollments.append(wake.Enrollment(keyword))
        return enrollments[-1]
    dialog = WakeSetupDialog(None, microphone="Microphone (Realtek)", problem=problem, check=wake.check_keyword,
                             enroll=enroll, new_recorder=lambda: mic,
                             save=(saved.append if saved is not None else lambda setup: None))
    return dialog, mic, enrollments


def _try(dialog, mic, heard: bool) -> None:
    mic.next = 0.2 if heard else 0.01
    count = len(dialog.results)
    dialog.record_button.click()  # Record
    assert dialog.recording
    dialog.record_button.click()  # Stop
    assert _wait(lambda: len(dialog.results) == count + 1 or dialog.pages.currentIndex() == 2)


def test_the_keyword_is_checked_as_it_is_typed(wake):
    dialog, _, _ = _dialog(wake)
    assert not dialog.start_button.isEnabled() and dialog.warning.isHidden()  # nothing typed yet
    dialog.keyword.setText("hey")
    assert not dialog.start_button.isEnabled() and "said too often" in dialog.warning.text()
    dialog.keyword.setText("okay type")
    assert dialog.start_button.isEnabled() and dialog.warning.isHidden()
    dialog.done(0)


def test_no_microphone_to_record_on_says_why_and_doesnt_start(wake):
    dialog, _, _ = _dialog(wake, problem=BLUETOOTH)
    dialog.keyword.setText("okay type")
    assert not dialog.start_button.isEnabled() and dialog.mic_words.text() == BLUETOOTH
    dialog.done(0)


def test_each_try_says_heard_or_not_and_a_prompt_not_heard_is_tried_once_more(wake):
    saved = []
    dialog, mic, enrollments = _dialog(wake, saved=saved)
    dialog.keyword.setText("okay type")
    dialog.start_button.click()
    assert _wait(lambda: dialog.record_button.isEnabled()) and dialog.pages.currentIndex() == 1
    assert mic.opened and mic.warm_seconds == float("inf")  # open from now on: each try starts at once
    prompts = wake.prompts("okay type")
    assert dialog.counter.text() == "Try 1 of 10" and dialog.prompt_label.text() == prompts[0]
    _try(dialog, mic, heard=False)
    assert dialog.verdict.text() == "Not heard: try again" and dialog.prompt_label.text() == prompts[0]
    assert dialog.counter.text() == "Try 2 of 10" and dialog.record_button.text() == "Try again"  # it counted
    assert [d.state for d in dialog.dots[:2]] == ["err", "off"]
    _try(dialog, mic, heard=True)
    assert dialog.verdict.text() == "Heard" and dialog.prompt_label.text() == prompts[1]
    _try(dialog, mic, heard=False)
    _try(dialog, mic, heard=False)  # its one retry, not heard either: on to the next
    assert dialog.prompt_label.text() == prompts[2]
    for _ in range(6):
        _try(dialog, mic, heard=True)
    assert _wait(lambda: dialog.result_title.text() == "Not saved")  # 7 of 10: one short
    assert dialog.results == [False, True, False, False] + [True] * 6 and enrollments[0].heard == 7
    assert saved == [] and "only 7 of 10 times" in dialog.result_words.text()
    dialog.done(0)


def test_a_keyword_heard_8_of_10_times_is_saved(wake):
    saved = []
    dialog, mic, enrollments = _dialog(wake, saved=saved)
    dialog.keyword.setText("okay type")
    dialog.start_button.click()
    assert _wait(lambda: dialog.record_button.isEnabled())
    for heard in [True, False, True, True, False, True, True, True, True, True]:
        _try(dialog, mic, heard)
    assert _wait(lambda: dialog.result_title.text() == "Saved")
    assert [s.heard for s in saved] == [8] and saved[0].keyword == "okay type"
    assert dialog.result_words.text() == "Rflow heard “okay type” 8 of 10 times."
    assert all(len(take) == 16_000 for take in enrollments[0]._takes)  # 16 kHz, from the microphone's 48 kHz
    assert mic.closed  # the microphone is let go of once the tries are done
    dialog.done_button.click()
    assert dialog.result() == 1


def test_a_keyword_heard_too_rarely_is_refused_with_advice(wake):
    saved = []
    dialog, mic, _ = _dialog(wake, saved=saved)
    dialog.keyword.setText("computer")
    dialog.start_button.click()
    assert _wait(lambda: dialog.record_button.isEnabled())
    for heard in [True, False, False, True, False, False, True, True, True, True]:
        _try(dialog, mic, heard)
    assert _wait(lambda: dialog.result_title.text() == "Not saved")
    assert saved == [] and "only 6 of 10 times" in dialog.result_words.text()
    assert "longer keyword" in dialog.result_advice.text() and "quieter room" in dialog.result_advice.text()
    dialog.again_button.click()  # another keyword
    assert dialog.pages.currentIndex() == 0 and dialog.start_button.isEnabled()
    dialog.done(0)


def test_a_try_too_short_doesnt_count_and_closing_lets_the_microphone_go(wake):
    mic = FakeMic()
    dialog, _, _ = _dialog(wake, mic=mic)
    dialog.keyword.setText("okay type")
    dialog.start_button.click()
    assert _wait(lambda: dialog.record_button.isEnabled())
    mic.stop = lambda: np.zeros(1000, dtype=np.float32)
    dialog.record_button.click()
    dialog.record_button.click()
    assert dialog.results == [] and "Too short" in dialog.verdict.text()
    dialog.record_button.click()  # recording when the window is closed
    dialog.reject()
    assert mic.closed and not dialog.recording


def test_wake_api_is_imported_when_needed(wake):
    assert wakeui.wake_api() is wake
