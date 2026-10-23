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
from sst.wake.enrol import LEAD, TAIL  # noqa: E402
from sst.wakeui import BLUETOOTH, FRAME, WakeController, WakeFeeder, WakeSetupDialog, WakeTools, keyword_trim  # noqa: E402


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


# ---- where it listens: a Bluetooth headset's microphone only when the user lets it

def _controller(microphone="", microphones=("Microphone (Realtek)",), bluetooth=(), default="Microphone (Realtek)",
                **settings):
    app = SimpleNamespace(settings=Settings(microphone=microphone, wake_enabled=True, **settings))
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
    assert "Listen on my Bluetooth headset" in BLUETOOTH  # the way to listen there


def test_the_always_on_microphone_doesnt_change_where_it_listens():
    headset = "Headset (Buds)"  # dictation never keeps a headset's microphone open, always-on or not (sst.audio)
    for always_on in (True, False):
        c = _controller(headset, [headset, "Microphone (Realtek)"], bluetooth=[headset], always_on_mic=always_on)
        assert c.choose_microphone() == (False, "Microphone (Realtek)", "Microphone (Realtek)", "")
        assert c.instead == headset  # for the card: not the headset, and why


def test_with_listening_on_the_headset_allowed_it_listens_there():
    headset = "Headset (OnePlus Nord Buds 3r)"
    c = _controller(headset, [headset, "Microphone (Realtek)"], bluetooth=[headset], wake_bluetooth=True)
    assert c.choose_microphone() == (True, headset, headset, "") and c.instead == ""
    c = _controller("", [headset], bluetooth=[headset], default=headset, wake_bluetooth=True)  # the headset is the default
    assert c.choose_microphone() == (True, None, headset, "")


def test_a_turns_trim_takes_the_keyword_and_the_end_phrase_out(wake):
    setup = wake.WakeSetup("okay type", "okay type", None, 9)
    trim = keyword_trim(setup, ["stop it", "end it"])
    assert trim("Okay type, send the report tomorrow. Stop it.") == "send the report tomorrow."
    assert trim("Okay, type the report. End it") == "the report."
    assert trim("Thanks, see you.") == "Thanks, see you."  # neither said: the words as heard


# ---- the setup: ten tries heard

class FakeMic:
    """The setup's microphone: each take records what the test says next (a level above 0.05 is heard: conftest)."""

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


def _dialog(wake, problem="", saved=None, mic=None, enroll=None, hearing=None):
    mic = mic or FakeMic()
    enrollments = []

    def default_enroll(keyword):
        enrollments.append(wake.Enrollment(keyword))
        return enrollments[-1]
    dialog = WakeSetupDialog(None, microphone="Headset (OnePlus Nord Buds 3r)", problem=problem, check=wake.check_keyword,
                             enroll=enroll or default_enroll, new_recorder=lambda: mic, hearing=hearing,
                             save=(saved.append if saved is not None else lambda setup: None))
    return dialog, mic, enrollments


def _begin(dialog, keyword="okay type"):
    dialog.keyword.setText(keyword)
    dialog.start_button.click()
    assert _wait(lambda: dialog.record_button.isEnabled()) and dialog.pages.currentIndex() == 1


def _take(dialog, mic, heard: bool) -> None:
    mic.next = 0.2 if heard else 0.01  # 0.01: -40 dBFS, loud enough, but the fake spotter doesn't hear it
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


def test_the_setup_says_which_microphone_it_listens_on(wake):
    dialog, _, _ = _dialog(wake)
    assert dialog.mic_words.text() == "Listening on Headset (OnePlus Nord Buds 3r)"
    assert dialog.try_mic.text() == "Listening on Headset (OnePlus Nord Buds 3r)"
    dialog.done(0)


def test_no_microphone_to_record_on_says_why_and_doesnt_start(wake):
    dialog, _, _ = _dialog(wake, problem=BLUETOOTH)
    dialog.keyword.setText("okay type")
    assert not dialog.start_button.isEnabled() and dialog.mic_words.text() == BLUETOOTH
    dialog.done(0)


def _hearing(results):
    """WakeTools.hearing, faked: each keyword's (Hearing, suggestion), and the threads it ran on."""
    import threading

    from sst.wake import Hearing
    calls = []

    def hearing(keyword):
        calls.append((keyword, threading.current_thread().name))
        heard, suggestion = results.get(keyword, (5, None))
        return Hearing(keyword, keyword, heard, 5), suggestion
    return hearing, calls


def test_a_keyword_the_spotter_cant_hear_is_said_at_once_with_a_phrase_to_try(wake):
    hearing, calls = _hearing({"karthi": (0, "hello karthi")})
    dialog, _, _ = _dialog(wake, hearing=hearing)
    for typed in ("k", "kar", "karthi"):  # typed quickly: checked once, when the typing stops
        dialog.keyword.setText(typed)
    assert _wait(lambda: "can't recognise" in dialog.warning.text())
    assert calls == [("karthi", "wake-setup")]  # on a thread: the window doesn't wait for it
    assert dialog.warning.text() == ("Rflow can't recognise “karthi” reliably. Try a phrase of 2-3 words, like "
                                     "“hello karthi” or “okay type”.")
    assert dialog.suggestion.isVisibleTo(dialog) and dialog.suggestion.text() == "hello karthi"
    assert dialog.start_button.isEnabled()  # the synthetic voice isn't the user's: they may still try
    dialog.suggestion.click()
    assert dialog.keyword.text() == "hello karthi"
    assert _wait(lambda: calls[-1][0] == "hello karthi") and _wait(lambda: dialog.warning.isHidden())
    assert not dialog.suggestion.isVisibleTo(dialog)
    dialog.done(0)


def test_with_no_phrase_heard_the_advice_is_a_keyword_known_to_work(wake):
    hearing, _ = _hearing({"karthi": (0, "okay type")})
    dialog, _, _ = _dialog(wake, hearing=hearing)
    dialog.keyword.setText("karthi")
    assert _wait(lambda: dialog.warning.text() == "Rflow can't recognise “karthi” reliably. Try a phrase of 2-3 words, "
                                                  "like “okay type”.")
    assert not dialog.suggestion.isVisibleTo(dialog)  # "okay type" has its own chip already
    dialog.done(0)


def test_without_the_synthetic_voice_there_is_no_check(wake):
    calls = []
    dialog, _, _ = _dialog(wake, hearing=lambda keyword: calls.append(keyword))  # None: the voice isn't downloaded
    dialog.keyword.setText("karthi")
    assert _wait(lambda: calls == ["karthi"]) and _wait(lambda: not dialog._checking)
    assert dialog.warning.isHidden() and dialog.start_button.isEnabled()
    dialog.done(0)


def test_a_take_not_heard_stays_on_the_same_try(wake):
    dialog, mic, enrollments = _dialog(wake)
    _begin(dialog)
    assert mic.opened and mic.warm_seconds == float("inf")  # open from now on: each take starts at once
    prompts = wake.prompts("okay type")
    assert dialog.counter.text() == "Try 1 of 10" and dialog.prompt_label.text() == prompts[0]
    _take(dialog, mic, heard=False)
    assert dialog.verdict.text() == "Not heard: say it again" and dialog.prompt_label.text() == prompts[0]
    assert dialog.counter.text() == "Try 1 of 10" and dialog.record_button.text() == "Try again"  # not used up
    assert [d.state for d in dialog.dots[:2]] == ["err", "off"] and dialog.hint.isHidden()
    _take(dialog, mic, heard=True)
    assert dialog.verdict.text() == "Heard" and dialog.prompt_label.text() == prompts[1]
    assert dialog.counter.text() == "Try 2 of 10" and [d.state for d in dialog.dots[:2]] == ["ok", "off"]
    assert enrollments[0].said == 2 and enrollments[0].tries == 1
    dialog.done(0)


def test_three_misses_in_a_row_bring_advice_and_another_keyword(wake):
    dialog, mic, _ = _dialog(wake)
    _begin(dialog)
    _take(dialog, mic, heard=False)
    _take(dialog, mic, heard=False)
    assert dialog.hint.isHidden() and dialog.change_button.isHidden()
    _take(dialog, mic, heard=False)
    assert dialog.hint.isVisibleTo(dialog) and "louder or closer" in dialog.hint.text()
    assert "Headset (OnePlus Nord Buds 3r)" in dialog.hint.text() and dialog.counter.text() == "Try 1 of 10"
    assert dialog.change_button.isVisibleTo(dialog)
    dialog.change_button.click()  # another keyword
    assert dialog.pages.currentIndex() == 0 and mic.closed and dialog.start_button.isEnabled()
    dialog.done(0)


def test_a_take_too_quiet_isnt_counted_and_says_so(wake, caplog):
    import logging
    dialog, mic, enrollments = _dialog(wake)
    _begin(dialog)
    mic.next = 0.001  # -60 dBFS: a microphone far away
    with caplog.at_level(logging.INFO, logger="sst.wakeui"):
        dialog.record_button.click()
        dialog.record_button.click()
    assert dialog.verdict.text() == "Too quiet: speak closer or choose another microphone."
    assert enrollments[0].said == 0 and dialog.counter.text() == "Try 1 of 10" and dialog.missed == 1
    assert "-60 dBFS" in caplog.text and "Headset (OnePlus Nord Buds 3r)" in caplog.text
    dialog.done(0)


def test_ten_heard_tries_are_saved_however_many_takes_it_took(wake):
    saved = []
    dialog, mic, enrollments = _dialog(wake, saved=saved)
    _begin(dialog)
    for heard in [True, False, True, True, False, True, True, True, True, True, True, True]:
        _take(dialog, mic, heard)
    assert _wait(lambda: dialog.result_title.text() == "Saved")
    assert [(s.heard, s.said) for s in saved] == [(10, 12)] and saved[0].keyword == "okay type"
    assert dialog.result_words.text() == "Rflow heard “okay type” 10 of the 12 times you said it."
    assert len(enrollments[0]._takes[0]) == 16_000 + int((LEAD + TAIL) * 16_000)  # at 16 kHz, from the mic's 48
    assert mic.closed  # the microphone is let go of once the tries are done
    dialog.done_button.click()
    assert dialog.result() == 1


def test_a_keyword_heard_as_different_words_is_refused_with_advice(wake):
    class Split:  # ten takes heard, but by different spellings: none of them 8 times
        prompts, heard, said, done, tries = wake.prompts("computer"), 5, 10, False, 0

        def add(self, audio):
            self.tries += 1
            self.done = self.tries == 10
            return True

        def result(self):
            return None
    saved = []
    dialog, mic, _ = _dialog(wake, saved=saved, enroll=lambda keyword: Split())
    _begin(dialog, "computer")
    for _ in range(10):
        _take(dialog, mic, True)
    assert _wait(lambda: dialog.result_title.text() == "Not saved")
    assert saved == [] and "as different words" in dialog.result_words.text() and "5 of the 10" in dialog.result_words.text()
    assert "2-3 words" in dialog.result_advice.text() and "quieter room" in dialog.result_advice.text()
    dialog.again_button.click()  # another keyword
    assert dialog.pages.currentIndex() == 0 and dialog.start_button.isEnabled()
    dialog.done(0)


def test_a_take_too_short_doesnt_count_and_closing_lets_the_microphone_go(wake):
    mic = FakeMic()
    dialog, _, _ = _dialog(wake, mic=mic)
    _begin(dialog)
    mic.stop = lambda: np.zeros(1000, dtype=np.float32)
    dialog.record_button.click()
    dialog.record_button.click()
    assert dialog.results == [] and "Too short" in dialog.verdict.text()
    dialog.record_button.click()  # recording when the window is closed
    dialog.reject()
    assert mic.closed and not dialog.recording


# ---- what the setup borrows: the recogniser for its tries, the voice for the check

class FakeApi:
    def __init__(self):
        self.transcribers, self.checks = [], []

    def Transcriber(self):  # noqa: N802 - sst.wake's class
        self.transcribers.append(lambda audio: "CARTHY")
        return self.transcribers[-1]

    Spotter = object()

    def check_hearing(self, keyword, say, spotter_factory, transcribe):
        self.checks.append((keyword, say("Well, karthi said yes.", 1.0), spotter_factory, transcribe))
        return "heard"


class FakeVoice:
    rate = 22_050

    def synthesize(self, text, speed):
        return np.zeros(22_050, np.float64)  # a second


def test_the_tools_load_what_they_need_once_and_keep_the_check_apart():
    api, made = FakeApi(), []
    tools = WakeTools(api, lambda: made.append(FakeVoice()) or made[-1])
    assert api.transcribers == [] and made == []  # nothing loaded before it's needed
    assert tools.transcribe(np.zeros(10)) == "CARTHY" and tools.transcribe(np.zeros(10)) == "CARTHY"
    assert len(api.transcribers) == 1
    assert tools.hearing("karthi") == "heard" and tools.hearing("hey karthi") == "heard"
    assert len(made) == 1 and len(api.transcribers) == 2  # the check's own recogniser: a try needn't wait for it
    keyword, audio, spotter, transcribe = api.checks[0]
    assert keyword == "karthi" and spotter is api.Spotter and transcribe is api.transcribers[1]
    assert audio.dtype == np.float32 and len(audio) == 16_000  # said at the voice's rate, heard at 16 kHz


def test_without_the_voice_downloaded_the_check_is_skipped():
    api, asked = FakeApi(), []
    tools = WakeTools(api, lambda: asked.append(1))
    assert tools.hearing("karthi") is None and tools.hearing("karthi") is None
    assert asked == [1] and api.checks == []  # looked for once


def test_wake_api_is_imported_when_needed(wake):
    assert wakeui.wake_api() is wake
