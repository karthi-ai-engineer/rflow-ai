"""The dictation logic: hold, tap, Ctrl+Win+Space, cancel, interrupt, auto-stop. The microphone, engine and
typing are fakes, and hotkey events are fed in with explicit times, so the tests are exact and fast:
no keys are pressed, the clipboard is untouched and no model is needed."""
import numpy as np
import pytest

from sst import dictate
from sst.dictate import Dictation

TAP = 0.05  # released this soon after the press = a tap


class FakeRecorder:
    rate = 16_000

    def __init__(self, seconds=1.0, fail=False):
        self.seconds, self.fail, self.starts = seconds, fail, 0

    def start(self):
        if self.fail:
            raise OSError("no microphone")
        self.starts += 1

    def stop(self):
        return np.full(int(self.seconds * self.rate), 0.1, dtype=np.float32)


class FakeEngine:
    name = "fake"

    def __init__(self, text="hello world"):
        self.text = text

    def transcribe(self, audio, rate):
        return self.text


class FakeListener:
    recording = False


@pytest.fixture
def make():
    def make(text="hello world", seconds=1.0, fail=False):
        typed, states = [], []
        d = Dictation(FakeEngine(text), FakeRecorder(seconds, fail), paste=typed.append, sounds=False, save=False)
        d.listener = FakeListener()
        d.on_state = lambda state, message: states.append(state)
        return d, typed, states
    return make


def feed(d, *events):
    for at, event in events:
        d.handle(event, at)
    d.wait()


def test_hold_then_release_types_the_text(make):
    d, typed, states = make()
    feed(d, (0, "press"), (0.7, "release"))
    assert typed == ["hello world "] and states == ["recording", "transcribing", "typed"]


def test_tap_then_tap_types_the_text(make):
    d, typed, _ = make()
    feed(d, (0, "press"), (TAP, "release"), (5.0, "press"), (5.05, "release"))
    assert typed == ["hello world "]


def test_a_tap_keeps_recording_until_the_next_press(make):
    d, typed, _ = make()
    feed(d, (0, "press"), (TAP, "release"))
    assert d.recording and typed == []


def test_esc_is_captured_only_while_recording(make):
    d, typed, states = make()
    feed(d, (0, "press"))
    assert d.listener.recording
    feed(d, (TAP, "release"), (0.3, "cancel"))
    assert not d.listener.recording and typed == [] and states[-1] == "cancelled"


def test_a_windows_shortcut_drops_the_recording_quietly(make):
    d, typed, states = make()
    feed(d, (0, "press"), (0.2, "interrupt"))
    assert typed == [] and states == ["recording", "idle"]
    feed(d, (0.5, "press"), (1.2, "release"))  # the next hold works normally
    assert typed == ["hello world "]


def test_ctrl_win_space_records_hands_free_until_the_next_press(make):
    d, typed, _ = make()
    feed(d, (0, "press"), (0.01, "handsfree"))
    assert d.recording  # the keys were let go, but this is hands-free
    feed(d, (4.0, "press"), (4.05, "release"))
    assert typed == ["hello world "] and d.recorder.starts == 1


def test_ctrl_win_space_again_stops_without_starting_a_new_recording(make):
    d, typed, _ = make()
    feed(d, (0, "press"), (0.01, "handsfree"), (4.0, "press"), (4.01, "handsfree"))
    assert typed == ["hello world "] and not d.recording and d.recorder.starts == 1


def test_space_while_holding_keeps_recording_after_the_keys_are_released(make):
    d, typed, _ = make()
    feed(d, (0, "press"), (0.3, "handsfree"))
    assert d.recording
    feed(d, (3.0, "press"))
    assert typed == ["hello world "] and d.recorder.starts == 1


def test_recording_stops_by_itself_at_the_limit(make):
    d, typed, states = make()
    feed(d, (0, "press"), (TAP, "release"))
    d.tick(dictate.MAX_SECONDS - 1)
    assert d.recording
    d.tick(dictate.MAX_SECONDS + 1)
    d.wait()
    assert typed == ["hello world "] and "warning" in states


def test_accidental_short_press_is_ignored(make):
    d, typed, states = make(seconds=0.1)
    feed(d, (0, "press"), (0.5, "release"))
    assert typed == [] and states[-1] == "ignored"


def test_nothing_is_typed_when_nothing_is_recognised(make):
    d, typed, states = make(text="")
    feed(d, (0, "press"), (0.5, "release"))
    assert typed == [] and states[-1] == "idle"


def test_a_missing_microphone_is_reported_not_raised(make):
    d, typed, states = make(fail=True)
    feed(d, (0, "press"))
    assert not d.recording and states == ["error"]


def test_a_failing_paste_is_reported_and_the_next_dictation_still_works(make):
    d, typed, states = make()
    calls = []

    def flaky_paste(text):
        calls.append(text)
        if len(calls) == 1:
            raise OSError("the clipboard is busy")
        typed.append(text)

    d.paste = flaky_paste
    feed(d, (0, "press"), (0.7, "release"))
    assert states[-1] == "error"
    feed(d, (2, "press"), (2.7, "release"))
    assert typed == ["hello world "]


class FakeCleanup:
    def __init__(self, result="Hello, world.", error=""):
        self.result, self.error, self.prepared, self.last_error = result, error, 0, ""

    def prepare(self):
        self.prepared += 1

    def polish(self, text):
        self.last_error = self.error
        return text if self.error else self.result


def test_the_cleaned_text_is_typed_and_both_texts_are_reported(make):
    d, typed, states = make()
    results = []
    d.cleanup = FakeCleanup()
    d.on_result = lambda heard, text, seconds: results.append((heard, text, seconds))
    feed(d, (0, "press"), (0.7, "release"))
    assert typed == ["Hello, world. "] and states[-1] == "typed"
    assert results == [("hello world", "Hello, world.", 1.0)]  # with the recording's length, for the speaking speed
    assert d.cleanup.prepared == 1  # the gateway connection was opened while the user was speaking


def test_when_the_cleanup_cannot_help_the_heard_text_is_typed(make):
    d, typed, states = make()
    d.cleanup = FakeCleanup(error="the gateway could not be reached")
    feed(d, (0, "press"), (0.7, "release"))
    assert typed == ["hello world "] and states[-1] == "typed_raw"


class CloudLikeEngine(FakeEngine):
    """Like sst.engines.cloud.CloudEngine: connects while the user speaks, and says when Parakeet stood in for it."""

    def __init__(self, error=""):
        super().__init__()
        self.error, self.prepared, self.last_error = error, 0, ""

    def prepare(self):
        self.prepared += 1

    def transcribe(self, audio, rate):
        self.last_error = self.error
        return self.text


def test_a_cloud_speech_model_connects_while_the_user_speaks(make):
    d, typed, states = make()
    d.engine = CloudLikeEngine()
    feed(d, (0, "press"))
    assert d.engine.prepared == 1
    feed(d, (0.7, "release"))
    assert typed == ["hello world "] and states[-1] == "typed"


def test_when_the_cloud_cannot_help_parakeets_text_is_typed_and_said_so(make):
    d, typed, _ = make()
    d.engine = CloudLikeEngine(error="OpenAI: cannot reach api.openai.com")
    d.cleanup = FakeCleanup(error="the gateway could not be reached")
    seen = []
    d.on_state = lambda state, message: seen.append((state, message))
    feed(d, (0, "press"), (0.7, "release"))
    assert typed == ["hello world "] and seen[-1] == ("typed_local", "OpenAI: cannot reach api.openai.com")


class WarmRecorder:
    """Like the real Recorder: stop_later() hands over a take whose tail arrives later, and tick() is passed on."""
    rate = 16_000

    def __init__(self, seconds=1.0, preroll=0.4):
        from sst.audio import Take
        self.Take, self.seconds, self.preroll = Take, seconds, preroll
        self.ticks, self.pending, self.closed = [], [], False

    def start(self):
        pass

    def stop_later(self):
        take = self.Take([np.full(int(self.preroll * self.rate), 0.1, dtype=np.float32)], self.rate)
        take.chunks.append(np.full(int(self.seconds * self.rate), 0.1, dtype=np.float32))
        self.pending.append(take)
        return take

    def tick(self, now):
        self.ticks.append(now)

    def close(self):
        self.closed = True


def test_the_text_is_typed_once_the_tail_after_the_release_has_arrived():
    typed = []
    recorder = WarmRecorder()
    d = Dictation(FakeEngine(), recorder, paste=typed.append, sounds=False, save=False)
    d.handle("press", 0.0)
    d.handle("release", 1.0)
    assert typed == [] and len(recorder.pending) == 1  # the worker waits for the tail
    recorder.pending[0].done.set()
    d.wait()
    assert typed == ["hello world "]


def test_the_lead_in_does_not_make_an_accidental_tap_count():
    states = []
    d = Dictation(FakeEngine(), WarmRecorder(seconds=0.1), paste=lambda text: None, sounds=False, save=False)
    d.on_state = lambda state, message: states.append(state)
    d.handle("press", 0.0)
    d.handle("press", 0.1)  # a quick tap on, tap off: 0.1 s said, plus 0.4 s from before the press
    assert states[-1] == "ignored"


def test_the_recorder_is_ticked_and_closed_with_the_dictation():
    recorder = WarmRecorder()
    d = Dictation(FakeEngine(), recorder, paste=lambda text: None, sounds=False, save=False)
    d.tick(5.0)
    d.close()
    assert recorder.ticks == [5.0] and recorder.closed


class FailingEngine(FakeEngine):
    def transcribe(self, audio, rate):
        raise RuntimeError("OpenAI: cannot reach api.openai.com (Parakeet isn't downloaded to take over)")


def test_a_failed_transcription_keeps_the_recording(monkeypatch):
    kept, seen = [], []
    monkeypatch.setattr(dictate, "save_recording", lambda audio, rate, text: kept.append((len(audio), text)) or "x")
    d = Dictation(FailingEngine(), FakeRecorder(), paste=lambda text: None, sounds=False, save=True)
    d.on_state = lambda state, message: seen.append((state, message))
    feed(d, (0, "press"), (0.7, "release"))
    assert kept == [(16_000, "")] and seen[-1][0] == "error" and "recording is kept" in seen[-1][1]


def test_a_voice_command_is_carried_out_not_typed(make):
    d, typed, states = make(text="Make it concise.")
    commands, results = [], []
    d.command = lambda text: "concise" if text == "Make it concise." else None
    d.on_command = commands.append
    d.on_result = lambda heard, text, seconds: results.append(text)
    feed(d, (0, "press"), (0.7, "release"))
    assert commands == ["concise"] and typed == [] and results == [] and states[-1] == "idle"
    d.engine.text = "Make it concise and send it."
    feed(d, (2, "press"), (2.7, "release"))
    assert commands == ["concise"] and typed == ["Make it concise and send it. "]


def test_the_classic_way_types_a_snippet_said_alone(make):
    from sst.snippets import Snippet
    d, typed, states = make(text="My email.")
    d.snippets = lambda: [Snippet("my email", "xyz@gmail.com")]
    d.cleanup = FakeCleanup(result="Something else.")
    d.command = lambda text: "concise"  # a snippet wins over a voice command
    feed(d, (0, "press"), (0.7, "release"))
    assert typed == ["xyz@gmail.com "]


@pytest.mark.parametrize("keeps, expected", [(True, "Send it to xyz@gmail.com. "), (False, "send it to xyz@gmail.com ")])
def test_the_classic_way_puts_an_anywhere_snippet_in_after_the_cleanup(make, keeps, expected):
    from sst.snippets import Snippet
    d, typed, _ = make(text="send it to my email")
    d.snippets = lambda: [Snippet("my email", "xyz@gmail.com", anywhere=True)]
    cleanup = FakeCleanup()
    cleanup.polish = lambda text: (text.capitalize().replace("rfsnip1", "RFSNIP1") + ".") if keeps else "Send it to me."
    d.cleanup = cleanup
    feed(d, (0, "press"), (0.7, "release"))
    assert typed == [expected]  # a cleanup that lost the placeholder: the words heard, with the snippet


class DeadRecorder(FakeRecorder):
    """A microphone that opens but sends no sound (unplugged, muted, or switched off when a headset was plugged in)."""
    notice = ""

    def stop(self):
        return np.zeros(int(self.seconds * self.rate), dtype=np.float32)

    def describe(self):
        return {"device": "Headset (Buds)"}


class CountingEngine(FakeEngine):
    calls = 0

    def transcribe(self, audio, rate):
        self.calls += 1
        return super().transcribe(audio, rate)


def test_a_recording_with_no_sound_is_not_transcribed_and_names_the_microphone():
    typed, states = [], []
    engine = CountingEngine()
    d = Dictation(engine, DeadRecorder(), paste=typed.append, sounds=False, save=False)
    d.listener = FakeListener()
    d.on_state = lambda state, message: states.append((state, message))
    feed(d, (0, "press"), (0.7, "release"))
    assert typed == [] and engine.calls == 0  # nothing sent to the speech model
    state, message = states[-1]
    assert state == "error" and "No sound came from the microphone (Headset (Buds))" in message


def test_the_recorders_notice_is_shown_once_per_change(make):
    d, typed, _ = make()
    notices = []
    d.on_notice = notices.append
    d.recorder.notice = "Headset (Buds) isn't connected: Rflow uses Microphone (Realtek(R) Audio) until it is."
    feed(d, (0, "press"), (0.7, "release"))
    feed(d, (2, "press"), (2.7, "release"))
    assert notices == [d.recorder.notice]  # once, not at every dictation
    d.recorder.notice = ""  # connected again
    feed(d, (4, "press"), (4.7, "release"))
    d.recorder.notice = "Headset (Buds) isn't connected: Rflow uses Microphone (Realtek(R) Audio) until it is."
    feed(d, (6, "press"), (6.7, "release"))
    assert len(notices) == 2 and typed == ["hello world "] * 4  # unplugged again: said again; dictation goes on


def test_a_key_brushed_while_speaking_keeps_the_dictation(make):
    d, typed, states = make()
    feed(d, (0, "press"), (1.5, "interrupt"), (2.0, "interrupt"), (4.0, "release"))  # Shift brushed twice, then let go
    assert typed == ["hello world "] and states == ["recording", "transcribing", "typed"]


def test_a_dictation_that_cant_be_typed_still_goes_to_the_history():
    results, messages = [], []

    def busy_clipboard(text):
        raise OSError("the clipboard is busy (another app is holding it open)")
    d = Dictation(FakeEngine(), FakeRecorder(), paste=busy_clipboard, sounds=False, save=False)
    d.listener = FakeListener()
    d.on_state = lambda state, message: messages.append((state, message))
    d.on_result = lambda heard, typed, seconds: results.append(typed)
    feed(d, (0, "press"), (0.7, "release"))
    assert results == ["hello world"]  # in the history, on Home
    assert messages[-1] == ("error", "Couldn't type it here (the clipboard is busy (another app is holding it open)). "
                                     "It's on Home, ready to copy.")


def test_busy_while_recording_or_typing(make):
    d, _, _ = make()
    assert not d.busy
    feed(d, (0, "press"))
    assert d.busy
    feed(d, (0.7, "release"))
    assert not d.busy


# ---- the wake word's turns (sst.wakeui): started hands-free, ended from outside, the keyword trimmed

def _without_keyword(text: str) -> str:
    return text.removeprefix("Okay type, ").removesuffix(" Stop it.")


def test_a_trim_set_for_one_recording_cuts_the_keyword_before_commands_and_cleanup(make):
    d, typed, _ = make(text="Okay type, hello there. Stop it.")
    heard, cleaned = [], []
    d.cleanup = FakeCleanup()
    d.cleanup.polish = lambda text: cleaned.append(text) or text
    d.on_result = lambda h, text, seconds: heard.append(h)
    d.trim = _without_keyword
    d.handle("handsfree", 10.0)  # what the wake word does: hands-free, like Ctrl+Win+Space
    d.trim = None  # the app clears it once the recording has started: it went with this one
    assert d.recording
    d.end(keep=True)
    d.wait()
    assert typed == ["hello there. "] and cleaned == ["hello there."] and heard == ["hello there."]
    feed(d, (20.0, "press"), (20.7, "release"))  # the next recording, with the hotkey: as heard
    assert typed[-1] == "Okay type, hello there. Stop it. "


def test_a_command_said_after_the_keyword_is_still_a_command(make):
    d, typed, states = make(text="Okay type, make it concise.")
    commands = []
    d.command = lambda text: "concise" if text == "make it concise." else None
    d.on_command = commands.append
    d.trim = _without_keyword
    d.handle("handsfree", 10.0)
    d.end(keep=True)
    d.wait()
    assert commands == ["concise"] and typed == []


def test_a_trim_that_fails_types_the_words_as_heard(make):
    d, typed, _ = make(text="Okay type, hello.")
    d.trim = lambda text: 1 / 0
    d.handle("handsfree", 10.0)
    d.end(keep=True)
    d.wait()
    assert typed == ["Okay type, hello. "]


def test_ending_from_outside_keeps_or_drops_quietly(make):
    d, typed, states = make()
    d.handle("handsfree", 10.0)
    d.end(keep=False, quiet=True)  # the voice check failed: nothing typed, no beep, no "Cancelled"
    d.wait()
    assert typed == [] and states == ["recording", "idle"] and not d.listener.recording
    d.end(keep=True)  # nothing recording: nothing happens
    assert states == ["recording", "idle"]
