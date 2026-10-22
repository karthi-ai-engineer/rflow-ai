"""The live captions session (one or two ways), its transcript, and the rule that keeps live captions apart from
dictation."""
import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import sst.live
from sst.live.contracts import MIC, SYSTEM, Kind, LiveConfig, LiveEvent, already_in, language_name
from sst.live.session import LiveSession, dbfs
from sst.live.speaker import Speaker
from sst.live.transcript import Transcript


class FakeCapture:
    def __init__(self, fail=False):
        self.on_frame, self.fail, self.stopped = None, fail, False

    def start(self, on_frame):
        if self.fail:
            raise OSError("no microphone")
        self.on_frame = on_frame

    def stop(self):
        self.stopped = True


class FakeEngine:
    def __init__(self, on_event):
        self.on_event, self.frames, self.started, self.stopped = on_event, [], False, False

    def start(self):
        self.started = True

    def feed(self, frame):
        self.frames.append(frame)

    def stop(self):
        self.stopped = True


def engine(session, lane):
    return session.lanes[lane][1]


def test_the_session_feeds_the_engine_and_passes_its_events_on(tmp_path):
    shown = []
    capture = FakeCapture()
    session = LiveSession(LiveConfig(target="en"), Transcript(tmp_path, LiveConfig(target="en")), shown.append)
    session.add(SYSTEM, capture, FakeEngine)
    capture.on_frame(b"\0" * 3200)
    assert engine(session, SYSTEM).started and engine(session, SYSTEM).frames == [b"\0" * 3200]
    engine(session, SYSTEM).on_event(LiveEvent(Kind.SOURCE, "今日は"))
    engine(session, SYSTEM).on_event(LiveEvent(Kind.LINE, "Hello today.", source="今日は", seconds=2.4))
    assert [e.kind for e in shown] == [Kind.SOURCE, Kind.LINE]
    assert session.lags == {SYSTEM: [2.4]} and session.lines == {SYSTEM: 1}
    stopped = engine(session, SYSTEM)
    session.stop()
    assert capture.stopped and stopped.stopped and not session.running
    assert session.transcript.lines == 1


def test_a_way_that_cant_start_leaves_nothing_running_and_the_other_goes_on():
    session = LiveSession(LiveConfig())
    session.add(SYSTEM, FakeCapture(), FakeEngine)
    made = []
    with pytest.raises(OSError):
        session.add(MIC, FakeCapture(fail=True), lambda on_event: made.append(FakeEngine(on_event)) or made[-1])
    assert made[0].stopped and list(session.lanes) == [SYSTEM]


def test_why_a_capture_hears_nothing_is_shown_like_an_engines_problem():
    shown = []
    session = LiveSession(LiveConfig(), on_event=shown.append)
    capture = FakeCapture()
    capture.on_problem = None  # a real Capture has it: Windows' sound muted
    session.add(SYSTEM, capture, FakeEngine)
    capture.on_problem("Windows' sound is muted")
    assert [(e.kind, e.text, e.lane) for e in shown] == [(Kind.ERROR, "Windows' sound is muted", SYSTEM)]


def test_each_way_stops_alone():
    session = LiveSession(LiveConfig(source="both"))
    them, me = FakeCapture(), FakeCapture()
    session.add(SYSTEM, them, FakeEngine)
    session.add(MIC, me, FakeEngine)
    session.remove(MIC)
    assert me.stopped and not them.stopped and list(session.lanes) == [SYSTEM]


def test_untranslated_lines_from_the_microphone_are_echo_and_dropped(tmp_path):
    shown = []
    config = LiveConfig(source="both", target="en", mic_target="ja")
    session = LiveSession(config, Transcript(tmp_path, config), shown.append)
    session.add(MIC, FakeCapture(), FakeEngine)
    mic = engine(session, MIC)
    mic.on_event(LiveEvent(Kind.LINE, "", source="本日はよろしくお願いします。", lane=MIC))  # the meeting, from the speakers
    mic.on_event(LiveEvent(Kind.LINE, "予算は来週決めましょう。", source="Let's set the budget next week.", lane=MIC))
    assert [e.text for e in shown] == ["予算は来週決めましょう。"] and session.lines == {MIC: 1}
    assert "本日" not in session.transcript.path.read_text(encoding="utf-8")


def test_with_the_microphone_alone_lines_already_in_its_language_are_shown_and_kept_but_not_spoken(tmp_path):
    shown = []
    config = LiveConfig(source="microphone", mic_target="en")
    session = LiveSession(config, Transcript(tmp_path, config), shown.append)
    session.add(MIC, FakeCapture(), FakeEngine)
    speaker = Speaker(lambda: None, player=None, lanes=config.spoken_lanes("en"), language="en")
    session.speaker = speaker  # not started: what it would say waits in its queue
    mic = engine(session, MIC)
    mic.on_event(LiveEvent(Kind.SOURCE, "Let's set the budget", language="en", lane=MIC))
    mic.on_event(LiveEvent(Kind.LINE, "", source="Let's set the budget next week.", language="en", lane=MIC))
    assert [e.kind for e in shown if e.kind is not Kind.NOTE] == [Kind.SOURCE, Kind.LINE] and session.lines == {MIC: 1}
    assert "Let's set the budget next week." in session.transcript.path.read_text(encoding="utf-8")
    assert not speaker._queue  # the voice says translations only
    mic.on_event(LiveEvent(Kind.LINE, "Thank you.", source="ありがとう。", language="ja", lane=MIC))
    assert [sentence for _, _, sentence in speaker._queue] == ["Thank you."]


def test_a_line_already_in_the_target_language_is_told_from_one_whose_translation_didnt_come():
    assert already_in(LiveEvent(Kind.LINE, "", source="Let's start.", language="en-US"), "en")
    assert already_in(LiveEvent(Kind.LINE, "", source="Let's start."), "en")  # the engine didn't say: as heard
    assert not already_in(LiveEvent(Kind.LINE, "", source="始めましょう。", language="ja"), "en")
    assert not already_in(LiveEvent(Kind.LINE, "Let's start.", source="始めましょう。"), "en")


# ---- the status line: why the bar is still empty

LOUD, QUIET = np.full(1600, 3000, dtype="<i2").tobytes(), bytes(3200)  # 100 ms frames: about -21 dBFS, and silence


def notes(shown, lane=SYSTEM):
    return [e.text for e in shown if e.kind is Kind.NOTE and e.lane == lane]


def listening(source="computer", lane=SYSTEM, **config):
    shown = []
    session = LiveSession(LiveConfig(source=source, **config), on_event=shown.append)
    capture = FakeCapture()
    session.add(lane, capture, FakeEngine)
    engine(session, lane).on_event(LiveEvent(Kind.STATUS, "Listening", lane=lane))
    return session, capture, shown


def test_the_status_line_says_listening_then_that_nothing_is_heard_then_that_sound_has_no_speech():
    session, capture, shown = listening()
    assert notes(shown) == ["Listening…"]
    for _ in range(79):
        capture.on_frame(QUIET)
    assert notes(shown) == ["Listening…"]  # not before 8 s
    capture.on_frame(QUIET)
    assert notes(shown)[-1] == "Nothing heard yet: is the sound playing on this laptop?"
    for _ in range(19):
        capture.on_frame(LOUD)
    assert notes(shown)[-1].startswith("Nothing heard")  # 1.9 s of sound in the last 8: still mostly silence
    capture.on_frame(LOUD)
    assert notes(shown)[-1] == "Hearing sound, but no speech yet"
    assert len(notes(shown)) == 3  # each change once, not every frame


def test_words_clear_the_status_line_for_good():
    session, capture, shown = listening()
    for _ in range(80):
        capture.on_frame(LOUD)
    engine(session, SYSTEM).on_event(LiveEvent(Kind.SOURCE, "今日は"))
    assert notes(shown)[-1] == ""
    for _ in range(200):  # the video paused afterwards: not "nothing heard yet"
        capture.on_frame(QUIET)
    engine(session, SYSTEM).on_event(LiveEvent(Kind.STATUS, "Listening"))  # a new connection
    assert notes(shown) == ["Listening…", "Hearing sound, but no speech yet", ""]


def test_nothing_is_judged_before_the_engine_listens():
    shown = []
    session = LiveSession(LiveConfig(), on_event=shown.append)
    capture = FakeCapture()
    session.add(SYSTEM, capture, FakeEngine)
    for _ in range(100):  # still connecting: the title says so
        capture.on_frame(QUIET)
    assert notes(shown) == []


def test_the_microphone_alone_says_it_hears_nothing_but_with_both_its_silence_is_the_user_listening():
    session, capture, shown = listening("microphone", MIC)
    for _ in range(80):
        capture.on_frame(QUIET)
    assert notes(shown, MIC) == ["Listening…", "Nothing heard from the microphone"]
    session, capture, shown = listening("both", MIC)
    for _ in range(80):
        capture.on_frame(QUIET)
    assert notes(shown, MIC) == []


def test_speaking_the_language_it_translates_into_is_said_until_a_line_is_translated():
    session, capture, shown = listening("microphone", MIC, mic_target="en")
    mic = engine(session, MIC)
    mic.on_event(LiveEvent(Kind.SOURCE, "Let's set", language="en", lane=MIC))
    mic.on_event(LiveEvent(Kind.LINE, "", source="Let's set the budget.", language="en", lane=MIC))
    you = "You're speaking English, the language it translates into"
    assert notes(shown, MIC) == ["Listening…", "", you]
    assert [e.kind for e in shown if e.kind in (Kind.LINE, Kind.NOTE)][-2:] == [Kind.LINE, Kind.NOTE]  # line first
    mic.on_event(LiveEvent(Kind.SOURCE, "Next", language="en", lane=MIC))  # words in progress: the note stays
    mic.on_event(LiveEvent(Kind.LINE, "", source="Next week.", language="en", lane=MIC))
    assert notes(shown, MIC)[-1] == you and len(notes(shown, MIC)) == 3
    mic.on_event(LiveEvent(Kind.LINE, "Thank you.", source="ありがとう。", language="ja", lane=MIC))
    assert notes(shown, MIC)[-1] == ""


def test_a_capture_can_say_what_it_found_on_the_status_line_until_words_come():
    shown = []
    session = LiveSession(LiveConfig(), on_event=shown.append)
    capture = FakeCapture()
    capture.on_note = None  # a Capture that has it gets a callable: the session's note() for its way
    session.add(SYSTEM, capture, FakeEngine)
    engine(session, SYSTEM).on_event(LiveEvent(Kind.STATUS, "Listening"))
    capture.on_note("Nothing plays on Speakers: the call may be on Headset")
    for _ in range(100):
        capture.on_frame(QUIET)  # its finding stays: the session's own guess doesn't replace it
    assert notes(shown) == ["Listening…", "Nothing plays on Speakers: the call may be on Headset"]
    session.note(SYSTEM, "")  # taken back: the session's own note again
    capture.on_frame(QUIET)
    assert notes(shown)[-2:] == ["", "Nothing heard yet: is the sound playing on this laptop?"]
    session.note(SYSTEM, "The call may be on Headset")
    engine(session, SYSTEM).on_event(LiveEvent(Kind.TRANSLATION, "Good morning"))
    assert notes(shown)[-1] == ""
    session.note(MIC, "no such way")  # a way that isn't running: nothing
    assert all(e.lane == SYSTEM for e in shown)


def test_the_level_of_a_frame():
    assert dbfs(QUIET) == -np.inf and round(dbfs(LOUD)) == -21 and dbfs(b"") == -np.inf


def test_the_settings_of_each_way():
    config = LiveConfig(target="en", source="both", mic_target="ja")
    assert config.for_lane(SYSTEM).target == "en" and config.for_lane(MIC).target == "ja"


def test_the_transcript_is_bilingual_and_starts_with_the_first_line(tmp_path):
    now = [datetime(2026, 10, 5, 14, 3, 12)]
    transcript = Transcript(tmp_path / "live", LiveConfig(target="en"), now=lambda: now[0])
    transcript.add(LiveEvent(Kind.SOURCE, "今日は"))  # only finished lines are written
    assert transcript.path is None
    transcript.add(LiveEvent(Kind.LINE, "Today we have a meeting.", source="今日は会議です"))
    now[0] = datetime(2026, 10, 5, 14, 3, 20)
    transcript.add(LiveEvent(Kind.LINE, "", source="Let's start."))  # already English: shown as heard
    text = transcript.path.read_text(encoding="utf-8")
    assert transcript.path.name == "2026-10-05 14-03-12 live captions.txt"
    assert text == ("Rflow live translation, 2026-10-05 14:03, translated into English\n\n"
                    "[14:03:12] 今日は会議です\n           Today we have a meeting.\n\n"
                    "[14:03:20] Let's start.\n\n")


def test_the_transcript_marks_the_users_own_lines(tmp_path):
    config = LiveConfig(source="both", target="en", mic_target="ja")
    transcript = Transcript(tmp_path, config, now=lambda: datetime(2026, 10, 5, 15, 0, 0))
    transcript.add(LiveEvent(Kind.LINE, "Thank you.", source="ありがとうございます。"))
    transcript.add(LiveEvent(Kind.LINE, "来週にしましょう。", source="Let's do it next week.", lane=MIC))
    assert transcript.path.read_text(encoding="utf-8") == (
        "Rflow live translation, 2026-10-05 15:00, translated into English; your own speech into Japanese\n\n"
        "[15:00:00] ありがとうございます。\n           Thank you.\n\n"
        "[15:00:00] You: Let's do it next week.\n           来週にしましょう。\n\n")


def test_with_the_microphone_alone_the_transcript_marks_no_one(tmp_path):
    config = LiveConfig(source="microphone", mic_target="ja")
    transcript = Transcript(tmp_path, config, now=lambda: datetime(2026, 10, 5, 16, 0, 0))
    transcript.add(LiveEvent(Kind.LINE, "予算は来週です。", source="The budget is next week.", lane=MIC))
    assert transcript.path.read_text(encoding="utf-8") == (
        "Rflow live translation, 2026-10-05 16:00, the microphone, translated into Japanese\n\n"
        "[16:00:00] The budget is next week.\n           予算は来週です。\n\n")


def test_language_names():
    assert language_name("ja") == "Japanese" and language_name("xx") == "xx"


# ---- rule number 1: live captions are a pipeline of their own

FORBIDDEN = ("sst.pipeline", "sst.dictate", "sst.audio")  # dictation's pipeline, its controller and its recorder


def test_live_captions_import_nothing_from_dictation():
    folder = Path(sst.live.__file__).parent
    for path in folder.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                assert not name.startswith(FORBIDDEN), f"{path.name} imports {name}: keep live captions apart"


# ---- the spoken translation: a way that could hear the voice gets silence while it speaks

class FakeSpeaker:
    def __init__(self, speaking=True, private=False):
        self.speaking, self.private, self.heard, self.started, self.stopped = speaking, private, [], False, False
        self.said, self.skipped, self.repeats = 7, 1, 3

    def hear(self, event):
        self.heard.append(event)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


@pytest.mark.parametrize(("lane", "hears_self", "private", "speaking", "silenced"), [
    (MIC, False, False, True, True),  # the microphone near speakers would hear the voice
    (MIC, False, True, True, False),  # headphones: it can't
    (MIC, False, False, False, False),  # the voice is quiet
    (SYSTEM, True, True, True, True),  # the whole output, Rflow's voice in it (before Windows 11)
    (SYSTEM, False, False, True, False),  # process loopback: Rflow's voice is left out already
])
def test_a_way_that_could_hear_the_voice_gets_silence_while_it_speaks(lane, hears_self, private, speaking, silenced):
    session = LiveSession(LiveConfig(source="both"))
    capture = FakeCapture()
    capture.hears_self = hears_self
    session.add(lane, capture, FakeEngine)
    session.set_speaker(FakeSpeaker(speaking=speaking, private=private))
    capture.on_frame(b"\x01\x02" * 1600)
    assert engine(session, lane).frames == [bytes(3200) if silenced else b"\x01\x02" * 1600]


def test_the_speaker_hears_what_is_shown_and_stops_with_the_session(tmp_path, caplog):
    caplog.set_level("INFO", logger="sst.live.session")
    config = LiveConfig(source="both", target="en", mic_target="ja")
    session = LiveSession(config, Transcript(tmp_path, config))
    session.add(MIC, FakeCapture(), FakeEngine)
    speaker = FakeSpeaker()
    session.set_speaker(speaker)
    mic = engine(session, MIC)
    mic.on_event(LiveEvent(Kind.LINE, "", source="本日は。", lane=MIC))  # echo: dropped before anyone hears it
    mic.on_event(LiveEvent(Kind.TRANSLATION, "予算は", lane=MIC))
    assert speaker.started and [e.text for e in speaker.heard] == ["予算は"]
    session.stop()
    assert speaker.stopped and session.speaker is None
    assert "The voice said 7 sentences (1 skipped to keep up); the voice skipped a repeat 3 times" in caplog.text


# ---- Both on speakers: the microphone hears the others' words from the speakers (#201)

class Speakers(FakeCapture):
    def __init__(self, private, sounding):
        super().__init__()
        self.private, self._sounding, self.asked = private, sounding, []

    def sounding(self, seconds):
        self.asked.append(seconds)
        return self._sounding


@pytest.mark.parametrize(("private", "sounding", "silenced"), [
    (False, True, True),  # the speakers play the meeting: the microphone hears it, not the user
    (False, False, False),  # the speakers are quiet: the user's words
    (True, True, False),  # headphones: the microphone can't hear them
    (None, True, False),  # not known (no output read): left as it was
])
def test_with_both_on_speakers_the_microphone_is_silenced_while_the_computer_plays(private, sounding, silenced):
    session = LiveSession(LiveConfig(source="both"))
    speakers = Speakers(private, sounding)
    session.add(SYSTEM, speakers, FakeEngine)
    session.add(MIC, FakeCapture(), FakeEngine)
    session.lanes[MIC][0].on_frame(b"\x01\x02" * 1600)
    assert engine(session, MIC).frames == [bytes(3200) if silenced else b"\x01\x02" * 1600]
    speakers.on_frame(b"\x01\x02" * 1600)
    assert engine(session, SYSTEM).frames == [b"\x01\x02" * 1600]  # the computer's way is never gated
    if silenced:
        assert speakers.asked == [pytest.approx(0.5)]


def test_the_microphone_alone_is_never_gated():
    session = LiveSession(LiveConfig(source="microphone"))
    session.add(MIC, FakeCapture(), FakeEngine)
    session.lanes[MIC][0].on_frame(b"\x01\x02" * 1600)
    assert engine(session, MIC).frames == [b"\x01\x02" * 1600]
