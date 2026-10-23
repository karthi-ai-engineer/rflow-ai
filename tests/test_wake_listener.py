"""The wake word's turns, frame by frame (100 ms, a fake clock): the spotters and the voice print are faked, speech and
silence are noise at two levels for the energy gate, and a phrase's last frame carries a mark the fake spotters hear."""
import numpy as np
import pytest

from sst.wake import SAFETY_SILENCE, VOICE_THRESHOLD, VoicePrint, WakeListener, WakeSetup
from sst.wake.listener import KEYWORD_AUDIO, RECHECK_OFFSETS, ROOM_SECONDS, VOICE_SECONDS, Gate

FRAME = 0.1
RATE = 16000
MARKS = {"keyword": 1.5, "stop it": 1.6, "end it": 1.7}  # far outside the noise: a phrase was said in this frame


class FakeSpotter:
    def __init__(self, phrases):
        self.phrases, self.fed, self.resets, self.deaf = phrases, 0, 0, False

    def reset(self):
        self.resets += 1

    def feed(self, frame):
        self.fed += 1
        heard = [name for name in self.phrases if np.isclose(frame, MARKS[name]).any()]
        return heard[0] if heard and not self.deaf else None


class FakeVoice(VoicePrint):
    def __init__(self, score):
        super().__init__([1.0])
        self.result, self.heard = score, []

    def score(self, audio, model_dir=None):
        self.heard.append(len(audio))
        return self.result


class Mic:
    """Feeds the listener: speech (noise at -20 dBFS) and silence (-60 dBFS), and the spotters' hits."""

    def __init__(self, voice=None, end_phrases=("stop it",), voice_check=True):
        self.spotters = []

        def factory(phrases):
            self.spotters.append(FakeSpotter(phrases))
            return self.spotters[-1]

        self.listener = WakeListener(WakeSetup("Rflow", "our flow", voice, 10), end_phrases, voice_check=voice_check,
                                     spotter_factory=factory)
        self.keyword, self.end, *self.rechecks = self.spotters
        self.events, self.now = [], 100.0
        self.listener.on_wake = lambda: self.events.append(("wake", round(self.now, 1)))
        self.listener.on_pause = lambda: self.events.append(("pause", round(self.now, 1)))
        self.listener.on_end = lambda keep: self.events.append(("end", keep, round(self.now, 1)))
        self.rng = np.random.default_rng(0)
        self.silence(2.0)  # the room, before anything is said

    def _feed(self, seconds, level, phrase=None):
        frames = round(seconds / FRAME)
        for i in range(frames):
            frame = (self.rng.standard_normal(int(FRAME * RATE)) * level).astype(np.float32)
            if phrase and i == frames - 1:  # the spotter hears the phrase as it ends
                frame[-1] = MARKS[phrase]
            self.now += FRAME
            self.listener.feed(frame, self.now)
        return self

    def speech(self, seconds):
        return self._feed(seconds, 0.1)

    def silence(self, seconds):
        return self._feed(seconds, 0.001)

    def say_keyword(self):
        return self._feed(0.6, 0.1, "keyword")

    def say_end(self, phrase="stop it"):
        return self._feed(0.5, 0.1, phrase)

    def names(self):
        return [e[0] for e in self.events]


def test_the_keyword_starts_a_turn():
    mic = Mic()
    assert not mic.listener.active and mic.end.fed == 0  # idle: only the keyword is listened for
    mic.say_keyword()
    assert mic.events == [("wake", 102.6)] and mic.listener.active and mic.listener.turns == 1
    fed = mic.keyword.fed
    mic.speech(1.0)
    assert mic.keyword.fed == fed and mic.end.fed == 10  # in a turn: only the end phrase
    assert mic.keyword.phrases == {"keyword": "our flow"} and mic.end.phrases == {"stop it": "stop it"}


def test_a_short_sentence_then_a_pause_lets_the_app_check_for_a_command():
    mic = Mic().say_keyword().speech(2.0).silence(1.1)
    assert mic.names() == ["wake"]
    mic.silence(0.2)
    assert mic.names() == ["wake", "pause"] and mic.listener.active
    mic.speech(2.0).silence(3.0)
    assert mic.names() == ["wake", "pause"]  # once a turn


def test_no_pause_event_after_long_speech_or_before_any_speech():
    mic = Mic().say_keyword().silence(3.0)  # the keyword, then thinking: no command was said
    assert mic.names() == ["wake"]
    mic.speech(5.0).silence(2.0)  # the speech ended more than 4 s after the keyword: dictation
    assert mic.names() == ["wake"]


def test_the_end_phrase_ends_the_turn_only_with_silence_after_it():
    mic = Mic().say_keyword().speech(6.0).say_end().silence(0.6)
    assert mic.names() == ["wake"]
    mic.silence(0.2)
    assert mic.events[-1][:2] == ("end", True) and not mic.listener.active
    assert mic.keyword.resets >= 2  # back to idle: the keyword spotter starts afresh


def test_stop_it_mid_sentence_does_not_end_the_turn():
    mic = Mic().say_keyword().speech(6.0).say_end().speech(1.5)  # "... stop it from sending the email"
    mic.silence(5.0)
    assert mic.names() == ["wake"] and mic.listener.active
    mic.speech(1.0).say_end().silence(1.0)  # the real one
    assert mic.names() == ["wake", "end"] and mic.events[-1][1] is True


def test_the_phrases_own_last_sound_just_after_the_hit_is_not_speaking_on():
    mic = Mic().say_keyword().speech(6.0).say_end().speech(0.1).silence(0.8)
    assert mic.names() == ["wake", "end"]


def test_an_end_phrase_the_running_stream_missed_is_heard_on_a_second_listen():
    mic = Mic()
    mic.end.deaf = True
    mic.say_keyword().speech(4.0).say_end().silence(0.6)
    assert mic.names() == ["wake"] and all(s.fed == 0 for s in mic.rechecks)  # nothing to recheck while speaking
    mic.silence(0.2)
    assert mic.events[-1][:2] == ("end", True)
    assert len(mic.rechecks) == len(RECHECK_OFFSETS) and mic.rechecks[0].resets == 1


def test_the_second_listen_keeps_stop_it_from_sending():
    mic = Mic()
    mic.end.deaf = True
    mic.say_keyword().speech(4.0).say_end().speech(1.0).silence(3.0)  # the phrase is in the audio, but words followed
    assert mic.names() == ["wake"]
    fed = [s.fed for s in mic.rechecks]
    assert all(fed) and mic.rechecks[0].resets == 1  # once a pause
    mic.silence(3.0)
    assert [s.fed for s in mic.rechecks] == fed
    mic.speech(1.0).silence(1.0)
    assert mic.rechecks[0].resets == 2 and mic.names() == ["wake"]


def test_long_silence_ends_the_turn_anyway():
    mic = Mic().say_keyword().speech(6.0).silence(SAFETY_SILENCE - 0.5)
    assert mic.names() == ["wake"]
    mic.silence(1.0)
    assert mic.events[-1][:2] == ("end", True)


def test_the_voice_check_rejects_another_voice_after_3_seconds():
    voice = FakeVoice(VOICE_THRESHOLD - 0.01)
    mic = Mic(voice).say_keyword().speech(2.9)
    assert voice.heard == [] and mic.listener.active
    mic.speech(0.2)
    assert mic.events[-1][:2] == ("end", False) and not mic.listener.active
    assert voice.heard == [int((KEYWORD_AUDIO + VOICE_SECONDS) * RATE)]  # the keyword and the first 3 s of speech
    assert mic.listener.voice_score == pytest.approx(VOICE_THRESHOLD - 0.01)
    mic.speech(2.0).say_end().silence(1.0)
    assert mic.names() == ["wake", "end"]  # back to idle: nothing more without the keyword


def test_the_voice_check_lets_the_user_through_once():
    voice = FakeVoice(0.6)
    mic = Mic(voice).say_keyword().speech(8.0).say_end().silence(1.0)
    assert mic.names() == ["wake", "end"] and mic.events[-1][1] is True
    assert len(voice.heard) == 1


def test_a_short_turn_is_checked_when_it_ends():
    voice = FakeVoice(0.1)
    mic = Mic(voice).say_keyword().speech(1.0).say_end().silence(1.0)
    assert mic.names() == ["wake", "end"] and mic.events[-1][1] is False  # discarded
    assert voice.heard == [int((KEYWORD_AUDIO + 1.5) * RATE)]  # the keyword, the speech and the end phrase


def test_a_command_is_voice_checked_before_the_app_acts_on_it():
    voice = FakeVoice(0.1)
    mic = Mic(voice).say_keyword().speech(1.5).silence(1.3)
    assert mic.names() == ["wake", "end"] and mic.events[-1][1] is False  # no pause: the app never runs it
    voice.result = 0.7
    mic.say_keyword().speech(1.5).silence(1.3)
    assert mic.names() == ["wake", "end", "wake", "pause"]


def test_with_the_voice_check_off_no_voice_is_scored():
    voice = FakeVoice(0.0)
    mic = Mic(voice, voice_check=False).say_keyword().speech(5.0).say_end().silence(1.0)
    assert mic.events[-1][:2] == ("end", True) and voice.heard == []
    mic = Mic(None).say_keyword().speech(5.0).say_end().silence(1.0)  # set up without a voice print
    assert mic.events[-1][:2] == ("end", True)


def test_a_voice_check_that_cannot_run_keeps_the_text_out():
    class Broken(FakeVoice):
        def score(self, audio, model_dir=None):
            raise FileNotFoundError("the voice model was deleted")

    mic = Mic(Broken(0.0)).say_keyword().speech(4.0)
    assert mic.events[-1][:2] == ("end", False)


def test_the_app_can_end_the_turn_and_the_keyword_starts_another():
    mic = Mic().say_keyword().speech(2.0)
    mic.listener.end()
    assert not mic.listener.active and mic.names() == ["wake"]  # no callback: the app knows
    mic.speech(1.0).say_end().silence(1.0)
    assert mic.names() == ["wake"]
    mic.say_keyword()
    assert mic.names() == ["wake", "wake"] and mic.listener.turns == 2
    mic.listener.end()
    mic.listener.end()  # twice is harmless


def test_more_end_phrases():
    mic = Mic(end_phrases=("stop it", "end it", "stop it"))
    assert mic.end.phrases == {"stop it": "stop it", "end it": "end it"}
    mic.say_keyword().speech(3.0).say_end("end it").silence(1.0)
    assert mic.events[-1][:2] == ("end", True)
    with pytest.raises(ValueError):
        Mic(end_phrases=())


def test_the_gate_hears_speech_above_the_room():
    gate, rng = Gate(), np.random.default_rng(1)

    def noise(level):
        return (rng.standard_normal(1600) * level).astype(np.float32)

    assert not any(gate(noise(0.001)) for _ in range(20))  # the room (-60 dBFS)
    assert gate(noise(0.1))  # speech
    assert not gate(np.zeros(0, np.float32))
    assert gate(noise(0.01))  # a fan switched on (-40 dBFS): louder than the room at first
    for _ in range(int(ROOM_SECONDS / FRAME)):
        gate(noise(0.01))  # then the room
    assert not gate(noise(0.01)) and gate(noise(0.1))
    quiet = Gate()
    assert not any(quiet(noise(level)) for level in (0.0, 0.0, 0.0003, 0.0004))  # never speech under -65 dBFS
