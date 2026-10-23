"""The wake word's setup (spotters, the transcriber and the voice print faked) and what it saves; and the check, before
any try, of whether the spotter can hear a keyword at all."""
import json
import logging

import numpy as np
import pytest

from sst.wake import MISSES, PASS, TRIES, Enrollment, VoicePrint, WakeSetup, check_hearing, level_db, sounds_like
from sst.wake import enrol as enrol_module
from sst.wake.enrol import FALLBACK, LEAD, MAX_LEARNED, TAIL


class ScriptedSpotter:
    """Hears its spelling in the takes `script[spelling]` marks True, one entry per take it's fed."""

    def __init__(self, script, phrases):
        (self.name, self.spelling), = phrases.items()
        self.script, self.fed, self.resets = script, [], 0

    def reset(self):
        self.resets += 1

    def feed(self, audio):
        self.fed.append(len(audio))
        marks = self.script.get(self.spelling, [])
        return self.name if len(self.fed) <= len(marks) and marks[len(self.fed) - 1] else None


def enrol(script, keyword="Rflow", voice_factory=None, transcribe=None):
    made = []

    def factory(phrases):
        made.append(ScriptedSpotter(script, phrases))
        return made[-1]

    return Enrollment(keyword, spotter_factory=factory, voice_factory=voice_factory, transcribe=transcribe), made


def take(seconds=1.0, level=0.1):
    return np.full(int(seconds * 16000), level, np.float32)


def test_a_take_missed_doesnt_use_up_the_try():
    enrolment, _ = enrol({"our flow": [False, False, True, False, True] + [True] * 20})
    assert [enrolment.add(take()) for _ in range(3)] == [False, False, True]
    assert enrolment.tries == 1 and enrolment.said == 3 and enrolment.missed == 0  # the try heard: misses start over
    assert enrolment.add(take()) is False and enrolment.missed == 1 and enrolment.tries == 1
    while not enrolment.done:
        enrolment.add(take())
    assert enrolment.tries == TRIES and enrolment.said == TRIES + 3 and enrolment.heard == TRIES
    setup = enrolment.result()
    assert (setup.spelling, setup.heard, setup.said) == ("our flow", TRIES, TRIES + 3)
    with pytest.raises(RuntimeError):
        enrolment.add(take())


def test_misses_in_a_row_are_counted_for_the_advice():
    enrolment, _ = enrol({"our flow": [False] * MISSES + [True]})
    for n in range(1, MISSES + 1):
        enrolment.add(take())
        assert enrolment.missed == n and enrolment.tries == 0
    enrolment.add(take())
    assert enrolment.missed == 0 and enrolment.tries == 1


def test_the_spelling_heard_most_is_kept_and_the_voice_print_gets_the_tries_heard():
    takes = []
    enrolment, spotters = enrol({"our flow": [True] * 9 + [False, True], "are flow": [True] * 6 + [False] * 5},
                                voice_factory=lambda t: takes.extend(t) or VoicePrint([1.0, 0.0]))
    assert [s.spelling for s in spotters] == ["our flow", "are flow"]  # one spotter each: together they compete
    assert len(enrolment.prompts) == TRIES and not enrolment.done
    while not enrolment.done:
        enrolment.add(take())
    assert enrolment.said == 11 and enrolment.heard == 10 and enrolment.counts == [10, 6]
    setup = enrolment.result()
    assert (setup.keyword, setup.spelling, setup.heard, setup.said) == ("Rflow", "our flow", 10, 11)
    assert len(takes) == TRIES and all(len(t) == 16000 for t in takes)  # the 10 heard, as they were recorded
    assert np.allclose(setup.voice.vector, [1.0, 0.0])
    assert all(s.resets == 11 for s in spotters)  # each take heard on its own
    assert spotters[0].fed[0] == 16000 + int((LEAD + TAIL) * 16000)  # with quiet around it: the end of the keyword counts


def test_tries_heard_by_different_spellings_count_but_no_spelling_reached_pass():
    enrolment, _ = enrol({"our flow": [True, False] * 5, "are flow": [False, True] * 5})
    for _ in range(TRIES):
        assert enrolment.add(take())
    assert enrolment.done and enrolment.heard == 5 and enrolment.result() is None  # heard as different words


def test_exactly_pass_is_saved_and_a_tie_keeps_the_likelier_spelling():
    first, rest = [True] * PASS + [False] * (TRIES - PASS), [False] * PASS + [True] * (TRIES - PASS)
    enrolment, spotters = enrol({"our tea flow": first, "our tee flow": first, "are tea flow": rest}, keyword="RTflow")
    for _ in range(TRIES):
        assert enrolment.add(take())
    assert enrolment.counts == [PASS, PASS, TRIES - PASS, 0]
    setup = enrolment.result()
    assert setup.spelling == "our tea flow" and setup.heard == PASS and setup.voice is None  # no voice factory: no check


def test_a_name_is_also_listened_for_as_the_spellings_that_sound_like_it():
    enrolment, spotters = enrol({"carthy": [True] * TRIES}, keyword="Karthi")
    assert [s.spelling for s in spotters] == sounds_like("karthi") == [
        "karthi", "karthy", "carthi", "kathi", "carthy", "kathy"]
    while not enrolment.done:
        enrolment.add(take())
    assert enrolment.result().spelling == "carthy"


def test_a_spelling_the_spotter_cant_listen_for_is_skipped_unless_it_is_the_keywords_own():
    def factory(phrases):
        if next(iter(phrases)) == "karthy":
            raise ValueError("unknown pieces")
        return ScriptedSpotter({}, phrases)
    assert "karthy" not in Enrollment("karthi", spotter_factory=factory, voice_factory=None).spellings
    with pytest.raises(ValueError):
        Enrollment("karthy", spotter_factory=factory, voice_factory=None)


def test_what_the_model_heard_in_a_take_missed_is_learned_as_a_spelling():
    heard_as = iter(["CARDI", "PARTY", "CARDI"])
    script = {"cardi": [True] * 30}  # the learned spelling hears every take, its own included
    enrolment, spotters = enrol(script, keyword="karthi", transcribe=lambda audio: next(heard_as))
    assert enrolment.add(take()) is False  # none of the spellings heard it: "cardi" is learned from it
    assert enrolment.spellings[-1] == "cardi" and enrolment.counts[-1] == 0  # its own take doesn't count
    assert enrolment.add(take()) is True and enrolment.counts[-1] == 1  # the next take: heard by "cardi"
    while not enrolment.done:
        enrolment.add(take())
    setup = enrolment.result()
    assert (setup.spelling, setup.heard, setup.said) == ("cardi", TRIES, TRIES + 1)
    assert spotters[-1].fed[0] == spotters[0].fed[0]  # scored on the take it came from, padded as the others


def test_words_that_cant_stand_for_the_keyword_are_never_learned():
    said = iter(["PARTY", "THE", "", "OKAY GUY STAPLE", "KARTHI"])  # "karthi" is already listened for
    enrolment, _ = enrol({}, keyword="karthi", transcribe=lambda audio: next(said))
    known = list(enrolment.spellings)
    for _ in range(5):
        enrolment.add(take())
    assert enrolment.spellings == known


def test_at_most_a_few_spellings_are_learned_and_a_failing_transcriber_is_harmless(caplog):
    words = iter(["CARDI", "CARDY", "CARTY", "CARTI", "CARTEE"])
    enrolment, _ = enrol({}, keyword="karthi", transcribe=lambda audio: next(words))
    for _ in range(5):
        enrolment.add(take())
    assert enrolment.spellings[-MAX_LEARNED:] == ["cardi", "cardy", "carty"]

    def broken(audio):
        raise RuntimeError("no model")
    enrolment, _ = enrol({}, keyword="karthi", transcribe=broken)
    with caplog.at_level(logging.WARNING):
        assert enrolment.add(take()) is False
    assert "couldn't be transcribed" in caplog.text


def test_each_takes_level_is_logged_never_its_audio(caplog):
    enrolment, _ = enrol({"our flow": [True]})
    with caplog.at_level(logging.INFO, logger="sst.wake.enrol"):
        enrolment.add(take(level=0.01))
    assert "level -40 dBFS" in caplog.text and "heard" in caplog.text


def test_a_takes_level_is_its_loudest_100_ms():
    quiet = np.zeros(16000, np.float32)
    loud = quiet.copy()
    loud[8000:9600] = 0.1  # 100 ms of speech in a second of silence
    assert level_db(loud) == pytest.approx(-20, abs=0.1)
    assert level_db(quiet) < -100 and level_db(np.zeros(0, np.float32)) < -100


def test_an_unusable_keyword_is_refused_before_any_try():
    with pytest.raises(ValueError, match="too often"):
        enrol({}, keyword="done")


def test_the_setup_survives_json_and_an_older_one_still_reads():
    setup = WakeSetup("Rflow", "our flow", VoicePrint([0.6, 0.8]), 9, 12)
    data = json.loads(json.dumps(setup.to_json()))
    assert "audio" not in json.dumps(data)
    back = WakeSetup.from_json(data)
    assert (back.keyword, back.spelling, back.heard, back.said) == ("Rflow", "our flow", 9, 12)
    assert np.allclose(back.voice.vector, [0.6, 0.8])
    old = {"keyword": "okay type", "spelling": "okay type", "heard": 10, "voice": None}  # saved before "said"
    assert WakeSetup.from_json(old) == WakeSetup("okay type", "okay type", None, 10, 0)


# ---- can the spotter hear it at all? A synthetic voice says it; faked here: the spotter hears what `hears` allows

class Voice:
    def __init__(self):
        self.said: list[tuple[str, float]] = []

    def __call__(self, text, speed):
        self.said.append((text, speed))
        return np.full(16000, 0.5, np.float32)


def _check(keyword, hears, transcribe=None):
    voice = Voice()

    class Spotter:
        def __init__(self, phrases):
            self.spelling = next(iter(phrases))

        def reset(self):
            pass

        def feed(self, audio):
            return self.spelling if hears(self.spelling, voice.said[-1][0]) else None
    return check_hearing(keyword, voice, Spotter, transcribe), voice


def test_a_keyword_heard_needs_no_suggestion_and_stops_early():
    (hearing, suggestion), voice = _check("computer", lambda spelling, text: True)
    assert hearing.ok and suggestion is None and hearing.spelling == "computer"
    assert hearing.said == 5 < len(enrol_module.CARRIERS) * len(enrol_module.SPEEDS)  # 5 of 9 heard: it can't fail
    assert voice.said[0] == ("I asked computer about it.", 1.0)  # in a sentence: a name said alone is garbled
    assert {speed for _, speed in voice.said} == {1.0, 0.9}


def test_a_name_heard_only_as_a_spelling_that_sounds_like_it_is_heard():
    (hearing, suggestion), _ = _check("karthi", lambda spelling, text: spelling == "kathy")
    assert hearing.ok and hearing.spelling == "kathy" and suggestion is None


def test_a_keyword_not_heard_gets_the_phrase_with_it_heard_best():
    def hears(spelling, text):  # "hello karthi" in every sentence, "hey karthi" in one of three
        return (spelling == "hello karthi" and "hello" in text) or (spelling == "hey karthi" and "So hey" in text)
    (hearing, suggestion), voice = _check("karthi", hears)
    assert not hearing.ok and hearing.heard == 0 and suggestion == "hello karthi"
    assert len([t for t, _ in voice.said if "karthi" in t and "hey" not in t and "hello" not in t
                and "okay" not in t]) == 5  # none of 5 heard: it can't pass any more


def test_with_nothing_heard_the_suggestion_is_a_keyword_known_to_work():
    (hearing, suggestion), _ = _check("hey karthi", lambda spelling, text: False)
    assert not hearing.ok and suggestion == FALLBACK


def test_a_greeting_already_in_the_keyword_isnt_doubled():
    (_, suggestion), voice = _check("hey karthi", lambda spelling, text: spelling == "okay karthi" and "okay" in text)
    assert suggestion == "okay karthi"
    assert not any("hey hey" in t or "hello hey" in t or "okay hey" in t for t, _ in voice.said)
