"""The wake word's 10-try setup (spotters and voice print faked) and what it saves."""
import json

import numpy as np
import pytest

from sst.wake import PASS, TRIES, Enrollment, VoicePrint, WakeSetup
from sst.wake.enrol import LEAD, TAIL


class ScriptedSpotter:
    """Hears its spelling in the tries `script[spelling]` marks True, one entry per try."""

    def __init__(self, script, phrases):
        (self.name, self.spelling), = phrases.items()
        self.script, self.fed, self.resets = script, [], 0

    def reset(self):
        self.resets += 1

    def feed(self, audio):
        self.fed.append(len(audio))
        return self.name if self.script.get(self.spelling, [False] * TRIES)[len(self.fed) - 1] else None


def enrol(script, keyword="Rflow", voice_factory=None):
    made = []

    def factory(phrases):
        made.append(ScriptedSpotter(script, phrases))
        return made[-1]

    return Enrollment(keyword, spotter_factory=factory, voice_factory=voice_factory), made


def take(seconds=1.0):
    return np.full(int(seconds * 16000), 0.1, np.float32)


def heard(n):
    return [True] * n + [False] * (TRIES - n)


def test_the_spelling_heard_most_is_kept():
    takes = []
    enrolment, spotters = enrol({"our flow": heard(9), "are flow": heard(6)},
                                voice_factory=lambda t: takes.extend(t) or VoicePrint([1.0, 0.0]))
    assert [s.spelling for s in spotters] == ["our flow", "are flow"]  # one spotter each: together they compete
    assert len(enrolment.prompts) == TRIES and not enrolment.done
    answers = [enrolment.add(take()) for _ in range(TRIES)]
    assert answers == heard(9) and enrolment.tries == TRIES and enrolment.done and enrolment.heard == 9
    setup = enrolment.result()
    assert (setup.keyword, setup.spelling, setup.heard) == ("Rflow", "our flow", 9)
    assert len(takes) == TRIES and all(len(t) == 16000 for t in takes)  # the voice print gets the recordings as they were
    assert np.allclose(setup.voice.vector, [1.0, 0.0])
    assert all(s.resets == TRIES for s in spotters)  # each try heard on its own
    assert spotters[0].fed[0] == 16000 + int((LEAD + TAIL) * 16000)  # with quiet around it: the end of the keyword counts


def test_any_spelling_hearing_a_try_counts_for_that_try():
    enrolment, _ = enrol({"our flow": [True, False] * 5, "are flow": [False, True] * 5})
    assert all(enrolment.add(take()) for _ in range(TRIES))
    assert enrolment.heard == 5 and enrolment.result() is None  # but no spelling reached PASS on its own


def test_a_keyword_heard_under_8_of_10_is_refused():
    enrolment, _ = enrol({"our flow": heard(PASS - 1), "are flow": heard(3)})
    for _ in range(TRIES):
        enrolment.add(take())
    assert enrolment.heard == PASS - 1 and enrolment.result() is None


def test_exactly_8_passes_and_a_tie_keeps_the_likelier_spelling():
    enrolment, _ = enrol({"our flow": heard(PASS), "are flow": heard(PASS)})
    for _ in range(TRIES):
        enrolment.add(take())
    setup = enrolment.result()
    assert setup.spelling == "our flow" and setup.heard == PASS and setup.voice is None  # no voice factory: no check


def test_no_result_before_the_tries_are_done_and_none_after():
    enrolment, _ = enrol({"our flow": heard(10)})
    for _ in range(TRIES - 1):
        enrolment.add(take())
    assert enrolment.result() is None and not enrolment.done
    enrolment.add(take())
    assert enrolment.result().heard == 10
    with pytest.raises(RuntimeError):
        enrolment.add(take())


def test_an_unusable_keyword_is_refused_before_any_try():
    with pytest.raises(ValueError, match="too often"):
        enrol({}, keyword="done")


def test_the_setup_survives_json():
    setup = WakeSetup("Rflow", "our flow", VoicePrint([0.6, 0.8]), 9)
    data = json.loads(json.dumps(setup.to_json()))
    assert "audio" not in json.dumps(data)
    back = WakeSetup.from_json(data)
    assert (back.keyword, back.spelling, back.heard) == ("Rflow", "our flow", 9)
    assert np.allclose(back.voice.vector, [0.6, 0.8])
    plain = WakeSetup.from_json(json.loads(json.dumps(WakeSetup("okay type", "okay type", None, 10).to_json())))
    assert plain == WakeSetup("okay type", "okay type", None, 10)
