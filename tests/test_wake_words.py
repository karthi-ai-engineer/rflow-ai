"""The wake word's words: which keywords can be used, their spellings, the setup's prompts, and cutting the keyword
and the end phrase out of the dictated text."""
import pytest

from sst.wake import END_PHRASES, check_keyword, prompts, spellings, strip_end, strip_keyword

# ---- which keywords can be used

@pytest.mark.parametrize("keyword", ["Rflow", "hey Rflow", "okay type", "computer", "Jarvis", "Siri", "RTflow"])
def test_good_keywords_pass(keyword):
    assert check_keyword(keyword) is None


@pytest.mark.parametrize("keyword, why", [
    ("", "Type"), ("   ", "Type"), ("R2D2", "letters"), ("über", "letters"), ("hello!", "letters"),
    ("done", "too often"), ("Done", "too often"), ("hey", "too often"), ("OK", "too often"), ("Okay", "too often"),
    ("yes", "too often"), ("stop it", "too often"), ("Thank you", "too often"), ("new line", "too often"),
    ("Max", "short"), ("x", "short"), ("R", "short"),
    ("one two three four five", "at most"),
])
def test_unusable_keywords_say_why(keyword, why):
    assert why in check_keyword(keyword)


def test_the_end_phrases_cant_start_dictation_too():
    for phrase in (*END_PHRASES, "end it"):
        assert check_keyword(phrase)


# ---- spellings

@pytest.mark.parametrize("typed, expected", [
    ("Rflow", ["our flow", "are flow"]),  # no English word starts "rf": the R is said by its name
    ("RFlow", ["our flow", "are flow"]),
    ("R flow", ["our flow", "are flow"]),
    ("hey Rflow", ["hey our flow", "hey are flow"]),
    ("Cflow", ["see flow", "sea flow"]),
    ("Bflow", ["be flow", "bee flow"]),
    ("Tflow", ["tea flow", "tee flow"]),
    ("UFlow", ["you flow"]),
    ("iPhone", ["eye phone"]),
    ("Xbox", ["ex box"]),
    ("Yflow", ["why flow"]),
    ("AI", ["ay eye"]),
    ("GPT", ["gee pea tea"]),
    ("RTflow", ["our tea flow", "our tee flow", "are tea flow", "are tee flow"]),  # at most 4
])
def test_letter_names_are_written_as_the_words_they_sound_like(typed, expected):
    assert spellings(typed) == expected


@pytest.mark.parametrize("typed", ["computer", "okay type", "Jarvis", "Siri", "Llama", "Knight", "Psyche", "Street"])
def test_real_words_stay_as_they_are(typed):
    assert spellings(typed) == [typed.lower()]


def test_spellings_are_plain_words():
    assert spellings("Hey-Rflow") == ["hey our flow", "hey are flow"]
    assert spellings("I") == ["i"] and spellings("a computer") == ["a computer"]  # "a" and "I" are words
    assert all(1 <= len(spellings(k)) <= 4 for k in ("Rflow", "RTCflow", "a b c d"))


def test_ten_prompts_with_the_keyword_in_each():
    said = prompts(" Rflow ")
    assert len(said) == 10 and all("“Rflow" in p for p in said)
    assert any("quiet" in p for p in said) and any("Lean back" in p for p in said)
    assert any("Rflow, make it concise" in p for p in said)  # once in a sentence, as it's used


# ---- cutting the keyword and the end phrase out of the text

@pytest.mark.parametrize("heard, left", [
    ("Our flow, hi", "hi"),
    ("Rflow, hi there.", "hi there."),
    ("R flow. Hello", "Hello"),
    ("Are, flow, hello", "hello"),
    ("Arflow hello", "hello"),
    ("RFlow: make it concise", "make it concise"),
    ("our flow make it concise", "make it concise"),
    ("Our flow", ""),
    ("Airflow, hello", "hello"),
    ("Hour flow, hi", "hi"),
    ("Hello there", "Hello there"),  # no keyword: unchanged
    ("We are flowing", "We are flowing"),
    ("", ""),
])
def test_strip_keyword(heard, left):
    assert strip_keyword(heard, "Rflow", "our flow") == left


def test_strip_keyword_of_a_real_word():
    assert strip_keyword("Computer, send the email.", "computer", "computer") == "send the email."
    assert strip_keyword("Okay, type the note", "okay type", "okay type") == "the note"
    assert strip_keyword("The computer is slow", "computer", "computer") == "The computer is slow"  # only at the start


@pytest.mark.parametrize("heard, left", [
    ("Send it, stop it.", "Send it."),
    ("Hello there. Stop it.", "Hello there."),
    ("Hello there stop it", "Hello there"),
    ("Is it done? Stop it!", "Is it done?"),
    ("Stop it.", ""),
    ("stopit", ""),
    ("Fine, stop it. Stop it.", "Fine."),  # said twice: the first wasn't heard
    ("Don't stop it from sending", "Don't stop it from sending"),  # not at the end: kept
    ("I said stop it 2", "I said stop it 2"),
    ("It went well. End it.", "It went well."),
    ("Nothing to cut", "Nothing to cut"),
])
def test_strip_end(heard, left):
    assert strip_end(heard, ["stop it", "end it"]) == left


def test_strip_end_with_no_phrases_changes_nothing():
    assert strip_end("Send it, stop it.", []) == "Send it, stop it."
