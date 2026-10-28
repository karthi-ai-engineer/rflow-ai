"""Voice commands for Text Transform (sst.commands): a whole dictation that is one command phrase is a command; anything
else, even with a command's words in it, is dictation and is typed."""
import pytest

from sst.commands import DEFAULT_PHRASES, MAX_WORDS, UNDO, match_command, normalize, parse_phrases, phrases_for

DEFAULTS = phrases_for({})


@pytest.mark.parametrize("said, command", [
    ("Make it concise.", "concise"),
    ("make it shorter", "concise"),
    ("Make this shorter, please.", "concise"),  # "this" is "it"; "please" around it
    ("Okay, make it concise.", "concise"),
    ("Make it more professional.", "professional"),
    ("make that formal", "professional"),
    ("Bullet points.", "bullets"),
    ("Turn it into a list.", "bullets"),
    ("Action items.", "actions"),
    ("Make it a to-do list.", "actions"),
    ("make it a todo list", "actions"),
    ("Rewrite it.", "rewrite"),
    ("Undo that.", UNDO),
    ("Undo.", UNDO),
    ("Put it back.", UNDO),
    ("Make it consise.", "concise"),  # misheard a little
    ("Make it profesional.", "professional"),
    ("MAKE IT CONCISE!", "concise"),
    ("Professional.", "professional"),  # one word, the whole dictation (the owner's way, 2026-10-28)
    ("Concise.", "concise"),
    ("Formal", "professional"),
    ("Shorter.", "concise"),
    ("Proofread.", "grammar"),
])
def test_command_phrases_are_found(said, command):
    assert match_command(said, DEFAULTS) == command


@pytest.mark.parametrize("said", [
    "Make it concise and send it to John.",  # more than the command: dictation
    "Please make it concise before the meeting on Monday.",
    "We need a professional.",  # the word in a sentence is dictation
    "Concise and clear is how I like it.",
    "Make it.",  # a fragment of a phrase is not the phrase
    "The action items are on the board.",
    "Can you make it shorter for the client?",
    "I need bullet points for the slides tomorrow.",
    "",
    "   ",
    "Make it count.",
])
def test_dictation_is_not_a_command(said):
    assert match_command(said, DEFAULTS) is None


def test_long_dictation_is_never_a_command():
    long = " ".join(["make it concise"] * 3)
    assert len(long.split()) > MAX_WORDS and match_command(long, DEFAULTS) is None


def test_the_users_own_phrases_replace_the_defaults():
    phrases = phrases_for({"concise": "trim it, tighten this up", "bullets": ""})
    assert match_command("Trim it.", phrases) == "concise"
    assert match_command("Tighten this up.", phrases) == "concise"
    assert match_command("Make it concise.", phrases) is None  # the defaults for Concise were replaced
    assert match_command("Bullet points.", phrases) is None  # no phrase: that command is off
    assert match_command("Make it professional.", phrases) == "professional"  # untouched: the defaults
    assert phrases["professional"] == list(DEFAULT_PHRASES["professional"])


def test_a_one_word_phrase_works_when_the_user_adds_it():
    assert match_command("Professional!", phrases_for({"professional": "professional"})) == "professional"


@pytest.mark.parametrize("text, phrases", [
    ("make it short, trim it", ["make it short", "trim it"]),
    ("make it short\ntrim it;  shorten   this ", ["make it short", "trim it", "shorten this"]),
    (" , ,, ", []),
    ("trim it, trim it", ["trim it"]),
])
def test_parse_phrases(text, phrases):
    assert parse_phrases(text) == phrases


def test_normalize():
    assert normalize("Um, so make THIS shorter, please!") == "make it shorter"
    assert normalize("Make it a to-do list") == "make it a to do list"
    assert normalize("Hey Rflow, rewrite that for me, thanks.") == "rewrite it"


@pytest.mark.parametrize("said, normal", [
    ("英語にしてください。", "英語にして"),
    ("えっと、英語にしてくれる？", "英語にして"),
    ("えっと英語にしてね", "英語にして"),
    ("校正お願いします", "校正して"),
    ("翻訳をお願いします。", "翻訳して"),
    ("Teams 用に して ください", "teams用にして"),
    ("英語に直して", "英語にして"),
    ("簡潔にしてもらえますか", "簡潔にして"),
])
def test_normalize_japanese(said, normal):
    assert normalize(said) == normal


# ---- phase 40: Fix grammar, the Japanese Teams message and emails, Translate (with a language and a register)

@pytest.mark.parametrize("said, command", [
    ("Fix the grammar.", "grammar"), ("fix grammar", "grammar"), ("Proofread it, please.", "grammar"),
    ("Fix the typos.", "grammar"), ("校正して", "grammar"), ("校正してください。", "grammar"), ("誤字を直して", "grammar"),
    ("Make it a Teams message.", "teams"), ("Make it a Japanese Teams message.", "teams"),
    ("Translate to Japanese for Teams.", "teams"), ("Teams用にして", "teams"), ("チャット用にしてください", "teams"),
    ("Make it an internal email.", "email_internal"), ("Make it a Japanese email.", "email_internal"),
    ("社内メールにして", "email_internal"),
    ("Make it an external email.", "email_external"), ("make it a client email", "email_external"),
    ("社外メールにしてください", "email_external"), ("取引先向けにして", "email_external"),
    ("Translate it.", "translate"), ("Translate this.", "translate"), ("翻訳して", "translate"),
    ("翻訳お願いします", "translate"),
    ("Translate it to Japanese.", "translate:Japanese"), ("Translate this into English.", "translate:English"),
    ("Translate into English, formally.", "translate:English:formal"),
    ("translate to japanese casually", "translate:Japanese:casual"),
    ("Make it Japanese.", "translate:Japanese"), ("In English, please.", "translate:English"),
    ("Translate to Chinese.", "translate:Chinese (Simplified)"), ("translate into traditional chinese",
                                                                 "translate:Chinese (Traditional)"),
    ("英語にして", "translate:English"), ("英訳して", "translate:English"), ("日本語にしてください", "translate:Japanese"),
    ("和訳して", "translate:Japanese"), ("日本語に訳して", "translate:Japanese"), ("英語に翻訳して", "translate:English"),
    ("英語で丁寧に", "translate:English:formal"), ("丁寧な英語にして", "translate:English:formal"),
    ("英語でお願いします", "translate:English"), ("韓国語にして", "translate:Korean"),
    ("元に戻して", UNDO),
])
def test_phase_40_phrases(said, command):
    assert match_command(said, DEFAULTS) == command


@pytest.mark.parametrize("said", [
    "Translate the report into Japanese before Friday.",  # more than the command
    "Make it a Teams message for the whole team about Friday.",
    "Translate to Klingon.",  # no such language
    "英語",  # a word, not a command
    "日本語で",
    "Make it formal in a way.",
    "I need to fix the grammar in the report.",
])
def test_phase_40_dictation_is_not_a_command(said):
    assert match_command(said, DEFAULTS) is None


def test_translate_with_a_language_needs_translate_on():
    phrases = phrases_for({"translate": ""})  # no phrase: Translate is off, languages too
    assert match_command("Translate it to Japanese.", phrases) is None and match_command("英語にして", phrases) is None
    assert match_command("Make it a Teams message.", phrases) == "teams"
