"""Text Transform on long texts (the owner's 3-4 minutes of dictation, 500-1,000 words, and a bigger one): split into parts
between paragraphs or sentences, each part transformed and checked on its own, in parallel, and joined again. A part the
check rejects stays as written where that is safe, else the whole text stays; nothing is ever lost or typed with a hole."""
import threading
import time

import pytest

from sst import usage
from sst.transform import NO_ACTIONS, PART_WORDS, SPLIT_OVER, Transformer, split_parts

SENTENCES = [
    "So I checked the deployment of release {n} this morning and everything looks good.",
    "We still have one issue with the database migration on server {n}, and I think we should fix that before production.",
    "Sarah said she can look at ticket {n} on Friday, or maybe Monday depending on testing.",
    "The budget for phase {n} is around $25,000 and we have not spent more than 40 percent of it.",
    "Can someone check whether the backup job number {n} finished last night?",
]


def dictation(words: int, paragraphs: bool = False) -> str:
    """A long dictation of about `words` words, each sentence with its own number (so every part has values to keep)."""
    out, n = [], 0
    while sum(len(s.split()) for s in out) < words:
        out.append(SENTENCES[n % len(SENTENCES)].format(n=n + 1))
        n += 1
    if not paragraphs:
        return " ".join(out)
    return "\n\n".join(" ".join(out[i:i + 5]) for i in range(0, len(out), 5))


class Model:
    """A fake model: answers each part with `answer(text)` (the part itself by default), counting calls, threads and the
    cost tracker's feature it was called under."""

    def __init__(self, answer=lambda prompt, text: text, delay=0.0):
        self.answer, self.delay = answer, delay
        self.calls, self.features, self.threads, self.running, self.most = [], [], set(), 0, 0
        self._lock = threading.Lock()

    def __call__(self, prompt, text):
        with self._lock:
            self.calls.append((prompt, text))
            self.features.append(usage._feature.get())
            self.threads.add(threading.current_thread().name)
            self.running += 1
            self.most = max(self.most, self.running)
        try:
            time.sleep(self.delay)
            return self.answer(prompt, text)
        finally:
            with self._lock:
                self.running -= 1


# ---- splitting


@pytest.mark.parametrize("words, paragraphs", [(700, False), (1000, True), (3500, False)])
def test_a_long_text_is_split_between_sentences_and_joins_back_to_itself(words, paragraphs):
    text = dictation(words, paragraphs)
    parts = split_parts(text)
    assert "".join(p + gap for p, gap in parts) == text
    # A short last part joins the one before it.
    assert len(parts) >= 2 and all(len(p.split()) <= PART_WORDS * 1.25 + 20 for p, _ in parts)
    assert all(p.rstrip().endswith((".", "?")) for p, _ in parts)  # never in the middle of a sentence
    if paragraphs:
        assert all(gap in ("\n\n", "") for _, gap in parts)


def test_a_short_text_is_one_part():
    text = dictation(SPLIT_OVER - 30)
    assert split_parts(text) == [(text, "")]


def test_a_text_without_punctuation_is_split_between_words():
    text = " ".join(["word"] * 1200)
    parts = split_parts(text)
    assert "".join(p + gap for p, gap in parts) == text and len(parts) >= 3


def test_japanese_is_measured_by_its_letters_and_split_after_its_full_stops():
    text = "明日の会議は午後3時から始まりますので、資料の準備をお願いします。" * 60  # no spaces: one "word" by split()
    parts = split_parts(text)
    assert "".join(p + gap for p, gap in parts) == text and len(parts) >= 2
    assert all(p.endswith("。") for p, _ in parts)


# ---- transforming in parts


@pytest.mark.parametrize("words", [700, 1000, 3500])
def test_a_long_dictation_is_transformed_in_parts(words):
    text = dictation(words)
    model = Model(delay=0.02)
    with usage.feature("transform"):
        got = Transformer(model).transform(text, "rewrite")
    assert got.accepted, got.reasons
    assert got.parts == len(split_parts(text)) == len(model.calls) and not got.kept
    assert " ".join(got.plain.split()) == " ".join(text.split())  # every part back, in order
    assert all(f"part {i + 1} of {got.parts}" in prompt for i, (prompt, _) in
               enumerate(sorted(model.calls, key=lambda c: text.index(c[1]))))
    assert set(model.features) == {"transform"}  # the cost tracker counts every part as Text Transform
    assert 1 < model.most <= 3  # in parallel, a few at a time


def test_concise_and_professional_give_each_part_a_paragraph():
    text = dictation(1000)
    got = Transformer(Model()).transform(text, "concise")
    assert got.accepted and got.plain.count("\n\n") == got.parts - 1


def test_a_part_the_check_rejects_twice_stays_as_written_and_the_user_is_told():
    text = dictation(1000)
    second = split_parts(text)[1][0]

    def answer(prompt, part):  # the model loses a value in the second part, twice
        return part.replace("$25,000", "a lot") if part == second else part
    model = Model(answer)
    got = Transformer(model).transform(text, "concise")
    assert got.accepted and got.parts >= 3
    assert got.kept == [f"part 2 of {got.parts} stayed as written: lost '$25,000'"]
    assert second in got.plain and "a lot" not in got.plain  # the user's own words, never the bad version
    assert sum(part == second for _, part in model.calls) == 2  # one repair attempt, naming what went wrong
    assert any("rejected: lost '$25,000'" in prompt for prompt, part in model.calls if part == second)


def test_every_part_rejected_keeps_the_whole_text():
    got = Transformer(Model(lambda p, part: part.replace("$25,000", "a lot"))).transform(dictation(1000), "concise")
    assert not got.accepted and got.plain == "" and got.reasons[0].startswith("part 1 of")


def test_a_translation_with_a_part_rejected_keeps_the_whole_text():
    """Half the text in another language is no translation: one part rejected, and nothing replaces the text."""
    text = dictation(1000)
    first = split_parts(text)[0][0]

    def answer(prompt, part):
        return "Hola." if part == first else part
    got = Transformer(Model(answer)).transform(text, "translate:English")
    assert not got.accepted and got.reasons[0].startswith(f"part 1 of {got.parts}:")


def test_action_items_of_a_long_text_are_one_list():
    text = dictation(1000)
    parts = split_parts(text)

    def answer(prompt, part):
        if part == parts[1][0]:
            return NO_ACTIONS  # a part without tasks adds nothing
        n = part.split("ticket ")[1].split()[0]
        return f"**Action items**\n- Sarah — look at ticket {n} on Friday, or maybe Monday depending on testing"
    got = Transformer(Model(answer)).transform(text, "actions")
    assert got.accepted, got.reasons
    lines = got.plain.splitlines()
    assert lines[0] == "Action items" and lines.count("Action items") == 1  # one heading, not one a part
    assert len([line for line in lines if line.startswith("- ")]) == got.parts - 1
    assert got.html.count("<ul>") == 1


def test_a_long_text_without_any_action_says_so():
    got = Transformer(Model(lambda p, part: NO_ACTIONS)).transform(dictation(1000), "actions")
    assert not got.accepted and got.reasons == ["no action items in the text"]


def test_a_japanese_template_is_never_split():
    """A Teams message or an email has one greeting and one closing: the whole text goes in one request."""
    model = Model(lambda p, part: "お疲れ様です。\n" + part + "\nよろしくお願いいたします。")
    text = dictation(1000)
    Transformer(model).transform(text, "teams")
    assert all(part == text and "part 1 of" not in prompt for prompt, part in model.calls)


def test_a_part_that_gets_no_answer_fails_the_whole_transform():
    text = dictation(1000)
    second = split_parts(text)[1][0]

    def answer(prompt, part):
        if part == second:
            raise TimeoutError("gemini took too long")
        return part
    with pytest.raises(TimeoutError):  # Text Transform says why, and the text stays as it was
        Transformer(Model(answer)).transform(text, "concise")
