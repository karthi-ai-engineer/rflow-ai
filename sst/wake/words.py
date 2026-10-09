"""The wake word's words: which keywords can be used, how the spotter is told to listen for them, and cutting the
keyword and the end phrase out of the dictated text.

Spellings: the spotter hears what a keyword sounds like, written in ordinary words. Letter names are heard badly
("Rflow" as written: 17-20% of tries in the spike; "R flow" 22%; "are flow" 55%; "our flow" 76%), so a letter said by
its name is written as the word that sounds like it. Such a letter is found three ways: a letter on its own ("R flow"),
a single letter before a capitalised part ("RFlow", "iPhone"), or a consonant before another that no English word
starts with ("Rflow": no word starts "rf"). Short capitals ("AI") and words without a vowel ("GPT") are said letter by
letter. Real words are kept as they are.
"""
import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from itertools import product

END_PHRASES = ("stop it",)  # the default; the user may add others ("end it"; never "done": it's said all the time)
MAX_SPELLINGS = 4
MAX_WORDS = 4

# A letter said by its name, as words that sound like it: the spotter's best first (R: "our flow" beat "are flow").
LETTERS = {
    "a": ["ay"], "b": ["be", "bee"], "c": ["see", "sea"], "d": ["dee"], "e": ["ee"], "f": ["ef"], "g": ["gee"],
    "h": ["aitch"], "i": ["eye"], "j": ["jay"], "k": ["kay"], "l": ["el"], "m": ["em"], "n": ["en"], "o": ["oh", "owe"],
    "p": ["pea"], "q": ["queue", "cue"], "r": ["our", "are"], "s": ["ess"], "t": ["tea", "tee"], "u": ["you"],
    "v": ["vee"], "w": ["double you"], "x": ["ex"], "y": ["why"], "z": ["zee"],
}
# Two consonants English words start with; any other pair at a word's start means its first letter is said by name.
ONSETS = set("""bl br ch cl cr dr dw fl fr gh gl gn gr kl kn kr ll mn ph pl pn pr ps pt qu rh sc sh sk sl sm sn sp sq st
sv sw th tr ts tw tz vl wh wr zh""".split())
VOWELS = set("aeiouy")  # y too: "Lynn" (but a word starting with y starts with a consonant: "Yflow")
# Said too often to start dictation on their own (the spike: 23 of 24 real "done"s were heard as the keyword).
COMMON = set("""a about all and are be but done end enter finish go hello hey hi i it just like listen next no now ok okay
please right send so sorry start stop thanks that the this type uh um well what wait write yeah yep yes you""".split()) | {
    "all right", "end it", "excuse me", "hey there", "i mean", "new line", "new paragraph", "of course", "stop it",
    "thank you", "wake up", "you know"} | set(END_PHRASES)


def prompts(keyword: str) -> list[str]:
    """What to say in each of the 10 setup tries: the ways the keyword will really be said, so that the spelling chosen
    and the voice print fit them all."""
    k = keyword.strip()
    return [f"Say “{k}” the way you normally would.",
            f"Say “{k}” quietly.",
            f"Say “{k}” a little louder.",
            f"Say “{k}” quickly.",
            f"Say “{k}” slowly.",
            f"Lean in close to the microphone and say “{k}”.",
            f"Lean back, a little further from the microphone, and say “{k}”.",
            f"Say it in a sentence: “{k}, make it concise.”",
            f"Say “{k}” normally again.",
            f"One last time: say “{k}” normally."]


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s\-]+", text.strip()) if w]


def _plain(text: str) -> str:
    """Lower case, words only: "Hey, Rflow!" -> "hey rflow"."""
    return " ".join(re.findall(r"[a-z']+", text.lower()))


def check_keyword(text: str) -> str | None:
    """Why `text` can't be the wake word, or None when it can."""
    text = text.strip()
    if not text:
        return "Type a word or two to say before you dictate."
    if not re.fullmatch(r"[A-Za-z' \-]+", text) or not re.search(r"[A-Za-z]", text):
        return "Use English letters only (A to Z)."
    if _plain(text) in COMMON or (len(_words(text)) == 1 and _plain(text).replace("'", "") in COMMON):
        return f"“{text}” is said too often in ordinary speech: choose a word you wouldn't say by chance."
    if len(_words(text)) > MAX_WORDS:
        return f"Use at most {MAX_WORDS} words: a short keyword is quicker to say and to hear."
    if syllables(spellings(text)[0]) < 2:
        return "Too short: one short word is easily heard by chance. Choose a longer word, or two words."
    return None


def syllables(text: str) -> int:
    """About how many syllables: vowel groups, less a silent final e ("done" 1, "computer" 3, "our flow" 2)."""
    count = 0
    for word in re.findall(r"[a-z]+", text.lower()):
        groups = len(re.findall(r"[aeiouy]+", word))
        if groups > 1 and word.endswith("e") and not word.endswith(("le", "ee")):
            groups -= 1
        count += max(groups, 1)
    return count


def _parts(word: str) -> list[str]:
    """A typed word in the pieces it's said in: "RFlow" -> ["R", "Flow"], "iPhone" -> ["i", "Phone"]."""
    return re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z']+|[A-Z']+", word) or [word]


def _say(part: str, alone: bool) -> list[str]:
    """How one piece of a word sounds, as ordinary words: the alternatives, best first."""
    letters = re.sub(r"[^a-z]", "", part.lower())
    if len(letters) == 1:
        return [letters] if alone and letters in ("a", "i") else LETTERS[letters]  # "a" and "I" are words
    if not VOWELS & set(letters):  # "GPT", "Rflw": said letter by letter
        return [" ".join(LETTERS[c][0] for c in letters)]
    if letters[0] not in "aeiou" and letters[1] not in VOWELS and letters[:2] not in ONSETS:  # "Rflow"; "Yflow"; not "Lynn"
        return [f"{name} {rest}" for name, rest in product(LETTERS[letters[0]], _say(letters[1:], False))]
    return [part.lower()]


def spellings(text: str) -> list[str]:
    """1-4 ways to write `text` for the spotter, the likeliest to be heard first: "Rflow" -> ["our flow", "are flow"];
    real words are kept as they are ("okay type" -> ["okay type"])."""
    options = []
    for word in _words(text):
        parts = _parts(word)
        if len(parts) == 1 and parts[0].isupper() and len(parts[0]) in (2, 3):  # an acronym said letter by letter: "AI"
            options.append([" ".join(LETTERS[c][0] for c in parts[0].lower() if c in LETTERS)])
            continue
        alternatives = [""]
        for part in parts:
            alternatives = [f"{a} {b}".strip() for a, b in product(alternatives, _say(part, len(parts) == 1))]
        options.append(alternatives)
    if not options:
        return [_plain(text)]
    ranked = sorted(product(*(range(len(o)) for o in options)), key=sum)  # the best alternatives together first
    out = list(dict.fromkeys(" ".join(o[i] for o, i in zip(options, pick, strict=True)) for pick in ranked))
    return out[:MAX_SPELLINGS]


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def strip_keyword(text: str, keyword: str, spelling: str) -> str:
    """`text` without the keyword it starts with, however the speech model wrote it ("Our flow, hi" -> "hi", and
    "Rflow", "R flow", "Arflow" alike). Unchanged when it doesn't start with the keyword."""
    targets = {t for t in (_letters(keyword), _letters(spelling), *map(_letters, spellings(keyword))) if t}
    words = list(re.finditer(r"\S+", text))
    longest = max((len(s.split()) for s in (keyword, spelling, *spellings(keyword))), default=1) + 1
    best = None  # (similarity, words)
    for n in range(1, min(len(words), longest) + 1):
        head = _letters(" ".join(w.group() for w in words[:n]))
        for target in targets if head else ():
            match = SequenceMatcher(None, head, target)
            score = match.ratio()  # and the likeness starts at the start: not "The computer" for "computer"
            if score >= 0.8 and match.get_matching_blocks()[0].a <= 1 and (best is None or score > best[0]):
                best = (score, n)
    if best is None:
        return text
    return text[words[best[1] - 1].end():].lstrip(" \t\r\n,.;:!?-–—")


def strip_end(text: str, phrases: Sequence[str]) -> str:
    """`text` without the end phrase it ends with ("Send it, stop it." -> "Send it."); a phrase said twice goes too.
    Only a whole match at the very end: "stop it from sending" stays."""
    targets = sorted({_letters(p) for p in phrases if _letters(p)}, key=len, reverse=True)
    while True:
        words = list(re.finditer(r"[A-Za-z']+", text))
        cut = None
        for n in range(1, min(len(words), MAX_WORDS + 1) + 1):
            if _letters(" ".join(w.group() for w in words[-n:])) in targets and not re.search(r"[A-Za-z0-9]",
                                                                                              text[words[-1].end():]):
                cut = n
                break
        if cut is None:
            return text
        mark = next((c for c in text[words[-1].end():] if c in ".!?"), "")
        head = text[:words[-cut].start()].rstrip().rstrip(",;:-–—").rstrip()
        if mark and head and head[-1] not in ".!?":
            head += mark
        text = head
