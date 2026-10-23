"""The wake word's words: which keywords can be used, how the spotter is told to listen for them, and cutting the
keyword and the end phrase out of the dictated text.

Spellings: the spotter hears what a keyword sounds like, written in ordinary words. Letter names are heard badly
("Rflow" as written: 17-20% of tries in the spike; "R flow" 22%; "are flow" 55%; "our flow" 76%), so a letter said by
its name is written as the word that sounds like it. Such a letter is found three ways: a letter on its own ("R flow"),
a single letter before a capitalised part ("RFlow", "iPhone"), or a consonant before another that no English word
starts with ("Rflow": no word starts "rf"). Short capitals ("AI") and words without a vowel ("GPT") are said letter by
letter. Real words are kept as they are.

Names: the spotter knows ~500 English word pieces, and a name like "karthi" is heard as whatever English spelling the
voice sounds like, which differs from voice to voice (measured, the same "karthi": Windows' Zira as "karthy" and
"carthy", Piper's danny as "carfi" and "kathy"; as written, 1 of 3 and 1 of 9). So the setup also tries the spellings
that sound alike (sounds_like) and the words the model itself heard in the user's tries (same_sound decides which of
those can stand for the keyword), and keeps the one heard most.
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


# Ways an English reader writes the same sound, tried on the words that end in "i" (names like Karthi, Preethi, Ravi;
# English words rarely do, and "start" mustn't become "stat"): the ending "i" as "y" (Zira's "karthi" was
# heard as "karthy" 3 of 3, as written 1 of 3), a hard "k" as "c" ("carthi", "carthy"), "ar" before a consonant without
# its r ("kathi", "kathy": non-rhotic, and Piper's "karthi" was heard as "kathy" 4 of 9). Not "th" as a dental t:
# "karti" was heard 0 of 12 in both voices; when a voice says it so, the setup learns it from what the model heard.
_ALIKE = ((r"i\b", "y"), (r"\bk(?=[aou])", "c"), (r"(?<=[a-z])ar(?=[^aeiouy\s])", "a"))
MAX_ALIKE = 6


def sounds_like(text: str) -> list[str]:
    """`text`'s first spelling and up to MAX_ALIKE - 1 others that sound alike, the fewest changes first:
    "karthi" -> ["karthi", "karthy", "carthi", "kathi", "carthy", "kathy"]; "okay type" -> ["okay type"]."""
    base = spellings(text)[0].split()
    found = {" ".join(base): 0}
    for subset in range(1, 1 << len(_ALIKE)):
        words = list(base)
        for i, (pattern, replacement) in enumerate(_ALIKE):
            if subset >> i & 1:
                words = [re.sub(pattern, replacement, w) if len(b) > 2 and b.endswith("i") else w
                         for w, b in zip(words, base, strict=True)]
        spelled = " ".join(words)
        found[spelled] = min(found.get(spelled, len(_ALIKE)), bin(subset).count("1"))
    return sorted(found, key=found.get)[:MAX_ALIKE]


def _sound(text: str) -> str:
    """Letters as they sound, roughly: "Carthy" and "karti" alike ("karti"), "cardi" too (a d and a t)."""
    s = re.sub(r"[^a-z ]", "", text.lower())
    for pattern, replacement in ((r"ph", "f"), (r"th", "t"), (r"ck", "k"), (r"c(?=[aoulr ]|$)", "k"), (r"c", "s"),
                                 (r"q", "k"), (r"d", "t"), (r"z", "s"), (r"y\b", "i"), (r"ee", "i"), (r"(.)\1", r"\1"),
                                 (r"(?<=[aeiou])r(?=[^aeiou ])", "")):
        s = re.sub(pattern, replacement, s)
    return s.replace(" ", "")


def _close(a: str, b: str, ratio: float = 0.6) -> bool:
    return bool(a) and bool(b) and a[0] == b[0] and SequenceMatcher(None, a, b).ratio() >= ratio


def same_sound(heard: str, keyword: str) -> bool:
    """Could `heard`, the words the spotter's model wrote for a try, stand for `keyword`? Close in sound, each of the
    keyword's own words (not "okay", "hey", "start") found in its words from the same first sound ("CARTHY", "CARDI" for
    "karthi"; not "PARTY": a word said in dictation wouldn't start it; "carth is start" for "okay karthi start", never
    "ok start"), and a keyword that could be chosen on its own (not "the", not one syllable)."""
    a, b = _sound(heard), _sound(spellings(keyword)[0])
    own = [word for word in spellings(keyword)[0].split() if word not in COMMON]
    if check_keyword(heard) is not None or not a or SequenceMatcher(None, a, b).ratio() < 0.6:
        return False
    if not own:  # only words like "okay type": the whole of it, from its first sound
        return _close(a, b)
    said = heard.lower().split()  # "hey" may go unheard ("carth had please"), but never the name
    runs = {_sound(" ".join(said[i:j])) for i in range(len(said)) for j in range(i + 1, min(len(said), i + 3) + 1)}
    return all(any(_close(run, _sound(word), 0.65) for run in runs) for word in own)  # whole words, not "okay guy"


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
