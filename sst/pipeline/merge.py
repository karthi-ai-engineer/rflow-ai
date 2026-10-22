"""Merging the chunks' transcripts into one (the plan's §33-39).

A chunk cut at the maximum length is followed by one that starts a second earlier, so the words spoken in that second are
transcribed twice; after a natural pause that second is silence. Only the boundary is inspected: the end of the merged
transcript against the start of the next chunk, never the whole text, so long dictations stay fast.

Four levels, strongest first; the first confident one decides:
  1. timestamps: both sides have word times. A word of the next chunk is a duplicate when a previous word at the same
     time (TIME_TOLERANCE) has the same, a normalised or a similar text. When both sides are timed this level decides
     alone: times that disagree mean the user really said it twice ("very very"), whatever the text looks like.
  2. exact: the longest end of the previous text that is also the start of the next.
  3. normalized: the same, ignoring case, punctuation, apostrophes, unicode forms and "five" vs "5".
  4. fuzzy: the same, allowing spelling variants ("piza" for "pizza") over at least two words.
A text-only match must also be confident (`_confident`), or both are kept: losing a word the user said is worse than
typing one twice. Partial words at a forced cut ("Post" + "PostgreSQL") keep the whole word.
"""
import bisect
import difflib
import functools
import math
import unicodedata
from dataclasses import dataclass, field, replace

from sst.pipeline.contracts import ChunkingConfig, ChunkResult, MergedTranscript, WordInfo

TIME_TOLERANCE = 0.35  # seconds: one word's times in two chunks (engines' word times jitter by 0.1-0.3 s)
TAIL_LEAD = 0.3  # previous words ending this long before the next chunk starts still count (an end time may be early)
HEAD_LAG = 0.1  # next words starting this long after the overlap still count (a start time may be late)
WORDS_PER_SECOND = 4  # fast dictation: without word times the window is overlap seconds x this + WINDOW_SLACK tokens
WINDOW_SLACK = 3
TAIL_CAP = 64  # tokens: never look further back, however long the previous chunk was
FUZZY_TOKEN = 0.84  # difflib ratio of a spelling variant: "piza"/"pizza" 0.89; "Tuesday"/"Thursday" (0.80) is not one
FUZZY_AVERAGE = 0.88  # over the matched words of a fuzzy run
FUZZY_TIMED = 0.75  # with timestamps agreeing, a looser likeness is enough

_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
          "seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
# One engine writes "five", another "5": both normalise to "5" ("twenty-five" -> "twentyfive" -> "25").
_NUMBERS = {word: str(n) for n, word in enumerate(_UNITS)} | {
    tens + (_UNITS[unit] if unit else ""): str(10 * n + unit) for n, tens in enumerate(_TENS, 2) for unit in range(10)}
# Words so frequent that seeing one at both edges proves nothing (normalised: no apostrophes). Only words of five letters
# or more can be "strong" alone, and only fragments outside this list count as a cut-off piece of a longer word.
COMMON = frozenset("""
a an the and or but if so as of at by for from in into on onto to up with about over after before than then
i me my we us our you your he him his she her it its they them their there here this that these those
is am are was were be been being do does did done have has had will would can could shall should may might must
not no yes yeah yep ok okay oh uh um hmm ah well just very really too also now still even only
what which who whom whose when where why how all any some each every both more most much many such other another
get got go going gonna wanna want like know think say said see make made take come came look thing things
something anything nothing everything way time people right good new first last
im ive ill id youre youve theyre weve thats theres whats lets dont doesnt didnt isnt arent wasnt cant wont
""".split())
# Never spelling variants of each other, whatever difflib says: a different day or month is a different meaning.
_NAMES = frozenset("monday tuesday wednesday thursday friday saturday sunday january february march april may june july "
                   "august september october november december".split())
_NEGATIONS = frozenset("not no never none nothing nobody nowhere neither nor".split())
_OPPOSITE_PREFIXES = ("un", "in", "im", "il", "ir", "dis", "non", "mis")  # "likely"/"unlikely" are 0.86 alike
_SENTENCE_ENDS = frozenset(".!?…。！？")


def _letters(text: str) -> str:
    """Letters, digits and combining marks (Tamil's vowel signs are marks), lowercase, in one unicode form."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if unicodedata.category(c)[0] in "LNM")


def normalize(token: str) -> str:
    """What two transcriptions of one word are compared by: "Five," -> "5", "don't" -> "dont", "Ｐｉｚｚａ" -> "pizza".
    Punctuation alone normalises to ""."""
    letters = _letters(token)
    return _NUMBERS.get(letters, letters)


@dataclass
class Token:
    """One whitespace-separated piece of a chunk's text, as the engine wrote it (casing and punctuation are kept), with
    the engine's words that belong to it."""
    text: str
    start: float | None = None  # session seconds; None: its chunk has no usable word times
    end: float | None = None
    words: list[WordInfo] = field(default_factory=list)
    chunk_end: float = math.inf  # where its chunk's audio ends: a later chunk can only repeat what came before this
    norm: str = field(init=False)

    def __post_init__(self):
        self.norm = normalize(self.text)


@dataclass
class BoundaryMerge:
    """How one boundary was joined: `tail` replaces the tail that was given, then `head` is appended."""
    tail: list[Token]
    head: list[Token]
    removed: list[str] = field(default_factory=list)  # the duplicates taken out, from either side
    level: str = ""  # "timestamps", "exact", "normalized", "fuzzy" or "partial word"; "" when nothing was removed
    note: str = ""  # for MergedTranscript.dedup: "removed 'large pizza' (exact)", "kept both: weak match 'the'"


def tokenize(result: ChunkResult) -> list[Token]:
    """A chunk's text as tokens, with times when the engine gave usable word times. The text stays the source: it has
    the punctuation and casing that the words may lack."""
    texts = result.text.split()
    groups = _align(texts, result.words) if texts and result.words and _plausible(result) else None
    if groups is None:
        return [Token(text, chunk_end=result.end) for text in texts]
    tokens, last = [], result.start
    for text, words in zip(texts, groups, strict=True):
        # A piece without a word of its own (a dash, a symbol) sits where the previous word ended.
        start, end = (min(w.start for w in words), max(w.end for w in words)) if words else (last, last)
        tokens.append(Token(text, start, end, words, result.end))
        last = end
    return tokens


def _plausible(result: ChunkResult) -> bool:
    """Word times inside the chunk and in order, in session seconds as the contract says. Chunk-relative times (a stage
    that forgot to convert them) would make the timestamp level match the wrong words: such a chunk counts as untimed."""
    previous = -math.inf
    for word in result.words:
        if not (result.start - 0.5 <= word.start <= word.end + 0.05 and word.end <= result.end + 0.5
                and word.start >= previous - 0.2):
            return False
        previous = word.start
    return True


def _align(texts: list[str], words: list[WordInfo]) -> list[list[WordInfo]] | None:
    """Gives each text token the engine's words that belong to it. Usually one each; otherwise ("New York" as one word,
    Japanese without spaces) by their letters, a word without a counterpart going with the one before it."""
    words = [w for w in words if w.text.strip()]
    if not words:
        return None
    if len(words) == len(texts):
        return [[w] for w in words]
    a, a_owner, b, b_owner = [], [], [], []
    for owner, text in enumerate(texts):
        letters = _letters(text)
        a.append(letters)
        a_owner += [owner] * len(letters)
    for owner, word in enumerate(words):
        letters = _letters(word.text)
        b.append(letters)
        b_owner += [owner] * len(letters)
    first: dict[int, int] = {}
    for block in difflib.SequenceMatcher(None, "".join(a), "".join(b), autojunk=False).get_matching_blocks():
        for x in range(block.size):
            first.setdefault(b_owner[block.b + x], a_owner[block.a + x])
    groups: list[list[WordInfo]] = [[] for _ in texts]
    current = 0
    for owner, word in enumerate(words):
        current = max(current, first.get(owner, current))  # never backwards
        groups[current].append(word)
    return groups


# ---------------------------------------------------------------- comparing two tokens

def _digits(text: str) -> bool:
    return any(c.isdigit() for c in text)


def _opposite(a: str, b: str) -> bool:
    """A negation or an opposite: "should"/"shouldnt", "likely"/"unlikely". Never one word misheard."""
    if (a in _NEGATIONS) != (b in _NEGATIONS) or a.endswith("nt") != b.endswith("nt"):
        return True
    return any(a == prefix + b or b == prefix + a for prefix in _OPPOSITE_PREFIXES)


@functools.lru_cache(maxsize=4096)
def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _variants(a: str, b: str, threshold: float, min_length: int) -> bool:
    """Two spellings of one word. Numbers, days and months must match exactly: "1234567" and "1234568" are 0.86 alike."""
    if min(len(a), len(b)) < min_length or 2 * min(len(a), len(b)) / (len(a) + len(b)) < threshold:
        return False
    if _digits(a + b) or a in _NAMES or b in _NAMES or _opposite(a, b):
        return False
    return _similar(a, b) >= threshold


def _differ(a: str, b: str) -> bool:
    """Two transcriptions of the same moment that disagree on something that matters (a number, a negation): neither
    may silently replace the other."""
    return (a != b and _digits(a + b)) or _opposite(a, b)


def _part(piece: str, whole: str, at_start: bool) -> bool:
    """`piece` is a cut-off start (or end) of `whole`: "post" of "postgresql", "ployment" of "deployment"."""
    if len(piece) >= len(whole) or _digits(whole):
        return False
    return whole.startswith(piece) if at_start else whole.endswith(piece)


def _strong(norm: str) -> bool:
    """A word that can prove an overlap alone: long and not one of the words every sentence has."""
    return len(norm) >= 5 and norm not in COMMON


def _piece_points(piece: str) -> int:
    """How much a cut-off piece proves: "post" (2) much, "po" (1) a little, "the" or "an" (0) nothing ("the" is the
    start of "theory", "an" of "answer")."""
    if piece in COMMON or len(piece) < 2:
        return 0
    return 2 if len(piece) >= 4 else 1


def _trailing(text: str) -> str:
    end = len(text)
    while end and not text[end - 1].isalnum():
        end -= 1
    return text[end:]


def _drop_cut_period(prev: Token, new: Token, continuation: Token | None) -> Token:
    """A chunk's engine ends its text with a period even when the speaker didn't stop ("a large pizza." + "large pizza
    for tonight"). The next chunk heard what followed: when its copy of the word has no sentence end and the text goes
    on in lowercase, the previous word takes the next chunk's punctuation instead."""
    tail_prev, tail_new = _trailing(prev.text), _trailing(new.text)
    first = next((c for c in continuation.text if c.isalpha()), "") if continuation else ""
    if _SENTENCE_ENDS & set(tail_prev) and not _SENTENCE_ENDS & set(tail_new) and first.islower():
        return replace(prev, text=prev.text[:len(prev.text) - len(tail_prev)] + tail_new)
    return prev


def _quote(texts: list[str]) -> str:
    return "'" + " ".join(texts) + "'"


# ---------------------------------------------------------------- one boundary

def merge_boundary(tail: list[Token], head: list[Token], start: float, overlap_end: float,
                   window: int | None = None) -> BoundaryMerge:
    """Joins the end of the merged transcript and a new chunk. `tail`: the merged tokens whose chunk's audio reaches past
    `start` (the only ones the new chunk can repeat); `head`: all the new chunk's tokens; [start, overlap_end): the
    audio the new chunk shares with the previous one. `window`: tokens compared on a side without word times."""
    unchanged = BoundaryMerge(list(tail), list(head))
    if overlap_end <= start or not tail or not head:
        return unchanged
    if window is None:
        window = math.ceil((overlap_end - start) * WORDS_PER_SECOND) + WINDOW_SLACK
    tail_timed = all(t.start is not None for t in tail)
    head_timed = all(t.start is not None for t in head)
    lo, count = len(tail), 0
    if tail_timed:
        while lo and tail[lo - 1].end > start - TAIL_LEAD:
            lo -= 1
    else:
        while lo and count < window:
            lo -= 1
            count += bool(tail[lo].norm)
    hi, count = 0, 0
    if head_timed:
        while hi < len(head) and head[hi].start < overlap_end + HEAD_LAG:
            hi += 1
    else:
        while hi < len(head) and count < window:
            count += bool(head[hi].norm)
            hi += 1
    prev = [i for i in range(lo, len(tail)) if tail[i].norm]
    new = [j for j in range(hi) if head[j].norm]
    if not prev or not new:
        return unchanged  # a timed side shows no speech in the overlap (a pause): nothing can be repeated
    if tail_timed and head_timed:
        return _by_time(tail, head, prev, new, start, overlap_end)
    return _by_text(tail, head, prev, new)


def _link(p: Token, n: Token, last_prev: bool, first_new: bool, start: float, overlap_end: float):
    """(weight, time distance, kind) when `n` can be the same spoken word as `p`, else None."""
    ds, de = abs(p.start - n.start), abs(p.end - n.end)
    near = min(ds, de)
    if near <= TIME_TOLERANCE:
        if p.norm == n.norm:
            return 3, near, "same"
        if _variants(p.norm, n.norm, FUZZY_TIMED, 3):
            return 2, near, "fuzzy"
    # The previous chunk was cut inside the word: its piece starts with the word and ends at the cut.
    if (last_prev and ds <= TIME_TOLERANCE and p.end >= overlap_end - TIME_TOLERANCE and _part(p.norm, n.norm, True)
            and _piece_points(p.norm)):
        return 1, ds, "prefix"
    # The next chunk started inside the word: its piece ends with the word and starts with the chunk.
    if (first_new and de <= TIME_TOLERANCE and n.start <= start + TIME_TOLERANCE and _part(n.norm, p.norm, False)
            and _piece_points(n.norm)):
        return 1, de, "suffix"
    return None


def _same_time(p: Token, n: Token) -> bool:
    a0, a1, b0, b1 = p.start, max(p.end, p.start + 0.05), n.start, max(n.end, n.start + 0.05)
    return min(a1, b1) - max(a0, b0) >= 0.5 * min(a1 - a0, b1 - b0)


def _by_time(tail: list[Token], head: list[Token], prev: list[int], new: list[int], start: float,
             overlap_end: float) -> BoundaryMerge:
    links = {}
    for i in prev:
        for j in new:
            if link := _link(tail[i], head[j], i == prev[-1], j == new[0], start, overlap_end):
                links[i, j] = link
    if not links:
        said = [head[j].text for j in new if any(head[j].norm == tail[i].norm for i in prev)]
        return BoundaryMerge(list(tail), list(head), note=f"kept both: {_quote(said)} said again (timestamps)" if said else "")
    # The most (and best) matches in order, then the closest in time: an alignment, so "very very" pairs up one by one.
    best = [[(0, 0.0)] * (len(new) + 1) for _ in range(len(prev) + 1)]
    for a in range(1, len(prev) + 1):
        for b in range(1, len(new) + 1):
            options = [best[a - 1][b], best[a][b - 1]]
            if link := links.get((prev[a - 1], new[b - 1])):
                options.append((best[a - 1][b - 1][0] + link[0], best[a - 1][b - 1][1] - link[1]))
            best[a][b] = max(options)
    anchors, a, b = [], len(prev), len(new)
    while a and b:
        link = links.get((prev[a - 1], new[b - 1]))
        if link and best[a][b] == (best[a - 1][b - 1][0] + link[0], best[a - 1][b - 1][1] - link[1]):
            anchors.append((prev[a - 1], new[b - 1], link[2]))
            a, b = a - 1, b - 1
        elif best[a][b] == best[a - 1][b]:
            a -= 1
        else:
            b -= 1
    anchors.reverse()

    tail = list(tail)
    last_i, last_j, last_kind = anchors[-1]
    prefix = last_kind == "prefix"
    keep_from = last_j if prefix else last_j + 1  # a cut-off previous word gives way to the next chunk's whole word
    if not prefix and last_i == prev[-1]:
        tail[last_i] = _drop_cut_period(tail[last_i], head[last_j], next((t for t in head[keep_from:] if t.norm), None))
    matched = {i for i, _, _ in anchors}
    anchored = {j for _, j, _ in anchors}
    new_order = [j for _, j, _ in anchors]
    removed, added, inserts = [], [], {}
    for j in range(keep_from):
        token = head[j]
        if j in anchored:
            removed.append(token.text)
            continue
        if not token.norm:
            continue  # punctuation inside the repeated stretch
        # A word between two matches: another transcription of what the previous chunk has at that moment, or a word
        # it missed. The second is added at its time; the first is dropped unless the two disagree on what matters.
        k = bisect.bisect(new_order, j)
        lo, hi = (anchors[k - 1][0] + 1 if k else prev[0]), anchors[k][0]
        rivals = [i for i in range(lo, hi) if tail[i].norm and i not in matched and _same_time(tail[i], token)]
        if rivals and not any(_differ(tail[i].norm, token.norm) for i in rivals):
            removed.append(token.text)
            continue
        at = lo
        while at < hi and tail[at].start <= token.start:
            at += 1
        inserts.setdefault(at, []).append(token)
        added.append(token.text)
    new_tail = []
    for i, token in enumerate(tail):
        new_tail += inserts.get(i, [])
        if not (prefix and i >= last_i):  # the cut-off piece and any punctuation after it
            new_tail.append(token)
    new_tail += inserts.get(len(tail), [])
    if prefix:
        removed.append(tail[last_i].text)
    note = f"removed {_quote(removed)} (timestamps)" + (f", added {_quote(added)}" if added else "")
    return BoundaryMerge(new_tail, head[keep_from:], removed, "timestamps", note)


_LEVELS = ("exact", "normalized", "fuzzy")
# Where a run may sit: right at both edges; after the next chunk's first word (cut at its start, or one the previous
# chunk missed); before the previous chunk's last word (garbled at the cut). The skips need two whole words.
_SKIPS = ((0, 0), (0, 1), (1, 0))


def _pair(p: Token, n: Token, level: str, prefix: bool, suffix: bool) -> tuple[str, float] | None:
    if p.text == n.text or (level != "exact" and p.norm == n.norm):
        return "same", 1.0
    if level == "exact":
        return None
    if prefix and _part(p.norm, n.norm, True):
        return "prefix", 1.0
    if suffix and _part(n.norm, p.norm, False):
        return "suffix", 1.0
    if level == "fuzzy" and _variants(p.norm, n.norm, FUZZY_TOKEN, 4):
        return "fuzzy", _similar(p.norm, n.norm)
    return None


def _confident(tail: list[Token], head: list[Token], ps: list[int], ns: list[int], kinds: list[tuple[str, float]],
               level: str, skipped: bool) -> bool:
    """Text alone is evidence only when it is unlikely to be chance: two words ("large pizza", even "of the"), one long
    uncommon word ("PostgreSQL"), or a long cut-off piece ("Post"). A lone "the" at both edges is kept twice."""
    whole = [similarity for kind, similarity in kinds if kind in ("same", "fuzzy")]
    if level == "fuzzy" and (len(whole) < 2 or sum(whole) / len(whole) < FUZZY_AVERAGE):
        return False
    if skipped:
        return len(whole) >= 2
    points = 0
    for (kind, _), i, j in zip(kinds, ps, ns, strict=True):
        if kind in ("same", "fuzzy"):
            points += 2 if _strong(tail[i].norm) else 1
        else:
            points += _piece_points(tail[i].norm if kind == "prefix" else head[j].norm)
    return points >= 2


def _by_text(tail: list[Token], head: list[Token], prev: list[int], new: list[int]) -> BoundaryMerge:
    weak = ""
    for skip_prev, skip_new in _SKIPS:
        for level in _LEVELS:
            for k in range(min(len(prev) - skip_prev, len(new) - skip_new), 0, -1):
                ps, ns = prev[len(prev) - skip_prev - k:len(prev) - skip_prev], new[skip_new:skip_new + k]
                kinds = []
                for x, (i, j) in enumerate(zip(ps, ns, strict=True)):
                    kind = _pair(tail[i], head[j], level, not skip_prev and x == k - 1, not skip_new and x == 0)
                    if kind is None:
                        break
                    kinds.append(kind)
                else:
                    if _confident(tail, head, ps, ns, kinds, level, bool(skip_prev or skip_new)):
                        return _apply_text(tail, head, prev, new, ps, ns, kinds, level, skip_prev, skip_new)
                    if not weak and not skip_prev and not skip_new:
                        weak = " ".join(head[j].text for j in ns)
    return BoundaryMerge(list(tail), list(head), note=f"kept both: weak match '{weak}'" if weak else "")


def _apply_text(tail: list[Token], head: list[Token], prev: list[int], new: list[int], ps: list[int], ns: list[int],
                kinds: list[tuple[str, float]], level: str, skip_prev: int, skip_new: int) -> BoundaryMerge:
    new_tail, removed, added, cut_off = list(tail), [], [], []
    keep_from = ns[-1] + 1
    if kinds[-1][0] == "prefix":  # the previous chunk was cut inside this word: the next chunk has all of it
        keep_from -= 1
        cut_off.append(tail[ps[-1]].text)
        del new_tail[ps[-1]:]  # with any punctuation after the piece
    elif not skip_prev:
        new_tail[ps[-1]] = _drop_cut_period(tail[ps[-1]], head[ns[-1]], next((t for t in head[keep_from:] if t.norm), None))
    skipped = new[0] if skip_new else -1
    if skip_new:
        # The next chunk's first word: the cut-off end of the previous word, or a word the previous chunk missed (kept).
        before = prev[len(prev) - skip_prev - len(ps) - 1] if len(prev) > skip_prev + len(ps) else None
        p, n = (tail[before] if before is not None else None), head[skipped]
        if p and (p.norm == n.norm or _part(n.norm, p.norm, False) or _variants(p.norm, n.norm, FUZZY_TIMED, 3)):
            removed.append(n.text)
        else:
            new_tail.insert(ps[0], n)
            added.append(n.text)
    removed += [head[j].text for j in range(keep_from) if head[j].norm and j != skipped] + cut_off
    label = "partial word" if all(kind in ("prefix", "suffix") for kind, _ in kinds) else level
    note = f"removed {_quote(removed)} ({label})" + (f", added {_quote(added)}" if added else "")
    return BoundaryMerge(new_tail, head[keep_from:], removed, label, note)


# ---------------------------------------------------------------- a whole session

class TranscriptMerger:
    """Merges one session's chunk results, given in sequence order, as they arrive. Each add() returns the provisional
    transcript (shown while the key is held, never typed); final() the one that goes on to the dictionary stage."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self._tokens: list[Token] = []
        self._notes: list[str] = []
        self._chunks = 0
        self._sequence: int | None = None
        self._final = False

    def add(self, result: ChunkResult) -> MergedTranscript:
        if self._final:
            raise RuntimeError("the merged transcript is already final")
        if self._sequence is not None and result.sequence <= self._sequence:
            raise ValueError(f"chunk #{result.sequence} came after #{self._sequence}: results must be in sequence order")
        self._sequence = result.sequence
        self._chunks += 1
        tokens = tokenize(result)
        overlap = result.overlap_end - result.start
        if tokens and self._tokens and overlap > 0:
            cut = len(self._tokens)  # back to the first token whose chunk ended before this one starts
            while cut and len(self._tokens) - cut < TAIL_CAP and self._tokens[cut - 1].chunk_end > result.start:
                cut -= 1
            if cut < len(self._tokens):
                # The text-only window follows the configured overlap, so a result claiming a longer one can't widen it.
                seconds = min(overlap, self.config.overlap_ms / 1000) if self.config.overlap_ms > 0 else overlap
                window = math.ceil(seconds * WORDS_PER_SECOND) + WINDOW_SLACK
                merged = merge_boundary(self._tokens[cut:], tokens, result.start, result.overlap_end, window)
                self._tokens[cut:] = merged.tail
                tokens = merged.head
                if merged.note:
                    self._notes.append(f"#{result.sequence}: {merged.note}")
        self._tokens += tokens
        return self._snapshot(final=False)

    @property
    def tokens(self) -> list[Token]:
        """The merged tokens so far (a copy), each with its chunk's end: what the session cleans up early."""
        return list(self._tokens)

    def final(self) -> MergedTranscript:
        """The session is complete: the transcript that goes on (no more add() after this)."""
        self._final = True
        return self._snapshot(final=True)

    def _snapshot(self, final: bool) -> MergedTranscript:
        return MergedTranscript(text=" ".join(t.text for t in self._tokens),
                                words=[w for t in self._tokens for w in t.words],
                                chunks=self._chunks, dedup=list(self._notes), final=final)
