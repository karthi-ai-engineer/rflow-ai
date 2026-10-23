"""A typed keyword as the spotter's word pieces, in process: no sentencepiece package, and not sherpa-onnx's own
tokenizer (it prints hundreds of warnings and ends the process on a bad path).

The gigaspeech spotter's bpe.model is a sentencepiece ModelProto (protobuf): field 1 = the pieces (1 = text, 2 = score,
3 = type; 1 is a normal piece), field 2 = the trainer spec (3 = model type; 1 = unigram). A unigram model splits each word
("▁" + the word) into the pieces whose scores add up highest: a Viterbi search, as sentencepiece does. Measured: the
same lines as the model's own keywords.txt (tests/test_wake_tokens.py).
"""
import struct
from pathlib import Path

SPACE = "▁"  # sentencepiece's word start
NORMAL, UNIGRAM = 1, 1


def _varint(data: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[i]
        value |= (byte & 0x7F) << shift
        i, shift = i + 1, shift + 7
        if byte < 0x80:
            return value, i


def fields(data: bytes):
    """A protobuf message's fields: (number, wire type, value): an int, or bytes for the other wire types."""
    i = 0
    while i < len(data):
        key, i = _varint(data, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(data, i)
        elif wire == 1:
            value, i = data[i:i + 8], i + 8
        elif wire == 2:
            size, i = _varint(data, i)
            value, i = data[i:i + size], i + size
        elif wire == 5:
            value, i = data[i:i + 4], i + 4
        else:
            raise ValueError(f"not a sentencepiece model (protobuf wire type {wire})")
        yield number, wire, value


def read_model(path: Path) -> tuple[list[tuple[str, float, int]], int]:
    """The pieces (text, score, type) in id order, and the model type."""
    pieces, model_type = [], UNIGRAM
    for number, _, value in fields(Path(path).read_bytes()):
        if number == 1:
            piece = {1: b"", 2: b"\0\0\0\0", 3: NORMAL}
            piece.update((n, v) for n, _, v in fields(value))
            pieces.append((piece[1].decode("utf-8"), struct.unpack("<f", piece[2])[0], piece[3]))
        elif number == 2:
            model_type = next((v for n, _, v in fields(value) if n == 3), UNIGRAM)
    return pieces, model_type


class Unigram:
    """encode("our flow") -> ["▁OUR", "▁F", "LO", "W"]: the spotter's pieces for a keyword."""

    def __init__(self, path: Path):
        pieces, model_type = read_model(path)
        if model_type != UNIGRAM or not pieces:
            raise ValueError(f"{path} isn't a unigram sentencepiece model")
        self.scores = {text: score for text, score, kind in pieces if kind == NORMAL}
        self.longest = max(len(text) for text in self.scores)
        self.unknown = min(self.scores.values()) - 10  # a character no piece covers: kept alone, at a high cost

    def encode_word(self, word: str) -> list[str]:
        word = SPACE + word
        best = [(0.0, 0)] + [(float("-inf"), 0)] * len(word)  # best score of word[:j], and where its last piece starts
        for i in range(len(word)):
            if best[i][0] == float("-inf"):
                continue
            for j in range(i + 1, min(len(word), i + self.longest) + 1):
                score = self.scores.get(word[i:j])
                if score is None and j == i + 1:
                    score = self.unknown
                if score is not None and best[i][0] + score > best[j][0]:
                    best[j] = (best[i][0] + score, i)
        out, j = [], len(word)
        while j > 0:
            i = best[j][1]
            out.append(word[i:j])
            j = i
        return out[::-1]

    def encode(self, text: str) -> list[str]:
        return [piece for word in text.upper().split() for piece in self.encode_word(word)]


def read_tokens(path: Path) -> set[str]:
    """The pieces the spotter knows (tokens.txt: "piece id" per line)."""
    return {line.rsplit(" ", 1)[0] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


def keyword_line(pieces: list[str], name: str, boost: float | None = None, threshold: float | None = None) -> str:
    """One keyword as sherpa-onnx reads it: "▁OUR ▁F LO W :1.0 #0.25 @name" (the name is what a hit returns)."""
    line = " ".join(pieces)
    if boost is not None:
        line += f" :{boost}"
    if threshold is not None:
        line += f" #{threshold}"
    return f"{line} @{name}"
