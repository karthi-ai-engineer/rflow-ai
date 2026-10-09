"""A typed keyword as the wake word spotter's word pieces, read from its bpe.model in process: against the model's own
keywords files (made by sentencepiece), with a small cut of the model (the real one isn't in the repository)."""
from pathlib import Path

import pytest

from sst.wake.tokens import Unigram, keyword_line, read_model, read_tokens

# The gigaspeech spotter's bpe.model cut down to the pieces these words can use (its trainer spec and piece messages
# byte for byte; the 237 KB normaliser left out), with their ids from tokens.txt.
FIXTURE = Path(__file__).parent / "fixtures" / "wake"

# The model's own keywords_raw.txt -> keywords.txt, and test_wavs/test_keywords.txt.
SHIPPED = {
    "HELLO WORLD": "▁HE LL O ▁WORLD",
    "HI GOOGLE": "▁HI ▁GO O G LE",
    "HEY SIRI": "▁HE Y ▁S I RI",
    "ALEXA": "▁A LE X A",
    "LOVE AND PEACE": "▁LOVE ▁AND ▁P E A CE",
    "PLAY MUSIC": "▁PLAY ▁MU S IC",
    "GO HOME": "▁GO ▁HOME",
    "HAPPY NEW YEAR": "▁HA PP Y ▁NEW ▁YEAR",
    "MERRY CHRISTMAS": "▁ME R RY ▁ CH R IST MA S",
    "LIGHT UP": "▁ L IGHT ▁UP",
    "LOVELY CHILD": "▁LOVE LY ▁CHI L D",
    "FOREVER": "▁FOR E VER",
}


@pytest.fixture(scope="module")
def unigram():
    return Unigram(FIXTURE / "bpe.model")


@pytest.mark.parametrize("raw", SHIPPED)
def test_the_tokenizer_matches_the_models_own_keywords(unigram, raw):
    assert " ".join(unigram.encode(raw)) == SHIPPED[raw]


def test_keywords_are_tokenized_in_capitals_word_by_word(unigram):
    assert unigram.encode("our flow") == ["▁OUR", "▁F", "LO", "W"]
    assert unigram.encode("  Stop   it ") == ["▁ST", "O", "P", "▁IT"]
    assert unigram.encode("q") == ["▁", "Q"]  # no piece for it: the character alone (the spotter refuses it)


def test_the_model_file_is_read_as_sentencepiece_protobuf():
    pieces, model_type = read_model(FIXTURE / "bpe.model")
    assert model_type == 1  # unigram
    assert pieces[:3] == [("<blk>", 0.0, 4), ("<sos/eos>", 0.0, 4), ("<unk>", 0.0, 2)]
    assert all(kind == 1 and score < 0 for _, score, kind in pieces[3:])
    ids = dict(line.split() for line in (FIXTURE / "tokens.txt").read_text(encoding="utf-8").splitlines())
    assert [text for text, _, _ in pieces] == list(ids)  # tokens.txt lists the same pieces in the same order
    assert "▁OUR" in read_tokens(FIXTURE / "tokens.txt") and "Q" not in read_tokens(FIXTURE / "tokens.txt")


def test_a_file_that_isnt_a_unigram_model_is_refused(tmp_path):
    bpe = tmp_path / "bpe.model"
    bpe.write_bytes(bytes([0x12, 0x02, 0x18, 0x02]))  # trainer spec: model type 2 (BPE), no pieces
    with pytest.raises(ValueError):
        Unigram(bpe)
    bpe.write_bytes(bytes([0x0B]))  # wire type 3: not a protobuf sentencepiece writes
    with pytest.raises(ValueError):
        Unigram(bpe)


def test_a_keyword_line_as_sherpa_onnx_reads_it():
    assert keyword_line(["▁OUR", "▁F", "LO", "W"], "k0", 1.0, 0.25) == "▁OUR ▁F LO W :1.0 #0.25 @k0"
    assert keyword_line(["▁GO", "▁HOME"], "home") == "▁GO ▁HOME @home"
