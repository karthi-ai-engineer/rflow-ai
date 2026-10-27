"""NVIDIA Parakeet (English) running locally on the CPU via sherpa-onnx.

With the model's word-piece vocabulary (bpe.vocab) the recogniser decodes with beam search and listens for the user's
own words ("hotwords"): on the owner's reading test that halved the errors on names and tech terms (40% -> 24.5%)
without hurting the other words. Without it, it decodes greedily as before.

The model is downloaded when the user chooses it (sst.downloads, ~660 MB), from the sherpa-onnx author's copy on
Hugging Face at a pinned revision: byte for byte the files Rflow 1.4 and older installed next to the program, where an
update keeps finding them. bpe.vocab (10 KB, cut from NVIDIA's 2.5 GB .nemo archive) comes with Rflow.
"""
import hashlib
import logging
import re
import threading
from pathlib import Path

import numpy as np

from sst import MODELS_DIR
from sst.audio import condition, split_at_pauses
from sst.downloads import Download, ModelFile
from sst.pipeline.contracts import RawTranscript, WordInfo

FOLDER = "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming"
MODEL_DIR = MODELS_DIR / FOLDER  # next to the program (Rflow 1.4 and older), or the source checkout's models/
REPO = "csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming"
REVISION = "8c3a10fb13408c7a7054f6898958bf1c64a8d6c7"
MODEL = Download(FOLDER, f"https://huggingface.co/{REPO}/resolve/{REVISION}/", (
    ModelFile("encoder.int8.onnx", 654040552, "6716910b7a0833997fec7a410494c995d70124001a0e9b66d6370d6aced577e0"),
    ModelFile("decoder.int8.onnx", 7257753, "a5e223392c90e75f8144cdb5eb95af7625db389e39edef2bd1a9c872b3298fe6"),
    ModelFile("joiner.int8.onnx", 1735860, "869f43f7d24595c55581ad3bf249a935fb8a71389fbdaa7504b9f46f93140f8a"),
    ModelFile("tokens.txt", 8952, "dc0b4584ab2e4ddbf888425c076c61b736e7356a015250db7d307e6f1a8188ff"),
    # NVIDIA's model card, kept with the model (see NOTICES.txt)
    ModelFile("bias.md", 1008, "634920adfccd7897759e495166aef1561a04f3d13026afa4fa64fb4097edb9e0"),
    ModelFile("explainability.md", 2428, "bd13ed3acf31895d29327ad9649cf1c27f45a0e0acaeb8faf8c1e12feba1a748"),
    ModelFile("privacy.md", 4066, "10ec3440299669a2430cf5450732b3c2b04f7af2008350c231b335797fbec55c"),
    ModelFile("safety.md", 782, "6307181203b91a766440a7b8276d4d53893fb30c8058636ea2965c1756bf5e8c"),
))
VOCAB = Path(__file__).resolve().parent.parent / "static" / "parakeet" / "bpe.vocab"  # comes with Rflow
# Audio longer than this is transcribed in pieces cut at pauses: as one piece, onnxruntime fails somewhere
# between 4 and 9 minutes (257 s worked, 514 s crashed). Up to 3 minutes stays one piece (the voice
# pipeline's chunks are ~20 s; a longer dictation transcribed whole, as a last resort, goes in 3-minute pieces):
# cutting at 30 s dropped a whole sentence that the model gets right with the full context.
MAX_PIECE_SECONDS = 180
# How strongly the user's words are favoured. Chosen on the owner's tuning sets (0.5 / 1.0 / 1.5 / 2.0): 1.0 was best
# there and on the held-out sets. From 1.5 the model starts hearing names where they weren't said ("the cloud" ->
# "Claude"), and at 2.5 it repeats them ("Claude Claude Claude").
HOTWORD_SCORE = 1.0
BEAM = 4

log = logging.getLogger(__name__)


def find_model() -> Path | None:
    """Where Parakeet is: next to the program (an older Rflow's, or the source checkout's), else downloaded."""
    if _find(MODEL_DIR, "encoder"):
        return MODEL_DIR
    return MODEL.path() if MODEL.installed() else None


def _find(model_dir: Path, prefix: str) -> Path | None:
    # Prefer the int8 file when both float and int8 versions are present.
    matches = sorted(model_dir.glob(f"{prefix}*.onnx"), key=lambda p: "int8" not in p.name)
    return matches[0] if matches else None


def _vocab_matches(vocab: Path, tokens: Path) -> bool:
    """bpe.vocab must list the model's own word pieces in the model's order, or hotwords would be spelled wrong."""
    try:
        pieces = [line.split("\t")[0] for line in vocab.read_text(encoding="utf-8").splitlines()]
        known = [line.rsplit(" ", 1)[0] for line in tokens.read_text(encoding="utf-8").splitlines()]
    except OSError:
        return False
    return bool(pieces) and pieces == known[:len(pieces)]


def hotword_text(words: list[str]) -> str:
    """sherpa-onnx's per-recording hotwords: phrases separated by "/"."""
    cleaned = (re.sub(r"\s+", " ", w.replace("/", " ")).strip() for w in words)
    return "/".join(dict.fromkeys(w for w in cleaned if w))


def runaway(text: str, words: list[str]) -> bool:
    """Boosting gone wrong: one of the user's words said twice in a row (seen as "Claude Claude Claude" when the
    score was too high). Such a result is decoded again without the words."""
    boosted = {w.lower() for phrase in words for w in re.findall(r"[\w']+", phrase)}
    said = re.findall(r"[\w']+", text.lower())
    return any(a == b and a in boosted for a, b in zip(said, said[1:], strict=False))


def token_words(tokens: list[str], times: list[float], end: float, offset: float = 0.0) -> list[WordInfo]:
    """Words with times from the decoder's word pieces and each piece's start (seconds). A piece starting with "▁"
    (sherpa-onnx hands it over as a space) begins a word, the others continue it ("We" "ll" "," -> "Well,"); a word
    ends where the next begins, the last at `end` (the audio's). `offset` moves them all (a piece of longer audio).
    No times ([]) when there isn't one per piece."""
    if not tokens or len(tokens) != len(times):
        return []
    spans: list[list] = []  # [text, start]
    for token, at in zip(tokens, times, strict=True):
        piece = token.replace("\u2581", " ")
        if piece.startswith(" ") or not spans:
            spans.append([piece.strip(), float(at)])  # a lone "▁" starts a word the next piece spells
        else:
            spans[-1][0] += piece.strip()
    spans = [s for s in spans if s[0]]
    ends = [start for _, start in spans[1:]] + [end]
    return [WordInfo(text, offset + start, offset + max(start, stop)) for (text, start), stop in zip(spans, ends, strict=True)]


class ParakeetEngine:
    name = "parakeet"
    title = "Parakeet"

    def __init__(self, model_dir: Path | None = None, num_threads: int = 4, conditioned: bool = True,
                 hotwords: bool = True):
        model_dir = model_dir or find_model()
        if not model_dir or not model_dir.exists():
            raise FileNotFoundError("Parakeet isn't downloaded: choose it on the Speech recognition page (or run "
                                    "uv run python scripts/download_model.py parakeet)")
        import sherpa_onnx

        self.conditioned = conditioned  # audio.condition() first: the laptop microphone is quiet, see there
        self.words: list[str] = []  # the user's words to listen for ("Your words"); the app keeps it up to date
        self._model = model_dir.name
        self._lock = threading.Lock()  # dictation and the reading test may transcribe at the same time
        tokens = model_dir / "tokens.txt"
        encoder = _find(model_dir, "encoder")
        vocab = model_dir / "bpe.vocab" if (model_dir / "bpe.vocab").exists() else VOCAB
        self.biased = bool(hotwords and encoder and _vocab_matches(vocab, tokens))
        if hotwords and encoder and not self.biased:
            log.warning("No usable bpe.vocab in %s: Your words won't be used while recognising. Run "
                        "scripts/download_model.py parakeet to fetch it.", model_dir)
        if encoder:  # transducer (TDT) export: encoder + decoder + joiner
            options = dict(decoding_method="modified_beam_search", max_active_paths=BEAM, modeling_unit="bpe",
                           bpe_vocab=str(vocab), hotwords_score=HOTWORD_SCORE) if self.biased else {}
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(_find(model_dir, "decoder")),
                joiner=str(_find(model_dir, "joiner")),
                tokens=str(tokens),
                model_type="nemo_transducer",
                num_threads=num_threads,
                **options,
            )
        else:  # CTC export: a single model file
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(_find(model_dir, "model")),
                tokens=str(tokens),
                num_threads=num_threads,
            )

    @property
    def signature(self) -> str:
        """What the text depends on (sst.evaluate caches by it): model, decoding, the words, the audio's preparation."""
        decoding = "greedy"
        if self.biased:
            words = hashlib.sha1(hotword_text(self.words).encode("utf-8")).hexdigest()[:10]
            decoding = f"beam{BEAM}|hotwords{HOTWORD_SCORE}:{words}|guard"
        return f"parakeet|{self._model}|{decoding}" + ("|peak-1|retry" if self.conditioned else "")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        with self._lock:
            return self._transcribe(audio, sample_rate)

    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self.conditioned:
            audio = condition(audio)
        texts = (self._decode(piece, sample_rate) for piece in split_at_pauses(audio, sample_rate, MAX_PIECE_SECONDS))
        text = " ".join(text for text in texts if text)
        if not text and self.conditioned and len(audio) > sample_rate and float(np.abs(audio).max()) > 0.05:
            # Audible speech that decoded to nothing (seen on the owner's reading test): each half on its own worked.
            half = len(audio) // 2
            text = " ".join(t for t in (self._decode(audio[:half], sample_rate), self._decode(audio[half:], sample_rate)) if t)
        return text

    def _decode(self, audio: np.ndarray, sample_rate: int) -> str:
        words = list(self.words) if self.biased else []  # read once: the app may change them meanwhile
        text = self._run(audio, sample_rate, hotword_text(words))
        if words and runaway(text, words):
            log.warning("Hotwords ran away (%r); decoding again without them", text)
            text = self._run(audio, sample_rate, "")
        return text

    def _run(self, audio: np.ndarray, sample_rate: int, hotwords: str) -> str:
        return self._result(audio, sample_rate, hotwords).text.strip()

    def _result(self, audio: np.ndarray, sample_rate: int, hotwords: str):
        """sherpa-onnx's result: the text, and for a transducer its word pieces (tokens) with their start times."""
        stream = self._recognizer.create_stream(hotwords=hotwords) if hotwords else self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)
        return stream.result

    # ---- the voice pipeline: the same text, with each word's time

    def transcribe_chunk(self, audio: np.ndarray, sample_rate: int) -> RawTranscript:
        """transcribe() with word times in seconds from the chunk's start, for merging overlapping chunks: the same
        conditioning, Your words, runaway guard, pieces and retry in halves. No times (words=[]) when the model gives
        none."""
        with self._lock:
            if self.conditioned:
                audio = condition(audio)
            decoded, at = [], 0
            for piece in split_at_pauses(audio, sample_rate, MAX_PIECE_SECONDS):
                decoded.append(self._decode_timed(piece, sample_rate, at / sample_rate))
                at += len(piece)
            if (not any(text for text, _ in decoded) and self.conditioned and len(audio) > sample_rate
                    and float(np.abs(audio).max()) > 0.05):  # as in _transcribe: each half on its own
                half = len(audio) // 2
                decoded = [self._decode_timed(audio[:half], sample_rate, 0.0),
                           self._decode_timed(audio[half:], sample_rate, half / sample_rate)]
        text = " ".join(text for text, _ in decoded if text)
        timed = all(words is not None for _, words in decoded)
        return RawTranscript(text, [w for _, words in decoded for w in words or []] if timed else [], "en", self.name)

    def _decode_timed(self, audio: np.ndarray, sample_rate: int, offset: float) -> tuple[str, list[WordInfo] | None]:
        """_decode() with the words' times (None: the model gave none for this text)."""
        words = list(self.words) if self.biased else []
        result = self._result(audio, sample_rate, hotword_text(words))
        if words and runaway(result.text.strip(), words):
            log.warning("Hotwords ran away (%r); decoding again without them", result.text.strip())
            result = self._result(audio, sample_rate, "")
        text = result.text.strip()
        timed = token_words(list(getattr(result, "tokens", None) or []), list(getattr(result, "timestamps", None) or []),
                            len(audio) / sample_rate, offset)
        return text, timed if timed or not text else None
