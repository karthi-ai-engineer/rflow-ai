"""Listening for a few phrases in the microphone's frames: sherpa-onnx's keyword spotter with the gigaspeech model
(the spike: 1.9% of one core, a hit ~0.4 s after the phrase ends).

The model is loaded once (0.8-2.2 s) and shared: each Spotter is a stream of its own with its own phrases. sherpa-onnx
wants a keywords file even then, so the shared one holds a placeholder that can't fire (threshold 0.99, no boost); a
stream's own phrases are added to it (measured on the spike's clips: the same hits as phrases in the file). The model
gives no confidence, only hits at the threshold: 0.25 (0.4 cost 6-20 points, 0.1 gained nothing), boost 1.0 (2.0 was
much worse).
"""
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np

from sst.wake import models
from sst.wake.tokens import Unigram, keyword_line, read_tokens

THRESHOLD = 0.25
BOOST = 1.0
RATE = 16_000
PLACEHOLDER = "everything nothing public health"  # the shared keywords file's phrase, never heard (see above)

log = logging.getLogger(__name__)
_lock = threading.Lock()
_loaded: dict[str, "_Model"] = {}


class _Model:
    def __init__(self, folder: Path):
        models.check(models.KWS, folder)  # before sherpa-onnx sees it: its checks end the process
        self.tokenizer = Unigram(folder / models.BPE)
        self.known = read_tokens(folder / models.TOKENS)
        self.lock = threading.Lock()  # one decoder for every stream
        import sherpa_onnx  # its native library: only when the wake word is used

        with tempfile.TemporaryDirectory(prefix="rflow-kws-") as temp:
            keywords = Path(temp) / "keywords.txt"
            keywords.write_text(keyword_line(self.tokenizer.encode(PLACEHOLDER), "placeholder", 0.0, 0.99) + "\n",
                                encoding="utf-8")
            self.spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(folder / models.TOKENS), encoder=str(folder / models.ENCODER),
                decoder=str(folder / models.DECODER), joiner=str(folder / models.JOINER), keywords_file=str(keywords),
                num_threads=1, keywords_score=BOOST, keywords_threshold=THRESHOLD, num_trailing_blanks=1,
                max_active_paths=4)


def _model(folder: Path) -> _Model:
    with _lock:
        if str(folder) not in _loaded:
            _loaded[str(folder)] = _Model(folder)
            log.info("Wake word: the keyword spotter is loaded")
        return _loaded[str(folder)]


def unload() -> None:
    """Free the shared model (the wake word turned off); the next Spotter loads it again."""
    with _lock:
        _loaded.clear()


class Spotter:
    """Listens for `phrases` (name -> spelling, as spellings() writes it): feed() returns the name of a phrase heard."""

    def __init__(self, phrases: dict[str, str], threshold: float = THRESHOLD, model_dir: Path | None = None):
        if not phrases:
            raise ValueError("nothing to listen for")
        self._model = _model(Path(model_dir or models.KWS.path()))
        lines, self._names = [], {}
        for i, (name, spelling) in enumerate(phrases.items()):
            pieces = self._model.tokenizer.encode(spelling)
            if not pieces or any(p not in self._model.known for p in pieces):  # sherpa-onnx would end the process
                raise ValueError(f"the keyword spotter can't listen for {spelling!r}")
            self._names[f"k{i}"] = name
            lines.append(keyword_line(pieces, f"k{i}", BOOST, threshold))
        self._keywords = "/".join(lines)
        self.reset()

    def reset(self) -> None:
        """Forget what was heard so far (a new stream: nothing half-heard carries over)."""
        with self._model.lock:
            self._stream = self._model.spotter.create_stream(self._keywords)

    def feed(self, frame: np.ndarray) -> str | None:
        """16 kHz mono float32 samples, any length; the name of a phrase heard in them, or None."""
        spotter, hit = self._model.spotter, None
        with self._model.lock:
            self._stream.accept_waveform(RATE, np.asarray(frame, dtype=np.float32).reshape(-1))
            while spotter.is_ready(self._stream):
                spotter.decode_stream(self._stream)
                found = spotter.get_result(self._stream)
                if found:
                    spotter.reset_stream(self._stream)
                    hit = hit or self._names.get(found)
        return hit
