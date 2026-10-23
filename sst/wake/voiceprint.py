"""The voice check: is it the user's voice saying the keyword? NVIDIA's TitaNet-small speaker model (sherpa-onnx's
export) turns speech into 192 numbers; the user's voice print is the mean of the setup tries' (each scaled to length 1),
and a voice scores its cosine similarity to it.

The spike (the owner enrolled from 10 recordings, tested on 27; 40 LibriSpeech speakers, two synthetic voices and others
as impostors): on 3 s, the owner's lowest 5% scored 0.23 and the highest other voice 0.21 (equal error 1.6%); on 1.5 s
they overlap more (5.5%). So the check takes the keyword and the turn's first 3 s, at 0.2. 58 ms per check of 3 s.
Only these numbers are kept, never the audio.
"""
import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sst.wake import models

VOICE_THRESHOLD = 0.2
RATE = 16_000
TRIM_DB = 40.0  # the silence before and after speech is cut: what's this far under the loudest 10 ms

log = logging.getLogger(__name__)
_lock = threading.Lock()
_extractors: dict[str, object] = {}


def _extractor(folder: Path):
    with _lock:
        if str(folder) not in _extractors:
            models.check(models.VOICE, folder)  # before sherpa-onnx sees it: its checks end the process
            import sherpa_onnx

            _extractors[str(folder)] = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(folder / models.VOICE_MODEL), num_threads=1, provider="cpu"))
            log.info("Wake word: the voice model is loaded")
        return _extractors[str(folder)]


def unload() -> None:
    with _lock:
        _extractors.clear()


def trim(audio: np.ndarray) -> np.ndarray:
    """Without the near-silence before and after the speech."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    frames = audio[: len(audio) // 160 * 160].reshape(-1, 160)
    if not len(frames):
        return audio
    level = 10 * np.log10((frames.astype(np.float64) ** 2).mean(1) + 1e-12)
    loud = np.nonzero(level > level.max() - TRIM_DB)[0]
    return audio[loud[0] * 160:(loud[-1] + 1) * 160]


def embed(audio: np.ndarray, model_dir: Path | None = None) -> np.ndarray | None:
    """The voice's 192 numbers, scaled to length 1; None for audio too short to tell."""
    extractor = _extractor(Path(model_dir or models.VOICE.path()))
    with _lock:  # one model for the listener and the setup
        stream = extractor.create_stream()
        stream.accept_waveform(RATE, trim(audio))
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        vector = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else None


class VoicePrint:
    MODEL = "nemo_en_titanet_small"  # a print only compares with prints of the same model

    def __init__(self, vector: Sequence[float]):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not norm:
            raise ValueError("an empty voice print")
        self.vector = vector / norm

    @classmethod
    def from_takes(cls, takes: Sequence[np.ndarray], model_dir: Path | None = None) -> "VoicePrint":
        """The voice print of the setup's recordings (16 kHz float32)."""
        vectors = [v for v in (embed(take, model_dir) for take in takes) if v is not None]
        if not vectors:
            raise ValueError("no speech to make a voice print from")
        return cls(np.mean(vectors, axis=0))

    def score(self, audio: np.ndarray, model_dir: Path | None = None) -> float:
        """How alike the voice in `audio` is (cosine similarity: 1 the same, around 0 another voice)."""
        vector = embed(audio, model_dir)
        return float(vector @ self.vector) if vector is not None and len(vector) == len(self.vector) else 0.0

    def to_json(self) -> dict:
        return {"model": self.MODEL, "vector": [round(float(x), 6) for x in self.vector]}

    @classmethod
    def from_json(cls, data: dict) -> "VoicePrint":
        if data.get("model") != cls.MODEL:
            raise ValueError(f"a voice print of another model: {data.get('model')!r}")
        return cls(data["vector"])
