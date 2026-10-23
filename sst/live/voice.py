"""The voice on the laptop that speaks live translations aloud when Gemini's own voice isn't there: Piper's "Ryan"
(English, medium quality, 22 kHz), run on the CPU by sherpa-onnx, which Rflow already has for Parakeet (no Piper
program needed). It replaced "Danny" (low quality, 16 kHz, flat); an older Rflow's Danny is still used until Ryan is
downloaded (local()), so nobody is left without a voice meanwhile.

Downloaded when the user turns speaking on (sst.downloads: pinned, checked, Hugging Face only):
    rhasspy/piper-voices  en_US-ryan-medium.onnx (63 MB) and .onnx.json (its settings and phoneme ids); the voice's
                          dataset is CC BY-NC-SA 4.0 (packaging/NOTICES.txt)
    espeak-ng-data        the 6 files of it English needs (795 KB of 17 MB, measured: the same speech), from the
                          sherpa-onnx author's copy of Danny (the same files for every English voice): the phonemes
                          Piper speaks from

Then prepared, once: sherpa-onnx reads a Piper voice's settings from the model's metadata, which rhasspy's files keep
in the .onnx.json. An ONNX model is a protobuf ModelProto, and protobuf merges repeated fields that come later in the
bytes, so the metadata (field 14) is appended to a copy of the model: no ONNX library needed. tokens.txt comes from the
json's phoneme ids; the original model is removed afterwards (63 MB not kept twice). Measured on the owner's laptop:
Danny loaded in 1.1 s, a sentence in ~0.2 s (about 20 times faster than it takes to say); Ryan: see RYAN.
"""
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sst.downloads import Download, ModelFile, download

PREPARED, TOKENS, MARKER = "model.onnx", "tokens.txt", "prepared.json"
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/c10ece1aade47bb51c153c893d14e5bf8e5b7117/"
ESPEAK = ("https://huggingface.co/csukuangfj/vits-piper-en_US-danny-low/resolve/"
          "9b9d4d57bb63c5fd0b17b4ef002bbf43b05c0d7f/espeak-ng-data/")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Voice:
    key: str
    name: str  # what the user sees
    language: str  # what it speaks (a LANGUAGES code)
    model: Download  # the Piper voice: model_file and its .json
    model_file: str
    data: Download  # espeak-ng-data for its language

    @property
    def size(self) -> int:
        return self.model.size + self.data.size

    def folder(self, root: Path | None = None) -> Path:
        return self.model.path(root)

    def ready(self, root: Path | None = None) -> bool:
        """Downloaded and prepared: it can speak."""
        folder = self.folder(root)
        return all((folder / name).exists() for name in (MARKER, PREPARED, TOKENS, self.model_file + ".json")) \
            and self.data.installed(root)


ESPEAK_EN = Download("espeak-ng-data-en", ESPEAK, (
    ModelFile("phontab", 55796, "886f3fa402cb0ba73d483aa8ad000af47a6b7cc06293c75a97913fba68a530f6"),
    ModelFile("phonindex", 39074, "3ca7b8fa3b42624e4b0f152707e7a39245fce569aa99ea47c055d9e622fcf0c4"),
    ModelFile("phondata", 550424, "4e0288957874029a8c3c9f41a8f517ad4bf18127046decbdd4b9d1d6807ce3a3"),
    ModelFile("intonations", 2040, "3f8af65fd3eda9759a10f021d61361c120871f463515229c925995c7f90918cc"),
    ModelFile("en_dict", 166944, "71bd330ba8a2e3e8076e631508208ef49449d6147c17b7bd2b4b1e1468292e35"),
    ModelFile("lang/gmw/en-US", 257, "41534c2a22df5dd4f1052ff9e1a33a3ea7bff5a26b5c02bdad5ba8ddb7524704"),
))
DANNY = Voice(
    "en_US-danny-low", "Danny", "en",
    Download("piper-en_US-danny-low", PIPER + "en/en_US/danny/low/", (
        ModelFile("en_US-danny-low.onnx", 63104526, "56a9ae9499e961514f060aac2866cb323a21e5989592fc4f208e56fdc323ab64"),
        ModelFile("en_US-danny-low.onnx.json", 4166, "191dea1ba9863199d8ca2e9048f83b219219fb450f62767ee02f1bc4568ce4f4"),
        ModelFile("MODEL_CARD", 275, "28a96e3a87f774b964f9b8ce21d9a504dc2fd1c685ab7992ba2487a8a883f5f7"),  # its licence
    )),
    "en_US-danny-low.onnx",
    ESPEAK_EN,
)
# Clearer than Danny at the same size and speed. Measured on the dev laptop (sherpa-onnx, 2 threads): loaded in 1.0 s;
# 5, 18 and 40 words in 70, 264 and 566 ms (1.2, 5.0 and 10.6 s of speech: ~19 times faster than it takes to say)
RYAN = Voice(
    "en_US-ryan-medium", "Ryan", "en",
    Download("piper-en_US-ryan-medium", PIPER + "en/en_US/ryan/medium/", (
        ModelFile("en_US-ryan-medium.onnx", 63201294, "abf4c274862564ed647ba0d2c47f8ee7c9b717d27bdad9219100eb310db4047a"),
        ModelFile("en_US-ryan-medium.onnx.json", 4883, "44034c056cb15681b2ad494307c7f3f2e4499d1253c700c711fa0a4607ffe78d"),
        ModelFile("MODEL_CARD", 306, "3a24ae5f7aa856199f338ad9af784cda796734f397f7e1b4a1bd3c07d06945b1"),  # its licence
    )),
    "en_US-ryan-medium.onnx",
    ESPEAK_EN,
)
LOCAL = RYAN  # the voice downloaded for the laptop now
VOICES = {RYAN.key: RYAN, DANNY.key: DANNY}


def local(root: Path | None = None) -> Voice:
    """The voice on this laptop to speak with: Ryan once he's downloaded; until then Danny where an older Rflow
    downloaded him (never left without a voice while Ryan comes); else Ryan, to be downloaded."""
    return DANNY if not LOCAL.ready(root) and DANNY.ready(root) else LOCAL


def install(voice: Voice, progress=lambda done, total: None, cancelled=lambda: False, root: Path | None = None) -> Path:
    """Download what's missing of `voice` and prepare it; its folder. Raises sst.downloads' Cancelled or
    DownloadError."""
    folder = voice.folder(root)
    if voice.ready(root):
        return folder
    download(voice.data, lambda done, total: progress(done, voice.size), cancelled, root)
    if not (folder / voice.model_file).exists() and (folder / MARKER).exists():
        (folder / MARKER).unlink()  # prepared once, then damaged: start again from the download
    download(voice.model, lambda done, total: progress(voice.data.size + done, voice.size), cancelled, root)
    prepare(voice, root)
    return folder


def prepare(voice: Voice, root: Path | None = None) -> None:
    """The downloaded voice as sherpa-onnx reads it: model.onnx with the voice's settings in its metadata, tokens.txt."""
    folder = voice.folder(root)
    config = json.loads((folder / (voice.model_file + ".json")).read_text(encoding="utf-8"))
    metadata = {"model_type": "vits", "comment": "piper", "language": config["language"]["name_english"],
                "voice": config["espeak"]["voice"], "has_espeak": 1, "n_speakers": config["num_speakers"],
                "sample_rate": config["audio"]["sample_rate"]}
    with (folder / (TOKENS + ".tmp")).open("w", encoding="utf-8", newline="\n") as out:
        for symbol, ids in config["phoneme_id_map"].items():
            out.write(f"{symbol} {ids[0]}\n")
    (folder / (TOKENS + ".tmp")).replace(folder / TOKENS)
    with (folder / voice.model_file).open("rb") as source, (folder / (PREPARED + ".tmp")).open("wb") as out:
        shutil.copyfileobj(source, out)
        out.write(onnx_metadata(metadata))
    (folder / (PREPARED + ".tmp")).replace(folder / PREPARED)
    (folder / MARKER).write_text(json.dumps({"from": voice.model_file, "metadata": metadata}), encoding="utf-8")
    (folder / voice.model_file).unlink()
    log.info("Prepared the voice %s (%s)", voice.name, voice.key)


def onnx_metadata(entries: dict) -> bytes:
    """ONNX ModelProto.metadata_props entries (field 14, each a StringStringEntryProto: key = 1, value = 2) as protobuf
    bytes, to append to a model file."""
    return b"".join(_field(14, _field(1, str(k).encode()) + _field(2, str(v).encode())) for k, v in entries.items())


def _field(number: int, payload: bytes) -> bytes:  # a length-delimited protobuf field
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


class PiperVoice:
    """A prepared voice, loaded (~1 s): synthesize(text, speed) gives float32 samples at `rate`. Used from one thread."""

    def __init__(self, voice: Voice, root: Path | None = None, threads: int = 2):
        import sherpa_onnx  # its native library: only when a voice is wanted
        folder = voice.folder(root)
        config = json.loads((folder / (voice.model_file + ".json")).read_text(encoding="utf-8"))
        inference = config.get("inference", {})
        self.voice, self.rate = voice, int(config["audio"]["sample_rate"])
        self._tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(folder / PREPARED), lexicon="", tokens=str(folder / TOKENS), data_dir=str(voice.data.path(root)),
                noise_scale=inference.get("noise_scale", 0.667), noise_scale_w=inference.get("noise_w", 0.8),
                length_scale=inference.get("length_scale", 1.0)),
            provider="cpu", num_threads=threads)))

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        return np.asarray(self._tts.generate(text, sid=0, speed=speed).samples, dtype=np.float32)
