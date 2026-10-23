"""The wake word's two models, downloaded when the user turns the wake word on (sst.downloads: pinned, checked), from
their makers, never bundled:

    KWS    k2-fsa's keyword spotter sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 (Apache-2.0), from k2-fsa's own
           GitHub release, the only place they publish it: the archive (17.6 MB) is pinned, and five of its files are
           kept (6.3 MB): the int8 encoder and joiner, the float decoder, tokens.txt and bpe.model.
    VOICE  NVIDIA's TitaNet-small speaker model as sherpa-onnx's ONNX export (NeMo toolkit licence, Apache-2.0; 40 MB),
           from the copy of sherpa-onnx's author (the same file as k2-fsa's release): the voice check.

sherpa-onnx's C++ checks end the whole process on a missing or damaged file, so a folder is checked (every file there,
at its size) before sherpa-onnx is given it: check().
"""
from collections.abc import Callable
from pathlib import Path

from sst.downloads import Download, ModelFile, download

KWS_ARCHIVE = ModelFile("sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2", 17626723,
                        "f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a")
KWS = Download(
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/", (
        ModelFile("encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx", 4807159,
                  "1e721676515bcd42a186979733981213c66c80db680e1cc582dfedf3be76e678"),
        ModelFile("decoder-epoch-12-avg-2-chunk-16-left-64.onnx", 1063189,
                  "f61ebd3eed3773a44d088d53dfae92dbb6aec4839f4dcaee2d402414741663a3"),
        ModelFile("joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx", 163380,
                  "eae9da0c7e1e6c6a3f4cc42d167899c388f6c6701b94cb96320e4f55df79624c"),
        ModelFile("tokens.txt", 5006, "fd2ded4050a55d2b1578870ba8697d02371980217806b7558bd0a5cc60f3ba53"),
        ModelFile("bpe.model", 244837, "c8a2a0129c4ab8e463164c142f82d25649661b122c8cd0b7aab5c9e80b90ad24"),
    ), KWS_ARCHIVE)
ENCODER, DECODER, JOINER, TOKENS, BPE = (f.name for f in KWS.files)

VOICE_REVISION = "0743f301363dec56491a490f6d6cbc9d67f9a3bf"
VOICE = Download(
    "nemo-en-titanet-small",
    f"https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/{VOICE_REVISION}/", (
        ModelFile("nemo_en_titanet_small.onnx", 40257283, "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"),
    ))
VOICE_MODEL = VOICE.files[0].name


def check(model: Download, folder: Path) -> Path:
    """`folder` if it holds every file of `model` at its size; else FileNotFoundError (sherpa-onnx would end the
    process instead)."""
    bad = [f.name for f in model.files if not (folder / f.name).is_file() or (folder / f.name).stat().st_size != f.size]
    if bad:
        raise FileNotFoundError(f"The wake word's model isn't downloaded (or is damaged): {', '.join(bad)} in {folder}")
    return folder


def installed(root: Path | None = None, voice: bool = True) -> bool:
    """Both models downloaded and checked (the voice model only when the voice check is wanted)."""
    return KWS.installed(root) and (not voice or VOICE.installed(root))


def size(voice: bool = True) -> int:
    return KWS.size + (VOICE.size if voice else 0)


def install(progress: Callable[[int, int], None] = lambda done, total: None, cancelled: Callable[[], bool] = lambda: False,
            root: Path | None = None, voice: bool = True) -> None:
    """Download what's missing (the spotter first: it's small). Raises sst.downloads' Cancelled or DownloadError."""
    total = size(voice)
    download(KWS, lambda done, _: progress(done, total), cancelled, root)
    if voice:
        download(VOICE, lambda done, _: progress(KWS.size + done, total), cancelled, root)
