"""The wake word's models: their pins, the folder check before sherpa-onnx sees it, the keyword spotter (sherpa-onnx
faked: no model needed) and the voice print (its model faked)."""
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from sst import downloads
from sst.downloads import Download, ModelFile
from sst.wake import KWS, VOICE, Spotter, VoicePrint, models, spotter, voiceprint

FIXTURE = Path(__file__).parent / "fixtures" / "wake"


# ---- the downloads

@pytest.mark.parametrize("model", [KWS, VOICE])
def test_each_model_is_pinned_and_checked(model):
    # From their makers: k2-fsa's own release (an archive), or the sherpa-onnx author's Hugging Face repository
    assert downloads.allowed(model.base_url)
    assert (model.base_url.startswith(downloads.K2_RELEASES) if model.archive else
            re.match(r"https://huggingface\.co/csukuangfj/[^/]+/resolve/[0-9a-f]{40}/$", model.base_url))
    pinned = (*model.files, model.archive) if model.archive else model.files
    assert all(f.size > 0 and re.fullmatch(r"[0-9a-f]{64}", f.sha256) for f in pinned)


def test_the_spotter_is_small_and_the_voice_model_is_titanet_small():
    assert {f.name for f in KWS.files} == {models.ENCODER, models.DECODER, models.JOINER, models.TOKENS, models.BPE}
    assert sum(f.size for f in KWS.files) < 7_000_000 and KWS.size < 18_000_000  # kept, and downloaded
    assert VOICE.files[0].name == "nemo_en_titanet_small.onnx"


def test_install_downloads_the_spotter_then_the_voice_model(monkeypatch, tmp_path):
    fetched, seen = [], []

    def fake_download(model, progress, cancelled, root):
        fetched.append((model.folder, root))
        progress(model.size, model.size)

    monkeypatch.setattr(models, "download", fake_download)
    models.install(lambda done, total: seen.append((done, total)), root=tmp_path)
    assert fetched == [(KWS.folder, tmp_path), (VOICE.folder, tmp_path)]
    assert seen == [(KWS.size, KWS.size + VOICE.size), (KWS.size + VOICE.size, KWS.size + VOICE.size)]
    fetched.clear()
    models.install(root=tmp_path, voice=False)  # no voice check: the spotter only
    assert fetched == [(KWS.folder, tmp_path)]
    assert not models.installed(tmp_path)


def test_a_folder_is_checked_before_sherpa_onnx_sees_it(tmp_path):
    model = Download("m", "https://huggingface.co/x/resolve/0/", (ModelFile("a.onnx", 3, "x"), ModelFile("b.txt", 2, "x")))
    (tmp_path / "a.onnx").write_bytes(b"abc")
    with pytest.raises(FileNotFoundError, match="b.txt"):
        models.check(model, tmp_path)
    (tmp_path / "b.txt").write_bytes(b"too long")  # damaged: the wrong size
    with pytest.raises(FileNotFoundError, match="b.txt"):
        models.check(model, tmp_path)
    (tmp_path / "b.txt").write_bytes(b"ok")
    assert models.check(model, tmp_path) == tmp_path


# ---- the keyword spotter, with sherpa-onnx faked

class FakeStream:
    def __init__(self, keywords):
        self.keywords, self.chunks, self.result, self.resets = keywords, [], "", 0

    def accept_waveform(self, rate, samples):
        assert rate == 16000 and samples.dtype == np.float32 and samples.ndim == 1
        self.chunks.append(samples)


class FakeKeywordSpotter:
    """A chunk whose loudest sample is 0.9 is heard as the stream's first phrase; 0.7 as the placeholder."""
    made: list["FakeKeywordSpotter"] = []

    def __init__(self, **config):
        self.config = config
        self.keywords_file = Path(config["keywords_file"]).read_text(encoding="utf-8")
        self.streams = []
        FakeKeywordSpotter.made.append(self)

    def create_stream(self, keywords=""):
        self.streams.append(FakeStream(keywords))
        return self.streams[-1]

    def is_ready(self, stream):
        return bool(stream.chunks)

    def decode_stream(self, stream):
        loudest = float(np.abs(stream.chunks.pop(0)).max())
        stream.result = (stream.keywords.split("@")[1].split("/")[0] if loudest > 0.85
                         else "placeholder" if 0.65 < loudest < 0.75 else "")

    def get_result(self, stream):
        return stream.result

    def reset_stream(self, stream):
        stream.resets += 1
        stream.result = ""


@pytest.fixture
def fake_sherpa(monkeypatch):
    """The downloaded model's folder (the tokenizer fixture, stand-in onnx files) passing the check; sherpa-onnx faked."""
    folder = KWS.path()
    folder.mkdir(parents=True)
    files = []
    for f in KWS.files:
        source = FIXTURE / f.name
        data = source.read_bytes() if source.exists() else b"onnx"
        (folder / f.name).write_bytes(data)
        files.append(ModelFile(f.name, len(data), "x"))
    monkeypatch.setattr(models, "KWS", Download(KWS.folder, KWS.base_url, tuple(files)))
    monkeypatch.setitem(sys.modules, "sherpa_onnx", types.SimpleNamespace(KeywordSpotter=FakeKeywordSpotter))
    FakeKeywordSpotter.made = []
    spotter.unload()
    yield folder
    spotter.unload()


def frame(peak=0.1):
    out = np.zeros(1600, np.float32)
    out[0] = peak
    return out


def test_phrases_are_tokenized_into_the_streams_own_keywords(fake_sherpa):
    found = Spotter({"rflow": "our flow", "stop it": "stop it"}, model_dir=fake_sherpa)
    stream = FakeKeywordSpotter.made[0].streams[-1]
    assert stream.keywords == "▁OUR ▁F LO W :1.0 #0.25 @k0/▁ST O P ▁IT :1.0 #0.25 @k1"
    assert found.feed(frame()) is None
    assert found.feed(frame(0.9)) == "rflow" and stream.resets == 1  # reset right after a hit, as sherpa-onnx wants
    assert found.feed(frame(0.7)) is None  # the placeholder is never anyone's phrase


def test_the_model_is_loaded_once_and_shared(fake_sherpa):
    Spotter({"a": "our flow"}, model_dir=fake_sherpa)
    other = Spotter({"b": "stop it"}, threshold=0.3)  # the downloaded folder by default
    assert len(FakeKeywordSpotter.made) == 1
    kws = FakeKeywordSpotter.made[0]
    assert kws.config["keywords_file"] and "#0.99" in kws.keywords_file and ":0.0" in kws.keywords_file
    assert kws.config["num_threads"] == 1 and kws.config["keywords_threshold"] == 0.25
    assert kws.config["encoder"] == str(fake_sherpa / models.ENCODER)
    assert kws.streams[-1].keywords.endswith("#0.3 @k0")
    before = len(kws.streams)
    other.reset()
    assert len(kws.streams) == before + 1  # a new stream: nothing half-heard carries over
    spotter.unload()
    Spotter({"a": "our flow"})
    assert len(FakeKeywordSpotter.made) == 2


def test_frames_of_any_shape_are_fed_as_float32_mono(fake_sherpa):
    found = Spotter({"a": "our flow"})
    assert found.feed(np.zeros((1600, 1), np.float64)) is None
    assert found.feed(np.full(800, 0.9, np.float32)) == "a"


def test_a_missing_model_never_reaches_sherpa_onnx(fake_sherpa):
    (fake_sherpa / models.JOINER).unlink()
    with pytest.raises(FileNotFoundError):
        Spotter({"a": "our flow"})
    assert FakeKeywordSpotter.made == []


def test_a_phrase_the_model_has_no_pieces_for_is_refused(fake_sherpa):
    with pytest.raises(ValueError):
        Spotter({"a": "q"})  # the fixture's tokens.txt has no "Q": sherpa-onnx would end the process
    with pytest.raises(ValueError):
        Spotter({"a": ""})
    with pytest.raises(ValueError):
        Spotter({})


# ---- the voice print, with its model faked

@pytest.fixture
def fake_embed(monkeypatch):
    """Each recording's "voice" is the vector in its first 3 samples."""
    def embed(audio, model_dir=None):
        vector = np.asarray(audio[:3], dtype=np.float32)
        return vector / np.linalg.norm(vector) if np.linalg.norm(vector) else None

    monkeypatch.setattr(voiceprint, "embed", embed)


def test_the_voice_print_is_the_mean_of_the_tries(fake_embed):
    print_ = VoicePrint.from_takes([np.array([1, 0, 0, 5], np.float32), np.array([0, 2, 0], np.float32),
                                    np.zeros(3, np.float32)])  # silence: nothing to learn from it
    assert np.allclose(print_.vector, [2 ** -0.5, 2 ** -0.5, 0])
    assert print_.score(np.array([1, 1, 0], np.float32)) == pytest.approx(1.0)
    assert print_.score(np.array([0, 0, 1], np.float32)) == pytest.approx(0.0)
    assert print_.score(np.zeros(3, np.float32)) == 0.0  # too short to tell: not the user
    with pytest.raises(ValueError):
        VoicePrint.from_takes([np.zeros(3, np.float32)])


def test_the_voice_print_survives_json():
    print_ = VoicePrint(np.arange(1, 193, dtype=np.float32))
    back = VoicePrint.from_json(json.loads(json.dumps(print_.to_json())))
    assert np.allclose(back.vector, print_.vector, atol=1e-6) and len(back.vector) == 192
    with pytest.raises(ValueError):
        VoicePrint.from_json({"model": "another", "vector": [1.0]})
    with pytest.raises(ValueError):
        VoicePrint([0.0, 0.0])


def test_silence_around_speech_is_trimmed():
    audio = np.concatenate([np.zeros(1600), np.full(3200, 0.5), np.full(1600, 1e-4)]).astype(np.float32)
    assert len(voiceprint.trim(audio)) == 3200
    assert len(voiceprint.trim(np.zeros(100, np.float32))) == 100  # shorter than 10 ms: as it is


def test_a_missing_voice_model_is_an_error_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)  # importing it would fail: it's never reached
    voiceprint.unload()
    with pytest.raises(FileNotFoundError):
        voiceprint.embed(np.zeros(16000, np.float32), tmp_path)
