"""The speech model catalog: what the Speech recognition page lists, and which model a setting loads."""
import json

from sst import engines, evaluate
from sst.settings import Settings


def test_every_model_says_where_it_runs_and_what_it_is():
    for key, model in engines.SPEECH_MODELS.items():
        assert model.key == key and model.where in engines.WHERE
        assert model.name and model.summary and model.languages and model.size


def test_parakeet_is_the_default_and_the_models_ready_now_can_be_loaded():
    assert engines.DEFAULT_MODEL == "parakeet" and engines.SPEECH_MODELS["parakeet"].ready
    assert engines.ENGINES == [k for k, m in engines.SPEECH_MODELS.items() if m.ready]
    assert engines.SPEECH_MODELS["whisper-turbo"].download is not None  # downloaded when chosen


def test_a_setting_this_version_cannot_use_falls_back_to_the_default():
    assert engines.usable("parakeet") == "parakeet"
    assert engines.usable("whisper-turbo") == "parakeet"  # chosen but not downloaded (removed, or another laptop)
    assert engines.usable("from-a-newer-version") == "parakeet"


def test_a_downloaded_model_is_used(whisper_downloaded):
    assert engines.usable("whisper-turbo") == "whisper-turbo"


def test_the_chosen_model_is_a_profile_setting_and_old_files_get_parakeet(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"hotkey": "menu"}), encoding="utf-8")  # from before the choice existed
    assert Settings.load(path).speech_model == "parakeet"
    Settings(speech_model="whisper-turbo").save(path)
    assert Settings.load(path).speech_model == "whisper-turbo"


def test_reports_name_the_model_in_use():
    names = [p.name for p in evaluate.pipelines_for({"qwen": object()}, ["narrowband"], title="Whisper")]
    assert names == ["Whisper alone", "Whisper, narrowband", "Whisper + qwen"]
    assert evaluate.pipelines_for({})[0].name == "Parakeet alone"  # the default, as before


# ---- Parakeet downloaded on demand

def test_parakeet_is_downloaded_from_a_pinned_revision_with_every_file_checked():
    from sst import downloads
    from sst.engines import parakeet
    model = parakeet.MODEL
    assert parakeet.REVISION in model.base_url and downloads.allowed(model.base_url)
    assert {"encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"} <= {f.name for f in model.files}
    assert all(len(f.sha256) == 64 for f in model.files) and 650e6 < model.size < 680e6
    assert model.folder == parakeet.FOLDER  # the folder Rflow 1.4 used next to the program
    assert engines.SPEECH_MODELS["parakeet"].download is model


def test_parakeet_is_found_next_to_an_older_rflow_or_downloaded(no_parakeet, monkeypatch, tmp_path):
    from sst import downloads
    from sst.engines import parakeet
    assert parakeet.find_model() is None and not engines.SPEECH_MODELS["parakeet"].installed()
    monkeypatch.setattr(downloads.Download, "installed", lambda self, root=None: True)
    assert parakeet.find_model() == parakeet.MODEL.path()  # downloaded
    old = tmp_path / parakeet.FOLDER
    old.mkdir()
    (old / "encoder.int8.onnx").write_bytes(b"")
    monkeypatch.setattr(parakeet, "MODEL_DIR", old)
    assert parakeet.find_model() == old  # next to the program, as Rflow 1.4 installed it: no download needed


def test_with_no_speech_model_yet_nothing_is_loaded(no_parakeet):
    from sst.gateway import GatewayConfig
    assert engines.usable("parakeet") == "" and engines.usable("whisper-turbo") == ""
    assert engines.usable("gemini", GatewayConfig().with_key("gemini", "AIza")) == "gemini"  # a cloud model: no download
    # Groq is no longer offered: Gemini stands in with its key, else Parakeet (not here: none yet)
    assert engines.usable("groq", GatewayConfig().with_key("groq", "gsk").with_key("gemini", "AIza")) == "gemini"
    assert engines.usable("groq", GatewayConfig().with_key("groq", "gsk")) == ""
    assert engines.usable("openai", GatewayConfig().with_key("openai", "sk")) == ""


def test_parakeet_without_its_model_says_how_to_get_it(no_parakeet):
    import pytest

    from sst.engines.parakeet import ParakeetEngine
    with pytest.raises(FileNotFoundError, match="isn't downloaded"):
        ParakeetEngine()


def test_the_hotword_vocabulary_and_the_sample_sentence_come_with_rflow():
    from sst.app import SAMPLE_FILE
    from sst.audio import load_wav
    from sst.engines import parakeet
    vocab = parakeet.VOCAB.read_bytes()
    assert b"\r" not in vocab and len(vocab.splitlines()) == 1024  # byte for byte NVIDIA's (sherpa-onnx parses it)
    audio, rate = load_wav(SAMPLE_FILE)
    assert rate == 16_000 and 7 < len(audio) / rate < 8
