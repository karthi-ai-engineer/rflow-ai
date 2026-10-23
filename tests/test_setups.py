"""The ready-made setups (sst.setups): what each uses and costs, which one the settings match, and using one."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from sst import costs, setups  # noqa: E402
from sst.gateway import GatewayConfig  # noqa: E402
from sst.settings import Settings  # noqa: E402
from sst.window import PreviewApp  # noqa: E402

S = setups.SETUPS


def test_the_setups_use_the_models_of_the_research_notes():
    assert setups.ORDER == ["recommended", "multilingual", "local", "custom"]  # Recommended first; no Fastest (Groq)
    assert (S["recommended"].speech, S["recommended"].provider, S["recommended"].model) == \
        ("parakeet", "gemini", "gemini-3.5-flash-lite")
    assert (S["multilingual"].speech_model, S["multilingual"].model) == ("gemini-3.5-transcribe", "gemini-3.5-flash-lite")
    assert S["local"].provider == "" and S["local"].needs == ""  # AI cleanup off until a model on this PC comes
    assert {S[k].needs for k in ("recommended", "multilingual")} == {"gemini"}


@pytest.mark.parametrize("key, amount, tier", [("recommended", 1.05, costs.LOW), ("multilingual", 3.57, costs.AMBER)])
def test_each_setup_costs_about_what_the_notes_say(key, amount, tier):
    estimate = setups.cost(S[key])
    assert abs(estimate.monthly - amount) < 0.03 and estimate.tier == tier


def test_local_is_free_and_multilingual_hears_tamil_with_flash_lite():
    assert setups.cost(S["local"]).tier == costs.FREE and setups.cost(S["local"], local="whisper-turbo").tier == costs.FREE
    assert setups.speech_model_for(S["multilingual"], "ta") == "gemini-3.5-flash-lite"  # Transcribe has no Tamil
    assert setups.speech_model_for(S["multilingual"], "ja") == "gemini-3.5-transcribe"
    assert abs(setups.cost(S["multilingual"], "ta").monthly - 1.90) < 0.05  # the notes: about $1.90 with Tamil


def test_the_setup_in_use_is_found_from_the_settings():
    gemini = GatewayConfig(provider="gemini", api_key="AIza-key")
    on = {"cleanup": True, "cleanup_model": "gemini-3.5-flash-lite"}
    assert setups.current(Settings(**on), gemini) == "recommended"
    assert setups.current(Settings(speech_model="gemini", **on), gemini) == "multilingual"  # Transcribe, the default
    assert setups.current(Settings(speech_model="gemini", speech_cloud_models={"gemini": "gemini-3.6-flash"}, **on),
                          gemini) == "custom"
    assert setups.current(Settings(speech_model="gemini", speech_language="ta",
                                   speech_cloud_models={"gemini": "gemini-3.5-flash-lite"}, **on), gemini) == "multilingual"
    groq = GatewayConfig(provider="groq", api_key="gsk")  # the old Fastest: Groq isn't offered, so no setup matches
    assert setups.current(Settings(speech_model="groq", cleanup=True, cleanup_model="openai/gpt-oss-20b"), groq) == "custom"
    assert setups.current(Settings(), GatewayConfig()) == "local"  # on this PC, no AI cleanup
    assert setups.current(Settings(speech_model="whisper-turbo", cleanup_model="gpt-4o-mini"), gemini) == "local"  # off
    assert setups.current(Settings(cleanup=True, cleanup_model="gpt-4o-mini"), GatewayConfig(provider="openai")) == "custom"
    assert setups.current(Settings(speech_model="groq", **on), gemini, pending="parakeet") == "recommended"  # arriving


def test_using_a_setup_sets_both_parts_and_keeps_the_other_keys():
    gateway = GatewayConfig("", "sk-openai", "openai").with_key("gemini", "AIza-key").with_key("groq", "gsk-key")
    app = PreviewApp(settings=Settings(welcomed=True, cleanup=True, cleanup_model="gpt-4o-mini",
                                       cleanup_fallback="gpt-4.1-mini"), gateway=gateway)
    assert setups.apply(app, "recommended") == ""  # Parakeet is here: in use at once
    assert ("choose_speech_model", "parakeet") in app.calls
    assert (app.gateway.chosen, app.settings.cleanup, app.settings.cleanup_model) == ("gemini", True,
                                                                                       "gemini-3.5-flash-lite")
    assert app.settings.cleanup_fallback == ""  # OpenAI's backup model can't be Gemini's
    assert app.gateway.key_for("openai") == "sk-openai" and app.gateway.key_for("gemini") == "AIza-key"
    assert setups.current(app.settings, app.gateway) == "recommended"
    assert app.gateway.key_for("groq") == "gsk-key"  # the keys of providers no longer offered stay saved
    app.settings.speech_language = "ta"
    setups.apply(app, "multilingual")
    assert app.settings.speech_cloud_models["gemini"] == "gemini-3.5-flash-lite"  # Tamil
    assert setups.current(app.settings, app.gateway) == "multilingual"
    setups.apply(app, "local")
    assert app.settings.speech_model == "parakeet" and not app.settings.cleanup
    assert app.settings.cleanup_model == "gemini-3.5-flash-lite"  # kept for when it's switched on again
    assert setups.current(app.settings, app.gateway) == "local"


def test_a_setup_downloads_parakeet_first_when_it_isnt_here(no_parakeet):
    app = PreviewApp(settings=Settings(welcomed=True), gateway=GatewayConfig().with_key("gemini", "AIza-key"))
    assert setups.apply(app, "recommended") == "parakeet"
    assert ("download_speech_model", "parakeet") in app.calls and app.settings.cleanup_model == "gemini-3.5-flash-lite"
    app.downloading = ("parakeet", 1, 2)
    calls = len(app.calls)
    setups.apply(app, "recommended")
    assert ("download_speech_model", "parakeet") not in app.calls[calls:]  # one download, not two
    setups.apply(app, "local", local="whisper-turbo")
    assert ("cancel_download",) in app.calls  # Parakeet would have been used once it was here: not any more


def test_a_missing_key_is_named():
    assert setups.key_missing(S["recommended"], GatewayConfig()) == "gemini"
    assert setups.key_missing(S["recommended"], GatewayConfig().with_key("gemini", "AIza")) == ""
    assert setups.key_missing(S["local"], GatewayConfig()) == ""
