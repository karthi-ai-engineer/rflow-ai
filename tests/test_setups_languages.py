"""Recommended chooses by the languages spoken (sst.setups.resolve): Parakeet for English, else the fastest offered
cloud model that hears every language well (Gemini, while it is the only cloud provider offered); a setup in use stays in
use (off-screen, PreviewApp)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst import setups  # noqa: E402
from sst import window as w  # noqa: E402
from sst.gateway import GatewayConfig  # noqa: E402
from sst.settings import Settings  # noqa: E402

NO_KEYS, GEMINI_KEY = GatewayConfig(), GatewayConfig().with_key("gemini", "AIza-key")
BOTH_KEYS = GEMINI_KEY.with_key("groq", "gsk-key")


@pytest.fixture(scope="module", autouse=True)
def qt():
    return QApplication.instance() or QApplication([])


def _picks(key, languages, gateway=NO_KEYS, scan=None) -> tuple[str, str, str, str]:
    """(speech, its cloud model, AI provider, AI model) a setup chooses."""
    s = setups.resolve(key, languages, gateway, scan)
    return s.speech, setups.speech_model_for(s, languages) if s.speech_model else "", s.provider, s.model


PARAKEET_GEMINI = ("parakeet", "", "gemini", "gemini-3.5-flash-lite")
TRANSCRIBE = ("gemini", "gemini-3.5-transcribe", "gemini", "gemini-3.5-flash-lite")
FLASH_LITE = ("gemini", "gemini-3.5-flash-lite", "gemini", "gemini-3.5-flash-lite")


@pytest.mark.parametrize("languages, recommended, multilingual", [
    ("en", PARAKEET_GEMINI, TRANSCRIBE),
    ("", PARAKEET_GEMINI, TRANSCRIBE),  # not chosen: English, Rflow's default
    ("ja", TRANSCRIBE, TRANSCRIBE),  # Gemini, not Groq's Whisper: Groq isn't offered
    ("ta", FLASH_LITE, FLASH_LITE),  # Transcribe has no Tamil
    (("en", "ja"), TRANSCRIBE, TRANSCRIBE),
    (("en", "ta"), FLASH_LITE, FLASH_LITE),
])
def test_each_setup_picks_by_the_languages_spoken(languages, recommended, multilingual):
    assert _picks("recommended", languages) == recommended
    assert _picks("recommended", languages, BOTH_KEYS) == recommended  # a Groq key saved changes nothing
    assert _picks("multilingual", languages) == multilingual
    assert setups.resolve("local", languages) == setups.SETUPS["local"]  # Local and Custom as they are
    assert setups.resolve("custom", languages) == setups.SETUPS["custom"]


def test_only_gemini_and_this_pc_are_chosen_for_any_language():
    assert setups.CLOUD_SPEED == (setups.GEMINI,) and "fastest" not in setups.SETUPS
    for languages in ("en", "ja", "ta", "hi", ("en", "ja", "ta")):
        for key in setups.ORDER:
            chosen = setups.resolve(key, languages, BOTH_KEYS)
            assert chosen.speech in ("", "parakeet", "gemini") and chosen.provider in ("", "gemini")
            assert chosen.needs in ("", "gemini")
    assert setups.resolve("recommended", "ja").needs == "gemini"  # no key: Gemini's is asked for


def test_the_texts_say_what_was_chosen_and_how_quick_it_is(no_parakeet):
    english = setups.resolve("recommended", "en")
    assert english.name == "Recommended" and "one-time download of 663 MB" in english.blurb
    japanese = setups.resolve("recommended", "ja")
    assert japanese.blurb == "Japanese: Parakeet hears English only, so Gemini hears it in about 2 seconds, and its AI " \
                             "polishes it."
    assert "Groq" not in setups.resolve("recommended", "ta").blurb
    assert setups.uses_line(english) == "Parakeet on this PC + Gemini 3.5 Flash-Lite"
    assert setups.uses_line(japanese, "ja") == "Gemini 3.5 Transcribe + Gemini 3.5 Flash-Lite"
    assert setups.uses_line(setups.resolve("recommended", "ta"), "ta") == "Gemini 3.5 Flash-Lite hears you and polishes " \
                                                                            "the text"
    assert setups.uses_line(setups.SETUPS["multilingual"]) == "Gemini 3.5 Transcribe + Gemini 3.5 Flash-Lite"


def test_a_setup_in_use_stays_in_use_whatever_it_would_choose_now():
    on = {"cleanup": True, "cleanup_model": "gemini-3.5-flash-lite"}
    gemini = GatewayConfig(provider="gemini", api_key="AIza-key")
    # Recommended chosen before the language was Japanese: still Recommended (its own models), nothing changed
    assert setups.in_use(Settings(speech_language="ja", **on), gemini) == ["recommended"]
    # The same models as two setups: both are in use, a setup's own models first
    tamil = Settings(speech_model="gemini", speech_language="ta", speech_cloud_models={"gemini": "gemini-3.5-flash-lite"},
                     **on)
    assert setups.in_use(tamil, gemini) == ["multilingual", "recommended"]
    assert setups.current(tamil, gemini) == "multilingual"
    japanese = Settings(speech_model="gemini", speech_language="ja", **on)
    assert setups.in_use(japanese, gemini) == ["multilingual", "recommended"]
    # The old Fastest on Groq: no setup offered uses it
    groq = GatewayConfig(provider="groq", api_key="gsk")
    assert setups.in_use(Settings(speech_model="groq", cleanup=True, cleanup_model="openai/gpt-oss-20b"), groq) == \
        ["custom"]


def test_applying_a_setup_uses_what_it_chooses_for_the_language(no_parakeet):
    app = w.PreviewApp(settings=Settings(welcomed=True, speech_language="ja"), gateway=GEMINI_KEY)
    assert setups.apply(app, "recommended") == ""  # no download: Japanese goes to the cloud
    assert ("use_cloud_speech", "gemini", "gemini-3.5-transcribe") in app.calls
    assert (app.gateway.chosen, app.settings.cleanup_model) == ("gemini", "gemini-3.5-flash-lite")
    assert "recommended" in setups.in_use(app.settings, app.gateway)
    app = w.PreviewApp(settings=Settings(welcomed=True, speech_language="ta"), gateway=BOTH_KEYS)
    setups.apply(app, "recommended")
    assert ("use_cloud_speech", "gemini", "gemini-3.5-flash-lite") in app.calls  # Tamil: Flash-Lite, as before
    assert app.gateway.chosen == "gemini" and app.gateway.key_for("groq") == "gsk-key"  # the Groq key stays saved


def test_the_window_shows_what_each_setup_chooses_for_the_language():
    gateway = GatewayConfig("", "AIza-key", "gemini")
    app = w.PreviewApp(settings=Settings(welcomed=True, speech_language="ta", speech_model="gemini", cleanup=True,
                                         cleanup_model="gemini-3.5-flash-lite",
                                         speech_cloud_models={"gemini": "gemini-3.5-flash-lite"}), gateway=gateway)
    window = w.MainWindow(app)
    window.show_page("models")
    card = window.pages["models"].setup_card
    tiles = card.tiles
    assert list(tiles) == ["recommended", "multilingual", "local", "custom"]
    assert tiles["recommended"].speech.text() == "Gemini 3.5 Flash-Lite, on Gemini"  # Tamil: not Parakeet
    assert tiles["recommended"].needs.text() == "Gemini key: saved"
    assert [k for k, t in tiles.items() if t.state.text() == "In use"] == ["recommended", "multilingual"]
    card.select("recommended")
    assert "Tamil: Parakeet hears English only" in card.blurb.text() and card.use.isHidden()  # in use already
    welcome = w.MainWindow(w.PreviewApp(settings=Settings(speech_language="ja"))).pages["welcome"]
    assert welcome.option_uses["recommended"].text() == "Gemini 3.5 Transcribe + Gemini 3.5 Flash-Lite"
    assert welcome.foot_note.text() == "Next: your Gemini key."  # no Parakeet download for Japanese
