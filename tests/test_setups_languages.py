"""Recommended and Fastest choose by the languages spoken (sst.setups.resolve): Parakeet for English, else the fastest
cloud model that hears every language well; a setup in use stays in use (off-screen, PreviewApp)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst import setups  # noqa: E402
from sst import window as w  # noqa: E402
from sst.gateway import GatewayConfig  # noqa: E402
from sst.settings import Settings  # noqa: E402

QUICK = {"verdicts": [{"key": "parakeet", "level": "recommended", "seconds": 0.9, "reason": "about 0.9 s"}]}
SLOW = {"verdicts": [{"key": "parakeet", "level": "usable", "seconds": 3.2, "reason": "about 3.2 s"}]}
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
PARAKEET_GROQ = ("parakeet", "", "groq", "openai/gpt-oss-20b")
GROQ = ("groq", "whisper-large-v3-turbo", "groq", "openai/gpt-oss-20b")
TRANSCRIBE = ("gemini", "gemini-3.5-transcribe", "gemini", "gemini-3.5-flash-lite")
FLASH_LITE = ("gemini", "gemini-3.5-flash-lite", "gemini", "gemini-3.5-flash-lite")


@pytest.mark.parametrize("languages, recommended, fastest, multilingual", [
    ("en", PARAKEET_GEMINI, GROQ, TRANSCRIBE),  # Fastest: Groq until Check this PC finds Parakeet quick
    ("", PARAKEET_GEMINI, GROQ, TRANSCRIBE),  # not chosen: English, Rflow's default
    ("ja", GROQ, GROQ, TRANSCRIBE),
    ("ta", FLASH_LITE, FLASH_LITE, FLASH_LITE),  # Groq's Whisper is weak at Tamil; Transcribe has none
    (("en", "ja"), GROQ, GROQ, TRANSCRIBE),
    (("en", "ta"), FLASH_LITE, FLASH_LITE, FLASH_LITE),
])
def test_each_setup_picks_by_the_languages_spoken(languages, recommended, fastest, multilingual):
    assert _picks("recommended", languages) == recommended
    assert _picks("fastest", languages) == fastest
    assert _picks("multilingual", languages) == multilingual
    assert setups.resolve("local", languages) == setups.SETUPS["local"]  # Local and Custom as they are
    assert setups.resolve("custom", languages) == setups.SETUPS["custom"]


def test_fastest_uses_parakeet_for_english_only_where_this_pc_runs_it_quickly():
    assert _picks("fastest", "en", scan=QUICK) == PARAKEET_GROQ
    assert _picks("fastest", "en", scan=SLOW) == GROQ  # a weak processor: Groq's half a second
    assert _picks("fastest", "ja", scan=QUICK) == GROQ  # Parakeet hears English only
    assert setups.resolve("fastest", "en", scan=QUICK).needs == "groq"
    assert "under a second" in setups.resolve("fastest", "en", scan=QUICK).blurb


def test_recommended_takes_the_fastest_cloud_model_whose_key_the_user_has():
    assert _picks("recommended", "ja", GEMINI_KEY) == TRANSCRIBE  # no Groq key: Gemini, on the key there is
    assert _picks("recommended", "ja", BOTH_KEYS) == GROQ
    assert setups.resolve("recommended", "ja").needs == "groq"  # no key: the fastest's is asked for
    assert _picks("fastest", "ja", GEMINI_KEY) == GROQ  # Fastest is always the fastest
    assert _picks("recommended", "en", BOTH_KEYS) == PARAKEET_GEMINI  # English: on this PC whatever the keys


def test_the_texts_say_what_was_chosen_and_how_quick_it_is(no_parakeet):
    english = setups.resolve("recommended", "en")
    assert english.name == "Recommended" and "one-time download of 663 MB" in english.blurb
    japanese = setups.resolve("recommended", "ja")
    assert japanese.blurb == "Japanese: Parakeet hears English only, so Groq hears it in about half a second, and its AI " \
                             "polishes it."
    assert "Gemini hears it in about 2 seconds (Groq's Whisper is weak at Tamil)" in setups.resolve("fastest", "ta").blurb
    assert setups.resolve("fastest", "en").blurb == setups.SETUPS["fastest"].blurb
    assert setups.uses_line(english) == "Parakeet on this PC + Gemini 3.5 Flash-Lite"
    assert setups.uses_line(japanese, "ja") == "Whisper turbo + GPT-OSS 20B, both on Groq"
    assert setups.uses_line(setups.resolve("recommended", "ta"), "ta") == "Gemini 3.5 Flash-Lite hears you and polishes " \
                                                                            "the text"
    assert setups.uses_line(setups.SETUPS["multilingual"]) == "Gemini 3.5 Transcribe + Gemini 3.5 Flash-Lite"


def test_a_setup_in_use_stays_in_use_whatever_it_would_choose_now():
    on = {"cleanup": True, "cleanup_model": "gemini-3.5-flash-lite"}
    gemini = GatewayConfig(provider="gemini", api_key="AIza-key")
    # Recommended chosen before the language was Japanese: still Recommended (its own models), nothing changed
    assert setups.in_use(Settings(speech_language="ja", **on), gemini) == ["recommended"]
    # Fastest on Groq, then Check this PC found Parakeet quick: Groq is still one of Fastest's
    groq = GatewayConfig(provider="groq", api_key="gsk")
    fastest = Settings(speech_model="groq", cleanup=True, cleanup_model="openai/gpt-oss-20b")
    assert setups.in_use(fastest, groq) == ["fastest"]
    assert setups.in_use(Settings(cleanup=True, cleanup_model="openai/gpt-oss-20b"), groq) == ["fastest"]  # Parakeet
    # The same models as two setups: both are in use, a setup's own models first
    tamil = Settings(speech_model="gemini", speech_language="ta", speech_cloud_models={"gemini": "gemini-3.5-flash-lite"},
                     **on)
    assert setups.in_use(tamil, gemini) == ["multilingual", "recommended", "fastest"]
    assert setups.current(tamil, gemini) == "multilingual"
    japanese = Settings(speech_model="groq", speech_language="ja", cleanup=True, cleanup_model="openai/gpt-oss-20b")
    assert setups.in_use(japanese, groq) == ["fastest", "recommended"]


def test_applying_a_setup_uses_what_it_chooses_for_the_language(no_parakeet):
    app = w.PreviewApp(settings=Settings(welcomed=True, speech_language="ja"), gateway=GEMINI_KEY)
    assert setups.apply(app, "recommended") == ""  # no download: Japanese goes to the cloud
    assert ("use_cloud_speech", "gemini", "gemini-3.5-transcribe") in app.calls
    assert (app.gateway.chosen, app.settings.cleanup_model) == ("gemini", "gemini-3.5-flash-lite")
    assert "recommended" in setups.in_use(app.settings, app.gateway)
    app = w.PreviewApp(settings=Settings(welcomed=True), gateway=GatewayConfig().with_key("groq", "gsk"))
    app.last_scan = QUICK
    assert setups.apply(app, "fastest") == "parakeet"  # English on a quick PC: Parakeet, downloaded once
    assert ("download_speech_model", "parakeet") in app.calls
    assert (app.gateway.chosen, app.settings.cleanup_model) == ("groq", "openai/gpt-oss-20b")
    assert setups.current(app.settings, app.gateway, pending="parakeet") == "fastest"


def test_the_window_shows_what_each_setup_chooses_for_the_language():
    gateway = GatewayConfig("", "AIza-key", "gemini")
    app = w.PreviewApp(settings=Settings(welcomed=True, speech_language="ta", speech_model="gemini", cleanup=True,
                                         cleanup_model="gemini-3.5-flash-lite",
                                         speech_cloud_models={"gemini": "gemini-3.5-flash-lite"}), gateway=gateway)
    window = w.MainWindow(app)
    window.show_page("models")
    card = window.pages["models"].setup_card
    tiles = card.tiles
    assert tiles["recommended"].speech.text() == "Gemini 3.5 Flash-Lite, on Gemini"  # Tamil: not Parakeet
    assert tiles["recommended"].needs.text() == "Gemini key: saved"
    assert [k for k, t in tiles.items() if t.state.text() == "In use"] == ["recommended", "fastest", "multilingual"]
    card.select("recommended")
    assert "Tamil: Parakeet hears English only" in card.blurb.text() and card.use.isHidden()  # in use already
    welcome = w.MainWindow(w.PreviewApp(settings=Settings(speech_language="ja"))).pages["welcome"]
    assert welcome.option_uses["recommended"].text() == "Whisper turbo + GPT-OSS 20B, both on Groq"
    assert welcome.foot_note.text() == "Next: your Groq key."  # no Parakeet download for Japanese
