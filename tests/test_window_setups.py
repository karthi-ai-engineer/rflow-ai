"""Setups on AI & models and in the first run, and what each model costs, on the window (off-screen, PreviewApp)."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst import window as w  # noqa: E402
from sst.gateway import GatewayConfig  # noqa: E402
from sst.settings import Settings  # noqa: E402

SCAN = {"time": "2026-10-06 10:00", "computer": {}, "verdicts": [
    {"key": "parakeet", "level": "recommended", "seconds": 0.9, "measured": False,
     "reason": "about 0.9 s per sentence (estimated)"},
    {"key": "whisper-turbo", "level": "no", "seconds": 9.5, "measured": False,
     "reason": "needs about 3.3 GB of memory; this computer has 4 GB"}]}


@pytest.fixture(scope="module", autouse=True)
def qt():
    return QApplication.instance() or QApplication([])


def _window(**kwargs) -> tuple[w.MainWindow, w.PreviewApp]:
    app = w.PreviewApp(**kwargs)
    return w.MainWindow(app), app


def _recommended() -> dict:
    return {"gateway": GatewayConfig("", "AIza-key-0000-9f3c", "gemini"),
            "settings": Settings(welcomed=True, cleanup=True, cleanup_model="gemini-3.5-flash-lite")}


def _wait_until(condition, seconds: float = 2.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


# ---- AI & models

def test_ai_and_models_opens_with_the_setups_and_marks_the_one_in_use():
    window, _ = _window(**_recommended())
    window.show_page("models")
    card = window.pages["models"].setup_card
    assert list(card.tiles) == ["recommended", "multilingual", "local", "custom"]  # no Fastest: Groq isn't offered
    tiles = card.tiles
    assert tiles["recommended"].state.text() == "In use" and tiles["multilingual"].state_row.isHidden()
    assert tiles["recommended"].speech.text() == "Parakeet, on this PC"
    assert tiles["recommended"].ai.text() == "Gemini 3.5 Flash-Lite, on Gemini"
    assert tiles["recommended"].cost.text() == "About $1.05 a month" and tiles["recommended"].needs.text() == \
        "Gemini key: saved"
    assert tiles["multilingual"].needs.text() == "Gemini key: saved" and tiles["multilingual"].cost.text() == \
        "About $3.58 a month"
    assert tiles["multilingual"].cost.property("tone") == "warn" and not tiles["multilingual"].cost_lamp.isHidden()
    assert tiles["local"].cost.text() == "Free" and tiles["custom"].cost.text() == "Its cost shows as you choose"
    assert card.panel.isHidden() and not card.hint.isHidden()  # a tile first


def test_the_setups_go_two_by_two_on_a_narrow_window():
    window, _ = _window(settings=Settings(welcomed=True))
    window.show_page("models")
    page = window.pages["models"]
    window.resize(1000, 700)
    window.grab()
    card = page.setup_card
    assert card.columns == 3
    places = {key: card.grid.getItemPosition(card.grid.indexOf(tile)) for key, tile in card.tiles.items()}
    assert places == {"recommended": (0, 0, 1, 1), "multilingual": (0, 1, 1, 1), "local": (0, 2, 1, 1),
                      "custom": (1, 0, 1, 3)}  # Custom across, under the three
    window.resize(780, 540)
    page.setup_card.select("multilingual")  # its details open, with the key to paste: never wider than the window
    window.grab()
    page.widget().layout().activate()
    assert page.setup_card.columns == 2 and page.widget().width() <= page.viewport().width()
    assert not card.key_area.isHidden() and card.grid.getItemPosition(card.grid.indexOf(card.tiles["custom"])) == \
        (1, 1, 1, 1)  # two by two
    assert page.widget().minimumSizeHint().width() <= page.viewport().width()


def test_choosing_multilingual_asks_for_the_gemini_key_then_the_voice_then_uses_both_parts():
    gateway = GatewayConfig("http://gw.example/v1", "gw-key", "vllm").with_key("openai", "sk-openai-0000")
    window, app = _window(gateway=gateway, settings=Settings(welcomed=True, cleanup=True, cleanup_model="qwen3"))
    window.show_page("models")
    page = window.pages["models"]
    card = page.setup_card
    card.tiles["multilingual"].clicked.emit()
    assert card.tiles["multilingual"].chosen and not card.panel.isHidden() and not card.key_area.isHidden()
    assert "Your voice goes to Google Gemini" in card.speech_words.text()
    assert "speech $2.52, AI $1.05" in card.cost_words.text() and "Google may use" in card.free_note.text()
    card.use.click()
    assert "Paste your Google Gemini key first" in card.result.text() and not app.calls  # nothing without the key
    card.key.setText("AIza-test-key-1111")
    assert page.unsaved()  # a key typed and not used yet
    questions = []
    page.confirm = lambda question: questions.append(question) or False
    card.use.click()
    assert "sent to Google Gemini" in questions[0] and not app.calls  # said no: not even the key is saved
    page.confirm = lambda question: True
    card.use.click()
    names = [call[0] for call in app.calls]
    assert names[:3] == ["save_key", "use_cloud_speech", "save_cleanup"]  # the key first, alone
    assert ("use_cloud_speech", "gemini", "gemini-3.5-transcribe") in app.calls
    assert (app.gateway.chosen, app.settings.cleanup_model, app.settings.speech_model) == ("gemini", "gemini-3.5-flash-lite",
                                                                                           "gemini")
    assert app.gateway.key_for("vllm") == "gw-key" and app.gateway.key_for("openai") == "sk-openai-0000"
    assert app.gateway.entries()["vllm"][0] == "http://gw.example/v1"  # your own server stays, for going back
    assert card.tiles["multilingual"].state.text() == "In use" and card.use.isHidden() and card.key_area.isHidden()
    assert "in use from the next dictation" in card.result.text() and not page.unsaved()


def test_recommended_downloads_parakeet_first_when_it_isnt_here(no_parakeet):
    gateway = GatewayConfig("", "gsk-key", "groq").with_key("gemini", "AIza-key")
    window, app = _window(gateway=gateway, settings=Settings(welcomed=True, speech_model="groq", cleanup=True,
                                                             cleanup_model="openai/gpt-oss-20b"))
    window.show_page("models")
    card = window.pages["models"].setup_card
    assert card.tiles["custom"].state.text() == "In use"  # Groq's AI: no setup offered uses it
    card.select("recommended")
    assert card.key_area.isHidden()  # the Gemini key is saved already
    card.use.click()
    assert ("download_speech_model", "parakeet") in app.calls and app.settings.cleanup_model == "gemini-3.5-flash-lite"
    assert "Parakeet is downloading" in card.result.text()
    app.downloading = ("parakeet", 331_000_000, 663_043_117)
    card.refresh()
    assert card.tiles["recommended"].state.text() == "Downloading 49%"  # the setup chosen, its speech on its way


def test_local_checks_this_pc_first_and_says_plainly_what_it_can_run():
    window, app = _window(**_recommended())
    window.show_page("models")
    card = window.pages["models"].setup_card
    card.select("local")
    assert not card.local.isHidden() and card.key_area.isHidden() and "coming soon" in card.ai_words.text()
    assert "Check this PC first" in card.check_words.text() and not card.use.isEnabled()
    card.scan.click()
    assert ("scan_computer",) in app.calls
    app.last_scan = SCAN
    card.local.buttons["whisper-turbo"].click()
    assert "can't run OpenAI Whisper large-v3 turbo" in card.check_words.text() and not card.use.isEnabled()
    card.local.buttons["parakeet"].click()
    assert "This PC runs NVIDIA Parakeet well" in card.check_words.text() and card.use.isEnabled()
    card.use.click()
    assert not app.settings.cleanup and app.settings.speech_model == "parakeet"
    assert app.settings.cleanup_model == "gemini-3.5-flash-lite"  # kept for later
    assert card.tiles["local"].state.text() == "In use"


def test_custom_shows_what_the_choice_costs_and_why_it_is_red():
    settings = Settings(welcomed=True, cleanup=True, cleanup_model="gemini-flash-latest", speech_model="gemini",
                        speech_cloud_models={"gemini": "gemini-3.6-flash"})
    window, _ = _window(gateway=GatewayConfig("", "AIza-key", "gemini"), settings=settings)
    window.show_page("models")
    card = window.pages["models"].setup_card
    assert card.tiles["custom"].state.text() == "In use" and card.tiles["custom"].cost.property("tone") == "err"
    card.select("custom")
    assert card.use.isHidden() and not card.links.isHidden()
    assert "speech $5.67, AI $6.34" in card.cost_words.text()
    assert "Speech: About $5.67 a month for typical use: an expensive model, over $5.00 a month." in card.warnings.text()
    card.links.findChildren(w.QPushButton)[1].click()
    assert window.current_page() == "cleanup"


# ---- each model's cost, and a red one asks once

def test_the_ai_connection_shows_each_model_s_cost_and_asks_once_for_an_expensive_one():
    window, app = _window(**_recommended())
    window.show_page("cleanup")
    page = window.pages["cleanup"]
    assert page.cost.words.text() == "About $1.05 a month for typical use: 20 minutes of dictation a day."
    assert page.cost.lamp.isHidden()
    delegate = page.model.itemDelegate()
    page.model.addItems(["gemini-flash-latest", "gemini-9-unknown"])
    assert delegate._words(page.model.model().index(1, 0)) == ("≈ $6.34/mo", "err")
    assert delegate._words(page.model.model().index(2, 0)) == ("price unknown", "text3")
    page.model.setCurrentText("gemini-flash-latest")
    assert "an expensive model, over $5.00" in page.cost.words.text() and not page.cost.lamp.isHidden()
    assert page.cost.words.property("tone") == "err"
    questions = []
    page.confirm = lambda question: questions.append(question) or False
    page.bar.save.click()
    assert "gemini-flash-latest is an expensive model" in questions[0] and app.settings.cleanup_model == \
        "gemini-3.5-flash-lite"
    page.confirm = lambda question: questions.append(question) or True
    page.bar.save.click()
    assert app.settings.cleanup_model == "gemini-flash-latest" and len(questions) == 2
    page.fallback.setCurrentText("gemini-3.5-flash-lite")
    page.bar.save.click()
    assert len(questions) == 2 and app.settings.cleanup_fallback == "gemini-3.5-flash-lite"  # asked once only
    page.model.setCurrentText("my-own-model")
    assert page.cost.words.text().startswith("Price unknown") and page.cost.lamp.isHidden()  # never red


def test_a_cloud_speech_model_shows_its_cost_and_an_expensive_one_asks_with_the_voice_question():
    window, app = _window(gateway=GatewayConfig().with_key("gemini", "AIza-key"))
    page = window.pages["speech"]
    page.show_where("gemini")
    gemini = page.models["gemini"]
    assert gemini.cost.words.text() == "About $2.52 a month for typical use: a mid-priced model."
    assert gemini.cost.words.property("tone") == "warn"
    gemini.model_box.setCurrentText("gemini-flash-latest")
    questions = []
    page.confirm = lambda question: questions.append(question) or True
    gemini.choose.click()
    assert len(questions) == 1 and "sent to Google Gemini" in questions[0] and "expensive model" in questions[0]
    assert app.settings.speech_cloud_models["gemini"] == "gemini-flash-latest"
    gemini.model_box.setCurrentText("gemini-3.5-transcribe")
    gemini.choose.click()
    assert len(questions) == 1  # a model that isn't red asks nothing more


# ---- the first run

def test_the_first_run_starts_by_choosing_a_setup_recommended_first(no_parakeet):
    window, app = _window(settings=Settings())
    welcome = window.pages["welcome"]
    assert welcome.step == 0 and welcome.setup == "recommended" and welcome.options["recommended"].chosen
    assert list(welcome.options) == ["recommended", "multilingual", "local", "custom"]  # no Fastest: Groq isn't offered
    assert [n.text() for n in welcome.stepper.names] == ["Setup", "Key", "Try it"]
    assert welcome.option_costs["recommended"].text() == "About $1.05 a month"
    assert welcome.option_costs["multilingual"].property("tone") == "warn"
    assert welcome.primary.text() == "Continue" and "Parakeet (663 MB) downloads" in welcome.foot_note.text()
    welcome.primary.click()
    assert welcome.current() == "key" and not app.calls  # nothing changes before the key
    assert "Gemini" in welcome.key_title.text() and not welcome.primary.isEnabled()  # a key first
    assert "Parakeet on this PC" in welcome.key_about.text() and "Google may use the text" in welcome.key_note.text()
    welcome.setup_key.setText("AIza-test-key")
    assert welcome.primary.isEnabled() and welcome.primary.text() == "Connect and continue"
    welcome.primary.click()
    assert _wait_until(lambda: welcome.current() == "try")
    assert ("check_ai", "gemini", "gemini-3.5-flash-lite") in app.calls and ("save_key", "gemini") in app.calls
    assert ("download_speech_model", "parakeet") in app.calls  # Parakeet comes with the setup
    assert app.settings.cleanup and app.settings.cleanup_model == "gemini-3.5-flash-lite"
    assert welcome.primary.text() == "Finish"
    welcome.primary.click()
    assert app.settings.welcomed and window.current_page() == "home"


def test_the_first_run_with_multilingual_uses_gemini_for_both_parts():
    window, app = _window(settings=Settings())
    welcome = window.pages["welcome"]
    welcome.options["multilingual"].clicked.emit()
    assert welcome.setup == "multilingual" and welcome.primary.text() == "Continue"
    welcome.primary.click()
    assert welcome.current() == "key" and "Your voice is sent to Google Gemini" in welcome.key_voice.text()
    assert "both at Google Gemini" in welcome.key_about.text() and welcome.key_title.text() == "Google Gemini API key"
    welcome.setup_key.setText("AIza-test")
    welcome.primary.click()
    assert _wait_until(lambda: welcome.current() == "try")
    assert ("check_ai", "gemini", "gemini-3.5-flash-lite") in app.calls
    assert app.settings.speech_model == "gemini" and app.gateway.key_for("gemini") == "AIza-test"


def test_a_key_the_provider_refuses_is_said_and_nothing_is_saved():
    window, app = _window(settings=Settings())
    welcome = window.pages["welcome"]

    def refuse(gateway, model):
        raise RuntimeError("HTTP 400 API key not valid")
    app.check_ai = refuse
    welcome.primary.click()
    welcome.setup_key.setText("AIza-wrong")
    welcome.primary.click()
    assert _wait_until(lambda: "Couldn't connect" in welcome.key_status.text())
    assert "not valid" in welcome.key_status.text() and welcome.current() == "key"
    assert not any(call[0] in ("save_key", "save_cleanup") for call in app.calls)


def test_the_first_run_with_local_checks_this_pc(no_parakeet):
    window, app = _window(settings=Settings())
    welcome = window.pages["welcome"]
    welcome.choose_setup("local")
    assert [n.text() for n in welcome.stepper.names] == ["Setup", "This PC", "Try it"]
    welcome.primary.click()
    assert welcome.current() == "check" and ("scan_computer",) in app.calls
    assert not welcome.primary.isEnabled()  # the check first
    app.last_scan = SCAN
    welcome.refresh(False)
    assert "This PC runs NVIDIA Parakeet well" in welcome.check_words.text()
    welcome.local.buttons["whisper-turbo"].click()
    assert "can't run" in welcome.check_words.text() and not welcome.primary.isEnabled()
    welcome.local.buttons["parakeet"].click()
    assert welcome.primary.text() == "Download Parakeet and continue" and welcome.primary.isEnabled()
    welcome.primary.click()
    assert ("download_speech_model", "parakeet") in app.calls and welcome.current() == "try"
    assert not app.settings.cleanup


def test_the_first_run_with_custom_keeps_the_old_steps():
    window, app = _window(settings=Settings())
    welcome = window.pages["welcome"]
    welcome.choose_setup("custom")
    assert [n.text() for n in welcome.stepper.names] == ["Setup", "Hear you", "Try it", "AI"]
    welcome.primary.click()
    assert welcome.current() == "hear" and welcome.local_option.chosen
    welcome.primary.click()
    assert welcome.current() == "try" and welcome.primary.text() == "Continue"
    welcome.primary.click()
    assert welcome.current() == "ai" and welcome.tiles["gemini"].chosen and welcome.primary.text() == "Connect and finish"
