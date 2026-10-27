"""The Rflow window, built off-screen with PreviewApp in place of the tray app (no model, no microphone, no hook)."""
import dataclasses
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import date, datetime, timedelta  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFontMetrics  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from sst import bench  # noqa: E402
from sst import window as w  # noqa: E402
from sst.audio import save_wav  # noqa: E402
from sst.commands import DEFAULT_PHRASES  # noqa: E402
from sst.gateway import GatewayConfig  # noqa: E402
from sst.settings import Settings, Stats  # noqa: E402
from sst.snippets import load  # noqa: E402
from sst.transform import TRANSFORMS, TransformGuard  # noqa: E402
from sst.ui import KeyCap  # noqa: E402
from sst.wakeui import BLUETOOTH  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qt():
    return QApplication.instance() or QApplication([])


def _labels(widget) -> str:
    return " | ".join(label.text() for label in widget.findChildren(QLabel))


def _shown(widget) -> str:
    """The labels shown, not those of a list just filled again (hidden until Qt deletes them)."""
    return " | ".join(label.text() for label in widget.findChildren(QLabel) if label.isVisibleTo(widget))


def _keys(widget) -> list[str]:
    """The keys drawn as keys, as shown."""
    return [cap.key_text for cap in widget.findChildren(KeyCap) if cap.isVisibleTo(widget)]


def _button(widget, caption: str) -> QPushButton:
    return next(b for b in widget.findChildren(QPushButton) if b.text() == caption)


def _window(**kwargs) -> tuple[w.MainWindow, w.PreviewApp]:
    app = w.PreviewApp(**kwargs)
    return w.MainWindow(app), app


# ---- the window

def test_a_new_user_sees_the_welcome_and_then_home():
    window, app = _window(settings=Settings())
    assert window.current_page() == "welcome" and window.sidebar.isHidden()  # the welcome has the whole window
    welcome = window.pages["welcome"]
    welcome.go_to_step("try")
    window.set_status("Loading the speech model...", False)
    assert "Loading the speech model" in welcome.status.text()
    window.set_status("Ready: hold Ctrl+Win", True)
    assert "Ready" in welcome.status.text() and welcome.orb.state == "ready"
    _button(welcome, "Skip setup").click()
    assert app.settings.welcomed and window.current_page() == "home" and not window.sidebar.isHidden()


def test_someone_who_has_been_welcomed_starts_at_home_and_can_open_every_page():
    window, _ = _window()
    assert window.current_page() == "home"
    for key, _label in w.NAV:
        window.nav[key].click()
        assert window.current_section() == key and window.nav[key].isChecked()
        assert not window.grab().isNull()
    for key in window.pages:  # and every page one level down, each under its section's button
        window.show_page(key)
        assert window.current_page() == key and not window.grab().isNull()
        if key != "welcome":
            assert window.nav[w.SECTION[key]].isChecked()


def test_building_and_opening_the_window_flashes_no_other_window():
    """A widget made visible before it has a parent shows as a window of its own for a moment: at start-up that was
    dozens of small "Rflow" windows opening and closing (#181)."""
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QWidget

    class Spy(QObject):
        shown: list[str] = []

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow() \
                    and not isinstance(obj, w.MainWindow):
                self.shown.append(f"{type(obj).__name__} {obj.objectName()} {obj.size().toTuple()}")
            return False

    spy = Spy()
    QApplication.instance().installEventFilter(spy)
    try:
        for kwargs in ({}, {"settings": Settings()}, {"gateway": GatewayConfig("http://localhost:8000/v1", "k", "vllm")}):
            window, _ = _window(**kwargs)
            window.open()
            for key in window.pages:
                window.show_page(key)
            window.hide()
    finally:
        QApplication.instance().removeEventFilter(spy)
    assert spy.shown == []


def test_pages_are_built_when_first_wanted_as_the_window_is_then(monkeypatch):
    """Building all fifteen pages took 3-5 s and 107 MB at every start: Home (and the welcome) now, the rest when used.
    A page built after the theme, the window's width or the app's settings changed is as if it had been there all along."""
    window, _ = _window(settings=Settings())
    assert set(window.pages.built) == {"home", "welcome"}
    monkeypatch.setattr(w, "dark_mode", lambda: False)
    born_light, _ = _window(settings=Settings(welcomed=True, hotkey="menu"))  # light and "menu" from the start
    early = born_light.pages["settings"]  # built in light (the theme is global: before the next window sets dark)
    monkeypatch.setattr(w, "dark_mode", lambda: True)
    window, app = _window()
    assert set(window.pages.built) == {"home"} and len(window.pages) == 16 and "settings" in window.pages
    assert list(window.pages)[:3] == ["home", "dictionary", "snippets"]
    for win in (window, born_light):
        win.open("home")
        win.resize(820, 600)  # narrow: the pages' tighter margins
        win.set_update_status("Rflow is up to date.")
        win.set_status("Ready: hold Ctrl+Win", True)
    window.apply_theme("light")
    app.settings = dataclasses.replace(app.settings, hotkey="menu")  # e.g. another profile's settings
    window.show_page("settings")
    born_light.show_page("settings")
    page = window.pages["settings"]
    assert "settings" in window.pages.built and window.current_page() == "settings" and window.pages["home"].ready
    assert page.update_status.text() == "Rflow is up to date." and page.hotkey.currentData() == "menu"
    assert page.body.contentsMargins().left() == early.body.contentsMargins().left() == 24
    assert page.grab().toImage() == early.grab().toImage()  # in the theme in use, icons and all
    for win in (window, born_light):
        win.hide()
    window.apply_theme("dark")


def test_pages_one_level_down_lead_back():
    window, _ = _window()
    window.show_page("speech")
    _button(window.pages["speech"], "AI & models").click()
    assert window.current_page() == "models"
    window.show_page("words")
    assert window.current_page() == "dictionary"  # Words opens on Your words


def test_words_and_snippets_are_sections_of_their_own():
    window, _ = _window()
    window.nav["snippets"].click()
    assert window.current_page() == "snippets" and window.nav["snippets"].isChecked()
    assert window.pages["snippets"].title.text() == "Snippets" and not hasattr(window.pages["snippets"], "tabs")
    window.nav["words"].click()
    assert window.current_page() == "dictionary" and window.pages["dictionary"].title.text() == "Words"
    assert not hasattr(window.pages["dictionary"], "tabs")


def test_home_shows_the_stats_and_the_dictations_by_day():
    today = date.today()
    stats = Stats()
    stats.add("one two three", 1.5, today)
    stats.add("four five", 1.0, today - timedelta(days=1))
    history = [{"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": "Hello from today.", "heard": "hello from today"},
               {"time": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), "text": "And yesterday."}]
    window, _ = _window(history=history, stats=stats)
    window.set_status("Ready: hold Ctrl+Win", True)
    home = window.pages["home"]
    labels = _labels(home)
    assert home.list_title.text() == "Today" and "Yesterday" in labels
    assert "Hello from today." in labels and "And yesterday." in labels
    assert home.stat_values["week"].text() == "5" and home.stat_values["total"].text() == "5"
    assert home.stat_values["streak"].text() == "2"
    assert home.stat_values["speed"].text() == "–"  # not yet half a minute of speech
    assert not home.hero_keys.isHidden() and home.hero_keys.accessibleName().startswith("Hold Ctrl+Win and talk")
    assert [cap.key_text for cap in home.hero_keys.findChildren(KeyCap)] == ["Ctrl", "Win"]  # drawn as keys
    assert home.orb.state == "ready" and window.status_title.text() == "Ready"


def test_the_orb_moves_only_while_listening_or_cleaning_up():
    """Each orb frame repaints the soft surfaces under it too: a breathing "ready" cost half a core while Home was open."""
    from sst.ui import Orb
    window, app = _window()
    app.speech_in_use = lambda: "parakeet"
    window.open("home")
    window.set_status("Ready: hold Ctrl+Win", True)
    home = window.pages["home"]
    assert home.orb.state == "ready" and home.orb.isVisible() and not home.orb.animating
    window.hide()
    orb = Orb()
    orb.show()
    for state in ("ready", "off", "loading", "done"):
        orb.set_state(state, 0.5)
        assert not orb.animating, state
    for state in ("live", "ai"):
        orb.set_state(state)
        assert orb.animating, state
    orb.hide()
    assert not orb.animating
    orb.show()
    assert orb.animating  # still listening when shown again
    orb.set_state("ready")
    assert not orb.animating and not orb.grab().isNull()
    orb.hide()


def test_home_explains_what_to_do_before_the_first_dictation():
    window, _ = _window(settings=Settings(welcomed=True, hotkey="menu"))
    labels = _labels(window.pages["home"])
    assert "Nothing dictated yet" in labels and "hold Menu key" in labels
    assert window.pages["home"].search_box.isHidden()  # nothing to search yet


def test_home_searches_the_dictations():
    history = [{"time": f"2026-09-30 10:{m:02}:00", "text": text} for m, text in
               [(10, "Book a room for Friday."), (12, "Send the Q3 roadmap."), (14, "Lunch on Friday?")]]
    window, _ = _window(history=history)
    home = window.pages["home"]
    home.search.setText("friday")
    shown = [row.body.text() for row in home.findChildren(w.HistoryRow) if not row.isHidden()]
    assert sorted(shown) == ["Book a room for Friday.", "Lunch on Friday?"]
    home.search.setText("nowhere")
    assert "No dictation has" in _labels(home)


def test_home_says_what_rflow_is_waiting_for(no_parakeet):
    window, app = _window()
    home = window.pages["home"]
    home.refresh()
    assert home.hero_title.text() == "Choose how Rflow hears you" and not home.get_parakeet.isHidden()
    home.get_parakeet.click()
    assert ("download_speech_model", "parakeet") in app.calls
    app.downloading = ("parakeet", 331_000_000, 663_043_117)
    window.refresh()
    assert home.hero_title.text().startswith("Getting Parakeet") and home.orb.state == "loading"
    assert round(home.orb.progress, 2) == 0.5 and not home.progress_row.isHidden()
    assert window.status_title.text() == "Not ready yet" and "49%" in window.status_label.text()
    home.pause.click()
    assert ("cancel_download",) in app.calls


def test_a_dictation_can_be_copied_again():
    window, _ = _window(history=[{"time": "2026-09-30 10:15:00", "text": "Copy me."}])
    copy = next(b for b in window.pages["home"].findChildren(w.IconButton) if b.toolTip() == "Copy")
    copy.click()
    assert QApplication.clipboard().text() == "Copy me." and copy.icon_name == "check" and copy.toolTip() == "Copied"


def test_the_dictionary_adds_several_words_and_removes_one():
    window, app = _window(settings=Settings(welcomed=True, vocabulary=["GitHub"]))
    window.show_page("dictionary")
    page = window.pages["dictionary"]
    assert page.cleanup_off.isVisibleTo(window)  # AI cleanup is off: the page says so
    page.entry.setText("Tamil,  CodeQL , github")
    page._add()
    assert app.settings.vocabulary == ["GitHub", "Tamil", "CodeQL"] and page.entry.text() == ""
    assert page.count.text() == "3 words"
    remove = next(b for b in page.findChildren(w.IconButton) if b.toolTip() == "Remove GitHub")
    remove.click()
    assert app.settings.vocabulary == ["Tamil", "CodeQL"]
    assert window.toast.text.text() == "Removed “GitHub”"
    window.toast.action.click()  # Undo
    assert "GitHub" in app.settings.vocabulary


def test_typing_in_the_word_box_finds_a_word():
    window, _ = _window(settings=Settings(welcomed=True, vocabulary=["GitHub", "Kubernetes", "Priya"]))
    page = window.pages["dictionary"]
    page.refresh()
    page.entry.setText("kub")
    assert [chip.word for chip in page.findChildren(w.WordChip) if not chip.isHidden()] == ["Kubernetes"]
    page.entry.setText("Vercel")
    assert not page.none_found.isHidden() and "press Enter to add it" in page.none_found.text()


def test_settings_apply_at_once_and_keep_the_rest():
    window, app = _window(settings=Settings(welcomed=True, vocabulary=["Tamil"], cleanup=True, cleanup_model="m"))
    page = window.pages["settings"]
    page.hotkey.setCurrentIndex(page.hotkey.findData("menu"))
    page.sounds.setChecked(False)
    s = app.settings
    assert (s.hotkey, s.sounds) == ("menu", False)
    caps = [cap.key_text for cap in page.hotkey_caps.findChildren(KeyCap) if cap.isVisibleTo(page)]
    assert caps == ["Menu"]  # the key, drawn as a key
    assert (s.vocabulary, s.cleanup, s.cleanup_model, s.welcomed) == (["Tamil"], True, "m", True)  # untouched


def test_starting_with_the_voice_is_off_until_switched_on_next_to_the_dictation_key():
    window, app = _window()
    card = window.pages["settings"].wake
    assert not card.toggle.isChecked() and card.details.isHidden()
    card.toggle.setChecked(True)
    assert ("set_wake_enabled", True) in app.calls and app.settings.wake_enabled and not card.details.isHidden()
    assert not card.models.isHidden() and card.download.text() == "Download (58 MB)"  # the models come first
    assert not card.setup.isEnabled() and card.setup.text() == "Set up" and "Not set up yet" in card.keyword.text()
    card.download.click()
    assert ("download_wake_models",) in app.calls
    app._wake = w.WakeState(models="downloading", percent=40, size=57_900_000)
    card.refresh()
    assert card.download.isHidden() and not card.cancel.isHidden() and "40%" in card.models_caption.text()
    assert card.lamp.state == "warn" and "40%" in card.status.text()


def test_the_wake_card_says_where_it_listens_and_sets_the_keyword_up_again():
    window, app = _window(settings=Settings(welcomed=True, wake_enabled=True))
    app._wake = w.WakeState(keyword="okay type", heard=9, models="ready", listening="Microphone (Realtek)")
    card = window.pages["settings"].wake
    card.refresh()
    assert card.models.isHidden() and card.lamp.state == "ok"
    assert card.status.text().startswith("Listening for “okay type” on Microphone (Realtek)")
    assert card.keyword.text() == "“okay type”  ·  heard 9 of 10 in the setup" and card.setup.text() == "Set up again"
    card.setup.click()
    assert ("open_wake_setup",) in app.calls
    app._wake = w.WakeState(keyword="okay type", heard=9, models="ready", note=BLUETOOTH)
    card.refresh()
    assert card.lamp.state == "warn" and card.status.text() == BLUETOOTH  # why it doesn't listen


def test_the_wake_card_says_which_microphone_and_lets_it_listen_on_the_headset():
    window, app = _window(settings=Settings(welcomed=True, wake_enabled=True))
    headset = "Headset (OnePlus Nord Buds 3r)"
    app._wake = w.WakeState(keyword="karthi", heard=10, said=13, models="ready", listening="Microphone (Realtek)",
                            instead=headset)
    card = window.pages["settings"].wake
    card.refresh()
    assert card.keyword.text() == "“karthi”  ·  heard 10 of 13 in the setup"  # the takes said again count too
    assert card.status.text().startswith("Listening for “karthi” on Microphone (Realtek).")
    assert f"Not on {headset}: turn on “Listen on my Bluetooth headset”" in card.status.text()
    assert not card.bluetooth.isChecked()
    card.bluetooth.setChecked(True)
    assert app.settings.wake_bluetooth
    app._wake = w.WakeState(keyword="karthi", heard=10, said=13, models="ready", listening=headset)
    card.refresh()
    assert card.status.text().startswith(f"Listening for “karthi” on {headset}.") and "Not on" not in card.status.text()
    assert card.bluetooth.isChecked()


def test_end_phrases_are_saved_as_typed_and_done_is_warned_against():
    window, app = _window(settings=Settings(welcomed=True, wake_enabled=True))
    card = window.pages["settings"].wake
    assert card.phrases.text() == "stop it" and "Not “done”" in card.phrases_caption.text()
    card.phrases.setText(" stop it ,End it, stop it,")
    card.phrases.editingFinished.emit()
    assert app.settings.wake_end_phrases == ["stop it", "End it"] and card.phrases.text() == "stop it, End it"
    card.phrases.setText("stop it, Done.")
    card.phrases.editingFinished.emit()
    assert app.settings.wake_end_phrases == ["stop it", "Done."]
    assert card.phrases_caption.text().startswith("“Done.” ends a dictation whenever you say the word")
    card.phrases.setText("")
    card.phrases.editingFinished.emit()
    assert app.settings.wake_end_phrases == ["stop it"]  # there's always one
    card.voice.setChecked(False)
    assert not app.settings.wake_voice_check


def test_the_microphone_is_chosen_on_ai_and_models():
    window, app = _window(microphones=["Mic A", "Mic B"])
    page = window.pages["models"]
    page.microphone.combo.setCurrentIndex(page.microphone.combo.findData("Mic B"))
    assert app.settings.microphone == "Mic B"


def test_settings_keep_a_custom_hotkey_and_an_unplugged_microphone():
    chosen = Settings(welcomed=True, hotkey="ctrl+shift+f9", microphone="Old headset")
    window, _ = _window(settings=chosen, microphones=["Mic A"])
    page = window.pages["settings"]
    assert page.result(chosen) == chosen
    assert "not connected" in window.pages["models"].microphone.combo.currentText()


def test_the_cleanup_page_saves_what_was_typed():
    window, app = _window(settings=Settings(welcomed=True, cleanup_model="model-a"),
                          gateway=GatewayConfig("https://gw.example/v1", "key-1"))
    page = window.pages["cleanup"]
    # settings from before the provider choice: the address says it's a server of the user's own (vLLM or the like)
    assert page.result() == (False, "model-a", "", GatewayConfig("https://gw.example/v1", "key-1", "vllm"))
    page.cleanup_on.setChecked(True)
    page.api_key.setText("  key-2 ")
    page.model.setCurrentText("  typed-model ")  # any model name can be typed
    _button(page, "Save").click()
    assert app.calls[-1] == ("save_cleanup", True, "typed-model", "", GatewayConfig("https://gw.example/v1", "key-2", "vllm"))
    assert "Active from the next dictation" in page.bar.state.text()


def test_a_new_install_has_no_endpoint_or_model_and_cleanup_off():
    window, _ = _window()
    assert window.pages["cleanup"].result() == (False, "", "", GatewayConfig("", "", "gemini"))  # the first provider


def test_the_update_banner_leads_to_the_update():
    window, app = _window()
    window.show_update("Rflow 9.9.9 is available (you have 1.1.0).", version="9.9.9")
    assert window.banner.isVisibleTo(window) and "9.9.9" in window.update_link.text()
    window.update_button.click()
    window.update_link.click()
    assert app.calls.count(("start_update",)) == 2
    window.show_update("Downloading... 40%", busy=True)
    assert not window.update_button.isEnabled()


def test_closing_the_window_keeps_rflow_running():
    window, app = _window()
    window.open()
    window.close()
    assert not window.isVisible() and ("window_closed",) in app.calls


def test_both_themes_have_every_colour_and_image():
    from sst import theme
    assert theme.TOKENS["dark"].keys() == theme.TOKENS["light"].keys() == theme.POPUP.keys()
    for name in ("dark", "light"):
        sheet = w.stylesheet(name)
        assert "None" not in sheet and theme.css(theme.TOKENS[name]["text"]) in sheet and f"arrow-{name}.png" in sheet
        assert theme.SHADOWS[name].keys() >= {"card", "tile", "key", "in1", "in2"}
    for name in ("check", "arrow-light", "arrow-dark"):  # made by scripts/make_ui_images.py, shipped in sst/static
        assert (w.UI_IMAGES / f"{name}.png").exists() and (w.UI_IMAGES / f"{name}@2x.png").exists()


def test_the_geist_fonts_ship_with_rflow():
    from PySide6.QtGui import QFontInfo

    from sst import theme
    assert theme.load_fonts() and (theme.FONT_DIR / "OFL.txt").exists()  # their licence goes with them
    assert QFontInfo(theme.font(14, 600)).family() == "Geist" and QFontInfo(theme.font(12, mono=True)).family() == "Geist Mono"


def test_the_logo_ships_with_rflow_in_every_size():
    import struct

    from sst import ui
    data = w.ICON_FILE.read_bytes()  # the app icon: the window, the taskbar, the tray, Rflow.exe, the installer
    count = struct.unpack("<HHH", data[:6])[2]
    sizes = [struct.unpack("<BBBBHHII", data[6 + 16 * i:22 + 16 * i])[0] or 256 for i in range(count)]
    assert {16, 20, 24, 32, 48, 256} <= set(sizes)
    for size in ui.BRAND_SIZES:  # the mark the window draws, from scripts/make_brand.py
        assert not ui._brand_pixmap(size).isNull() and ui._brand_pixmap(size).width() == size
    assert ui.brand_mark(48).width() == 64 and ui.brand_mark(2000).width() == 512  # scaled down a little, never up
    window, _ = _window()
    assert window.mark.accessibleName() == "Rflow" and not window.mark.grab().isNull()


def test_the_theme_follows_windows_and_repaints_both_ways():
    from sst import theme
    window, _ = _window()
    window.apply_theme("light")
    assert theme.theme() == "light" and theme.css(theme.TOKENS["light"]["base"]) in window.styleSheet()
    assert not window.grab().isNull()
    window.apply_theme("dark")
    assert theme.theme() == "dark" and not window.grab().isNull()


def test_soft_shadows_are_drawn_without_seams():
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter

    from sst import theme
    image = QImage(300, 200, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(theme.tok("base"))
    p = QPainter(image)
    for kind in theme.RADIUS:
        theme.paint_surface(p, kind, QRectF(60, 50, 180, 100))
    theme.paint_surface(p, "orb", QRectF(100, 50, 100, 100))  # a circle: one scaled tile, no slices meeting in it
    p.end()
    assert image.pixelColor(150, 100).isValid()


# ---- the reading test

class FakeRecorder:
    rate, level = 16_000, 0.0

    def __init__(self, seconds=1.0):
        self.seconds, self.starts, self.stops = seconds, 0, 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1
        return np.zeros(int(self.seconds * self.rate), dtype=np.float32)


def _reading_test(tmp_path, recorder=None, score=None, add_words=None):
    return w.ReadingTest(recorder or FakeRecorder(), score or (lambda folders, progress: None),
                         add_words or (lambda words: 0), "Test microphone", folder=tmp_path / "test")


def test_reading_test_saves_each_sentence_and_moves_on(tmp_path):
    test = _reading_test(tmp_path)
    assert test.sentence.text() == bench.SENTENCES[0] and not test.score_button.isEnabled()
    test.toggle_recording()
    assert test.recording and test.record_button.text() == "Stop"
    test.toggle_recording()
    assert (tmp_path / "test" / "01.wav").exists()
    assert (tmp_path / "test" / "01.txt").read_text(encoding="utf-8") == bench.SENTENCES[0]
    assert test.index == 1 and test.sentence.text() == bench.SENTENCES[1]  # moved on to the next sentence
    assert test.score_button.isEnabled() and "1 recorded" in test.score_button.text()


def test_space_records_and_a_too_short_recording_is_not_kept(tmp_path):
    test = _reading_test(tmp_path, recorder=FakeRecorder(seconds=0.2))
    test._space()
    test._space()
    assert not (tmp_path / "test" / "01.wav").exists() and test.index == 0 and "too short" in test.status.text()


def test_an_unfinished_test_continues_at_the_first_sentence_not_read(tmp_path):
    first = _reading_test(tmp_path)
    for _ in range(2):
        first.toggle_recording()
        first.toggle_recording()
    again = _reading_test(tmp_path)  # the same folder, as the page passes bench.unfinished()
    assert again.index == 2 and again.sentence.text() == bench.SENTENCES[2] and "2 of 30 read" in again.status.text()


def test_redo_replaces_a_recording(tmp_path):
    recorder = FakeRecorder()
    test = _reading_test(tmp_path, recorder=recorder)
    test.toggle_recording()
    test.toggle_recording()
    test.go(0)
    assert test.redo_button.isEnabled() and not test.record_button.isEnabled()
    test.redo_button.click()
    test.toggle_recording()
    assert recorder.starts == 2 and test.recorded() == 1


def test_leaving_the_page_stops_a_recording_without_keeping_it(tmp_path):
    recorder = FakeRecorder()
    test = _reading_test(tmp_path, recorder=recorder)
    test.toggle_recording()
    test.stop()
    assert not test.recording and recorder.stops == 1 and test.recorded() == 0


def test_a_test_notes_its_set_and_microphone_with_the_first_recording(tmp_path):
    class DescribedRecorder(FakeRecorder):
        def describe(self):
            return {"device": "Headset (Earbuds)", "host_api": "MME", "rate": self.rate}
    test = w.ReadingTest(DescribedRecorder(), lambda folders, progress: None, lambda words: 0, "Earbuds",
                         folder=tmp_path / "test", block="C")
    assert test.sentence.text() == bench.BLOCKS["C"][0] and "Set C" in test.counter.text()
    test.toggle_recording()
    test.toggle_recording()
    session = bench.read_session(tmp_path / "test")
    assert (session["block"], session["microphone"], session["device"]) == ("C", "Earbuds", "Headset (Earbuds)")
    assert (tmp_path / "test" / "01.txt").read_text(encoding="utf-8") == bench.BLOCKS["C"][0]
    again = w.ReadingTest(FakeRecorder(), lambda folders, progress: None, lambda words: 0, "Laptop",
                          folder=tmp_path / "test", block="A")  # resumed: keeps its own set
    assert again.block == "C" and again.sentence.text() == bench.BLOCKS["C"][1]


def test_score_all_tests_scores_every_test_next_to_this_one(tmp_path):
    asked = []
    test = w.ReadingTest(FakeRecorder(), lambda folders, progress: asked.append(folders), lambda words: 0, "mic",
                         folder=tmp_path / "2026-10-01_100000")
    old = tmp_path / "2026-10-01_090000"
    bench.write_session(old, "A", "mic", {})
    save_wav(old / "01.wav", np.zeros(16000, dtype=np.float32), 16000)
    (old / "01.txt").write_text("x", encoding="utf-8")
    test.toggle_recording()
    test.toggle_recording()
    test.start_scoring(every=True)
    test.start_scoring()
    deadline = time.monotonic() + 5
    while len(asked) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sorted(asked, key=len) == [[test.folder], [old, test.folder]]


def test_results_show_every_setup_and_add_the_ticked_words(tmp_path):
    from sst.audio import AudioStats
    from sst.evaluate import Recording, Results, Score
    wide, narrow = AudioStats(2.0, -20, -70, -3, 0, -30), AudioStats(2.0, -20, -60, -3, 0, -80)
    recordings = [Recording("1.wav", "s1", "t1", "A", "Laptop", wide), Recording("2.wav", "s2", "t2", "C", "Earbuds", narrow)]
    alone = Score("Parakeet alone", ["a", "b"], [3, 1], [10, 10], [1, 0], [2, 1], [0, 0], [5, 1], [50, 50], [0.4, 0.6],
                  interval=(0.1, 0.3))
    cleaned = Score("Parakeet + m", ["a", "b"], [1, 1], [10, 10], [0, 0], [2, 1], [0, 0], [2, 1], [50, 50], [0.7, 0.9],
                    interval=(0.05, 0.15), difference=(-0.1, -0.2, -0.02))
    results = Results([str(tmp_path)], recordings, [alone, cleaned], [("tamil", "tamar", 2)], ["Tamil", "CodeQL"])
    added = []
    test = _reading_test(tmp_path, add_words=lambda words: added.extend(words) or len(words))
    test._show_results(results)
    assert test.pages.currentIndex() == 1
    html = test.report.toHtml()
    assert "Parakeet alone" in html and "20.0%" in html and "10.0%" in html and "tamar" in html
    assert "better" in html and "from 2 tests" in html
    assert "Earbuds" in html and "phone call" in html and "Laptop</b>" not in html  # only the narrowband one warned
    test.suggestions.item(1).setCheckState(Qt.CheckState.Unchecked)
    test._add_selected()
    assert added == ["Tamil"] and "Added 1 word" in test.results_status.text()


def test_new_test_starts_a_fresh_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "BENCH_DIR", tmp_path)
    window, _ = _window()
    page = window.pages["reading"]
    first = page.ensure_test()
    first.restart.emit()
    assert page.test is not first and page.test.index == 0


# ---- providers and profiles

def test_each_provider_asks_for_what_it_needs():
    window, _ = _window()
    page = window.pages["cleanup"]
    for key, provider in w.PROVIDERS.items():
        page.provider.setCurrentIndex(page.provider.findData(key))
        assert page.form.isRowVisible(page.gateway_url) is provider.own_server  # an address only for your own server
        assert (not page.key_link.isHidden()) is bool(provider.key_page)  # "Get a key" for the cloud ones


def test_switching_providers_keeps_each_one_s_key_and_address():
    window, app = _window()
    page = window.pages["cleanup"]
    page.provider.setCurrentIndex(page.provider.findData("gemini"))
    page.api_key.setText("gemini-key")
    page.model.setCurrentText("gemini-3.5-flash-lite")
    page.provider.setCurrentIndex(page.provider.findData("vllm"))
    assert page.api_key.text() == "" and page.model.currentText() == ""  # a fresh start for the other provider
    page.gateway_url.setText("http://my-server:8000/v1")
    page.provider.setCurrentIndex(page.provider.findData("gemini"))
    assert page.api_key.text() == "gemini-key" and page.model.currentText() == "gemini-3.5-flash-lite"
    page.cleanup_on.setChecked(True)
    _button(page, "Save").click()
    on, model, _, gateway = app.calls[-1][1:]
    assert (on, model) == (True, "gemini-3.5-flash-lite")
    assert gateway == GatewayConfig("", "gemini-key", "gemini", {"vllm": ("http://my-server:8000/v1", "")})
    assert gateway.address == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_a_saved_provider_opens_with_its_other_keys(qt):
    gateway = GatewayConfig("http://gw.example/v1", "gw-key", "vllm", {"gemini": ("", "gemini-key")})
    window, _ = _window(gateway=gateway)
    page = window.pages["cleanup"]
    assert page.provider.currentData() == "vllm" and page.api_key.text() == "gw-key"
    assert page.gateway_url.text() == "http://gw.example/v1"  # your own server (a company gateway) as saved
    page.provider.setCurrentIndex(page.provider.findData("gemini"))
    assert page.api_key.text() == "gemini-key"


def test_the_cleanup_offers_only_gemini_ollama_and_your_own_server(qt):
    page = _window()[0].pages["cleanup"]
    assert [page.provider.itemData(i) for i in range(page.provider.count())] == ["gemini", "ollama", "vllm"]
    # A profile still on a provider no longer offered: shown as Gemini, nothing saved changes, nothing crashes
    gateway = GatewayConfig("", "gsk-key", "groq", {"anthropic": ("", "ant-key")})
    window, app = _window(gateway=gateway, settings=Settings(welcomed=True, cleanup=True,
                                                             cleanup_model="openai/gpt-oss-20b"))
    page = window.pages["cleanup"]
    assert page.provider.currentData() == "gemini" and page.model.currentText() == "" and not page.unsaved()
    page.refresh()
    assert page.provider.currentData() == "gemini" and not page.unsaved()
    assert (app.gateway.chosen, app.settings.cleanup_model) == ("groq", "openai/gpt-oss-20b")
    _, _, _, saved = page.result()
    assert saved.key_for("groq") == "gsk-key" and saved.key_for("anthropic") == "ant-key"  # their keys stay saved


def test_openai_groq_and_anthropic_are_offered_nowhere(qt):
    import re

    from PySide6.QtWidgets import QComboBox
    # Their keys saved, and Groq the speech model chosen before they were hidden: still named nowhere (OpenAI's Whisper
    # on this PC and servers with OpenAI's API are other things)
    named = re.compile(r"\b(Groq|Anthropic|OpenAI)\b(?![- ](compatible|Whisper)|'s transcription API)")
    gateway = GatewayConfig("", "AIza-key", "gemini", {"groq": ("", "gsk"), "openai": ("", "sk"), "anthropic": ("", "a")})
    for settings in (Settings(welcomed=True, speech_model="groq"), Settings()):
        window, _ = _window(gateway=gateway, settings=settings)
        for name in ("models", "speech", "cleanup", "welcome"):
            page = window.pages[name]
            if name == "welcome":
                for key in ("custom", "recommended"):
                    page.show_step(0)
                    page.choose_setup(key)
                    for step in range(len(page.flow)):
                        page.show_step(step)
            texts = [label.text() for label in page.findChildren(QLabel)]
            texts += [b.text() for b in page.findChildren(QPushButton)]
            texts += [box.itemText(i) for box in page.findChildren(QComboBox) for i in range(box.count())]
            assert not [text for text in texts if named.search(text)], name


def test_load_models_needs_an_address_for_your_own_server():
    window, _ = _window()
    page = window.pages["cleanup"]
    page.provider.setCurrentIndex(page.provider.findData("vllm"))
    page._load_models()
    assert "address" in page.test_result.text()


def _profiles_window():
    from sst.settings import Profiles
    profiles = Profiles()
    profiles.current.name = "Karthi Raj"
    profiles.add("Rahul")
    return _window(profiles=profiles)


def test_the_sidebar_shows_whose_profile_it_is_when_there_are_several():
    window, _ = _profiles_window()
    assert "Karthi Raj" in window.profile_button.text() and not window.profile_button.isHidden()
    alone, _ = _window()
    assert alone.profile_button.isHidden()  # one profile: nothing to switch to


def test_the_profiles_page_switches_renames_and_deletes():
    window, app = _profiles_window()
    window.show_page("profiles")
    page = window.pages["profiles"]
    assert "In use" in _labels(page)
    _button(page, "Switch to this profile").click()
    assert ("switch_profile", "rahul") in app.calls
    names = [box for box in page.findChildren(w.QLineEdit) if box.text() == "Rahul"]
    names[0].setText("Rahul K")
    names[0].editingFinished.emit()
    assert app.profiles.get("rahul").name == "Rahul K"
    app.profiles.active = "default"
    page.refresh()
    page.confirm = lambda question: "Rahul K" in question
    _button(page, "Delete").click()
    assert ("delete_profile", "rahul") in app.calls and [p.id for p in app.profiles.items] == ["default"]


def test_a_new_profile_is_made_from_its_name():
    window, app = _profiles_window()
    page = window.pages["profiles"]
    page.new_name.setText("Priya")
    page._create()
    assert app.profiles.current.name == "Priya" and page.new_name.text() == ""


def test_the_welcome_connects_an_ai_and_picks_its_fast_model():
    window, app = _window(settings=Settings())
    welcome = window.pages["welcome"]
    welcome.choose_setup("custom")  # each part by hand: the AI last
    welcome.go_to_step("ai")
    assert welcome.tiles["gemini"].chosen and not welcome.primary.isEnabled()  # a key first
    welcome.ai_key.setText("AIza-test-key")
    assert welcome.primary.text() == "Connect and finish" and welcome.primary.isEnabled()
    welcome.primary.click()
    assert _wait_until(lambda: app.settings.welcomed)
    assert ("check_ai", "gemini", "gemini-3.5-flash-lite") in app.calls  # the newest Flash-Lite, not Pro or an embedder
    assert app.settings.cleanup and app.settings.cleanup_model == "gemini-3.5-flash-lite"
    assert app.gateway.key_for("gemini") == "AIza-test-key" and window.current_page() == "home"


def test_the_fast_model_of_each_provider():
    assert w.fast_model("gemini", ["gemini-2.0-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-pro",
                                   "gemini-3.5-flash-lite-preview", "text-embedding-004"]) == "gemini-3.5-flash-lite"
    assert w.fast_model("openai", ["gpt-4o", "gpt-4o-mini", "gpt-4o-mini-tts", "gpt-4.1-mini"]) == "gpt-4.1-mini"
    assert w.fast_model("openai", ["gpt-4o", "o4-mini", "gpt-5.4-mini", "gpt-4o-mini"]) == "gpt-4o-mini"  # not "any mini"
    assert w.fast_model("anthropic", ["claude-sonnet-5-5", "claude-haiku-4-5"]) == "claude-haiku-4-5"
    assert w.fast_model("ollama", ["llama3.2"]) == "llama3.2" and w.fast_model("ollama", []) == ""


def test_the_fast_model_on_groq_is_gpt_oss_and_never_a_speech_model():
    # The research notes of 2026-10-06: llama-3.1-8b-instant was shut down, and the old fallback (the first model in
    # the list) would have picked a text-to-speech model.
    groq = ["canopylabs/orpheus-arabic-saudi", "canopylabs/orpheus-v1-english", "llama-prompt-guard-2-22m",
            "openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b", "whisper-large-v3",
            "whisper-large-v3-turbo"]
    assert w.fast_model("groq", groq) == "openai/gpt-oss-20b"
    assert w.fast_model("groq", [m for m in groq if m != "openai/gpt-oss-20b"]) == "openai/gpt-oss-120b"  # priced
    assert w.fast_model("groq", ["canopylabs/orpheus-v1-english", "whisper-large-v3-turbo", "playai-tts"]) == ""
    assert w.fast_model("gemini", ["gemini-3.8-flash", "gemini-3.6-flash"]) == "gemini-3.6-flash"  # the cheapest priced
    assert [key for key, _, _ in w.PROVIDER_TILES] == ["gemini", "ollama"]  # the providers offered, Gemini first


# ---- capture settings

def test_the_microphone_settings_apply_at_once_and_warn_about_bluetooth(monkeypatch):
    monkeypatch.setattr(w, "call_quality", lambda device: device == "Headset (Buds)")
    window, app = _window()
    page, models = window.pages["settings"], window.pages["models"]
    assert page.mic_ready.currentData() == "always" and not page.raw_audio.isChecked()  # always ready by default
    assert models.call_warning.isHidden()
    page.raw_audio.setChecked(True)
    page.mic_ready.setCurrentIndex(page.mic_ready.findData("off"))
    assert app.settings.raw_audio and not app.settings.warm_mic and not app.settings.always_on_mic
    page.mic_ready.setCurrentIndex(page.mic_ready.findData("warm"))
    assert app.settings.warm_mic and not app.settings.always_on_mic
    models.microphone.combo.addItem("Headset (Buds)", "Headset (Buds)")
    models.microphone.combo.setCurrentIndex(models.microphone.combo.findData("Headset (Buds)"))
    assert not models.call_warning.isHidden() and app.settings.microphone == "Headset (Buds)"


def test_the_reading_test_saves_what_came_after_record_not_the_lead_in(tmp_path):
    from sst.audio import Take

    class Warm(FakeRecorder):
        tail = 0.3

        def stop_later(self):
            self.stops += 1
            take = Take([np.full(int(0.4 * self.rate), 0.5, dtype=np.float32)], self.rate)  # lead-in
            take.chunks.append(np.zeros(int(0.3 * self.rate), dtype=np.float32))  # too short a reading
            take.done.set()
            return take

        def close(self):
            self.closed = True
    recorder = Warm()
    test = _reading_test(tmp_path, recorder=recorder)
    test.toggle_recording()
    test.toggle_recording()
    assert "too short" in test.status.text() and test.recorded() == 0
    test.stop()
    assert recorder.closed  # leaving the page closes even a warm microphone


# ---- speech recognition

def test_the_speech_page_shows_the_model_in_use_and_what_comes_next():
    window, _ = _window()
    window.show_page("speech")
    page = window.pages["speech"]
    assert page.where["local"].isChecked()
    parakeet, whisper = page.models["parakeet"], page.models["whisper-turbo"]
    assert "In use" in parakeet.status.text() and parakeet.choose.isHidden() and parakeet.download.isHidden()
    assert parakeet.remove.isHidden()  # it comes with Rflow
    assert not whisper.download.isHidden() and "1.6 GB" in whisper.download.text()  # not downloaded yet
    assert whisper.choose.isHidden()
    assert "In use: NVIDIA Parakeet" in page.in_use.title.text() and "hears English" in page.in_use.note.text()
    assert list(page.where) == ["local", "cloud"]  # two groups: on this PC, and the cloud
    page.where["cloud"].click()
    assert list(page.tiles) == ["gemini", "server"]  # OpenAI and Groq aren't offered

    def shown() -> list[str]:
        return [key for key in page.tiles if not page.models[key].frame.isHidden()]
    assert page.groups.currentIndex() == 1 and shown() == [] and not page.pick.isHidden()  # a tile first
    assert [page.tiles[key].state.text() for key in page.tiles] == ["Needs a key", "Needs an address"]
    assert all(tile.logo.width() == 28 for tile in page.tiles.values())  # room for each provider's logo
    page.tiles["gemini"].clicked.emit()
    assert page.tiles["gemini"].chosen and shown() == ["gemini"] and page.pick.isHidden()
    assert "Your voice is sent to Google Gemini" in _labels(page)
    page.tiles["server"].clicked.emit()
    assert shown() == ["server"] and not page.tiles["gemini"].chosen
    page.where["local"].click()
    assert page.groups.currentIndex() == 0 and page.where["local"].isChecked()


def test_a_cloud_model_asks_first_then_keeps_its_key_and_model():
    window, app = _window()
    page = window.pages["speech"]
    page.show_where("cloud")
    gemini = page.models["gemini"]
    assert gemini.choose.text() == "Use this model" and not gemini.choose.isHidden() and gemini.status.text() == ""
    gemini.choose.click()
    assert "Enter your Google Gemini API key" in gemini.result.text() and not app.calls  # no key: nothing happens
    gemini.key.setText("AIza-test")
    gemini.model_box.setCurrentText("gemini-3.6-flash")
    questions = []
    page.confirm = lambda question: questions.append(question) or False
    gemini.choose.click()
    assert "sent to Google Gemini" in questions[0] and not app.calls  # the user said no
    page.confirm = lambda question: True
    gemini.choose.click()
    assert ("use_cloud_speech", "gemini", "gemini-3.6-flash") in app.calls
    assert app.gateway.key_for("gemini") == "AIza-test" and app.settings.speech_model == "gemini"
    page.refresh()
    assert "In use" in gemini.status.text() and not gemini.choose.isEnabled()  # nothing to save
    assert "In use: Google Gemini" in page.in_use.title.text() and "gemini-3.6-flash" in page.in_use.detail.text()
    gemini.model_box.setCurrentText("gemini-3.5-flash-lite")
    assert gemini.choose.text() == "Save" and gemini.choose.isEnabled() and "Unsaved changes" in gemini.bar.state.text()
    questions.clear()
    page.confirm = lambda question: questions.append(question) or True
    gemini.choose.click()
    assert not questions and app.settings.speech_cloud_models["gemini"] == "gemini-3.5-flash-lite"  # asked once only


def test_a_speech_model_no_longer_offered_shows_as_gemini_or_parakeet():
    groq = Settings(welcomed=True, speech_model="groq")
    page = _window(settings=groq, gateway=GatewayConfig().with_key("groq", "gsk"))[0].pages["speech"]
    assert page.where["local"].isChecked() and "groq" not in page.tiles  # no Gemini key: Parakeet, on this PC
    page = _window(settings=groq, gateway=GatewayConfig().with_key("groq", "gsk").with_key("gemini", "AIza"))[0]
    page = page.pages["speech"]
    assert page.where["cloud"].isChecked() and page.tiles["gemini"].chosen


def test_testing_a_cloud_model_shows_its_answer():
    window, app = _window()
    page = window.pages["speech"]
    gemini = page.models["gemini"]
    gemini._test()
    assert "Enter your Google Gemini API key" in gemini.result.text()
    gemini.key.setText("AIza-test")
    gemini._test()
    for _ in range(200):
        QApplication.processEvents()
        if gemini.result.text().startswith("OK"):
            break
        time.sleep(0.01)
    assert gemini.result.text().startswith("OK: gemini-3.5-transcribe answered")
    assert ("test_cloud_speech", "gemini", "gemini-3.5-transcribe") in app.calls


def test_a_key_saved_on_one_page_shows_on_the_other():
    window, app = _window(gateway=GatewayConfig(provider="gemini", api_key="sk-old"))
    speech, cleanup = window.pages["speech"], window.pages["cleanup"]
    gemini = speech.models["gemini"]
    assert gemini.key.text() == "sk-old" and cleanup.api_key.text() == "sk-old"
    speech.confirm = lambda question: True
    gemini.key.setText("sk-new")
    gemini.choose.click()
    window.show_page("cleanup")
    assert cleanup.api_key.text() == "sk-new"
    _, _, _, gateway = cleanup.result()
    assert gateway.key_for("gemini") == "sk-new"
    cleanup.api_key.setText("sk-typed")  # being typed: a refresh leaves it alone
    app.gateway = app.gateway.with_key("gemini", "sk-other")
    cleanup.refresh()
    assert cleanup.api_key.text() == "sk-typed"


def test_saving_the_cleanup_keeps_a_key_saved_for_speech():
    window, app = _window(gateway=GatewayConfig("http://gw.example/v1", "gw-cleanup", "vllm"))
    cleanup = window.pages["cleanup"]
    app.gateway = app.gateway.with_key("gemini", "AIza-speech")  # saved on the speech page after this page was built
    _, _, _, gateway = cleanup.result()
    assert gateway.key_for("gemini") == "AIza-speech" and gateway.key_for("vllm") == "gw-cleanup"
    cleanup.provider.setCurrentIndex(cleanup.provider.findData("gemini"))
    assert cleanup.api_key.text() == "AIza-speech"  # the same key, ready for cleanup with Gemini


def test_choosing_a_downloaded_speech_model_and_its_language(whisper_downloaded):
    window, app = _window()
    page = window.pages["speech"]
    page.refresh()
    whisper = page.models["whisper-turbo"]
    assert whisper.download.isHidden() and not whisper.choose.isHidden() and whisper.status.text() == "Downloaded"
    assert not whisper.remove.isHidden()
    whisper.choose.click()
    assert ("choose_speech_model", "whisper-turbo") in app.calls
    app.loading_speech = "whisper-turbo"
    page.refresh()
    assert whisper.status.text() == "Loading..." and whisper.remove.isHidden()  # the chosen model can't be removed
    assert "Switching to OpenAI Whisper" in page.in_use.title.text()
    language = page.in_use.language
    language.setCurrentIndex(language.findData("ta"))
    assert not any(c[0] == "set_speech_language" for c in app.calls)  # chosen, not saved yet
    assert "Unsaved changes" in page.in_use.bar.state.text() and page.unsaved()
    page.in_use.bar.save.click()
    assert ("set_speech_language", "ta") in app.calls and app.settings.speech_language == "ta" and not page.unsaved()


def test_downloading_a_speech_model_shows_its_progress():
    window, app = _window()
    page = window.pages["speech"]
    whisper, parakeet = page.models["whisper-turbo"], page.models["parakeet"]
    whisper.download.click()
    assert ("download_speech_model", "whisper-turbo") in app.calls
    app.downloading = ("whisper-turbo", 400_000_000, 1_600_000_000)
    page.refresh()
    assert whisper.status.text().startswith("Downloading 25%") and whisper.progress.value() == 250
    assert whisper.download.isHidden() and not whisper.cancel.isHidden()
    assert "In use" in parakeet.status.text()  # dictation goes on with Parakeet meanwhile
    whisper.cancel.click()
    assert ("cancel_download",) in app.calls


def test_the_scan_card_runs_shows_progress_and_the_verdicts(tmp_path):
    from sst import scan
    window, app = _window()
    card = window.pages["speech"].scan
    card.refresh()
    assert card.button.isEnabled() and card.button.text() == "Scan this PC" and card.result.isHidden()
    card.button.click()
    assert ("scan_computer",) in app.calls
    app.scanning = "Trying NVIDIA Parakeet on a short sentence..."
    card.refresh()
    assert not card.button.isEnabled() and "Trying" in card.status.text()
    pc = scan.Computer(processor="Test CPU", cores=10, threads=12, memory_gb=32, free_memory_gb=16, free_disk_gb=200,
                       graphics=["Intel(R) Iris(R) Xe Graphics"], score=scan.REFERENCE_SCORE)
    local = [m for m in w.SPEECH_MODELS.values() if m.where == "local"]
    app.scanning, app.last_scan = "", scan.save(pc, scan.judge(pc, local, {"parakeet": 0.9}),
                                                 tmp_path / "scan.json")
    card.refresh()
    lines = [label.text() for label in card.result.findChildren(w.QLabel)]
    assert any("Test CPU" in line and "no NVIDIA card" in line for line in lines)
    assert any("NVIDIA Parakeet: Recommended. About 0.9 s" in line for line in lines)
    assert any("Slow on this computer" in line and "other languages" in line for line in lines)
    assert card.button.text() == "Scan again" and "Last scan" in card.status.text()


def test_the_server_card_starts_from_ai_cleanups_server_and_lists_its_speech_models():
    window, app = _window(gateway=GatewayConfig("http://gateway.example/v1", "gw-key", "vllm"))
    page = window.pages["speech"]
    page.show_where("server")
    server = page.models["server"]
    assert (server.address.text(), server.key.text()) == ("http://gateway.example/v1", "gw-key")
    assert "Filled in from your own server in Your API keys" in server.result.text()
    server.choose.click()
    assert "Choose a model first" in server.result.text() and not app.calls
    server._load_models()
    assert _wait_until(lambda: server.result.text().startswith("Loaded"))
    assert server.model_box.currentText() == "whisper-1" and "1 of them for speech" in server.result.text()
    server._test()
    assert _wait_until(lambda: server.result.text().startswith("OK"))
    assert ("test_server_speech", "http://gateway.example/v1", "whisper-1") in app.calls
    server.choose.click()  # no question: the user's own server
    assert ("use_server_speech", "http://gateway.example/v1", "whisper-1") in app.calls
    assert app.gateway.speech_server() == ("http://gateway.example/v1", "gw-key") and app.gateway.service.key == "vllm"
    page.refresh()
    assert "In use" in server.status.text() and not server.choose.isEnabled()
    server.address.setText("http://localhost:8000/v1")
    assert server.choose.text() == "Save" and server.choose.isEnabled()


def test_the_server_card_needs_an_address():
    window, app = _window()
    server = window.pages["speech"].models["server"]
    assert server.address.text() == "" and server.result.isHidden()
    server._load_models()
    assert "Enter your server's address first" in server.result.text() and not app.calls


def _wait_until(condition) -> bool:
    for _ in range(200):
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_a_new_install_chooses_its_speech_model_in_the_welcome(no_parakeet):
    window, app = _window(settings=Settings())
    welcome = window.pages["welcome"]
    window.set_status("Choose a speech model to start dictating", False)
    welcome.choose_setup("custom")  # each part by hand: how Rflow hears you first
    welcome.go_to_step("hear")
    assert welcome.current() == "hear" and welcome.local_option.chosen  # on this PC is the recommended choice
    assert welcome.get_parakeet.text() == "Download Parakeet and continue" and welcome.get_parakeet.isEnabled()
    welcome.get_parakeet.click()
    assert ("download_speech_model", "parakeet") in app.calls and welcome.current() == "try"  # tried while it downloads
    app.downloading = ("parakeet", 331_000_000, 663_043_117)
    welcome.refresh(False)
    assert welcome.orb.state == "loading" and "Downloading Parakeet, 49%" in welcome.orb_text.text()
    assert "Waiting for Parakeet" in welcome.status.text()
    welcome.go_to_step("hear")
    assert "Downloading Parakeet: 49%" in welcome.speech_status.text() and welcome.get_parakeet.text() == "Continue"


def test_the_welcome_can_lead_to_a_cloud_model_instead(no_parakeet):
    window, app = _window(settings=Settings())
    welcome = window.pages["welcome"]
    welcome.choose_setup("custom")
    welcome.go_to_step("hear")
    welcome.cloud_option.clicked.emit()
    assert welcome.cloud_option.chosen and welcome.primary.text() == "Set up a cloud model"
    welcome.primary.click()
    assert app.settings.welcomed and window.current_page() == "speech"
    assert window.pages["speech"].where["cloud"].isChecked()


def test_the_welcome_shows_the_speech_model_already_there():
    window, _ = _window(settings=Settings())
    welcome = window.pages["welcome"]
    welcome.refresh(True)
    assert "NVIDIA Parakeet" in welcome.speech_status.text() and welcome.get_parakeet.text() == "Continue"


def test_a_first_dictation_lights_the_orb():
    window, _ = _window(settings=Settings())
    welcome = window.pages["welcome"]
    welcome.go_to_step("try")
    welcome.refresh(True)
    welcome.try_box.setPlainText("Hello Rflow, this is my first dictation.")
    assert welcome.orb.state == "done" and welcome.status.text() == "It works. Now try it in any app."
    welcome.secondary.click()  # Try again
    assert welcome.try_box.toPlainText() == "" and welcome.orb.state == "ready"


def test_without_parakeet_the_pages_offer_it_and_promise_no_fallback(no_parakeet):
    window, _ = _window()
    page = window.pages["speech"]
    page.refresh()
    parakeet = page.models["parakeet"]
    assert parakeet.download.text() == "Download and use (663 MB)" and not parakeet.download.isHidden()
    assert parakeet.choose.isHidden() and parakeet.remove.isHidden() and "In use" not in parakeet.status.text()
    assert "Download Parakeet too" in page.models["gemini"].privacy.text()
    assert "Download Parakeet too" in page.models["server"].note.text()


# ---- settings that change only on purpose, with long lists searchable and the saved state shown (phase 23)

def test_the_mouse_wheel_never_changes_a_dropdown_anywhere():
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QComboBox
    window, _ = _window()
    list(window.pages.values())  # every page built (they are built when first shown)
    boxes = window.findChildren(QComboBox)
    assert boxes and all(isinstance(box, w.Choice) for box in boxes)  # every page, every dropdown
    box = window.pages["speech"].in_use.language
    before = box.currentIndex()
    wheel = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(box, wheel)
    assert box.currentIndex() == before and not wheel.isAccepted()  # left to the page, which scrolls


def test_a_long_list_has_a_search_box():
    from PySide6.QtTest import QTest
    window, _ = _window()
    language = window.pages["speech"].in_use.language
    language.showPopup()
    popup = language._popup
    assert popup.isVisible() and popup.search.hasFocus()
    QTest.keyClicks(popup.search, "zzz")
    assert not popup.empty.isHidden()  # "Nothing matches"
    popup.search.clear()
    QTest.keyClicks(popup.search, "tam")
    QTest.keyClick(popup.search, Qt.Key.Key_Return)
    assert language.currentText() == "Tamil" and not popup.isVisible()


def test_a_model_list_filters_by_what_is_typed():
    window, _ = _window()
    completer = window.pages["speech"].models["gemini"].model_box.completer()
    completer.setCompletionPrefix("LITE")  # in the middle of the name, any case
    found = [completer.completionModel().index(i, 0).data() for i in range(completer.completionCount())]
    assert found == ["gemini-3.5-flash-lite"]


def test_an_api_key_shows_masked_with_a_pen_to_change_it(monkeypatch):
    key = w.KeyField("sk-abcdefghijkl1234")
    assert key.view.text().endswith("1234") and "abcdefghijkl" not in key.view.text()
    assert not key.editing and not key.edited() and key.text() == "sk-abcdefghijkl1234"
    key.edit.click()
    assert key.editing and key.field.echoMode() == w.QLineEdit.EchoMode.Password and not key.edited()
    key.reveal.click()
    assert key.field.echoMode() == w.QLineEdit.EchoMode.Normal
    monkeypatch.setattr(w, "_clipboard_text", lambda: "  sk-pasted-key  ")
    key.paste.click()
    assert key.text() == "sk-pasted-key" and key.edited()
    key.undo.click()
    assert not key.editing and key.text() == "sk-abcdefghijkl1234" and not key.edited()
    empty = w.KeyField()
    assert empty.editing and empty.undo.isHidden()  # nothing saved: the box to paste one


def test_the_cleanup_page_shows_what_isnt_saved_and_cancel_undoes_it():
    window, app = _window(settings=Settings(welcomed=True, cleanup_model="model-a"),
                          gateway=GatewayConfig("", "sk-saved-key-0000", "gemini"))
    page = window.pages["cleanup"]
    assert not page.unsaved() and not page.bar.save.isEnabled() and page.bar.cancel.isHidden()
    page.cleanup_on.setChecked(True)
    assert page.unsaved() and page.bar.save.isEnabled() and "Unsaved changes" in page.bar.state.text()
    page.bar.cancel.click()
    assert not page.cleanup_on.isChecked() and not page.unsaved() and page.bar.state.text() == ""
    page.provider.setCurrentIndex(page.provider.findData("ollama"))
    page.provider.setCurrentIndex(page.provider.findData("gemini"))
    assert not page.unsaved()  # there and back again is no change
    page.model.setCurrentText("model-b")
    page.bar.save.click()
    assert app.settings.cleanup_model == "model-b" and not page.unsaved() and not page.bar.save.isEnabled()
    assert "Saved" in page.bar.state.text() and page.api_key.text() == "sk-saved-key-0000"


def test_a_cloud_card_cancel_brings_back_what_is_saved():
    window, _ = _window(gateway=GatewayConfig(provider="gemini", api_key="AIza-gemini-0000"))
    gemini = window.pages["speech"].models["gemini"]
    gemini.key.setText("AIza-typo")
    gemini.model_box.setCurrentText("gemini-3.6-flash")
    assert gemini.edited() and not gemini.bar.cancel.isHidden()
    gemini.bar.cancel.click()
    assert not gemini.edited() and gemini.key.text() == "AIza-gemini-0000"
    assert gemini.model_box.currentText() == "gemini-3.5-transcribe"


def test_leaving_a_page_with_unsaved_changes_asks_first():
    window, _ = _window()
    window.show_page("cleanup")
    page = window.pages["cleanup"]
    page.cleanup_on.setChecked(True)
    asked = []
    window.leave_unsaved = lambda title: asked.append(title) or False  # Stay
    window.nav["home"].click()
    assert asked == ["AI connection"] and window.current_page() == "cleanup" and window.nav["models"].isChecked()
    assert page.cleanup_on.isChecked()  # nothing lost
    window.leave_unsaved = lambda title: True  # Discard changes
    window.nav["home"].click()
    assert window.current_page() == "home" and not page.unsaved() and not page.cleanup_on.isChecked()


# ---- the voice pipeline's controls

def test_a_sound_alike_is_added_and_shown_with_its_word():
    window, app = _window(settings=Settings(welcomed=True, vocabulary=["PostgreSQL"]))
    window.show_page("dictionary")
    page = window.pages["dictionary"]
    assert page.alike_note.isHidden()
    page.heard.setText("post grass")
    page._add_sound_alike()
    assert "Fill in both" in page.alike_note.text()
    page.meant.setText("PostgreSQL")
    page._add_sound_alike()
    assert ("add_sound_alike", "post grass", "PostgreSQL") in app.calls
    assert "post grass" in _labels(page) and page.heard.text() == ""
    page.heard.setText("cube er net ease")
    page.meant.setText("Kubernetes")  # a word not in Your words yet: listed with its sound-alike
    page._add_sound_alike()
    assert "Kubernetes" in _labels(page) and "2 words" in page.count.text()


def test_a_correction_made_twice_becomes_a_suggestion_to_accept():
    window, app = _window(settings=Settings(welcomed=True))
    home = window.pages["home"]
    home.ask_correction = lambda typed: "I pushed it to GitHub today"
    for _ in range(2):
        home._correct("I pushed it to get hub today")
    page = window.pages["dictionary"]
    window.show_page("dictionary")
    assert not page.suggestions_card.isHidden() and "get hub" in _labels(page)
    assert window.nav["words"].badge == "1"  # the sidebar says there is something to look at
    _button(page.suggestions_card, "Learn it").click()
    assert page.suggestions_card.isHidden()
    assert any(t.preferred == "GitHub" and "get hub" in t.aliases for t in app.dictionary_terms())


def test_the_pipeline_switches_are_settings():
    window, app = _window(settings=Settings(welcomed=True))
    page = window.pages["settings"]
    assert page.mic_ready.currentData() == "always"
    page.mic_ready.setCurrentIndex(page.mic_ready.findData("warm"))
    assert app.settings.always_on_mic is False and app.settings.warm_mic is True
    page.debug_pipeline.setChecked(True)
    page.voice_pipeline.setChecked(False)
    s = app.settings
    assert (s.debug_pipeline, s.voice_pipeline) == (True, False)


# ---- Formatting

def test_formatting_is_a_section_with_the_switch_moved_from_settings():
    window, app = _window(settings=Settings(welcomed=True))
    assert not hasattr(window.pages["settings"], "format_text")
    window.nav["formatting"].click()
    page = window.pages["formatting"]
    assert window.current_page() == "formatting" and page.format_text.isChecked() and page.pipeline_off.isHidden()
    page.format_text.setChecked(False)
    assert app.settings.format_text is False and "Off" in page.switch_words.text()
    page.trial.setText("it costs five dollars")
    assert "Formatting is off" in page.trial_result.text()
    page.format_text.setChecked(True)
    assert app.settings.format_text is True and "$5" in page.trial_result.text()
    page.trial.setText("we have two options")
    assert "Nothing to change" in page.trial_result.text()


def test_the_formatting_examples_are_what_the_real_stage_types():
    expected = {"sales went up twenty five percent this quarter": "sales went up 25% this quarter",
                "the budget is twenty five thousand dollars": "the budget is $25,000",
                "it costs nine dollars ninety nine": "it costs $9.99",
                "let's meet at three thirty pm": "let's meet at 3:30 PM",
                "the launch is on october first twenty twenty six": "the launch is on October 1, 2026",
                "we need twenty five hundred copies": "we need 2,500 copies",
                "send it to john dot smith at gmail dot com": "send it to john.smith@gmail.com"}
    assert list(expected) == w.FORMAT_EXAMPLES
    for said, typed in expected.items():
        assert w.formatted(said)[0] == typed
    for said in w.FORMAT_KEPT:
        assert w.formatted(said)[0] == said  # prose stays as said
    window, _ = _window()
    page = window.pages["formatting"]
    page.refresh()
    shown = _shown(page.changed_box)
    assert all(w.formatted(said)[1] in shown for said in expected) and "25%</span>" in shown  # its changes in Iris


def test_formatting_says_when_the_voice_pipeline_is_off_and_turns_it_on():
    window, app = _window(settings=Settings(welcomed=True, voice_pipeline=False))
    window.show_page("formatting")
    page = window.pages["formatting"]
    assert not page.pipeline_off.isHidden()
    _button(page.pipeline_off, "Turn it on").click()
    assert app.settings.voice_pipeline and page.pipeline_off.isHidden()



def test_everyday_words_in_the_dictionary_are_marked():
    window, _ = _window(settings=Settings(welcomed=True, vocabulary=["GitHub", "move"]))
    window.show_page("dictionary")
    labels = _labels(window.pages["dictionary"])
    assert labels.count("everyday word") == 1



# ---- Snippets

def test_snippets_can_be_added_edited_tried_and_removed():
    window, app = _window(settings=Settings(welcomed=True))
    window.show_page("snippets")
    page = window.pages["snippets"]
    assert page.list_card.isHidden() and page.count.text() == "0 snippets"
    page.cue.setText("  my   email ")
    page.snippet_text.setPlainText("xyz@gmail.com\n")
    page.save_button.click()
    assert app.settings.snippets == [{"cue": "my email", "text": "xyz@gmail.com", "anywhere": False}]
    assert page.count.text() == "1 snippet" and page.cue.text() == "" and "say \u201cmy email\u201d" in page.note.text()
    page.cue.setText("My E-mail")  # the same cue, written differently
    page.snippet_text.setPlainText("other")
    page.save_button.click()
    assert len(app.settings.snippets) == 1 and "already a snippet" in page.note.text()
    page._edit(load(app.settings.snippets)[0])
    assert page.cue.text() == "my email" and page.save_button.text() == "Save" and not page.cancel_button.isHidden()
    page.snippet_text.setPlainText("Best regards,\nKarthi")
    page.anywhere.setChecked(True)
    page.save_button.click()
    assert app.settings.snippets == [{"cue": "my email", "text": "Best regards,\nKarthi", "anywhere": True}]
    page.trial.setText("send it to my email please")
    assert page.trial_result.text() == "Types: send it to Best regards,\nKarthi please"
    page.trial.setText("hello there")
    assert "No snippet here" in page.trial_result.text()
    page._remove(load(app.settings.snippets)[0])
    assert app.settings.snippets == [] and page.list_card.isHidden()


def test_a_snippet_needs_both_parts():
    window, app = _window(settings=Settings(welcomed=True))
    page = window.pages["snippets"]
    page.cue.setText("my email")
    page.save_button.click()
    assert app.settings.snippets == [] and "Fill in both" in page.note.text()


# ---- Translate

def test_the_translate_page_sets_the_shortcut_and_the_languages():
    window, app = _window(gateway=GatewayConfig(provider="openai", api_key="sk"),
                          settings=Settings(welcomed=True, cleanup_model="gpt-4o-mini"))
    window.show_page("translate")
    page = window.pages["translate"]
    assert page.shortcut.currentData() == "ctrl+c+c" and "Ctrl+C+C" in page.how.text()
    assert "Uses your AI connection: gpt-4o-mini" in page.model.text() and page.setup.isHidden()
    page.target.setCurrentText("Japanese")
    page.second.setCurrentIndex(page.second.findData("English"))
    assert (app.settings.translate_to, app.settings.translate_second) == ("Japanese", "English")
    assert "already in Japanese: in English" in page.how.text()
    page.shortcut.setCurrentIndex(page.shortcut.findData(""))
    assert app.settings.translate_shortcut == "" and "off" in page.how.text()


def test_translate_can_be_tried_on_the_page():
    window, app = _window(gateway=GatewayConfig(provider="openai", api_key="sk"),
                          settings=Settings(welcomed=True, cleanup_model="gpt-4o-mini", translate_to="Spanish"))
    page = window.pages["translate"]
    page.refresh()
    page.try_button.click()
    for _ in range(200):
        QApplication.processEvents()
        if not page.result.isHidden():
            break
        time.sleep(0.01)
    assert "informe" in page.result.toPlainText() and ("run_translation", "Spanish") in app.calls


def test_translate_without_an_ai_model_points_to_the_setup():
    window, _ = _window()
    window.show_page("translate")
    page = window.pages["translate"]
    assert not page.setup.isHidden() and not page.try_button.isEnabled()


# ---- Text Transform

def test_the_text_transform_page_sets_the_shortcut_and_the_menu():
    window, app = _window(gateway=GatewayConfig(provider="openai", api_key="sk"),
                          settings=Settings(welcomed=True, cleanup_model="gpt-4o-mini"))
    window.show_page("transform")
    page = window.pages["transform"]
    assert "Uses your AI connection: gpt-4o-mini" in page.model.text() and page.setup.isHidden()
    assert "double-tap Ctrl" in page.how.text() and page.hotkey.currentText() == "Double-tap Ctrl"
    page.choices["rewrite"].setChecked(True)
    page.choices["professional"].setChecked(False)
    assert app.settings.transforms == ["concise", "rewrite", "bullets", "actions"]  # as the page groups them
    page.choices["translate"].setChecked(True)
    assert "then 1-4 or T or a click" in page.how.text() and _keys(page.steps)[-5:] == ["1", "2", "3", "4", "T"]
    page.hotkey.setCurrentIndex(page.hotkey.findData(""))
    assert app.settings.transform_shortcut == "" and "off" in page.how.text()


def test_the_name_for_japanese_emails_is_saved():
    window, app = _window(settings=Settings(welcomed=True))
    window.show_page("transform")
    page = window.pages["transform"]
    assert page.signature_name.text() == "" and page.signature_company.text() == ""
    page.signature_name.setText("  山田   太郎 ")
    page.signature_company.setText("株式会社ABC")
    page.signature_company.editingFinished.emit()
    assert (app.settings.signature_name, app.settings.signature_company) == ("山田 太郎", "株式会社ABC")


def test_the_examples_are_grouped_tone_format_language():
    window, _ = _window()
    page = window.pages["transform"]
    tabs = page.example_tabs
    assert [list(row.buttons) for row in tabs.rows] == [["concise", "professional", "rewrite", "grammar"],
                                                        ["bullets", "actions", "teams", "email_internal", "email_external"],
                                                        ["translate"]]
    tabs.buttons["teams"].click()
    assert page.example_key == "teams" and tabs.buttons["teams"].isChecked()
    assert not any(b.isChecked() for k, b in tabs.buttons.items() if k != "teams")


def test_the_voice_command_phrases_can_be_edited():
    window, app = _window(settings=Settings(welcomed=True))
    window.show_page("transform")
    page = window.pages["transform"]
    assert page.phrases["concise"].text() == ", ".join(DEFAULT_PHRASES["concise"]) and page.reset.isHidden()
    assert set(page.phrases) == {"concise", "professional", "bullets", "actions", "rewrite", "grammar", "teams",
                                 "email_internal", "email_external", "translate", "undo"}
    page.phrases["concise"].setText("trim it,  tighten this up,")
    page.phrases["concise"].editingFinished.emit()
    assert app.settings.command_phrases == {"concise": "trim it, tighten this up"}
    assert page.phrases["concise"].text() == "trim it, tighten this up" and not page.reset.isHidden()
    page.phrases["rewrite"].setText("rewrite it, trim it")
    page.phrases["rewrite"].editingFinished.emit()
    assert not page.phrase_note.isHidden() and "is in Concise and Rewrite" in page.phrase_note.text()
    page.phrases["concise"].setText(", ".join(DEFAULT_PHRASES["concise"]))  # back to the defaults: not kept as custom
    page.phrases["concise"].editingFinished.emit()
    assert app.settings.command_phrases == {"rewrite": "rewrite it, trim it"} and page.phrase_note.isHidden()
    page.phrases["undo"].setText("")  # no phrase: that command is off
    page.phrases["undo"].editingFinished.emit()
    assert app.settings.command_phrases["undo"] == ""
    page.reset.click()
    assert app.settings.command_phrases == {} and page.phrases["undo"].text().startswith("undo")
    page.voice.setChecked(False)
    assert not app.settings.voice_commands and not page.phrases["concise"].isEnabled()


def test_text_transform_can_be_tried_on_the_page():
    window, app = _window(gateway=GatewayConfig(provider="openai", api_key="sk"),
                          settings=Settings(welcomed=True, cleanup_model="gpt-4o-mini"))
    page = window.pages["transform"]
    page.refresh()
    page.try_buttons["concise"].click()
    for _ in range(200):
        QApplication.processEvents()
        if not page.result.isHidden():
            break
        time.sleep(0.01)
    assert "database migration issue" in page.result.toPlainText() and ("run_transform", "concise") in app.calls


def test_text_transform_without_an_ai_model_points_to_the_setup():
    window, _ = _window()
    window.show_page("transform")
    page = window.pages["transform"]
    assert not page.setup.isHidden() and not page.try_buttons["concise"].isEnabled()


# ---- AI & models, Tools

def test_ai_and_models_shows_both_halves_and_switches_the_cleanup():
    window, app = _window(gateway=GatewayConfig(provider="gemini", api_key="AIza-key-0000-9f3c"),
                          settings=Settings(welcomed=True, cleanup=True, cleanup_model="gemini-3.5-flash-lite"))
    page = window.pages["models"]
    page.refresh()
    assert page.speech_name.text() == "NVIDIA Parakeet" and page.ai_name.text() == "Google Gemini"
    assert page.ai_model.text() == "gemini-3.5-flash-lite" and "Google Gemini key" in page.key_line.text()
    assert page.keys.rows["gemini"].value.text().endswith("9f3c")  # the key itself is in Your API keys, masked
    assert page.cleanup.isChecked() and page.ai_state.text() == "Connected"
    page.cleanup.setChecked(False)
    assert app.calls[-1][:2] == ("save_cleanup", False) and not app.settings.cleanup
    page.test_button.click()
    assert _wait_until(lambda: page.ai_state.text().startswith("Connected ·"))
    assert ("check_ai", "gemini", "gemini-3.5-flash-lite") in app.calls


def test_the_cost_card_shows_today_this_month_and_each_feature():
    window, app = _window(gateway=GatewayConfig(provider="gemini", api_key="AIza-key-0000-9f3c"))
    today = date.today()
    app.usage.record("dictation", "gemini-3.5-transcribe", audio_in=500_000, seconds=1500)  # $1.00, 25 minutes
    app.usage.record("cleanup", "gemini-3.5-flash-lite", text_in=100_000, out=20_000)  # $0.03 + $0.05
    if today.day > 1:  # earlier this month: in the month, not today
        app.usage.days[today.replace(day=1).isoformat()] = {"translate|gemini-3.5-flash-lite": dict(
            dict.fromkeys(("audio_in", "text_in", "out", "thoughts", "audio_out", "seconds"), 0.0), calls=3.0,
            text_in=1_000_000.0)}
    window.show_page("models")
    card = window.pages["models"].costs
    assert card.frame.isVisibleTo(window.pages["models"]) and card.today.text() == "$1.08"
    assert card.month.text() == ("$1.38" if today.day > 1 else "$1.08")
    shown = _shown(card.frame)
    assert "What Gemini costs (estimate)" in shown and "Dictation | 25 min | $1.00" in shown
    assert "AI cleanup | 1 call | $0.08" in shown and f"as of {w.nice_day(w.costs.SEEN)}" in shown
    assert ("Translate | 3 calls | $0.30" in shown) == (today.day > 1)
    assert _button(card.frame, "Billing in AI Studio")
    # A call counted while the page is shown: the card follows
    app.usage.record("live_translation", "gemini-3.5-live-translate-preview", audio_in=1500, audio_out=1500, seconds=60,
                     calls=0)
    window.usage_changed()
    assert "Live translation | 1 min | $0.04" in _shown(card.frame)
    assert card.saved.isHidden()  # nothing kept from being sent: no line for the cost saver
    app.usage.record("live_translation", "gemini-3.5-live-translate-preview", seconds=600, calls=0, unsent=600)
    window.usage_changed()
    assert card.saved.text().startswith("Saved by the cost saver this month: $") and not card.saved.isHidden()


def test_the_cost_card_waits_for_a_gemini_key_and_says_a_free_key_is_billed_nothing():
    window, app = _window()
    window.show_page("models")
    card = window.pages["models"].costs
    assert not card.frame.isVisibleTo(window.pages["models"])  # no Gemini key, nothing counted
    app.gateway = GatewayConfig(provider="gemini", api_key="AIza-key")
    window.refresh()
    assert card.frame.isVisibleTo(window.pages["models"]) and card.today.text() == "$0.00"
    assert "Nothing counted this month yet" in _shown(card.frame)
    app.usage.record("dictation", "gemini-3.5-transcribe", audio_in=500_000, seconds=60)
    app.settings.gemini_free_tier = True
    window.refresh()
    assert card.month.text() == "$0.00" and "$0 billed: a free key (would cost $1.00 this month)." in _shown(card.frame)
    assert w.dollars(0.004) == "<$0.01" and w.dollars(0) == "$0.00" and w.dollars(12.4) == "$12"


def test_a_key_added_on_ai_and_models_changes_nothing_else():
    # The user testing's M-04: a Gemini key added for live translation switched AI cleanup to Gemini, with no model.
    window, app = _window(gateway=GatewayConfig("http://gw.example/v1", "gw-key-0000", "vllm"),
                          settings=Settings(welcomed=True, cleanup=True, cleanup_model="qwen3"))
    window.show_page("models")
    page = window.pages["models"]
    keys = page.keys
    server, gemini = keys.rows["vllm"], keys.rows["gemini"]
    assert list(keys.rows) == ["gemini", "vllm", w.SPEECH_SERVER]  # OpenAI, Anthropic and Groq aren't offered
    assert set(keys.add_buttons) == {"gemini", "vllm"}
    assert not server.isHidden() and "gw-key" not in server.value.text()
    assert server.uses.text() == "Used for AI cleanup" and keys.add_buttons["vllm"].isHidden()
    assert gemini.isHidden() and not keys.add_buttons["gemini"].isHidden() and app.live_problem()
    keys.add_buttons["gemini"].click()
    assert gemini.editing and not gemini.isHidden() and not gemini.bar.save.isEnabled()  # nothing typed yet
    assert "AI cleanup keeps vLLM" in gemini.note.text() and gemini.remove.isHidden()
    gemini.key.setText("AIza-test-key-41ab")
    assert gemini.bar.save.isEnabled() and page.unsaved()
    gemini.bar.save.click()
    assert ("save_key", "gemini") in app.calls and app.gateway.key_for("gemini") == "AIza-test-key-41ab"
    assert (app.gateway.chosen, app.gateway.api_key, app.settings.cleanup_model) == ("vllm", "gw-key-0000", "qwen3")
    assert not gemini.editing and gemini.value.text().endswith("41ab") and keys.add_buttons["gemini"].isHidden()
    assert gemini.uses.text() == "Used for live translation" and "AI cleanup keeps vLLM" in keys.state.text()
    assert not app.live_problem() and not page.unsaved()


def test_a_key_is_changed_or_removed_in_your_api_keys():
    window, app = _window(gateway=GatewayConfig("", "sk-openai-key-0000", "gemini"),
                          settings=Settings(welcomed=True, cleanup=True, cleanup_model="gemini-3.5-flash-lite"))
    window.show_page("models")
    page = window.pages["models"]
    openai = page.keys.rows["gemini"]
    openai.edit.click()
    assert openai.editing and openai.key.editing and not openai.remove.isHidden() and not openai.bar.save.isEnabled()
    openai.key.setText("sk-typo")
    asked = []
    window.leave_unsaved = lambda title: asked.append(title) or False  # Stay
    window.nav["home"].click()
    assert asked == ["AI & models"] and window.current_page() == "models"
    openai.bar.cancel.click()
    assert not openai.editing and app.gateway.api_key == "sk-openai-key-0000" and not page.unsaved()
    openai.edit.click()
    openai.key.setText("sk-new-key-1111")
    openai.bar.save.click()
    assert app.gateway.api_key == "sk-new-key-1111" and openai.value.text().endswith("1111")
    questions = []
    page.confirm = lambda question: questions.append(question) or True
    openai.edit.click()
    openai.remove.click()
    assert "AI cleanup stops" in questions[0] and app.gateway.key_for("gemini") == ""
    assert app.gateway.chosen == "gemini" and openai.isHidden() and not page.keys.add_buttons["gemini"].isHidden()
    assert not page.keys.empty.isHidden()  # no keys at all now
    page.add_key("gemini")  # e.g. from live translation's "Add a Gemini key"
    assert page.keys.rows["gemini"].editing and not page.keys.rows["gemini"].isHidden()


def test_your_own_server_is_one_line_for_ai_cleanup_and_speech():
    from sst.gateway import SPEECH_SERVER
    gateway = GatewayConfig("http://gateway.example/v1", "gw-key", "vllm").with_entry(
        SPEECH_SERVER, "http://gateway.example/v1", "gw-key")
    window, app = _window(gateway=gateway, settings=Settings(welcomed=True, cleanup_model="m", speech_model="server",
                                                             speech_server_model="whisper-1"))
    page = window.pages["models"]
    page.refresh()
    server = page.keys.rows["vllm"]
    assert not server.isHidden() and page.keys.rows[SPEECH_SERVER].isHidden() and page.keys.add_buttons["vllm"].isHidden()
    assert server.value.text().startswith("http://gateway.example/v1") and "gw-key" not in server.value.text()
    assert server.uses.text() == "Used for AI cleanup and speech"
    server.edit.click()
    server.address.setText("http://localhost:8000/v1")
    server.bar.save.click()
    assert ("save_server", "http://localhost:8000/v1", ("vllm", SPEECH_SERVER)) in app.calls
    assert app.gateway.base_url == app.gateway.speech_server()[0] == "http://localhost:8000/v1"


def test_a_cloud_tile_takes_its_key_from_your_api_keys():
    window, app = _window()
    keys, speech = window.pages["models"].keys, window.pages["speech"]
    keys.add_buttons["gemini"].click()
    keys.rows["gemini"].key.setText("AIza-test-key-0000")
    keys.rows["gemini"].bar.save.click()
    assert app.gateway.chosen == "" and app.settings.speech_model == "parakeet"  # nothing chosen by saving a key
    window.show_page("speech")
    speech.show_where("gemini")
    gemini = speech.models["gemini"]
    assert speech.tiles["gemini"].chosen and speech.tiles["gemini"].state.text() == "Key saved"
    assert gemini.key.text() == "AIza-test-key-0000" and "From Your API keys" in gemini.key_note.text()


def test_switching_the_cleanup_s_provider_needs_a_model_first():
    # M-04 on the AI connection page: choosing Gemini there only to save its key must not leave the cleanup without a model.
    window, app = _window(gateway=GatewayConfig("http://gw.example/v1", "gw", "vllm"),
                          settings=Settings(welcomed=True, cleanup=True, cleanup_model="qwen3"))
    page = window.pages["cleanup"]
    page.provider.setCurrentIndex(page.provider.findData("gemini"))
    page.api_key.setText("AIza-live")
    page.bar.save.click()
    assert not any(call[0] == "save_cleanup" for call in app.calls) and "Your API keys" in page.test_result.text()
    assert (app.gateway.chosen, app.settings.cleanup_model) == ("vllm", "qwen3")
    page.model.setCurrentText("gemini-3.5-flash-lite")
    page.bar.save.click()
    assert app.gateway.chosen == "gemini" and app.settings.cleanup_model == "gemini-3.5-flash-lite"


def test_a_long_microphone_name_is_cut_short_and_whole_in_its_tooltip():
    # The user testing's M-03: the owner's own laptop microphone made AI & models 1418 px wide in a 696 px view.
    long = "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)"
    box = w.MicrophoneBox(long, [long, "Headset (Buds)"], long)
    assert box.minimumSizeHint().width() < 300 and box.combo.sizeHint().width() < 300  # not the name's width
    assert box.combo.currentText() == long and box.combo.toolTip() == long  # cut with … in the box, whole in its tooltip
    assert box.combo.itemData(box.combo.findData(long), Qt.ItemDataRole.ToolTipRole) == long


def test_a_long_microphone_name_never_makes_ai_and_models_wider_than_the_window():
    # The user testing's M-03: the owner's own laptop microphone made the page 1418 px wide in a 696 px view.
    long = "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)"
    window, _ = _window(microphones=[long, "Headset (Buds)"], settings=Settings(welcomed=True, microphone=long))
    window.resize(780, 540)
    window.show_page("models")
    page = window.pages["models"]
    window.grab()  # lays the page out
    assert page.widget().width() <= page.viewport().width()
    box = page.microphone.combo
    assert box.currentText() == long and box.toolTip() == long  # cut with … in the box, whole in its tooltip
    assert box.itemData(box.findData(long), Qt.ItemDataRole.ToolTipRole) == long
    assert page.pair.direction() == w.QBoxLayout.Direction.TopToBottom  # too narrow for the two cards side by side


def test_ai_and_models_without_a_connection_offers_one():
    window, _ = _window()
    page = window.pages["models"]
    page.refresh()
    assert page.ai_name.text() == "Not connected" and page.ai_change.text() == "Connect"
    assert not page.cleanup.isEnabled()
    page.ai_change.click()
    assert window.current_page() == "cleanup"


def test_text_transform_and_translate_are_sections_with_the_tools_switches():
    window, app = _window(gateway=GatewayConfig(provider="openai", api_key="sk"),
                          settings=Settings(welcomed=True, cleanup_model="gpt-4o-mini"))
    assert "tools" not in window.pages and "tools" not in dict(w.NAV)
    window.nav["translate"].click()
    page = window.pages["translate"]
    assert window.current_page() == "translate" and page.translate_on.isChecked() and page.setup.isHidden()
    page.translate_on.setChecked(False)
    assert app.settings.translate_shortcut == "" and page.steps.isHidden() and "Off" in page.switch_words.text()
    page.translate_on.setChecked(True)
    assert app.settings.translate_shortcut == "ctrl+c+c" and not page.steps.isHidden()
    window.nav["transform"].click()
    page = window.pages["transform"]
    assert window.current_page() == "transform" and page.transform_on.isChecked()
    page.transform_on.setChecked(False)
    assert app.settings.transform_shortcut == "" and not app.settings.voice_commands and page.steps.isHidden()
    page.transform_on.setChecked(True)
    assert (app.settings.transform_shortcut, app.settings.voice_commands) == ("double ctrl", True)
    window.show_page("tools")  # a link to the old Tools section (another page, the tray menu) opens Text Transform
    assert window.current_page() == "transform" and window.nav["transform"].isChecked()


def test_text_transform_says_how_to_use_it_with_the_users_own_keys_and_phrases():
    window, app = _window(settings=Settings(welcomed=True, hotkey="menu", command_phrases={"concise": "trim it"},
                                            transforms=["bullets", "concise"]))
    window.show_page("transform")
    page = window.pages["transform"]
    steps = _shown(page.steps)
    assert "and say “trim it”" in steps and "Or double-tap" in steps and "Bullet points" in steps
    assert _keys(page.steps) == ["Menu", "Ctrl", "1", "2"]  # the menu: as it is numbered
    app.apply_settings(dataclasses.replace(app.settings, voice_commands=False))
    page.refresh()
    assert "trim it" not in _shown(page.steps) and "Then double-tap" in _shown(page.steps)


def test_text_transform_shows_examples_of_each_transform():
    window, _ = _window()
    page = window.pages["transform"]
    page.refresh()
    for key, transform in TRANSFORMS.items():
        page.example_tabs.buttons[key].click()
        shown = page.findChildren(w.Example)
        assert len([e for e in shown if e.isVisibleTo(page)]) == len(w.TRANSFORM_EXAMPLES[key]) >= 2
        assert page.example_words.text() == transform.description


@pytest.mark.parametrize("key, before, after", [(key, before, after) for key, pairs in w.TRANSFORM_EXAMPLES.items()
                                                for before, after in pairs])
def test_every_text_transform_example_is_one_rflow_would_accept(key, before, after):
    got = TransformGuard().validate(before, after, key)  # the real check: nothing invented, nothing lost
    assert got.accepted, got.reasons


def test_translate_says_how_to_use_it_and_shows_examples():
    window, app = _window(settings=Settings(welcomed=True, translate_to="German", translate_second="English"))
    window.show_page("translate")
    page = window.pages["translate"]
    steps = _shown(page.steps)
    assert "shows it in German" in steps and "already in German goes into English" in steps
    assert _keys(page.steps) == ["Ctrl", "C", "C", "C", "↵"]
    assert len([e for e in page.findChildren(w.Example) if e.isVisibleTo(page)]) == len(w.TRANSLATE_EXAMPLES)
    page.shortcut.setCurrentIndex(page.shortcut.findData("ctrl+alt+l"))
    assert app.settings.translate_shortcut == "ctrl+alt+l" and "to translate it." in _shown(page.steps)


def test_text_transform_and_translate_without_an_ai_connection_say_so():
    window, _ = _window()
    for key in ("transform", "translate"):
        page = window.pages[key]
        page.refresh()
        assert not page.setup.isHidden() and "needs an AI connection" in page.model.text()


def test_the_smallest_window_folds_the_sidebar_into_a_rail():
    window, _ = _window()
    window.resize(780, 540)
    window._set_compact(True)
    assert window.sidebar.width() == 84 and window.wordmark.isHidden() and window.nav["models"].text() == "AI"
    assert not window.grab().isNull()
    window._set_compact(False)
    assert window.nav["models"].text() == "AI & models"


def test_the_sidebar_has_the_eleven_sections_in_order():
    window, _ = _window()
    assert [key for key, _ in w.NAV] == ["home", "live", "meeting", "words", "snippets", "transform", "translate", "formatting",
                                         "models", "settings", "report"]
    assert all(key in w.NAV_ICONS and w.NAV_ICONS[key] in w.theme.ICONS for key in window.nav)


@pytest.mark.parametrize("size", [(780, 540), (1000, 540), (1000, 700)])
def test_every_section_fits_the_smallest_window_in_the_sidebar_and_the_rail(size):
    profiles = w.Profiles()
    profiles.add("Rahul")  # two profiles: the profile button shows too
    window, _ = _window(profiles=profiles)
    window.show_update("Rflow 9.9.9 is available (you have 1.0.0).", version="9.9.9")
    window.resize(*size)
    assert not window.grab().isNull()
    buttons = [window.nav[key] for key, _ in w.NAV]
    tops = [b.geometry().top() for b in buttons]
    assert tops == sorted(tops)
    assert all(a.geometry().bottom() < b.geometry().top() for a, b in zip(buttons, buttons[1:], strict=False))  # no overlap
    below = [x for x in (window.update_link, window.status_card, window.profile_button) if x.isVisibleTo(window)]
    assert window.status_card in below and all(buttons[-1].geometry().bottom() < x.geometry().top() for x in below)
    assert max(x.geometry().bottom() for x in below) < window.sidebar.height()  # nothing cut off at the bottom
    assert min(b.height() for b in buttons) >= w.NAV_HEIGHTS[window.compact][1]
    if window.compact:  # the rail: each short name fits under its icon
        metrics = QFontMetrics(w.font(11, 600))
        assert all(metrics.horizontalAdvance(b.text()) <= b.width() - 4 for b in buttons)
    else:
        assert window.update_link.isVisibleTo(window) and window.profile_button.isVisibleTo(window)


# ---- Report a problem

def test_report_a_problem_opens_an_issue_and_copies_only_the_version_info(monkeypatch):
    opened, copied, folders = [], [], []
    monkeypatch.setattr(w, "open_link", opened.append)
    monkeypatch.setattr(w, "copy_text", copied.append)
    monkeypatch.setattr(w, "open_folder", folders.append)
    window, app = _window(settings=Settings(welcomed=True, vocabulary=["Priya"],
                                            snippets=[{"cue": "my email", "text": "alex@example.com"}]))
    window.nav["report"].click()
    page = window.pages["report"]
    assert window.current_page() == "report" and page.title.text() == "Report a problem"
    page.issue_button.click()
    assert opened == [w.REPO + "/issues/new/choose"]
    page.copy_button.click()
    lines = copied[0].splitlines()
    assert lines[0].startswith(f"Rflow {w.__version__}") and lines[1].startswith("Windows") and "x64" in lines[2]
    assert "Priya" not in copied[0] and "alex@example.com" not in copied[0] and page.copy_button.text() == "Copied"
    assert "can contain text you dictated" in page.privacy.text()
    page.logs_button.click()
    assert folders == [w.LOG_DIR]


def test_settings_name_the_key_plainly_and_list_the_reading_test():
    window, _ = _window()
    page = window.pages["settings"]
    assert page.hotkey.currentText() == "Ctrl+Win" and "Wispr" not in page.hotkey.itemText(0)
    assert not page.advanced_card.isAncestorOf(page.reading) and page.reading.isVisibleTo(page)  # not under Advanced
    page.reading.click()
    assert window.current_page() == "reading" and window.nav["settings"].isChecked()


def test_start_over_asks_first_and_can_keep_the_speech_models():
    window, app = _window()
    page = window.pages["settings"]
    layout = page.body
    cards = [layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget() is not None]
    assert cards[-1].isAncestorOf(page.start_over)  # at the very end
    assert page.start_over.kind == "danger" and page.start_over.text().startswith("Start over")
    page.confirm_start_over = lambda: None  # cancelled
    page.start_over.click()
    assert not any(call[0] == "start_over" for call in app.calls)
    page.confirm_start_over = lambda: True  # "Keep the downloaded speech models" ticked
    page.start_over.click()
    page.confirm_start_over = lambda: False
    page.start_over.click()
    assert [call for call in app.calls if call[0] == "start_over"] == [("start_over", True), ("start_over", False)]


def test_the_start_over_dialog_says_plainly_what_is_deleted(monkeypatch):
    seen = {}

    def answer(box):  # in place of showing it: read it, untick, press the red button
        seen["words"] = f"{box.text()} {box.informativeText()}"
        seen["keep"] = box.checkBox().isChecked() and box.checkBox().text() == "Keep the downloaded speech models"
        box.checkBox().setChecked(False)
        next(b for b in box.buttons() if b.text() == "Delete everything and restart").click()
        return 0
    monkeypatch.setattr(w.QMessageBox, "exec", answer)
    window, _ = _window()
    assert window.pages["settings"].confirm_start_over() is False  # go on, without keeping the models
    assert seen["keep"]  # ticked at first
    assert all(word in seen["words"] for word in ("every profile", "settings", "words and snippets", "history", "stats",
                                                  "API keys", "recordings", "reading tests", "live translation", "logs",
                                                  "welcome"))


def test_report_a_problem_left_settings_advanced():
    window, _ = _window()
    advanced = window.pages["settings"].advanced_card
    names = [b.accessibleName() for b in advanced.findChildren(QPushButton) if b.accessibleName()]
    assert "Report a problem" not in names and "Source code" in names and "Profiles" in names


# ---- the microphone list follows Windows (phase 26)

def test_the_microphone_list_follows_windows_and_keeps_the_choice():
    laptop, headset = "Microphone (Realtek(R) Audio)", "Headset (Buds)"
    windows = {"list": ([laptop], laptop)}
    box = w.MicrophoneBox("", [laptop], laptop, source=lambda: windows["list"])
    followed = []
    box.followed.connect(lambda: followed.append(True))
    assert box.combo.itemText(0) == f"Windows default (now: {laptop})" and box.device() == ""
    windows["list"] = ([headset, laptop], headset)  # a headset plugged in: Windows makes it the default
    box._follow()
    assert [box.combo.itemText(i) for i in range(box.combo.count())] == [f"Windows default (now: {headset})", headset,
                                                                         laptop]
    assert box.device() == "" and followed == [True]  # still "Windows default", which now means the headset
    box._follow()
    assert followed == [True]  # nothing changed: nothing redone


def test_a_chosen_microphone_that_is_unplugged_stays_chosen_and_says_what_records():
    laptop, headset = "Microphone (Realtek(R) Audio)", "Headset (Buds)"
    windows = {"list": ([headset, laptop], headset)}
    box = w.MicrophoneBox(headset, [headset, laptop], headset, source=lambda: windows["list"])
    chosen = []
    box.changed.connect(chosen.append)
    assert box.device() == headset and box.note.isHidden()
    windows["list"] = ([laptop], laptop)  # unplugged
    box._follow()
    assert box.device() == headset and box.combo.currentText() == f"{headset} (not connected)"
    assert not box.note.isHidden() and box.note.text() == f"Not connected now: Rflow uses {laptop} until it is."
    assert chosen == []  # following Windows never changes the setting
    windows["list"] = ([headset, laptop], headset)  # back
    box._follow()
    assert box.combo.currentText() == headset and box.note.isHidden()


# ---- live translation (sst.live): a section of its own

def _live_window(**settings):
    return _window(gateway=GatewayConfig(provider="gemini", api_key="AIza-test"), settings=Settings(welcomed=True, **settings))


def test_live_translation_is_a_section_of_its_own_and_starts_and_stops_there():
    window, app = _live_window()
    assert ("live", "Live translation") in w.NAV and "tools" not in window.pages
    window.show_page("live")
    page = window.pages["live"]
    assert page.state.text() == "Off" and page.start_button.text() == "Start" and page.connect_card.isHidden()
    assert page.doing.text() == "Ready to translate what the computer plays into English" and not page.light.blinking
    assert not page.start_button.running and not page.start_card.grab().isNull()  # green, with its glow
    page.start_button.click()
    assert ("start_live", "computer") in app.calls and page.start_button.text() == "Stop" and page.start_button.running
    assert page.state.text() == "Live" and page.light.state == "live" and page.light.blinking
    assert page.doing.text() == "Translating what the computer plays into English"
    assert not page.start_card.grab().isNull()  # red
    page.start_button.click()
    assert ("stop_live",) in app.calls and page.start_button.text() == "Start" and page.state.text() == "Off"
    assert not page.start_button.running and not page.light.blinking


def test_the_page_follows_live_translation_started_or_stopped_elsewhere():
    window, app = _live_window()
    window.show_page("live")
    page = window.pages["live"]
    app._live = True  # started with the shortcut or from the tray menu
    window.refresh()  # what the TrayApp does when live translation starts or stops
    assert page.start_button.text() == "Stop" and page.state.text() == "Live" and page.light.blinking
    app._live = False  # stopped with the bar's ✕
    window.refresh()
    assert page.start_button.text() == "Start" and page.state.text() == "Off" and not page.light.blinking


def test_the_page_says_why_the_bar_is_still_empty():
    window, app = _live_window()
    window.show_page("live")
    page = window.pages["live"]
    app._live_note = "Nothing heard yet: is the sound playing on this laptop?"
    window.refresh()
    assert page.hearing.isHidden()  # not running: nothing to say
    app._live = True
    window.refresh()  # what the TrayApp does when the bar's status line changes
    assert page.hearing.text() == app._live_note and not page.hearing.isHidden()
    app._live_note = ""  # words came
    window.refresh()
    assert page.hearing.isHidden()


def test_the_live_light_blinks_only_while_it_runs_and_is_shown():
    from PySide6.QtGui import QHideEvent, QShowEvent
    window, _ = _live_window()
    window.show_page("live")
    light = window.pages["live"].light
    window.pages["live"].start_button.click()
    assert light.blinking and not light._timer.isActive()  # the window isn't on the screen: nothing ticks
    QApplication.sendEvent(light, QShowEvent())  # shown
    assert light._timer.isActive() and light._timer.interval() == 500  # lit half a second, dimmed the other half
    light._tick()
    assert not light.lit
    QApplication.sendEvent(light, QHideEvent())  # another page, or the window closed or minimised
    assert not light._timer.isActive()
    QApplication.sendEvent(light, QShowEvent())
    assert light._timer.isActive() and light.lit
    window.pages["live"].start_button.click()  # stopped
    assert not light.blinking and not light._timer.isActive() and light.lit


def test_what_it_does_names_the_languages_and_a_new_one_waits_for_the_next_start():
    window, app = _live_window(live_source="both", live_target="en", live_mic_target="ja")
    window.show_page("live")
    page = window.pages["live"]
    page.start_button.click()
    assert page.doing.text() == "Translating what the computer plays into English, and what you say into Japanese"
    page.target.setCurrentIndex(page.target.findData("ta"))
    assert app.settings.live_target == "ta" and page.doing.text().startswith("Translating what the computer plays into "
                                                                            "English")  # the session keeps its language
    assert page.note.text() == "Saved: the next start uses the new language." and not page.note.isHidden()
    page.sources.buttons["computer"].click()  # a source changes at once
    assert page.doing.text() == "Translating what the computer plays into English"
    app._live = False  # stopped elsewhere: the note about that session goes
    window.refresh()
    assert page.note.isHidden() and page.doing.text() == "Ready to translate what the computer plays into Tamil"


def test_the_source_decides_which_languages_are_asked_for():
    window, app = _live_window()
    window.show_page("live")
    page = window.pages["live"]
    assert not page.target_row.isHidden() and page.mic_row.isHidden()
    page.sources.buttons["microphone"].click()
    assert ("set_live_source", "microphone") in app.calls and app.settings.live_source == "microphone"
    assert page.target_row.isHidden() and not page.mic_row.isHidden() and page.mic_title.text() == "Microphone into"
    page.sources.buttons["both"].click()
    assert not page.target_row.isHidden() and not page.mic_row.isHidden() and page.mic_title.text() == "Your speech into"
    assert "headphones" in page.source_words.text()


def test_the_languages_the_shortcut_and_screen_sharing_are_saved():
    window, app = _live_window()
    window.show_page("live")
    page = window.pages["live"]
    page.target.setCurrentIndex(page.target.findData("ta"))
    page.mic_target.setCurrentIndex(page.mic_target.findData("ko"))
    page.shortcut.setCurrentIndex(page.shortcut.findData("win+alt+l"))
    page.hide_share.setChecked(False)
    s = app.settings
    assert (s.live_target, s.live_mic_target, s.live_shortcut, s.live_hide_from_share) == ("ta", "ko", "win+alt+l", False)
    assert ("set_live_hidden", False) in app.calls  # applied at once, also while it runs


def test_the_cost_saver_is_a_switch_and_the_page_says_what_a_session_cost_and_saved():
    window, app = _live_window()
    window.show_page("live")
    page = window.pages["live"]
    assert not page.save.isChecked() and page.spend.isHidden()  # off by default
    page.save.setChecked(True)
    assert ("set_live_save", True) in app.calls and app.settings.live_save
    app._live, app._live_spend = True, SimpleNamespace(cost=0.42, saved=0.26, seconds=660.0, unsent_s=420.0)
    window.refresh()
    assert page.spend.text() == "This session: $0.42 · 11 min · saved $0.26" and not page.spend.isHidden()
    app._live = False
    app._live_last, app._live_why = app._live_spend, "no speech for 15 min"  # the cost saver stopped a quiet session
    window.refresh()
    assert page.spend.text() == "Stopped: no speech for 15 min. Last session: $0.42 for 11 min, saved $0.26"
    app.settings = dataclasses.replace(app.settings, live_save=False)  # switched off on the bar: the page follows
    window.refresh()
    assert not page.save.isChecked()


def test_the_shortcut_is_shown_as_keys_and_one_another_tool_uses_stays_with_it():
    window, _ = _live_window()
    window.show_page("live")
    page = window.pages["live"]
    assert [cap.key_text for cap in page.start_card.findChildren(KeyCap)] == ["Ctrl", "Alt", "L"]
    window, _ = _live_window(translate_shortcut="ctrl+alt+l")
    window.show_page("live")
    page = window.pages["live"]
    assert "is Translate's shortcut, so it stays with Translate" in page.shortcut_caption.text()
    assert not page.start_card.findChildren(KeyCap)


def test_without_a_gemini_key_it_says_where_to_add_one():
    window, _ = _window(gateway=GatewayConfig(provider="openai", api_key="sk-test"), settings=Settings(welcomed=True))
    window.show_page("live")
    page = window.pages["live"]
    assert not page.connect_card.isHidden() and not page.start_button.isEnabled()
    assert page.state.text() == "Off" and page.doing.text() == "Add a Gemini key to start it"
    assert not page.start_card.grab().isNull()  # pressed in, quiet
    _button(page.connect_card, "Add a Gemini key").click()
    assert window.current_page() == "models"
    gemini = window.pages["models"].keys.rows["gemini"]
    assert gemini.editing and not gemini.isHidden() and gemini.key.editing  # its line open, ready to paste


def test_past_sessions_are_listed_to_open():
    from pathlib import Path
    window, app = _live_window()
    app._live_sessions = ((datetime(2026, 10, 5, 14, 3, 12), 23, Path("a.txt")), (datetime(2026, 10, 4, 9, 0), 1, Path("b.txt")))
    window.show_page("live")
    page = window.pages["live"]
    texts = [label.text() for label in page.findChildren(QLabel)]
    assert datetime(2026, 10, 5).strftime("%a 5 %b, 14:03") in texts and "23 lines" in texts and "1 line" in texts
    next(b for b in page.findChildren(QPushButton) if b.text() == "Open").click()
    assert ("open_live_session", Path("a.txt")) in app.calls
    copy = next(b for b in page.findChildren(QPushButton) if b.text() == "Copy")
    copy.click()
    assert ("copy_live_session", Path("a.txt"), False) in app.calls and copy.text() == "Copied"
    next(b for b in page.findChildren(QPushButton) if b.text() == "Copy translation").click()
    assert ("copy_live_session", Path("a.txt"), True) in app.calls


def test_hearing_the_translation_is_switched_on_in_the_section():
    window, app = _live_window()
    window.show_page("live")
    page = window.pages["live"]
    assert not page.speak.isChecked() and page.speed_row.isHidden()
    assert "Downloaded the first time (64 MB)" in page.speak_caption.text()
    page.speak.setChecked(True)
    assert ("set_live_speak", True) in app.calls and app.settings.live_speak and not page.speed_row.isHidden()
    page.speed.setCurrentIndex(page.speed.findData(1.3))
    assert app.settings.live_speak_speed == 1.3
    app._voice_state = ("downloading", 37, "")
    page.refresh()
    assert page.speak_caption.text().startswith("Downloading Ryan, the voice on this laptop: 37%")


def test_how_low_the_other_apps_stay_while_it_runs_is_a_slider_from_0_to_100():
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    window, app = _live_window(live_speak=True)
    window.show_page("live")
    page = window.pages["live"]
    assert page.duck.value() == 10 and page.duck_value.text() == "10%"  # the default: 10%
    assert "stay this quiet while live translation runs, and come back when it stops" in page.duck_caption.text()
    page.duck.setFocus()
    QTest.keyClick(page.duck, Qt.Key.Key_Right)  # the arrow keys: a step, chosen at once
    assert page.duck.value() == 15 and app.settings.live_duck == 0.15
    QTest.keyClick(page.duck, Qt.Key.Key_Home)  # all the way down: 0%, still heard by live translation (the Ducker)
    assert page.duck.value() == 0 and app.settings.live_duck == 0.0 and page.duck_value.text() == "0%"
    assert "go silent for you" in page.duck_caption.text() and "still hears and translates" in page.duck_caption.text()
    page.duck.resize(160, 24)
    QTest.mousePress(page.duck, Qt.MouseButton.LeftButton, pos=QPoint(150, 12))  # a drag: chosen when let go
    assert page.duck_value.text() == "Unchanged" and app.settings.live_duck == 0.0
    QTest.mouseRelease(page.duck, Qt.MouseButton.LeftButton, pos=QPoint(150, 12))
    assert ("set_live_duck", 1.0) in app.calls and app.settings.live_duck == 1.0
    assert "speaks over them" in page.duck_caption.text()
    assert page.duck.value_at(0) == 0 and page.duck.value_at(80) == 50  # whole steps where the pointer is


def test_the_section_says_when_the_voice_cant_speak_the_language_chosen():
    window, _ = _live_window(live_target="ja")
    window.show_page("live")
    words = window.pages["live"].speak_caption.text()
    assert words.startswith("Gemini says the translation in its own voice")
    assert "Ryan, the voice on this laptop, speaks only English" in words


# ---- Meeting mode (phase 49)

def _meeting_window(found=(), **settings):
    from sst.meeting.apps import MeetingWindow
    window, app = _live_window(**settings)
    app._meeting_windows = tuple(MeetingWindow(name, title, hwnd, 7) for name, title, hwnd in found)
    window.show_page("meeting")
    return window, app, window.pages["meeting"]


def test_meeting_mode_is_a_section_after_live_translation_and_finds_the_meeting_by_itself():
    window, app, page = _meeting_window([("Microsoft Teams", "Weekly sync | Microsoft Teams", 101)])
    keys = [key for key, _ in w.NAV]
    assert keys.index("meeting") == keys.index("live") + 1 and w.RAIL_NAMES["meeting"] == "Meeting"
    assert window.nav["meeting"].isChecked() and w.NAV_ICONS["meeting"] in w.theme.ICONS
    assert page.chosen.app == "Microsoft Teams" and page.picker.isHidden() and not page.app_line.isHidden()
    assert page.app_name.text() == "Microsoft Teams" and page.start_button.isEnabled() and page.connect_card.isHidden()
    assert page.state.text() == "Off" and page.doing.text() == "Ready for your meeting in Microsoft Teams"
    assert {p: t.isChecked() for p, t in page.parts.items()} == {"translate": True, "speak": False, "record": True,
                                                                 "minutes": True, "screen": True}  # silent captions


def test_the_in_person_tab_needs_no_meeting_app_and_has_its_own_switches():
    window, app, page = _meeting_window()  # no meeting app open
    assert [b.text() for b in page.tabs.buttons.values()] == ["Online", "In person"] and page.place == "online"
    assert not page.start_button.isEnabled() and page.room_card.isHidden()
    page.tabs.buttons["room"].click()
    assert page.place == "room" and app.settings.meeting_place == "room"  # remembered for the shortcut
    assert page.app_card.isHidden() and page.parts_card.isHidden() and not page.room_card.isHidden()
    assert set(page.room_parts) == {"translate", "record", "minutes"} and "room" in page.subtitle.text()
    assert page.start_button.isEnabled() and page.doing.text().startswith("Ready: put the laptop")
    page.room_parts["minutes"].click()
    assert ("set_meeting_option", "minutes", False, "in person") in app.calls and not app.settings.room_minutes
    assert app.settings.meeting_minutes  # the online switches stay as they were
    page.start_button.click()
    assert ("start_meeting", "In person") in app.calls and page.start_button.text() == "Stop"
    assert not page.tabs.buttons["online"].isEnabled()  # one meeting at a time
    page.start_button.click()
    assert page.state.text() == "Saved" and page.doing.text() == "The meeting in person is saved on this laptop."
    page.tabs.buttons["online"].click()
    assert page.state.text() == "Off"  # the room's meeting isn't the online tab's


def test_several_meetings_are_picked_from_and_none_says_start_one_first():
    window, app, page = _meeting_window([("Microsoft Teams", "Sync", 101), ("Zoom", "Zoom Meeting", 102)])
    assert page.chosen is None and not page.picker.isHidden() and not page.start_button.isEnabled()
    assert page.doing.text() == "Pick your meeting below" and page.picker.count() == 3
    page.picker.setCurrentIndex(2)
    assert page.chosen.app == "Zoom" and page.start_button.isEnabled()
    app._meeting_windows = app._meeting_windows[1:]  # Teams closed: Look again
    page._scan()
    assert page.chosen.app == "Zoom" and page.picker.isHidden()
    app._meeting_windows = ()
    page._scan()
    assert page.chosen is None and not page.start_button.isEnabled() and page.doing.text() == "Start a meeting first"
    assert "Start a meeting first" in page.app_words.text()


def test_a_meeting_starts_records_with_its_cost_then_shows_its_minutes():
    from sst.meeting.minutes import MinutesResult
    from sst.meeting.session import MeetingCost
    window, app, page = _meeting_window([("Microsoft Teams", "Weekly sync", 101)])
    page.start_button.click()
    assert ("start_meeting", "Microsoft Teams") in app.calls and page.start_button.text() == "Stop"
    app._meeting_cost = MeetingCost(total=0.12, translation=0.0, minutes_estimate=0.12, seconds=840.0, frames=6)
    window.refresh()
    assert page.state.text() == "Recording" and page.light.state == "live" and page.light.blinking
    assert page.doing.text() == "Microsoft Teams · 14 min · $0.12 so far · 6 slides"
    # a running meeting keeps the parts it started with; speaking changes at once, like the bar's speaker button
    assert [p for p, t in page.parts.items() if t.isEnabled()] == ["speak"]
    page.start_button.click()
    assert ("stop_meeting",) in app.calls and page.state.text() == "Writing the minutes…" and page.light.blinking
    assert not page.start_button.isEnabled()  # one meeting's minutes at a time
    state, meeting, folder, _, _ = app._meeting
    app._meeting = ("done", meeting, folder, MinutesResult("Weekly sync: the release moves to Friday", "…"), "")
    window.refresh()  # what the TrayApp does when the minutes are written
    assert page.state.text() == "Minutes ready" and page.doing.text() == "Weekly sync: the release moves to Friday"
    assert not page.done_row.isHidden() and not page.open_minutes.isHidden() and page.open_last_folder.isHidden()
    page.open_minutes.click()
    page.copy_last.click()
    assert ("open_meeting", folder) in app.calls and ("copy_minutes", folder) in app.calls
    assert page.copy_last.text() == "Copied" and page.start_button.isEnabled()


def test_failed_minutes_keep_the_recording_and_open_its_folder():
    window, app, page = _meeting_window([("Zoom", "Zoom Meeting", 102)])
    app._meeting = ("failed", app._meeting_windows[0], w.Path("m"), None, "Google didn't answer")
    window.refresh()
    assert page.state.text() == "The minutes failed" and "The recording is kept" in page.doing.text()
    assert page.open_minutes.isHidden() and not page.open_last_folder.isHidden()
    page.open_last_folder.click()
    assert ("open_meeting_folder", w.Path("m")) in app.calls


def test_the_parts_switch_for_the_next_meeting_and_follow_each_other():
    window, app, page = _meeting_window([("Zoom", "Zoom Meeting", 102)])
    page.parts["record"].click()
    assert ("set_meeting_option", "record", False) in app.calls and not app.settings.meeting_record
    assert not page.parts["minutes"].isChecked() and not page.parts["minutes"].isEnabled()  # no sound, no minutes
    assert not page.parts["screen"].isEnabled() and app.settings.meeting_minutes  # kept for when Record is on again
    page.parts["translate"].click()
    assert "Switch on live translation or Record" in page.connect_words.text() and not page.start_button.isEnabled()
    page.parts["record"].click()
    assert page.parts["minutes"].isChecked() and page.parts["screen"].isEnabled() and page.start_button.isEnabled()


def test_spoken_translation_is_its_own_row_that_needs_live_translation():
    window, app, page = _meeting_window([("Zoom", "Zoom Meeting", 102)], live_speak=True)
    keys = list(page.parts)
    assert keys.index("speak") == keys.index("translate") + 1 and "speak" not in page.room_parts  # never in the room
    assert not page.parts["speak"].isChecked() and page.parts["speak"].isEnabled()  # the Live page's speaking aside
    page.parts["speak"].click()
    assert ("set_meeting_option", "speak", True) in app.calls and app.settings.meeting_speak
    page.parts["translate"].click()
    assert not page.parts["speak"].isChecked() and not page.parts["speak"].isEnabled()
    assert page.part_rows["speak"].toolTip() == "Needs Live translation" and app.settings.meeting_speak  # kept
    page.parts["translate"].click()
    assert page.parts["speak"].isChecked() and page.part_rows["speak"].toolTip() == ""


def test_without_a_gemini_key_only_recording_can_start():
    window, app = _window(settings=Settings(welcomed=True))
    window.show_page("meeting")
    page = window.pages["meeting"]
    assert not page.connect_card.isHidden() and "Gemini key" in page.connect_words.text()
    for part in ("translate", "minutes"):
        page.parts[part].click()
    assert page.connect_card.isHidden() and app.meeting_problem() == ""


def test_past_meetings_open_and_copy_their_minutes_and_the_shortcut_can_change():
    window, app, page = _meeting_window()
    app._past_meetings = ((w.Path("a"), datetime(2026, 10, 24, 15, 30), "Zoom", 2710.0, True),
                          (w.Path("b"), datetime(2026, 10, 22, 9, 0), "Google Meet", 1260.0, False))
    window.refresh()
    texts = [b.text() for b in page.findChildren(QPushButton) if b.isVisibleTo(page)]
    assert texts.count("Copy minutes") == 1 and texts.count("Open minutes") == 1 and texts.count("Open folder") == 2
    labels = [x.text() for x in page.findChildren(QLabel)]
    assert "Zoom · 45 min" in labels and "Google Meet · 21 min · no minutes" in labels
    assert "Write minutes" not in texts  # no recording in "b" either: nothing to write them from
    page.shortcut.setCurrentIndex(page.shortcut.findData(""))
    assert app.settings.meeting_shortcut == "" and "tray menu" in page.how.itemAt(0).widget().text()
    window, app, page = _meeting_window(meeting_shortcut="ctrl+alt+l")
    assert app.meeting_shortcut_clash() == "Live translation" and "stays with Live translation" in page.shortcut_caption.text()


def test_a_past_meeting_with_a_recording_but_no_minutes_can_have_them_written(tmp_path):
    window, app, page = _meeting_window()
    (tmp_path / "others.wav").write_bytes(b"RIFF")
    app._past_meetings = ((tmp_path, datetime(2026, 10, 22, 9, 0), "Google Meet", 1260.0, False),)
    window.refresh()
    (write,) = [b for b in page.findChildren(QPushButton) if b.text() == "Write minutes"]
    write.click()
    assert app.calls[-1] == ("write_minutes", tmp_path)
    app.writing_minutes = {tmp_path}
    window.refresh()
    (busy,) = [b for b in page.findChildren(QPushButton) if b.text() == "Writing the minutes…"]
    assert not busy.isEnabled()


# ---- the top bar: the look and Check for updates

def test_the_top_bar_switches_light_and_dark_at_once_and_remembers_it():
    window, app = _window(settings=Settings(welcomed=True))
    assert list(window.appearance.buttons) == ["system", "light", "dark"]
    assert window.appearance.buttons["system"].isChecked() and w.theme.appearance() == "system"
    window.appearance.buttons["light"].click()
    assert app.settings.appearance == "light" and w.theme.theme() == "light" and not w.theme.dark_mode()
    window.appearance.buttons["dark"].click()
    assert app.settings.appearance == "dark" and w.theme.theme() == "dark"
    window.show_page("settings")
    page = window.pages["settings"]
    assert page.appearance.buttons["dark"].isChecked()
    page.appearance.buttons["light"].click()  # the same choice in Settings
    assert app.settings.appearance == "light" and window.appearance.buttons["light"].isChecked()
    assert w.theme.theme() == "light"
    again, _ = _window(settings=Settings(welcomed=True, appearance="dark"))  # the next start (or another profile)
    assert again.appearance.buttons["dark"].isChecked() and w.theme.theme() == "dark"
    _window(settings=Settings(welcomed=True))  # back to the default for the tests after
    assert w.theme.appearance() == "system"


def test_check_for_updates_in_the_top_bar_says_checking_then_the_answer():
    window, app = _window(settings=Settings(welcomed=True))
    assert window.check_button.text() == "Check for updates" and window.check_button.kind == "primary"
    window.check_button.click()
    assert ("check_for_updates", True) in app.calls
    assert window.check_button.text() == "Checking…" and not window.check_button.isEnabled()
    window.set_update_status("Checking for updates...")
    assert not window.check_button.isEnabled()
    window.set_update_status("You have the latest version (2.7.0).")
    assert window.check_button.isEnabled() and window.check_button.text() == "Check for updates"
    assert window.toast.text.text() == "You have the latest version (2.7.0)."
    window.check_button.click()
    window.show_update("Rflow 2.8.0 is available (you have 2.7.0).", version="2.8.0")  # the banner says it
    window.set_update_status("Rflow 2.8.0 is available (you have 2.7.0).")
    assert window.check_button.isEnabled() and not window.banner_holder.isHidden()
