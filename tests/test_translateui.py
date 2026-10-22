"""Translate's flow: Ctrl+C+C, the copied text, the popup, the translation, Copy and Replace. Windows is faked (no key is
pressed, no window activated, the clipboard isn't touched) and so is the AI model."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import queue  # noqa: E402
import threading  # noqa: E402

import pytest  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst import translate as translate_module  # noqa: E402
from sst import translateui  # noqa: E402
from sst.hotkey import VK_ESCAPE  # noqa: E402
from sst.settings import Settings  # noqa: E402
from sst.translate import Translation  # noqa: E402
from sst.translateui import TranslateController  # noqa: E402

SOURCE = "Could you send me the updated report by Friday?"
JAPANESE = "金曜日までに更新したレポートを送っていただけますか？"


@pytest.fixture(scope="module", autouse=True)
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def windows_in_english(monkeypatch):
    monkeypatch.setattr(translate_module, "system_language", lambda: "English")  # the same on every test machine


class FakeAccess:
    """The user's app and the clipboard: what it copies on Ctrl+C+C, what is selected, what gets pasted."""

    def __init__(self, copied=SOURCE, window=7):
        self.window, self.copied, self.selection = window, copied, copied
        self.sequence, self.copies_on_trigger = 1, True  # the app's copy lands just after the shortcut
        self.cls, self.pid = "Chrome_WidgetWin_1", 4242
        self.clipboard, self.pasted, self.activated, self.selected_last = None, [], [], []
        self.copy_calls = 0

    def foreground_window(self):
        return self.window

    def window_class(self, hwnd):
        return self.cls

    def window_process(self, hwnd):
        return os.getpid() if hwnd == 99 else self.pid  # window 99 is one of Rflow's own

    def clipboard_sequence(self):
        if self.copies_on_trigger:
            self.copies_on_trigger = False
            return self.sequence  # read at the shortcut: the copy lands right after
        return self.sequence + 1

    def clipboard_text(self):
        return self.copied

    def copy_selection(self, timeout=0.6, fallback=False):
        self.copy_calls += 1
        return self.selection

    def select_last(self, text):
        self.selected_last.append(text)
        return None

    def activate(self, hwnd):
        self.activated.append(hwnd)
        self.window = hwnd
        return True

    def set_clipboard(self, text, html=None):
        self.clipboard = text

    def paste_rich(self, text, html=None):
        self.pasted.append(text)


class FakeListener:
    def __init__(self, hotkey):
        self.hotkey, self.events, self.captured, self.on_event = hotkey, queue.Queue(), None, None

    def send(self, event):
        """As the keyboard hook does: queue the event, then wake the controller up (it polls nothing)."""
        self.events.put((event, 0.0))
        self.on_event()

    def start(self):
        pass

    def stop(self):
        pass

    def capture(self, keys):
        self.captured = keys


class FakeApp:
    def __init__(self, ready=True, answer=JAPANESE, gate=None):
        self.settings = Settings(welcomed=True, translate_to="Japanese")
        self.ready, self.answer, self.gate = ready, answer, gate
        self.said, self.asked, self.opened = [], [], []

    def open_window(self, page):
        self.opened.append(page)

    def translate_ready(self):
        return self.ready

    def transform_model(self):
        return "gemini-3.5-flash-lite"

    def run_translation(self, text, target, second=""):
        self.asked.append((text, target, second))
        if self.gate is not None:
            self.gate.wait(2)
        if isinstance(self.answer, Exception):
            raise self.answer
        return Translation(f"{self.answer} [{target}]", target, text, 0.4)

    def say(self, state, message=""):
        self.said.append((state, message))

    def apply_settings(self, new):
        self.settings = new


_LIVE: list[TranslateController] = []  # never collected while a worker thread may still signal them


@pytest.fixture(autouse=True)
def finish_work():
    yield
    for controller in _LIVE:
        wait_until(lambda c=controller: not c.busy)
        QTest.qWait(30)
        controller.stop()


def make(shortcut="ctrl+c+c", access=None, **app):
    access, fake_app = access or FakeAccess(), FakeApp(**app)
    controller = TranslateController(fake_app, access=access, listener_factory=FakeListener)
    controller.start(shortcut)
    _LIVE.append(controller)
    return controller, access, fake_app


def wait_until(condition, ms=2000):
    for _ in range(ms // 10):
        if condition():
            return True
        QTest.qWait(10)
    return condition()


def press(controller):
    controller.listener.send("release")


def shown(controller):
    return wait_until(lambda: controller.result is not None)


def test_ctrl_c_c_shows_the_copied_text_translated(qt):
    controller, access, app = make()
    assert controller.double_copy
    press(controller)
    assert shown(controller) and controller.popup.isVisible()
    assert app.asked == [(SOURCE, "Japanese", "English")]  # no second language chosen: Windows' own, else English
    assert controller.popup.result.toPlainText() == f"{JAPANESE} [Japanese]"
    assert controller.popup.route.text() == "English →"  # the text's language, then the one it went into
    assert access.copy_calls == 0  # the app copied: Rflow pressed nothing
    assert controller.listener.captured == {VK_ESCAPE, translateui.VK_C, translateui.VK_RETURN}  # Esc, Copy, Replace


def test_a_copy_that_landed_before_the_shortcut_is_read_after_a_moment(qt, monkeypatch):
    monkeypatch.setattr(translateui, "COPY_WAIT", 0.05)
    access = FakeAccess()
    access.copies_on_trigger = False
    access.clipboard_sequence = lambda: 1  # nothing changes any more
    controller, _, app = make(access=access)
    press(controller)
    assert shown(controller) and app.asked[0][0] == SOURCE


@pytest.mark.parametrize("copied, said", [
    (None, "Select some text first, then try again."),
    ("   ", "Select some text first, then try again."),
    ("x" * 5001, "That's 5,001 characters: Translate takes up to 5,000 at a time."),
])
def test_nothing_to_translate_says_why(qt, copied, said):
    controller, access, app = make(access=FakeAccess(copied=copied))
    press(controller)
    assert wait_until(lambda: app.said)
    assert app.said == [("warning", said)] and not controller.popup.isVisible() and not app.asked


def test_without_an_ai_model_the_popup_says_so_and_leads_to_ai_cleanup(qt):
    controller, _, app = make(ready=False)
    press(controller)
    popup = controller.popup
    assert wait_until(popup.isVisible) and "needs an AI model" in popup.status.text()
    assert not popup.setup_button.isHidden() and popup.copy_button.isHidden() and not app.asked  # nothing sent
    popup.setup.emit()
    assert app.opened == ["cleanup"] and not popup.isVisible()


def test_the_second_language_is_used_for_text_already_in_the_first(qt):
    controller, _, app = make()
    app.settings.translate_second = "English"
    press(controller)
    assert shown(controller) and app.asked == [(SOURCE, "Japanese", "English")]


def test_picking_a_language_translates_again_into_exactly_it_and_remembers_it(qt):
    controller, _, app = make()
    press(controller)
    assert shown(controller)
    controller.popup.language.emit("French")
    assert wait_until(lambda: controller.result is not None and controller.result.target == "French")
    assert app.asked[-1] == (SOURCE, "French", "") and app.settings.translate_to == "French"


def test_an_older_answer_never_replaces_a_newer_one(qt):
    gate = threading.Event()
    controller, _, app = make(gate=gate)
    press(controller)
    assert wait_until(lambda: len(app.asked) == 1)
    controller.popup.language.emit("French")  # asked again before the first answer came
    gate.set()
    assert wait_until(lambda: controller.result is not None and controller.result.target == "French")
    QTest.qWait(50)
    assert controller.result.target == "French"


def test_copy_puts_the_translation_on_the_clipboard(qt):
    controller, access, app = make()
    press(controller)
    assert shown(controller)
    controller.popup.copy.emit()
    assert access.clipboard == f"{JAPANESE} [Japanese]" and controller.popup.isVisible()  # it stays, and says so
    assert controller.popup.copy_button.text() == "Copied" and not app.said


def test_replace_puts_the_translation_in_place_of_the_text(qt):
    controller, access, app = make()
    press(controller)
    assert shown(controller)
    access.window = 8  # the user looked at another window meanwhile: the text's window comes back first
    controller.popup.replace.emit()
    assert wait_until(lambda: access.pasted)
    assert access.activated == [7] and access.pasted == [f"{JAPANESE} [Japanese]"]
    assert wait_until(lambda: app.said and app.said[-1] == ("transformed", "Translated"))


def test_replace_never_pastes_over_other_text(qt):
    controller, access, app = make()
    press(controller)
    assert shown(controller)
    access.selection = "something else the user selected"
    controller.popup.replace.emit()
    assert wait_until(lambda: app.said and app.said[-1][0] == "warning")
    assert not access.pasted and access.clipboard == f"{JAPANESE} [Japanese]" and "press Ctrl+V" in app.said[-1][1]


def test_esc_closes_the_popup(qt):
    controller, _, _ = make()
    press(controller)
    assert shown(controller)
    controller.listener.send(f"key:{VK_ESCAPE}")
    assert wait_until(lambda: not controller.popup.isVisible()) and controller.listener.captured is None


def test_the_popup_closes_when_the_user_goes_to_another_app_but_not_for_rflows_own_windows(qt):
    controller, access, _ = make()
    press(controller)
    assert shown(controller)
    access.window = 99  # the language list: one of Rflow's own windows
    QTest.qWait(60)
    assert controller.popup.isVisible()
    access.window = 8  # another app
    assert wait_until(lambda: not controller.popup.isVisible())
    assert wait_until(lambda: not controller.watch.isActive())  # the popup is gone: nothing is watched any more


def test_nothing_runs_while_waiting_for_the_shortcut(qt):
    """Rflow sits in the tray all day: Translate wakes up for its shortcut's events, it never polls for them."""
    from PySide6.QtCore import QTimer
    controller, _, _ = make()
    assert not any(timer.isActive() for timer in controller.findChildren(QTimer))
    press(controller)
    assert shown(controller) and controller.watch.isActive()  # only while the popup is open


def test_a_failing_model_is_explained_with_try_again(qt):
    controller, _, app = make(answer=RuntimeError("HTTP 429 quota"))
    press(controller)
    popup = controller.popup
    assert wait_until(lambda: "limit is reached" in popup.status.text())  # in plain words
    assert "HTTP 429 quota" in popup.detail.toolTip() and not popup.detail.isHidden()  # the provider's own, small
    assert popup.copy_button.isHidden() and not popup.retry_button.isHidden() and popup.setup_button.isHidden()
    app.answer = JAPANESE
    popup.retry.emit()
    assert wait_until(lambda: controller.result is not None) and len(app.asked) == 2


def test_a_refused_key_leads_to_ai_cleanup(qt):
    controller, _, app = make(answer=RuntimeError("HTTP 401: invalid API key"))
    press(controller)
    popup = controller.popup
    assert wait_until(lambda: "refused the API key" in popup.status.text()) and not popup.setup_button.isHidden()
    popup.setup.emit()
    assert app.opened == ["cleanup"]


def test_another_shortcut_copies_the_selection_itself_but_never_in_a_terminal(qt):
    controller, access, app = make(shortcut="ctrl+alt+l")
    assert not controller.double_copy
    press(controller)
    assert shown(controller) and access.copy_calls == 1
    controller.popup.close_popup()
    access.cls = "CASCADIA_HOSTING_WINDOW_CLASS"
    app.said.clear()
    press(controller)
    assert wait_until(lambda: app.said) and access.copy_calls == 1 and "terminal" in app.said[-1][1]


def test_off_means_no_shortcut(qt):
    controller, _, _ = make(shortcut="")
    assert controller.listener is None
    controller.start("not a key")
    assert controller.listener is None


# ---- the popup's UX (phase 24)

def test_the_popup_s_buttons_take_clicks_instead_of_letting_them_through(qt, monkeypatch):
    from sst import app as sst_app
    calls = []
    monkeypatch.setattr(sst_app, "_no_activate", lambda hwnd, click_through=True: calls.append(click_through))
    controller, _, _ = make()
    press(controller)
    assert shown(controller) and calls == [False]  # click-through sent every click to the app below (until 1.10.1)

def test_the_header_says_from_what_into_what_with_one_click_languages(qt):
    controller, _, app = make(access=FakeAccess(copied=JAPANESE))
    app.settings.translate_to = "English"
    press(controller)
    assert shown(controller)
    popup = controller.popup
    chips = {chip.text(): chip.isChecked() for chip in popup.chips}
    assert popup.route.text() == "Japanese →" and chips.get("English") is True
    assert "Japanese" not in chips  # never into the language the text is already in
    popup.chips[-1].click()
    assert wait_until(lambda: app.asked[-1][1] == popup.chips[-1].text())


def test_more_shows_every_language_and_esc_goes_back(qt):
    controller, _, app = make()
    press(controller)
    assert shown(controller)
    popup = controller.popup
    popup.more.click()
    assert not popup.grid_box.isHidden() and popup.result.isHidden() and len(popup.all_languages) >= 25
    controller.listener.send(f"key:{VK_ESCAPE}")
    assert wait_until(lambda: popup.grid_box.isHidden()) and popup.isVisible() and not popup.result.isHidden()
    popup.more.click()
    popup.all_languages["Korean"].click()
    assert wait_until(lambda: controller.result is not None and controller.result.target == "Korean")
    assert popup.grid_box.isHidden() and app.settings.translate_to == "Korean"


def test_a_click_outside_closes_the_popup_but_not_one_inside(qt, monkeypatch):
    access = FakeAccess()
    access.mouse_down = lambda: access.down
    access.down = False
    controller, _, _ = make(access=access)
    press(controller)
    assert shown(controller)
    popup, where = controller.popup, {}
    monkeypatch.setattr(translateui, "_pointer", lambda: where["at"])
    where["at"] = popup.frameGeometry().center()
    QTest.qWait(40)
    access.down = True  # a click on the popup itself
    QTest.qWait(60)
    assert popup.isVisible()
    access.down = False
    QTest.qWait(40)
    where["at"] = popup.frameGeometry().bottomRight() + translateui.QPoint(40, 40)
    access.down = True  # and one outside it
    assert wait_until(lambda: not popup.isVisible())


def test_a_long_translation_never_covers_the_buttons(qt):
    controller, _, app = make(answer="A long translation. " * 120)
    press(controller)
    assert shown(controller)
    popup = controller.popup
    QTest.qWait(30)
    assert popup.result.geometry().bottom() < popup.buttons.geometry().top()  # scrolls inside its box instead
    area = popup.screen().availableGeometry()
    assert area.contains(popup.frame.geometry().translated(popup.pos()))  # and the grown panel is still on the screen


def test_english_text_with_english_chosen_goes_into_windows_language(qt, monkeypatch):
    monkeypatch.setattr(translate_module, "system_language", lambda: "Japanese")
    controller, _, app = make()
    app.settings.translate_to = "English"
    press(controller)
    assert shown(controller) and app.asked == [(SOURCE, "English", "Japanese")]  # never English into English
