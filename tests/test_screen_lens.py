"""Screen translator's frame, its worker and the app's side (sst.screen.lens, sst.screen.control), off-screen, with a fake
screen, OCR and translator: nothing is captured, read by Windows or sent anywhere."""
import ast
import dataclasses
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtGui import QAction, QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import sst.screen  # noqa: E402
from sst.gateway import GatewayConfig  # noqa: E402
from sst.screen.capture import Frame  # noqa: E402
from sst.screen.control import ScreenControl  # noqa: E402
from sst.screen.lens import ScreenLens  # noqa: E402
from sst.screen.render import paint_stickers  # noqa: E402
from sst.screen.winocr import Word  # noqa: E402
from sst.settings import Settings  # noqa: E402

FORBIDDEN = ("sst.pipeline", "sst.dictate", "sst.audio")  # dictation's pipeline, its controller and its recorder


@pytest.fixture(scope="module", autouse=True)
def qt():
    return QApplication.instance() or QApplication([])


def test_screen_translation_imports_nothing_from_dictation():
    for path in Path(sst.screen.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                assert not name.startswith(FORBIDDEN), f"{path.name} imports {name}: keep screen translation apart"


def _page() -> np.ndarray:
    page = np.full((300, 400), 250, np.uint8)
    page[60:74, 40:240] = np.random.default_rng(3).integers(0, 120, (14, 200), dtype=np.uint8)
    return page


class FakeCapturer:
    def __init__(self):
        self.frame = Frame.of_gray(_page(), 0, 0)

    def grab(self, rect, t=0.0):
        return self.frame

    def close(self):
        pass


class FakeEngine:
    max_dim = 10000

    def recognize(self, gray, width, height):
        s = width / 400
        return [[Word("請求書を送ります", 40 * s, 60 * s, 200 * s, 14 * s)]]


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = []

    def translate_batch(self, texts):
        self.calls.append(texts)
        return ["We send the invoice" for _ in texts]


def test_the_worker_reads_a_settled_screen_and_its_text_gets_a_translated_sticker():
    backend = FakeBackend()
    lens = ScreenLens(backend, make_ocr=FakeEngine, make_capturer=FakeCapturer, light=lambda: False)
    lens._rect = (0, 0, 400, 300)
    lens._update_rect = lambda: None  # off-screen windows have no real rectangle
    cap = FakeCapturer()
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        lens.tick(cap)
        if lens.tracker.stickers and lens.svc.get("請求書を送ります"):
            break
        time.sleep(0.05)
    (sticker,) = lens.tracker.stickers
    assert sticker.src == "請求書を送ります" and (sticker.x, sticker.y) == (40, 60)
    assert lens.svc.get(sticker.src) == "We send the invoice" and backend.calls == [["請求書を送ります"]]
    assert "1 translated" in lens.status_text()
    lens.toggle_pause()
    assert lens.status_text() == "Paused" and lens.tick(cap) == 400
    lens.svc.shutdown()


def test_the_stickers_are_painted_over_their_text_in_its_colours():
    lens = ScreenLens(FakeBackend(), make_ocr=FakeEngine, make_capturer=FakeCapturer, light=lambda: False)
    from sst.screen.layout import build_blocks
    from sst.screen.ocr import OcrLine
    frame = FakeCapturer().frame
    lens.tracker.replace_from_ocr(build_blocks([OcrLine("請求書を送ります", 40, 60, 200, 14, 14)]), frame)
    image = QImage(400, 300, QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    p = QPainter(image)
    drawn = paint_stickers(p, lens.tracker.stickers, lambda s: "We send the invoice", (0, 0), 1.0, clip_w=400)
    p.end()
    assert drawn == 1
    assert image.pixelColor(45, 62).alpha() == 255  # covered
    assert image.pixelColor(300, 200).alpha() == 0  # the rest left alone
    lens.svc.shutdown()


# ---- the app's side

class FakeApp:
    def __init__(self, settings=None, key="KEY"):
        self.settings = settings or Settings()
        self.gateway = GatewayConfig().with_key("gemini", key)
        self.applied, self.notes = [], []
        self.profile = SimpleNamespace(settings_file=None)

    def apply_settings(self, new):
        self.applied.append(new)
        self.settings = new

    def _notify(self, title, text, icon=None):
        self.notes.append(text)


class FakeLens:
    def __init__(self, on_close):
        self.running, self.on_close, self.started = False, on_close, None

    def start(self, geometry=None, fit=None):
        self.running, self.started = True, (geometry, fit)

    def close(self):
        self.running = False
        self.on_close([10, 20, 900, 600])


def _control(app, asked=True):
    control = ScreenControl(app, QAction("Screen translator"), check_ocr=lambda: "", ask=lambda text: asked)
    control._make_lens = lambda: FakeLens(control._closed)
    return control


def test_the_first_start_asks_once_and_the_frame_is_remembered(monkeypatch, tmp_path):
    app = FakeApp()
    app.profile.settings_file = tmp_path / "settings.json"
    control = _control(app)
    assert control.start() == "" and control.running and control.action.isChecked()
    assert app.settings.screen_told
    control.toggle()  # stop
    assert not control.running and not control.action.isChecked()
    assert app.settings.screen_lens == [10, 20, 900, 600]
    assert Settings.load(app.profile.settings_file).screen_lens == [10, 20, 900, 600]
    control.start()
    assert control.lens.started == ([10, 20, 900, 600], None)


def test_no_gemini_key_or_a_no_means_not_started():
    assert "Gemini API key" in _control(FakeApp(key="")).start()
    app = FakeApp()
    assert _control(app, asked=False).start() == "Not started." and not app.settings.screen_told
    control = ScreenControl(FakeApp(), QAction("x"), check_ocr=lambda: "Windows' OCR for Japanese isn't installed")
    assert "OCR" in control.start()


def test_the_shortcut_gives_way_to_another_tools():
    app = FakeApp(Settings(screen_shortcut="ctrl+alt+l"))
    assert _control(app).clash() == "Live translation"
    assert _control(FakeApp()).clash() == ""
    assert Settings().screen_shortcut == "ctrl+alt+s"


def test_translate_page_has_the_screen_translation_card():
    from sst import window as w
    app = w.PreviewApp(settings=Settings(welcomed=True))
    app.screen = _control(FakeApp())
    window = w.MainWindow(app)
    window.show_page("translate")
    card = window.pages["translate"].screen_card
    card.refresh()
    assert card.start_button.text() == "Start" and card.shortcut.currentData() == "ctrl+alt+s"
    assert "Ctrl+Alt+S" in card.note.text()
    app.apply_settings = lambda new: setattr(app, "settings", new)
    card.shortcut.setCurrentIndex(card.shortcut.findData(""))
    assert app.settings.screen_shortcut == "" and "No shortcut" in card.note.text()
    assert dataclasses.replace(app.settings, screen_shortcut="ctrl+alt+s")
