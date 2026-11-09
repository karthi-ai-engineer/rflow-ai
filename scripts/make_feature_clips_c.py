"""Render two short clips of Rflow's translators for the website, each the feature and nothing else, made to loop:

- site/media/clip-screen.mp4: the Screen translator (sst.screen) over an app in Japanese: Ctrl+Alt+S, the frame around
  the app, its strings waiting for Gemini (dashed), then English stickers in their place, in the app's own colours;
  the eye shows the original and back; the frame closed;
- site/media/clip-text.mp4: Translate's Text tab (TextTranslator): Japanese typed (Detect language finds it), the
  English after the pause, then the swap: English into Japanese.

And a poster for each (site/img/clip-<name>-poster.png, 1280x720), a frame that shows the result. Like the other clips
(scripts/make_feature_clips_a.py and _b.py, whose helpers in make_demo_video.py they use): Rflow's own parts with example
data, drawn off the screen: the screen translator's real frame and its sticker painter (sst.screen.render), the real
window on its Translate page. Nothing appears, takes focus, presses keys or touches the clipboard; the app under the
stickers is painted here, a generic one in no real product's look. On the tour's 1280x720 stage at 150% (demo.WIDE).

For the tour, CLIPS maps each name to (seconds, frame(t) -> QImage, 1920x1080): importing this file renders nothing.
Pillow (posters, and imported by make_demo_video) is added only for the run:

  uv run --no-sync --with pillow python scripts/make_feature_clips_c.py [screen | text]
"""
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]  # this checkout's Rflow, whichever one is installed
JOBS = ("screen", "text")
JOB = sys.argv[1] if len(sys.argv) > 1 else ""

import make_demo_video as demo  # noqa: E402  (before Qt: it sets Qt's platform, and a scratch APPDATA)

os.environ["QT_SCALE_FACTOR"] = "3"  # Rflow's parts grabbed at 3x: the text clip zooms in on the window

import make_feature_clips_b as clips  # noqa: E402  (the encoder and the posters)
import numpy as np  # noqa: E402
from make_demo_video import POP, alpha, clamp, draw_image, faded, mix, ramp  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolButton  # noqa: E402

from sst import theme  # noqa: E402

LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
LOOP = 0.8  # the last moments: back to the first frame


def cursor(p: QPainter, at: QPointF, shown: float, pressed: float = 0.0) -> None:
    """The mouse pointer, Windows' white arrow; pressed, a ring in Iris around its tip."""
    if shown <= 0:
        return
    with faded(p, shown):
        if pressed > 0:
            p.setPen(QPen(alpha(POP["iris"], 0.85 * pressed), 2))
            p.setBrush(alpha(POP["iris"], 0.18 * pressed))
            p.drawEllipse(at, 10 + 6 * pressed, 10 + 6 * pressed)
        arrow = QPainterPath(at)
        for x, y in ((0, 17), (4.2, 13.2), (7, 19.6), (9.6, 18.5), (6.9, 12.2), (12.4, 12.2)):
            arrow.lineTo(at + QPointF(x, y))
        arrow.closeSubpath()
        p.setPen(QPen(QColor(0, 0, 0, 230), 1.1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawPath(arrow)


def glide(t: float, path: list[tuple[float, QPointF]]) -> QPointF:
    """Where the pointer is at t, eased from point to point: (time it arrives, where)."""
    if t <= path[0][0]:
        return path[0][1]
    for (t0, a), (t1, b) in zip(path, path[1:], strict=False):
        if t <= t1:
            k = demo.ease((t - t0) / (t1 - t0))
            return QPointF(mix(a.x(), b.x(), k), mix(a.y(), b.y(), k))
    return path[-1][1]


def tap(t: float, at: float) -> float:
    """A click at `at`: the ring in, then out."""
    return ramp(t, at - 0.05, 0.08) * (1 - ramp(t, at + 0.12, 0.25))


def crossfade(a: QImage, b: QImage, amount: float) -> QImage:
    image = a.copy()
    p = QPainter(image)
    p.setOpacity(clamp(amount))
    p.drawImage(0, 0, b)
    p.end()
    return image


# ---------------------------------------------------------------- 1. the screen translator

# An expense report in a company's own tool, in Japanese: a header with its menu, a notice, a table, buttons.
APP = QRectF(160, 22, 960, 532)  # the app's window on the stage
TITLE_BAR, HEADER = 28, 50  # the title bar as tall as the frame's toolbar, which covers it
BLUE, PAGE, INK, GREY = QColor("#2D5BD3"), QColor("#F4F5F8"), QColor("#1F2430"), QColor("#6B7280")
JA = "Yu Gothic UI"  # a Japanese app's own font on Windows


@dataclass
class Item:
    """A string in the app: where its text is, how it's drawn, and what Gemini makes of it."""
    ja: str
    en: str
    x: float  # the text's left, its row's centre, on the stage
    cy: float
    px: float
    fg: QColor
    bg: QColor
    bold: bool = False
    box: QRectF | None = None  # its pill or button, if it's in one: then centred in it
    room: float | None = None  # the stage px from the text's left to its cell's right edge


def items() -> list[Item]:
    left, top = APP.left(), APP.top() + TITLE_BAR
    white, head = QColor("#FFFFFF"), QColor("#EEF1F6")
    card = QRectF(left + 28, top + HEADER + 20, APP.width() - 56, APP.height() - TITLE_BAR - HEADER - 40)
    x, cols = card.left() + 26, (0, 150, 470, 620)
    rows = [("11月4日", "Nov 4", "東京→大阪 新幹線", "Tokyo → Osaka, bullet train", "¥14,720", "承認済み", "Approved",
             "#E3F5EA", "#1E7B43"),
            ("11月5日", "Nov 5", "ホテル宿泊（1泊）", "Hotel (1 night)", "¥12,800", "確認中", "In review", "#FFF1D6", "#9A6100"),
            ("11月6日", "Nov 6", "顧客との会食", "Client dinner", "¥8,400", "差し戻し", "Returned", "#FDE4E4", "#B42318")]
    out = [Item("経費精算", "Expenses", left + 58, top + HEADER / 2, 17, white, BLUE, bold=True)]
    nx = APP.right() - 400
    for ja, en in (("ホーム", "Home"), ("申請一覧", "Requests"), ("承認", "Approvals"), ("設定", "Settings")):
        out.append(Item(ja, en, nx, top + HEADER / 2, 14, white, BLUE))
        nx += 84
    y = card.top() + 34
    out += [Item("出張費の申請", "Business trip expenses", x, y, 22, INK, white, bold=True),
            Item("申請者：田中 太郎", "Applicant: Taro Tanaka", x, y + 36, 13, GREY, white),
            Item("部署：営業部", "Department: Sales", x + 210, y + 36, 13, GREY, white),
            Item("申請は月末までに提出してください。領収書の画像を添付する必要があります。",
                 "Please submit by the end of the month. A photo of each receipt is required.", x + 16, y + 78, 13.5,
                 QColor("#6B4E00"), QColor("#FFF7E0"))]
    ty = y + 124
    for (ja, en), cx in zip((("日付", "Date"), ("内容", "Item"), ("金額", "Amount"), ("状態", "Status")), cols, strict=True):
        out.append(Item(ja, en, x + 14 + cx, ty, 13, QColor("#4B5563"), head, bold=True, room=(140 if cx < 470 else 130)))
    for i, (date, date_en, what, what_en, _, state, state_en, bg, fg) in enumerate(rows):
        cy = ty + 36 + i * 38
        out += [Item(date, date_en, x + 14, cy, 14, INK, white, room=136),
                Item(what, what_en, x + 14 + cols[1], cy, 14, INK, white, room=306)]
        pill = QRectF(x + 14 + cols[3], cy - 13, 96, 26)
        out.append(Item(state, state_en, pill.left() + 12, cy, 12.5, QColor(fg), QColor(bg), bold=True, box=pill))
    cy = ty + 36 + 3 * 38
    out.append(Item("合計", "Total", x + 14 + cols[1], cy, 14, INK, white, bold=True, room=300))
    by = card.bottom() - 34
    out += [Item("下書き保存", "Save draft", 0, by, 14, BLUE, white, bold=True, box=QRectF(card.right() - 290, by - 18, 126, 36)),
            Item("申請する", "Submit", 0, by, 14, white, BLUE, bold=True, box=QRectF(card.right() - 152, by - 18, 126, 36))]
    return out


def ja_font(item: Item) -> QFont:
    font = QFont(JA)
    font.setPixelSize(round(item.px))
    font.setBold(item.bold)
    return font


def text_rect(item: Item) -> QRectF:
    """Where the item's Japanese is on the stage (centred in its pill or button)."""
    m = QFontMetricsF(ja_font(item))
    w, h = m.horizontalAdvance(item.ja), m.height()
    if item.box is not None:
        return QRectF(item.box.center().x() - w / 2, item.box.center().y() - h / 2, w, h)
    return QRectF(item.x, item.cy - h / 2, w, h)


def draw_app(p: QPainter) -> None:
    """The app, in Japanese, as it is without Rflow."""
    r, top = APP, APP.top() + TITLE_BAR
    theme.outer_shadow(p, r, 10, [(0, 18, 44, theme.rgba(0, 0, 0, .55)), (0, 2, 6, theme.rgba(0, 0, 0, .30))], demo.STAGE.s)
    p.save()
    p.setClipPath(theme.rounded(r, 10))
    p.fillRect(r, PAGE)
    p.fillRect(QRectF(r.left(), r.top(), r.width(), TITLE_BAR), QColor("#E9EBEF"))
    p.fillRect(QRectF(r.left(), top, r.width(), HEADER), BLUE)
    p.restore()
    title = QFont(JA)
    title.setPixelSize(12)
    p.setFont(title)
    p.setPen(QColor("#4B5563"))
    p.drawText(QRectF(r.left() + 16, r.top(), 400, TITLE_BAR), LEFT, "経費精算システム")
    cy = r.top() + TITLE_BAR / 2
    p.setPen(QPen(QColor("#6B7280"), 1.2))
    x = r.right() - 96
    p.drawLine(QPointF(x - 5, cy), QPointF(x + 5, cy))
    p.drawRect(QRectF(x + 32 - 5, cy - 5, 10, 10))
    p.drawLine(QPointF(x + 64 - 5, cy - 5), QPointF(x + 64 + 5, cy + 5))
    p.drawLine(QPointF(x + 64 + 5, cy - 5), QPointF(x + 64 - 5, cy + 5))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#FFFFFF"))
    p.drawRoundedRect(QRectF(r.left() + 26, top + HEADER / 2 - 11, 22, 22), 6, 6)  # the tool's mark
    p.setBrush(BLUE)
    p.drawRoundedRect(QRectF(r.left() + 32, top + HEADER / 2 - 5, 10, 10), 2, 2)
    p.setBrush(QColor(255, 255, 255, 60))
    p.drawEllipse(QPointF(r.right() - 34, top + HEADER / 2), 14, 14)  # the user
    card = QRectF(r.left() + 28, top + HEADER + 20, r.width() - 56, r.height() - TITLE_BAR - HEADER - 40)
    theme.outer_shadow(p, card, 10, [(0, 2, 10, theme.rgba(20, 30, 60, .10))], demo.STAGE.s)
    p.setBrush(QColor("#FFFFFF"))
    p.drawRoundedRect(card, 10, 10)
    found = items()
    notice = next(i for i in found if i.ja.startswith("申請は"))
    rect = text_rect(notice)
    p.setBrush(notice.bg)
    p.setPen(QPen(QColor("#F2D58A"), 1))
    p.drawRoundedRect(QRectF(rect.left() - 16, rect.top() - 9, card.width() - 52, rect.height() + 18), 7, 7)
    head = next(i for i in found if i.ja == "日付")
    table = QRectF(head.x - 14, head.cy - 17, card.width() - 52, 34)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(head.bg)
    p.drawRoundedRect(table, 6, 6)
    for k in range(4):  # the rows' lines
        p.fillRect(QRectF(table.left(), head.cy + 36 + 38 * k + 19, table.width(), 1), QColor("#E5E7EB"))
    for item in found:
        if item.box is not None:
            p.setBrush(item.bg)
            p.setPen(QPen(BLUE, 1.2) if item.ja == "下書き保存" else Qt.PenStyle.NoPen)
            p.drawRoundedRect(item.box, item.box.height() / 2 if item.box.height() < 30 else 7,
                              item.box.height() / 2 if item.box.height() < 30 else 7)
        p.setFont(ja_font(item))
        p.setPen(item.fg)
        p.drawText(text_rect(item), Qt.AlignmentFlag.AlignCenter, item.ja)
    amount = QFont(JA)
    amount.setPixelSize(14)
    for i, value in enumerate(("¥14,720", "¥12,800", "¥8,400", "¥35,920")):  # numbers: nothing to translate
        amount.setBold(i == 3)
        p.setFont(amount)
        p.setPen(INK)
        p.drawText(QRectF(head.x + 470 - 100, head.cy + 36 + 38 * i - 12, 120, 24),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, value)


def stickers() -> list:
    """The screen translator's stickers for the app's strings, as its OCR and layout would make them (in stage pixels:
    the painter's origin is the stage's, one pixel each)."""
    from sst.screen.tracker import Sticker
    out = []
    for n, item in enumerate(items()):
        r = text_rect(item)
        if item.box is not None:  # its pill or button, inside the rounded ends: the English centred there
            r = item.box.adjusted(item.box.height() / 2 - 2, 4, 2 - item.box.height() / 2, -4)
        room = None
        if item.room is not None:
            room = round(item.x + item.room - r.right())
        out.append(Sticker(id=n, src=item.ja, x=round(r.left()), y=round(r.top()), w=round(r.width()), h=round(r.height()),
                           char_h=item.px, n_lines=1, patch=np.zeros((4, 4), np.uint8), px=round(r.left()),
                           py=round(r.top()), bg=item.bg.getRgb()[:3], fg=item.fg.getRgb()[:3], bold=item.bold,
                           centered=item.box is not None, room_r=room))
    return out


INSIDE = QRectF(APP.left(), APP.top() + TITLE_BAR, APP.width(), APP.height() - TITLE_BAR)  # what the frame covers


class Lens:
    """What the frame's buttons are wired to: nothing here (the frame is only drawn)."""
    paused = False

    class overlay:  # noqa: N801  (as ScreenLens.overlay)
        @staticmethod
        def setGeometry(rect) -> None:  # noqa: N802
            pass

    def __getattr__(self, name):
        return lambda *args: None


@lru_cache(maxsize=1)
def frame_window():
    """The real frame (sst.screen.lens.FrameWindow), sized to the app's page, never shown."""
    from sst.screen.lens import BORDER, TOOLBAR, FrameWindow
    frame = FrameWindow(Lens())
    frame.resize(round(INSIDE.width()) + 2 * BORDER, round(INSIDE.height()) + TOOLBAR + BORDER)
    frame.grab()  # laid out at that size (a hidden window gets its resize when first drawn)
    return frame


@lru_cache(maxsize=8)
def frame_image(status: str, original: bool = False) -> QImage:
    """The frame with `status` on its toolbar (the eye pressed while it shows the original); its inside cut out, as the
    window's mask cuts it."""
    frame = frame_window()
    frame.status.setText(status)
    frame.original_button.setChecked(original)
    demo.settle()
    image = frame.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    p = QPainter(image)  # in the window's own pixels: the image keeps its 3x
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
    p.fillRect(frame.interior(), Qt.GlobalColor.transparent)
    p.end()
    return image


def button_at(tip: str) -> QPointF:
    """The centre of the frame's toolbar button with this tooltip, on the stage."""
    from sst.screen.lens import BORDER, TOOLBAR
    button = next(b for b in frame_window().toolbar.findChildren(QToolButton) if b.toolTip() == tip)
    c = button.geometry().center()
    return QPointF(INSIDE.left() - BORDER + c.x() + 0.5, INSIDE.top() - TOOLBAR + c.y() + 0.5)


class ScreenClip:
    """The app; Ctrl+Alt+S; the frame; the strings waiting (dashed), then English; the eye: the original, and back; the
    frame closed. The app alone at the start and the end, so the clip loops."""
    KEYS, DOWN, UP, KEYS_OFF = 0.5, 0.95, 1.25, 1.85
    FRAME, PENDING, ARRIVE = 1.3, 1.9, 2.85
    EYE, EYE_OFF, CLOSE = 6.0, 7.5, 9.3  # the clicks
    duration = 10.9

    def __init__(self):
        self.stickers = stickers()
        self.english = {item.ja: item.en for item in items()}
        n = len(self.stickers)
        self.statuses = [(0.0, "Starting…"), (self.PENDING, f"0 translated · translating {n}…"),
                         (self.ARRIVE, f"{n} translated"), (self.EYE + 0.05, "Showing the original"),
                         (self.EYE_OFF + 0.05, f"{n} translated")]
        eye, close = button_at("Show the original"), button_at("Stop the screen translator")
        self.path = [(4.4, QPointF(980, 470)), (5.75, eye), (self.EYE_OFF + 0.3, eye), (9.05, close), (9.6, close),
                     (10.3, QPointF(close.x() + 40, close.y() + 70))]

    def frame(self, t: float) -> QImage:
        image = demo.backdrop().copy()
        p = demo.painter(image)
        draw_app(p)
        shown = ramp(t, self.FRAME, 0.3) * (1 - ramp(t, self.CLOSE + 0.08, 0.3))
        if shown > 0:
            self.draw_stickers(p, t, shown)
            from sst.screen.lens import BORDER, TOOLBAR
            status = next(s for at, s in reversed(self.statuses) if at <= t)
            original = self.EYE + 0.05 <= t < self.EYE_OFF + 0.05
            with faded(p, shown):
                p.save()
                p.translate(INSIDE.left() - BORDER, INSIDE.top() - TOOLBAR - 6 * (1 - shown))
                draw_image(p, frame_image(status, original), 0, 0)
                p.restore()
        pressed = ramp(t, self.DOWN, 0.1) * (1 - ramp(t, self.UP, 0.1))
        demo.draw_keys(p, [("Ctrl", pressed), "+", ("Alt", pressed), "+", ("S", pressed)],
                       ramp(t, self.KEYS, 0.3) * (1 - ramp(t, self.KEYS_OFF, 0.35)))
        here = ramp(t, self.path[0][0], 0.3) * (1 - ramp(t, self.path[-1][0] - 0.4, 0.4))
        cursor(p, glide(t, self.path), here, max(tap(t, self.EYE), tap(t, self.EYE_OFF), tap(t, self.CLOSE)))
        p.end()
        return image

    def draw_stickers(self, p: QPainter, t: float, shown: float) -> None:
        """Dashed while Gemini translates; then the English, all at once (one request for the screen's strings); gone
        while the eye shows the original."""
        from sst.screen.render import paint_stickers
        waiting = ramp(t, self.PENDING, 0.25) * (1 - ramp(t, self.ARRIVE, 0.2))
        english = ramp(t, self.ARRIVE, 0.25) * (1 - ramp(t, self.EYE, 0.18)) + ramp(t, self.EYE_OFF, 0.2)
        english = min(english, 1.0)
        for amount, lookup in ((waiting * shown, lambda src: None), (english * shown, self.english.get)):
            if amount > 0:
                with faded(p, amount):
                    paint_stickers(p, self.stickers, lookup, (0, 0), 1.0, clip_w=INSIDE.right())


# ---------------------------------------------------------------- 2. the text translator

JAPANESE = ("来週の定例会議は木曜日の午後3時からに変更になりました。"
            "資料は水曜日までに共有フォルダにアップロードしてください。")
ENGLISH = demo.ENGLISH  # "Next week's regular meeting has moved to Thursday at 3 PM. Please upload the materials ..."
CHUNKS = ["来週の", "定例会議は", "木曜日の", "午後3時から", "に変更に", "なりました。", "資料は", "水曜日までに",
          "共有フォルダに", "アップロード", "してください。"]  # as an IME commits them, a phrase at a time
assert "".join(CHUNKS) == JAPANESE
SIZE = (1000, 680)  # the window, as in the screenshots


@lru_cache(maxsize=1)
def text_window():
    """The real window on Translate's Text tab, Detect language → English, nothing typed yet."""
    import make_site_screenshots as site

    from sst import window as w
    site.parakeet.find_model = lambda: Path(".")  # as if Parakeet were downloaded: Rflow shows as ready
    app = site.preview()
    app.settings.text_from, app.settings.text_to = "", "English"
    window = w.MainWindow(app)
    window.apply_theme("dark")
    window.resize(*SIZE)
    window.set_status("Ready: hold Ctrl+Win · cleanup: gemini-3.5-flash-lite", True)
    window.show_page("translate")
    page = window.pages["translate"]
    page.tabs.set_current("text")
    page._tab_chosen("text")
    page.text.refresh()
    demo.settle()
    window.grab()  # laid out at that size: where the translator is is measured from it
    return window


def translator():
    return text_window().pages["translate"].text


def where(widget) -> QRectF:
    """A widget's rectangle in the window."""
    top_left = widget.mapTo(text_window(), QPoint(0, 0))
    return QRectF(top_left.x(), top_left.y(), widget.width(), widget.height())


@lru_cache(maxsize=32)
def text_state(typed: str, output: str, status: str, swapped: bool) -> QImage:
    """The window with `typed` in the left pane, `output` on the right and `status` under it; swapped: English →
    Japanese. Set as the translator itself sets them (its own _typed for the count and Detect language), with its pause
    timer stopped: the translations are the example's, nothing is asked."""
    text = translator()
    for box, value in ((text.source, "English" if swapped else ""), (text.target, "Japanese" if swapped else "English")):
        box.blockSignals(True)
        box.setCurrentIndex(max(0, box.findData(value)))
        box.blockSignals(False)
    text.input.blockSignals(True)
    text.input.setPlainText(typed)
    text.input.blockSignals(False)
    text._typed()
    text.timer.stop()
    text.output.setPlainText(output)
    text._say(status)
    cursor_at = text.input.textCursor()
    cursor_at.movePosition(cursor_at.MoveOperation.End)
    text.input.setTextCursor(cursor_at)
    demo.settle()
    return text_window().grab().toImage()


class TextClip:
    """The window, then closer on the translator; Japanese typed, a phrase at a time; after the pause, translating, then
    the English; the swap clicked: English → Japanese, the English translated after the pause; back out, to the start."""
    ZOOM_IN, TYPE, TYPED = (0.3, 1.1), 1.35, 3.6
    ASK, ANSWER = TYPED + 0.8, TYPED + 1.25  # the 800 ms pause, then Gemini's answer
    SWAP = 6.6
    ASK2, ANSWER2 = SWAP + 0.8, SWAP + 1.2
    ZOOM_OUT = (9.6, 10.5)
    duration = 10.5 + LOOP

    def __init__(self):
        card = where(translator().widget)
        self.whole = (SIZE[0] / 2, SIZE[1] / 2, 0.98, 640, 360)  # the window's point x, y at the stage's x, y
        self.close = (card.center().x(), card.center().y(), 1180 / card.width(), 640, 372)
        swap = where(translator().swap).center()
        self.swap_at = self.stage_point(QPointF(swap.x(), swap.y()), self.close)
        self.path = [(self.ANSWER + 0.3, QPointF(1010, 640)), (self.SWAP - 0.25, self.swap_at), (self.SWAP + 0.6, self.swap_at),
                     (self.SWAP + 1.4, self.swap_at + QPointF(70, 120))]
        self.first = self.at(0.0)
        self.end = None

    @staticmethod
    def stage_point(point: QPointF, view: tuple) -> QPointF:
        fx, fy, scale, cx, cy = view
        return QPointF(cx + (point.x() - fx) * scale, cy + (point.y() - fy) * scale)

    def state(self, t: float) -> QImage:
        if t < self.SWAP + 0.05:
            n = sum(1 for k in range(len(CHUNKS)) if t >= self.TYPE + k * (self.TYPED - self.TYPE) / (len(CHUNKS) - 1))
            typed = "".join(CHUNKS[:n])
            if t < self.ASK:
                return text_state(typed, "", "", False)
            return text_state(typed, ENGLISH if t >= self.ANSWER else "", "0.9 s" if t >= self.ANSWER else "Translating…",
                              False)
        if t < self.ASK2:  # swapped: the translation is the text now; the one on the right stays until the new one
            return text_state(ENGLISH, ENGLISH, "0.9 s", True)
        return text_state(ENGLISH, JAPANESE if t >= self.ANSWER2 else ENGLISH, "0.8 s" if t >= self.ANSWER2 else "Translating…",
                          True)

    def at(self, t: float) -> QImage:
        k = ramp(t, self.ZOOM_IN[0], self.ZOOM_IN[1] - self.ZOOM_IN[0]) * (1 - ramp(t, self.ZOOM_OUT[0],
                                                                                  self.ZOOM_OUT[1] - self.ZOOM_OUT[0]))
        fx, fy, scale, cx, cy = (mix(a, b, k) for a, b in zip(self.whole, self.close, strict=True))
        window = self.state(t)
        dpr = window.devicePixelRatio()
        rect = QRectF(0, 0, window.width() / dpr, window.height() / dpr)
        image = demo.backdrop().copy()
        p = demo.painter(image)
        p.save()
        p.translate(cx, cy)
        p.scale(scale, scale)
        p.translate(-fx, -fy)
        theme.outer_shadow(p, rect, 14, [(0, 18, 44, theme.rgba(0, 0, 0, .55)), (0, 2, 6, theme.rgba(0, 0, 0, .3))],
                           demo.STAGE.s)
        p.setClipPath(theme.rounded(rect, 14))
        p.drawImage(rect, window)
        p.setClipping(False)
        p.setPen(QPen(theme.rgba(255, 255, 255, .08), 1 / scale))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)
        p.restore()
        here = ramp(t, self.path[0][0], 0.3) * (1 - ramp(t, self.path[-1][0] - 0.4, 0.4))
        cursor(p, glide(t, self.path), here, tap(t, self.SWAP))
        p.end()
        return image

    def frame(self, g: float) -> QImage:
        settled = self.duration - LOOP
        if g <= settled:
            return self.at(g)
        if self.end is None:
            self.end = self.at(settled)
        return crossfade(self.end, self.first, demo.ease((g - settled) / LOOP))


# ---------------------------------------------------------------- the tour's way in, and the files

def _clip(kind):
    """(seconds, frame(t)) for a clip, made on its first frame: nothing is drawn while this file is imported. Each frame
    is drawn on demo.WIDE and the caller's stage put back after (the portrait cut draws on another)."""
    made = []

    def frame(t: float) -> QImage:
        before = demo.STAGE
        if before is not demo.WIDE:
            demo.use(demo.WIDE)
        try:
            if not made:
                made.append(kind())
            return made[0].frame(t)
        finally:
            if before is not demo.WIDE:
                demo.use(before)
    return kind.duration, frame


CLIPS = {"screen": _clip(ScreenClip), "text": _clip(TextClip)}
POSTERS = {"screen": 4.5, "text": 5.4}  # the English over the app; the translation beside the Japanese


def main() -> None:
    try:
        if not JOB:  # each clip in a process of its own, side by side
            jobs = [subprocess.Popen([sys.executable, __file__, job]) for job in JOBS]
            sys.exit(max(job.wait() for job in jobs))
        if JOB not in JOBS:
            sys.exit(f"usage: make_feature_clips_c.py [{' | '.join(JOBS)}]")
        QApplication([])
        theme.load_fonts()
        clips.MEDIA.mkdir(parents=True, exist_ok=True)
        demo.use(demo.WIDE)
        seconds, frame = CLIPS[JOB]
        clips.encode(f"clip-{JOB}.mp4", seconds, frame)
        clips.poster(JOB, frame(POSTERS[JOB]))
    finally:
        shutil.rmtree(demo.DATA, ignore_errors=True)


if __name__ == "__main__":
    main()
