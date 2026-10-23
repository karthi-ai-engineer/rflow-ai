"""Render six short GIFs of Rflow for LinkedIn posts, each one idea told in about five seconds without sound: a big caption
at the top says the action, the real Rflow moment plays below it, the result is held long enough to read, and the last
frame leads back into the first.

  1-talk-it-types.gif         hold Ctrl+Win, speak; the sentence is typed into a chat box
  2-make-it-concise.gif       a rambling paragraph, "make it concise", the short version
  3-bullet-points.gif         the same paragraph, "bullet points", the tidy list
  4-numbers-written-right.gif what was said, and how the formatting stage writes it (25%, 3:30 PM, $25,000, an email)
  5-translate-any-text.gif    Japanese selected, Ctrl+C twice, the real Translate popup in English
  6-translate-anything.gif    a video in Spanish and a meeting in Japanese, translated live on the real translation bar

LinkedIn shows GIFs in the feed only up to 5 MB, and most people see them on a phone: so portrait 4:5 (900x1125), 15
frames a second, one palette per GIF chosen from its own frames (no dithering, so the text stays crisp), frames that
don't change merged into longer ones, and each frame storing only the pixels that moved.

Everything is drawn like the feature clips (scripts/make_demo_video.py, make_feature_clips_a.py and _b.py, imported, not
copied): the pill, the Translate popup and the translation bar are Rflow's own widgets with example data, grabbed off the
screen, so nothing appears, takes focus, presses keys or touches the clipboard. The transforms' results are the ones the
real TransformGuard accepts (checked before drawing: the run stops if one isn't), and the numbers come from the real
formatting stage.
Pillow is added only for the run:

  uv run --no-sync --with pillow python scripts/make_linkedin_gifs.py [1 2 ...] [--out DIR] [--frames DIR]
"""
import argparse
import math
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]  # this checkout's Rflow, whichever one is installed
import make_demo_video as demo  # noqa: E402  (sets Qt's platform and a scratch APPDATA before Qt starts)
import make_feature_clips_a as clips_a  # noqa: E402
import make_feature_clips_b as clips_b  # noqa: E402

os.environ["QT_SCALE_FACTOR"] = "3"  # Rflow's parts grabbed at 3x: sharp when drawn a little larger than life

import numpy as np  # noqa: E402
from make_demo_video import POP, T, alpha, blend, clamp, faded, mix, ramp, write  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPainterPath, QPen  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst import theme  # noqa: E402

FPS = 15
W, H = 900, 1125  # LinkedIn's portrait 4:5
SW = 540  # the stage: 540x675, drawn at 900/540 (Rflow's parts at about 167%, as big as on a phone screen they need be)
WHITE = QColor("#FFFFFF")
LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
TOP_LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
CAPTION = QRectF(16, 30, SW - 32, 118)  # two lines at most, clear of LinkedIn's own buttons
CAPTION_PX = 36
BASE = demo.Stage(W, H, W / SW, QRectF(20, 172, 500, 300), 22, 17, 27, 520, 1.45, 600, 1.15, 0, CAPTION_PX, 700)


# ---------------------------------------------------------------- what every GIF has

def draw_caption(p: QPainter, captions: list[tuple[float, str]], t: float) -> None:
    """The caption at t: (from, text) each, lines split by hand. The first is there from the first frame (the loop
    comes back to it); a later one comes in as the one before goes."""
    i = max(k for k, (start, _) in enumerate(captions) if start <= t or k == 0)
    start, text = captions[i]
    if i == 0:
        write(p, CAPTION, text, CAPTION_PX, 700, WHITE)
        return
    if t < start + 0.18:
        with faded(p, 1 - ramp(t, start, 0.18)):
            write(p, CAPTION, captions[i - 1][1], CAPTION_PX, 700, WHITE)
    k = ramp(t, start + 0.15, 0.3)
    with faded(p, k):
        write(p, CAPTION.translated(0, 6 * (1 - k)), text, CAPTION_PX, 700, WHITE)


def check_caption(text: str) -> None:
    """Each caption fits the frame on its own lines (two at most): a phone shows it no smaller."""
    metrics = QFontMetricsF(theme.font(CAPTION_PX, 700))
    lines = text.split("\n")
    widest = max(metrics.horizontalAdvance(line) for line in lines)
    if len(lines) > 2 or widest > CAPTION.width():
        sys.exit(f"the caption {text!r} doesn't fit: {len(lines)} lines, {widest:.0f} of {CAPTION.width():.0f} px")


def draw_mark(p: QPainter) -> None:
    """Rflow's mark and name, small, in the bottom right corner."""
    width = 18 + 7 + QFontMetricsF(theme.font(13, 600)).horizontalAdvance("Rflow")
    x = SW - 18 - width
    p.drawImage(QRectF(x, 649, 18, 18), demo.logo())
    write(p, QRectF(x + 25, 646, 80, 24), "Rflow", 13, 600, T["text3"], LEFT)


def wrapped(text: str, px: float, width: float, weight: int = 400) -> list[str]:
    """The text's lines, wrapped at word boundaries to `width`."""
    metrics, lines = QFontMetricsF(theme.font(px, weight)), [""]
    for word in text.split(" "):
        line = f"{lines[-1]} {word}".strip()
        if lines[-1] and metrics.horizontalAdvance(line) > width:
            lines.append(word)
        else:
            lines[-1] = line
    return lines


def window(p: QPainter, r: QRectF, radius: float = 14, fill: str = "#1C1F25") -> None:
    """A plain dark window: shadow, body, hairline edge."""
    theme.outer_shadow(p, r, radius, [(0, 18, 44, theme.rgba(0, 0, 0, .55)), (0, 2, 6, theme.rgba(0, 0, 0, .30))], BASE.s)
    p.fillPath(theme.rounded(r, radius), QColor(fill))
    p.setPen(QPen(theme.rgba(255, 255, 255, .08), 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


def avatar(p: QPainter, c: QPointF, radius: float, initials: str, tint: str) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(blend(QColor("#2A2E3A"), T[tint], 0.42))
    p.drawEllipse(c, radius, radius)
    write(p, QRectF(c.x() - radius, c.y() - radius, 2 * radius, 2 * radius), initials, radius * 0.72, 600, T["text"])


class Gif:
    """One GIF: its stage, its length, its captions and what's drawn between them."""
    name = ""
    duration = 6.0
    stage = BASE
    captions: list[tuple[float, str]] = []
    checks: tuple[float, ...] = ()  # moments to look at (exported as PNG with --frames)
    SHADOWS = False  # a window that changes size over the glows: its shadow dithered (see write_gif)

    def draw(self, p: QPainter, t: float) -> None:
        raise NotImplementedError

    def frame(self, t: float) -> np.ndarray:
        image = demo.backdrop().copy()
        p = demo.painter(image)
        self.draw(p, t)
        draw_mark(p)
        draw_caption(p, self.captions, t)
        p.end()
        return np.ascontiguousarray(demo.pixels(image)[..., 2::-1])


# ---------------------------------------------------------------- 1. talk, it types

class TalkItTypes(Gif):
    """Ctrl+Win held, the pill listening while the words are heard (the line under it), let go: the sentence typed into
    a chat's message box, the pill's "Typed"."""
    name = "1-talk-it-types.gif"
    duration = 6.2
    text = demo.DICTATED  # "Can you review the pull request before lunch and leave a comment if anything looks wrong?"
    KEYS, DOWN, UP, TYPED, HIDE, CLEAR = 0.1, 0.35, 2.7, 3.15, 5.2, 5.4
    SPEECH = (0.55, 2.5)
    CHAT = QRectF(20, 172, 500, 262)
    BOX_PX, BOX_LINE = 17, 25
    stage = replace(BASE, pill_y=548, keys_y=478, keys_scale=1.1)
    captions = [(0, "Hold Ctrl + Win.\nSpeak."), (3.3, "It types.\nIn any app."), (5.55, "Hold Ctrl + Win.\nSpeak.")]
    checks = (1.6, 2.9, 4.6)

    def __init__(self):
        self.pill = demo.PillTrack(self.DOWN + 0.05, self.HIDE, [(self.DOWN + 0.05, "recording", False, ""),
                                                                 (self.UP, "transcribing", True, ""),
                                                                 (self.TYPED, "typed", False, "")], [self.SPEECH])
        self.box = QRectF(self.CHAT.left() + 14, self.CHAT.bottom() - 14 - 80, self.CHAT.width() - 28, 80)

    def draw(self, p, t):
        self.chat(p)
        typing = ramp(t, self.TYPED - 0.05, 0.55) if t >= self.TYPED - 0.05 else 0.0
        text, gone = demo.words_typed(self.text, typing), ramp(t, self.CLEAR, 0.4)
        at = QPointF(self.box.left() + 16, self.box.top() + 13)
        width = self.box.width() - 32 - 44
        if not text or gone >= 1:
            write(p, QRectF(at.x(), at.y() - 2, width, self.BOX_LINE), "Message Release team", self.BOX_PX, 400,
                  T["text3"], TOP_LEFT)
        lay = demo.layout(text, self.BOX_PX, width, self.BOX_LINE) if text else None
        if lay is not None and gone < 1:
            with faded(p, 1 - gone):
                p.setPen(T["text"])
                lay.draw(p, at)
        if gone == 0 and lay is not None:
            line = lay.lineAt(lay.lineCount() - 1)
            caret = QPointF(at.x() + demo._x(line, len(text)), at.y() + line.y())
        else:
            caret = at
        on = 1.0 if 0 < typing < 1 else clips_a.blink(t, self.duration)
        clips_a.caret(p, caret, on, self.BOX_PX + 6)
        pressed = ramp(t, self.DOWN, 0.12) * (1 - ramp(t, self.UP, 0.12))
        demo.draw_keys(p, [("Ctrl", pressed), "+", ("Win", pressed)],
                       ramp(t, self.KEYS, 0.25) * (1 - ramp(t, self.UP + 0.2, 0.3)))
        self.pill.draw(p, t)
        self.heard(p, t)

    def chat(self, p: QPainter) -> None:
        """A team chat, plain: its name, a message from a colleague, and the message box the dictation goes into."""
        r = self.CHAT
        window(p, r)
        p.save()
        p.setClipPath(theme.rounded(r, 14))
        p.fillRect(QRectF(r.left(), r.top(), r.width(), 46), QColor("#16181D"))
        p.fillRect(QRectF(r.left(), r.top() + 45, r.width(), 1), theme.rgba(255, 255, 255, .06))
        p.restore()
        avatar(p, QPointF(r.left() + 30, r.top() + 23), 13, "RT", "iris")
        write(p, QRectF(r.left() + 52, r.top(), 240, 46), "Release team", 15, 600, T["text"], LEFT)
        write(p, QRectF(r.right() - 160, r.top(), 144, 46), "5 members", 12, 400, T["text3"],
              Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        avatar(p, QPointF(r.left() + 34, r.top() + 80), 16, "AK", "violet")
        write(p, QRectF(r.left() + 60, r.top() + 60, 200, 22), "Arjun", 13, 600, T["text"], LEFT)
        write(p, QRectF(r.left() + 108, r.top() + 60, 80, 22), "10:42", 12, 400, T["text3"], LEFT)
        message = "I'm free this morning if anything needs a look."
        bubble = QRectF(r.left() + 60, r.top() + 86, QFontMetricsF(theme.font(15)).horizontalAdvance(message) + 28, 38)
        p.fillPath(theme.rounded(bubble, 12), QColor("#262A33"))
        write(p, bubble, message, 15, 400, T["text"])
        b = self.box
        p.fillPath(theme.rounded(b, 12), QColor("#14161B"))
        p.setPen(QPen(alpha(T["iris"], 0.45), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(b.adjusted(0.6, 0.6, -0.6, -0.6), 12, 12)
        send = QPointF(b.right() - 26, b.bottom() - 24)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T["iris"])
        p.drawEllipse(send, 15, 15)
        p.drawPixmap(QPointF(send.x() - 8, send.y() - 8), theme.icon_pixmap("arrow-right", T["on_primary"].name(), 16,
                                                                           2 * BASE.s))

    def heard(self, p: QPainter, t: float) -> None:
        """What is being said, word by word, under the pill: on the lines the whole sentence will take, so nothing
        shifts as it grows."""
        start, end = self.SPEECH
        shown = ramp(t, start - 0.1, 0.25) * (1 - ramp(t, self.TYPED + 0.2, 0.35))
        if shown <= 0:
            return
        px, height, top = 16, 23, 590
        said = demo.words_typed(self.text, clamp((t - start) / (end - start - 0.15)))
        lines = wrapped(f"“{self.text}”", px, 430)
        metrics, n = QFontMetricsF(theme.font(px)), len(said) + 1  # the opening quote
        speaking = demo.voice(t, [self.SPEECH]) if t < end else 0.0
        with faded(p, shown):
            first = SW / 2 - metrics.horizontalAdvance(lines[0]) / 2
            demo.icon(p, "mic", blend(T["text3"], T["live"], clamp(speaking * 1.6)), first - 24, top + 3)
            for row, line in enumerate(lines):
                piece = line[:max(0, n)]
                n -= len(line) + 1
                if said and piece:
                    if said == self.text and row == len(lines) - 1:
                        piece = line
                    x = SW / 2 - metrics.horizontalAdvance(line) / 2
                    write(p, QRectF(x, top + row * height, 460, height), piece, px, 400, T["text2"], TOP_LEFT)


# ---------------------------------------------------------------- 2 and 3. say a transform

class Transform(Gif):
    """The paragraph selected, Ctrl+Win held, the command said: the selection replaced by the result, the window
    shrinking to it, the pill naming the transform; then the paragraph back, as at the start."""
    KEY = ""
    BEFORE = AFTER = ""  # the paragraph and the result: the feature clip's (make_feature_clips_b), unless given here
    COUNT = False  # the word count under the window
    duration = 6.4
    SELECT, KEYS, DOWN, UP, DONE, HIDE, BACK = 0.55, 0.95, 1.15, 2.3, 2.85, 5.1, 5.4
    SPEECH = (1.3, 2.1)
    TOP = 150
    stage = replace(BASE, editor=QRectF(20, TOP, 500, 300), pad=24, text_px=19, line=30, keys_scale=1.05)
    checks = (0.3, 1.9, 4.2)

    def __init__(self):
        from sst.transform import TRANSFORMS
        clips_b.STAGE = self.stage  # its result drawing reads the stage's text size, line and width
        clips_b.block.cache_clear()
        self.before, self.after = self.BEFORE or clips_b.PARAGRAPH, self.AFTER or clips_b.RESULTS[self.KEY]
        if self.BEFORE:  # a pair of its own: one Rflow would type (the real guard accepts it)
            from sst.transform import TransformGuard
            got = TransformGuard().validate(self.before, self.after, self.KEY)
            if not got.accepted:
                sys.exit(f"TransformGuard rejects the {self.KEY} result: {got.reasons}")
        self.before_h, self.after_h = self.height(self.before), self.height(self.after)
        bottom = self.TOP + max(self.before_h, self.after_h)  # the pill, the words said and the keys under the window
        self.stage = clips_b.STAGE = demo.STAGE = replace(self.stage, pill_y=bottom + 46, keys_y=bottom + 148)
        self.command_y = bottom + 98
        if self.stage.keys_y > 626:
            sys.exit(f"{self.name}: the window is too tall for the keys under it")
        self.command = clips_b.COMMANDS[self.KEY]
        self.captions = [(0, f"Say: “{self.command}”")]
        self.pill = demo.PillTrack(self.DOWN + 0.05, self.HIDE, [
            (self.DOWN + 0.05, "recording", False, ""), (self.UP, "transforming", False, ""),
            (self.DONE, "transformed", False, f"Transformed: {TRANSFORMS[self.KEY].name}")], [self.SPEECH])

    def height(self, text: str) -> float:
        """The window's height around the text: its title bar, the text's top margin, the lines, a bottom margin."""
        s = self.stage
        if "\n" not in text:
            body = demo.text_layout(text).lineCount() * s.line
        else:
            width = s.editor.width() - 2 * s.pad
            body = sum(s.line + 4 if line.startswith("**") else s.line * clips_b.block(line[2:], 400, width - 28).lineCount()
                       for line in text.splitlines())
        return demo.BAR + 32 + body + 12

    def draw(self, p, t):
        k = ramp(t, self.DONE + 0.05, 0.6) * (1 - ramp(t, self.BACK, 0.45))  # the window eases to the text's height
        editor = QRectF(20, self.TOP, 500, mix(self.before_h, self.after_h, k))
        demo.STAGE = replace(self.stage, editor=editor)
        demo.draw_editor(p)
        demo.STAGE = self.stage
        if self.COUNT:  # how long it is, under the window: the words before, then after
            for text, amount, colour in ((self.before, 1 - k, T["text3"]), (self.after, k, T["iris"])):
                if amount > 0:
                    with faded(p, amount):
                        write(p, QRectF(editor.right() - 200, editor.bottom() + 8, 196, 22), f"{len(text.split())} words",
                              14, 500, colour, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        gone, came = ramp(t, self.DONE - 0.15, 0.2), ramp(t, self.DONE + 0.05, 0.3)  # one text out, then the other in
        out, back = ramp(t, self.BACK - 0.1, 0.2), ramp(t, self.BACK + 0.4, 0.3)  # the paragraph back once there's room
        paragraph = self.before
        if gone < 1:
            with faded(p, 1 - gone):
                demo.draw_text(p, paragraph, round(len(paragraph) * ramp(t, self.SELECT, 0.5)))
        if came > 0 and out < 1:
            with faded(p, came * (1 - out)):
                clips_b.draw_result(p, self.after, 1 - ramp(t, self.DONE + 0.5, 1.0))
        if back > 0:
            with faded(p, back):
                demo.draw_text(p, paragraph)
        if t < self.SELECT or t > self.BACK + 0.6:
            clips_a.caret(p, demo.caret_at(paragraph, len(paragraph)), clips_a.blink(t, self.duration),
                          self.stage.text_px + 8)
        pressed = ramp(t, self.DOWN, 0.12) * (1 - ramp(t, self.UP, 0.12))
        demo.draw_keys(p, [("Ctrl", pressed), "+", ("Win", pressed)],
                       ramp(t, self.KEYS, 0.25) * (1 - ramp(t, self.UP + 0.2, 0.3)))
        self.pill.draw(p, t)
        self.said(p, t)

    def said(self, p: QPainter, t: float) -> None:
        """The command, word by word as it's heard, in quotes under the pill."""
        start, end = self.SPEECH
        said = demo.words_typed(self.command, clamp((t - start + 0.15) / (end - start)))
        shown = ramp(t, start, 0.2) * (1 - ramp(t, self.DONE + 0.1, 0.3))
        if not said or shown <= 0:
            return
        x = SW / 2 - QFontMetricsF(theme.font(24, 500)).horizontalAdvance(f"“{self.command}”") / 2
        with faded(p, shown):
            write(p, QRectF(x, self.command_y - 20, 460, 40), f"“{said}" + ("”" if said == self.command else ""), 24, 500,
                  T["text"], LEFT)


class Concise(Transform):
    """A rambling message about moving a meeting, about a third as long after "make it concise": its name, day and time
    kept."""
    name, KEY, COUNT, SHADOWS = "2-make-it-concise.gif", "concise", True, True
    checks = (0.3, 1.9, 3.1, 4.2, 5.75)
    TOP = 140  # a longer paragraph: the window a little higher, its text a little smaller
    stage = replace(Transform.stage, editor=QRectF(20, TOP, 500, 300), text_px=18, line=28)
    BEFORE = ("Hey team, so, um, I just wanted to quickly mention that, basically, we need to kind of move tomorrow's design "
              "review, because Maya is actually going to be out in the morning, so, like, the plan is that we push the design "
              "review to Thursday at 3:30 PM instead, which honestly works better for everyone anyway, so yeah, please, you "
              "know, just update your calendars and sort of let me know that you saw this, thanks.")
    AFTER = ("Maya is out tomorrow morning, so the design review moves to Thursday at 3:30 PM. Please update your calendars "
             "and confirm you saw this.")


class Bullets(Transform):
    name, KEY = "3-bullet-points.gif", "bullets"


# ---------------------------------------------------------------- 4. numbers, written right

class Numbers(Gif):
    """Four things said, one under the other: what was heard (small, the spoken numbers lit), then what Rflow types
    (large, the parts the formatting stage wrote in Iris)."""
    name = "4-numbers-written-right.gif"
    duration = 6.0
    PHRASES = ["sales went up twenty five percent", "let's meet at three thirty pm", "the budget is twenty five thousand dollars",
               "send it to john dot smith at gmail dot com"]
    START, STEP, OUT = 0.2, 0.95, 5.3
    TOP, ROW, GAP = 172, 104, 12
    TYPED_PX = 27
    captions = [(0, "Numbers,\nwritten right.")]
    checks = (1.3, 4.5)

    def __init__(self):
        self.lines = [clips_a.run_formatter(said) for said in self.PHRASES]
        for line in self.lines:
            width = QFontMetricsF(theme.font(self.TYPED_PX, 500)).horizontalAdvance(line.text)
            if width > 452:
                sys.exit(f"{line.text!r} is too wide for its card: {width:.0f} px")

    def draw(self, p, t):
        out = ramp(t, self.OUT, 0.4)
        with faded(p, 1 - out):
            for i, line in enumerate(self.lines):
                self.card(p, t - self.START - i * self.STEP, line, self.TOP + i * (self.ROW + self.GAP))

    def card(self, p: QPainter, u: float, line, y: float) -> None:
        """One phrase at u seconds into its turn: the card, the words heard, then the line typed and lit."""
        shown = ramp(u, 0.0, 0.25)
        if shown <= 0:
            return
        r = QRectF(20, y + 8 * (1 - shown), 500, self.ROW)
        heard = demo.words_typed(line.said, clamp(u / 0.45))
        typed, lit = ramp(u, 0.5, 0.25), ramp(u, 0.65, 0.3)
        with faded(p, shown):
            theme.paint_surface(p, "float", r, 16, popup=True, dpr=BASE.s)
            speaking = demo.voice(u, [(0.0, 0.45)]) if u < 0.45 else 0.0
            demo.icon(p, "mic", blend(POP["text3"], T["live"], clamp(speaking * 1.6)), r.left() + 20, r.top() + 17)
            x = r.left() + 44
            for text, changed in clips_a.pieces(heard, [(a, min(b, len(heard))) for a, b in line.heard if a < len(heard)]):
                w = QFontMetricsF(theme.font(15)).horizontalAdvance(text)
                write(p, QRectF(x, r.top() + 12, w + 4, 26), text, 15, 400, blend(POP["text2"], T["iris"], 0.9 * lit if changed
                                                                                     else 0), LEFT)
                x += w
            if typed > 0:
                with faded(p, typed):
                    self.typed(p, r.left() + 22, r.top() + 48 + 6 * (1 - typed), line, lit)

    def typed(self, p: QPainter, x: float, y: float, line, lit: float) -> None:
        h = QFontMetricsF(theme.font(self.TYPED_PX, 500)).height()
        for text, changed in clips_a.pieces(line.text, line.spans):
            weight = 600 if changed else 500
            w = QFontMetricsF(theme.font(self.TYPED_PX, weight)).horizontalAdvance(text)
            if changed and lit > 0:
                p.fillPath(theme.rounded(QRectF(x - 3, y - 3, w + 9, h + 6), 8), alpha(T["iris"], 0.2 * lit))
            write(p, QRectF(x, y, w + 4, h), text, self.TYPED_PX, weight, blend(T["text"], T["iris"], lit if changed else 0),
                  TOP_LEFT)
            x += w


# ---------------------------------------------------------------- 5. translate any text

class Translate(Gif):
    """Japanese selected; Ctrl+C, C; the real popup by the selection, translating, then the English."""
    name = "5-translate-any-text.gif"
    duration = 5.5
    SELECT, KEYS, CTRL, FIRST, SECOND, KEYS_OFF = 0.3, 0.95, 1.1, 1.3, 1.6, 2.0
    OPEN, DONE, CLOSE = 2.15, 2.75, 4.55
    POPUP_SCALE = 1.08
    stage = replace(BASE, editor=QRectF(20, 190, 500, 192), pad=24, text_px=21, line=34, keys_y=452, keys_scale=1.25)
    captions = [(0, "Select text.\nPress Ctrl + C twice."), (2.8, "Translate any text."),
                (4.7, "Select text.\nPress Ctrl + C twice.")]
    checks = (1.35, 2.4, 3.8)

    def draw(self, p, t):
        demo.draw_editor(p)
        japanese = demo.JAPANESE
        selected = 0 if t > self.CLOSE + 0.15 else round(len(japanese) * ramp(t, self.SELECT, 0.6))
        demo.draw_text(p, japanese, selected)
        if t < self.SELECT or t > self.CLOSE + 0.15:
            clips_a.caret(p, demo.caret_at(japanese, len(japanese)), clips_a.blink(t, self.duration), self.stage.text_px + 8)

        def tap(at: float) -> float:
            return ramp(t, at, 0.08) * (1 - ramp(t, at + 0.16, 0.08))
        ctrl = ramp(t, self.CTRL, 0.1) * (1 - ramp(t, self.SECOND + 0.3, 0.12))
        demo.draw_keys(p, [("Ctrl", ctrl), "+", ("C", tap(self.FIRST)), ",", ("C", tap(self.SECOND))],
                       ramp(t, self.KEYS, 0.25) * (1 - ramp(t, self.KEYS_OFF, 0.25)))
        shown = ramp(t, self.OPEN, 0.3) * (1 - ramp(t, self.CLOSE, 0.3))
        if shown <= 0:
            return
        busy, result, _ = clips_a.translate_popups()
        k = self.POPUP_SCALE
        width = result.width() / result.devicePixelRatio() * k
        x, y = SW / 2 - width / 2, self.stage.editor.bottom() + 4 - 24 * k + 8 * (1 - ramp(t, self.OPEN, 0.3))
        done = ramp(t, self.DONE, 0.18)
        p.save()
        p.translate(x, y)
        p.scale(k, k)
        with faded(p, shown):
            if done < 1:
                with faded(p, 1 - done):
                    demo.draw_image(p, busy[int(max(0.0, t - self.OPEN) / 0.35) % 3], 0, 0)
            if done > 0:
                with faded(p, done):
                    demo.draw_image(p, result, 0, 0)
        p.restore()


# ---------------------------------------------------------------- 6. anything the laptop plays, translated live

LECTURE = [  # (start, end, heard, translation): a lecture in Spanish about climate data
    (0.15, 1.15, "Desde 1900, el planeta se ha calentado 1,2 grados.", "Since 1900, the planet has warmed by 1.2 degrees."),
    (1.25, 2.25, "Y los últimos diez años fueron los más cálidos registrados.",
     "And the last ten years were the warmest on record."),
]
MEETING_LINE = clips_b.LINES[0]  # Kenji: the release moves to Thursday (said in Japanese)


class Anything(Gif):
    """(a) a video in Spanish, a lecture on climate data, the real translation bar under it; (b) the meeting of the live
    clip, a schedule change said in Japanese. Each beat fades out and the next in, the last back into the first."""
    name = "6-translate-anything.gif"
    A, B, FADE = 3.4, 2.9, 0.36  # the beats' lengths, and the crossfades between them
    duration = A + B + 2 * FADE
    PLAYER = QRectF(30, 152, 480, 270 + 36)
    BAR_AT, BAR_SIZE = QPointF(20, 466), (500, 178)
    captions = [(0, "Anything your laptop plays,\ntranslated live.")]
    checks = (1.0, 2.9, 3.5, 3.7, 5.0, 6.3)

    def __init__(self):
        self.lecture = demo.Feed([e for line in LECTURE for e in demo.line_events(*line)], self.BAR_SIZE)
        *_, heard, said = MEETING_LINE
        self.meeting = demo.Feed(demo.line_events(0.15, 1.65, heard, said), self.BAR_SIZE)
        clips_b.CALL = QRectF(30, 152, 480, 306)  # the live clip's meeting, in this frame
        self.first: QImage | None = None

    def frame(self, t):
        if t < self.A:
            return super().frame(t)
        if self.A + self.FADE <= t < self.A + self.FADE + self.B:
            return super().frame(t)
        image = demo.backdrop().copy()
        p = demo.painter(image)
        if t < self.A + self.FADE:  # the video out, then the meeting in (no text over text)
            k = (t - self.A) / self.FADE
            layer = self.beat(t, "a") if k < 0.5 else self.beat(self.A + self.FADE, "b")
        else:  # the meeting out, then the video in, as at the start
            k = (t - self.A - self.FADE - self.B) / self.FADE
            layer = self.beat(t, "b") if k < 0.5 else self.beat(0.0, "a")
        p.resetTransform()
        p.setOpacity(1 - demo.ease(2 * k) if k < 0.5 else demo.ease(2 * k - 1))
        p.drawImage(0, 0, layer)
        p.setOpacity(1)
        p.scale(BASE.s, BASE.s)
        draw_mark(p)
        draw_caption(p, self.captions, t)
        p.end()
        return np.ascontiguousarray(demo.pixels(image)[..., 2::-1])

    def beat(self, t: float, which: str) -> QImage:
        """One beat alone (no caption), on a transparent layer."""
        layer = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
        layer.fill(Qt.GlobalColor.transparent)
        p = demo.painter(layer)
        if which == "a":
            self.video(p, t)
        else:
            self.call(p, t - self.A - self.FADE)
        p.end()
        return layer

    def draw(self, p, t):
        if t < self.A:
            self.video(p, t)
        else:
            self.call(p, t - self.A - self.FADE)

    def bar(self, p: QPainter, image: QImage) -> None:
        rect = QRectF(self.BAR_AT.x(), self.BAR_AT.y(), *self.BAR_SIZE)
        theme.outer_shadow(p, rect, 14, [(0, 16, 40, theme.rgba(0, 0, 0, .55)), (0, 2, 6, theme.rgba(0, 0, 0, .3))], BASE.s)
        demo.draw_image(p, image, rect.left(), rect.top())

    def video(self, p: QPainter, t: float) -> None:
        """A plain video player: a lecture slide (a temperature chart), the controls, the progress moving."""
        r = self.PLAYER
        window(p, r, 14, "#111317")
        screen = QRectF(r.left(), r.top(), r.width(), r.width() * 9 / 16)
        p.save()
        p.setClipPath(theme.rounded(r, 14))
        p.fillRect(screen, QColor("#1C2029"))  # flat: a gradient would band in the GIF's 256 colours
        p.restore()
        write(p, QRectF(screen.left() + 26, screen.top() + 18, 400, 26), "Global temperature, 1900–2020", 16, 600,
              T["text"], LEFT)
        write(p, QRectF(screen.left() + 26, screen.top() + 42, 400, 20), "Change from the 1951–1980 average, °C", 12, 400,
              T["text3"], LEFT)
        chart = QRectF(screen.left() + 52, screen.top() + 78, 290, 160)
        p.setPen(QPen(theme.rgba(255, 255, 255, .08), 1))
        for k in range(4):
            yy = chart.top() + chart.height() * k / 3
            p.drawLine(QPointF(chart.left(), yy), QPointF(chart.right(), yy))
        for k, label in enumerate(("+1.0", "+0.5", "0", "−0.5")):
            write(p, QRectF(chart.left() - 46, chart.top() + chart.height() * k / 3 - 9, 40, 18), label, 11, 400,
                  T["text3"], Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, mono=True)
        for k, label in enumerate(("1900", "1960", "2020")):
            write(p, QRectF(chart.left() + chart.width() * k / 2 - 30, chart.bottom() + 4, 60, 18), label, 11, 400,
                  T["text3"], mono=True)
        path, area = QPainterPath(), QPainterPath()
        n = 60
        for k in range(n + 1):
            x = k / n
            value = -0.25 + 0.05 * math.sin(k * 1.7) + 0.06 * math.sin(k * 0.6) + (1.25 * max(0.0, x - 0.55) ** 1.5 / 0.45 ** 1.5)
            point = QPointF(chart.left() + chart.width() * x, chart.top() + chart.height() * (1.0 - value) / 1.5)
            path.lineTo(point) if k else path.moveTo(point)
        area.addPath(path)
        area.lineTo(chart.bottomRight())
        area.lineTo(chart.bottomLeft())
        area.closeSubpath()
        p.fillPath(area, alpha(T["live"], 0.09))  # flat: a gradient would band in the GIF's 256 colours
        p.setPen(QPen(T["live"], 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        speaker = QRectF(screen.right() - 96, screen.bottom() - 90, 78, 72)  # the lecturer, small, in the corner
        p.fillPath(theme.rounded(speaker, 10), QColor("#2A2F3C"))
        p.save()
        p.setClipPath(theme.rounded(speaker, 10))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(blend(QColor("#2A2E3A"), T["warn"], 0.35))
        c = speaker.center()
        p.drawEllipse(QPointF(c.x(), c.y() - 8), 12, 12)
        p.drawEllipse(QRectF(c.x() - 26, c.y() + 10, 52, 48))
        p.restore()
        bar = QRectF(r.left(), screen.bottom(), r.width(), r.bottom() - screen.bottom())
        cy = bar.center().y()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T["text"])
        p.drawRect(QRectF(bar.left() + 20, cy - 7, 4, 14))  # pause: it's playing
        p.drawRect(QRectF(bar.left() + 28, cy - 7, 4, 14))
        track = QRectF(bar.left() + 48, cy - 2, 280, 4)
        done = 0.41 + 0.004 * t
        p.fillPath(theme.rounded(track, 2), theme.rgba(255, 255, 255, .14))
        p.fillPath(theme.rounded(QRectF(track.left(), track.top(), track.width() * done, 4), 2), T["iris"])
        p.setBrush(WHITE)
        p.drawEllipse(QPointF(track.left() + track.width() * done, cy), 6, 6)
        seconds = 12 * 60 + 48 + int(t)
        write(p, QRectF(track.right() + 12, bar.top(), 100, bar.height()), f"{seconds // 60}:{seconds % 60:02} / 31:05", 12,
              500, T["text2"], LEFT, mono=True)
        demo.icon(p, "speaker", T["text2"], bar.right() - 34, cy - 8)
        self.bar(p, self.lecture.at(t))

    def call(self, p: QPainter, u: float) -> None:
        """The live clip's meeting (its first line, a schedule change, said here a little faster), the bar under it."""
        clips_b.draw_meeting(p, min(0.3 + (u - 0.15) * 1.2, 2.2))  # (and quiet after it: no one else speaks)
        self.bar(p, self.meeting.at(u))


# ---------------------------------------------------------------- the GIF file

GIFS = [TalkItTypes, Concise, Bullets, Numbers, Translate, Anything]


def palette_for(frames: list[np.ndarray]) -> list[int]:
    """One palette for the whole GIF: 255 colours (index 255 is "unchanged"), most chosen from frames across it, the
    rest ramps from the background to white, Rflow's colours and the selection, so text edges and lamps keep theirs."""
    picks = frames[:: max(1, len(frames) // 10)][:10]
    sample = Image.fromarray(np.concatenate(picks, axis=0))
    ground = QColor("#15171C")
    fixed = [blend(ground, colour, k) for colour in (WHITE, T["text2"], T["iris"], T["violet"], T["ok"], T["live"])
             for k in (0.2, 0.4, 0.6, 0.8, 1.0)] + [T["selection"], T["text3"], QColor("#262B3A")]
    count = 255 - len(fixed)
    colours = sample.quantize(colors=count, method=Image.Quantize.MEDIANCUT).getpalette()[:3 * count]
    colours += [v for c in fixed for v in (c.red(), c.green(), c.blue())]
    return colours + [255, 0, 255] * (256 - len(colours) // 3)


BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16 - 0.5


def write_gif(path: Path, frames: list[np.ndarray], ground: np.ndarray, shadows: bool = False) -> tuple[float, int]:
    """The frames as a looping GIF: indexed without dithering (crisp text), except the backdrop's soft glows, which would
    band: where a frame shows the bare backdrop (`ground`), its pixels come from one dithered copy of it, the same in
    every frame. With `shadows` (a window that moves, its soft shadow over the glows), the dark parts get a faint ordered
    dither too: the same pattern in every frame, so nothing shimmers. Still frames are merged, and each frame stores only
    what moved."""
    colours = palette_for(frames)
    palette = Image.new("P", (1, 1))
    palette.putpalette(colours)
    dithered = np.asarray(Image.fromarray(ground).quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG))
    indexed = []
    for f in frames:
        bare = (f == ground).all(axis=2)
        if shadows:
            dark = (f.max(axis=2) < 72)[..., None]
            noise = np.tile(BAYER, (f.shape[0] // 4 + 1, f.shape[1] // 4 + 1))[:f.shape[0], :f.shape[1], None] * 5
            f = np.where(dark, np.clip(f + noise, 0, 255), f).astype(np.uint8)
        index = np.asarray(Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE)).copy()
        index[bare] = dithered[bare]
        indexed.append(index)
    ticks = [round(n * 100 / FPS) for n in range(len(frames) + 1)]  # centiseconds: GIF's own unit
    kept, durations = [indexed[0]], [ticks[1] - ticks[0]]
    for n in range(1, len(indexed)):
        if np.array_equal(indexed[n], kept[-1]):
            durations[-1] += ticks[n + 1] - ticks[n]
        else:
            kept.append(indexed[n])
            durations.append(ticks[n + 1] - ticks[n])
    out = []
    for n, now in enumerate(kept):
        image = Image.fromarray(now if n == 0 else np.where(now == kept[n - 1], 255, now).astype(np.uint8), "P")
        image.putpalette(colours)
        out.append(image)
    out[0].save(path, save_all=True, append_images=out[1:], duration=[d * 10 for d in durations], loop=0, transparency=255,
                disposal=1, optimize=False)
    expected = Image.fromarray(kept[-1], "P")
    expected.putpalette(colours)
    with Image.open(path) as gif:  # the file shows what was meant: its last frame, decoded, is the last frame drawn
        gif.seek(gif.n_frames - 1)
        if not np.array_equal(np.asarray(gif.convert("RGB")), np.asarray(expected.convert("RGB"))):
            sys.exit(f"{path.name}: the decoded GIF differs from its frames")
    return sum(durations) / 100, len(kept)


def export_frames(path: Path, folder: Path, times: tuple[float, ...]) -> None:
    """A few moments of the finished GIF, decoded from the file, as PNGs to look at."""
    with Image.open(path) as gif:
        at, n = 0.0, 0
        wanted = sorted(times)
        for n in range(gif.n_frames):
            gif.seek(n)
            length = gif.info.get("duration", 0) / 1000
            while wanted and at <= wanted[0] < at + length:
                gif.convert("RGB").save(folder / f"{path.stem}-{wanted.pop(0):.2f}s.png")
            at += length


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Rflow's LinkedIn GIFs.")
    parser.add_argument("only", nargs="*", help="which GIFs (1-6); all if none")
    parser.add_argument("--out", type=Path, default=Path.home() / "Videos" / "Rflow-gifs")
    parser.add_argument("--frames", type=Path, help="also save a few frames of each GIF here, as PNG")
    args = parser.parse_args()
    try:
        QApplication([])
        theme.load_fonts()
        args.out.mkdir(parents=True, exist_ok=True)
        if args.frames:
            args.frames.mkdir(parents=True, exist_ok=True)
        for n, kind in enumerate(GIFS, 1):
            if args.only and str(n) not in args.only:
                continue
            if kind in (Concise, Bullets):
                clips_b.check_results()
            demo.use(kind.stage)
            gif = kind()
            for _, text in gif.captions:
                check_caption(text)
            frames = [gif.frame(n / FPS) for n in range(round(gif.duration * FPS))]
            path = args.out / gif.name
            seconds, kept = write_gif(path, frames, np.ascontiguousarray(demo.pixels(demo.backdrop())[..., 2::-1]),
                                      gif.SHADOWS)
            print(f"Wrote {path.name}: {seconds:.1f} s, {kept} of {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")
            if args.frames:
                export_frames(path, args.frames, (0.0, *gif.checks, gif.duration - 0.5 / FPS))  # (the last: it meets the first)
    finally:
        shutil.rmtree(demo.DATA, ignore_errors=True)


if __name__ == "__main__":
    main()
