"""Render two short feature clips for the website, each only the feature, made to loop:

- site/media/clip-transform.mp4: one paragraph in a Notes window, transformed five ways by voice, a word or two each
  (Concise, Professional, Bullet points, Action items, Rewrite), each result put back with Undo before the next;
- site/media/clip-live.mp4: a meeting in Japanese translated as people speak, the translation read aloud while the
  meeting is turned down, and a line of the user's own (Both); the cost saver on (what the session cost and saved, in
  the bar's title), and the session copied at the end;

and a poster for each (site/img/clip-<name>-poster.png, 1280x720), a frame that shows the result.

They are drawn like the demo video (scripts/make_demo_video.py, whose helpers they use): the pill and the translation bar
are Rflow's own widgets with example data, drawn off the screen; nothing appears, takes focus, presses keys or touches
the clipboard. The five results are ones Rflow itself would type: each passes the real TransformGuard (checked first, the
run stops if one doesn't), and each command is a phrase sst.commands knows. Pillow (posters, and make_demo_video) is
added only for the run:

  uv run --no-sync --with pillow python scripts/make_feature_clips_b.py [transform | live]
"""
import math
import shutil
import subprocess
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
JOBS = ("transform", "live")
JOB = sys.argv[1] if len(sys.argv) > 1 else ""

import av  # noqa: E402
import make_demo_video as demo  # noqa: E402  (before Qt: it sets Qt's platform and scale, and a scratch APPDATA)
from av.video.reformatter import Colorspace  # noqa: E402
from make_demo_video import POP, T, alpha, blend, chip, clamp, draw_image, faded, mix, ramp, write  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPen, QTextLayout, QTextOption  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst import theme  # noqa: E402

MEDIA, IMG = ROOT / "site" / "media", ROOT / "site" / "img"
FPS = 30
LOOP = 0.8  # the live clip's last moments: back to its first frame

# ---------------------------------------------------------------- the transform clip's text

PARAGRAPH = ("Quick update on the release: we won't make Tuesday, because the database migration failed on staging last night, "
             "so it's probably moving to Thursday, October 15. Priya is going to confirm the new date with the client, and "
             "I'm going to fix the migration script. We still have 3 open bugs, and I think we should close them before we "
             "ship.")
ORDER = ("concise", "professional", "bullets", "actions", "rewrite")
COMMANDS = {"concise": "concise", "professional": "professional", "bullets": "bullet points", "actions": "action items",
            "rewrite": "rewrite"}  # one word is enough (2.8.1); "make it concise" works as well
RESULTS = {  # in the light markdown the model writes (sst.transform.render), as TransformPage's examples
    "concise": ("The release won't make Tuesday: the database migration failed on staging last night, so it's probably "
                "moving to Thursday, October 15. Priya will confirm the date with the client; I'll fix the migration script. "
                "3 bugs are still open, and I think we should close them before we ship."),
    "professional": ("A brief update on the release: we will not make Tuesday, as the database migration failed on staging "
                     "last night. The release will therefore probably move to Thursday, October 15. Priya will confirm the "
                     "new date with the client, and I will fix the migration script. There are still 3 open bugs, which I "
                     "believe we should resolve before we ship."),
    "bullets": ("**Release update**\n- Not making Tuesday: the database migration failed on staging last night\n"
                "- Probably moving to Thursday, October 15\n- Priya: confirm the new date with the client\n"
                "- Me: fix the migration script\n- 3 open bugs; I think we should close them before we ship"),
    "actions": ("**Action items**\n- Priya — Confirm the new date with the client\n- Me — Fix the migration script\n"
                "- Close the 3 open bugs before we ship"),
    "rewrite": ("Quick update on the release: we won't make Tuesday. The database migration failed on staging last night, so "
                "the release will probably move to Thursday, October 15. Priya is going to confirm the new date with the "
                "client, and I'm going to fix the migration script. We still have 3 open bugs, which I think we should close "
                "before we ship."),
}


def check_results() -> None:
    """Every result is one Rflow would type (the real guard accepts it), and every command one it hears as that command."""
    from sst.commands import match_command, phrases_for
    from sst.transform import TransformGuard
    guard, phrases = TransformGuard(), phrases_for({})
    for key in ORDER:
        got = guard.validate(PARAGRAPH, RESULTS[key], key)
        if not got.accepted:
            sys.exit(f"TransformGuard rejects the {key} result: {got.reasons}")
        if match_command(COMMANDS[key], phrases) != key:
            sys.exit(f"{COMMANDS[key]!r} isn't the {key} command")


# ---------------------------------------------------------------- the transform clip

# The demo's stage, the editor a little taller and the pill a little higher: below them, the command said, the keys and
# the strip of the five transforms.
STAGE = replace(demo.WIDE, editor=QRectF(170, 34, 940, 356), text_px=24, line=38, pill_y=448, keys_y=562)
COMMAND_Y, STRIP_Y = 503, 652
ROUND = 6.0  # one transform: selected, said, replaced, shown, undone
TOTAL = ROUND * len(ORDER)
BLINK = 1.06 / (TOTAL / round(TOTAL / 1.06))  # the caret's blink stretched a little, so it loops with the clip


@lru_cache(maxsize=64)
def block(text: str, weight: int, width: float) -> QTextLayout:
    """A paragraph laid out in the editor's font at `weight`, wrapped at `width`, one line every STAGE.line."""
    lay = QTextLayout(text, theme.font(STAGE.text_px, weight))
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    lay.setTextOption(option)
    lay.beginLayout()
    y = 0.0
    while (line := lay.createLine()).isValid():
        line.setLineWidth(width)
        line.setPosition(QPointF(0, y))
        y += STAGE.line
    lay.endLayout()
    return lay


def draw_result(p: QPainter, text: str, tint: float) -> None:
    """A result as Rflow pastes it: plain paragraphs, or a bold heading line and bullets. `tint` as in demo.draw_text."""
    if "\n" not in text:
        demo.draw_text(p, text, tint=tint)
        return
    at, width = demo.text_at(), STAGE.editor.width() - 2 * STAGE.pad
    p.setPen(blend(T["text"], T["violet"], tint))
    y = at.y()
    for line in text.splitlines():
        if line.startswith("**"):
            block(line.strip("*"), 600, width).draw(p, QPointF(at.x(), y))
            y += STAGE.line + 4
        elif line.startswith("- "):
            block("•", 400, 20).draw(p, QPointF(at.x() + 4, y))
            lay = block(line[2:], 400, width - 28)
            lay.draw(p, QPointF(at.x() + 28, y))
            y += STAGE.line * lay.lineCount()


def draw_modes(p: QPainter, current: int, k: float) -> None:
    """The five transforms in a row, as the menu numbers them; the current one lights up in Iris (from the one before, by
    k)."""
    from sst.transform import TRANSFORMS
    name_font, digit_font = theme.font(14, 500), theme.font(12, 600, mono=True)
    names, digits = QFontMetricsF(name_font), QFontMetricsF(digit_font)
    items = [(TRANSFORMS[key].key_hint, TRANSFORMS[key].name) for key in ORDER]
    widths = [names.horizontalAdvance(name) + digits.horizontalAdvance(digit) + 48 for digit, name in items]
    gap, h = 10, 34
    x = STAGE.sw / 2 - (sum(widths) + gap * (len(items) - 1)) / 2
    for i, ((digit, name), w) in enumerate(zip(items, widths, strict=True)):
        lit = k if i == current else (1 - k) if i == (current - 1) % len(items) else 0.0
        rect = QRectF(x, STRIP_Y - h / 2, w, h)
        p.fillPath(theme.rounded(rect, h / 2), blend(QColor("#191B21"), QColor("#252A3B"), lit))
        p.setPen(QPen(blend(theme.rgba(255, 255, 255, .08), alpha(T["iris"], 0.85), lit), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.6, 0.6, -0.6, -0.6), h / 2 - 0.6, h / 2 - 0.6)
        box = QRectF(rect.left() + 9, rect.center().y() - 10, digits.horizontalAdvance(digit) + 12, 20)
        p.fillPath(theme.rounded(box, 5), blend(theme.rgba(255, 255, 255, .06), alpha(T["iris"], 0.22), lit))
        write(p, box, digit, 12, 600, blend(T["text3"], T["iris"], lit), mono=True)
        write(p, QRectF(box.right() + 9, rect.top(), w, h), name, 14, 500, blend(T["text3"], T["text"], lit),
              Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        x += w + gap


class Round(demo.Scene):
    """One transform: the paragraph selected; Ctrl+Win held and the command said; the selection replaced by the result,
    the pill naming the transform; then U (Undo) in the menu puts the original back."""
    duration = ROUND
    chrome = staticmethod(demo.draw_editor)
    SELECT, KEYS, DOWN, UP, DONE = 0.15, 0.4, 0.6, 1.8, 2.4  # selected; keys shown, pressed, let go; the text replaced
    SPEECH = (0.8, 1.55)
    UNDO_KEYS, UNDO, HIDE = 4.35, 4.75, 5.25  # the U key shown; the original back; the pill gone

    def __init__(self, n: int):
        from sst.transform import TRANSFORMS
        self.n, self.key, self.start = n, ORDER[n], n * ROUND
        self.after = RESULTS[self.key]
        self.pill = demo.PillTrack(self.DOWN + 0.05, self.HIDE, [
            (self.DOWN + 0.05, "recording", False, ""), (self.UP, "transforming", False, ""),
            (self.DONE, "transformed", False, f"Transformed: {TRANSFORMS[self.key].name}"),
            (self.UNDO + 0.1, "transformed", False, "Original restored")], [self.SPEECH])

    def caret(self, p: QPainter, t: float) -> None:
        demo.draw_caret(p, PARAGRAPH, len(PARAGRAPH), (self.start + t) * 1.06 / BLINK)

    def draw(self, p, t):
        gone, came = ramp(t, self.DONE - 0.15, 0.2), ramp(t, self.DONE + 0.05, 0.3)  # one text out, then the other in
        out, back = ramp(t, self.UNDO - 0.1, 0.2), ramp(t, self.UNDO + 0.1, 0.3)
        if gone < 1:
            with faded(p, 1 - gone):
                demo.draw_text(p, PARAGRAPH, round(len(PARAGRAPH) * ramp(t, self.SELECT, 0.55)))
                if t < self.SELECT:
                    self.caret(p, t)
        if came > 0 and out < 1:
            with faded(p, came * (1 - out)):
                draw_result(p, self.after, 1 - ramp(t, self.DONE + 0.5, 1.0))
        if back > 0:
            with faded(p, back):
                demo.draw_text(p, PARAGRAPH)
                self.caret(p, t)
        pressed = ramp(t, self.DOWN, 0.12) * (1 - ramp(t, self.UP, 0.12))
        demo.draw_keys(p, [("Ctrl", pressed), "+", ("Win", pressed)],
                       ramp(t, self.KEYS, 0.3) * (1 - ramp(t, self.UP + 0.3, 0.35)))
        u = ramp(t, self.UNDO - 0.2, 0.08) * (1 - ramp(t, self.UNDO + 0.05, 0.1))
        demo.draw_keys(p, [("U", u), "Undo"], ramp(t, self.UNDO_KEYS, 0.25) * (1 - ramp(t, self.UNDO + 0.35, 0.3)))
        self.draw_command(p, t)
        self.pill.draw(p, t)
        draw_modes(p, self.n, ramp(t, 0.0, 0.45))

    def draw_command(self, p: QPainter, t: float) -> None:
        """What's said, word by word as it's heard, in quotes under the pill."""
        start, end = self.SPEECH
        said = demo.words_typed(COMMANDS[self.key], clamp((t - start + 0.15) / (end - start)))
        shown = ramp(t, start, 0.25) * (1 - ramp(t, self.DONE + 0.1, 0.35))
        if not said or shown <= 0:
            return
        whole = f"“{COMMANDS[self.key]}”"
        x = STAGE.sw / 2 - QFontMetricsF(theme.font(22, 500)).horizontalAdvance(whole) / 2
        with faded(p, shown):
            write(p, QRectF(x, COMMAND_Y - 20, 600, 40), f"“{said}" + ("”" if said == COMMANDS[self.key] else ""), 22,
                  500, T["text"], Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)


def transform_frame(rounds: list[Round], g: float) -> QImage:
    return demo.render(rounds[min(int(g // ROUND), len(rounds) - 1)], g % ROUND)


# ---------------------------------------------------------------- the live clip

PEOPLE = [("Kenji Sato", "KS", "iris"), ("Yui Tanaka", "YT", "violet"), ("Haruto Ito", "HI", "warn"),
          ("Aiko Mori", "AM", "violet"), ("Ren Kobayashi", "RK", "iris"), ("You", "", "iris")]  # Mint: who's speaking
YOU = len(PEOPLE) - 1
LINES = [  # (start, end, who, heard, translation): Japanese from the meeting, English from the user (Both)
    (0.3, 2.1, 0, "来週のリリースは木曜日に延期します。", "We're moving next week's release to Thursday."),
    (2.4, 4.4, 1, "追加の予算は50万円で、部長の承認が必要です。",
     "The extra budget is 500,000 yen, and it needs the director's approval."),
    (7.1, 8.9, YOU, "Thursday works for me. I'll send the approval request today.",
     "木曜日で大丈夫です。今日、承認依頼を送ります。"),
    (9.2, 10.8, 0, "ありがとうございます。では、木曜日で確定です。", "Thank you. Then Thursday it is."),
]
SPEAK_ON = 4.6  # the bar's speaker button turned on: the translation read aloud from then on
SPEND = [(2.2, 0.01, 0.01), (4.5, 0.02, 0.02), (9.0, 0.04, 0.03), (11.0, 0.05, 0.04)]  # the cost saver's note: (t, cost, saved)
COPY, COPIED = 12.0, 13.0  # the bar's copy button clicked: "Copied 4 lines" in its title for a moment
READS = [(4.85, 6.85), (10.95, 12.15)]  # the voice reading the second line's translation, then the last one's (not the user's)
LIVE_TOTAL = 13.1 + LOOP
CALL, BAR_AT, BAR_SIZE = QRectF(110, 14, 1060, 340), QPointF(220, 364), (840, 262)
CHIPS_Y = 640


class LiveFeed:
    """The real translation bar (sst.live.captions.CaptionBar) in Both, fed the meeting as it goes, like demo.Feed."""

    def __init__(self):
        from sst.live.contracts import MIC, Kind, LiveEvent
        events = []
        for start, end, who, heard, said in LINES:
            line = demo.line_events(start, end, heard, said)
            if who == YOU:  # into Japanese: no spaces to come word by word, so a few characters at a time
                line = [(at, e) for at, e in line if e.kind is not Kind.TRANSLATION] + [
                    (start + 0.35 + (end - start - 0.4) * k / 6, LiveEvent(Kind.TRANSLATION, said[:math.ceil(len(said) * k / 6)]))
                    for k in range(1, 7)]
                line = [(at, replace(e, lane=MIC)) for at, e in line]
            events += line
        events.append((SPEAK_ON, lambda bar: bar.set_speaking(True)))
        events += [(at, lambda bar, c=cost, s=saved: bar.set_spend(c, s)) for at, cost, saved in SPEND]
        events += [(COPY + 0.1, lambda bar: (bar.flash("Copied 4 lines"), bar._flash_timer.stop())),
                   (COPIED, lambda bar: bar.flash(""))]  # the timer stopped: the note goes when the clip says
        self.events = sorted(events, key=lambda e: e[0])
        self.bar, self.last, self.applied, self.image = None, 0.0, 0, None

    def at(self, t: float) -> QImage:
        from sst.live.captions import CaptionBar
        from sst.live.contracts import Kind, LiveConfig, LiveEvent
        if self.bar is None or t < self.last:
            self.bar = CaptionBar(LiveConfig(target="en", source="both", mic_target="ja", save=True))
            self.bar.resize(*BAR_SIZE)
            self.bar.show_event(LiveEvent(Kind.STATUS, "Listening"))
            self.applied, self.image = 0, None
        self.last = t
        while self.applied < len(self.events) and self.events[self.applied][0] <= t:
            event = self.events[self.applied][1]
            if callable(event):
                event(self.bar)
            else:
                self.bar.show_event(event)
            self.applied, self.image = self.applied + 1, None
        if self.image is None:
            demo.settle()
            self.image = self.bar.grab().toImage()
        return self.image


def draw_meeting(p: QPainter, t: float) -> None:
    """An online meeting, abstract: tiles with initials, the one speaking ringed in Mint."""
    r = CALL
    theme.outer_shadow(p, r, 14, [(0, 18, 44, theme.rgba(0, 0, 0, .5))], demo.STAGE.s)
    p.fillPath(theme.rounded(r, 14), QColor("#14161B"))
    p.setPen(QPen(theme.rgba(255, 255, 255, .07), 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)
    head = 40
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(T["ok"])
    p.drawEllipse(QPointF(r.left() + 22, r.top() + head / 2), 4, 4)
    write(p, QRectF(r.left() + 34, r.top(), 400, head), "Release planning", 13, 500, T["text2"], Qt.AlignmentFlag.AlignVCenter)
    write(p, QRectF(r.right() - 220, r.top(), 202, head), f"{len(PEOPLE)} people", 13, 400, T["text3"],
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
    speaking = next((who for start, end, who, *_ in LINES if start - 0.1 <= t <= end), -1)
    level = demo.voice(t, [(s, e) for s, e, *_ in LINES])
    cols, rows, gap, pad = 3, 2, 12, 16
    tw = (r.width() - 2 * pad - gap * (cols - 1)) / cols
    th = (r.height() - head - pad - gap * (rows - 1)) / rows
    for i, (name, initials, tint) in enumerate(PEOPLE):
        tile = QRectF(r.left() + pad + (i % cols) * (tw + gap), r.top() + head + (i // cols) * (th + gap), tw, th)
        colour = T[tint]
        p.fillPath(theme.rounded(tile, 12), blend(QColor("#1B1E26"), colour, 0.06))
        c = QPointF(tile.center().x(), tile.center().y() - 10)
        on = i == speaking
        if on:
            p.setPen(QPen(alpha(T["ok"], 0.35 + 0.4 * level), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, 33 + 6 * level, 33 + 6 * level)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(blend(QColor("#2A2E3A"), colour, 0.38))
        p.drawEllipse(c, 28, 28)
        if initials:
            write(p, QRectF(c.x() - 28, c.y() - 28, 56, 56), initials, 18, 600, T["text"])
        else:
            p.drawPixmap(QPointF(c.x() - 13, c.y() - 13), theme.icon_pixmap("profile", T["text"].name(), 26, 2 * demo.STAGE.s))
        write(p, QRectF(tile.left() + 14, tile.bottom() - 28, tw - 28, 22), name, 12, 500, T["text2"],
              Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        if on:
            p.setPen(QPen(alpha(T["ok"], 0.6 + 0.4 * level), 2.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(tile.adjusted(1.25, 1.25, -1.25, -1.25), 11, 11)


def volume(t: float) -> float:
    """The meeting app's volume: lowered to 30% while the voice reads (sst.live.ducking's default), back after."""
    return mix(1.0, 0.3, max(ramp(t, a, 0.15) * (1 - ramp(t, b + 0.2, 0.6)) for a, b in READS))


def draw_live(feed: LiveFeed, t: float, bar: QImage | None = None, chips: bool = True) -> QImage:
    """The meeting at t: the call, the bar over it (the feed's, or `bar`), and what Rflow does meanwhile (chips)."""
    image = demo.backdrop().copy()
    p = demo.painter(image)
    draw_meeting(p, t)
    bar = feed.at(t) if bar is None else bar
    rect = QRectF(BAR_AT.x(), BAR_AT.y(), *BAR_SIZE)
    theme.outer_shadow(p, rect, 14, [(0, 16, 40, theme.rgba(0, 0, 0, .55)), (0, 2, 6, theme.rgba(0, 0, 0, .3))],
                       demo.STAGE.s)
    draw_image(p, bar, rect.left(), rect.top())
    for at, name in ((SPEAK_ON, "speak_button"), (COPY, "copy_button")):  # the speaker button pressed, then copy
        click = ramp(t, at - 0.15, 0.15) * (1 - ramp(t, at + 0.35, 0.5))
        if click <= 0:
            continue
        button = getattr(feed.bar, name).geometry()
        centre = QPointF(rect.left() + button.center().x() + 0.5, rect.top() + button.center().y() + 0.5)
        p.setPen(QPen(alpha(POP["iris"], 0.8 * click), 2))
        p.setBrush(alpha(POP["iris"], 0.16 * click))
        p.drawEllipse(centre, 14 + 4 * click, 14 + 4 * click)
    reading = max(ramp(t, a - 0.1, 0.25) * (1 - ramp(t, b, 0.25)) for a, b in READS) if chips else 0.0
    meter = ramp(t, SPEAK_ON, 0.3) if chips else 0.0
    left = 640 - (214 + 12 + 276) / 2
    if reading > 0:
        with faded(p, reading):
            r = QRectF(left, CHIPS_Y, 214, 36)
            chip(p, r, "speaker", POP["iris"], "Reading it aloud")
            for k in range(4):  # the voice, as it speaks
                h = 4 + 12 * abs(math.sin(t * 7.3 + k * 1.1)) * (0.6 + 0.4 * math.sin(t * 2.1 + k))
                p.fillRect(QRectF(r.right() - 46 + k * 7, r.center().y() - h / 2, 3.5, h), POP["iris"])
    if meter > 0:
        with faded(p, meter):
            level = volume(t)
            r = QRectF(left + 226, CHIPS_Y, 276, 36)
            chip(p, r, "speaker", POP["text2"], "Meeting")
            track = QRectF(r.left() + 108, r.center().y() - 3, 100, 6)
            p.fillPath(theme.rounded(track, 3), POP["well"])
            p.fillPath(theme.rounded(QRectF(track.left(), track.top(), max(6.0, track.width() * level), 6), 3), POP["iris"])
            write(p, QRectF(track.right() + 8, r.top(), 52, r.height()), f"{round(level * 100)}%", 13, 500, POP["text"],
                  Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, mono=True)
    p.end()
    return image


@lru_cache(maxsize=1)
def empty_bar() -> QImage:
    """The bar with nothing in it yet: between the meeting's last frame and its first, so no text fades over text."""
    from sst.live.captions import CaptionBar
    from sst.live.contracts import Kind, LiveConfig, LiveEvent
    bar = CaptionBar(LiveConfig(target="en", source="both", mic_target="ja", save=True))
    bar.resize(*BAR_SIZE)
    bar.show_event(LiveEvent(Kind.STATUS, ""))
    demo.settle()
    return bar.grab().toImage()


class LiveClip:
    """The clip's frames: the meeting, then in the last LOOP seconds back to its first frame through an empty bar."""

    def __init__(self):
        self.feed = LiveFeed()
        self.first = draw_live(self.feed, 0.0)
        self.end = self.blank = None

    def frame(self, g: float) -> QImage:
        settled = LIVE_TOTAL - LOOP
        if g <= settled:
            return draw_live(self.feed, g)
        if self.end is None:
            self.end = draw_live(self.feed, settled)
            self.blank = draw_live(self.feed, settled, empty_bar(), chips=False)
        k = (g - settled) / LOOP
        a, b, amount = (self.end, self.blank, demo.ease(2 * k)) if k < 0.5 else (self.blank, self.first, demo.ease(2 * k - 1))
        image = a.copy()
        p = QPainter(image)
        p.setOpacity(amount)
        p.drawImage(0, 0, b)
        p.end()
        return image


# ---------------------------------------------------------------- files

def encode(name: str, total: float, frame) -> None:
    """H.264 (yuv420p, BT.709, no sound), the index first so it plays while it loads."""
    path = MEDIA / name
    with av.open(str(path), "w", container_options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=FPS, options={
            "crf": "21", "preset": "slow", "tune": "animation",
            "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709"})
        stream.width, stream.height, stream.pix_fmt = demo.STAGE.w, demo.STAGE.h, "yuv420p"
        for n in range(round(total * FPS)):
            video = av.VideoFrame.from_ndarray(demo.pixels(frame(n / FPS)), format="bgra")
            container.mux(stream.encode(video.reformat(format="yuv420p", dst_colorspace=Colorspace.ITU709)))
        container.mux(stream.encode())
    print(f"Wrote {name}: {total:.1f} s, {demo.STAGE.w}x{demo.STAGE.h}, {FPS} fps, {path.stat().st_size / 1e6:.2f} MB")


def poster(name: str, image: QImage) -> None:
    path = IMG / f"clip-{name}-poster.png"
    Image.fromarray(demo.pixels(image)[..., 2::-1].copy()).resize((1280, 720), Image.Resampling.LANCZOS).save(
        path, optimize=True)
    print(f"Wrote {path.name}: {path.stat().st_size / 1e3:.0f} kB")


def main() -> None:
    try:
        if not JOB:  # each clip in a process of its own, side by side
            jobs = [subprocess.Popen([sys.executable, __file__, job]) for job in JOBS]
            sys.exit(max(job.wait() for job in jobs))
        if JOB not in JOBS:
            sys.exit(f"usage: make_feature_clips_b.py [{' | '.join(JOBS)}]")
        QApplication([])
        theme.load_fonts()
        MEDIA.mkdir(parents=True, exist_ok=True)
        if JOB == "transform":
            check_results()
            demo.use(STAGE)
            rounds = [Round(n) for n in range(len(ORDER))]
            encode("clip-transform.mp4", TOTAL, lambda g: transform_frame(rounds, g))
            poster("transform", demo.render(rounds[ORDER.index("bullets")], 3.9))
        else:
            demo.use(demo.WIDE)
            clip = LiveClip()
            encode("clip-live.mp4", LIVE_TOTAL, clip.frame)
            poster("live", draw_live(clip.feed, 11.6))
    finally:
        shutil.rmtree(demo.DATA, ignore_errors=True)


if __name__ == "__main__":
    main()
