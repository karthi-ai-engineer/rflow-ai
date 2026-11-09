"""Render two short clips of Rflow's newer features for the website, each only the feature, made to loop:

- site/media/clip-meeting.mp4: Meeting mode. The real Meeting mode page finds the Teams meeting and Start is clicked;
  the meeting runs in Japanese with the real translation bar over it while it's recorded; back on the page, Stop, then
  "Writing the minutes…", "Minutes ready", and the minutes themselves (minutes.md as sst.meeting.minutes.render writes
  it from a MinutesResult: the summary, the decisions, the action items with owners and dates);
- site/media/clip-wake.mp4: the wake word. No key is pressed: the keyword said, the pill listening, a sentence, the
  end phrase ("stop it", the default), and the sentence typed into a Notes window without the keyword or the phrase.

And a poster for each (site/img/clip-<name>-poster.png, 1280x720), a frame that shows the result. They are drawn like
the demo video (scripts/make_demo_video.py, whose helpers they use): the window, the pill and the translation bar are
Rflow's own widgets with example data (a PreviewApp: nothing is recorded or sent), drawn off the screen; nothing
appears, takes focus, presses keys or touches the clipboard. The tour imports CLIPS (name: (seconds, frame(t))) to play
them as scenes; importing renders nothing. Pillow (posters, and make_demo_video) is added only for the run:

  uv run --no-sync --with pillow python scripts/make_feature_clips_d.py [meeting | wake]
"""
import os
import shutil
import subprocess
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
JOBS = ("meeting", "wake")
JOB = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else ""

import make_demo_video as demo  # noqa: E402  (before Qt: it sets Qt's platform and scale, and a scratch APPDATA)
from make_demo_video import POP, T, alpha, blend, clamp, draw_image, faded, ramp, write  # noqa: E402
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPen  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst import theme  # noqa: E402

LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
LOOP = 0.8  # each clip's last moments: back to its first frame


def crossfade(a: QImage, b: QImage, amount: float) -> QImage:
    image = a.copy()
    p = QPainter(image)
    p.setOpacity(clamp(amount))
    p.drawImage(0, 0, b)
    p.end()
    return image


def click(p: QPainter, centre: QPointF, t: float, at: float) -> None:
    """A click on a button: an Iris ring that grows and fades, as the live clip's speaker button."""
    k = ramp(t, at - 0.15, 0.15) * (1 - ramp(t, at + 0.35, 0.5))
    if k > 0:
        p.setPen(QPen(alpha(POP["iris"], 0.8 * k), 2))
        p.setBrush(alpha(POP["iris"], 0.16 * k))
        p.drawEllipse(centre, 22 + 6 * k, 22 + 6 * k)


# ---------------------------------------------------------------- 1. Meeting mode

MEETING_STAGE = demo.WIDE  # the tour's stage: Rflow's window at 150%, as on a laptop
LINES = [  # (start, end, who, heard, translation), in the call's seconds: a weekly sync in Japanese
    (0.3, 1.9, 0, "では、今週の進捗を確認しましょう。", "Let's go over this week's progress."),
    (2.2, 4.0, 1, "新しい画面のデザインは金曜日までに仕上げます。", "I'll finish the new screen designs by Friday."),
    (4.3, 5.9, 2, "リリースは来週の木曜日に決定です。", "The release is set for next Thursday."),
]
TINTS = ("iris", "violet", "ok", "warn")
CALL, BAR_AT, BAR_SIZE = QRectF(110, 52, 1060, 330), QPointF(220, 394), (840, 236)
VIEW = QRectF(140, 34, 1000, 600)  # where the page is shown, and the part of the window it shows (window pixels)
PAGE = QRectF(236, 76, 730, 438)
# The clip's beats, in seconds: the page; Start; the call; back to the page; Stop; writing; ready; the minutes
START, CALL_IN, CALL_OUT, STOP, WRITTEN, MINUTES, MEETING_TOTAL = 1.3, 2.3, 8.9, 10.0, 11.4, 12.2, 16.4 + LOOP
CALL_LEN = CALL_OUT - CALL_IN


def example_minutes():
    """The minutes Gemini would write of that meeting, as Rflow keeps them (MinutesResult), and their minutes.md."""
    from sst.meeting.minutes import ActionItem, MinutesResult, render
    result = MinutesResult(
        "Weekly sync: designs and the release date",
        "The team went over the week's progress. The new screen designs are on track for Friday, and the release was set "
        "for next Thursday once testing is done.",
        ["Release next Thursday, after testing", "New screens reviewed on Monday before the build"],
        [ActionItem("Finish the new screen designs", "Yui Tanaka", "Friday"),
         ActionItem("Book the release review", "Kenji Sato", "Wednesday"),
         ActionItem("Share the test plan with the team", "You", "")],
        speakers=["Kenji Sato", "Yui Tanaka", "Haruto Ito", "You"], language="ja")
    meta = {"started": datetime.now().replace(hour=10, minute=0).isoformat(), "app": "Microsoft Teams", "seconds": 1690}
    return result, render(result, meta)


class MeetingWindow:
    """The real window on Meeting mode (a PreviewApp: Rflow's own flow, nothing recorded or sent), grabbed as the
    meeting goes: found, recording, recording later on, writing the minutes, minutes ready."""

    def __init__(self):
        from sst import window as w
        from sst.meeting.apps import MeetingWindow as Found
        demo.site.parakeet.find_model = lambda: Path(".")  # as if Parakeet were downloaded: Rflow shows as ready
        self.app = demo.site.preview()
        self.app._meeting_windows = (Found("Microsoft Teams", "Weekly sync | Microsoft Teams", 101, 7),)
        self.window = w.MainWindow(self.app)
        self.window.apply_theme("dark")
        self.window.resize(1000, 680)
        self.window.set_status("Ready: hold Ctrl+Win · cleanup: gemini-3.5-flash-lite", True)
        self.window.show_page("meeting")
        self.page = self.window.pages["meeting"]
        self.states = self._states()

    def grab(self) -> QImage:
        self.window.refresh()
        demo.settle()
        return self.window.grab().toImage()

    def button(self) -> QPointF:
        """The Start / Stop button's centre, in the window's pixels."""
        b = self.page.start_button
        return QPointF(b.mapTo(self.window, b.rect().center()))

    def _states(self) -> dict[str, QImage]:
        from sst.meeting.session import MeetingCost
        images = {"found": self.grab()}
        self.page.start_button.click()  # PreviewApp.start_meeting: "recording", nothing heard or saved
        self.app._meeting_cost = MeetingCost(total=0.0, translation=0.0, minutes_estimate=0.0, seconds=8.0, frames=0)
        images["recording"] = self.grab()
        self.app._meeting_cost = MeetingCost(total=0.39, translation=0.27, minutes_estimate=0.12, seconds=1690.0, frames=6)
        images["later"] = self.grab()
        self.page.start_button.click()  # PreviewApp.stop_meeting: "writing minutes"
        images["writing"] = self.grab()
        state, window, folder, _, _ = self.app._meeting
        self.app._meeting = ("done", window, folder, example_minutes()[0], "")
        images["ready"] = self.grab()
        return images


def draw_page(p: QPainter, image: QImage, amount: float, lift: float = 0.0) -> None:
    """The part of the window that matters (the page, not the sidebar), in a rounded view a little larger than life."""
    if amount <= 0:
        return
    scale = VIEW.width() / PAGE.width()
    with faded(p, amount):
        view = VIEW.translated(0, lift)
        theme.outer_shadow(p, view, 16, [(0, 18, 44, theme.rgba(0, 0, 0, .55))], demo.STAGE.s)
        p.save()
        p.setClipPath(theme.rounded(view, 16))
        p.fillRect(view, QColor("#16181D"))
        p.translate(view.left(), view.top())
        p.scale(scale, scale)
        p.translate(-PAGE.left(), -PAGE.top())
        dpr = image.devicePixelRatio()
        p.drawImage(QRectF(0, 0, image.width() / dpr, image.height() / dpr), image)
        p.restore()
        p.setPen(QPen(theme.rgba(255, 255, 255, .08), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(view.adjusted(0.5, 0.5, -0.5, -0.5), 16, 16)


def on_page(point: QPointF) -> QPointF:
    """A point in the window's pixels, on the stage where draw_page shows it."""
    scale = VIEW.width() / PAGE.width()
    return QPointF(VIEW.left() + (point.x() - PAGE.left()) * scale, VIEW.top() + (point.y() - PAGE.top()) * scale)


def draw_people(p: QPainter, t: float) -> None:
    """The call, abstract (make_demo_video's draw_call): who's speaking ringed in Mint."""
    spans = [(s, e) for s, e, *_ in LINES]
    who = next((w for s, e, w, *_ in LINES if s - 0.1 <= t <= e), -1)
    demo.draw_call(p, CALL, "Weekly sync", TINTS, 1, who, demo.voice(t, spans), you=3)


def draw_recording(p: QPainter, t: float) -> None:
    """Rflow's part while the meeting runs: a capsule over the call, its red light blinking, the time going on."""
    r = QRectF(CALL.center().x() - 110, CALL.top() + 2, 220, 30)  # in the call's title bar, between its name and count
    theme.paint_surface(p, "float", r, r.height() / 2, popup=True, dpr=demo.STAGE.s)
    on = 0.55 + 0.45 * abs(((t % 1.2) / 1.2) * 2 - 1)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(alpha(T["live"], on))
    p.drawEllipse(QPointF(r.left() + 17, r.center().y()), 5, 5)
    seconds = 28 * 60 + 2 + t
    write(p, QRectF(r.left() + 30, r.top(), 104, r.height()), "Recording", 13, 600, POP["text"], LEFT)
    write(p, QRectF(r.left() + 108, r.top(), 100, r.height()), f"{int(seconds // 60)}:{int(seconds % 60):02d}", 13, 500,
          POP["text2"], Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, mono=True)


@lru_cache(maxsize=1)
def meeting_feed():
    events = [e for s, e_, w, heard, said in LINES for e in demo.line_events(s, e_, heard, said)]
    return demo.Feed(events, BAR_SIZE)


def draw_call(p: QPainter, t: float) -> None:
    draw_people(p, t)
    draw_recording(p, t)
    rect = QRectF(BAR_AT.x(), BAR_AT.y(), *BAR_SIZE)
    theme.outer_shadow(p, rect, 14, [(0, 16, 40, theme.rgba(0, 0, 0, .55)), (0, 2, 6, theme.rgba(0, 0, 0, .3))],
                       demo.STAGE.s)
    draw_image(p, meeting_feed().at(t), rect.left(), rect.top())


@lru_cache(maxsize=64)
def minutes_lines(width: float) -> list[tuple[str, str]]:
    """minutes.md up to its transcript, as (kind, text) lines wrapped to `width`: a title, headings, plain lines, bullets
    and checklist items, the way a Markdown viewer shows them."""
    _, md = example_minutes()
    md = md.split("\n## Transcript")[0]
    out = []
    for raw in md.splitlines():
        if not raw.strip():
            out.append(("gap", ""))
        elif raw.startswith("# "):
            out.append(("title", raw[2:]))
        elif raw.startswith("## "):
            out.append(("heading", raw[3:]))
        elif raw.startswith("- [ ] "):
            out += wrapped("check", raw[6:], width - 34, 17)
        elif raw.startswith("- "):
            out += wrapped("bullet", raw[2:], width - 26, 17)
        else:
            out += wrapped("text", raw, width, 17)
    return out


def wrapped(kind: str, text: str, width: float, px: float) -> list[tuple[str, str]]:
    metrics, lines, line = QFontMetricsF(theme.font(px)), [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and metrics.horizontalAdvance(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    return [(kind if i == 0 else f"{kind}+", ln) for i, ln in enumerate(lines)]


def draw_minutes(p: QPainter, t: float, amount: float) -> None:
    """minutes.md opened from "Open the minutes": a document window, its lines coming in from the top."""
    if amount <= 0:
        return
    r = QRectF(230, 24 + 16 * (1 - amount), 820, 640)
    with faded(p, amount):
        theme.outer_shadow(p, r, 12, [(0, 18, 44, theme.rgba(0, 0, 0, .6)), (0, 2, 6, theme.rgba(0, 0, 0, .3))],
                           demo.STAGE.s)
        p.fillPath(theme.rounded(r, 12), QColor("#1C1F25"))
        p.save()
        p.setClipPath(theme.rounded(r, 12))
        p.fillRect(QRectF(r.left(), r.top(), r.width(), demo.BAR), QColor("#16181D"))
        p.fillRect(QRectF(r.left(), r.top() + demo.BAR - 1, r.width(), 1), theme.rgba(255, 255, 255, .06))
        p.restore()
        p.setPen(QPen(theme.rgba(255, 255, 255, .08), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        demo.icon(p, "reading", T["text3"], r.left() + 16, r.top() + 12)
        name = f"minutes.md · {datetime.now():%Y-%m-%d} 1000 Microsoft Teams"  # in the meeting's folder, named by it
        write(p, QRectF(r.left() + 42, r.top(), 500, demo.BAR), name, 13, 500, T["text2"], LEFT)
        x, y, width = r.left() + 40, r.top() + demo.BAR + 26, r.width() - 80
        lines = minutes_lines(width)
        for n, (kind, text) in enumerate(lines):
            k = ramp(t, 0.15 + 0.035 * n, 0.3)
            if k <= 0:
                break
            with faded(p, k):
                y = minutes_line(p, kind, text, x, y, width)
            if y > r.bottom() - 20:
                break


def minutes_line(p: QPainter, kind: str, text: str, x: float, y: float, width: float) -> float:
    """One line of the minutes at x, y; where the next one goes."""
    base = kind.rstrip("+")
    if base == "gap":
        return y + 8
    if base == "title":
        write(p, QRectF(x, y, width, 36), text, 24, 600, T["text"], LEFT)
        return y + 38
    if base == "heading":
        write(p, QRectF(x, y, width, 28), text, 18, 600, T["iris"], LEFT)
        return y + 30
    indent = {"bullet": 26, "check": 34}.get(base, 0)
    if not kind.endswith("+"):
        if base == "bullet":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(T["text2"])
            p.drawEllipse(QPointF(x + 8, y + 13), 3, 3)
        elif base == "check":
            p.setPen(QPen(T["text3"], 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(x + 2, y + 5, 16, 16), 4, 4)
    colour = T["text2"] if base == "text" and text.count(" · ") >= 2 or text.startswith("People:") else T["text"]
    write(p, QRectF(x + indent, y, width - indent, 26), text, 17, 400, colour, LEFT)
    return y + 26


def meeting_at(t: float) -> QImage:
    """The Meeting mode clip at t: the page and Start; the call while it's recorded; the page, Stop, the minutes written;
    the minutes; then back to the first frame."""
    mw = meeting_window()
    states = mw.states
    image = demo.backdrop().copy()
    p = demo.painter(image)
    call = ramp(t, CALL_IN - 0.3, 0.5) * (1 - ramp(t, CALL_OUT - 0.2, 0.5))
    page = 1 - call
    if page > 0:
        if t < CALL_IN:
            shown = crossfade(states["found"], states["recording"], ramp(t, START + 0.05, 0.2))
        elif t < STOP + 0.05:
            shown = states["later"]
        else:
            shown = crossfade(states["writing"], states["ready"], ramp(t, WRITTEN, 0.25))
            if t < STOP + 0.3:
                shown = crossfade(states["later"], shown, ramp(t, STOP + 0.05, 0.2))
        draw_page(p, shown, page * (1 - ramp(t, MINUTES - 0.1, 0.4) * 0.6))
        centre = on_page(mw.button())
        click(p, centre, t, START)
        click(p, centre, t, STOP)
    if call > 0:
        with faded(p, call):
            draw_call(p, max(0.0, t - CALL_IN))
    draw_minutes(p, t - MINUTES, ramp(t, MINUTES, 0.45))
    p.end()
    return image


def meeting_frame(t: float) -> QImage:
    if demo.STAGE is not MEETING_STAGE:
        demo.use(MEETING_STAGE)
    settled = MEETING_TOTAL - LOOP
    if t <= settled:
        return meeting_at(t)
    return crossfade(meeting_at(settled), meeting_at(0.0), demo.ease((t - settled) / LOOP))


@lru_cache(maxsize=1)
def meeting_window() -> MeetingWindow:
    return MeetingWindow()


# ---------------------------------------------------------------- 2. the wake word

# The dictation clip's stage (scripts/make_feature_clips_a.py): a 960x540 stage at 200%, the Notes window at the top,
# the pill below it, and what is said under the pill.
WAKE_STAGE = demo.Stage(1920, 1080, 2.0, QRectF(110, 40, 740, 214), 36, 28, 44, 336, 1.25, 420, 1.15, 450, 19, 400)
KEYWORD, END_PHRASE = "Hey Rflow", "stop it"  # the keyword the user typed in its setup; the end phrase, the default
SENTENCE = demo.site.EXAMPLES[1][1]  # "Thanks for the quick reply! I'll send you the updated draft this afternoon."
SAID = [(0.5, 1.25, KEYWORD), (1.9, 4.9, SENTENCE), (5.25, 5.95, END_PHRASE)]  # (from, to, words): what's said
HEARD, UP, TYPED, HIDE, CLEAR = 1.3, 6.5, 7.15, 8.3, 9.1  # the pill: listening, cleaning up, typed, gone; the text gone
WAKE_TOTAL = 9.8 + LOOP


@lru_cache(maxsize=1)
def wake_pill() -> demo.PillTrack:
    return demo.PillTrack(HEARD, HIDE, [(HEARD, "recording", False, ""), (UP, "transcribing", True, ""),
                                        (TYPED, "typed", False, "")], [(s, e) for s, e, _ in SAID[1:]])


def blink(t: float) -> float:
    """The caret's blink (about 1 s), timed so the clip's end meets its start."""
    period = WAKE_TOTAL / max(1, round(WAKE_TOTAL / 1.06))
    return clamp(abs(((t % period) / period) * 2 - 1) * 6 - 2.5)


def draw_said(p: QPainter, t: float) -> None:
    """What's being said, one phrase at a time, under the pill: the keyword and the end phrase in Iris, in capsules
    (what Rflow listens for), the sentence word by word between them."""
    s = demo.STAGE
    for start, end, words in SAID:
        shown = ramp(t, start - 0.1, 0.2) * (1 - ramp(t, end + 0.35, 0.25))
        if shown <= 0:
            continue
        cue = words in (KEYWORD, END_PHRASE)
        heard = words if cue else demo.words_typed(words, clamp((t - start) / (end - start - 0.15)))
        px = 22 if cue else 18
        metrics = QFontMetricsF(theme.font(px, 600 if cue else 400))
        full = metrics.horizontalAdvance(f"“{words}”")
        width = full + (40 if cue else 0)
        x = s.sw / 2 - (width + 30) / 2
        speaking = demo.voice(t, [(start, end)])
        with faded(p, shown):
            demo.icon(p, "mic", blend(T["text3"], T["live"], clamp(speaking * 1.6) if t < end else 0), x, s.caption_y - 8)
            if cue:
                r = QRectF(x + 30, s.caption_y - 21, width, 42)
                p.fillPath(theme.rounded(r, 21), alpha(T["iris"], 0.16))
                p.setPen(QPen(alpha(T["iris"], 0.6), 1.2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(r.adjusted(0.6, 0.6, -0.6, -0.6), 20.4, 20.4)
                write(p, r, f"“{words}”", px, 600, T["iris"])
            elif heard:
                write(p, QRectF(x + 30, s.caption_y - 16, full + 4, 32), f"“{heard}" + ("”" if heard == words else ""),
                      px, 400, T["text2"], LEFT)
    for start, label in ((SAID[0][0], "Your keyword starts it"), (SAID[2][0], "Your end phrase types it")):
        shown = ramp(t, start + 0.1, 0.3) * (1 - ramp(t, start + 1.3, 0.3))
        if shown > 0:
            with faded(p, shown):
                write(p, QRectF(0, s.caption_y + 32, s.sw, 26), label, 14, 500, T["text3"])


def wake_frame(t: float) -> QImage:
    """The wake word clip at t: no keys, the keyword said, the pill, the sentence, the end phrase, the text typed."""
    if demo.STAGE is not WAKE_STAGE:
        demo.use(WAKE_STAGE)
    image = demo.backdrop().copy()
    p = demo.painter(image)
    demo.draw_editor(p)
    s = demo.STAGE
    typing = ramp(t, TYPED - 0.05, 0.7) if t >= TYPED - 0.05 else 0.0
    text, gone = demo.words_typed(SENTENCE, typing), ramp(t, CLEAR, 0.55)
    if gone < 1:
        with faded(p, 1 - gone):
            demo.draw_text(p, text)
    height = s.text_px + 8
    if gone == 0:
        on = 1.0 if 0 < typing < 1 else blink(t)
        if on > 0:
            at = demo.caret_at(text, len(text))
            p.fillRect(QRectF(at.x(), at.y() - 1, 2, height), alpha(T["iris"], on))
    elif gone >= 1 and blink(t) > 0:
        at = demo.text_at()
        p.fillRect(QRectF(at.x(), at.y() - 1, 2, height), alpha(T["iris"], blink(t)))
    wake_pill().draw(p, t)
    draw_said(p, t)
    p.end()
    return image


# ---------------------------------------------------------------- files

CLIPS = {"meeting": (MEETING_TOTAL, meeting_frame), "wake": (WAKE_TOTAL, wake_frame)}
POSTERS = {"meeting": MINUTES + 1.6, "wake": TYPED + 0.9}


def main() -> None:
    try:
        if not JOB:  # each clip in a process of its own, side by side
            jobs = [subprocess.Popen([sys.executable, __file__, job]) for job in JOBS]
            sys.exit(max(job.wait() for job in jobs))
        if JOB not in JOBS:
            sys.exit(f"usage: make_feature_clips_d.py [{' | '.join(JOBS)}]")
        os.environ["QT_SCALE_FACTOR"] = "3"  # Rflow's parts grabbed at 3x: sharp when shown larger than life
        import make_feature_clips_b as clips  # its encoder and posters (H.264, yuv420p, BT.709, no sound)
        QApplication([])
        theme.load_fonts()
        demo.site.parakeet.find_model = lambda: Path(".")  # as if Parakeet were downloaded: Rflow shows as ready
        clips.MEDIA.mkdir(parents=True, exist_ok=True)
        total, frame = CLIPS[JOB]
        frame(0.0)  # the clip's stage, before the encoder reads its size
        clips.encode(f"clip-{JOB}.mp4", total, frame)
        clips.poster(JOB, frame(POSTERS[JOB]))
    finally:
        shutil.rmtree(demo.DATA, ignore_errors=True)


if __name__ == "__main__":
    main()
