"""Render the website's screenshots (site/img) from the real Rflow window, popups and pill, with generic example data.

Everything is drawn off the screen (nothing appears, nothing takes focus), at the screen's scaling, as if Parakeet were
downloaded. The popups and the pill are drawn without being opened, on a transparent background:

  uv run python scripts/make_site_screenshots.py
"""
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")  # the real fonts and scaling
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from sst.engines import parakeet

OUT = Path(__file__).resolve().parent.parent / "site" / "img"
EXAMPLES = [  # (minutes ago, text)
    (3, "Can you review the pull request before lunch and leave a comment if anything looks wrong?"),
    (30, "Thanks for the quick reply! I'll send you the updated draft this afternoon."),
    (54, "Let's move the stand-up to Thursday morning and share the notes in Slack."),
    (75, "Please book a room for six people on Friday at 2 o'clock."),
    (93, "Ship the release notes after the review is merged, then tag the build."),
    (107, "Remind Priya that the Q3 roadmap review moved to Monday."),
    (125, "Draft a short thank-you note to the design team for the new screens."),
    (60 * 24 + 5, "Let's move the stand-up to Thursday morning and share the notes in Slack."),
]


def preview():
    from sst import window as w
    from sst.gateway import GatewayConfig
    from sst.settings import Profiles, Settings, Stats
    now = datetime.now()
    history = [{"time": (now - timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S"), "text": t} for m, t in EXAMPLES]
    stats = Stats()
    for n, words in enumerate([412, 386, 520, 264, 598, 331, 190, 0, 455, 380, 610]):  # the last three weeks
        if words:
            stats.add(" ".join(["word"] * words), words / 2.3, date.today() - timedelta(days=n))
    stats.add(" ".join(["word"] * 9800), 9800 / 2.3, date.today() - timedelta(days=30))  # the month before
    settings = Settings(welcomed=True, cleanup=True, cleanup_model="gemini-3.5-flash-lite",
                        microphone="Microphone array",
                        vocabulary=["Rflow", "GitHub", "Kubernetes", "Priya", "Q3 roadmap", "Parakeet", "CodeQL",
                                    "Vercel", "Snapdragon"],
                        snippets=[{"cue": "my email", "text": "alex@example.com", "anywhere": False}])
    profiles = Profiles()
    profiles.current.name = "Alex"
    app = w.PreviewApp(settings=settings, history=history, stats=stats,
                       microphones=["Microphone array", "Headset (Bluetooth)"],
                       gateway=GatewayConfig("", "AIza-example-key-9f3c", "gemini"), profiles=profiles)
    app.add_sound_alike("post grass", "PostgreSQL")
    spend(app.usage)
    return app


def spend(ledger) -> None:
    """A month of Gemini use for the Home page's card, recorded as Rflow records it (tokens and seconds, priced by
    sst.costs): dictation's cleanup, live translation with the cost saver on, a meeting's minutes, translations."""
    today = date.today()
    for back in range(min(today.day, 12)):
        ledger._today = lambda back=back: today - timedelta(days=back)
        ledger.record("cleanup", "gemini-3.5-flash-lite", text_in=24_000, out=6_000, calls=40)
        ledger.record("translate", "gemini-3.5-flash-lite", text_in=3_000, out=1_500, calls=6)
        if back % 3 == 0:  # a meeting every few days: translated as it runs, half its silence not sent, then minutes
            ledger.record("live_translation", "gemini-3.5-live-translate-preview", audio_in=25 * 1500,
                          audio_out=25 * 1500, seconds=1500, calls=0, unsent=900)
            ledger.record("meeting", "gemini-3.6-flash", audio_in=32 * 2400, text_in=6_000, out=12_000, seconds=2400)
        if back % 4 == 1:
            ledger.record("screen", "gemini-3.5-flash-lite", text_in=8_000, out=3_000, calls=12)
    ledger._today = date.today


def shot(name: str, page: str, theme_name: str = "dark", size=(1000, 680), setup=None, app=None) -> None:
    from sst import window as w
    window = w.MainWindow(app or preview())
    window.apply_theme(theme_name)
    window.resize(*size)
    window.set_status("Ready: hold Ctrl+Win · cleanup: gemini-3.5-flash-lite", True)
    window.show_page(page)
    if setup:
        setup(window)
    for _ in range(3):
        QApplication.processEvents()
    window.grab().save(str(OUT / name))


JAPANESE_TEXT = ("来週の定例会議は木曜日の午後3時からに変更になりました。"
                 "資料は水曜日までに共有フォルダにアップロードしてください。ご不明な点があれば、田中までご連絡ください。")
ENGLISH_TEXT = ("Next week's regular meeting has moved to Thursday at 3 PM. Please upload the materials to the shared "
                "folder by Wednesday. If you have any questions, please contact Tanaka.")


def text_translator(window) -> None:
    """Translate's Text tab with a paragraph translated, as the 800 ms pause leaves it."""
    page = window.pages["translate"]
    page.tabs.set_current("text")
    page._tab_chosen("text")
    text = page.text
    text.source.setCurrentIndex(max(0, text.source.findData("Japanese")))
    text.target.setCurrentIndex(max(0, text.target.findData("English")))
    text.input.setPlainText(JAPANESE_TEXT)
    text.timer.stop()
    text.output.setPlainText(ENGLISH_TEXT)
    text._say("0.9 s")


def meeting_app():
    """An app where a Teams meeting is open and three meetings are already in the list, with their minutes."""
    from sst.meeting.apps import MeetingWindow
    app = preview()
    app._meeting_windows = (MeetingWindow("Microsoft Teams", "Weekly sync | Microsoft Teams", 101, 7),)
    app._past_meetings = ((Path("a"), datetime.now() - timedelta(days=1, hours=2), "Zoom", 2710.0, True),
                          (Path("b"), datetime.now() - timedelta(days=3, hours=5), "In person", 1840.0, True),
                          (Path("c"), datetime.now() - timedelta(days=6, hours=1), "Google Meet", 1260.0, True))
    return app


def popups(name: str) -> None:
    """The Translate popup and the Text Transform menu side by side, as they float over an app."""
    from sst.transformui import TransformMenu
    from sst.translate import Translation
    from sst.translateui import TranslatePopup
    popup = TranslatePopup()
    popup._choices = ["English", "Spanish", "Japanese"]
    popup.route.setText("Japanese →")
    popup.original.setText("来週の定例会議は木曜日の午後3時からに変更になりました。")
    popup.show_result(Translation("Next week's regular meeting has moved to Thursday at 3 PM. Please upload the materials "
                                  "to the shared folder by Wednesday.", "English", "", 0.9))
    menu = TransformMenu()
    menu.items = [("concise", "1", "Concise"), ("professional", "2", "Professional"), ("bullets", "3", "Bullet points"),
                  ("actions", "4", "Action items"), ("undo", "U", "Undo last change")]
    menu.source = "Selected text · 3 sentences"
    menu.resize(menu.WIDTH + 2 * menu.MARGIN, menu._height() + 2 * menu.MARGIN)
    left, right = popup.grab(), menu.grab()
    dpr = left.devicePixelRatio()  # sizes below in logical pixels; the image keeps the screen's scale
    width, height = (left.width() + right.width()) / dpr + 8, max(left.height(), right.height()) / dpr
    image = QImage(round(width * dpr), round(height * dpr), QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.drawPixmap(0, 0, left)
    p.drawPixmap(round(left.width() / dpr + 8), 0, right)
    p.end()
    image.save(str(OUT / name))


def pills(name: str) -> None:
    """The pill as it goes: listening, cleaning up, typed."""
    from sst.app import Pill
    pill = Pill(level=lambda: 0.0)
    shots = []
    for state, ai, levels in [("recording", False, [0.2, 0.5, 0.8, 0.4, 0.9, 0.6, 0.3, 0.7, 1.0, 0.5, 0.35, 0.65, 0.85,
                                                     0.45, 0.25, 0.55]),
                              ("transcribing", True, []), ("typed", False, [])]:
        pill.state, pill.message, pill.ai, pill._started, pill._phase = state, "", ai, time.monotonic() - 7, 0.18
        pill._levels.extend(levels)
        pill.resize(pill._width() + 2 * pill.MARGIN, pill.HEIGHT + 2 * pill.MARGIN)
        shots.append(pill.grab())
    dpr = shots[0].devicePixelRatio()  # sizes below in logical pixels; the image keeps the screen's scale
    width = max(s.width() for s in shots) / dpr
    height = sum(s.height() for s in shots) / dpr - 2 * 24
    image = QImage(round(width * dpr), round(height * dpr), QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    y = 0.0
    for s in shots:
        p.drawPixmap(round((width - s.width() / dpr) / 2), round(y), s)
        y += s.height() / dpr - 24
    p.end()
    image.save(str(OUT / name))


# A meeting in Japanese, translated into English; with Both, the user's own words go into Japanese, marked "You".
LIVE_LINES = [  # (way, words heard, translation)
    ("system", "皆さん、おはようございます。それでは定例会議を始めます。",
     "Good morning, everyone. Let's start the weekly meeting."),
    ("system", "新しいリリースは来週の木曜日に予定しています。", "The new release is planned for next Thursday."),
    ("mic", "Testing will be finished by Wednesday.", "テストは水曜日までに終わります。"),
]
LIVE_SPEAKING = ("ありがとうございます。では、資料は金曜日までに共有フォルダへ",  # the line being spoken
                 "Thank you. Then please put the materials in the shared folder by Friday")


def live_example(name: str) -> None:
    """The translation bar as it floats over an online meeting: the real CaptionBar with a few lines, over an abstract
    call (soft tiles and initials, no faces, no real product's look)."""
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QColor, QLinearGradient, QRadialGradient

    from sst import theme
    from sst.live.captions import CaptionBar
    from sst.live.contracts import MIC, SYSTEM, Kind, LiveConfig, LiveEvent
    bar = CaptionBar(LiveConfig(target="en", source="both", mic_target="ja", speak=True))
    bar.resize(760, 256)
    bar.status = "Listening"
    bar._update_title()
    bar.set_speaking(True)
    for way, heard, said in LIVE_LINES:
        bar.show_event(LiveEvent(Kind.LINE, said, source=heard, lane=MIC if way == "mic" else SYSTEM))
    heard, said = LIVE_SPEAKING
    bar.show_event(LiveEvent(Kind.SOURCE, heard))
    bar.show_event(LiveEvent(Kind.TRANSLATION, said))
    bar.layout().activate()
    for _ in range(3):
        QApplication.processEvents()
    bar._to_bottom()
    caption = bar.grab()
    dpr = caption.devicePixelRatio()

    width, height = 1200, 720  # logical pixels; the image keeps the screen's scale
    image = QImage(round(width * dpr), round(height * dpr), QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#0E1014"))
    p.drawRoundedRect(QRectF(0, 0, width, height), 22, 22)
    tiles = [("SK", "Sato", "#3B4A7A", "#1E2540"), ("TM", "Tanaka", "#4A3B6E", "#251E3A"),
             ("YN", "Yamada", "#2F5A5A", "#17302F"), ("AL", "Alex", "#5A4A30", "#2E2618")]
    gap, top, bottom = 14, 18, 84
    tile_w, tile_h = (width - 3 * gap - 4) / 2, (height - top - bottom - gap) / 2
    for i, (initials, who, light, dark) in enumerate(tiles):
        rect = QRectF(gap + 2 + (i % 2) * (tile_w + gap), top + (i // 2) * (tile_h + gap), tile_w, tile_h)
        fill = QLinearGradient(rect.topLeft(), rect.bottomRight())
        fill.setColorAt(0, QColor(dark).lighter(115))
        fill.setColorAt(1, QColor(dark).darker(130))
        p.setBrush(fill)
        p.drawRoundedRect(rect, 16, 16)
        glow = QRadialGradient(rect.center() + QPointF(rect.width() * .18, -rect.height() * .1), rect.width() * .55)
        soft = QColor(light)
        soft.setAlpha(70)
        glow.setColorAt(0, soft)
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(glow)
        p.drawRoundedRect(rect, 16, 16)
        disc = QRectF(0, 0, 92, 92)
        disc.moveCenter(rect.center() - QPointF(0, 10))
        p.setBrush(QColor(light))
        p.drawEllipse(disc)
        p.setPen(QColor("#ECEEF3"))
        p.setFont(theme.font(28, 600))
        p.drawText(disc, Qt.AlignmentFlag.AlignCenter, initials)
        p.setFont(theme.font(13, 500))
        label = QRectF(rect.left() + 14, rect.bottom() - 40, 110, 28)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 110))
        p.drawRoundedRect(label, 8, 8)
        p.setPen(QColor("#DDE1EA"))
        p.drawText(label.adjusted(10, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, who)
        p.setPen(Qt.PenStyle.NoPen)
    for i, colour in enumerate(["#2A2D35", "#2A2D35", "#2A2D35", "#2A2D35", "#C23B3B"]):  # the call's buttons
        size = 44
        x = width / 2 - (5 * size + 4 * 14) / 2 + i * (size + 14)
        p.setBrush(QColor(colour))
        p.drawEllipse(QRectF(x, height - bottom / 2 - size / 2 - 4, size, size))
    p.drawPixmap(QPointF((width - caption.width() / dpr) / 2, height - bottom - caption.height() / dpr - 6), caption)
    p.end()
    image.save(str(OUT / name))


def main() -> None:
    QApplication([])
    parakeet.find_model = lambda: Path(".")  # as if Parakeet were downloaded: the screenshots show Rflow ready
    OUT.mkdir(parents=True, exist_ok=True)
    shot("app.png", "home")
    shot("app-light.png", "home", "light")
    shot("models.png", "models")
    shot("live.png", "live")
    shot("transform.png", "transform")
    shot("words.png", "dictionary")
    shot("translate.png", "translate", setup=text_translator)
    shot("translate-light.png", "translate", "light", setup=text_translator)
    shot("meeting.png", "meeting", app=meeting_app())
    app = preview()
    app.settings.welcomed = False
    shot("welcome.png", "welcome", app=app)
    popups("popups.png")
    pills("pill.png")
    live_example("live-example.png")
    print("Wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
