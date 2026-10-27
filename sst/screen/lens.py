"""The frame: a hollow, movable, resizable window over any app, whose inside is covered with English stickers.

    FrameWindow    the ring and the toolbar (a mask cuts the inside out, so clicks there reach the app below); drag the
                   toolbar to move it, an edge or the corner to resize it
    OverlayWindow  the inside: click-through, never activated, left out of screen capture; it paints the stickers
    ScreenLens     owns both, and one worker thread that captures, tracks and decides when to read again (the UI thread
                   only paints); OCR has a thread of its own and hands its results back through a queue, so only the
                   worker changes the tracker; translations come from TranslationService's threads

The worker's pace (the owner's measurements, kaipulla's README): 33 ms while the user scrolls, 120 ms while the screen
changes, 300 then 700 ms as it stays quiet; Light (on battery) 100 / 300 / 600-1500 ms and OCR at most every 1.5 s, 1.5x.
Raw input (wheel, keys) wakes the worker and hides the stickers before the text under them moves. An animation or a
video changes the screen by itself: after 8 ticks of that the search stops and the pace calms.

At the start a magenta square is painted in the overlay and the screen read under it: seen, capture exclusion doesn't
work here (Remote Desktop, a VM), and the status line warns, since Rflow would read its own English.
"""
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QToolButton, QWidget

from sst import theme
from sst.screen import win32
from sst.screen.capture import Capturer, Frame
from sst.screen.layout import build_blocks
from sst.screen.ocr import ScreenOcr
from sst.screen.render import paint_stickers
from sst.screen.tracker import ChangeDetector, Tracker
from sst.screen.translate import Backend, TranslationService

log = logging.getLogger(__name__)

BORDER, TOOLBAR, GRIP = 3, 28, 14
MIN_SIZE, DEFAULT_SIZE = QSize(220, 120), QSize(960, 640)
MIN_INSIDE = (60, 30)  # physical px: a smaller inside isn't read


@dataclass(frozen=True)
class Pace:
    """How hard the worker works. idle: ((quiet seconds, tick ms), ...), ascending."""

    name: str
    base: int  # ms a tick while the screen changes or just changed
    idle: tuple
    fast: int  # ms a tick while the user scrolls or types
    settle: float  # s the screen must stand still before it's read
    min_gap: float  # s between two readings at least
    force: float  # s: read at least this often while the screen never settles
    ocr_scale: float  # the upscale before OCR: lower is cheaper


NORMAL = Pace("normal", 120, ((3.0, 300), (20.0, 700)), 33, 0.30, 0.6, 3.0, 2.0)
LIGHT = Pace("light", 300, ((3.0, 600), (20.0, 1500)), 100, 0.50, 1.5, 5.0, 1.5)


class _Bridge(QObject):
    """Worker threads' news, onto the UI thread."""

    repaint = Signal()
    translated = Signal()


class OverlayWindow(QWidget):
    def __init__(self, lens: "ScreenLens"):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.lens = lens
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.marker = False
        self.excluded = False

    def showEvent(self, e) -> None:
        hwnd = int(self.winId())
        win32.add_ex_style(hwnd, win32.WS_EX_TRANSPARENT | win32.WS_EX_LAYERED | win32.WS_EX_NOACTIVATE
                           | win32.WS_EX_TOOLWINDOW)
        self.excluded = win32.exclude_from_capture(hwnd)
        super().showEvent(e)

    def phys_rect(self) -> tuple[int, int, int, int]:
        return win32.window_rect(int(self.winId()))

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        if self.marker:
            p.fillRect(QRect(0, 0, 8, 8), QColor(255, 0, 255))
            return
        x, y, w, _ = self.lens.inside()
        scale = w / max(1, self.width()) if w else self.devicePixelRatioF()
        paint_stickers(p, self.lens.tracker.stickers, self.lens.svc.get, (x, y), scale, hover_id=self.lens.hover_id,
                       clip_w=self.width())


class _Toolbar(QWidget):
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self.window().windowHandle():
            self.window().windowHandle().startSystemMove()

    def mouseDoubleClickEvent(self, e) -> None:
        self.window().lens.refresh()


class FrameWindow(QWidget):
    def __init__(self, lens: "ScreenLens"):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.lens = lens
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setMinimumSize(MIN_SIZE)
        self.setWindowTitle("Rflow: screen translation")
        self.toolbar = _Toolbar(self)
        layout = QHBoxLayout(self.toolbar)
        layout.setContentsMargins(10, 2, 4, 2)
        layout.setSpacing(2)
        text, muted = theme.tok("text", popup=True).name(), theme.tok("text3", popup=True).name()
        title = QLabel("Screen translation")
        title.setFont(theme.font(12, 600))
        title.setStyleSheet(f"color: {text}; background: transparent;")
        self.status = QLabel("Starting…")
        self.status.setFont(theme.font(11))
        self.status.setStyleSheet(f"color: {muted}; background: transparent; padding-left: 8px;")
        for w in (title, self.status):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(title)
        layout.addWidget(self.status, 1)
        self.pause_button = self._button("Ⅱ", "Pause, or go on", lens.toggle_pause, checkable=True)
        self.original_button = self._button("", "Show the original", lens.toggle_original, checkable=True, icon="eye")
        self._button("⛶", "Fit to the window you were in", lens.fit_to_last)
        self._button("⟳", "Read again now", lens.refresh)
        self._button("", "Stop screen translation", lens.close, icon="close")

    def _button(self, text: str, tip: str, slot, checkable: bool = False, icon: str = "") -> QToolButton:
        b = QToolButton(self.toolbar)
        b.setText(text)
        if icon:
            b.setIcon(QIcon(theme.icon_pixmap(icon, theme.tok("text2", popup=True).name(), 14, 2.0)))
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setFixedSize(QSize(26, 22))
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        glow = theme.tok("glow", popup=True)
        b.setStyleSheet(f"QToolButton {{ color: {theme.tok('text2', popup=True).name()}; background: transparent; "
                        f"border: none; border-radius: 5px; font-size: 13px; }}"
                        f"QToolButton:hover {{ background: {theme.css(theme.tok('chip_bg', popup=True))}; }}"
                        f"QToolButton:checked {{ background: rgba({glow.red()}, {glow.green()}, {glow.blue()}, 150); }}")
        b.clicked.connect(slot)
        self.toolbar.layout().addWidget(b)
        return b

    def interior(self) -> QRect:
        return QRect(BORDER, TOOLBAR, self.width() - 2 * BORDER, self.height() - TOOLBAR - BORDER)

    def _update_mask(self) -> None:
        grip = QRegion(QRect(self.width() - GRIP, self.height() - GRIP, GRIP, GRIP))
        self.setMask(QRegion(self.rect()).subtracted(QRegion(self.interior())).united(grip))

    def sync_overlay(self) -> None:
        r = self.interior()
        self.lens.overlay.setGeometry(QRect(self.mapToGlobal(r.topLeft()), r.size()))
        self.lens._update_rect()

    def resizeEvent(self, e) -> None:
        self.toolbar.setGeometry(0, 0, self.width(), TOOLBAR)
        self._update_mask()
        self.sync_overlay()
        super().resizeEvent(e)

    def moveEvent(self, e) -> None:
        self.sync_overlay()
        super().moveEvent(e)

    def showEvent(self, e) -> None:
        hwnd = int(self.winId())
        win32.exclude_from_capture(hwnd)
        if not win32.register_raw_input(hwnd):
            log.info("Screen translation: no raw input; scrolling is seen from the screen only")
        super().showEvent(e)

    def nativeEvent(self, event_type, message):
        ptr = int(message)
        if win32.message_id(ptr) == win32.WM_INPUT and (kind := win32.raw_input_kind(ptr)):
            self.lens.on_input(kind)
        return False, 0

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar = QPainterPath()
        bar.addRoundedRect(0, 0, self.width(), TOOLBAR + 8, 8, 8)
        base = QColor(theme.tok("base", popup=True))
        base.setAlpha(245)
        p.fillPath(bar, base)
        accent = QColor(theme.tok("iris", popup=True))
        accent.setAlpha(140 if self.lens.paused else 235)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(accent)
        w, h = self.width(), self.height()
        p.drawRect(0, TOOLBAR, BORDER, h - TOOLBAR)
        p.drawRect(w - BORDER, TOOLBAR, BORDER, h - TOOLBAR)
        p.drawRect(0, h - BORDER, w, BORDER)
        p.drawPolygon([QPoint(w, h - GRIP), QPoint(w, h), QPoint(w - GRIP, h)])
        p.setPen(QPen(QColor(255, 255, 255, 180), 1.2))
        for i in (4, 8):
            p.drawLine(w - i, h - 2, w - 2, h - i)

    def _edges(self, pos: QPoint):
        x, y, w, h = pos.x(), pos.y(), self.width(), self.height()
        if x >= w - GRIP and y >= h - GRIP:
            return Qt.Edge.RightEdge | Qt.Edge.BottomEdge
        edges = Qt.Edge(0)
        if x <= BORDER + 1:
            edges |= Qt.Edge.LeftEdge
        if x >= w - BORDER - 1:
            edges |= Qt.Edge.RightEdge
        if y >= h - BORDER - 1:
            edges |= Qt.Edge.BottomEdge
        if y <= 3:
            edges |= Qt.Edge.TopEdge
        return edges

    def mouseMoveEvent(self, e) -> None:
        ed = self._edges(e.position().toPoint())
        if ed in (Qt.Edge.LeftEdge | Qt.Edge.TopEdge, Qt.Edge.RightEdge | Qt.Edge.BottomEdge):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif ed in (Qt.Edge.RightEdge | Qt.Edge.TopEdge, Qt.Edge.LeftEdge | Qt.Edge.BottomEdge):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif ed & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif ed & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, e) -> None:
        if e.button() != Qt.MouseButton.LeftButton or not self.windowHandle():
            return
        ed = self._edges(e.position().toPoint())
        if ed:
            self.windowHandle().startSystemResize(ed)
        else:
            self.windowHandle().startSystemMove()


class ScreenLens(QObject):
    """The frame and its worker. backend: Gemini (sst.screen.translate), a fake in tests; make_ocr: an OCR engine, made
    on the OCR thread (sst.screen.winocr.WinOcr); make_capturer: a Capturer, made on the worker's thread. on_close is
    called after the frame closed (the ✕, or close()), with the frame's [x, y, w, h] to remember."""

    def __init__(self, backend: Backend | None, *, make_ocr: Callable, make_capturer: Callable = Capturer,
                 on_close: Callable[[list[int]], None] | None = None, light: Callable[[], bool] = win32.on_battery):
        super().__init__()
        self.bridge = _Bridge()
        self.bridge.translated.connect(self._on_translated)
        self.bridge.repaint.connect(self._repaint)
        self.svc = TranslationService(backend, self.bridge.translated.emit)
        self.tracker, self.det = Tracker(), ChangeDetector()
        self._make_ocr, self._make_capturer, self._on_close, self._light = make_ocr, make_capturer, on_close, light
        self._pid = os.getpid()
        self._ocr: ScreenOcr | None = None
        self._ocr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="screen-ocr")
        self._ocr_results: queue.SimpleQueue = queue.SimpleQueue()
        self.ocr_busy, self.ocr_ms, self.ocr_error, self.tick_error = False, 0.0, "", ""
        self.paused = self.original = False
        self.hover_id: int | None = None
        self.capture_ok: bool | None = None
        self.pace = NORMAL
        self._battery_at, self._on_battery = 0.0, False
        self.fast_until = self.input_until = 0.0
        self.last_motion = time.monotonic()
        self._moving_ticks = 0
        self._last_retry = 0.0
        self._last_other: int | None = None  # the last active window that isn't Rflow's
        self._rect = (0, 0, 0, 0)  # the inside, in physical px (the UI thread keeps it current for the worker)
        self._stop, self._wake = threading.Event(), threading.Event()
        self.running = False
        self.overlay = OverlayWindow(self)
        self.frame = FrameWindow(self)
        self.status_timer = QTimer(self, interval=400, timeout=self.update_status)
        self.worker: threading.Thread | None = None

    # ------------------------------------------------------------ start and stop

    def start(self, geometry: list[int] | None = None, fit: int | None = None) -> None:
        """Show the frame: over window `fit` (its page area for a browser), else where it was (geometry), else centred."""
        self._place(geometry)
        self.overlay.show()
        self.frame.show()
        self.frame.sync_overlay()
        self._update_rect()
        if fit:
            self._last_other = fit
            self.fit_to(fit)
        self.running = True
        self._stop.clear()
        self.status_timer.start()
        self.worker = threading.Thread(target=self._run, name="screen-lens", daemon=True)
        self.worker.start()
        QTimer.singleShot(700, self._self_test)

    def close(self) -> None:
        if not self.running:
            return
        self.running = False
        g = self.frame.geometry()
        self._stop.set()
        self._wake.set()
        self.status_timer.stop()
        win32.unregister_raw_input()
        self.frame.hide()
        self.overlay.hide()
        self.svc.shutdown()
        self._ocr_pool.shutdown(wait=False, cancel_futures=True)
        if self._on_close is not None:
            self._on_close([g.x(), g.y(), g.width(), g.height()])

    def _place(self, geometry: list[int] | None) -> None:
        if geometry and len(geometry) == 4 and geometry[2] >= MIN_SIZE.width() and geometry[3] >= MIN_SIZE.height():
            saved = QRect(*geometry)
            for screen in QGuiApplication.screens():
                avail = screen.availableGeometry()
                if avail.intersects(saved):  # on the screen it was on, inside it
                    w, h = min(saved.width(), avail.width()), min(saved.height(), avail.height())
                    self.frame.setGeometry(min(max(saved.x(), avail.left()), avail.right() - w + 1),
                                           min(max(saved.y(), avail.top()), avail.bottom() - h + 1), w, h)
                    return
        screen = QGuiApplication.primaryScreen().availableGeometry()
        w, h = min(DEFAULT_SIZE.width(), screen.width() - 80), min(DEFAULT_SIZE.height(), screen.height() - 80)
        self.frame.setGeometry(screen.x() + (screen.width() - w) // 2, screen.y() + (screen.height() - h) // 2, w, h)

    def inside(self) -> tuple[int, int, int, int]:
        return self._rect

    def _update_rect(self) -> None:
        self._rect = self.overlay.phys_rect()

    # ------------------------------------------------------------ the capture-exclusion check

    def _self_test(self) -> None:
        if not self.running:
            return
        self.overlay.marker = True
        self.overlay.update()
        QTimer.singleShot(250, self._self_test_check)

    def _self_test_check(self) -> None:
        self.overlay.marker = False
        self.overlay.update()
        if not self.running:
            return
        x, y, _, _ = self.overlay.phys_rect()
        try:
            cap = self._make_capturer()
            try:
                f = cap.grab((x, y, 8, 8))
            finally:
                cap.close()
        except OSError:
            return
        if f is not None:
            b, g, r = (int(v) for v in f.img[3, 3, :3])
            self.capture_ok = not (r > 230 and b > 230 and g < 40)
            if not self.capture_ok:
                log.warning("Screen translation: screen capture sees the stickers (Remote Desktop or a VM?)")

    # ------------------------------------------------------------ the worker

    def _repaint(self) -> None:
        self._update_rect()
        self.overlay.update()

    def _run(self) -> None:
        cap, errors = None, 0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                cap = cap or self._make_capturer()
                wait = self.tick(cap) / 1000.0
                errors, self.tick_error = 0, ""
            except Exception as e:  # the lock screen, UAC's desktop, a monitor unplugged
                errors += 1
                self.tick_error = str(e) or type(e).__name__
                if errors == 1 or errors % 30 == 0:
                    log.warning("Screen translation: %s", self.tick_error)
                if errors >= 3 and cap is not None:  # start again from a clean capturer
                    try:
                        cap.close()
                    except OSError:
                        pass
                    cap = None
                wait = min(5.0, 0.5 * errors)
            self._wake.wait(max(0.0, wait - (time.perf_counter() - t0)))
            self._wake.clear()
        if cap is not None:
            cap.close()

    def _pick_pace(self, now: float) -> Pace:
        if now - self._battery_at > 20.0:
            self._on_battery, self._battery_at = bool(self._light()), now
        pace = LIGHT if self._on_battery else NORMAL
        if pace is not self.pace:
            self.pace = pace
            self.det.settle_s, self.det.min_gap_s, self.det.force_s = pace.settle, pace.min_gap, pace.force
        return pace

    def tick(self, cap) -> float:
        """One capture, track and detect step on the worker; the ms to wait before the next."""
        now = time.monotonic()
        self._last_other = win32.foreground_other(self._pid) or self._last_other
        pace = self._pick_pace(now)
        while True:
            try:
                self._apply_ocr(self._ocr_results.get_nowait())
            except queue.Empty:
                break
        if self.paused or self.original or self.overlay.marker:
            return 400
        rect = self._rect
        if rect[2] < MIN_INSIDE[0] or rect[3] < MIN_INSIDE[1]:
            if self.tracker.stickers:
                self.tracker.clear()
                self.bridge.repaint.emit()
            return 400
        frame = cap.grab(rect, now)
        if frame is None:
            return pace.base
        self.det.update(frame)
        moving, by_user = self.det.moving, now <= self.input_until
        self._moving_ticks = self._moving_ticks + 1 if moving else 0
        busy = moving and not by_user and self._moving_ticks > 8  # an animation or a video: not followed tick by tick
        changed = self.tracker.track(frame, search=not busy)
        if moving:
            self.last_motion = now
            if by_user:
                self.fast_until = max(self.fast_until, now + 0.4)
        cx, cy = win32.cursor_pos()
        hover = next((s.id for s in self.tracker.stickers if s.visible and s.contains(cx, cy)), None)
        if hover != self.hover_id:
            self.hover_id, changed = hover, True
        if not self.ocr_busy and self.det.should_ocr():
            self.ocr_busy = True
            self.det.mark_ocr_started(frame)
            self._ocr_pool.submit(self._ocr_job, frame, pace.ocr_scale)
        if now - self._last_retry > 1.0:  # ask again for what failed
            self._last_retry = now
            missing = [s.src for s in self.tracker.stickers if s.visible and self.svc.get(s.src) is None]
            if missing:
                self.svc.request(missing)
        if changed:
            self.bridge.repaint.emit()
        return self._interval(pace, now, busy)

    def _interval(self, pace: Pace, now: float, busy: bool) -> float:
        if now < self.fast_until:
            return pace.fast
        if self.det.moving:
            return max(pace.base, 250) if busy else pace.base
        quiet, ms = now - self.last_motion, pace.base
        for after, idle_ms in pace.idle:
            if quiet >= after:
                ms = idle_ms
        return ms

    def on_input(self, kind: str) -> None:
        """Raw input comes before the app under the frame repaints: the stickers can go before the text moves."""
        if self.paused or self.original:
            return
        x, y, w, h = self._rect
        if kind in ("wheel", "click"):
            cx, cy = win32.cursor_pos()
            if not (x <= cx < x + w and y <= cy < y + h):
                return  # elsewhere on the screen
        else:  # keys matter only for a window under the frame
            fr = win32.foreground_rect()
            if fr is None or fr[0] >= x + w or fr[0] + fr[2] <= x or fr[1] >= y + h or fr[1] + fr[3] <= y:
                return
        now = time.monotonic()
        self.last_motion, self.input_until = now, now + 1.0
        self.fast_until = max(self.fast_until, now + 0.7)
        if kind == "wheel" and self.tracker.hide_all():
            self.overlay.repaint()
        self._wake.set()

    def _ocr_job(self, frame: Frame, scale: float) -> None:
        try:
            if self._ocr is None:
                self._ocr = ScreenOcr(self._make_ocr())
            self._ocr.max_scale = scale
            blocks = build_blocks(self._ocr.recognize(frame))
            self._ocr_results.put((frame, blocks, self._ocr.last_ms, ""))
        except Exception as e:  # the loop goes on; the status line says it
            self._ocr_results.put((frame, None, 0.0, str(e) or type(e).__name__))
        self._wake.set()

    def _apply_ocr(self, payload) -> None:
        frame, blocks, ms, error = payload
        self.ocr_busy, self.ocr_error = False, error
        if blocks is None:
            return
        self.ocr_ms = ms
        self.tracker.replace_from_ocr(blocks, frame)
        self.svc.request([b.text for b in blocks])
        self.bridge.repaint.emit()

    def _on_translated(self) -> None:
        self.overlay.update()

    # ------------------------------------------------------------ the toolbar

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.frame.pause_button.setChecked(self.paused)
        self._show_overlay()
        if not self.paused:
            self.refresh()
        self.frame.update()

    def toggle_original(self) -> None:
        self.original = not self.original
        self.frame.original_button.setChecked(self.original)
        self._show_overlay()

    def _show_overlay(self) -> None:
        self.overlay.setVisible(self.running and not self.paused and not self.original)
        self._wake.set()

    def refresh(self) -> None:
        self.det.force()
        self._wake.set()

    def fit_to_last(self) -> None:
        if self._last_other:
            self.fit_to(self._last_other)

    def fit_to(self, hwnd: int) -> None:
        """The inside onto a window (a browser's page area). Physical to logical pixels differs from screen to screen,
        so: place, measure, correct."""
        target = win32.fit_rect(hwnd)
        if target is None:
            return
        tx, ty, tw, th = target
        vy = win32.virtual_screen()[1]
        clip = max(0, vy + TOOLBAR - ty)  # the toolbar would be above the screen
        ty, th = ty + clip, th - clip
        for attempt in range(3):
            dpr = self.frame.devicePixelRatio() or 1.0
            ox, oy, ow, oh = self.overlay.phys_rect()
            if attempt and (ox, oy, ow, oh) == (tx, ty, tw, th):
                break
            g = self.frame.geometry()
            if attempt == 0:
                self.frame.setGeometry(round(tx / dpr) - BORDER, round(ty / dpr) - TOOLBAR, round(tw / dpr) + 2 * BORDER,
                                       round(th / dpr) + TOOLBAR + BORDER)
            else:
                self.frame.setGeometry(g.x() + round((tx - ox) / dpr), g.y() + round((ty - oy) / dpr),
                                       g.width() + round((tw - ow) / dpr), g.height() + round((th - oh) / dpr))
            QApplication.processEvents()
        self.frame.sync_overlay()
        self._update_rect()
        self.refresh()

    def status_text(self) -> str:
        if self.paused:
            return "Paused"
        if self.original:
            return "Showing the original"
        parts = [f"{sum(1 for s in self.tracker.stickers if s.visible)} translated"]
        if pending := len(self.svc.pending):
            parts.append(f"translating {pending}…")
        if self.pace is LIGHT:
            parts.append("light (on battery)")
        if self.svc.status == "refused":
            parts.append(f"⚠ {self.svc.last_error}")
        elif self.svc.status == "offline":
            parts.append("⚠ Gemini can't be reached")
        for problem in (self.ocr_error, self.tick_error):
            if problem:
                parts.append(f"⚠ {problem[:80]}")
        if self.capture_ok is False:
            parts.append("⚠ screen capture sees the stickers (Remote Desktop?)")
        return " · ".join(parts)

    def update_status(self) -> None:
        self._update_rect()
        text = self.status_text()
        if self.frame.status.text() != text:
            self.frame.status.setText(text)
            self.frame.status.setToolTip(text)
