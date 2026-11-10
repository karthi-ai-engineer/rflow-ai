"""Render the README's animations (docs/media) from the real Rflow window and pill, with the website's example data.

Drawn off the screen like scripts/make_site_screenshots.py (nothing appears, nothing takes focus). Writing GIFs needs
Pillow, which Rflow itself doesn't use, so it's added only for this run:

  uv run --with pillow python scripts/make_readme_media.py
"""
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_site_screenshots as site  # noqa: E402  (sets the Windows platform before Qt starts)
from PIL import Image  # noqa: E402
from PySide6.QtCore import QBuffer, QIODevice  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "media"
TOUR = ["home", "live", "meeting", "translate", "transform", "models"]  # the sections a new user meets first
WIDTH = 960  # pixels in the GIF: sharp on GitHub, small enough to load quickly
BACKDROP = "#191C22"  # the pill floats over the app's own graphite, as on the website


def pil(image: QImage) -> Image.Image:
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGB")


def scaled(frame: Image.Image) -> Image.Image:
    return frame.resize((WIDTH, round(frame.height * WIDTH / frame.width)), Image.Resampling.LANCZOS)


def save_gif(name: str, frames: list[Image.Image], durations: list[int]) -> None:
    # One palette for every frame keeps the colours steady from section to section.
    palette = frames[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(OUT / name, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"Wrote {name}: {len(frames)} frames, {(OUT / name).stat().st_size // 1024} KB")


def tour(name: str) -> None:
    """The window going from section to section, as a new user would click through it."""
    from sst import window as w
    window = w.MainWindow(site.meeting_app())  # a Teams meeting open: Meeting mode is ready for it
    window.apply_theme("dark")
    window.resize(1000, 680)
    window.set_status("Ready: hold Ctrl+Win · cleanup: gemini-3.5-flash-lite", True)
    frames = []
    for page in TOUR:
        window.show_page(page)
        if page == "translate":
            site.text_translator(window)  # the Text tab with a paragraph translated
        for _ in range(3):
            QApplication.processEvents()
        frames.append(scaled(pil(window.grab().toImage())))
    save_gif(name, frames, [2200] * len(frames))


def pill(name: str) -> None:
    """The pill while you speak (the waveform moving), then cleaning up, then typed."""
    from sst.app import Pill
    p = Pill(level=lambda: 0.0)
    shots, durations = [], []
    waves = [[0.2, 0.5, 0.8, 0.4, 0.9, 0.6, 0.3, 0.7], [0.6, 0.3, 0.9, 0.7, 0.4, 1.0, 0.5, 0.2],
             [0.4, 0.8, 0.5, 1.0, 0.6, 0.3, 0.85, 0.45], [0.7, 0.4, 0.6, 0.35, 0.95, 0.55, 0.25, 0.65]]
    for i in range(12):  # about 2.4 s of speaking
        p.state, p.message, p.ai, p._started, p._phase = "recording", "", False, time.monotonic() - 3 - i * 0.2, i * 0.3
        p._levels.extend(waves[i % len(waves)])
        shots.append(_grab(p))
        durations.append(200)
    for state, ai, wait in [("transcribing", True, 1400), ("typed", False, 1800)]:
        p.state, p.message, p.ai = state, "", ai
        shots.append(_grab(p))
        durations.append(wait)
    width = max(s.width() for s in shots)
    height = max(s.height() for s in shots)
    frames = []
    for s in shots:
        canvas = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.setDevicePixelRatio(s.devicePixelRatio())
        canvas.fill(QColor(BACKDROP))
        painter = QPainter(canvas)
        dpr = s.devicePixelRatio()
        painter.drawImage(round((width - s.width()) / 2 / dpr), round((height - s.height()) / 2 / dpr), s)
        painter.end()
        frames.append(pil(canvas))
    save_gif(name, frames, durations)


def _grab(p) -> QImage:
    p.resize(p._width() + 2 * p.MARGIN, p.HEIGHT + 2 * p.MARGIN)
    return p.grab().toImage()


def main() -> None:
    QApplication([])
    site.parakeet.find_model = lambda: Path(".")  # as if Parakeet were downloaded: Rflow shows as ready
    OUT.mkdir(parents=True, exist_ok=True)
    tour("tour.gif")
    pill("pill.gif")


if __name__ == "__main__":
    main()
