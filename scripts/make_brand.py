"""Make Rflow's logo files from the logo artwork (docs/brand/rflow-logo.webp, the owner's logo of 2026-10-05: a
blue-to-violet ribbon "R" above the word "Rflow", on white).

  sst/static/sst.ico            the app icon: the window, the taskbar, the tray, Rflow.exe and the installer, in every
                                size Windows asks for (16 to 256 px)
  sst/static/brand/             the mark alone, transparent, for the window (the sidebar, the first run)
  site/favicon.ico, site/img/   the website's favicon, logo, touch icon and social preview card

The mark is cut out of the white around it: a pixel as coloured as the ribbon (or more) is the ribbon, solid; a paler
one is an edge, made transparent in proportion and given back its own colour, so the mark has soft edges on any
background instead of a white fringe. Run it again after changing the artwork:

  uv run python scripts/make_brand.py
"""
import os
import struct
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # nothing is shown; the Geist fonts come with Rflow
import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QRadialGradient
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "brand" / "rflow-logo.webp"
BRAND = ROOT / "sst" / "static" / "brand"
ICON = ROOT / "sst" / "static" / "sst.ico"
SITE = ROOT / "site"
SOLID = 145  # how far from white a pixel must be to count as the ribbon itself (the ribbon's palest part is ~150)
ICON_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
MARK_SIZES = [64, 128, 256, 512]
OBSIDIAN, TEXT, TEXT_2 = QColor("#191C22"), QColor("#ECEEF3"), QColor("#A6ACBA")


def _pixels(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    raw = np.frombuffer(image.constBits(), np.uint8, image.sizeInBytes()).reshape(image.height(), image.bytesPerLine())
    return raw[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()


def _image(rgba: np.ndarray) -> QImage:
    height, width = rgba.shape[:2]
    image = QImage(np.ascontiguousarray(rgba).tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
    return image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)  # premultiplied: scaling keeps clean edges


def cut_mark() -> np.ndarray:
    """The ribbon "R" alone (above the wordmark), RGBA, cut out of the white around it."""
    rgb = _pixels(QImage(str(SOURCE)))[..., :3].astype(np.float32)
    distance = 255 - rgb.min(axis=2)  # 0 on white, ~150-230 on the ribbon
    ink = distance > 12
    rows = np.where(ink.any(axis=1))[0]
    blank = ~ink.any(axis=1)
    gap = next(y for y in range(rows.min(), rows.max()) if blank[y])  # the white band between the mark and the word
    cols = np.where(ink[:gap].any(axis=0))[0]
    top, bottom, left, right = rows.min(), gap, cols.min(), cols.max() + 1
    rgb, distance = rgb[top - 2:bottom + 2, left - 2:right + 2], distance[top - 2:bottom + 2, left - 2:right + 2]
    alpha = np.clip(distance / SOLID, 0, 1)
    alpha[distance <= 6] = 0  # the paper's own grain
    safe = np.maximum(alpha, 1e-6)[..., None]
    colour = np.clip((rgb - 255 * (1 - alpha[..., None])) / safe, 0, 255)  # an edge pixel's colour, without the white
    out = np.zeros((*alpha.shape, 4), np.uint8)
    out[..., :3] = np.round(colour)
    out[..., 3] = np.round(alpha * 255)
    return out


def square(rgba: np.ndarray, margin: float) -> np.ndarray:
    """The mark centred on a transparent square, `margin` (a fraction of the side) around it."""
    height, width = rgba.shape[:2]
    side = round(max(height, width) / (1 - 2 * margin))
    out = np.zeros((side, side, 4), np.uint8)
    y, x = (side - height) // 2, (side - width) // 2
    out[y:y + height, x:x + width] = rgba
    return out


def scaled(image: QImage, size: int) -> QImage:
    """Down to `size` px in halves first: one big smooth step blurs and aliases a small icon."""
    while image.width() >= 2 * size:
        image = image.scaled(image.width() // 2, image.height() // 2, Qt.AspectRatioMode.IgnoreAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
    return image.scaled(size, size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)


def _png(image: QImage) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(data)


def write_ico(path: Path, images: list[QImage]) -> None:
    """A Windows icon with every size in it, each as a PNG (Windows Vista and later, PyInstaller and Inno Setup)."""
    blobs = [_png(image) for image in images]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for image, blob in zip(images, blobs, strict=True):
        side = image.width()
        entries += struct.pack("<BBBBHHII", side % 256, side % 256, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    path.write_bytes(header + entries + b"".join(blobs))


def touch_icon(mark: QImage, size: int = 180) -> QImage:
    """The phone's home-screen icon: the mark on Obsidian (the phone rounds the corners)."""
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(OBSIDIAN)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    inner = round(size * 0.66)
    p.drawImage(QRectF((size - inner) / 2, (size - inner) / 2, inner, inner), scaled(mark, inner))
    p.end()
    return image


def social_card(mark: QImage) -> QImage:
    """The preview a link to the website shows (1200 x 630): the mark, Rflow and what it does, on Obsidian. Only it:
    uv run python -c "import sys; sys.path.insert(0, 'scripts'); import make_brand; make_brand.og()"
    """
    from sst import theme
    theme.load_fonts()
    width, height = 1200, 630
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(OBSIDIAN)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    for (x, y, r, colour) in ((260, 250, 420, QColor(33, 154, 254, 46)), (330, 390, 380, QColor(126, 72, 252, 40))):
        glow = QRadialGradient(QPointF(x, y), r)  # the mark's own blue and violet, as a faint light behind it
        glow.setColorAt(0, colour)
        glow.setColorAt(1, QColor(colour.red(), colour.green(), colour.blue(), 0))
        p.fillRect(image.rect(), glow)
    side = 260
    p.drawImage(QRectF(150, (height - side) / 2, side, side), scaled(mark, side))
    p.setPen(TEXT)
    p.setFont(theme.font(112, 600, spacing=-0.03))
    p.drawText(QRectF(470, 170, 700, 140), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Rflow")
    p.setPen(TEXT_2)
    p.setFont(theme.font(38, 500))
    p.drawText(QRectF(474, 315, 700, 60), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               "Work in any language.")
    p.setFont(theme.font(26, 500))
    p.setPen(QColor(138, 145, 161))
    p.drawText(QRectF(476, 385, 700, 44), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               "Translate, type and take meeting notes · Windows")
    p.end()
    return image


def og() -> None:
    """The social card alone, from the mark already cut (sst/static/brand), for when only its words change."""
    QApplication([])
    social_card(QImage(str(BRAND / "rflow-mark-512.png"))).save(str(SITE / "img" / "og.png"))


def main() -> None:
    QApplication([])
    mark = cut_mark()
    BRAND.mkdir(parents=True, exist_ok=True)
    tight = _image(square(mark, 0.01))  # the window places the mark itself
    for size in MARK_SIZES:
        scaled(tight, size).save(str(BRAND / f"rflow-mark-{size}.png"))
    icon = _image(square(mark, 0.04))  # an icon keeps a hair of room, as Windows' own do
    write_ico(ICON, [scaled(icon, size) for size in ICON_SIZES])
    (SITE / "favicon.ico").write_bytes(ICON.read_bytes())
    (SITE / "img").mkdir(exist_ok=True)
    scaled(tight, 512).save(str(SITE / "img" / "logo.png"))
    touch_icon(tight).save(str(SITE / "img" / "touch-icon.png"))
    social_card(tight).save(str(SITE / "img" / "og.png"))
    print("Wrote", ICON.relative_to(ROOT), ", ".join(p.name for p in sorted(BRAND.glob("*.png"))),
          "site/favicon.ico, site/img/logo.png, touch-icon.png, og.png")


if __name__ == "__main__":
    main()
