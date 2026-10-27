"""Screen translation's OCR lines (sst.screen.ocr): Windows OCR's quirks for Japanese, with a fake engine."""
import numpy as np
import pytest

from sst.screen.capture import Frame, to_gray
from sst.screen.ocr import ScreenOcr, join_words, upscale
from sst.screen.winocr import Word


class FakeEngine:
    """Answers with words given in the frame's own pixels, scaled as the frame was before reading."""

    max_dim = 10000

    def __init__(self, lines):
        self.lines, self.seen = lines, []

    def recognize(self, gray: bytes, width: int, height: int):
        self.seen.append(np.frombuffer(gray, np.uint8).reshape(height, width))
        s = width / self.frame_w
        return [[Word(t, x * s, y * s, w * s, h * s) for t, x, y, w, h in line] for line in self.lines]


def _ocr(lines, gray=None, frame_xy=(100, 200)):
    gray = np.full((120, 400), 250, np.uint8) if gray is None else gray
    engine = FakeEngine(lines)
    engine.frame_w = gray.shape[1]
    frame = Frame.of_gray(gray, *frame_xy)
    return ScreenOcr(engine).recognize(frame), engine


def test_japanese_is_rebuilt_without_the_spaces_windows_puts_between_characters():
    assert join_words(["請", "求", "書"]) == "請求書"
    assert join_words(["Hello", "World", "2026", "年"]) == "Hello World 2026年"
    assert join_words(["ＡＢＣ"]) == "ABC"  # NFKC


def test_the_thousands_comma_read_as_a_period_is_put_back_only_for_money_and_counts():
    assert join_words(["128.000", "円"]) == "128,000円"
    assert join_words(["¥", "1.500"]) == "¥1,500"
    assert join_words(["3.141"]) == "3.141"  # a decimal stays


def test_leading_junk_is_dropped_but_brackets_kept():
    assert join_words(["・", "設定"]) == "設定"
    assert join_words(["「", "完了", "」"]) == "「完了」"


def test_lines_come_back_in_absolute_screen_pixels_from_the_upscaled_reading():
    lines, engine = _ocr([[("保", 20, 40, 14, 14), ("存", 34, 40, 14, 14)]])
    assert engine.seen[0].shape == (240, 800)  # read at 2x
    (line,) = lines
    assert line.text == "保存"
    assert (line.x, line.y, line.w, line.h) == pytest.approx((120, 240, 28, 14))
    assert line.char_h == pytest.approx(14)


def test_a_wide_gap_splits_table_cells_into_lines():
    lines, _ = _ocr([[("氏", 20, 40, 14, 14), ("名", 34, 40, 14, 14), ("山", 120, 40, 14, 14), ("田", 134, 40, 14, 14)]])
    assert [ln.text for ln in lines] == ["氏名", "山田"]


def test_a_column_read_as_one_vertical_line_becomes_rows():
    column = [("担", 20, 20, 14, 14), ("佐", 20, 50, 14, 14), ("鈴", 20, 80, 14, 14)]
    lines, _ = _ocr([column, [("当", 34, 20, 14, 14)]])
    assert sorted(ln.text for ln in lines) == ["佐", "担当", "鈴"]


def test_a_dark_theme_is_read_inverted():
    _, engine = _ocr([], gray=np.full((120, 400), 30, np.uint8))
    assert engine.seen[0].min() > 200


def test_a_short_label_cut_by_the_frame_edge_is_dropped():
    gray = np.full((120, 400), 250, np.uint8)
    gray[40:54, 0:3] = 0  # ink between the edge and the line: a glyph the edge sliced
    lines, _ = _ocr([[("名", 6, 40, 14, 14)], [("住", 200, 40, 14, 14), ("所", 214, 40, 14, 14)]], gray=gray)
    assert [ln.text for ln in lines] == ["住所"]


def test_upscale_and_grey():
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    assert upscale(gray, 2.0).shape == (6, 8)
    assert upscale(gray, 1.0) is not gray or upscale(gray, 1.0).shape == (3, 4)
    bgra = np.zeros((1, 3, 4), np.uint8)
    bgra[0, 0, 2], bgra[0, 1, 1], bgra[0, 2, 0] = 255, 255, 255  # red, green, blue
    assert to_gray(bgra).tolist() == [[76, 150, 29]]
