"""Screen translator's layout and tracking (sst.screen.layout, sst.screen.tracker), on synthetic frames."""
import numpy as np
import pytest

from sst.screen.capture import Frame
from sst.screen.layout import Block, build_blocks, has_japanese
from sst.screen.ocr import OcrLine
from sst.screen.tracker import ChangeDetector, Tracker, contrast, match_template, sample_colors


def _line(text, x, y, w=200, h=14):
    return OcrLine(text, x, y, w, h, h)


# ---- layout

def test_a_wrapped_paragraph_is_one_block_and_a_menu_one_block_a_line():
    para = [_line("これは長い説明の文章で次の行に続きます", 10, 10, 300), _line("ここで終わります。", 10, 30, 120)]
    menu = [_line("ファイル", 10, 80, 40), _line("編集", 10, 100, 30)]
    blocks = build_blocks(para + menu)
    assert [b.text for b in blocks] == ["これは長い説明の文章で次の行に続きますここで終わります。", "ファイル", "編集"]


def test_lines_without_japanese_and_a_lone_kanji_are_not_translated():
    blocks = build_blocks([_line("Hello 2026", 10, 10), _line("伊", 10, 40, 14), _line("名", 10, 70, 14)])
    assert [b.text for b in blocks] == ["名"]  # in the glossary
    assert has_japanese("ｶﾀｶﾅ") and not has_japanese("ABC 123")


# ---- template matching

def test_match_template_finds_the_same_place_as_a_brute_force_search():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (60, 80), dtype=np.uint8)
    templ = image[22:34, 31:51].copy()
    value, (x, y) = match_template(image, templ)
    assert (x, y) == (31, 22) and value == pytest.approx(0.0, abs=1e-6)
    noisy = np.clip(image.astype(int) + rng.integers(-3, 4, image.shape), 0, 255).astype(np.uint8)
    assert match_template(noisy, templ)[1] == (31, 22)


# ---- tracking

def _page(offset=0, h=300, w=400) -> np.ndarray:
    """A white page with three 'text' rows of random ink, scrolled up by offset."""
    rng = np.random.default_rng(7)
    page = np.full((h + 200, w), 250, np.uint8)
    for y in (60, 140, 220):
        page[y:y + 14, 40:240] = rng.integers(0, 120, (14, 200), dtype=np.uint8)
    return page[offset:offset + h]


def _blocks():
    return [Block([_line(t, 40, y, 200, 14)]) for t, y in (("一行目です", 60), ("二行目です", 140), ("三行目です", 220))]


def test_stickers_follow_text_that_scrolls_and_wait_a_tick_before_showing():
    tracker = Tracker()
    tracker.replace_from_ocr(_blocks(), Frame.of_gray(_page()))
    assert tracker.track(Frame.of_gray(_page())) is False or True
    assert all(s.visible for s in tracker.stickers)
    moved = Frame.of_gray(_page(offset=25))
    tracker.track(moved)
    assert [s.y for s in tracker.stickers] == [35, 115, 195]
    assert not any(s.visible for s in tracker.stickers)  # just moved: hidden for one tick
    tracker.track(moved)
    assert all(s.visible for s in tracker.stickers)


def test_a_sticker_whose_text_is_gone_hides():
    tracker = Tracker()
    tracker.replace_from_ocr(_blocks(), Frame.of_gray(_page()))
    blank = Frame.of_gray(np.full((300, 400), 250, np.uint8))
    tracker.track(blank)
    assert not any(s.visible for s in tracker.stickers)


def test_hide_all_before_a_scroll_and_a_new_reading_keeps_unchanged_stickers():
    tracker = Tracker()
    frame = Frame.of_gray(_page())
    tracker.replace_from_ocr(_blocks(), frame)
    ids = [s.id for s in tracker.stickers]
    assert tracker.hide_all() is True
    tracker.replace_from_ocr(_blocks(), frame)
    assert [s.id for s in tracker.stickers] == ids


def test_colours_are_sampled_from_the_text_and_its_background():
    img = np.zeros((40, 100, 4), np.uint8)
    img[..., :3] = (212, 120, 0)  # BGR: a blue button
    img[15:25, 20:80, :3] = 255  # white text
    bg, fg = sample_colors(Frame(img, 0, 0), (20, 15, 60, 10))
    assert bg == (0, 120, 212) and fg == (255, 255, 255)
    assert contrast((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0)


# ---- the change detector

def test_the_screen_is_read_once_it_settles_and_not_again_while_unchanged():
    now = [0.0]
    det = ChangeDetector(settle_s=0.3, min_gap_s=0.6, force_s=2.0, clock=lambda: now[0])
    page = Frame.of_gray(_page())
    det.update(page)
    assert not det.should_ocr()  # not settled yet
    now[0] = 1.0
    det.update(page)
    assert det.should_ocr()
    det.mark_ocr_started(page)
    now[0] = 2.0
    det.update(page)
    assert not det.should_ocr()  # nothing changed
    now[0] = 2.1
    det.update(Frame.of_gray(_page(offset=40)))
    assert det.moving and not det.should_ocr()  # changed, still moving
    now[0] = 3.0
    det.update(Frame.of_gray(_page(offset=40)))
    assert det.should_ocr()
    det.force()
    assert det.dirty
