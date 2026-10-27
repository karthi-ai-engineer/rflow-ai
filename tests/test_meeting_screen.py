"""Seeing the meeting window: when a frame counts as changed, and which frames WindowFrames keeps (fake grabs only)."""
import numpy as np
from PySide6.QtGui import QImage

from sst.meeting.apps import TEAMS, MeetingWindow
from sst.meeting.screen import MAX_WIDTH, WindowFrames, changed, frame_name

WINDOW = MeetingWindow(TEAMS, "Meeting with Priya | Microsoft Teams", 11, 7)
H, W = 720, 1280


def slide(seed):
    """A white slide with dark text-like bars, different for each seed."""
    rng = np.random.default_rng(seed)
    image = np.full((H, W, 3), 245, np.uint8)
    for _ in range(40):
        y, x = rng.integers(40, H - 40), rng.integers(40, W - 400)
        image[y:y + 18, x:x + rng.integers(100, 360)] = 30
    return image


def with_tile(image, seed):
    """The same picture with a webcam tile in the corner showing something else (a moving head)."""
    out = image.copy()
    out[H - 200:H - 20, W - 260:W - 20] = np.random.default_rng(seed).integers(0, 255, (180, 240, 3), np.uint8)
    return out


def test_slide_change_counts():
    assert changed(slide(1), slide(2))


def test_webcam_tile_and_noise_dont_count():
    base = with_tile(slide(1), 1)
    assert not changed(base, with_tile(slide(1), 2))
    noisy = np.clip(base.astype(int) + np.random.default_rng(3).integers(-4, 5, base.shape), 0, 255).astype(np.uint8)
    assert not changed(base, noisy)


def test_a_resized_window_counts():
    assert changed(slide(1), slide(1)[:600])


def test_frame_name():
    assert frame_name(1, 35.4) == "0001 00m35s.jpg"
    assert frame_name(12, 3725) == "0012 62m05s.jpg"


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_keeps_only_changed_frames_and_skips_minimised(tmp_path):
    clock, kept = Clock(), []
    shown = iter([slide(1), slide(1), None, with_tile(slide(1), 5), slide(2), None, slide(2)])
    frames = WindowFrames(WINDOW, tmp_path, lambda path, seconds: kept.append((path.name, seconds)),
                          grab=lambda hwnd: next(shown), clock=clock)
    frames._started = clock()
    for _ in range(7):
        frames.look()
        clock.now += 5
    assert kept == [("0001 00m00s.jpg", 0.0), ("0002 00m20s.jpg", 20.0)]
    assert [(s, p.name) for s, p in frames.frames] == [(0.0, "0001 00m00s.jpg"), (20.0, "0002 00m20s.jpg")]
    assert (frames.seen, frames.kept, frames.skipped) == (7, 2, 2)
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["0001 00m00s.jpg", "0002 00m20s.jpg"]


def test_wide_frames_are_scaled_down(tmp_path):
    frames = WindowFrames(WINDOW, tmp_path, grab=lambda hwnd: np.zeros((1200, 2400, 3), np.uint8) + 128, clock=Clock())
    path = frames.look()
    picture = QImage(str(path))
    assert (picture.width(), picture.height()) == (MAX_WIDTH, 800)


def test_thread_starts_and_stops(tmp_path):
    grabs = []
    frames = WindowFrames(WINDOW, tmp_path, grab=lambda hwnd: grabs.append(hwnd), interval=0.01)
    frames.start()
    for _ in range(500):
        if len(grabs) >= 3:
            break
        frames._stop.wait(0.01)
    frames.stop()
    count = len(grabs)
    assert count >= 3 and set(grabs) == {11} and frames.skipped == frames.seen == count
    frames._stop.wait(0.05)
    assert len(grabs) == count  # nothing seen after stop
