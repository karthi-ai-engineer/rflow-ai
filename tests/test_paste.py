"""Typing a dictation (sst.paste), with the clipboard and the keys faked: nothing is pasted for real."""
from contextlib import contextmanager

import pytest

from sst import paste


@pytest.fixture(autouse=True)
def settled(monkeypatch):
    """The clipboard is put back on a timer thread: it must happen while the fakes are still in place, never on the real
    clipboard after the test."""
    yield
    assert paste.settle(5)


def test_line_breaks_reach_the_app_as_windows_line_breaks(monkeypatch):
    # A snippet's lines: a classic edit box shows a bare "\n" as nothing, so the clipboard gets "\r\n".
    puts = []

    @contextmanager
    def session():
        yield

    class User32:
        @staticmethod
        def GetClipboardSequenceNumber():
            return 1

    for name, fake in {"_clipboard": session, "_snapshot": lambda: [], "_put": puts.append, "_press_ctrl_v": lambda: None,
                       "_wait_until_keys_released": lambda: None, "user32": User32}.items():
        monkeypatch.setattr(paste, name, fake)
    monkeypatch.setattr(paste.time, "sleep", lambda seconds: None)
    paste.paste_text("Best regards,\nKarthi\r\nKarthi Labs ")
    text = puts[0][0][1].decode("utf-16-le").rstrip("\0")
    assert text == "Best regards,\r\nKarthi\r\nKarthi Labs "


def test_a_paste_waits_while_the_keys_are_held_for_the_next_dictation(monkeypatch):
    held = iter([True] * 50 + [False])
    clock = iter(range(0, 1000))
    monkeypatch.setattr(paste, "_modifiers_held", lambda: next(held))
    monkeypatch.setattr(paste.time, "monotonic", lambda: next(clock))  # 50 s pass: longer than the old 5 s
    monkeypatch.setattr(paste.time, "sleep", lambda seconds: None)
    paste._wait_until_keys_released()  # returns once they're let go


def test_a_paste_is_never_pressed_with_the_keys_still_held(monkeypatch):
    clock = iter(range(0, 100_000, 100))  # past HELD_WAIT (an hour, a dictation's longest)
    monkeypatch.setattr(paste, "_modifiers_held", lambda: True)
    monkeypatch.setattr(paste.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(paste.time, "sleep", lambda seconds: None)
    pressed = []
    monkeypatch.setattr(paste, "_press_ctrl_v", lambda: pressed.append(True))
    try:
        paste.paste_text("hello ")
        raise AssertionError("it should have refused")
    except OSError as e:
        assert "held down" in str(e) and pressed == []


def fake_clipboard(monkeypatch, log, held=None):
    """A clipboard that records what happens to it, in order."""
    state = {"seq": 0, "content": [(13, "the user's own\0".encode("utf-16-le"))]}

    @contextmanager
    def session():
        if held is not None:
            held.wait(5)
        yield

    def put(items):
        state["seq"] += 1
        state["content"] = items
        log.append(("put", items[0][1].decode("utf-16-le").rstrip("\0"), len(items)))

    class User32:
        @staticmethod
        def GetClipboardSequenceNumber():
            return state["seq"]

    for name, fake in {"_clipboard": session, "_snapshot": lambda: list(state["content"]), "_put": put,
                       "_press_ctrl_v": lambda: log.append(("ctrl+v",)), "_wait_until_keys_released": lambda: None,
                       "user32": User32}.items():
        monkeypatch.setattr(paste, name, fake)
    monkeypatch.setattr(paste, "RESTORE_DELAY", 0.05)
    return state


def test_the_dictation_goes_on_at_once_and_the_clipboard_comes_back_after_the_app_read_it(monkeypatch):
    import time
    log = []
    fake_clipboard(monkeypatch, log)
    monkeypatch.setattr(paste, "RESTORE_DELAY", 0.3)
    t0 = time.perf_counter()
    paste.paste_text("hello ")
    assert time.perf_counter() - t0 < 0.2  # no longer blocks the dictation's worker
    assert log == [("put", "hello ", 2), ("ctrl+v",)]  # the user's clipboard isn't back before the app has read it
    assert paste.settle(2)
    assert log[-1] == ("put", "the user's own", 2)  # with the format that keeps it out of the clipboard history


def test_a_next_paste_waits_for_the_restore_instead_of_saving_our_text_as_the_users(monkeypatch):
    log = []
    fake_clipboard(monkeypatch, log)
    paste.paste_text("one ")
    paste.paste_text("two ")  # right after: the first restore runs first
    assert paste.settle(2)
    users = ("put", "the user's own")
    assert [step[:2] for step in log] == [("put", "one "), ("ctrl+v",), users, ("put", "two "), ("ctrl+v",), users]


def test_something_copied_meanwhile_is_left_on_the_clipboard(monkeypatch):
    log = []
    state = fake_clipboard(monkeypatch, log)
    paste.paste_text("hello ")
    state["seq"] += 1  # the user copied something new before the restore
    assert paste.settle(2)
    assert log == [("put", "hello ", 2), ("ctrl+v",)]


def test_a_failed_paste_lets_the_next_one_go(monkeypatch):
    log = []
    fake_clipboard(monkeypatch, log)

    @contextmanager
    def busy():
        raise OSError("the clipboard is busy")
        yield
    monkeypatch.setattr(paste, "_clipboard", busy)
    try:
        paste.paste_text("hello ")
        raise AssertionError("it should have failed")
    except OSError:
        pass
    assert paste.settle(0.5)  # not held by the failed paste
