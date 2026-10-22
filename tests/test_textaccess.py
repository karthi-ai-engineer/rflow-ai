"""Text access for Text Transform, against a fake clipboard and a fake text box: no key is pressed, the real clipboard is
never touched and no window gets the focus."""
from contextlib import contextmanager

import pytest

from sst import textaccess
from sst.textaccess import (
    CF_EXCLUDE_FROM_HISTORY,
    CF_HTML,
    CF_UNICODETEXT,
    VK_A,
    VK_C,
    VK_CONTROL,
    VK_HOME,
    VK_INSERT,
    VK_LEFT,
    VK_RIGHT,
    VK_SHIFT,
    VK_V,
    activate,
    cf_html,
    collapse_selection,
    copy_selection,
    paste_rich,
    select_all,
    select_back,
    select_last,
    set_clipboard,
    text_before_caret,
    unselect_all,
)

CF_RTF = 0xC0F0  # stands for any registered format the user's copy carries besides its text


def utf16(text):
    return (text + "\0").encode("utf-16-le")


def decode(data):
    return data.decode("utf-16-le").split("\0", 1)[0]


USERS_COPY = {CF_UNICODETEXT: utf16("the user's own copy"), CF_RTF: b"{\\rtf1 rich}"}
RESTORED = {**USERS_COPY, CF_EXCLUDE_FROM_HISTORY: b"\0" * 4}  # put back, and kept out of Win+V history again
TAMIL = "வணக்கம்"  # 7 code points, 5 letters as a browser's caret steps: வ ண க் க ம்


class Desktop:
    """The clipboard and the focused text box, as textaccess sees them. The box holds `units`, the steps its caret takes
    (one per code point, or per cluster like a browser), and its selection runs from `anchor` to `caret`."""

    def __init__(self, text="", units=None, copies=True, insert_copies=True, select_all=True):
        self.clipboard, self.seq = dict(USERS_COPY), 1
        self.units = list(text) if units is None else list(units)
        self.caret = self.anchor = len(self.units)
        self.copies = copies  # False: an app where Ctrl+C copies nothing
        self.insert_copies = insert_copies  # False: an app that copies with Ctrl+C only, not with Ctrl+Insert
        self.select_all = select_all  # False: an app where Ctrl+A selects nothing
        self.log = []  # "wait" (for the modifiers to be let go) and the keys of each SendInput call, in order
        self.sessions, self.is_open, self.puts = 0, False, 0
        self.on_close = {}  # session number -> what happens right after we close the clipboard that time
        self.on_paste = None  # what happens right after the app pastes
        self.pasted = None
        self.held = set()

    # -- the text box

    @property
    def selected(self):
        start, end = sorted((self.anchor, self.caret))
        return "".join(self.units[start:end])

    def select(self, start, end):
        self.caret, self.anchor = start, end  # as Shift+Left leaves it: the caret at the start

    def keys(self, events):
        self.log.append(list(events))
        for vk, up in events:
            if up:
                self.held.discard(vk)
                continue
            self.held.add(vk)
            if vk == VK_LEFT:  # collapses a selection to its start, else moves back (selecting with Shift)
                if VK_SHIFT not in self.held and self.caret != self.anchor:
                    self.caret = self.anchor = min(self.caret, self.anchor)
                    continue
                self.caret = max(0, self.caret - 1)
                if VK_SHIFT not in self.held:
                    self.anchor = self.caret
            elif vk == VK_A and VK_CONTROL in self.held and self.select_all:
                self.anchor, self.caret = 0, len(self.units)
            elif vk == VK_HOME and VK_CONTROL in self.held:
                self.caret = 0
                if VK_SHIFT not in self.held:
                    self.anchor = 0
            elif vk == VK_RIGHT:  # collapses a selection to its end, else moves on
                end = max(self.caret, self.anchor) if self.caret != self.anchor else min(len(self.units), self.caret + 1)
                self.caret = self.anchor = end
            elif (vk == VK_C or vk == VK_INSERT and self.insert_copies) and VK_CONTROL in self.held and self.copies \
                    and self.selected:
                self.copy_by_user({CF_UNICODETEXT: utf16(self.selected.replace("\n", "\r\n"))})  # Windows apps copy \r\n
            elif vk == VK_V and VK_CONTROL in self.held:
                self.pasted = dict(self.clipboard)
                start, end = sorted((self.anchor, self.caret))
                self.units[start:end] = text = list(decode(self.clipboard[CF_UNICODETEXT]).replace("\r\n", "\n"))
                self.caret = self.anchor = start + len(text)
                if self.on_paste:
                    self.on_paste()

    # -- the clipboard

    def copy_by_user(self, items):  # anyone but textaccess: the app or the user
        self.clipboard = dict(items)
        self.seq += 1

    @contextmanager
    def session(self):
        assert not self.is_open
        self.is_open, self.sessions = True, self.sessions + 1
        try:
            yield
        finally:
            self.is_open = False
            if hook := self.on_close.pop(self.sessions, None):
                hook()

    def snapshot(self):
        assert self.is_open
        return list(self.clipboard.items())

    def put(self, items):
        assert self.is_open
        self.clipboard, self.seq, self.puts = dict(items), self.seq + 1, self.puts + 1

    def read_text(self):
        assert self.is_open
        data = self.clipboard.get(CF_UNICODETEXT)
        return decode(data) if data else None


@pytest.fixture(autouse=True)
def no_real_windows_calls(monkeypatch):
    """A test that forgets its fake fails instead of pressing keys or touching the clipboard."""
    def refuse(*args, **kwargs):
        raise AssertionError("a real Windows call in a test")
    for name in ("send_keys", "_clipboard", "_snapshot", "_put", "_read_text", "_sequence", "_wait_for_modifiers_released",
                 "foreground_window", "_is_window", "_is_iconic", "_show_window", "_bring_to_top", "_set_foreground",
                 "_window_thread", "_current_thread", "_attach_input"):
        monkeypatch.setattr(textaccess, name, refuse)


@pytest.fixture
def desktop(monkeypatch):
    def make(text="", **kwargs):
        d = Desktop(text, **kwargs)
        fakes = {"send_keys": d.keys, "_clipboard": d.session, "_snapshot": d.snapshot, "_put": d.put, "_read_text": d.read_text,
                 "_sequence": lambda: d.seq, "_wait_for_modifiers_released": lambda: d.log.append("wait")}
        for name, fake in fakes.items():
            monkeypatch.setattr(textaccess, name, fake)
        monkeypatch.setattr(textaccess, "RESTORE_DELAY", 0)
        monkeypatch.setattr(textaccess, "SETTLE_DELAY", 0)
        return d
    return make


def presses(d):
    return [keys for keys in d.log if keys != "wait"]


def count(d, vk):
    return sum(keys.count((vk, False)) for keys in presses(d))


CTRL_C = [(VK_CONTROL, False), (VK_C, False), (VK_C, True), (VK_CONTROL, True)]
CTRL_INSERT = [(VK_CONTROL, False), (VK_INSERT, False), (VK_INSERT, True), (VK_CONTROL, True)]


# ---------------------------------------------------------------- copy_selection

def test_copy_selection_returns_the_selection_and_puts_the_users_clipboard_back(desktop):
    d = desktop("Dear team, the report is late.")
    d.select(11, 30)
    assert copy_selection() == "the report is late."
    assert d.log == ["wait", CTRL_INSERT]  # the shortcut's modifiers are let go before the copy
    assert d.clipboard == RESTORED
    assert d.selected == "the report is late."  # still selected, ready to be replaced


def test_with_nothing_selected_the_clipboard_is_left_untouched(desktop):
    d = desktop("Dear team")
    assert copy_selection(timeout=0.05) is None
    assert presses(d) == [CTRL_INSERT]
    assert d.clipboard == USERS_COPY and d.puts == 0  # never rewritten, so nothing of the user's copy can get lost


def test_rflow_copies_with_ctrl_insert_so_ctrl_c_tools_stay_quiet(desktop):
    # A translator on the owner's laptop answered two Ctrl+C in a second by putting a translation on the clipboard.
    d = desktop("hello there")
    d.select(0, 11)
    assert copy_selection() == "hello there" and presses(d) == [CTRL_INSERT] and count(d, VK_C) == 0


def test_an_app_without_ctrl_insert_is_copied_with_ctrl_c_when_text_is_known_to_be_selected(desktop):
    d = desktop("hello there", insert_copies=False)
    d.select(0, 11)
    assert copy_selection(timeout=0.05) is None and presses(d) == [CTRL_INSERT]  # not asked to: no Ctrl+C
    d.log.clear()
    assert copy_selection(timeout=0.05, fallback=True) == "hello there" and presses(d) == [CTRL_INSERT, CTRL_C]
    assert d.clipboard == RESTORED


def test_select_last_works_in_an_app_without_ctrl_insert(desktop):
    d = desktop("Notes: hello there ", insert_copies=False)
    assert select_last("hello there ") and d.selected == "hello there "


def test_line_breaks_come_back_as_newlines(desktop):
    d = desktop("one\ntwo")
    d.select(0, 7)
    assert copy_selection() == "one\ntwo"  # the app copied "one\r\ntwo"


def test_a_selection_of_only_whitespace_counts_as_nothing(desktop):
    d = desktop("a \n b")
    d.select(1, 4)
    assert copy_selection() is None
    assert d.clipboard == RESTORED


def test_an_error_while_reading_still_puts_the_clipboard_back(desktop, monkeypatch):
    d = desktop("hello")
    d.select(0, 5)

    def broken():
        raise OSError("GlobalLock failed")
    monkeypatch.setattr(textaccess, "_read_text", broken)
    assert copy_selection() is None
    assert d.clipboard == RESTORED


def test_a_copy_the_user_makes_right_after_our_read_is_kept(desktop):
    d = desktop("hello")
    d.select(0, 5)
    d.on_close[2] = lambda: d.copy_by_user({CF_UNICODETEXT: utf16("copied meanwhile")})  # session 2 reads the copy
    assert copy_selection() == "hello"
    assert d.clipboard == {CF_UNICODETEXT: utf16("copied meanwhile")}


def test_a_busy_clipboard_presses_nothing(desktop, monkeypatch):
    d = desktop("hello")
    d.select(0, 5)

    def busy():
        raise OSError("the clipboard is busy (another app is holding it open)")
    monkeypatch.setattr(textaccess, "_clipboard", busy)
    assert copy_selection() is None
    assert presses(d) == []  # without a copy of the user's clipboard, Ctrl+C could lose it


# ---------------------------------------------------------------- selecting

def test_select_back_holds_shift_once_around_batches_of_50(desktop):
    d = desktop("x" * 200)
    select_back(120)
    pair = [(VK_LEFT, False), (VK_LEFT, True)]
    assert d.log == ["wait", [(VK_SHIFT, False)], pair * 50, pair * 50, pair * 20, [(VK_SHIFT, True)]]
    assert d.selected == "x" * 120


def test_select_back_of_nothing_presses_nothing(desktop):
    d = desktop("x")
    select_back(0)
    assert presses(d) == []


def test_collapse_selection_presses_right_once(desktop):
    d = desktop("hello")
    d.select(1, 5)
    collapse_selection()
    assert presses(d) == [[(VK_RIGHT, False), (VK_RIGHT, True)]]
    assert d.selected == "" and d.caret == 5


def test_select_last_selects_what_was_just_typed(desktop):
    d = desktop("Dear team, Hello world ")
    assert select_last("Hello world ")
    assert d.selected == "Hello world " and count(d, VK_LEFT) == 12
    assert d.clipboard == RESTORED


def test_a_line_break_is_one_step(desktop):
    d = desktop("Notes\nline one\nline two")
    assert select_last("line one\r\nline two")
    assert count(d, VK_LEFT) == 17 and d.selected == "line one\nline two"


def test_a_character_outside_the_bmp_is_one_step(desktop):
    d = desktop("Great job " + chr(0x1F680))
    assert select_last("job " + chr(0x1F680))
    assert count(d, VK_LEFT) == 5


def test_text_the_app_changed_a_little_is_still_found_as_it_is_there(desktop):
    d = desktop("Dear team, Hello, world")  # the app added a comma to what Rflow typed
    assert select_last("Hello world") == "Hello, world" and d.selected == "Hello, world"
    assert d.clipboard == RESTORED


def test_the_owners_case_a_dropped_space_and_full_stop_still_finds_the_dictation(desktop):
    # 2026-10-06: 11 steps back gave 9 characters (an app dropped the trailing space and the full stop), so it failed.
    d = desktop("Hi there. Thank you")
    assert select_last("Thank you. ") == "Thank you" and d.selected == "Thank you"


def test_text_that_isnt_close_is_never_selected(desktop):
    d = desktop("Dear team, see you on Monday")  # the caret moved, or the dictation was deleted
    assert select_last("Hello world ") is None
    assert d.selected == "" and d.caret == len(d.units)  # nothing left selected, the caret back where it was
    assert presses(d)[-1] == [(VK_RIGHT, False), (VK_RIGHT, True)]
    assert d.clipboard == RESTORED


def test_quotes_and_capitals_changed_by_autocorrect_still_match(desktop):
    d = desktop("So Don\u2019t wait")
    assert select_last("don't wait")
    assert d.selected == "Don\u2019t wait"


def test_a_browser_steps_over_a_tamil_letter_with_its_vowel_sign_at_once(desktop):
    d = desktop(units=["வ", "ண", "க்", "க", "ம்"])
    assert select_last(TAMIL)
    assert count(d, VK_LEFT) == 5 and d.selected == TAMIL and count(d, VK_RIGHT) == 0


def test_an_editor_stepping_over_each_code_point_gets_a_second_try(desktop):
    d = desktop(TAMIL)
    assert select_last(TAMIL)
    assert count(d, VK_LEFT) == 5 + 7 and count(d, VK_RIGHT) == 1 and d.selected == TAMIL


def test_when_nothing_gets_copied_there_is_no_second_try(desktop):
    d = desktop(TAMIL, copies=False)
    assert not select_last(TAMIL)
    assert count(d, VK_LEFT) == 5 and count(d, VK_RIGHT) == 1 and d.selected == ""
    assert d.clipboard == USERS_COPY


def test_whitespace_alone_is_not_selected(desktop):
    d = desktop("hello  ")
    assert not select_last(" \r\n ")
    assert presses(d) == []


@pytest.mark.parametrize("text, clusters", [
    ("hello", 5),
    ("a\nb", 3),
    (TAMIL, 5),
    ("e" + chr(0x301), 1),  # e and a combining acute accent
    (chr(0x301) + "a", 2),  # a stray mark at the start is a character of its own
    (chr(0x1F44D) + chr(0x1F3FD), 1),  # thumbs up, medium skin tone
    (chr(0x2764) + chr(0xFE0F), 1),  # heart with the emoji variation selector
    (chr(0x1F1EE) + chr(0x1F1F3) + chr(0x1F1EC) + chr(0x1F1E7), 2),  # two flags
    ("".join(map(chr, (0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467))), 1),  # a family joined with ZWJ
])
def test_clusters(text, clusters):
    assert textaccess._clusters(text) == clusters


# ---------------------------------------------------------------- all the text in the field (nothing selected)

CTRL_A = [(VK_CONTROL, False), (VK_A, False), (VK_A, True), (VK_CONTROL, True)]
CTRL_SHIFT_HOME = [(VK_CONTROL, False), (VK_SHIFT, False), (VK_HOME, False), (VK_HOME, True), (VK_SHIFT, True),
                   (VK_CONTROL, True)]
RIGHT = [(VK_RIGHT, False), (VK_RIGHT, True)]


def test_text_before_caret_reads_it_and_puts_the_caret_back(desktop):
    d = desktop("Hi team, the build is green.")
    d.caret = d.anchor = 8
    assert text_before_caret() == "Hi team,"
    assert presses(d) == [CTRL_SHIFT_HOME, CTRL_INSERT, RIGHT]
    assert d.caret == d.anchor == 8 and d.clipboard == RESTORED


def test_with_the_caret_at_the_start_nothing_more_is_pressed(desktop):
    d = desktop("Hi team")
    d.caret = d.anchor = 0
    assert text_before_caret() is None
    assert presses(d) == [CTRL_SHIFT_HOME, CTRL_INSERT] and d.caret == d.anchor == 0


def test_select_all_selects_and_returns_the_whole_field(desktop):
    d = desktop("Hi team,\nthe build is green.")
    d.caret = d.anchor = 3
    assert select_all() == "Hi team,\nthe build is green."
    assert presses(d) == [CTRL_A, CTRL_INSERT] and d.selected == "".join(d.units) and d.clipboard == RESTORED


def test_select_all_never_presses_ctrl_c(desktop):
    # With no text known to be selected, Ctrl+C could stop a program in a console Rflow doesn't recognise.
    d = desktop("Hi team", insert_copies=False)
    assert select_all() is None and count(d, VK_C) == 0
    d = desktop("")
    assert select_all() is None and count(d, VK_C) == 0 and d.clipboard == USERS_COPY


@pytest.mark.parametrize("caret, steps", [(0, 0), (3, 3), (20, 4), (24, 0)])  # "Hello team. Ship it now." is 24 long
def test_unselect_all_puts_the_caret_back_where_it_was(desktop, caret, steps):
    d = desktop("Hello team. Ship it now.")
    d.caret = d.anchor = caret
    before = text_before_caret()
    whole = select_all()
    d.log.clear()
    unselect_all(before, whole)
    assert d.caret == d.anchor == (caret if before else 24)  # the caret at the start reads as unknown: the end
    assert count(d, VK_LEFT) + count(d, VK_RIGHT) == steps + 1  # the fewest steps, from the start or from the end


def test_unselect_all_with_nothing_known_goes_to_the_end(desktop):
    d = desktop("Hello")
    d.select(0, 5)
    unselect_all(None, None)
    assert presses(d) == [RIGHT] and d.caret == d.anchor == 5


# ---------------------------------------------------------------- paste_rich

def test_paste_rich_pastes_text_and_html_then_puts_the_clipboard_back(desktop):
    d = desktop("Draft: notes")
    d.select(7, 12)
    html = "<h3>Summary</h3><ul><li>one</li></ul>"
    paste_rich("Summary\n- one", html)
    assert d.pasted == {CF_UNICODETEXT: utf16("Summary\r\n- one"), CF_HTML: cf_html(html) + b"\0",
                        CF_EXCLUDE_FROM_HISTORY: b"\0" * 4}
    assert "".join(d.units) == "Draft: Summary\n- one"
    assert d.log[0] == "wait" and presses(d) == [[(VK_CONTROL, False), (VK_V, False), (VK_V, True), (VK_CONTROL, True)]]
    assert d.clipboard == RESTORED


def test_paste_rich_gives_the_app_a_moment_to_see_the_new_clipboard_before_ctrl_v(desktop, monkeypatch):
    # An app that has just copied still answers Ctrl+V with its own copy until it hears the clipboard changed.
    d = desktop("x")
    d.select(0, 1)
    monkeypatch.setattr(textaccess, "SETTLE_DELAY", 0.15)
    monkeypatch.setattr(textaccess.time, "sleep", lambda s: d.log.append(("sleep", s)))
    paste_rich("y")
    assert d.log.index(("sleep", 0.15)) < d.log.index([(VK_CONTROL, False), (VK_V, False), (VK_V, True), (VK_CONTROL, True)])


def test_paste_rich_without_html_pastes_only_text(desktop):
    d = desktop("x")
    d.select(0, 1)
    paste_rich("plain")
    assert d.pasted == {CF_UNICODETEXT: utf16("plain"), CF_EXCLUDE_FROM_HISTORY: b"\0" * 4}
    assert d.clipboard == RESTORED


def test_paste_rich_keeps_a_copy_the_user_made_meanwhile(desktop):
    d = desktop("x")
    d.select(0, 1)
    d.on_paste = lambda: d.copy_by_user({CF_UNICODETEXT: utf16("copied meanwhile")})
    paste_rich("y")
    assert d.clipboard == {CF_UNICODETEXT: utf16("copied meanwhile")}


def test_an_empty_clipboard_comes_back_empty(desktop):
    d = desktop("x")
    d.clipboard = {}
    d.select(0, 1)
    paste_rich("y")
    assert d.clipboard == {}


# ---------------------------------------------------------------- set_clipboard (the user pastes it themselves)

def test_set_clipboard_leaves_text_and_html_for_the_user(desktop):
    d = desktop("x")
    html = "<p>one</p><p>two</p>"
    set_clipboard("one\ntwo", html)
    # Not put back, and without the marker that keeps a copy out of Win+V history: the user wants this one.
    assert d.clipboard == {CF_UNICODETEXT: utf16("one\r\ntwo"), CF_HTML: cf_html(html) + b"\0"}
    assert d.sessions == 1 and d.log == []  # no key pressed, nothing waited for
    assert "".join(d.units) == "x"


@pytest.mark.parametrize("use", [lambda: copy_selection(timeout=0.05), lambda: paste_rich("new"), lambda: set_clipboard("new")])
def test_the_clipboard_waits_for_a_dictations_restore_first(desktop, use):
    """A dictation's paste puts the user's clipboard back in the background, 0.5 s later: Text Transform copying or
    pasting meanwhile would save the dictation as the user's clipboard, or be overwritten by the restore."""
    import threading
    import time

    from sst import paste
    d = desktop("x")
    released = []
    paste._pasting.acquire()  # a paste whose restore is still to come
    threading.Timer(0.2, lambda: (released.append(time.monotonic()), paste._pasting.release())).start()
    use()
    assert released and d.sessions >= 1 and paste.settle(0)  # it went on only after the restore, which it left free


def test_set_clipboard_without_html_puts_only_text(desktop):
    d = desktop()
    set_clipboard("plain\r\ntext")
    assert d.clipboard == {CF_UNICODETEXT: utf16("plain\r\ntext")} and d.puts == 1


# ---------------------------------------------------------------- activate (back to the window the text came from)

ME, EDITOR, BROWSER, PANEL = 7, 0x1001, 0x2002, 0x3003  # our thread; the editor, a browser in front, a window of ours


class Windows:
    """The top-level windows as activate sees them. As in Windows, the foreground passes on only from a thread joined
    to the input of the window in front."""

    def __init__(self, front=BROWSER, minimised=(), refuses=False, gone=(), threads=None):
        self.front, self.minimised, self.refuses, self.gone = front, set(minimised), refuses, set(gone)
        self.threads = threads or {EDITOR: 11, BROWSER: 22, PANEL: ME}
        self.joined, self.calls = set(), []

    def attach(self, thread, to, on):
        assert thread == ME and to != ME  # Windows refuses to join a thread to itself
        self.calls.append(("attach" if on else "detach", to))
        (self.joined.add if on else self.joined.discard)(to)
        return True

    def show(self, hwnd, command):
        self.calls.append(("show", hwnd, command))
        self.minimised.discard(hwnd)

    def set_foreground(self, hwnd):
        self.calls.append(("foreground", hwnd))
        if not self.refuses and self.threads.get(self.front) in self.joined | {ME}:
            self.front = hwnd


@pytest.fixture
def windows(monkeypatch):
    def make(**kwargs):
        w = Windows(**kwargs)
        fakes = {"foreground_window": lambda: w.front, "_is_window": lambda h: h in w.threads and h not in w.gone,
                 "_is_iconic": lambda h: h in w.minimised, "_show_window": w.show, "_set_foreground": w.set_foreground,
                 "_bring_to_top": lambda h: w.calls.append(("top", h)), "_window_thread": lambda h: w.threads.get(h, 0),
                 "_current_thread": lambda: ME, "_attach_input": w.attach}
        for name, fake in fakes.items():
            monkeypatch.setattr(textaccess, name, fake)
        return w
    return make


def test_activate_joins_the_input_for_the_switch_and_leaves_again(windows):
    w = windows()
    assert activate(EDITOR)
    assert w.front == EDITOR and not w.joined
    assert w.calls == [("attach", 22), ("attach", 11), ("top", EDITOR), ("foreground", EDITOR), ("detach", 22), ("detach", 11)]


def test_a_minimised_window_is_restored_first(windows):
    w = windows(minimised={EDITOR})
    assert activate(EDITOR)
    assert w.calls[0] == ("show", EDITOR, textaccess.SW_RESTORE) and w.front == EDITOR and not w.minimised


def test_a_window_already_in_front_is_left_alone(windows):
    w = windows(front=EDITOR)
    assert activate(EDITOR) and w.calls == []


def test_a_closed_window_cannot_be_activated(windows):
    w = windows(gone={EDITOR})
    assert not activate(EDITOR) and not activate(0)
    assert w.calls == [] and w.front == BROWSER


def test_each_other_thread_is_joined_once_and_ours_never(windows):
    w = windows(front=PANEL)  # Rflow's own window is in front
    assert activate(EDITOR)
    assert [call for call in w.calls if call[0] in ("attach", "detach")] == [("attach", 11), ("detach", 11)]
    w = windows(threads={EDITOR: 11, BROWSER: 11})  # another window of the same app
    assert activate(EDITOR)
    assert [call for call in w.calls if call[0] in ("attach", "detach")] == [("attach", 11), ("detach", 11)]


def test_when_windows_keeps_another_window_in_front_activate_says_so(windows):
    w = windows(refuses=True)
    assert not activate(EDITOR, timeout=0.05)
    assert w.front == BROWSER and not w.joined


def test_the_input_is_always_detached(windows, monkeypatch):
    w = windows()

    def broken(hwnd):
        raise OSError("SetForegroundWindow failed")
    monkeypatch.setattr(textaccess, "_set_foreground", broken)
    with pytest.raises(OSError):
        activate(EDITOR)
    assert not w.joined and w.calls[-2:] == [("detach", 22), ("detach", 11)]


# ---------------------------------------------------------------- the HTML clipboard format

OFFSETS = ("StartHTML", "EndHTML", "StartFragment", "EndFragment")


def header(data):
    return dict(line.decode("ascii").split(":", 1) for line in data.split(b"\r\n")[:5])


def test_cf_html_header_for_a_small_fragment():
    data = cf_html("<b>hi</b>")
    assert data == (b"Version:0.9\r\nStartHTML:0000000105\r\nEndHTML:0000000182\r\nStartFragment:0000000139\r\n"
                    b"EndFragment:0000000148\r\n<html><body>\r\n<!--StartFragment--><b>hi</b><!--EndFragment-->\r\n"
                    b"</body></html>")


@pytest.mark.parametrize("fragment", [
    "<p>Hello <b>world</b></p>",
    "<h3>வணக்கம்</h3><ul><li>நன்றி</li></ul>",
    "<ul><li>Ship it " + chr(0x1F680) + "</li><li>" + chr(0x1F44D) + chr(0x1F3FD) + " done</li></ul>",
])
def test_cf_html_offsets_are_utf8_bytes(fragment):
    data = cf_html(fragment)
    fields = header(data)
    assert fields["Version"] == "0.9" and all(len(fields[name]) == 10 for name in OFFSETS)
    start, end, fragment_start, fragment_end = (int(fields[name]) for name in OFFSETS)
    assert data[:start].endswith(b"EndFragment:%010d\r\n" % fragment_end)  # the document starts right after the header
    assert end == len(data) and data[start:end].startswith(b"<html><body>") and data.endswith(b"</body></html>")
    assert data[fragment_start:fragment_end].decode("utf-8") == fragment
    assert data[:fragment_start].endswith(b"<!--StartFragment-->") and data[fragment_end:].startswith(b"<!--EndFragment-->")
    assert fragment_end - fragment_start == len(fragment.encode("utf-8"))


# ---------------------------------------------------------------- reading what the user copied (Translate's Ctrl+C+C)

def test_clipboard_text_reads_the_users_copy_without_pressing_anything(desktop):
    d = desktop("x")
    d.copy_by_user({CF_UNICODETEXT: utf16("line one\r\nline two")})
    assert textaccess.clipboard_text() == "line one\nline two" and presses(d) == []
    assert textaccess.clipboard_sequence() == d.seq


def test_clipboard_text_of_nothing_or_whitespace_is_none(desktop, monkeypatch):
    d = desktop("x")
    d.copy_by_user({CF_UNICODETEXT: utf16("  \r\n ")})
    assert textaccess.clipboard_text() is None

    def busy():
        raise OSError("the clipboard is busy (another app is holding it open)")
    monkeypatch.setattr(textaccess, "_clipboard", busy)
    assert textaccess.clipboard_text() is None
