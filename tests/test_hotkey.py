import ctypes
import struct

import pytest

from sst.hotkey import (
    INPUT,
    KEYEVENTF_EXTENDEDKEY,
    KEYEVENTF_KEYUP,
    OUR_INPUT,
    VK_ESCAPE,
    Hotkey,
    HotkeyListener,
    Matcher,
    key_input,
    parse_hotkey,
)

LCTRL, RCTRL, LWIN, RWIN, LALT, LSHIFT, MENU, D, LEFT, SPACE = 0xA2, 0xA3, 0x5B, 0x5C, 0xA4, 0xA0, 0x5D, 0x44, 0x25, 0x20
ONE, NUM1, UP, DOWN, ENTER, U, T, C, RSHIFT = 0x31, 0x61, 0x26, 0x28, 0x0D, 0x55, 0x54, 0x43, 0xA1
# What Text Transform's menu takes: 1-9 on both rows, Up, Down, Enter, Esc and U (undo).
MENU_KEYS = frozenset({*range(0x31, 0x3A), *range(0x61, 0x6A), UP, DOWN, ENTER, VK_ESCAPE, U})


def feed(matcher, *steps):
    """steps like ('down', LCTRL); returns the list of (hidden, event) results."""
    return [matcher.feed(vk, action == "down") for action, vk in steps]


# ---------------------------------------------------------------- parsing

@pytest.mark.parametrize("text, modifiers, key", [
    ("ctrl+win", {"ctrl", "win"}, None),
    ("Win + Ctrl", {"ctrl", "win"}, None),
    ("menu", set(), 0x5D),
    ("ctrl+alt+d", {"ctrl", "alt"}, ord("D")),
    ("control+shift+space", {"ctrl", "shift"}, 0x20),
    ("f9", set(), 0x78),
    ("muhenkan", set(), 0x1D),
])
def test_parse_hotkey(text, modifiers, key):
    hotkey = parse_hotkey(text)
    assert hotkey.modifiers == modifiers and hotkey.key == key


@pytest.mark.parametrize("text", ["ctrl+hyper+x", "ctrl+alt+", "d+ctrl", "", "esc"])
def test_parse_hotkey_rejects_unknown_keys(text):
    with pytest.raises(ValueError, match="unknown key"):
        parse_hotkey(text)


@pytest.mark.parametrize("text", ["ctrl", "win"])
def test_a_single_modifier_is_refused(text):
    with pytest.raises(ValueError, match="normal typing"):
        parse_hotkey(text)


def test_labels():
    assert parse_hotkey("ctrl+win").label == "Ctrl+Win"
    assert parse_hotkey("menu").label == "Menu key"
    assert parse_hotkey("double ctrl").label == "Double-tap Ctrl"
    assert parse_hotkey("double shift").label == "Double-tap Shift"


@pytest.mark.parametrize("text, modifier", [
    ("double ctrl", "ctrl"),
    ("Double Shift", "shift"),
    ("double-ctrl", "ctrl"),
    ("  DOUBLE   control ", "ctrl"),
])
def test_parse_double_tap(text, modifier):
    hotkey = parse_hotkey(text)
    assert hotkey == Hotkey(f"double {modifier}", frozenset({modifier}), None, double=True)


@pytest.mark.parametrize("text", ["double alt", "double win", "double d", "double ctrl+alt", "double-"])
def test_only_ctrl_and_shift_can_be_double_tapped(text):
    with pytest.raises(ValueError):
        parse_hotkey(text)


def test_other_hotkeys_are_not_double_taps():
    assert not parse_hotkey("ctrl+win").double and not parse_hotkey("menu").double
    assert Hotkey("ctrl+win", frozenset({"ctrl", "win"}), None).double is False  # positional construction still works


def test_input_struct_has_the_size_sendinput_expects():
    # SendInput silently does nothing when cbSize is wrong: 40 bytes on 64-bit Windows, 28 on 32-bit.
    assert ctypes.sizeof(INPUT) == (40 if struct.calcsize("P") == 8 else 28)


# ---------------------------------------------------------------- Ctrl+Win (modifiers only)

def test_ctrl_win_press_and_release_pass_through():
    m = Matcher(parse_hotkey("ctrl+win"))
    assert feed(m, ("down", LCTRL), ("down", LWIN), ("up", LWIN), ("up", LCTRL)) == [
        (False, None), (False, "press"), (False, "release"), (False, None)]


def test_ctrl_win_in_either_order_and_either_side():
    m = Matcher(parse_hotkey("ctrl+win"))
    assert feed(m, ("down", RWIN), ("down", RCTRL))[-1] == (False, "press")
    assert feed(m, ("up", RCTRL))[-1] == (False, "release")


def test_auto_repeat_while_holding_does_not_press_again():
    m = Matcher(parse_hotkey("ctrl+win"))
    events = [event for _, event in feed(m, ("down", LCTRL), ("down", LWIN), ("down", LWIN), ("down", LCTRL),
                                          ("down", LWIN), ("up", LWIN))]
    assert events == [None, "press", None, None, None, "release"]


def test_windows_shortcuts_still_work_and_interrupt_dictation():
    m = Matcher(parse_hotkey("ctrl+win"))
    results = feed(m, ("down", LCTRL), ("down", LWIN), ("down", D), ("up", D), ("up", LWIN), ("up", LCTRL))
    assert results == [(False, None), (False, "press"), (False, "interrupt"), (False, None), (False, "release"),
                       (False, None)]  # D reaches Windows (new desktop); the release still comes: the dictation decides


def test_ctrl_win_space_is_hands_free_and_windows_never_gets_the_space():
    # Wispr Flow's hands-free chord; on the dev laptop the Copilot key is remapped to it with PowerToys.
    m = Matcher(parse_hotkey("ctrl+win"))
    assert feed(m, ("down", LCTRL), ("down", LWIN), ("down", SPACE), ("down", SPACE), ("up", SPACE), ("up", LWIN),
                ("up", LCTRL)) == [(False, None), (False, "press"), (True, "handsfree"), (True, None), (True, None),
                                   (False, None), (False, None)]


def test_space_is_typed_normally_when_ctrl_win_is_not_held():
    m = Matcher(parse_hotkey("ctrl+win"))
    assert feed(m, ("down", SPACE), ("up", SPACE)) == [(False, None), (False, None)]


def test_ctrl_win_with_another_modifier_is_not_the_hotkey():
    m = Matcher(parse_hotkey("ctrl+win"))
    assert [e for _, e in feed(m, ("down", LSHIFT), ("down", LCTRL), ("down", LWIN))] == [None, None, None]


def test_ctrl_win_while_a_letter_is_held_is_not_the_hotkey():
    m = Matcher(parse_hotkey("ctrl+win"))
    assert [e for _, e in feed(m, ("down", LEFT), ("down", LCTRL), ("down", LWIN))] == [None, None, None]


def test_tap_again_after_release_presses_again():
    m = Matcher(parse_hotkey("ctrl+win"))
    feed(m, ("down", LCTRL), ("down", LWIN), ("up", LWIN), ("up", LCTRL))
    assert feed(m, ("down", LCTRL), ("down", LWIN))[-1] == (False, "press")


# ---------------------------------------------------------------- the Menu key

def test_menu_key_is_hidden_from_apps():
    m = Matcher(parse_hotkey("menu"))
    assert feed(m, ("down", MENU), ("down", MENU), ("up", MENU)) == [(True, "press"), (True, None), (True, "release")]


def test_menu_key_with_a_modifier_is_left_alone():
    m = Matcher(parse_hotkey("menu"))
    assert feed(m, ("down", LSHIFT), ("down", MENU), ("up", MENU)) == [(False, None), (False, None), (False, None)]


# ---------------------------------------------------------------- modifiers + key

def test_ctrl_alt_d_hides_only_the_d():
    m = Matcher(parse_hotkey("ctrl+alt+d"))
    assert feed(m, ("down", LCTRL), ("down", LALT), ("down", D), ("up", D), ("up", LALT), ("up", LCTRL)) == [
        (False, None), (False, None), (True, "press"), (True, "release"), (False, None), (False, None)]


def test_plain_d_is_typed_normally():
    m = Matcher(parse_hotkey("ctrl+alt+d"))
    assert feed(m, ("down", D), ("up", D)) == [(False, None), (False, None)]


# ---------------------------------------------------------------- Esc

def test_esc_cancels_only_while_recording():
    m = Matcher(parse_hotkey("ctrl+win"))
    assert feed(m, ("down", VK_ESCAPE), ("up", VK_ESCAPE)) == [(False, None), (False, None)]
    m.recording = True
    assert feed(m, ("down", VK_ESCAPE), ("down", VK_ESCAPE), ("up", VK_ESCAPE)) == [
        (True, "cancel"), (True, None), (True, None)]


# ---------------------------------------------------------------- missed key releases

def test_keys_released_while_the_hook_was_blind_are_forgotten():
    # Win+L: the hook sees Win go down but not up (the lock screen takes over).
    held = {LWIN}
    m = Matcher(parse_hotkey("ctrl+win"), is_held=lambda vk: vk in held)
    feed(m, ("down", LWIN))
    held.clear()  # back from the lock screen, Win is no longer down
    assert feed(m, ("down", LCTRL))[-1] == (False, None)  # Ctrl alone must not look like Ctrl+Win


# ---------------------------------------------------------------- capture (a menu that leaves the focus with the app)

def capturing(hotkey="ctrl+alt+t", **kwargs):
    m = Matcher(parse_hotkey(hotkey), **kwargs)
    m.capture = MENU_KEYS
    return m


def test_nothing_is_captured_by_default():
    m = Matcher(parse_hotkey("ctrl+alt+t"))
    assert feed(m, ("down", ONE), ("up", ONE), ("down", ENTER), ("up", ENTER)) == [(False, None)] * 4


def test_captured_presses_are_hidden_and_reported_and_their_releases_hidden():
    m = capturing()
    assert feed(m, ("down", ONE), ("up", ONE), ("down", NUM1), ("up", NUM1), ("down", ENTER), ("up", ENTER),
                ("down", VK_ESCAPE), ("up", VK_ESCAPE), ("down", U), ("up", U)) == [
        (True, "key:49"), (True, None), (True, "key:97"), (True, None), (True, "key:13"), (True, None),
        (True, "key:27"), (True, None), (True, "key:85"), (True, None)]


def test_holding_a_captured_arrow_repeats_it():
    m = capturing()
    assert feed(m, ("down", DOWN), ("down", DOWN), ("down", DOWN), ("up", DOWN)) == [
        (True, "key:40"), (True, "key:40"), (True, "key:40"), (True, None)]


def test_keys_outside_the_capture_reach_the_app():
    m = capturing()
    assert feed(m, ("down", D), ("up", D), ("down", LEFT), ("up", LEFT), ("down", SPACE), ("up", SPACE)) == [(False, None)] * 6


def test_a_key_held_when_the_capture_starts_is_left_to_the_app():
    # Its press reached the app; hiding its repeats or release would leave it stuck down there.
    m = Matcher(parse_hotkey("ctrl+alt+t"))
    assert feed(m, ("down", ENTER)) == [(False, None)]
    m.capture = MENU_KEYS
    assert feed(m, ("down", ENTER), ("up", ENTER)) == [(False, None), (False, None)]
    assert feed(m, ("down", ENTER), ("up", ENTER)) == [(True, "key:13"), (True, None)]  # the next press is the menu's


def test_a_captured_key_still_held_when_the_capture_ends_stays_hidden_without_events():
    # The app never saw it go down, so it mustn't see its repeats or release either; the menu has closed.
    m = capturing()
    assert feed(m, ("down", ENTER)) == [(True, "key:13")]
    m.capture = frozenset()
    assert feed(m, ("down", ENTER), ("up", ENTER)) == [(True, None), (True, None)]
    assert feed(m, ("down", ENTER), ("up", ENTER)) == [(False, None), (False, None)]  # typed normally again


def test_a_captured_key_released_while_the_hook_was_blind_is_forgotten():
    held = {ENTER}
    m = capturing(is_held=lambda vk: vk in held)
    assert feed(m, ("down", ENTER)) == [(True, "key:13")]
    held.clear()  # its release went missing (lock screen, admin window)
    m.capture = frozenset()
    assert feed(m, ("down", D), ("up", D), ("down", ENTER), ("up", ENTER)) == [(False, None)] * 4


def test_the_hotkey_still_works_while_capturing():
    m = capturing("ctrl+alt+t")
    assert feed(m, ("down", LCTRL), ("down", LALT), ("down", T), ("up", T), ("up", LALT), ("up", LCTRL)) == [
        (False, None), (False, None), (True, "press"), (True, "release"), (False, None), (False, None)]


def test_the_hotkey_wins_when_its_key_is_captured_too():
    m = capturing("ctrl+alt+1")
    assert feed(m, ("down", LCTRL), ("down", LALT), ("down", ONE), ("up", ONE), ("up", LALT), ("up", LCTRL)) == [
        (False, None), (False, None), (True, "press"), (True, "release"), (False, None), (False, None)]
    assert feed(m, ("down", ONE), ("up", ONE)) == [(True, "key:49"), (True, None)]  # plain 1 is the menu's


def test_ctrl_win_dictation_still_works_while_capturing():
    m = capturing("ctrl+win")
    assert feed(m, ("down", LCTRL), ("down", LWIN), ("up", LWIN), ("up", LCTRL)) == [
        (False, None), (False, "press"), (False, "release"), (False, None)]


def test_esc_cancels_a_recording_even_while_the_menu_takes_esc():
    m = capturing("ctrl+win")
    m.recording = True
    assert feed(m, ("down", VK_ESCAPE), ("down", VK_ESCAPE), ("up", VK_ESCAPE)) == [
        (True, "cancel"), (True, None), (True, None)]
    m.recording = False
    assert feed(m, ("down", VK_ESCAPE), ("up", VK_ESCAPE)) == [(True, "key:27"), (True, None)]


def test_capturing_other_keys_leaves_esc_as_it_was():
    m = Matcher(parse_hotkey("ctrl+win"))
    m.capture = frozenset({ONE, ENTER})
    assert feed(m, ("down", VK_ESCAPE), ("up", VK_ESCAPE)) == [(False, None), (False, None)]
    m.recording = True
    assert feed(m, ("down", VK_ESCAPE), ("up", VK_ESCAPE)) == [(True, "cancel"), (True, None)]


def test_the_listener_hands_the_capture_to_its_matcher():
    listener = HotkeyListener(parse_hotkey("ctrl+alt+t"))  # not started: no hook is installed
    listener.capture({ONE, ENTER})
    assert listener._matcher.capture == frozenset({ONE, ENTER})
    listener.capture(None)
    assert listener._matcher.capture == frozenset()


# ---------------------------------------------------------------- double tap (Text Transform's double ctrl)

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def tapping(hotkey="double ctrl", **kwargs):
    clock = Clock()
    return Matcher(parse_hotkey(hotkey), clock=clock, **kwargs), clock


def play(m, clock, *steps):
    """steps like ('down', LCTRL), or a number: that many seconds pass. Returns the (hidden, event) results."""
    results = []
    for step in steps:
        if isinstance(step, tuple):
            results.append(m.feed(step[1], step[0] == "down"))
        else:
            clock.now += step
    return results


def tap(vk=LCTRL, hold=0.1):
    return ("down", vk), hold, ("up", vk)


def events(results):
    return [event for _, event in results if event]


def test_a_double_tap_fires_once_on_the_second_release_and_hides_nothing():
    m, clock = tapping()
    assert play(m, clock, *tap(), 0.2, *tap()) == [(False, None), (False, None), (False, None), (False, "release")]


def test_slow_taps_are_not_a_double_tap():
    m, clock = tapping()
    assert events(play(m, clock, *tap(), 0.45, *tap())) == []
    assert events(play(m, clock, 0.1, *tap())) == ["release"]  # the slow second tap was the first of a new pair


def test_a_long_hold_is_not_a_tap():
    m, clock = tapping()
    assert events(play(m, clock, *tap(hold=0.4), 0.1, *tap())) == []
    m, clock = tapping()
    assert events(play(m, clock, *tap(), 0.1, *tap(hold=0.4))) == []


def test_a_shortcut_then_ctrl_is_not_a_double_tap():
    m, clock = tapping()
    copy = [("down", LCTRL), 0.05, ("down", C), ("up", C), 0.05, ("up", LCTRL)]
    assert events(play(m, clock, *copy, 0.1, *tap())) == []
    assert events(play(m, clock, 0.5, *tap(), 0.1, *copy)) == []  # Ctrl+C right after a tap neither


def test_another_modifier_in_between_breaks_the_double_tap():
    m, clock = tapping()
    assert events(play(m, clock, *tap(), 0.05, *tap(LSHIFT), 0.05, *tap())) == []
    assert events(play(m, clock, 0.5, ("down", LCTRL), ("down", RCTRL), ("up", RCTRL), ("up", LCTRL), 0.1, *tap())) == []


def test_a_key_held_meanwhile_breaks_the_double_tap():
    m, clock = tapping()
    assert events(play(m, clock, ("down", D), *tap(), 0.1, *tap(), ("up", D))) == []


def test_a_triple_tap_fires_once_and_a_fourth_tap_starts_a_new_pair():
    m, clock = tapping()
    assert events(play(m, clock, *tap(), 0.1, *tap(), 0.1, *tap())) == ["release"]
    assert events(play(m, clock, 0.1, *tap())) == ["release"]


def test_left_and_right_ctrl_mix():
    m, clock = tapping()
    assert events(play(m, clock, *tap(LCTRL), 0.1, *tap(RCTRL))) == ["release"]


def test_double_shift():
    m, clock = tapping("double shift")
    assert events(play(m, clock, *tap(LSHIFT), 0.1, *tap(RSHIFT))) == ["release"]
    assert events(play(m, clock, 0.5, *tap(LCTRL), 0.1, *tap(LCTRL))) == []


def test_ctrl_win_dictation_never_fires_a_double_ctrl():
    m, clock = tapping()
    ctrl_win = [("down", LCTRL), ("down", LWIN), 0.1, ("up", LWIN), ("up", LCTRL)]
    results = play(m, clock, *ctrl_win, 0.1, *ctrl_win, 0.1, *tap(), 0.1, *ctrl_win)
    assert events(results) == [] and not any(hidden for hidden, _ in results)


def test_repeats_while_held_are_not_taps():
    m, clock = tapping()
    assert events(play(m, clock, ("down", LCTRL), 0.05, ("down", LCTRL), 0.05, ("down", LCTRL), ("up", LCTRL),
                       0.1, *tap())) == []
    assert events(play(m, clock, 0.5, *tap(), 0.1, ("down", LCTRL), 0.05, ("down", LCTRL), 0.05, ("up", LCTRL))) == []


def test_the_menu_opened_by_a_double_tap_captures_keys():
    m, clock = tapping()
    assert events(play(m, clock, *tap(), 0.1, *tap())) == ["release"]
    m.capture = MENU_KEYS  # Text Transform's menu is open
    assert play(m, clock, 0.5, ("down", DOWN), ("up", DOWN), ("down", ONE), ("up", ONE)) == [
        (True, "key:40"), (True, None), (True, "key:49"), (True, None)]
    assert play(m, clock, ("down", VK_ESCAPE), ("up", VK_ESCAPE)) == [(True, "key:27"), (True, None)]
    assert events(play(m, clock, 0.5, *tap(), 0.1, *tap())) == ["release"]  # the double tap again closes the menu


def test_a_captured_key_breaks_the_double_tap():
    m, clock = tapping()
    m.capture = MENU_KEYS
    assert events(play(m, clock, *tap(), 0.05, ("down", ONE), ("up", ONE), 0.05, *tap())) == ["key:49"]


def test_a_double_tap_needs_no_mask():
    # Ctrl or Shift tapped alone opens no menu, so the listener presses nothing of its own.
    assert not HotkeyListener(parse_hotkey("double ctrl"))._mask


@pytest.mark.parametrize("vk, extended", [(0x25, True), (0x27, True), (0x2D, True), (0x24, True), (0x2E, True),
                                          (0x11, False), (0x10, False), (0x43, False), (0x56, False), (0xE8, False)])
def test_navigation_keys_are_sent_as_extended_keys(vk, extended):
    # Without the flag, Left is the number pad's: with NumLock on, Shift+Left then moved the caret instead of selecting.
    event = key_input(vk, up=False)
    assert bool(event.ki.dwFlags & KEYEVENTF_EXTENDEDKEY) == extended and event.ki.wVk == vk
    assert key_input(vk, up=True).ki.dwFlags & KEYEVENTF_KEYUP and event.ki.dwExtraInfo == OUR_INPUT


def test_ctrl_clicks_that_move_the_pointer_are_not_a_double_tap():
    at = [(100, 100)]
    m, clock = tapping(pointer=lambda: at[0])
    first = play(m, clock, *tap(), 0.1)
    at[0] = (100, 140)  # the mouse went to the next file and Ctrl+clicked it
    assert events(first + play(m, clock, *tap())) == []
    assert events(play(m, clock, 0.1, *tap())) == ["release"]  # still there: the last tap and this one are a double tap
    first = play(m, clock, 0.5, *tap(), 0.1)
    at[0] = (108, 146)  # a hand resting on the mouse nudges it a little: still a double tap
    assert events(first + play(m, clock, *tap())) == ["release"]


# ---------------------------------------------------------------- ctrl+c+c (Translate)

def test_parse_ctrl_c_c():
    hotkey = parse_hotkey("ctrl+c+c")
    assert hotkey.double and hotkey.key == C and hotkey.modifiers == {"ctrl"} and hotkey.label == "Ctrl+C+C"
    with pytest.raises(ValueError):
        parse_hotkey("c+c")  # without a modifier it would fire while typing "cc"


def press_c():
    return ("down", C), 0.05, ("up", C)


def test_ctrl_held_and_c_pressed_twice_fires_and_hides_nothing():
    m, clock = tapping("ctrl+c+c")
    results = play(m, clock, ("down", LCTRL), *press_c(), 0.1, *press_c(), ("up", LCTRL))
    assert events(results) == ["release"] and not any(hidden for hidden, _ in results)  # the app still copies


def test_ctrl_c_twice_with_ctrl_let_go_in_between_fires():
    m, clock = tapping("ctrl+c+c")
    assert events(play(m, clock, ("down", LCTRL), *press_c(), ("up", LCTRL), 0.1, ("down", RCTRL), *press_c(),
                       ("up", RCTRL))) == ["release"]


def test_slow_copies_shortcuts_and_held_keys_are_not_ctrl_c_c():
    m, clock = tapping("ctrl+c+c")
    assert events(play(m, clock, ("down", LCTRL), *press_c(), 0.6, *press_c(), ("up", LCTRL))) == []  # too slow
    assert events(play(m, clock, 1.0, ("down", LCTRL), *press_c(), ("down", 0x56), ("up", 0x56), *press_c(),
                       ("up", LCTRL))) == []  # Ctrl+C, Ctrl+V, Ctrl+C: a paste in between
    assert events(play(m, clock, 1.0, ("down", LCTRL), ("down", C), ("down", C), ("down", C), ("up", C),
                       ("up", LCTRL))) == []  # C held down: repeats aren't presses
    assert events(play(m, clock, 1.0, *press_c(), 0.1, *press_c())) == []  # "cc" typed, no Ctrl
    assert events(play(m, clock, 1.0, ("down", LCTRL), ("down", LSHIFT), *press_c(), *press_c(), ("up", LSHIFT),
                       ("up", LCTRL))) == []  # Ctrl+Shift+C is another shortcut


def test_a_third_copy_starts_a_new_pair():
    m, clock = tapping("ctrl+c+c")
    assert events(play(m, clock, ("down", LCTRL), *press_c(), *press_c(), *press_c(), *press_c(),
                       ("up", LCTRL))) == ["release", "release"]


def test_the_hook_is_renewed_while_no_key_is_held(monkeypatch):
    """Windows silently drops a hook that answers too slowly: a fresh one goes in now and then, never mid-chord."""
    from sst import hotkey
    from sst.hotkey import WM_TIMER, HotkeyListener

    listener = HotkeyListener(parse_hotkey("ctrl+win"))
    shared = hotkey.KeyboardHook()
    shared.listeners = (HotkeyListener(parse_hotkey("double ctrl")), listener)  # any listener mid-chord holds it back
    hooks, unhooked = iter(range(1, 10)), []

    def held(msg):
        listener._matcher._down = {LCTRL}  # Ctrl is down: not now
        msg.message = WM_TIMER

    def released(msg):
        listener._matcher._down = set()
        msg.message = WM_TIMER
    steps = iter([lambda msg: setattr(msg, "message", WM_TIMER), held, released])

    class User32:
        SetWindowsHookExW = staticmethod(lambda *args: next(hooks))
        UnhookWindowsHookEx = staticmethod(unhooked.append)
        SetTimer = staticmethod(lambda *args: 7)
        KillTimer = staticmethod(lambda *args: None)

        @staticmethod
        def GetMessageW(ref, *args):
            step = next(steps, None)
            if step is None:
                return 0
            step(ref._obj)
            return 1

    class Kernel32:
        GetCurrentThreadId = staticmethod(lambda: 1)
        GetModuleHandleW = staticmethod(lambda name: 0)

    monkeypatch.setattr(hotkey, "user32", User32)
    monkeypatch.setattr(hotkey, "kernel32", Kernel32)
    shared._run()
    assert unhooked == [1, 2, 3]  # hook 1 replaced by 2 (idle), kept while Ctrl was held, then by 3; 3 removed at the end


# ---------------------------------------------------------------- one hook for every shortcut

class FakeWindows:
    """user32 and kernel32 as the shared hook uses them: keys held as the test says, no real hook, no key sent."""

    def __init__(self):
        import threading
        self.held, self.hooks, self.unhooked, self.sent, self.calls = set(), 0, [], [], 0
        self.quit = threading.Event()

    def SetWindowsHookExW(self, *args):
        self.hooks += 1
        return self.hooks

    def UnhookWindowsHookEx(self, hook):
        self.unhooked.append(hook)

    def SetTimer(self, *args):
        return 0

    def KillTimer(self, *args):
        pass

    def GetMessageW(self, *args):
        self.quit.wait(5)  # until the last listener stops
        return 0

    def PostThreadMessageW(self, *args):
        self.quit.set()

    def GetAsyncKeyState(self, vk):
        return -0x8000 if vk in self.held else 0

    def GetCursorPos(self, point):
        pass

    def MapVirtualKeyW(self, vk, kind):
        return 0

    def SendInput(self, count, inputs, size):
        self.sent.extend(inputs[i].ki.wVk for i in range(count))

    def CallNextHookEx(self, *args):
        self.calls += 1
        return 0

    @staticmethod
    def GetCurrentThreadId():
        return 1

    @staticmethod
    def GetModuleHandleW(name):
        return 0


@pytest.fixture
def windows(monkeypatch):
    from sst import hotkey
    fake = FakeWindows()
    monkeypatch.setattr(hotkey, "user32", fake)
    monkeypatch.setattr(hotkey, "kernel32", fake)
    monkeypatch.setattr(hotkey, "HOOK", hotkey.KeyboardHook())
    return fake


def _shortcuts():
    """Dictation, Text Transform, Translate and live translation, as the app starts them; each counts its wake-ups."""
    listeners = {name: HotkeyListener(parse_hotkey(text)) for name, text in
                 (("dictation", "ctrl+win"), ("transform", "double ctrl"), ("translate", "ctrl+c+c"), ("live", "ctrl+alt+l"))}
    woken = {name: 0 for name in listeners}
    for name, listener in listeners.items():
        listener.on_event = lambda name=name: woken.__setitem__(name, woken[name] + 1)
    return listeners, woken


def _type(windows, *steps):
    """Physical key events through the shared hook's procedure, as Windows would call it; returns what it answered."""
    import threading

    from sst import hotkey
    hotkey.HOOK._ident = threading.get_ident()  # called on the hook's own thread
    answers = []
    for action, vk in steps:
        (windows.held.add if action == "down" else windows.held.discard)(vk)
        info = hotkey.KBDLLHOOKSTRUCT(vkCode=vk, scanCode=0, flags=0, time=0, dwExtraInfo=0)
        answers.append(hotkey.HOOK._on_key(hotkey.HC_ACTION, hotkey.WM_KEYDOWN if action == "down" else 0x0101,
                                           ctypes.addressof(info)))
    return answers


def _drain(listener):
    out = []
    while not listener.events.empty():
        out.append(listener.events.get_nowait()[0])
    return out


def test_every_shortcut_shares_one_keyboard_hook(windows):
    from sst import hotkey
    listeners, _ = _shortcuts()
    for listener in listeners.values():
        listener.start()
    assert windows.hooks == 1 and hotkey.HOOK.listeners == tuple(listeners.values())  # one hook, four listeners
    for listener in listeners.values():
        listener.stop()
    assert windows.quit.wait(2)
    for _ in range(100):
        if windows.unhooked:
            break
        __import__("time").sleep(0.01)
    assert windows.unhooked == [1] and hotkey.HOOK.listeners == ()  # removed with the last listener


def test_one_hook_feeds_every_shortcut_its_own_events(windows):
    from sst import hotkey
    listeners, woken = _shortcuts()
    hotkey.HOOK.listeners = tuple(listeners.values())
    A, L = 0x41, 0x4C
    # Ordinary typing: every app gets every key, and nobody is woken up.
    assert _type(windows, ("down", A), ("up", A)) == [0, 0] and windows.calls == 2
    assert woken == {"dictation": 0, "transform": 0, "translate": 0, "live": 0}
    # Ctrl+Win held: dictation's press and release (and the Start menu masked, as before).
    _type(windows, ("down", LCTRL), ("down", LWIN), ("up", LWIN), ("up", LCTRL))
    assert _drain(listeners["dictation"]) == ["press", "release"] and windows.sent == [0xE8, 0xE8]
    # Ctrl tapped twice: Text Transform.
    _type(windows, ("down", LCTRL), ("up", LCTRL), ("down", LCTRL), ("up", LCTRL))
    assert _drain(listeners["transform"]) == ["release"]
    # Ctrl+C+C: Translate; the app still copies (nothing hidden).
    answers = _type(windows, ("down", LCTRL), ("down", C), ("up", C), ("down", C), ("up", C), ("up", LCTRL))
    assert _drain(listeners["translate"]) == ["release"] and 1 not in answers
    # Ctrl+Alt+L: live translation; L is hidden from the app.
    answers = _type(windows, ("down", LCTRL), ("down", LALT), ("down", L), ("up", L), ("up", LALT), ("up", LCTRL))
    assert _drain(listeners["live"]) == ["press", "release"] and answers[2:4] == [1, 1]
    assert _drain(listeners["dictation"]) == [] and _drain(listeners["transform"]) == []  # nobody else's
    assert woken == {"dictation": 2, "transform": 1, "translate": 1, "live": 2}  # woken once per event, never polled


def test_a_listener_that_fails_never_costs_the_others_their_keys(windows):
    from sst import hotkey
    listeners, _ = _shortcuts()
    listeners["live"].feed = lambda vk, is_down: 1 / 0  # a bug in one shortcut, fed before dictation's
    hotkey.HOOK.listeners = (listeners["live"], listeners["dictation"])
    assert _type(windows, ("down", LCTRL), ("down", LWIN), ("up", LWIN), ("up", LCTRL)) == [0, 0, 0, 0]
    assert _drain(listeners["dictation"]) == ["press", "release"]  # and every app still got its keys
