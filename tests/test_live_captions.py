"""The translation bar and the Qt side of a live translation session, off-screen: no window shown, no focus taken,
no real mouse (Qt's test events only)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QPoint, QRect, QSize, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst.live.captions import CaptionBar, LiveCaptions, dragged  # noqa: E402
from sst.live.contracts import MIC, SYSTEM, Kind, LiveConfig, LiveEvent  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qt():
    return QApplication.instance() or QApplication([])


def line(text, source="", lane=SYSTEM):
    return LiveEvent(Kind.LINE, text, source=source, lane=lane)


# ---- what the bar shows

def test_each_line_shows_the_words_heard_above_its_translation_and_the_whole_session_stays():
    bar = CaptionBar()
    for i in range(40):
        bar.show_event(line(f"Line {i} in English.", f"日本語の{i}行目。"))
    bar.show_event(LiveEvent(Kind.SOURCE, "問題が一つ"))
    bar.show_event(LiveEvent(Kind.TRANSLATION, "One issue"))
    text = bar.text()
    assert "日本語の0行目。\nLine 0 in English." in text  # nothing scrolls away: the session is all there to read back
    assert text.index("Line 39") < text.index("問題が一つ") < text.index("One issue")
    bar.grab()


def test_words_in_progress_are_redrawn_not_repeated():
    bar = CaptionBar()
    for words in ("One", "One issue", "One issue left."):
        bar.show_event(LiveEvent(Kind.TRANSLATION, words))
    assert bar.text().count("One") == 1
    bar.show_event(line("One issue left.", "問題が一つ残っています。"))
    bar.show_event(LiveEvent(Kind.TRANSLATION, "Next"))
    assert bar.text().count("One issue left.") == 1 and bar.text().endswith("Next")


def test_a_line_already_in_the_target_language_shows_once():
    bar = CaptionBar()
    bar.show_event(line("", "Let's start."))
    assert bar.text().count("Let's start.") == 1


def test_with_both_the_users_own_lines_are_marked_and_their_words_left_out():
    bar = CaptionBar(LiveConfig(source="both"))
    bar.show_event(line("Shall we start?", "始めましょうか。"))
    bar.show_event(line("はい、始めましょう。", "Yes, let's start.", lane=MIC))
    bar.show_event(LiveEvent(Kind.SOURCE, "The budget", lane=MIC))  # the user's own words aren't shown
    text = bar.text()
    assert "You · はい、始めましょう。" in text and "始めましょうか。\nShall we start?" in text
    assert "Yes, let's start." not in text and "The budget" not in text


def test_with_the_microphone_alone_words_already_in_its_language_are_shown_as_heard_and_marked():
    bar = CaptionBar(LiveConfig(source="microphone", mic_target="en"))
    bar.show_event(line("", "Let's set the budget next week.", lane=MIC))
    bar.show_event(LiveEvent(Kind.LINE, "", source="来週にしましょう。", language="ja", lane=MIC))  # translation late
    assert bar.text() == "already English\nLet's set the budget next week.\n来週にしましょう。"
    computer = CaptionBar(LiveConfig(target="en"))  # what the laptop plays: as before, unmarked
    computer.show_event(line("", "Let's start."))
    assert computer.text() == "Let's start."


def test_with_the_microphone_alone_nothing_is_marked():
    bar = CaptionBar(LiveConfig(source="microphone"))
    bar.show_event(line("Shall we start?", "始めましょうか。", lane=MIC))
    assert "You" not in bar.text() and "始めましょうか。\nShall we start?" in bar.text()


def test_the_title_says_what_is_translated_into_what():
    assert CaptionBar(LiveConfig(target="en")).title.text() == "Live translation  ·  into English  ·  Starting…"
    both = CaptionBar(LiveConfig(source="both", target="en", mic_target="ja"))
    both.show_event(LiveEvent(Kind.STATUS, "Listening"))
    assert both.title.text() == "Live translation  ·  into English  ·  you into Japanese"
    assert "microphone into Japanese" in CaptionBar(LiveConfig(source="microphone", mic_target="ja")).title.text()


def test_before_anyone_speaks_it_says_it_listens():
    bar = CaptionBar(LiveConfig(target="ja"))
    bar.show_event(LiveEvent(Kind.STATUS, "Listening"))
    assert bar.text() == "Translations into Japanese appear here when someone speaks."


def test_the_status_line_shows_each_ways_note_until_the_session_clears_it():
    bar = CaptionBar(LiveConfig(source="both"))
    assert bar.footer.isHidden() and bar.note() == ""
    bar.show_event(LiveEvent(Kind.NOTE, "Listening…"))
    bar.show_event(LiveEvent(Kind.NOTE, "Listening…", lane=MIC))
    assert bar.note() == "Listening…" and not bar.footer.isHidden()  # the same words once
    bar.show_event(LiveEvent(Kind.NOTE, "Nothing heard from the microphone", lane=MIC))
    bar.show_event(LiveEvent(Kind.NOTE, "Hearing sound, but no speech yet"))
    assert bar.footer.text() == "Hearing sound, but no speech yet  ·  Nothing heard from the microphone"
    assert "Nothing heard" not in bar.text()  # a line of its own, not one of the captions
    bar.show_event(LiveEvent(Kind.NOTE, ""))
    bar.drop_lane(MIC)
    assert bar.note() == "" and bar.footer.isHidden()


def test_the_live_page_hears_when_the_status_line_changes():
    live, made = live_captions()
    noted = []
    live.noted.connect(noted.append)
    live.start(LiveConfig(target="en"))
    made[0].on_event(LiveEvent(Kind.NOTE, "Nothing heard yet: is the sound playing on this laptop?"))
    QApplication.processEvents()
    assert noted == [live.note] == ["Nothing heard yet: is the sound playing on this laptop?"]
    live.stop()
    assert live.note == ""


def test_a_problem_says_whose_it_is_and_goes_when_words_come():
    bar = CaptionBar(LiveConfig(source="both"))
    bar.show_event(LiveEvent(Kind.TRANSLATION, "We'll decide next week."))
    bar.show_event(LiveEvent(Kind.ERROR, "The microphone couldn't be opened.", lane=MIC))
    assert "We'll decide next week.\nYour speech: The microphone couldn't be opened." in bar.text()
    bar.show_event(LiveEvent(Kind.TRANSLATION, "はい", lane=MIC))
    assert "couldn't" not in bar.text()


def test_it_follows_new_lines_until_the_user_scrolls_up():
    bar = CaptionBar()
    bar.setGeometry(0, 0, 400, 160)
    bar.show()
    for i in range(30):
        bar.show_event(line(f"Line {i}.", f"{i}行目。"))
    QApplication.processEvents()
    scroll = bar.view.verticalScrollBar()
    assert scroll.maximum() > 0 and scroll.value() == scroll.maximum()
    scroll.setValue(0)  # the user reads back
    bar.show_event(LiveEvent(Kind.TRANSLATION, "New words"))
    bar.show_event(line("A new line.", "新しい行。"))
    QApplication.processEvents()
    assert scroll.value() == 0  # left where they were reading
    scroll.setValue(scroll.maximum())  # back at the bottom: it follows again
    bar.show_event(line("Another line.", "もう一行。"))
    QApplication.processEvents()
    assert scroll.value() == scroll.maximum()
    bar.close()


# ---- moving it, resizing it, closing it

def test_a_drag_moves_it_and_an_edge_resizes_it_never_below_its_smallest():
    start, smallest = QRect(100, 100, 600, 200), QSize(360, 140)
    assert dragged(start, (False, False, False, False), QPoint(50, 40), smallest) == QRect(150, 140, 600, 200)
    assert dragged(start, (False, False, True, False), QPoint(50, 0), smallest) == QRect(100, 100, 650, 200)
    assert dragged(start, (True, True, False, False), QPoint(-20, -30), smallest) == QRect(80, 70, 620, 230)
    assert dragged(start, (False, False, True, True), QPoint(-500, -500), smallest).size() == smallest


def test_the_mouse_moves_and_resizes_it_and_where_it_ends_up_is_remembered():
    bar = CaptionBar()
    bar.setGeometry(100, 100, 600, 200)
    bar.show()
    ended = []
    bar.moved.connect(ended.append)
    QTest.mousePress(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(300, 15))
    QTest.mouseMove(bar, QPoint(350, 55))
    QTest.mouseRelease(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(350, 55))
    assert bar.geometry() == QRect(150, 140, 600, 200) and ended == [[150, 140, 600, 200]]
    QTest.mousePress(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(597, 100))  # right edge
    QTest.mouseMove(bar, QPoint(677, 100))
    QTest.mouseRelease(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(677, 100))
    assert bar.geometry() == QRect(150, 140, 680, 200) and ended[-1] == [150, 140, 680, 200]
    viewport = bar.view.viewport()  # the text moves it too
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(40, 40))
    QTest.mouseMove(viewport, QPoint(30, 40))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(30, 40))
    assert bar.geometry().topLeft() == QPoint(140, 140)
    bar.close()


def test_it_opens_where_it_was_left_if_that_is_still_on_a_screen():
    area = QApplication.primaryScreen().availableGeometry()
    bar = CaptionBar()
    bar.place([area.left() + 50, area.top() + 60, 500, 180])
    assert bar.geometry() == QRect(area.left() + 50, area.top() + 60, 500, 180)
    bar.place([area.right() + 5000, area.top(), 500, 180])  # on a screen that's gone
    assert area.contains(bar.geometry().center())


# ---- the session behind it

class FakeSession:
    """A LiveSession's ways, without captures or engines."""

    def __init__(self, config, on_event, fail=()):
        self.config, self.on_event, self.fail = config, on_event, fail
        self.lanes, self.stopped, self.transcript, self.speaker = {}, False, None, None

    def set_speaker(self, speaker):
        self.speaker = speaker

    def add(self, lane, capture, engine_factory):
        if lane in self.fail:
            raise OSError("no microphone" if lane == MIC else "no output device")
        self.lanes[lane] = capture

    def remove(self, lane):
        self.lanes.pop(lane, None)

    def stop(self):
        self.stopped = True
        self.lanes.clear()


def live_captions(fail=()):
    made = []
    live = LiveCaptions(lambda config, on_event: made.append(FakeSession(config, on_event, fail)) or made[-1],
                        lambda lane, config: (f"{lane} into {config.for_lane(lane).target}", None))
    return live, made


def test_start_shows_the_bar_and_stop_removes_it():
    live, made = live_captions()
    running = []
    live.changed.connect(running.append)
    live.start(LiveConfig(target="ja"))
    assert live.running and live.bar is not None and live.bar.isVisible()
    assert made[0].lanes == {SYSTEM: "system into ja"}
    made[0].on_event(LiveEvent(Kind.TRANSLATION, "hello"))  # from the engine's thread in real life: queued to Qt's
    QApplication.processEvents()
    assert "hello" in live.bar.text()
    live.stop()
    assert not live.running and live.bar is None and made[0].stopped and running == [True, False]


@pytest.mark.parametrize(("source", "ways"), [("computer", {SYSTEM: "system into en"}),
                                              ("microphone", {MIC: "mic into ja"}),
                                              ("both", {SYSTEM: "system into en", MIC: "mic into ja"})])
def test_each_source_runs_its_ways(source, ways):
    live, made = live_captions()
    live.start(LiveConfig(source=source, target="en", mic_target="ja"))
    assert made[0].lanes == ways
    live.stop()


def test_the_source_can_change_while_it_runs():
    live, made = live_captions()
    live.start(LiveConfig(source="computer"))
    assert live.set_source("both") == "" and set(made[0].lanes) == {SYSTEM, MIC} and live.bar.config.source == "both"
    assert live.set_source("microphone") == "" and set(made[0].lanes) == {MIC} and live.running
    live.stop()


def test_a_microphone_that_cant_open_is_said_on_the_bar_and_the_computer_goes_on():
    live, made = live_captions(fail=(MIC,))
    live.start(LiveConfig(source="both"))
    QApplication.processEvents()
    assert live.running and list(made[0].lanes) == [SYSTEM]
    assert "Your speech: The microphone couldn't be opened (no microphone)." in live.bar.text()
    live.stop()


def test_nothing_that_can_start_leaves_nothing_behind():
    live, _ = live_captions(fail=(MIC,))
    with pytest.raises(OSError):
        live.start(LiveConfig(source="microphone"))
    assert not live.running and live.bar is None


def test_the_close_button_stops_live_translation():
    live, made = live_captions()
    running = []
    live.changed.connect(running.append)
    live.start(LiveConfig())
    live.bar.close_button.click()
    assert not live.running and made[0].stopped and running == [True, False]


def test_where_the_bar_is_put_is_passed_on_to_remember():
    live, _ = live_captions()
    places = []
    live.moved.connect(places.append)
    live.start(LiveConfig(), geometry=[40, 50, 500, 180])
    assert live.bar.geometry() == QRect(40, 50, 500, 180)
    live.bar.moved.emit([60, 50, 500, 180])
    assert places == [[60, 50, 500, 180]]
    live.stop()


def test_the_copy_button_asks_for_a_copy_and_the_bar_says_what_happened():
    live, made = live_captions()
    asked = []
    live.copy_requested.connect(lambda: asked.append(True))
    live.start(LiveConfig(target="en"))
    assert live.transcript_path is None  # nobody has spoken yet
    live.bar.copy_button.click()
    assert asked == [True] and not live.bar.copy_button.hasFocus()
    live.flash("Copied 3 lines")
    assert live.bar.title.text().endswith("Copied 3 lines")
    live.bar._flash_timer.timeout.emit()  # a moment later the title is itself again
    assert "Copied" not in live.bar.title.text()
    live.stop()
    live.flash("after it stopped")  # nothing to say it on: no error


def test_showing_the_bar_in_screen_shares_applies_at_once():
    live, _ = live_captions()
    live.start(LiveConfig())
    assert live.bar.hide_from_capture
    live.set_hidden(False)
    assert not live.bar.hide_from_capture and not live.config.hide_from_capture
    live.stop()


def test_when_the_engine_gives_up_the_bar_goes_and_the_reason_stays():
    live, made = live_captions()
    live.start(LiveConfig())
    made[0].on_event(LiveEvent(Kind.ERROR, "Google refused the key: check the Gemini key in AI & models."))
    made[0].on_event(LiveEvent(Kind.STATUS, "Stopped"))
    QApplication.processEvents()
    assert not live.running and live.last_problem.startswith("Google refused the key")


def test_when_one_way_gives_up_the_other_goes_on():
    live, made = live_captions()
    live.start(LiveConfig(source="both"))
    made[0].on_event(LiveEvent(Kind.ERROR, "Google's quota for this key is used up for now.", lane=MIC))
    made[0].on_event(LiveEvent(Kind.STATUS, "Stopped", lane=MIC))
    QApplication.processEvents()
    assert live.running and list(made[0].lanes) == [SYSTEM] and live.last_problem.startswith("Your speech:")
    live.stop()


def test_a_way_switched_off_on_purpose_doesnt_stop_the_rest():
    live, made = live_captions()
    live.start(LiveConfig(source="both"))
    live.set_source("computer")
    made[0].on_event(LiveEvent(Kind.STATUS, "Stopped", lane=MIC))  # what the microphone's engine says as it stops
    QApplication.processEvents()
    assert live.running and list(made[0].lanes) == [SYSTEM]
    live.stop()


# ---- the spoken translation on the bar

class FakeSpeaker:
    language = "en"

    def __init__(self, config):
        self.lanes, self.own, self.speed, self.config = None, None, None, config

    def set_lanes(self, lanes, own=None):
        self.lanes, self.own = tuple(lanes), tuple(own or ())

    def set_speed(self, speed):
        self.speed = speed


def speaking_captions(ready=True):
    made, speakers = [], []

    def make_speaker(config):
        if not ready:
            return None
        speakers.append(FakeSpeaker(config))
        return speakers[-1]
    live = LiveCaptions(lambda config, on_event: made.append(FakeSession(config, on_event)) or made[-1],
                        lambda lane, config: (lane, None), make_speaker)
    return live, made, speakers


def test_speaking_starts_with_the_session_gemini_says_any_language_the_laptops_voice_its_own():
    live, made, speakers = speaking_captions()
    live.start(LiveConfig(source="both", target="en", mic_target="ja", speak=True, speak_speed=1.15))
    assert made[0].speaker is speakers[0] and speakers[0].lanes == speakers[0].own == (SYSTEM,)  # not the user's words
    assert speakers[0].speed == 1.15 and live.bar.speak_button.isChecked()
    live.set_source("microphone")  # into Japanese: Gemini says it; nothing for an English voice on the laptop
    assert speakers[0].lanes == () and speakers[0].own == (MIC,)
    assert "speaks only English" in live.bar.speak_button.toolTip()
    live.set_speak(False)
    assert made[0].speaker is None and not live.bar.speak_button.isChecked()
    live.stop()


def test_how_low_the_other_apps_go_is_changed_while_it_speaks():
    live, made, speakers = speaking_captions()
    live.set_duck(0.5)  # nothing running: for the next start
    live.start(LiveConfig(speak=True, duck=0.5))
    speakers[0].ducker = type("Ducker", (), {"set_depth": lambda self, depth: setattr(self, "depth", depth)})()
    live.set_duck(0.2)
    assert live.config.duck == 0.2 and speakers[0].ducker.depth == 0.2
    live.stop()


def test_the_speaker_button_asks_for_speaking_and_the_bar_follows_what_happened():
    live, made, speakers = speaking_captions(ready=False)
    asked = []
    live.speak_toggled.connect(asked.append)
    live.start(LiveConfig())
    live.bar.speak_button.click()
    assert asked == [True]
    live.set_speak(True)  # the voice isn't downloaded yet: nothing speaks, the button stays off
    assert made[0].speaker is None and not live.bar.speak_button.isChecked()
    live.set_voice_note("Downloading the voice: 40%")
    assert live.bar.title.text().endswith("Downloading the voice: 40%")
    live.stop()


# ---- the cost saver

class SavingSession(FakeSession):
    """A FakeSession that says what it cost (session.Spend's fields) and how long nobody has spoken."""

    def __init__(self, config, on_event):
        super().__init__(config, on_event)
        self.saving, self.quiet, self.cost = config.save, 0.0, (0.42, 0.26, 660.0, 420.0)

    def set_saving(self, on):
        self.saving = on

    def spend(self):
        cost, saved, seconds, unsent = self.cost
        return type("Spend", (), {"cost": cost, "saved": saved, "seconds": seconds, "unsent_s": unsent})()

    def quiet_for(self, now=None):
        return self.quiet


def saving_captions():
    made = []
    live = LiveCaptions(lambda config, on_event: made.append(SavingSession(config, on_event)) or made[-1],
                        lambda lane, config: (lane, None))
    return live, made


def test_the_cost_saver_button_asks_for_it_and_the_bar_shows_what_the_session_cost_and_saved():
    live, made = saving_captions()
    asked = []
    live.save_toggled.connect(asked.append)
    live.start(LiveConfig())
    bar = live.bar
    assert not bar.save_button.isChecked() and "$" not in bar.title.text()  # off: the bar as it was
    assert bar.save_button.toolTip() == "Cost saver: send audio only while someone talks"
    head = bar.layout().itemAt(0).layout()
    assert head.indexOf(bar.save_button) == head.indexOf(bar.copy_button) - 1  # just before the copy button
    bar.save_button.click()
    assert asked == [True] and not bar.save_button.hasFocus()
    live.set_save(True)  # what the app does: saved, and at once on every way
    assert made[0].saving and bar.save_button.isChecked() and live.config.save
    assert bar.title.text().endswith("$0.42 · saved $0.26")
    made[0].cost = (0.5, 0.3, 700.0, 450.0)
    live._ticks.timeout.emit()  # every 5 s
    assert bar.title.text().endswith("$0.50 · saved $0.30")
    live.set_save(False)
    assert not made[0].saving and "$" not in bar.title.text() and not bar.save_button.isChecked()
    live.stop()
    assert live.last_spend is None  # it didn't save at the end: nothing to say about it


def test_a_saving_session_with_no_words_is_warned_then_stops_by_itself():
    live, made = saving_captions()
    notes, stops, running = [], [], []
    live.noted.connect(notes.append)
    live.quiet_stop.connect(stops.append)
    live.changed.connect(running.append)
    live.start(LiveConfig(save=True))
    assert live.bar.save_button.isChecked() and made[0].saving
    made[0].quiet = 700.0
    live.tick()
    assert live.running and not live.bar.warning
    made[0].quiet = 780.0
    live.tick()
    assert live.bar.warning == "No speech for 13 min: stops in 2 min" and live.note.startswith("No speech for 13 min")
    assert notes[-1] == live.note and not live.bar.footer.isHidden()
    made[0].quiet = 30.0  # someone spoke
    live.tick()
    assert live.bar.warning == "" and live.bar.footer.isHidden()
    made[0].quiet = 900.0
    live.tick()
    assert not live.running and made[0].stopped and running == [True, False]
    assert live.stopped_why == "no speech for 15 min" and stops == ["no speech for 15 min"]
    assert (live.last_spend.cost, live.last_spend.saved, live.last_spend.seconds) == (0.42, 0.26, 660.0)
    live.start(LiveConfig(save=True))  # the next session starts afresh
    assert live.stopped_why == "" and live.last_spend is None
    live.stop()


def test_without_the_cost_saver_a_quiet_session_goes_on():
    live, made = saving_captions()
    live.start(LiveConfig())
    made[0].quiet = 5000.0
    live.tick()
    assert live.running and not live.bar.warning and live.stopped_why == ""
    live.stop()
