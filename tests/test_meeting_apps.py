"""Finding the meeting app's window from a list of top-level windows (fakes only: no real window is looked at)."""
from sst.meeting import apps
from sst.meeting.apps import MEET, TEAMS, WEBEX, ZOOM, MeetingWindow, choose, find


def listed(*windows):
    return lambda: list(windows)


def test_several_teams_windows_give_one_meeting():
    found = find(listed(
        (10, 7, "ms-teams.exe", "Chat | Priya | Microsoft Teams"),
        (11, 7, "ms-teams.exe", "Meeting with Priya | Microsoft Teams"),
        (12, 7, "ms-teams.exe", "Sprint review | Microsoft Teams"),
        (13, 7, "ms-teams.exe", "Microsoft Teams"),
    ))
    assert found == [MeetingWindow(TEAMS, "Meeting with Priya | Microsoft Teams", 11, 7)]


def test_teams_main_window_alone_is_no_meeting():
    assert find(listed((10, 7, "ms-teams.exe", "Calendar | Microsoft Teams"),
                       (11, 7, "teams.exe", "Activity | Microsoft Teams"),
                       (12, 7, "ms-teams.exe", "Microsoft Teams Notification"))) == []


def test_teams_meeting_with_its_own_name_is_found():
    assert find(listed((10, 7, "teams.exe", "Sprint review | Microsoft Teams")))[0].title.startswith("Sprint review")


def test_zoom_meeting_first_and_one_per_app():
    found = find(listed((1, 3, "zoom.exe", "Zoom Workplace"), (2, 3, "zoom.exe", "Zoom Meeting"), (3, 3, "zoom.exe", "Zoom")))
    assert found == [MeetingWindow(ZOOM, "Zoom Meeting", 2, 3)]
    assert find(listed((1, 3, "zoom.exe", "Settings"))) == []


def test_meet_in_browsers_and_other_tabs_ignored():
    found = find(listed(
        (1, 5, "chrome.exe", "Inbox - Gmail - Google Chrome"),
        (2, 5, "chrome.exe", "Meet - abc-defg-hij - Google Chrome"),
        (3, 6, "msedge.exe", "Meet - xyz-abcd-efg and 2 more pages - Personal - Microsoft Edge"),
        (4, 8, "firefox.exe", "Google Meet — Mozilla Firefox"),
        (5, 9, "notepad.exe", "Meet - notes.txt - Notepad"),
    ))
    assert [(w.app, w.hwnd) for w in found] == [(MEET, 2), (MEET, 3), (MEET, 4)]  # each browser window its own meeting


def test_webex():
    found = find(listed((1, 4, "webex.exe", "Webex"), (2, 4, "ciscocollabhost.exe", "Weekly sync meeting")))
    assert found == [MeetingWindow(WEBEX, "Weekly sync meeting", 2, 4)]


def test_most_likely_first_then_topmost():
    found = find(listed(
        (1, 3, "zoom.exe", "Zoom"),                                  # weak, on top
        (2, 5, "chrome.exe", "Meet - abc-defg-hij - Google Chrome"),  # strong
        (3, 7, "ms-teams.exe", "Call with Ken | Microsoft Teams"),   # strong, below
    ))
    assert [w.app for w in found] == [MEET, TEAMS, ZOOM]


def test_nothing_found():
    assert find(listed()) == [] and find(listed((1, 2, "explorer.exe", "Downloads"))) == []


def test_choose():
    one, two = MeetingWindow(TEAMS, "a", 1, 2), MeetingWindow(ZOOM, "b", 3, 4)
    assert choose([one]) is one
    assert choose([one, two]) is None and choose([]) is None


def test_alive_checks_the_same_process(monkeypatch):
    window = MeetingWindow(TEAMS, "Meeting | Microsoft Teams", 42, 7)

    class FakeUser32:
        IsWindow = staticmethod(lambda hwnd: hwnd == 42)

    monkeypatch.setattr(apps, "user32", FakeUser32)
    monkeypatch.setattr(apps, "_pid", lambda hwnd: 7)
    assert apps.alive(window)
    monkeypatch.setattr(apps, "_pid", lambda hwnd: 8)  # the handle now belongs to another process
    assert not apps.alive(window)
    assert not apps.alive(MeetingWindow(TEAMS, "x", 43, 7))
