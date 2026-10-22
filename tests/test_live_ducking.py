"""The other apps kept lower while live translation speaks (sst.live.ducking), with fake apps."""
import json
import time

import pytest

from sst.live import ducking
from sst.live.ducking import CHANGED, DOWN, INAUDIBLE, KEEP, RESCAN, UP, Ducker, put_back


class Sound:
    """One app's sound (an audio session) as Windows would hand it out."""

    def __init__(self, name, volume=1.0, key=None, playing=True):
        self.name, self.key, self.level, self.playing = name, key or f"{name}|1", volume, playing
        self.closed, self.gone, self.sets = 0, False, []

    def volume(self):
        if self.gone:
            raise OSError("the app ended")
        return self.level

    def set_volume(self, value):
        if self.gone:
            raise OSError("the app ended")
        self.level = value
        self.sets.append(round(value, 3))
        self.lowest = min(getattr(self, "lowest", 1.0), value)

    def close(self):
        self.closed += 1


class Apps:
    """What find(playing) sees: the sounds of the apps running now."""

    def __init__(self, *sounds):
        self.sounds = list(sounds)

    def __call__(self, playing):
        return [s for s in self.sounds if s.playing or not playing]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def ducker_with(*sounds, depth=0.3, tmp_path):
    clock, apps = Clock(), Apps(*sounds)
    return Ducker(depth, find=apps, record=tmp_path / "lowered.json", clock=clock, wall=lambda: 5000.0), clock, apps


def run(ducker, clock, seconds, step=0.02):
    for _ in range(round(seconds / step)):
        clock.now += step
        ducker.tick()


def test_the_apps_stay_lowered_the_whole_session_and_come_back_when_it_stops(tmp_path):
    video, music = Sound("chrome.exe", 1.0), Sound("spotify.exe", 0.5)
    ducker, clock, _ = ducker_with(video, music, tmp_path=tmp_path)
    run(ducker, clock, DOWN + 0.1)
    assert video.level == pytest.approx(0.3) and music.level == pytest.approx(0.15)  # each from its own volume
    record = json.loads((tmp_path / "lowered.json").read_text(encoding="utf-8"))
    assert {(e["name"], e["original"]) for e in record} == {("chrome.exe", 1.0), ("spotify.exe", 0.5)}
    run(ducker, clock, 30)  # between sentences and through long pauses: still low, no pumping up and down
    assert video.level == pytest.approx(0.3) and music.level == pytest.approx(0.15)
    lowest = video.sets.index(min(video.sets))  # a ramp down, then left there
    assert video.sets[:lowest + 1] == sorted(video.sets[:lowest + 1], reverse=True) and lowest == len(video.sets) - 1
    ducker.release()  # live translation or speaking stopped, or Rflow quits
    assert video.level == 1.0 and music.level == 0.5 and not (tmp_path / "lowered.json").exists()
    assert not ducker._lowered and video.closed  # Windows' handles given back


def test_it_goes_down_quickly_and_comes_back_up_gently_when_raised(tmp_path):
    video = Sound("chrome.exe")
    ducker, clock, _ = ducker_with(video, tmp_path=tmp_path)
    run(ducker, clock, 0.06)
    assert 0.3 < video.level < 1.0  # on its way down
    run(ducker, clock, 1.0)
    ducker.set_depth(0.8)
    run(ducker, clock, UP / 2)
    assert 0.3 < video.level < 0.8  # on its way up
    run(ducker, clock, UP)
    assert video.level == pytest.approx(0.8)


def test_nothing_is_touched_when_the_apps_stay_as_they_are(tmp_path):
    video = Sound("chrome.exe")
    ducker, clock, _ = ducker_with(video, depth=1.0, tmp_path=tmp_path)
    run(ducker, clock, 1.0)
    assert video.sets == [] and not (tmp_path / "lowered.json").exists()


def test_an_app_that_starts_playing_during_the_session_is_lowered_too(tmp_path):
    video, call = Sound("chrome.exe"), Sound("teams.exe", playing=False)
    ducker, clock, _ = ducker_with(video, call, tmp_path=tmp_path)
    run(ducker, clock, 0.3)
    assert call.sets == []  # not playing: left alone
    call.playing = True
    run(ducker, clock, RESCAN + 0.1)
    assert call.level == pytest.approx(0.3)


def test_an_app_the_user_turns_up_meanwhile_keeps_the_users_volume(tmp_path):
    video = Sound("chrome.exe", 0.8)
    ducker, clock, _ = ducker_with(video, tmp_path=tmp_path)
    run(ducker, clock, 0.5)
    video.level = 0.6  # the user moves its slider in the volume mixer
    run(ducker, clock, 0.2)
    ducker.release()
    assert video.level == 0.6 and not (tmp_path / "lowered.json").exists()


def test_an_app_that_ends_is_let_go_quietly(tmp_path):
    video, music = Sound("chrome.exe"), Sound("spotify.exe")
    ducker, clock, _ = ducker_with(video, music, tmp_path=tmp_path)
    run(ducker, clock, 0.5)
    video.gone = True
    ducker.set_depth(0.5)
    run(ducker, clock, UP + 0.2)
    assert music.level == pytest.approx(0.5) and "chrome.exe|1" not in ducker._lowered and video.closed


def test_a_new_level_is_followed_at_once(tmp_path):
    video = Sound("chrome.exe")
    ducker, clock, _ = ducker_with(video, tmp_path=tmp_path)
    run(ducker, clock, 0.5)
    ducker.set_depth(0.2)
    run(ducker, clock, 0.1)
    assert video.level == pytest.approx(0.2)
    ducker.set_depth(1.0)  # "Unchanged": back up, and nothing more is lowered
    run(ducker, clock, UP + 0.1)
    assert video.level == 1.0 and not ducker._lowered and not (tmp_path / "lowered.json").exists()
    ducker.set_depth(-3)
    assert ducker.depth == 0.0


@pytest.mark.parametrize(("original", "level"), [(1.0, INAUDIBLE), (0.5, INAUDIBLE), (0.002, INAUDIBLE)])
def test_zero_percent_is_inaudible_but_never_silence_so_live_translation_still_hears_the_meeting(tmp_path, original,
                                                                                                    level):
    """Windows' mix after a session volume of exactly 0 is silence: live translation would hear nothing."""
    video = Sound("chrome.exe", original)
    ducker, clock, _ = ducker_with(video, depth=0.0, tmp_path=tmp_path)
    run(ducker, clock, DOWN + 0.1)
    assert video.level == pytest.approx(level) and 0 < INAUDIBLE <= 0.0003  # phase 37: translated in full at 0.03%
    assert video.lowest == pytest.approx(level)  # never 0 on the way down either
    ducker.release()
    assert video.level == original


def test_the_thread_lowers_until_it_is_stopped_then_puts_back(tmp_path):
    video = Sound("chrome.exe")
    ducker = Ducker(0.3, find=Apps(video), record=tmp_path / "lowered.json")
    ducker.start()
    end = time.monotonic() + 3
    while time.monotonic() < end and video.level > 0.31:
        time.sleep(0.01)
    assert video.level == pytest.approx(0.3)
    time.sleep(0.2)
    assert video.level == pytest.approx(0.3)  # and stays there
    ducker.stop()
    assert video.level == 1.0 and not any(t.name == "live-ducker" and t.is_alive() for t in __import__("threading").enumerate())


# ---- a Rflow that ended while the apps were low

def write_record(path, *entries):
    path.write_text(json.dumps(list(entries)), encoding="utf-8")


def test_apps_left_lowered_are_put_back_at_the_next_start(tmp_path):
    record = tmp_path / "lowered.json"
    video, moved, other = Sound("chrome.exe", 0.3), Sound("spotify.exe", 0.9), Sound("vlc.exe", 0.2)
    write_record(record, {"key": "chrome.exe|1", "name": "chrome.exe", "original": 1.0, "depth": 0.3, "at": 4000},
                 {"key": "spotify.exe|1", "name": "spotify.exe", "original": 0.5, "depth": 0.3, "at": 4000})
    assert put_back(Apps(video, moved, other), record, wall=lambda: 5000.0) == 1
    assert video.level == 1.0  # as it was before Rflow lowered it
    assert moved.level == 0.9 and other.level == 0.2  # the user's own volume, and an app never lowered: untouched
    assert not record.exists() and video.closed == moved.closed == other.closed == 1


def test_an_app_not_running_now_is_put_back_when_it_runs_again_within_a_week(tmp_path):
    record = tmp_path / "lowered.json"
    write_record(record, {"key": "teams.exe|1", "name": "teams.exe", "original": 0.8, "depth": 0.3, "at": 4000},
                 {"key": "old.exe|1", "name": "old.exe", "original": 1.0, "depth": 0.3, "at": 5000 - KEEP - 1})
    assert put_back(Apps(), record, wall=lambda: 5000.0) == 0
    assert [e["name"] for e in json.loads(record.read_text(encoding="utf-8"))] == ["teams.exe"]
    teams = Sound("teams.exe", 0.8 * 0.3, key="teams.exe|2", playing=False)  # a new run: Windows remembered it low
    assert put_back(Apps(teams), record, wall=lambda: 6000.0) == 1
    assert teams.level == 0.8 and not record.exists()


def test_a_ducker_keeps_what_was_left_in_the_record_while_it_lowers_others(tmp_path):
    record = tmp_path / "lowered.json"
    write_record(record, {"key": "teams.exe|1", "name": "teams.exe", "original": 0.8, "depth": 0.3, "at": 4000,
                          "left": True})
    video = Sound("chrome.exe")
    ducker = Ducker(0.3, find=Apps(video), record=record, clock=Clock(), wall=lambda: 5000.0)
    run(ducker, ducker._clock, 0.3)
    assert {e["name"] for e in json.loads(record.read_text(encoding="utf-8"))} == {"teams.exe", "chrome.exe"}
    ducker.release()
    assert [e["name"] for e in json.loads(record.read_text(encoding="utf-8"))] == ["teams.exe"]


def test_a_damaged_record_is_ignored(tmp_path):
    record = tmp_path / "lowered.json"
    record.write_text("{not json", encoding="utf-8")
    assert put_back(Apps(Sound("chrome.exe", 0.3)), record) == 0


def test_the_logs_name_the_program_not_the_whole_session_id():
    name = r"{0.0.0.00000000}.{a1b2}|\Device\HarddiskVolume3\Program Files\Google\Chrome\Application\chrome.exe%b{00000000-0000}"
    assert ducking._app_name(name) == "chrome.exe" and ducking._app_name("") == "an app"
    assert CHANGED < 0.05  # a slider moved by hand is noticed
