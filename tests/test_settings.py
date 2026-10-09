import json
from datetime import date, timedelta

import pytest

from sst import settings
from sst.settings import Profiles, Settings, Stats, add_to_history, read_history


def test_missing_file_gives_defaults(tmp_path):
    assert Settings.load(tmp_path / "none.json") == Settings()


def test_round_trip(tmp_path):
    path = tmp_path / "sub" / "settings.json"
    Settings(hotkey="menu", microphone="マイク (Realtek(R) Audio)", sounds=False, save_recordings=False).save(path)
    assert Settings.load(path) == Settings(hotkey="menu", microphone="マイク (Realtek(R) Audio)", sounds=False,
                                           save_recordings=False)


def test_translate_defaults():
    s = Settings()
    assert (s.translate_shortcut, s.translate_to, s.translate_second) == ("ctrl+c+c", "English", "")


def test_snippets_round_trip_and_a_wrong_shape_is_dropped(tmp_path):
    path = tmp_path / "settings.json"
    snippets = [{"cue": "my email", "text": "xyz@gmail.com", "anywhere": False},
                {"cue": "my signature", "text": "Best regards,\nKarthi", "anywhere": True}]
    Settings(snippets=snippets).save(path)
    assert Settings.load(path).snippets == snippets
    path.write_text('{"snippets": ["my email"]}', encoding="utf-8")  # strings, not snippets
    assert Settings.load(path).snippets == []


def test_damaged_file_gives_defaults_so_the_app_still_starts(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ not json", encoding="utf-8")
    assert Settings.load(path) == Settings()
    path.write_text("[1, 2]", encoding="utf-8")
    assert Settings.load(path) == Settings()


def test_unknown_keys_and_wrong_types_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"hotkey": "menu", "sounds": "yes", "save_recordings": 1, "future_option": 3}),
                    encoding="utf-8")
    assert Settings.load(path) == Settings(hotkey="menu")


def test_history_is_newest_first_and_skips_damaged_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    add_to_history("first", path=path)
    add_to_history("second — with ünïcödé", path=path)
    with path.open("a", encoding="utf-8") as f:
        f.write("{ broken line\n")
    assert [e["text"] for e in read_history(path)] == ["second — with ünïcödé", "first"]


def test_history_is_trimmed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_KEEP", 5)
    path = tmp_path / "history.jsonl"
    for i in range(12):
        add_to_history(f"entry {i}", path=path)
    assert [e["text"] for e in read_history(path)] == [f"entry {i}" for i in range(11, 6, -1)]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 5


def test_cleanup_settings_and_vocabulary_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    chosen = Settings(cleanup=True, cleanup_model="model-a", cleanup_fallback="model-b",
                      vocabulary=["Claude Code", "தமிழ்", "GitHub"])
    chosen.save(path)
    assert Settings.load(path) == chosen


def test_settings_from_before_the_cleanup_switch_keep_cleanup_on(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"cleanup_model": "model-a"}), encoding="utf-8")  # a 0.1 settings file
    assert Settings.load(path) == Settings(cleanup=True, cleanup_model="model-a")
    path.write_text(json.dumps({"cleanup": False, "cleanup_model": "model-a"}), encoding="utf-8")
    assert Settings.load(path).cleanup is False


def test_a_vocabulary_that_is_not_a_list_of_words_is_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"vocabulary": ["ok", 3], "cleanup_model": 5}), encoding="utf-8")
    assert Settings.load(path) == Settings()


def test_history_keeps_what_was_heard_when_the_cleanup_changed_it(tmp_path):
    path = tmp_path / "history.jsonl"
    add_to_history("Hello, world.", "hello world", path=path)
    add_to_history("Same.", "Same.", path=path)
    newest, oldest = read_history(path)
    assert oldest == {"time": oldest["time"], "text": "Hello, world.", "heard": "hello world"}
    assert "heard" not in newest


def test_the_first_run_flags_are_off_for_a_new_user_and_saved(tmp_path):
    path = tmp_path / "settings.json"
    assert not Settings().welcomed and not Settings().told_about_tray
    Settings(welcomed=True, told_about_tray=True).save(path)
    assert Settings.load(path) == Settings(welcomed=True, told_about_tray=True)


def test_stats_count_words_speed_week_and_streak():
    today = date(2026, 9, 30)
    stats = Stats()
    stats.add("one two three four", 2.0, today - timedelta(days=8))  # outside this week
    stats.add("five six", None, today - timedelta(days=1))  # length unknown: not in the speed
    stats.add("seven eight nine", 1.0, today)
    assert (stats.words, stats.dictations, stats.words_this_week(today)) == (9, 3, 5)
    assert stats.streak(today) == 2 and stats.streak(today + timedelta(days=1)) == 2  # today isn't over yet
    assert stats.streak(today + timedelta(days=2)) == 0
    assert stats.words_per_minute is None  # too little speech to say
    stats.add(" ".join(["word"] * 70), 30.0, today)
    assert stats.words_per_minute == 140  # 77 timed words in 33 s


def test_stats_round_trip_and_start_from_the_history(tmp_path):
    history = tmp_path / "history.jsonl"
    add_to_history("Hello there, world.", path=history)
    add_to_history("Again.", path=history)
    stats = Stats.load(tmp_path / "stats.json", history=history)
    assert (stats.words, stats.dictations, stats.seconds) == (4, 2, 0.0)
    assert stats.days == {date.today().isoformat(): 4}
    stats.add("more words", 1.0, date.today())
    stats.save(tmp_path / "stats.json")
    assert Stats.load(tmp_path / "stats.json", history=history) == stats


def test_damaged_stats_start_again_from_the_history(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text('{"words": "many"}', encoding="utf-8")
    assert Stats.load(path, history=tmp_path / "none.jsonl") == Stats()


def test_stats_keep_a_limited_number_of_days(monkeypatch):
    monkeypatch.setattr(settings, "STATS_DAYS", 3)
    stats = Stats()
    for n in range(5):
        stats.add("word", 1.0, date(2026, 9, 1) + timedelta(days=n))
    assert list(stats.days) == ["2026-09-03", "2026-09-04", "2026-09-05"] and stats.words == 5


def test_before_any_profile_there_is_the_first_one_using_the_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_DIR", tmp_path)
    profiles = Profiles.load(tmp_path / "profiles.json")
    first = profiles.current
    assert [p.id for p in profiles.items] == [settings.FIRST_PROFILE] and first.label == "My profile"
    assert first.settings_file == tmp_path / "settings.json"  # where they were before profiles: nothing moves
    assert first.history_file == tmp_path / "history.jsonl" and first.gateway_file == tmp_path / "gateway.json"


def test_new_profiles_get_their_own_folder_and_a_unique_id(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_DIR", tmp_path)
    profiles = Profiles()
    rahul, rahul_2, tamil = profiles.add("Rahul"), profiles.add(" Rahul "), profiles.add("தமிழ்")
    assert (rahul.id, rahul_2.id, tamil.id) == ("rahul", "rahul-2", "profile")
    assert rahul.name == "Rahul" and tamil.label == "தமிழ்"
    assert rahul.settings_file == tmp_path / "profiles" / "rahul" / "settings.json"
    assert rahul.folder(tmp_path / "bench") == tmp_path / "bench" / "profiles" / "rahul"


def test_profiles_round_trip_and_the_active_one_is_kept(tmp_path):
    path = tmp_path / "profiles.json"
    profiles = Profiles()
    profiles.current.name = "Karthi"
    profiles.active = profiles.add("Rahul").id
    profiles.save(path)
    loaded = Profiles.load(path)
    assert loaded == profiles and loaded.current.name == "Rahul"


def test_removing_a_profile_falls_back_to_the_first_and_the_first_stays(tmp_path):
    profiles = Profiles()
    profiles.active = profiles.add("Rahul").id
    profiles.remove("rahul")
    assert [p.id for p in profiles.items] == ["default"] and profiles.active == "default"
    with pytest.raises(ValueError):
        profiles.remove("default")  # its files are the settings folder itself


def test_a_damaged_or_odd_profiles_file_still_gives_a_usable_list(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text("{ not json", encoding="utf-8")
    assert Profiles.load(path) == Profiles()
    path.write_text(json.dumps({"active": "gone", "profiles": [{"id": "rahul", "name": "Rahul"}, {"id": "../evil"}]}),
                    encoding="utf-8")
    loaded = Profiles.load(path)
    assert [p.id for p in loaded.items] == ["default", "rahul"] and loaded.active == "default"


def test_cloud_speech_models_are_kept_and_bad_ones_ignored(tmp_path):
    path = tmp_path / "settings.json"
    Settings(speech_model="groq", speech_cloud_models={"groq": "whisper-large-v3"}).save(path)
    loaded = Settings.load(path)
    assert loaded.speech_model == "groq" and loaded.speech_cloud_models == {"groq": "whisper-large-v3"}
    path.write_text('{"speech_cloud_models": {"groq": 3}}', encoding="utf-8")
    assert Settings.load(path).speech_cloud_models == {}


@pytest.fixture(autouse=True)
def no_problems_left():
    yield
    settings.LOAD_PROBLEMS.clear()


def test_a_damaged_settings_file_comes_back_from_the_last_good_copy(tmp_path):
    path = tmp_path / "settings.json"
    Settings(vocabulary=["Rflow"], snippets=[{"cue": "my email", "text": "a@b.c", "anywhere": False}]).save(path)
    Settings(vocabulary=["Rflow", "Karthi"], welcomed=True).save(path)  # the first one is the copy now
    path.write_text("{ garbage", encoding="utf-8")  # a full disk, a sync conflict
    restored = Settings.load(path)
    assert restored.vocabulary == ["Rflow"] and restored.snippets[0]["cue"] == "my email"
    assert settings.LOAD_PROBLEMS == ["Your settings file was damaged, so Rflow went back to the copy from before your "
                                      "last change."]
    assert not path.exists() and len(list(tmp_path.glob("settings.json.damaged-*"))) == 1  # kept aside for a look


def test_a_damaged_settings_file_with_no_copy_gives_the_defaults_and_says_so(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("not json", encoding="utf-8")
    assert Settings.load(path) == Settings()
    assert settings.LOAD_PROBLEMS[0].startswith("Your settings file was damaged and there was no good copy")
    assert "settings.json.damaged-" in settings.LOAD_PROBLEMS[0]


def test_trimming_the_history_leaves_no_half_written_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_KEEP", 3)
    path = tmp_path / "history.jsonl"
    for i in range(10):
        add_to_history(f"dictation {i}", path=path)
    assert [e["text"] for e in read_history(path)] == ["dictation 9", "dictation 8", "dictation 7"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3 and not list(tmp_path.glob("*.tmp"))


def test_the_wake_word_is_off_until_chosen_and_ends_with_stop_it(tmp_path):
    s = Settings()
    assert not s.wake_enabled and s.wake_end_phrases == ["stop it"] and s.wake_voice_check
    path = tmp_path / "settings.json"
    Settings(wake_enabled=True, wake_end_phrases=["stop it", "end it"], wake_voice_check=False).save(path)
    loaded = Settings.load(path)
    assert loaded.wake_enabled and loaded.wake_end_phrases == ["stop it", "end it"] and not loaded.wake_voice_check


def test_the_wake_word_setup_is_kept_in_the_profile_and_goes_with_it(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_DIR", tmp_path)
    profiles = Profiles()
    first, rahul = profiles.current, profiles.add("Rahul")
    assert first.wake_file == tmp_path / "wake.json"
    assert rahul.wake_file == tmp_path / "profiles" / "rahul" / "wake.json"
    data = {"keyword": "okay type", "spelling": "okay type", "heard": 9, "voice": {"embedding": [0.1, -0.2]}}
    settings.write_wake(rahul.wake_file, data)
    assert settings.read_wake(rahul.wake_file) == data and not list(rahul.folder().glob("*.tmp"))
    settings.write_wake(rahul.wake_file, None)  # set up again from nothing
    assert settings.read_wake(rahul.wake_file) is None


def test_a_damaged_wake_word_setup_is_ignored(tmp_path):
    path = tmp_path / "wake.json"
    path.write_text("{ not json", encoding="utf-8")
    assert settings.read_wake(path) is None
    path.write_text("[1, 2]", encoding="utf-8")
    assert settings.read_wake(path) is None
