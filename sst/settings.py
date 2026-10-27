"""User settings and dictation history, kept in %APPDATA%\\sst (shared by the installed app and the source checkout)."""
import json
import logging
import os
import re
import shutil
import sys
import time
import winreg
from dataclasses import asdict, dataclass, field, fields
from datetime import date, timedelta
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "sst"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
HISTORY_FILE = CONFIG_DIR / "history.jsonl"
HISTORY_KEEP = 200  # entries shown and kept
STATS_FILE = CONFIG_DIR / "stats.json"
STATS_DAYS = 400  # days of per-day word counts kept (for "this week" and the streak)
PROFILES_FILE = CONFIG_DIR / "profiles.json"
FIRST_PROFILE = "default"

log = logging.getLogger(__name__)
LOAD_PROBLEMS: list[str] = []  # what Settings.load had to do about a damaged file, for the app to tell the user


@dataclass
class Settings:
    hotkey: str = "ctrl+win"
    microphone: str = ""  # a device name from input_device_names(); "" = the Windows default
    sounds: bool = True
    save_recordings: bool = True
    warm_mic: bool = True  # keep the microphone open a few minutes after dictating: instant start, no lost first word
    always_on_mic: bool = True  # keep it open while Rflow runs: 2 s of pre-roll in RAM (the owner's plan, 2026-10-01)
    voice_pipeline: bool = True  # chunks while speaking, then dictionary, formatting, LLM and guard (sst.pipeline)
    format_text: bool = True  # spoken numbers, dates, times and money written as such ("25%", "3:30 PM")
    debug_pipeline: bool = False  # keep every dictation's stages and chunk audio in %LOCALAPPDATA%\sst\debug
    transform_shortcut: str = "double ctrl"  # Text Transform's menu for selected text (sst.transformui); "" = off
    translate_shortcut: str = "ctrl+c+c"  # Translate's popup for the copied text (sst.translateui); "" = off
    translate_to: str = "English"  # the language Translate writes in
    translate_second: str = ""  # for text already in translate_to: this language instead ("" = none)
    # Screen translation (sst.screen): a frame over any app, its Japanese covered with English
    screen_shortcut: str = "ctrl+alt+s"  # starts and stops it from any app ("" = none)
    screen_lens: list[int] = field(default_factory=list, metadata={"items": int})  # the frame's [x, y, w, h], where left
    screen_told: bool = False  # the notice (the text read goes to Google, the cost) was accepted once
    live_target: str = "en"  # live translation (sst.live): what the laptop plays is translated into this language
    live_source: str = "computer"  # what live translation listens to: computer, microphone or both (sst.live SOURCES)
    live_mic_target: str = "ja"  # what the microphone hears is translated into this language
    live_shortcut: str = "ctrl+alt+l"  # starts and stops live translation from any app ("" = none)
    live_bar: list[int] = field(default_factory=list, metadata={"items": int})  # the bar's [x, y, w, h], where left
    live_hide_from_share: bool = True  # the translation bar isn't in screen shares and recordings
    live_speak: bool = False  # the translation spoken aloud too, by a voice on this laptop (sst.live.voice)
    live_speak_speed: float = 1.0  # how fast it speaks (faster by itself when it falls behind)
    live_duck: float = 0.1  # how loud the other apps stay while it speaks (0.0: barely, 1.0: as they are; sst.live.ducking)
    live_voice: str = "en_US-danny-low"  # the voice (sst.live.voice VOICES)
    live_save: bool = False  # the cost saver: audio goes to Google only while someone talks (sst.live LiveConfig.save)
    live_told: bool = False  # the notice (audio goes to Google, the cost) was accepted once
    # Meeting mode (sst.meeting): one switch per part, all started and stopped with the meeting
    meeting_translate: bool = True  # live translation's bar runs during the meeting
    meeting_record: bool = True  # the others (what the laptop plays) and you (the microphone), saved on this laptop
    meeting_minutes: bool = True  # after it: transcript with speakers, minutes, decisions, action items (needs the recording)
    meeting_screen: bool = True  # the meeting window's changed frames (slides) given to the minutes
    meeting_shortcut: str = "ctrl+alt+m"  # starts and stops a meeting from any app ("" = none)
    meeting_told: bool = False  # the notice (people should know it's recorded; audio and frames go to Google) accepted once
    # The Gemini key is on Google's free tier: the cost card says "$0 billed (would cost $X)". No key can tell its tier,
    # and nothing sets it yet (TODO: the owner decides how it is asked or found out).
    gemini_free_tier: bool = False
    # Start with your voice (sst.wake, sst.wakeui): the keyword starts a hands-free dictation, an end phrase types it.
    # The keyword itself, learned in its 10-try setup, is kept apart: Profile.wake_file.
    wake_enabled: bool = False
    wake_end_phrases: list[str] = field(default_factory=lambda: ["stop it"])  # never "done": said in dictation too often
    wake_voice_check: bool = True  # only the voice the setup learned starts it
    # Listen on a Bluetooth headset's microphone though it isn't kept open for dictation (always_on_mic off): it drops the
    # headset to call quality while it listens. With always_on_mic it's open anyway, and the wake word listens there.
    wake_bluetooth: bool = False
    voice_commands: bool = True  # hold the dictation key and say "make it concise" (sst.commands)
    command_phrases: dict[str, str] = field(default_factory=dict)  # transform -> the user's own phrases ("a, b"); else defaults
    # Snippets (sst.snippets): {"cue": "my email", "text": "xyz@gmail.com", "anywhere": false}, typed when said
    snippets: list[dict] = field(default_factory=list, metadata={"items": dict})
    transforms: list[str] = field(default_factory=lambda: ["concise", "professional", "bullets", "actions"])  # in the menu
    # The writer, for a Japanese email's line that names them (株式会社△△の□□です。); empty: no such line, never invented
    signature_name: str = ""
    signature_company: str = ""
    raw_audio: bool = False  # ask Windows for the microphone without its voice effects (noise suppression, gating)
    speech_model: str = "parakeet"  # the speech recognition model, a key of sst.engines.SPEECH_MODELS
    speech_language: str = ""  # for models that know many languages (Whisper): "" = detected, or a code such as "ta"
    speech_cloud_models: dict[str, str] = field(default_factory=dict)  # cloud provider -> its model chosen for speech
    speech_server_model: str = ""  # the model on the user's own server (its address and key: GatewayConfig.speech_server)
    cleanup: bool = False  # clean up the text with an AI model before typing it
    cleanup_model: str = ""  # a model id on the user's endpoint (sst.gateway)
    cleanup_fallback: str = ""  # optional backup model, tried when the first one fails
    vocabulary: list[str] = field(default_factory=list)  # the user's names and terms, for the cleanup
    welcomed: bool = False  # the first-run welcome was completed (or skipped)
    told_about_tray: bool = False  # closing the window keeps Rflow in the tray; said once

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "Settings":
        """Settings from disk, so the app always starts: a damaged file is kept aside and the last good copy (written at
        every save) used instead; with none, the defaults. What happened goes to LOAD_PROBLEMS for the user."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError) as e:
            log.warning("Unreadable settings file %s: %s", path, e)
            data = _recover(path)
            if data is None:
                return cls()
        defaults = cls()
        values = {}
        for f in fields(cls):  # keep only known keys with the right type
            default = getattr(defaults, f.name)
            value = data.get(f.name, default) if isinstance(data, dict) else default
            items = f.metadata.get("items", str)  # a list's items: strings, unless the field says otherwise
            ok = isinstance(value, type(default)) and (not isinstance(value, list) or all(isinstance(v, items) for v in value)) \
                and (not isinstance(value, dict) or all(isinstance(v, str) for v in (*value, *value.values())))
            values[f.name] = value if ok else default
        if isinstance(data, dict) and "cleanup" not in data and values["cleanup_model"]:
            values["cleanup"] = True  # settings from before the on/off switch: a chosen model meant "on"
        return cls(**values)

    def save(self, path: Path = SETTINGS_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        if path.exists():  # the last good copy, should this file ever be damaged (a full disk, a sync conflict)
            try:
                shutil.copyfile(path, _backup(path))
            except OSError as e:
                log.warning("Couldn't keep a copy of the settings: %s", e)
        tmp.replace(path)  # atomic: a crash mid-write never leaves a half-written file


def _backup(path: Path) -> Path:
    return path.with_name(path.name + ".bak")


def _recover(path: Path) -> dict | None:
    """A damaged settings file: kept aside for a look, and the last good copy's values, or None (the defaults)."""
    aside = path.with_name(f"{path.name}.damaged-{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        path.replace(aside)  # the next save writes a good file where it was
    except OSError:
        aside = None
    try:
        data = json.loads(_backup(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            LOAD_PROBLEMS.append("Your settings file was damaged, so Rflow went back to the copy from before your last "
                                 "change.")
            return data
    except (OSError, ValueError):
        pass
    kept = f" The damaged file is kept as {aside.name}." if aside else ""
    LOAD_PROBLEMS.append("Your settings file was damaged and there was no good copy, so Rflow started with its defaults, "
                         f"Your words and snippets too.{kept}")
    return None


def add_to_history(text: str, heard: str | None = None, path: Path = HISTORY_FILE) -> None:
    """`text` as typed; `heard` is the recognizer's text when the cleanup changed it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text}
    if heard is not None and heard != text:
        entry["heard"] = heard
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_history(path: Path = HISTORY_FILE) -> list[dict]:
    """Newest first. Damaged lines are skipped; the file is trimmed to HISTORY_KEEP entries."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
            if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                entries.append(entry)
        except ValueError:
            continue
    if len(lines) > HISTORY_KEEP * 2:  # trim now and then, not on every write
        entries = entries[-HISTORY_KEEP:]
        tmp = path.with_name(path.name + ".tmp")  # a new file, then a swap: a crash or full disk can't empty it
        tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
        tmp.replace(path)
    return entries[::-1][:HISTORY_KEEP]


@dataclass
class Stats:
    """Totals for the Home page. Kept apart from the history, which only keeps the last HISTORY_KEEP dictations."""
    words: int = 0
    dictations: int = 0
    timed_words: int = 0  # words of the dictations whose length is known, for the speaking speed
    seconds: float = 0.0
    days: dict[str, int] = field(default_factory=dict)  # "2026-09-30" -> words dictated that day

    def add(self, text: str, seconds: float | None, day: date) -> None:
        words = len(text.split())
        self.words += words
        self.dictations += 1
        if seconds:
            self.timed_words += words
            self.seconds += seconds
        key = day.isoformat()
        self.days[key] = self.days.get(key, 0) + words
        if len(self.days) > STATS_DAYS:
            self.days = dict(sorted(self.days.items())[-STATS_DAYS:])

    @property
    def words_per_minute(self) -> int | None:
        """Speaking speed, once there is enough to say (half a minute of speech)."""
        return round(self.timed_words / self.seconds * 60) if self.seconds >= 30 else None

    def words_this_week(self, today: date) -> int:
        return sum(self.days.get((today - timedelta(days=n)).isoformat(), 0) for n in range(7))

    def streak(self, today: date) -> int:
        """Days in a row with dictation, up to today (or up to yesterday: today isn't over yet)."""
        day = today if today.isoformat() in self.days else today - timedelta(days=1)
        count = 0
        while day.isoformat() in self.days:
            count, day = count + 1, day - timedelta(days=1)
        return count

    @classmethod
    def load(cls, path: Path = STATS_FILE, history: Path = HISTORY_FILE) -> "Stats":
        """The saved totals; the first time, a start from the history (its dictations have no length yet)."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(words=int(data["words"]), dictations=int(data["dictations"]), timed_words=int(data["timed_words"]),
                       seconds=float(data["seconds"]), days={str(k): int(v) for k, v in dict(data["days"]).items()})
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("Starting the stats again; unreadable %s: %s", path, e)
        stats = cls()
        for entry in reversed(read_history(history)):
            try:
                stats.add(entry["text"], None, date.fromisoformat(entry.get("time", "")[:10]))
            except ValueError:
                continue
        return stats

    def save(self, path: Path = STATS_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)


# ---- profiles: people sharing this computer, each with their own settings, words, AI provider, history and stats

@dataclass
class Profile:
    id: str
    name: str = ""  # "" = not named yet

    @property
    def label(self) -> str:
        return self.name or "My profile"

    def folder(self, root: Path | None = None) -> Path:
        """This profile's part of a data folder (the settings folder unless another is given). The first profile keeps
        the files from before profiles where they were, so nothing moves and an older Rflow still finds them."""
        root = root or CONFIG_DIR
        return root if self.id == FIRST_PROFILE else root / "profiles" / self.id

    @property
    def settings_file(self) -> Path:
        return self.folder() / "settings.json"

    @property
    def meetings_folder(self) -> Path:
        """Meeting mode's meetings, one folder each (sst.meeting): the recording, the frames, the minutes."""
        return self.folder() / "meetings"

    @property
    def history_file(self) -> Path:
        return self.folder() / "history.jsonl"

    @property
    def stats_file(self) -> Path:
        return self.folder() / "stats.json"

    @property
    def usage_file(self) -> Path:
        """What the profile's Gemini use cost (sst.usage): tokens and seconds a day, by feature and model. No text."""
        return self.folder() / "usage.json"

    @property
    def gateway_file(self) -> Path:
        return self.folder() / "gateway.json"

    @property
    def wake_file(self) -> Path:
        """The wake word's setup (sst.wake.WakeSetup): the keyword, its spelling, the voice print's numbers. No audio."""
        return self.folder() / "wake.json"


def read_wake(path: Path) -> dict | None:
    """The saved wake word setup as its JSON (sst.wake.WakeSetup.from_json reads it), or None: none yet, or damaged
    (set up again)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        log.warning("Ignoring the unreadable wake word setup %s: %s", path, e)
        return None
    return data if isinstance(data, dict) else None


def write_wake(path: Path, data: dict | None) -> None:
    """Keep a wake word setup (WakeSetup.to_json: words and numbers only, never audio), or remove it (None)."""
    if data is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


@dataclass
class Profiles:
    items: list[Profile] = field(default_factory=lambda: [Profile(FIRST_PROFILE)])
    active: str = FIRST_PROFILE

    @property
    def current(self) -> Profile:
        return next((p for p in self.items if p.id == self.active), self.items[0])

    def get(self, profile_id: str) -> Profile | None:
        return next((p for p in self.items if p.id == profile_id), None)

    def add(self, name: str) -> Profile:
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24] or "profile"
        profile_id, n = base, 2
        while self.get(profile_id) or profile_id == FIRST_PROFILE:
            profile_id, n = f"{base}-{n}", n + 1
        profile = Profile(profile_id, name.strip())
        self.items.append(profile)
        return profile

    def remove(self, profile_id: str) -> None:
        """Every profile but the first can go (the first one's files are the settings folder itself)."""
        if profile_id == FIRST_PROFILE:
            raise ValueError("The first profile can't be deleted; rename it instead.")
        self.items = [p for p in self.items if p.id != profile_id]
        if self.active == profile_id:
            self.active = FIRST_PROFILE

    @classmethod
    def load(cls, path: Path = PROFILES_FILE) -> "Profiles":
        """The saved profiles; before any were made (or if the file is damaged), just the first one."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items = [Profile(str(p["id"]), str(p.get("name") or "")) for p in data["profiles"]
                     if re.fullmatch(r"[a-z0-9-]+", str(p["id"]))]
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("Ignoring unreadable %s: %s", path, e)
            return cls()
        if not any(p.id == FIRST_PROFILE for p in items):
            items.insert(0, Profile(FIRST_PROFILE))
        profiles = cls(items, str(data.get("active") or FIRST_PROFILE))
        profiles.active = profiles.current.id  # an unknown active profile falls back to the first
        return profiles

    def save(self, path: Path = PROFILES_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"active": self.active, "profiles": [asdict(p) for p in self.items]}, indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


# ---- start with Windows: a value under HKCU\...\Run (the installer's "start when I sign in" option uses the same one)

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "Rflow"


def can_start_with_windows() -> bool:
    return bool(getattr(sys, "frozen", False))  # only the installed app has a fixed .exe to start


def starts_with_windows() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, RUN_VALUE)
            return True
    except OSError:
        return False


def set_start_with_windows(enabled: bool) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, f'"{sys.executable}" --startup')  # starts quietly
        else:
            try:
                winreg.DeleteValue(key, RUN_VALUE)
            except FileNotFoundError:
                pass
