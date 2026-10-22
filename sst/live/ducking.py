"""While live translation speaks, the other apps stay lower for the whole session, at the level the user chose.

    Ducker (its own thread, every STEP, from start() to stop()) -> each other app's sound, on every output: its volume in
    Windows' volume mixer (ISimpleAudioVolume) x depth, over DOWN seconds; stop() puts them back at once

The owner's choice (2026-10-08): not lowered for each sentence and back between them, which pumped the meeting up and
down, but low the whole time while Speak the translation is on, and back when live translation or speaking stops, or
Rflow quits. Rflow's own sound (the voice) and Windows' system sounds are left alone, and only apps playing are lowered
(one that starts later is found within RESCAN seconds). Windows remembers each app's mixer volume, so what is lowered
is written down first (RECORD, on this PC) and put back at the next start if Rflow ended while it was low; an app whose
volume the user changes meanwhile keeps the user's. Rflow hears the lowered apps lower too: the computer's sound has
its own level (sst.live.level) and Gemini translates speech down to -70 dBFS (measured), so the meeting is still
translated. Even at 0%: an app at a mixer volume of exactly 0 is silence in the mix Rflow hears, so 0% is INAUDIBLE
instead (phase 37 measured an app at 0.03% translated in full, and nobody hears it).
"""
import ctypes
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable
from ctypes import POINTER, byref, c_float, c_int, c_void_p, wintypes
from dataclasses import dataclass
from pathlib import Path

from sst.live.wasapi import _CLSCTX_ALL, _GUID, E_RENDER, _com, _enumerator, _method, _release

STEP = 0.02  # seconds between steps
DOWN = 0.15  # seconds from full volume to silence (a smaller step down takes less)
UP = 0.6  # and back up, when the user raises the level: gently
RESCAN = 0.5  # seconds between looks for apps that started playing while the others are lowered
INAUDIBLE = 0.0003  # the lowest mixer volume Rflow sets: 0% for the user, still heard by live translation (see above)
CHANGED = 0.02  # a volume this far from what Rflow set was changed by the user: theirs from now on
KEEP = 7 * 86_400  # seconds a lowered app that isn't running is remembered, to be put back when it runs again
RECORD = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "sst" / "lowered-apps.json"

log = logging.getLogger(__name__)


@dataclass
class _Lowered:
    sound: object  # an app's sound: key, name, volume(), set_volume(v), close()
    original: float  # its volume before Rflow lowered it
    now: float  # what Rflow set it to last


class Ducker:
    """start() keeps the other apps at `depth` on its own thread; stop() puts every app back at once. `find(playing)`
    lists the apps' sounds (all of them, or only those playing now): the real ones by default, fakes in the tests.
    `depth`: how loud the other apps stay, 0.0 to 1.0 (1.0: as they are; 0.0: INAUDIBLE)."""

    def __init__(self, depth: float, find: Callable[[bool], list] | None = None, record: Path = RECORD,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self.depth = _bounded(depth)
        self._find = find or apps
        self.record, self._clock, self._wall = record, clock, wall
        self._lowered: dict[str, _Lowered] = {}
        self._level = 1.0  # where the apps are: 1.0 as the user left them, `depth` lowered
        self._scanned = -math.inf
        self._ticked: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._said: set[str] = set()  # the apps the log has named

    def set_depth(self, depth: float) -> None:
        self.depth = _bounded(depth)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="live-ducker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(3)

    def _run(self) -> None:
        _com()
        try:
            put_back(self._find, self.record, self._wall)  # left low by a Rflow that ended while they were low
            while not self._stop.wait(STEP):
                self.tick()
        except Exception:
            log.exception("Lowering the other apps failed")
        finally:
            self.release()
            ctypes.WinDLL("ole32").CoUninitialize()

    def tick(self) -> None:
        """One step: the apps playing (new ones too) go towards `depth`, down quickly, up gently."""
        now = self._clock()
        step = 0.0 if self._ticked is None else min(now - self._ticked, 0.25)
        self._ticked = now
        if self.depth < 1.0 and now - self._scanned >= RESCAN:
            self._scanned = now
            self._scan()
        if self._level > self.depth:
            self._level = max(self.depth, self._level - step / DOWN)
        else:
            self._level = min(self.depth, self._level + step / UP)
        self._apply()
        if self.depth >= 1.0 and self._level >= 1.0 and self._lowered:
            self._forget()  # "Unchanged" chosen meanwhile: all back up

    def release(self) -> None:
        """Every app back to its own volume at once (stopping)."""
        self._level = 1.0
        self._apply()
        self._forget()

    def _scan(self) -> None:
        try:
            found = self._find(True)
        except OSError as e:
            log.debug("The apps' sound couldn't be listed: %s", e)
            return
        added = []
        for sound in found:
            try:
                volume = sound.volume() if sound.key not in self._lowered else 0.0
            except OSError:
                volume = 0.0
            if volume <= 0.001:  # already lowered, silent anyway, or gone
                sound.close()
                continue
            self._lowered[sound.key] = _Lowered(sound, volume, volume)
            added.append(sound)
        if added:
            self._write()
            names = {_app_name(sound.name) for sound in added} - self._said
            if names:  # once each, not for every sentence
                self._said |= names
                log.info("While live translation speaks, lowered to %d%%: %s", round(self.depth * 100),
                         ", ".join(sorted(names)))

    def _apply(self) -> None:
        for key, item in list(self._lowered.items()):
            want = item.original if self._level >= 1.0 else max(item.original * self._level, INAUDIBLE)
            if want == item.now:  # where it is already: Windows isn't asked again
                continue
            try:
                if abs(item.sound.volume() - item.now) > CHANGED:  # the user moved it meanwhile: left as they set it
                    log.info("%s's volume was changed while it was lowered: left as it is", _app_name(item.sound.name))
                    self._drop(key)
                    continue
                item.sound.set_volume(want)
                item.now = want
            except OSError:  # the app ended
                self._drop(key)

    def _drop(self, key: str) -> None:
        item = self._lowered.pop(key)
        item.sound.close()
        self._write()

    def _forget(self) -> None:
        for item in self._lowered.values():
            item.sound.close()
        self._lowered.clear()
        self._write()

    def _write(self) -> None:
        """The record: the apps lowered now, and any left low before that weren't found again yet."""
        keys = set(self._lowered)
        entries = [e for e in _read(self.record) if e.get("key") not in keys]
        entries += [{"key": key, "name": item.sound.name, "original": item.original, "depth": self.depth,
                     "at": self._wall()} for key, item in self._lowered.items()]
        lowered = [e for e in entries if e["key"] in keys or e.get("left")]
        _save(self.record, lowered)


def put_back(find: Callable[[bool], list], record: Path = RECORD, wall: Callable[[], float] = time.time) -> int:
    """Apps a Rflow that ended while live translation spoke left lowered: back to their own volume (where nobody changed them
    since). Those not running now are kept for KEEP seconds. How many were put back."""
    entries = _read(record)
    if not entries:
        return 0
    try:
        sounds = find(False)
    except OSError as e:
        log.debug("The apps' sound couldn't be listed: %s", e)
        return 0
    back, kept = 0, []
    try:
        for entry in entries:
            matching = [s for s in sounds if s.name == entry.get("name")]
            for sound in matching:
                try:
                    original, volume = float(entry["original"]), sound.volume()
                    if original * float(entry.get("depth", 0)) - CHANGED <= volume <= original - CHANGED:
                        sound.set_volume(original)
                        back += 1
                        log.info("%s was left lowered: back to %d%%", _app_name(sound.name), round(original * 100))
                except (OSError, KeyError, TypeError, ValueError):
                    continue
            if not matching and wall() - float(entry.get("at", 0)) < KEEP:
                kept.append({**entry, "left": True})
    finally:
        for sound in sounds:
            sound.close()
    _save(record, kept)
    return back


def put_back_left(record: Path = RECORD) -> None:
    """At Rflow's start, on a thread of its own (COM): put_back() with the real apps."""
    def work() -> None:
        _com()
        try:
            put_back(apps, record)
        except Exception:
            log.exception("Putting back apps left lowered failed")
        finally:
            ctypes.WinDLL("ole32").CoUninitialize()
    if _read(record):
        threading.Thread(target=work, name="put-back-apps", daemon=True).start()


def _bounded(depth: float) -> float:
    return min(max(float(depth), 0.0), 1.0)


def _app_name(name: str) -> str:
    """The program in a session's identifier ("...\\chrome.exe%b{...}"), for the log."""
    found = re.search(r"([^\\|%]+\.exe)", name, re.IGNORECASE)
    return found.group(1) if found else "an app"


def _read(record: Path) -> list[dict]:
    try:
        entries = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def _save(record: Path, entries: list[dict]) -> None:
    try:
        if entries:
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps(entries), encoding="utf-8")
        else:
            record.unlink(missing_ok=True)
    except OSError as e:
        log.warning("The record of lowered apps couldn't be written: %s", e)


# ---- the apps' sound, through Core Audio (ctypes)

_IID_SESSION_MANAGER2 = _GUID.of("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F")
_IID_SESSION_CONTROL2 = _GUID.of("BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D")
_IID_SIMPLE_VOLUME = _GUID.of("87CE5498-68D6-44E5-9215-6DA47EF883D8")
_ACTIVE = 1  # DEVICE_STATE_ACTIVE, and AudioSessionStateActive: a session playing now


class _AppSound:
    """One app's sound on one output (an audio session): `key` this run of it, `name` the app (Windows remembers the
    volume by it). Used on the thread that found it."""

    def __init__(self, volume: c_void_p, key: str, name: str, pid: int = 0):
        self._volume, self.key, self.name, self.pid = volume, key, name, pid

    def volume(self) -> float:
        value = c_float()
        _method(self._volume, 4, POINTER(c_float))(byref(value))  # GetMasterVolume
        return value.value

    def set_volume(self, value: float) -> None:
        _method(self._volume, 3, c_float, c_void_p)(c_float(min(max(value, 0.0), 1.0)), None)  # SetMasterVolume

    def close(self) -> None:
        _release(self._volume)
        self._volume = c_void_p()


def apps(playing: bool = True, own: int | None = None) -> list[_AppSound]:
    """The other apps' sound on every active output: Rflow's own and Windows' system sounds left out; `playing`: only
    those playing now. COM must be set up on the calling thread."""
    own = os.getpid() if own is None else own
    enumerator, devices, found = _enumerator(), c_void_p(), []
    try:
        _method(enumerator, 3, c_int, wintypes.DWORD, POINTER(c_void_p))(E_RENDER, _ACTIVE, byref(devices))
        count = wintypes.UINT()
        _method(devices, 3, POINTER(wintypes.UINT))(byref(count))  # IMMDeviceCollection::GetCount
        for i in range(count.value):
            device = c_void_p()
            try:
                _method(devices, 4, wintypes.UINT, POINTER(c_void_p))(i, byref(device))  # Item
                found += _sessions(device, playing, own)
            except OSError as e:
                log.debug("An output's apps couldn't be listed: %s", e)
            finally:
                _release(device)
    finally:
        _release(devices)
        _release(enumerator)
    return found


def _sessions(device, playing: bool, own: int) -> list[_AppSound]:
    manager, sessions, out = c_void_p(), c_void_p(), []
    try:
        _method(device, 3, POINTER(_GUID), wintypes.DWORD, c_void_p, POINTER(c_void_p))(
            byref(_IID_SESSION_MANAGER2), _CLSCTX_ALL, None, byref(manager))  # IMMDevice::Activate
        _method(manager, 5, POINTER(c_void_p))(byref(sessions))  # GetSessionEnumerator
        count = c_int()
        _method(sessions, 3, POINTER(c_int))(byref(count))
        for i in range(count.value):
            control, control2, volume = c_void_p(), c_void_p(), c_void_p()
            try:
                _method(sessions, 4, c_int, POINTER(c_void_p))(i, byref(control))  # GetSession
                state = c_int()
                _method(control, 3, POINTER(c_int))(byref(state))  # GetState
                if playing and state.value != _ACTIVE:
                    continue
                _query(control, _IID_SESSION_CONTROL2, control2)
                if _method(control2, 15)() == 0:  # IsSystemSoundsSession: S_OK means it is
                    continue
                pid = wintypes.DWORD()
                _method(control2, 14, POINTER(wintypes.DWORD))(byref(pid))  # GetProcessId
                if pid.value == own:
                    continue
                key, name = _text(control2, 13), _text(control2, 12)  # GetSessionInstanceIdentifier, -Identifier
                _query(control, _IID_SIMPLE_VOLUME, volume)
                out.append(_AppSound(volume, key, name, pid.value))
                volume = c_void_p()  # the sound keeps it
            except OSError as e:
                log.debug("An app's sound was skipped: %s", e)
            finally:
                _release(volume)
                _release(control2)
                _release(control)
    finally:
        _release(sessions)
        _release(manager)
    return out


def _query(obj, iid: _GUID, out: c_void_p) -> None:
    _method(obj, 0, POINTER(_GUID), POINTER(c_void_p))(byref(iid), byref(out))  # QueryInterface


def _text(obj, index: int) -> str:
    raw = ctypes.c_wchar_p()
    _method(obj, index, POINTER(ctypes.c_wchar_p))(byref(raw))
    value = raw.value or ""
    ctypes.WinDLL("ole32").CoTaskMemFree(raw)
    return value
