"""One meeting: the parts the user switched on, started and stopped together, then the minutes written after it.

    start():  meeting.json, the recorder (others.wav, you.wav), the window's frames, live translation  -> "recording"
    stop():   all of them stopped, meeting.json updated; with minutes on, the minutes on a worker thread
              -> "writing minutes" -> "done" (result) or "failed" (error); without, "done" at once
    the meeting window gone for GONE_S (apps.alive false): stopped by itself

In person (window apps.IN_PERSON): the microphone alone (room.wav), no frames (there's no window), and it runs until
the user stops it.

The rules between the switches: no sound, no minutes (they're made from the recording); frames are only for the
minutes, so without them none are kept. meeting.json holds numbers and names only (the app, the window's title, times,
switches, counts), never a word said: those stay in the recording and the minutes. A recording is never deleted here,
even when the minutes fail: the user can try again from it.
"""
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

META = "meeting.json"
GONE_S = 10.0  # the meeting window missing this long: the meeting ended (a short gap is Teams swapping its windows)
WATCH_S = 1.0  # seconds between looks at the meeting window
RECORDING, WRITING, DONE, FAILED = "recording", "writing minutes", "done", "failed"


@dataclass
class MeetingOptions:
    translate: bool = True  # sst.live's bar, started and stopped with the meeting
    record: bool = True  # others.wav and you.wav on this laptop
    minutes: bool = True  # after the meeting, from the recording
    screen: bool = True  # the window's changed frames, given to the minutes

    def effective(self) -> "MeetingOptions":
        """The switches as they work: minutes need the recording, frames are only kept for the minutes."""
        minutes = self.minutes and self.record
        return replace(self, minutes=minutes, screen=self.screen and minutes)

    def anything(self) -> bool:
        on = self.effective()
        return on.translate or on.record


@dataclass(frozen=True)
class MeetingCost:
    total: float  # USD: translation + minutes_estimate
    translation: float  # live translation's spend so far
    minutes_estimate: float  # what the minutes will cost for the meeting so far (0 with minutes off)
    seconds: float
    frames: int


class MeetingSession:
    """See the module docstring. recorder_factory(folder) -> a Recorder; frames_factory(window, folder) -> a
    WindowFrames (folder: the meeting's); live: an object with start()/stop()/spend() (spend() -> sst.live Spend or
    None), or None; summarise(folder) -> MinutesResult, raising on failure. on_change() is called from any thread when
    state or result change. alive(window) and estimate(seconds, frames) default to sst.meeting.apps and .minutes."""

    def __init__(self, window, options: MeetingOptions, folder: Path, *, recorder_factory: Callable, frames_factory: Callable,
                 live=None, summarise: Callable[[Path], object], clock: Callable[[], float] = time.monotonic,
                 alive: Callable[[object], bool] | None = None, estimate: Callable[[float, int], float] | None = None,
                 watch: float = WATCH_S):
        from sst.meeting.apps import in_person
        self.in_person = in_person(window)
        self.window, self.folder, self.options = window, Path(folder), options.effective()
        if self.in_person:
            self.options = replace(self.options, screen=False)  # no window to look at
        self._recorder_factory, self._frames_factory = recorder_factory, frames_factory
        self.live, self._summarise = live, summarise
        self._clock, self._alive, self._estimate, self._watch_s = clock, alive, estimate, watch
        self.recorder = None
        self.frames = None
        self.state = ""
        self.result = None
        self.error = ""
        self.on_change: Callable[[], None] | None = None
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._started_wall = ""
        self._translation = 0.0  # live's spend, kept once it stops (a stopped bar has no session to ask)
        self._gone_since: float | None = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._watcher: threading.Thread | None = None
        self._worker: threading.Thread | None = None

    # ---- running

    def start(self) -> None:
        """Start the parts switched on; a part that fails stops the ones started and raises (nothing half-running)."""
        if not self.options.anything():
            raise ValueError("Switch on live translation or recording first.")
        self.folder.mkdir(parents=True, exist_ok=True)
        self._started_at, self._started_wall = self._clock(), datetime.now().isoformat(timespec="seconds")
        try:
            if self.options.record:
                self.recorder = self._recorder_factory(self.folder)
                self.recorder.start()
            if self.options.screen:
                self.frames = self._frames_factory(self.window, self.folder)
                self.frames.start()
            if self.options.translate and self.live is not None:
                self.live.start()
        except Exception as e:
            self._stop_parts()
            self._set(FAILED, error=str(e))
            raise
        self._write_meta()
        self._set(RECORDING)
        if self._watch_s > 0 and not self.in_person:
            self._watcher = threading.Thread(target=self._watch, name="meeting-window", daemon=True)
            self._watcher.start()

    def stop(self) -> None:
        """Stop the parts, then (minutes on) write the minutes on a worker thread. Any thread; only the first counts."""
        with self._lock:
            if self.state != RECORDING:
                return
            self.state = WRITING if self.options.minutes else DONE  # taken under the lock: a second stop does nothing
        self._stopped.set()
        self._stop_parts()
        self._ended_at = self._clock()
        self._write_meta()
        self._changed()
        if not self.options.minutes:
            return
        self._worker = threading.Thread(target=self._write_minutes, name="meeting-minutes", daemon=True)
        self._worker.start()

    def wait(self, timeout: float | None = None) -> None:
        """Until the minutes are written (tests, and the app quitting)."""
        if self._worker is not None:
            self._worker.join(timeout)

    def check(self, now: float | None = None) -> None:
        """One look at the meeting window: gone for GONE_S, the meeting is stopped."""
        if self.state != RECORDING or self.in_person:  # in person: no window, it runs until the user stops it
            return
        now = self._clock() if now is None else now
        if self._window_alive():
            self._gone_since = None
        elif self._gone_since is None:
            self._gone_since = now
        elif now - self._gone_since >= GONE_S:
            log.info("Meeting: the %s window closed, so the meeting stopped", getattr(self.window, "app", "meeting"))
            self.stop()

    # ---- what it shows

    @property
    def seconds(self) -> float:
        if self.recorder is not None and self.recorder.seconds:
            return self.recorder.seconds  # what was heard, the minutes' length
        if self._started_at is None:
            return 0.0
        return max(0.0, (self._ended_at if self._ended_at is not None else self._clock()) - self._started_at)

    @property
    def frame_count(self) -> int:
        return len(self.frames.frames) if self.frames is not None else 0

    def cost(self) -> MeetingCost:
        seconds, frames = self.seconds, self.frame_count
        translation = self._live_cost() if self.state == RECORDING else self._translation
        minutes = 0.0
        if self.options.minutes:
            try:
                minutes = self._estimator()(seconds, frames)
            except Exception as e:  # a cost figure never stops the meeting
                log.warning("Meeting: the minutes' estimate: %s", e)
        return MeetingCost(total=translation + minutes, translation=translation, minutes_estimate=minutes,
                           seconds=seconds, frames=frames)

    # ---- inside

    def _stop_parts(self) -> None:
        if self.live is not None and self.options.translate:
            self._translation = self._live_cost()
        for part in (self.recorder, self.frames, self.live if self.options.translate else None):
            if part is None:
                continue
            try:
                part.stop()
            except Exception as e:  # each part stops whatever the others do
                log.warning("Meeting: stopping %s: %s", type(part).__name__, e)

    def _live_cost(self) -> float:
        if self.live is None or not self.options.translate:
            return self._translation
        try:
            spend = self.live.spend()
        except Exception as e:
            log.warning("Meeting: live translation's cost: %s", e)
            return self._translation
        if spend is not None:
            self._translation = float(spend.cost)
        return self._translation

    def _write_minutes(self) -> None:
        try:
            result = self._summarise(self.folder)
        except Exception as e:
            log.warning("Meeting minutes failed: %s", e)
            self._set(FAILED, error=str(e) or type(e).__name__)
            return
        self._set(DONE, result=result)

    def _watch(self) -> None:
        while not self._stopped.wait(self._watch_s):
            try:
                self.check()
            except Exception as e:  # a failed look is not a closed window
                log.warning("Meeting: looking at the window: %s", e)

    def _window_alive(self) -> bool:
        alive = self._alive
        if alive is None:
            from sst.meeting.apps import alive  # lazily: the window rules aren't needed until a meeting runs
        return bool(alive(self.window))

    def _estimator(self) -> Callable[[float, int], float]:
        if self._estimate is None:
            from sst.meeting.minutes import estimate
            tracks = 1 if self.in_person else 2  # in person, the microphone alone
            self._estimate = lambda seconds, frames: estimate(seconds, frames, tracks=tracks)
        return self._estimate

    def _write_meta(self) -> None:
        """meeting.json: numbers and names only, never words said (the minutes and the recording hold those)."""
        meta = {"app": getattr(self.window, "app", ""), "title": getattr(self.window, "title", ""),
                "started": self._started_wall, "seconds": round(self.seconds, 1), "options": asdict(self.options),
                "frames": self.frame_count, "in_person": self.in_person}
        try:
            (self.folder / META).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.warning("Meeting: couldn't write %s: %s", META, e)

    def _set(self, state: str, *, result=None, error: str = "") -> None:
        with self._lock:
            self.state = state
            if result is not None:
                self.result = result
            if error:
                self.error = error
        self._changed()

    def _changed(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception as e:
                log.warning("Meeting: on_change: %s", e)
