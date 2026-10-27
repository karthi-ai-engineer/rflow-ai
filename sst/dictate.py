"""Dictate into any app: put the cursor in a text box, press the hotkey, speak, and the text
is typed where the cursor is.

  hold the hotkey      record while held; let go to stop    (push-to-talk)
  tap the hotkey       start recording; tap again to stop   (hands-free)
  Ctrl+Win+Space       hands-free too, as in Wispr Flow
  Esc while recording  cancel, nothing is typed

`Dictation` holds the logic and is shared by the console command (`run` below), the tray app
(sst.app) and the tests. Transcription and typing run on a worker thread, so a new recording
can start right away.

With a voice pipeline (sst.pipeline.session.VoicePipeline), each recording is a session fed while the key is held:
chunks cut at pauses go to the speech engine while the user is still speaking, and after the key is let go only the
last chunk and the text stages (dictionary, formatting, LLM, guard) remain. Without one, the whole recording is
transcribed at once (the classic way, and the tests' fake recorders).
"""
import ctypes
import logging
import queue
import subprocess
import threading
import time
import winsound
from collections.abc import Callable, Sequence

import numpy as np

from sst import snippets
from sst.audio import Recorder, Take, save_recording
from sst.hotkey import HotkeyListener, parse_hotkey
from sst.paste import paste_text, settle
from sst.pipeline import streaming
from sst.snippets import Snippet

HOLD_SECONDS = 0.4   # key held longer than this = push-to-talk; a quicker tap = hands-free
MIN_SECONDS = 0.3    # shorter recordings are treated as accidental presses
# No length limit to speak of (the owner's wish of 2026-10-27): the pipeline sends 20 s chunks as the user speaks and
# skips the silence, a whole recording goes to an engine in pieces (Parakeet's MAX_PIECE_SECONDS, cloud.WHOLE_PIECE_SECONDS)
# and the AI cleanup gets a long text in pieces (sst.pipeline.polish). Only memory is left: the take is kept whole for the
# last resort, a retry and the saved recording, 230 MB an hour at 16 kHz (690 MB at a 48 kHz microphone's own rate), so a
# recording forgotten with hands-free on stops after an hour, and the text is still typed.
MAX_SECONDS = 60 * 60
SHORTCUT_SECONDS = 0.6  # another key this soon after Ctrl+Win: a Windows shortcut (Ctrl+Win+D), not a dictation
DEFAULT_HOTKEY = "ctrl+win"
START_BEEP = 880  # Hz: the voice pipeline filters it out of the recording (sst.pipeline.session._ToneNotch)
# A recording whose loudest sample is below this (-80 dBFS) has no sound at all: a muted, unplugged or switched-off
# microphone. Any working microphone hears more, even in a quiet room.
SILENT = 1e-4
_NOT_TYPED = "Couldn't type it here ({}). It's on Home, ready to copy."

log = logging.getLogger(__name__)


class Dictation:
    """Turns hotkey events into recordings and typed text.

    Feed it `handle(event, at)` for every hotkey event and `tick(now)` a few times a second.
    It reports progress through `on_state(state, message)`, where state is one of
    recording, transcribing, typed (message = the text), typed_raw (typed as heard because the cleanup
    couldn't help; message = the reason), typed_local (a cloud speech model couldn't help, so Parakeet transcribed on
    this computer; message = the reason), idle, cancelled, ignored, warning, error; and each typed text
    through `on_result(heard, typed, seconds of audio)`. Both are called from the caller's thread and from the worker
    thread.
    `cleanup` is an optional sst.gateway.Polisher (or anything with prepare(), polish(text) and last_error). An engine
    may have prepare() too (connect while the user speaks) and last_error (why it fell back on another engine).
    `pipeline` (set by the app) replaces engine + cleanup with the staged pipeline when the recorder offers the take
    in progress (Recorder.current_take).
    """

    def __init__(self, engine, recorder, *, paste: Callable[[str], None] = paste_text, sounds: bool = True,
                 save: bool = True):
        self.engine, self.recorder, self.paste = engine, recorder, paste
        self.sounds, self.save = sounds, save
        self.listener: HotkeyListener | None = None  # told when recording, so that Esc cancels only then
        self.cleanup = None  # set and replaced by the app when the chosen model changes; None = type what was heard
        self.pipeline = None  # sst.pipeline.session.VoicePipeline, set by the app; None = the classic way
        self.last_failed: tuple[np.ndarray, int] | None = None  # a recording nothing could be typed for: retry_last()
        self._session = None  # the pipeline session being recorded
        self.on_state: Callable[[str, str], None] = lambda state, message: None
        self.on_result: Callable[[str, str, float], None] = lambda heard, typed, seconds: None
        # Voice commands ("make it concise", sst.commands): `command` finds one in the words heard, `on_command` carries
        # it out (the app's Text Transform). None: every dictation is typed.
        self.command: Callable[[str], str | None] | None = None
        self.on_command: Callable[[str], None] = lambda command: None
        # The words heard, trimmed before the voice commands, the snippets and the cleanup (the wake word's keyword and
        # end phrase, sst.wakeui). Set by the app for one recording: the one started next takes it with it.
        self.trim: Callable[[str], str] | None = None
        self._trim_now: Callable[[str], str] | None = None  # the recording's own
        self._failed_trim: Callable[[str], str] | None = None  # last_failed's, for a retry
        self.snippets: Callable[[], Sequence[Snippet]] = tuple  # the user's snippets (sst.snippets), for the classic way
        # A word about the microphone (the chosen one isn't connected), once per change: the app shows a notification.
        self.on_notice: Callable[[str], None] = lambda message: None
        self._told = ""
        self.recording = False
        self._holding = False  # the press that started this recording has not been released yet
        self._started = 0.0
        self._stopped_by_press = -1.0  # when a hotkey press last ended a recording
        self._jobs: queue.Queue[tuple] = queue.Queue()  # (take, pipeline session or None, trim)
        threading.Thread(target=self._work, name="transcriber", daemon=True).start()

    def handle(self, event: str, at: float) -> None:
        if event == "press":
            if self.recording:
                self._stop(keep=True)
                self._stopped_by_press = at
            else:
                self._start(at)
        elif event == "handsfree":  # Ctrl+Win+Space, arriving right after that chord's "press"
            if self.recording:
                self._holding = False  # keep recording after the keys are let go
            elif at - self._stopped_by_press > 1.0:  # not the chord that has just stopped a recording
                self._start(at)
                self._holding = False
        elif not self.recording:
            return
        elif event == "release" and self._holding:
            self._holding = False
            if at - self._started >= HOLD_SECONDS:
                self._stop(keep=True)
        elif event == "cancel":
            self._stop(keep=False)
        elif event == "interrupt" and self._holding:
            if at - self._started < SHORTCUT_SECONDS:  # a shortcut such as Ctrl+Win+D, not dictation
                self._stop(keep=False, quiet=True)
            else:  # a key brushed while speaking (Shift, a Ctrl+click): what was said is kept, the dictation goes on
                log.info("Another key was pressed %.1f s into the dictation: it goes on", at - self._started)

    def tick(self, now: float) -> None:
        if self.recording and now - self._started > MAX_SECONDS:
            self.on_state("warning", f"Reached {MAX_SECONDS // 60} minutes, stopping.")
            self._stop(keep=True)
        tick = getattr(self.recorder, "tick", None)
        if tick:
            tick(now)  # close a warm microphone that has been idle long enough

    def end(self, keep: bool = True, quiet: bool = False) -> None:
        """Stop the recording from outside the hotkey (the wake word's end phrase or a command): typed when `keep`;
        dropped otherwise, silently with `quiet`."""
        if self.recording:
            self._stop(keep=keep, quiet=quiet)

    def close(self) -> None:
        if self.recording:
            self._stop(keep=False, quiet=True)
        close = getattr(self.recorder, "close", None)
        if close:
            close()
        settle()  # the user's clipboard back before Rflow quits (a paste puts it back half a second later)

    def wait(self) -> None:
        """Block until every recording so far is transcribed and typed."""
        self._jobs.join()

    @property
    def busy(self) -> bool:
        """Recording, or a dictation still being transcribed or typed: nothing may cut it (an update waits)."""
        return self.recording or self._jobs.unfinished_tasks > 0

    def _type(self, typed: str) -> str:
        """Type the text where the user is; "" or why it couldn't be. Either way it goes on to the history: a text that
        couldn't be typed is never lost."""
        if not typed:
            return ""
        try:
            self.paste(typed + " ")  # trailing space so the next dictation doesn't run into this one
            return ""
        except Exception as e:  # the clipboard held by another app, the keys held down...
            log.warning("Couldn't type the dictation: %s", e)
            return str(e) or type(e).__name__

    def _start(self, at: float) -> None:
        try:
            self.recorder.start()
        except Exception as e:
            log.exception("Could not open the microphone")
            self.on_state("error", f"Could not open the microphone: {e}")
            return
        notice = getattr(self.recorder, "notice", "")
        if notice and notice != self._told:
            log.info(notice)
            self.on_notice(notice)
        self._told = notice
        self._beep(START_BEEP)
        self._trim_now = self.trim
        prepare = getattr(self.engine, "prepare", None)
        if prepare:
            prepare()  # a cloud speech model: connect while the user speaks, not after
        self._session = self._open_session()
        llm = getattr(self.pipeline.stages, "llm", None) if self._session else None
        cleanup = llm if self._session else self.cleanup
        if cleanup and hasattr(cleanup, "prepare"):
            cleanup.prepare()  # connect to the gateway while the user speaks, not after
        if self.listener:
            self.listener.recording = True  # Esc now cancels
        self.recording, self._holding, self._started = True, True, at
        self.on_state("recording", "")

    def _stop(self, keep: bool, quiet: bool = False) -> None:
        self.recording = False
        if self.listener:
            self.listener.recording = False  # give Esc back to the other apps
        # The take keeps recording a short tail after the key release; the worker waits for it, not this thread.
        stop_later = getattr(self.recorder, "stop_later", None)
        take = stop_later() if stop_later else Take.ready(self.recorder.stop(), self.recorder.rate)
        session, self._session = self._session, None
        if quiet or not keep or take.seconds < MIN_SECONDS:
            if session is not None:
                session.cancel()  # its feeder stops; results still on their way are dropped
        if quiet:
            self.on_state("idle", "")
        elif not keep:
            self._beep(330)
            self.on_state("cancelled", "Cancelled.")
        elif take.seconds < MIN_SECONDS:
            self.on_state("ignored", "Too short, ignored.")
        else:
            self._beep(660, after=take.done)  # once the tail is in: the microphone heard it in the last part
            self._jobs.put((take, session, self._trim_now))
            self.on_state("transcribing", "")

    def retry_last(self) -> bool:
        """Transcribe the last recording nothing could be typed for again (e.g. the internet was down)."""
        if self.last_failed is None or self.pipeline is None:
            return False
        audio, rate = self.last_failed
        self.last_failed = None
        session = self.pipeline.start(rate)
        session.trim = self._failed_trim  # the keyword and end phrase are still in that recording
        _Feeder(Take.ready(audio, rate), session).start()
        self._jobs.put((Take.ready(audio, rate), session, self._failed_trim))
        self.on_state("transcribing", "")
        return True

    def _open_session(self):
        """A pipeline session for the recording just started, fed by its own thread; None for the classic way."""
        take = getattr(self.recorder, "current_take", None)
        if self.pipeline is None or take is None:
            return None
        try:
            preroll, n = [], 0
            for block in list(take.chunks):  # the blocks from before the key press come first
                if n >= take.preroll:
                    break
                preroll.append(block)
                n += len(block)
            preroll_audio = np.concatenate(preroll) if preroll else None
            session = self.pipeline.start(take.rate, preroll_audio, START_BEEP if self.sounds else None)
            session.trim = self._trim_now  # applied to the words heard before the voice command and the text stages
            session = streaming.attach(session, self.engine, preroll_audio)  # an engine that hears it as it's said
            _Feeder(take, session, skip=len(preroll)).start()
            return session
        except Exception:  # the classic way still types the text
            log.exception("Could not start a voice pipeline session")
            return None

    def _work(self) -> None:
        while True:
            take, session, trim = self._jobs.get()
            if session is not None:
                try:
                    self._finish_session(take, session, trim)
                except Exception as e:  # keep the worker alive for the next recording
                    log.exception("The voice pipeline failed")
                    self.on_state("error", f"Error: {e}")
                finally:
                    self._jobs.task_done()
                continue
            try:
                audio, rate = take.audio(), take.rate
                if self._no_sound(audio):
                    continue
                t0 = time.perf_counter()
                engine = self.engine  # read once: the app may swap it meanwhile
                try:
                    text = engine.transcribe(audio, rate)
                except Exception as e:
                    if self.save:  # e.g. a cloud model failed with no Parakeet to take over: the voice isn't lost
                        save_recording(audio, rate, "")
                        raise RuntimeError(f"{e}. The recording is kept in Rflow's recordings folder.") from e
                    raise
                fell_back = getattr(engine, "last_error", "")
                took = time.perf_counter() - t0
                log.info("%.1fs -> %.2fs  %s", len(audio) / rate, took, text or "(nothing recognised)")
                text = trimmed(trim, text)
                mine = self.snippets() if text else ()
                snippet = snippets.alone(text, mine) if mine else None
                if snippet is None and text and self.command is not None and (command := self.command(text)):
                    self.on_state("idle", "")
                    self.on_command(command)  # not typed, not in the history: a command, not a dictation
                    continue
                cleanup = self.cleanup  # read once: the app may swap it meanwhile
                if snippet is not None:  # "my email": the user's own text, as entered
                    typed = snippet.text
                else:
                    protected, slots = snippets.protect(text, mine) if mine else (text, {})
                    typed = cleanup.polish(protected) if cleanup and protected else protected
                    if slots:  # the AI may have lost a placeholder: then the text as heard, with the snippets
                        typed = snippets.expand(typed, slots) or snippets.expand(protected, slots) or text
                failed = self._type(typed)
                if float(np.abs(audio).max()) < 0.01:
                    self.on_state("warning", "Almost silent: check the microphone.")
                if self.save:
                    save_recording(audio, rate, typed)
                if typed:
                    self.on_result(text, typed, len(audio) / rate)
                if failed:
                    self.on_state("error", _NOT_TYPED.format(failed))
                elif typed and fell_back:
                    self.on_state("typed_local", fell_back)
                elif typed and cleanup and cleanup.last_error:
                    self.on_state("typed_raw", cleanup.last_error)
                else:
                    self.on_state("typed" if typed else "idle", typed)
            except Exception as e:  # keep the worker alive for the next recording
                log.exception("Transcription or typing failed")
                self.on_state("error", f"Error: {e}")
            finally:
                self._jobs.task_done()

    def _no_sound(self, audio: np.ndarray) -> bool:
        """A recording with no sound at all isn't sent to the speech model (it would only say it heard nothing): the user
        is told which microphone gave nothing, and where to choose another."""
        if len(audio) and float(np.abs(audio).max()) >= SILENT:
            return False
        describe = getattr(self.recorder, "describe", None)
        name = (describe() if describe else {}).get("device", "")
        log.warning("No sound from the microphone (%s): not transcribed", name or "unknown")
        self.on_state("error", f"No sound came from the microphone{f' ({name})' if name else ''}. Check that it isn't "
                               "muted or switched off, or choose another one in AI & models.")
        return True

    def _finish_session(self, take: Take, session, trim: Callable[[str], str] | None = None) -> None:
        # Without the pre-roll the session left out (speech before the key press that isn't part of the dictation):
        # the whole-recording last resort, a retry and the saved recording see what the session saw.
        audio, rate = take.audio()[getattr(session, "trimmed", 0):], take.rate
        if self._no_sound(audio):
            session.result()  # let the session end; with no sound it sent nothing to wait for
            return
        final = session.result()
        if final.command:
            self.on_state("idle", "")
            self.on_command(final.command)  # not typed, not saved, not in the history
            return
        if final.provenance.value == "failed" and self.engine is not None and len(audio):
            try:  # the last resort before failing: the whole recording in one piece, as before the voice pipeline
                text = self.engine.transcribe(audio, rate)
                if text:
                    log.warning("%s: %s; typed from the whole recording instead", final.session_id, final.error)
                    final = session.text_result(text)
            except Exception as e:
                log.warning("The whole recording couldn't be transcribed either: %s", e)
        if final.provenance.value == "failed":  # fail closed: nothing typed rather than a text with a hole in it
            self.last_failed, self._failed_trim = (audio, rate), trim
            kept = ""
            if self.save:
                save_recording(audio, rate, "")
                kept = " The recording is kept; Retry the last dictation is in the tray menu."
            self.on_state("error", f"Nothing was typed: {final.error}.{kept}")
            return
        typed = final.text
        heard = final.stages.get("trimmed", final.stages.get("merged")) or typed
        failed = self._type(typed)  # returns at once: the clipboard is put back in the background (sst.paste)
        if len(audio) and float(np.abs(audio).max()) < 0.01:
            self.on_state("warning", "Almost silent: check the microphone.")
        if typed:
            self.on_result(heard, typed, len(audio) / rate)
        if failed:
            self.on_state("error", _NOT_TYPED.format(failed))
        elif typed and final.notes:
            self.on_state("typed_local", final.notes[0])
        elif typed and final.provenance.value == "formatted_fallback" and final.error:
            self.on_state("typed_raw", final.error)
        else:
            self.on_state("typed" if typed else "idle", typed)
        if self.save:  # after the user is told: writing a long recording takes a moment
            save_recording(audio, rate, typed)

    def _beep(self, frequency: int, after: threading.Event | None = None) -> None:
        if not self.sounds:
            return

        def play() -> None:
            if after is not None:
                after.wait(2.0)
            winsound.Beep(frequency, 60)
        threading.Thread(target=play, daemon=True).start()


def trimmed(trim: Callable[[str], str] | None, text: str) -> str:
    """`text` through a recording's trim (the wake word's keyword and end phrase); as heard if the trim fails: typing a
    keyword is better than losing the dictation."""
    if trim is None or not text:
        return text
    try:
        return trim(text)
    except Exception:
        log.exception("Trimming the words heard failed: typed as heard")
        return text


class _Feeder(threading.Thread):
    """Passes a take's audio to its pipeline session while it is being recorded, then finishes the session once the
    tail after the key release has arrived. Polling keeps the audio callback free of any work."""

    POLL = 0.03

    def __init__(self, take: Take, session, skip: int = 0):
        super().__init__(name="pipeline-feeder", daemon=True)
        self.take, self.session, self.position = take, session, skip

    def run(self) -> None:
        try:
            while True:
                if getattr(self.session.state, "value", "") == "cancelled":
                    return
                done = self.take.done.is_set()  # read before the chunks: nothing appended after it is missed
                blocks = self.take.chunks[self.position:]
                self.position += len(blocks)
                for block in blocks:
                    self.session.feed(block)
                if done and self.position >= len(self.take.chunks):
                    break
                time.sleep(self.POLL)
            self.session.finish()
        except Exception:
            log.exception("Feeding the voice pipeline failed")
            self.session.cancel()


# ---- the console command: `sst dictate`

def run(load_engine: Callable, hotkey: str = DEFAULT_HOTKEY, device: int | str | None = None, save: bool = True) -> None:
    key = parse_hotkey(hotkey)  # a typo is reported before the slow model load
    if already_running():
        raise SystemExit("Dictation is already running in another window (the installed app or dictate.cmd).")
    if key.modifiers == {"ctrl", "win"} and key.key is None and wispr_flow_running():
        print("Warning: Wispr Flow is running and also listens to Ctrl+Win, so both would type.\n"
              "         Quit Wispr Flow (its tray icon -> Quit), or start this with --hotkey menu.")
    dictation = Dictation(load_engine(), Recorder(device), save=save)
    dictation.on_state = _print_state
    listener = HotkeyListener(key)  # started after the model load, so typing never waits for it
    dictation.listener = listener
    listener.start()
    print_ready(key)
    try:
        while True:
            try:
                event, at = listener.events.get(timeout=0.25)
                dictation.handle(event, at)
            except queue.Empty:
                pass
            dictation.tick(time.monotonic())
    finally:
        listener.stop()
        dictation.close()


def print_ready(key) -> None:
    rows = [(f"hold {key.label}", "talk while holding, let go to type the text"),
            (f"tap  {key.label}", "start recording, tap again to stop and type the text")]
    if key.key is None:
        rows.append((f"{key.label}+Space", "hands-free as well (like Wispr Flow)"))
    rows.append(("Esc", "cancel the recording"))
    width = max(len(keys) for keys, _ in rows)
    print("\nReady. Click in any text box, then:\n"
          + "".join(f"  {keys:<{width}}   {what}\n" for keys, what in rows)
          + "Keep this window open (you can minimise it). Close it or press Ctrl+C to quit.\n")


def _print_state(state: str, message: str) -> None:
    if state == "recording":
        print("● Recording...")
    elif state in ("cancelled", "ignored", "warning", "error"):
        print(f"  {message}")


def already_running() -> bool:
    # A named mutex shared by every copy (tray app, console, source checkout); Windows frees it when the process ends.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW(None, False, "SST-Dictation-dictate")
    return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS


def wispr_flow_running() -> bool:
    try:
        result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Wispr Flow.exe", "/NH"], capture_output=True,
                                text=True, errors="replace", timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "Wispr Flow.exe" in result.stdout
