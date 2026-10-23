"""A speech engine that transcribes while the user speaks (sst.engines.gemini_stream), as a shortcut in a voice pipeline
session: the session hears everything as before, and its chunks are kept instead of sent.

    attach(session, engine, preroll)  at the key press: the session as it was, or wrapped when the engine streams
    StreamedSession.feed(block)       the block goes to the session (chunked, held) and to the stream
    StreamedSession.finish()          the session finishes; the stream is told the audio has ended
    StreamedSession.result()          the stream's final text through the session's text stages; or, when the stream
                                      gave none, the held chunks go to the batch model and the session goes on as always

Never a text with a hole in it: the stream's text is used only when the stream says it is complete; anything else (no
connection, a break, no final text in time) sends the held chunks, the whole recording, the usual way. A stream that
breaks while the user speaks lets them go at once, so the batch model starts before the release.
"""
import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


def attach(session, engine, preroll: np.ndarray | None = None):
    """The session, wrapped when the engine has a stream for this dictation (open_stream(rate) -> stream or None).
    `preroll`: what the session was started with (it keeps the part from `session.trimmed` on)."""
    opener = getattr(engine, "open_stream", None)
    if opener is None:
        return session
    try:
        stream = opener(session.rate)
    except Exception:  # dictation goes the usual way
        log.exception("Could not open a stream")
        return session
    if stream is None:
        return session
    wrapped = StreamedSession(session, stream)
    if preroll is not None and len(preroll):
        stream.feed(preroll[session.trimmed:])  # the session fed itself this part when it was made
    return wrapped


class _HeldASR:
    """Stands in for a session's ASR (sst.pipeline.asr.SessionASR): its chunks and its finish are kept until release()
    (the stream failed: they go the usual way) or drop() (the stream's text is used)."""

    def __init__(self, real):
        self.real = real
        self._lock = threading.Lock()
        self._held: list = []
        self._total: int | None = None
        self.holding = True

    def submit(self, chunk) -> None:
        with self._lock:
            if self.holding:
                self._held.append(chunk)
                return
        self.real.submit(chunk)

    def finish(self, total: int) -> None:
        with self._lock:
            if self.holding:
                self._total = total
                return
        self.real.finish(total)

    def release(self) -> int:
        """The held chunks to the speech model, in order (and the finish, if it came); how many."""
        with self._lock:
            if not self.holding:
                return 0
            self.holding = False
            held, total, self._held = self._held, self._total, []
            for chunk in held:  # under the lock: a chunk cut meanwhile waits, so the order holds
                self.real.submit(chunk)
            if total is not None:
                self.real.finish(total)
        return len(held)

    def drop(self) -> None:
        with self._lock:
            self.holding, self._held = False, []
        self.real.cancel()  # a chunk sent before the stream began (the pre-roll's) is no longer wanted

    def __getattr__(self, name):  # wait, cancel, progress...
        return getattr(self.real, name)


class StreamedSession:
    """A sst.pipeline.session.Session with a stream beside it; anything else is the session's own."""

    def __init__(self, session, stream):
        self.session, self.stream = session, stream
        self._held = _HeldASR(session._asr)
        session._asr = self._held  # its chunks wait here; the session doesn't know

    def __getattr__(self, name):  # state, trimmed, session_id, metrics, text_result...
        return getattr(self.session, name)

    def feed(self, block: np.ndarray) -> None:
        self.session.feed(block)
        if not self.stream.failed:
            self.stream.feed(block)
        elif self._held.holding:
            self._let_go()

    def finish(self) -> None:
        self.session.finish()
        if self.stream.failed:
            self._let_go()
        else:
            self.stream.end()

    def cancel(self) -> None:
        self.stream.close()
        self.session.cancel()

    def result(self, timeout: float | None = None):
        kwargs = {} if timeout is None else {"timeout": timeout}
        t0 = time.perf_counter()
        text = self.stream.final(**kwargs) if self._held.holding else None
        if text is None:
            self._let_go()
            return self.session.result(**kwargs)
        self._held.drop()
        self.session.metrics.stage_ms["stream_wait"] = (time.perf_counter() - t0) * 1000
        log.info("%s: typed from the stream", self.session.session_id)
        return self.session._text_stages(text, {"raw": text, "merged": text, "streamed": text}, [])

    def _let_go(self) -> None:
        if self._held.holding:
            sent = self._held.release()
            log.info("%s: the stream couldn't help (%s); %d chunks go to the speech model", self.session.session_id,
                     self.stream.failed or "no text", sent)
