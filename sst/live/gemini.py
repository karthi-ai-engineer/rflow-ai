"""Gemini 3.5 Live Translate as a live captions engine (Google's Live API, a WebSocket; the profile's Gemini key).

    setup        {"setup": {"model": ..., "generationConfig": {"responseModalities": ["AUDIO"],
                  "translationConfig": {"targetLanguageCode": "en", "echoTargetLanguage": false}},
                  "inputAudioTranscription": {}, "outputAudioTranscription": {}, "sessionResumption": {"handle": ...}}}
                 (the transcripts are the setup's own fields, as in the API reference; Google's Live Translate
                 guide shows them in generationConfig, which the server closes with 1007 "Unknown name")
    audio        {"realtimeInput": {"audio": {"data": <base64 PCM16 16 kHz mono>, "mimeType": "audio/pcm;rate=16000"}}}
    silence      {"realtimeInput": {"audioStreamEnd": true}}: after SILENCE_SEND_S of digital silence no audio is sent
                 (Google bills the audio streamed); the next sound reopens the stream, after PREROLL_FRAMES of silence
    answers      serverContent.inputTranscription.text   the words heard, piece by piece
                 serverContent.outputTranscription.text  their translation, piece by piece
                 serverContent.modelTurn                 the translated voice (24 kHz), unused: captions only
                 sessionResumptionUpdate.newHandle       where the session can be taken up again (kept in memory only)
                 goAway                                  the connection ends soon: a new one is opened

The model translates continuously, a few seconds behind the speaker. It only produces audio (and bills it), so the
voice is thrown away and its transcript shown. A line ends when the model ends its turn, or after LiveConfig.line_pause_s
without new words, or once a long translation and the words heard are both at a sentence end (one runs ahead of the
other, either way). How far a line trailed the speaker is measured from when the audio last had
a voice in it to when the line's translation was complete, and only when the voice had paused. A connection lives ~10
minutes: before that, at a pause, a new one is opened (frames wait in the queue meanwhile, so no audio is lost), and it
takes up the same session with the last resumption handle, so the model keeps what was said (Live API session
resumption, https://ai.google.dev/gemini-api/docs/live-session; a setup Google refuses with it is sent again without).
Google's refusals (no credit, a key not accepted, a quota used up) are said plainly and end the captions; a short rate
limit is waited out. The key and the handle are never logged.
"""
import base64
import contextlib
import difflib
import json
import logging
import queue
import re
import socket
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable
from urllib.parse import urlsplit

import numpy as np

from sst import netconnect
from sst.live.contracts import Kind, LiveConfig, LiveEvent

URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
QUEUE_FRAMES = 50  # 5 s of audio: when the network falls further behind, the oldest frames go (captions stay live)
LONG_LINE = 120  # characters: a translation this long ends its line where both texts are at a sentence end
VOICE_LEVEL = 10 ** (-45 / 20)  # a frame louder than -45 dBFS has a voice in it (for measuring the lag only)
VOICE_PAUSE = 0.3  # seconds: the voice has paused, so a translation arriving now is measured from that pause
SILENT_PEAK = 32  # a frame whose samples all stay within this (of 32,768: -60 dBFS) is digital silence
SILENCE_SEND_S = 2.0  # this much silence is still sent (the model hears the pause and ends its line); after it, none
PREROLL_FRAMES = 2  # the last 200 ms of silence not sent go first when sound comes back
RECONNECT_GRACE_S = 45.0  # past reconnect_s, the new connection waits this long at most for a pause (Google's ~10 min)
QUOTA_TRIES = 3  # a quota refusal that doesn't say when it lifts is tried this often before the captions stop
QUOTA_WAIT = 30.0  # seconds between those tries
LONGEST_WAIT = 600.0  # a rate limit asking for longer than this ends the captions instead
BYTES_PER_S = 32_000  # 16 kHz, 16-bit
_SENTENCE_END = re.compile(r"[.!?。！？]\s*$")
_SENTENCES = re.compile(r"[。！？!?]+|\.+(?!\w)")  # each sentence end once ("?!", "..."); "3.5" is none
_CJK = "".join(f"{chr(a)}-{chr(b)}" for a, b in (
    (0x3000, 0x303F), (0x3040, 0x30FF), (0x31F0, 0x31FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xFF00, 0xFFEF)))
_CJK_GAP = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}?!])")  # Japanese and Chinese put no spaces between words

log = logging.getLogger(__name__)


ADDRESS_WAIT = 2.0  # seconds for each of Google's addresses to answer before the next one is tried


def _connect(url: str):
    from websockets.sync.client import connect
    sock = None if urllib.request.getproxies() else _socket(urlsplit(url).hostname)  # behind a proxy: websockets' way
    # legacy=True: the connection itself, as before websockets 17 (without it, 17 warns that this use will change)
    return connect(url, sock=sock, open_timeout=10, close_timeout=2, max_size=None, legacy=True)


def _socket(host: str, port: int = 443, each: float = ADDRESS_WAIT) -> socket.socket:
    """A connection to the first of the host's addresses that answers. Python tries them in turn and waits out the whole
    timeout on each: from the dev laptop's network the first of Google's eight addresses never answers (measured: 5 s
    timeout there, 0.03 s for the other seven), which made every start wait 10 s and fail once."""
    error: OSError | None = None
    for family, kind, proto, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        sock = socket.socket(family, kind, proto)
        sock.settimeout(each)
        try:
            sock.connect(address)
        except OSError as e:
            sock.close()
            error = e
            continue
        sock.settimeout(None)
        return sock
    raise error or OSError(f"no address for {host}")


class GeminiLiveTranslate:
    """Feed it frames from any thread; it calls `on_event` (from its own threads) with LiveEvents."""

    def __init__(self, key: str, config: LiveConfig, on_event: Callable[[LiveEvent], None], *,
                 connect: Callable = _connect, clock: Callable[[], float] = time.monotonic, lane: str = "system"):
        self.key, self.config, self.on_event, self.lane = key, config, on_event, lane
        self._connect, self._clock = connect, clock
        self._frames: queue.Queue[bytes] = queue.Queue(maxsize=QUEUE_FRAMES)
        self._stop = threading.Event()
        self._lock = threading.Lock()  # the line being built: touched by the receiving and the sending thread
        self._source = self._translation = self._language = ""
        self._last_text = 0.0
        self._lag = 0.0  # the line's: how long after the voice paused its latest translated words came (0 = unknown)
        self._voice_at = 0.0  # when the last frame with a voice in it was captured
        self._ws = None
        self._goaway = False
        self._opened = 0.0
        self._closed: str | None = None  # why the server closed the connection, once it did
        self._thread: threading.Thread | None = None
        self._resume = True  # ask for session resumption (dropped if Google refuses a setup with it)
        self._token = ""  # where the session can be taken up again: in memory only, never logged
        self._stream_end = True  # say when the audio pauses (dropped if Google refuses it)
        self._ended = False  # audioStreamEnd was the last thing sent on this connection
        self._quiet_bytes = 0  # of silence in a row so far (bytes: tenths of a second would add up to 2.0000000000000004)
        self._skipping = False  # silence long enough: frames aren't sent until sound comes back
        self._preroll: deque[bytes] = deque(maxlen=PREROLL_FRAMES)
        self._quota = 0  # quota refusals in a row
        self._started = 0.0
        # for the session's log line (report()): never any text
        self.dropped = 0  # frames dropped because the network fell behind
        self.sent = 0  # frames sent
        self.silence_s = 0.0  # seconds of silence not sent
        self.reconnects = 0
        self.reconnect_s = 0.0  # seconds without a connection between them, in all
        self.resumed = 0  # reconnects that took up the same session
        self.first_words: float | None = None  # seconds from start() to the first words shown
        self.error = ""  # the last error, as the user saw it

    # -- the caller's side

    def start(self) -> None:
        self._stop.clear()
        self._started = self._clock()
        self._thread = threading.Thread(target=self._run, name="live-gemini", daemon=True)
        self._thread.start()

    def feed(self, frame: bytes) -> None:
        """A 100 ms frame of 16 kHz mono PCM16. Never blocks the capture: a full queue loses its oldest frame."""
        if _level(frame) >= VOICE_LEVEL:
            self._voice_at = self._clock()
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._frames.get_nowait()
                self.dropped += 1
            with contextlib.suppress(queue.Full):
                self._frames.put_nowait(frame)

    def stop(self, wait: float = 3.0) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(Exception):
                ws.close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(wait)

    def report(self) -> str:
        """How the session went, for the log: numbers only."""
        first = f"first words after {self.first_words:.1f} s" if self.first_words is not None else "no words"
        parts = [first, f"{self.sent} frames sent, {self.dropped} dropped", f"{self.silence_s:.0f} s of silence not sent",
                 f"{self.reconnects} reconnects ({self.reconnect_s:.1f} s without a connection, {self.resumed} resumed)"]
        if self.error:
            parts.append(f"last error: {self.error}")
        return "; ".join(parts)

    # -- the connection

    def _run(self) -> None:
        backoff, failures = 1.0, 0
        lost: float | None = None  # when the last connection ended
        while not self._stop.is_set():
            resuming, in_setup = bool(self._token), True
            try:
                self._status("Connecting to Google…")
                ws = self._open()
                in_setup = False
                if lost is not None:
                    gap = self._clock() - lost
                    self.reconnects, self.reconnect_s, self.resumed = (
                        self.reconnects + 1, self.reconnect_s + gap, self.resumed + resuming)
                    log.info("Live captions: connected again in %.1f s (%s)", gap,
                             "the same session" if resuming else "a new session")
                    lost = None
                backoff, failures, self._quota = 1.0, 0, 0
                self._status("Listening")
                receiver = threading.Thread(target=self._receive, args=(ws,), name="live-gemini-in", daemon=True)
                receiver.start()
                self._send(ws)
                if not self._token:
                    self._finish_line()  # a new session starts with no memory of this line; a resumed one goes on
                with contextlib.suppress(Exception):
                    ws.close()
                receiver.join(2)
                self._ws = None
                lost = self._clock()
            except Exception as e:
                self._ws = None
                if self._stop.is_set():
                    break
                if lost is None and self._opened:  # a connection there was: the time until the next one counts
                    lost = self._clock()
                failures += 1
                log.warning("Live captions: %s", self._scrub(_describe(e)))
                if failures == 1 and isinstance(e, TimeoutError):
                    continue  # a first connection that stalls (seen on the dev laptop): again at once, quietly
                if self._without_extras(e, in_setup, resuming):
                    continue  # asked again at once without what Google didn't take
                message, wait = self._judge(e, backoff)
                if wait is not None and not in_setup and self._skipping:
                    log.info("Live captions: Google closed a connection that had no sound to hear: a new one")
                else:  # an idle connection closing is nothing the user needs to see; anything else is
                    self.error = message
                    self.on_event(LiveEvent(Kind.ERROR, message, lane=self.lane))
                if wait is None:
                    break  # trying again won't help
                self._status("Reconnecting…")
                self._stop.wait(wait)
                backoff = min(backoff * 2, 30.0)
        self._finish_line()
        self._status("Stopped")

    def _without_extras(self, error: Exception, in_setup: bool, resuming: bool) -> bool:
        """Google refused something this engine adds and can do without: a handle it no longer knows (they last 2 hours),
        session resumption (a preview model may not take it), or audioStreamEnd. Each is dropped once and the
        connection made again at once; a refusal of anything else is the error it is."""
        if in_setup and resuming and not isinstance(error, OSError):
            log.info("Live captions: the session couldn't be taken up again: a new one")
            self._token = ""
            return True
        if self._judge(error, 0.0, count=False)[0] != NOT_ACCEPTED:
            return False
        raw = self._scrub(_describe(error)).lower().replace("_", "")
        named = "unknown name" in raw  # Google names the field it doesn't know: only that one is dropped
        if in_setup and self._resume and (not named or "sessionresumption" in raw):
            log.info("Live captions: Google refused session resumption: reconnecting without it")
            self._resume, self._token = False, ""
            return True
        if not in_setup and self._ended and self._stream_end and (not named or "audiostreamend" in raw):
            log.info("Live captions: Google refused audioStreamEnd: long silences are left out without it")
            self._stream_end = False
            return True
        return False

    def _open(self):
        self._goaway, self._closed = False, None
        ws = self._connect(f"{URL}?key={self.key}")
        self._ws = ws
        ws.send(json.dumps({"setup": {
            "model": f"models/{self.config.model}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],  # the model only speaks; its transcript is what's shown
                "translationConfig": {"targetLanguageCode": self.config.target,
                                      "echoTargetLanguage": self.config.echo_target},
            },
            "inputAudioTranscription": {},  # the words heard
            "outputAudioTranscription": {},  # the translation, as text
            **({"sessionResumption": {"handle": self._token} if self._token else {}} if self._resume else {}),
        }}))
        self._ended = False
        reply = json.loads(ws.recv(timeout=15))
        if "setupComplete" not in reply:
            raise ConnectionError(f"setup not accepted: {str(reply)[:200]}")
        self._opened = self._clock()
        return ws

    def _send(self, ws) -> None:
        """Stream the frames until it's time for a new connection, the server closes it, or stop()."""
        while not self._stop.is_set():
            if self._closed is not None:
                raise self._closed
            if self._goaway or self._due():
                log.info("Live captions: a new connection (the old one ends soon)")
                return
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                frame = b""
            self._check_pause()
            if frame:
                self._stream(ws, frame)

    def _due(self) -> bool:
        """Time for a new connection: past reconnect_s at a pause (the voice paused, no line being built), or anyway
        RECONNECT_GRACE_S later, still before Google's ~10 minutes."""
        age = self._clock() - self._opened
        if age <= self.config.reconnect_s:
            return False
        if self._skipping or age > self.config.reconnect_s + RECONNECT_GRACE_S:
            return True
        with self._lock:
            pending = bool(self._source or self._translation)
        return not pending and self._clock() - self._voice_at >= VOICE_PAUSE

    def _stream(self, ws, frame: bytes) -> None:
        """Send a frame, unless it's more of a long digital silence: Google bills every second streamed. Once the
        silence has lasted SILENCE_SEND_S (enough for the model to hear the pause and finish its line), the stream is
        ended (audioStreamEnd: the model finalises what it heard) and frames wait for sound; the last PREROLL_FRAMES of
        the silence go first then, so the model hears the voice begin."""
        seconds = len(frame) / BYTES_PER_S
        self._quiet_bytes = self._quiet_bytes + len(frame) if _silent(frame) else 0
        if self._quiet_bytes > SILENCE_SEND_S * BYTES_PER_S:
            if not self._skipping:
                self._skipping = True
                if self._stream_end:  # without it, the 2 s sent are the pause the model hears
                    ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                    self._ended = True
            self._preroll.append(frame)
            self.silence_s += seconds
            return
        if self._skipping:
            self._skipping = False
            for before in self._preroll:
                self._send_audio(ws, before)
                self.silence_s -= len(before) / BYTES_PER_S
            self._preroll.clear()
        self._send_audio(ws, frame)

    def _send_audio(self, ws, frame: bytes) -> None:
        ws.send(json.dumps({"realtimeInput": {"audio": {
            "data": base64.b64encode(frame).decode("ascii"), "mimeType": "audio/pcm;rate=16000"}}}))
        self._ended = False
        self.sent += 1

    def _receive(self, ws) -> None:
        try:
            for raw in ws:
                self._handle(json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw))
        except Exception as e:  # closed by the server (a refused key arrives this way too), or by us
            if not self._stop.is_set():
                self._closed = e
            return
        if not self._stop.is_set() and not self._goaway:
            self._closed = ConnectionError("the connection closed")

    # -- the answers

    def _handle(self, message: dict) -> None:
        if "goAway" in message:
            self._goaway = True
        update = message.get("sessionResumptionUpdate") or {}
        if update.get("resumable") and update.get("newHandle"):
            self._token = update["newHandle"]
        content = message.get("serverContent") or {}
        heard = content.get("inputTranscription") or {}
        if heard.get("text"):
            self._add(Kind.SOURCE, heard["text"], heard.get("languageCode", ""))
        said = content.get("outputTranscription") or {}
        if said.get("text"):
            self._add(Kind.TRANSLATION, said["text"], said.get("languageCode", ""))
        if content.get("turnComplete"):
            self._finish_line()

    def _add(self, kind: Kind, piece: str, language: str) -> None:
        now = self._clock()
        if self.first_words is None:
            self.first_words = now - self._started
        with self._lock:
            if kind is Kind.SOURCE:
                self._source = _join(self._source, piece)
                self._language = language or self._language
                text = self._source
            else:
                self._translation = _join(self._translation, piece)
                text = self._translation
            self._last_text = now
            if kind is Kind.TRANSLATION or not self._translation:  # the line's latest translated (or heard) words
                quiet = now - self._voice_at
                self._lag = quiet if self._voice_at and quiet >= VOICE_PAUSE else 0.0  # still speaking: unknown
            cut = self._find_cut()
        self.on_event(LiveEvent(kind, _tidy(text), language=language, lane=self.lane))
        if cut:
            self._finish_line(*cut)

    def _find_cut(self) -> tuple[int, int, float] | None:
        """Where a long line ends: when its translation and the words heard are both at a sentence end. Google's
        translation runs ahead of its transcript at times and behind it at others, so a cut at the translation's full
        stop split the words heard mid-word ("…びっくり" | "しました。"), and matching sentence counts drifted for good
        once one sentence became two. Under the lock."""
        t, s = self._translation, self._source
        if len(t) < LONG_LINE or not _SENTENCE_END.search(t):
            return None
        if _SENTENCE_END.search(s):
            return len(t), len(s), self._lag
        if len(t) >= 2 * LONG_LINE:  # the two never pause together: end at the words heard's last sentence end
            ends = [m.end() for m in _SENTENCES.finditer(s)]
            return len(t), ends[-1] if ends else len(s), self._lag
        return None

    def _check_pause(self) -> None:
        with self._lock:
            pending = bool(self._source or self._translation)
            quiet = self._clock() - self._last_text
        if pending and quiet >= self.config.line_pause_s:
            self._finish_line()

    def _finish_line(self, t_cut: int | None = None, s_cut: int | None = None, lag: float | None = None) -> None:
        """Ends the line being built: all of it, or up to the cuts (what follows them begins the next line)."""
        with self._lock:
            t_cut = len(self._translation) if t_cut is None else t_cut
            s_cut = len(self._source) if s_cut is None else s_cut
            source, translation = self._source[:s_cut].strip(), self._translation[:t_cut].strip()
            self._source, self._translation = self._source[s_cut:].lstrip(), self._translation[t_cut:].lstrip()
            lag = self._lag if lag is None else lag
            language, rest = self._language, (self._source, self._translation)
            self._lag = 0.0
        if source or translation:
            self.on_event(LiveEvent(Kind.LINE, _tidy(translation), source=_tidy(source), language=language,
                                    lane=self.lane, seconds=lag))
        for kind, text in zip((Kind.SOURCE, Kind.TRANSLATION), rest, strict=True):
            if text:  # carried past the cut: the next line has begun
                self.on_event(LiveEvent(kind, _tidy(text), language=language, lane=self.lane))

    # -- words for the user

    def _status(self, message: str) -> None:
        self.on_event(LiveEvent(Kind.STATUS, message, lane=self.lane))

    def _scrub(self, text: str) -> str:
        return text.replace(self.key, "***") if self.key else text

    def _judge(self, error: Exception, backoff: float, count: bool = True) -> tuple[str, float | None]:
        """The error in plain words (the provider's own goes to the log), and how long to wait before trying again
        (None: trying again won't help, the captions stop). Google refuses on the WebSocket itself, in its close frame's
        code and reason (1008: the key or the model; 1011: a quota, as reported in the field); an HTTP status comes only
        when the handshake itself is refused. `count`: a quota refusal counts towards QUOTA_TRIES."""
        status, body = _http(error)
        raw = self._scrub(f"{_describe(error)} {body}").lower()
        if status == 402 or _CREDIT.search(raw) or ("billing" in raw and "quota" not in raw):
            return NO_CREDIT, None
        if status in (401, 403) or "api key" in raw or "api_key" in raw or "unauthenticated" in raw \
                or "unregistered caller" in raw:
            return REFUSED_KEY, None
        if "not found" in raw or "is not supported" in raw or status == 404 or "404" in raw or "permission" in raw:
            return NO_MODEL, None  # e.g. 1008 "models/... is not found for API version v1beta"
        if status == 429 or "quota" in raw or "429" in raw or "resource_exhausted" in raw or "rate limit" in raw:
            if _DAILY.search(raw):
                return QUOTA_DAY, None
            wait = netconnect.retry_after(body or _reason(error))
            if wait is not None and wait <= LONGEST_WAIT:
                return RATE_LIMITED.format(max(1, round(wait))), max(wait, 1.0)
            self._quota += count
            if wait is not None or self._quota >= QUOTA_TRIES:
                return QUOTA_STOP, None
            return QUOTA_NOW, QUOTA_WAIT
        if "(1007)" in raw or "invalid json" in raw or "unknown name" in raw or "invalid argument" in raw:
            return NOT_ACCEPTED, None  # the request itself: the same one would be refused again
        if isinstance(error, (OSError, TimeoutError)) or "timed out" in raw or "getaddrinfo" in raw:
            return "Can't reach Google: check the internet connection. Captions try again shortly.", backoff
        return "The connection to Google broke: captions try again shortly.", backoff


REFUSED_KEY = "Gemini didn't accept the key: check the Gemini key in AI & models."
NO_CREDIT = "Gemini credit used up: add credit in Google AI Studio."
NO_MODEL = "This key can't use Gemini Live Translate (a preview model) yet."
NOT_ACCEPTED = "Google didn't accept Rflow's request: live captions need an update to Rflow."
QUOTA_DAY = "Gemini's daily quota for this key is used up: live translation works again tomorrow."
QUOTA_NOW = "Gemini's quota for this key is used up for now: captions try again shortly."
QUOTA_STOP = "Gemini's quota for this key is used up: check its plan and billing in Google AI Studio."
RATE_LIMITED = "Gemini asks to wait {} s (a rate limit): captions try again then."
GIVE_UP = (REFUSED_KEY, NO_CREDIT, NO_MODEL, NOT_ACCEPTED, QUOTA_DAY, QUOTA_STOP)  # the captions stop and say why
# No credit: a 402, the prepaid credits run out ("prepayment credits are depleted"), or a billing problem that isn't
# Google's quota message ("You exceeded your current quota, please check your plan and billing details": a quota)
_CREDIT = re.compile(r"prepay|credits? (are |is )?(depleted|exhausted|used up)|out of credits?|insufficient[_ ](balance|funds)"
                     r"|credit balance|payment required")
_DAILY = re.compile(r"per[ _]?day|daily|\brpd\b")


def _http(error: Exception) -> tuple[int, str]:
    """The HTTP status and body of a refused WebSocket handshake (websockets' InvalidStatus, or the legacy
    InvalidStatusCode), else (0, "")."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None) or getattr(error, "status_code", None) or 0
    body = getattr(response, "body", b"") or b""
    return int(status), body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)


def _reason(error: Exception) -> str:
    close = getattr(error, "rcvd", None)
    return str(close.reason) if close is not None else str(error)


def _describe(error: Exception) -> str:
    """For the log and the wording: a connection Google closed as its code and reason (websockets' own text repeats
    them, "received ...; then sent ..."), anything else as itself."""
    close = getattr(error, "rcvd", None)
    if close is not None:
        return f"Google closed the connection ({close.code}): {close.reason}"
    return f"{type(error).__name__}: {error}"


def _silent(frame: bytes) -> bool:
    """Digital silence: what a capture sends while nothing plays, a muted microphone, the voice muted out."""
    samples = np.frombuffer(frame[: len(frame) // 2 * 2], dtype="<i2")
    return not len(samples) or int(np.max(np.abs(samples.astype(np.int32)))) <= SILENT_PEAK


def _level(frame: bytes) -> float:
    """A frame's loudness: RMS, 1.0 = full scale."""
    samples = np.frombuffer(frame[: len(frame) // 2 * 2], dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples))) / 32768 if len(samples) else 0.0


def _tidy(text: str) -> str:
    """Google's pieces of Japanese (or Chinese) come with spaces between them ("なんと 。そうなんです"): gone."""
    return _CJK_GAP.sub("", text)


_POSSESSIVE_GAP = re.compile(r"s['’]$")


def _join(text: str, piece: str) -> str:
    """The pieces arrive as the model hears them; a piece that repeats the whole line so far replaces it, also when the
    model revised it meanwhile (a word, a comma made a full stop): appended, the line would say it all twice."""
    if text and (piece.startswith(text) or piece.lstrip().startswith(text.lstrip()) or _resends(text, piece)):
        return piece
    if _POSSESSIVE_GAP.search(text) and piece[:1].isalpha() and piece[:1].isascii():
        return f"{text} {piece}"  # Google's pieces drop the space after a plural possessive: "parents'" + "house"
    return text + piece


RESENT = 0.85  # a piece whose start has this much of the line in it (their words without case or punctuation) re-sends it


def _resends(text: str, piece: str) -> bool:
    """A piece that is the line so far again, revised: about as long as the line, and its start has nearly all of the
    line in it, in order. The model's own pieces are a few words that go on from the line, never most of it again."""
    line, sent = (" ".join(re.findall(r"\w+", t.lower())) for t in (text, piece))
    if line.count(" ") < 2 or len(sent) < 0.9 * len(line):
        return False
    blocks = difflib.SequenceMatcher(None, line, sent[:round(1.2 * len(line))], autojunk=False).get_matching_blocks()
    return sum(block.size for block in blocks) >= RESENT * len(line)
