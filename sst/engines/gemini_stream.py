"""Gemini 3.5 Transcribe Live: the words while the user speaks (Google's Live API, a WebSocket; the profile's Gemini key).

Gemini Transcribe (sst.engines.cloud) answers ~2 s after the key is let go, however short the dictation: the recording
only leaves when it ends. The live model hears it as it is said and has the text ~0.4 s after the end of speech. It is
one of the Google Gemini speech provider's models (gemini-3.5-transcribe-live), so its key, its tile and its model menu
are Gemini's; GeminiEngine is that provider's engine, and streams only while a live model is chosen.

    setup    {"setup": {"model": "models/gemini-3.5-transcribe-live", "generationConfig": {"responseModalities": ["TEXT"]},
              "inputAudioTranscription": {"languageCodes": [...], "customVocabulary": [...], "mode": "VERBATIM"}}}
             -> {"setupComplete": {}}
    audio    {"realtimeInput": {"audio": {"data": <base64 PCM16 16 kHz mono>, "mimeType": "audio/pcm;rate=16000"}}}
    release  {"realtimeInput": {"audioStreamEnd": true}}: Google finalises what it has heard at once
    answers  serverContent.interimInputTranscription.text   a guess while the user speaks: not used
             serverContent.inputTranscription.text          a final piece, at each pause and after the release
             goAway                                         the session ends soon

Never loses words: the stream is a shortcut, not the only way. The voice pipeline keeps every chunk of the recording
(sst.pipeline.streaming); when the stream can't connect, breaks, runs out of time or has no final text soon after the
release, the chunks go to the batch model (gemini-3.5-transcribe, the live model's twin) as they always did. Batch
requests (a retry, the reading test, sst eval, the Test button) always use the twin. A session lasts 10 minutes at most
and a dictation 3; Tamil isn't one of the live model's languages, so Tamil goes the batch way. The key is never logged.
"""
import base64
import contextlib
import json
import logging
import queue
import re
import socket
import threading
import time
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit

import numpy as np

from sst import costs, usage
from sst.engines.cloud import CloudEngine, vocabulary_list

URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
TWINS = {"gemini-3.5-transcribe-live": "gemini-3.5-transcribe"}  # a live model and the batch model for its other work
RATE = 16_000  # what the Live API takes; the recorder's rate
FRAME_BYTES = RATE // 10 * 2  # 100 ms of PCM16 a message, as Google advises
CONNECT_WAIT = 5.0  # seconds to connect and have the setup accepted; the frames wait meanwhile
FINAL_WAIT = 1.5  # seconds after the release for the final text; then the batch model takes over
SETTLE = 0.25  # seconds without a new message after a final piece that came after the release: nothing more is coming
COVERED = 1.0  # seconds since the last frame with a voice: a final piece that came after it has all the words
VOICE_LEVEL = 10 ** (-55 / 20)  # a frame louder than -55 dBFS may have a voice in it (low: rather wait than lose a word)
SESSION_LIMIT = 540.0  # seconds; Google ends a session at 10 minutes
DOWN_FOR = 60.0  # after a failed connection, dictate the batch way this long (a refused key isn't tried every time)
# The live model's languages (its page lists BCP-47 codes); one Rflow doesn't map (Tamil) means no stream.
LANGUAGE_CODES = {"en": ["en-US", "en-GB", "en-IN"], "ja": ["ja-JP"], "hi": ["hi-IN"], "te": ["te-IN"], "ml": ["ml-IN"],
                  "kn": ["kn-IN"], "bn": ["bn-IN", "bn-BD"], "mr": ["mr-IN"], "zh": ["cmn-Hans-CN"], "ko": ["ko-KR"],
                  "ar": ["ar-EG"], "es": ["es-US", "es-419"], "fr": ["fr-FR"], "de": ["de-DE"], "pt": ["pt-BR", "pt-PT"],
                  "ru": ["ru-RU"], "": []}  # "": Google detects the language
_CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")
_END = object()

log = logging.getLogger(__name__)
# websockets logs the handshake, the address with the key in it, at DEBUG: never let it through
_WS_LOG = logging.getLogger(f"{__name__}.websocket")
_WS_LOG.setLevel(logging.WARNING)


class GeminiEngine(CloudEngine):
    """Google Gemini's speech engine. `model` is the batch model the requests use (a live model's twin); `chosen` is
    the model the user chose. open_stream() gives a GeminiStream while a live model is chosen."""

    def __init__(self, api_key: str, model: str = "", language: str = "", fallback: Callable[[], object] | None = None,
                 *, url: str = URL, connect: Callable | None = None):
        self.chosen = ""
        super().__init__("gemini", api_key, model, language, fallback)
        self.stream_url, self._ws_connect = url, connect or _connect
        self._stream_down_until = 0.0

    # The app sets `model` to the user's choice (speech_cloud_models); CloudEngine's requests read the batch model.
    @property
    def model(self) -> str:
        return TWINS.get(self.chosen, self.chosen)

    @model.setter
    def model(self, value: str) -> None:
        self.chosen = value

    @property
    def live(self) -> bool:
        return self.chosen in TWINS

    @property
    def title(self) -> str:
        return f"{self.provider.name} {self.chosen}"

    @property
    def signature(self) -> str:
        # The batch requests are the twin's (super's signature); sst eval scores those, not the stream.
        return super().signature + (f"|live:{self.chosen}" if self.live else "")

    def __repr__(self) -> str:  # never let the key reach a log
        return f"GeminiEngine({self.chosen!r})"

    def open_stream(self, rate: int) -> "GeminiStream | None":
        """A stream for the dictation just started, connecting in the background; None when it can't stream (no live
        model chosen, a language the live model doesn't know, another sample rate, a recent failure)."""
        if not self.live or rate != RATE or not self.api_key or self.language not in LANGUAGE_CODES:
            return None
        if time.monotonic() < self._stream_down_until:
            return None
        stream = GeminiStream(self.api_key, self.chosen, LANGUAGE_CODES[self.language], vocabulary_list(self.words),
                              url=self.stream_url, connect=self._ws_connect, on_down=self._down)
        stream.start()
        return stream

    def _down(self) -> None:
        self._stream_down_until = time.monotonic() + DOWN_FOR


class GeminiStream:
    """One dictation's stream. feed() and end() from the feeding thread, final() from the one waiting for the text;
    the connection runs on threads of its own. Every failure sets `failed` (why, in words without the key)."""

    def __init__(self, key: str, model: str, languages: list[str], words: list[str], *, url: str = URL,
                 connect: Callable | None = None, clock: Callable[[], float] = time.monotonic,
                 on_down: Callable[[], None] = lambda: None):
        self.key, self.model, self.languages, self.words = key, model, languages, words
        self.url, self._connect, self._clock, self._on_down = url, connect or _connect, clock, on_down
        self.failed = ""
        self._frames: queue.Queue = queue.Queue()
        self._cond = threading.Condition()
        self._finals: list[str] = []
        self._interim = False  # a guess came after the last final piece: words not yet final
        self._last_final = self._last_message = self._voice_at = 0.0
        self._ended = 0.0  # when end() was called
        self._finals_after_end = 0
        self._turn_done = False
        self._closing = False
        self._opened = 0.0
        self._ws = None
        self.sent = 0  # samples sent
        self._usage: dict = {}  # the last usageMetadata Google sent, if any
        self._billed = False  # into the cost tracker once, when the stream closes or fails

    # -- the caller's side

    def start(self) -> None:
        threading.Thread(target=self._run, name="gemini-stream", daemon=True).start()

    def feed(self, block: np.ndarray) -> None:
        """A block of 16 kHz float samples in [-1, 1]. Never blocks: the sending thread takes it from the queue."""
        block = np.asarray(block, dtype=np.float32)
        if not len(block) or self._ended:
            return
        if float(np.sqrt(np.mean(block * block))) >= VOICE_LEVEL:
            with self._cond:
                self._voice_at = self._clock()
        pcm = (np.clip(block, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm), FRAME_BYTES):  # Google asks for ~100 ms a message; the pre-roll comes as 2 s
            self._frames.put(pcm[i:i + FRAME_BYTES])

    def end(self) -> None:
        """The key was let go (and the tail recorded): Google is told the audio has ended, so it finalises at once."""
        with self._cond:
            if self._ended:
                return
            self._ended = self._clock()
            self._cond.notify_all()
        self._frames.put(_END)

    def final(self, timeout: float = 120.0) -> str | None:
        """The final text, or None when the stream couldn't give it (then the batch model must). Waits for end() (at most
        `timeout`), then FINAL_WAIT at most. Closes the stream."""
        try:
            with self._cond:
                deadline = self._clock() + timeout
                while not self._ended and not self.failed:
                    if not self._wait(deadline - self._clock()):
                        self._fail("the recording didn't end in time")
                deadline = self._ended + FINAL_WAIT
                while not self.failed and not self._done():
                    if not self._wait(min(deadline - self._clock(), SETTLE / 2)) and self._clock() >= deadline:
                        self._fail("no final text in time")
                if self.failed:
                    return None
                text = _joined(self._finals)
            if not text.strip():  # nothing heard: the batch model says so for sure, or finds what this missed
                self._fail("no words")
                return None
            log.info("Gemini Live: final text %.2f s after the release", self._clock() - self._ended)
            return text
        finally:
            self.close()

    def close(self) -> None:
        """Stop, without waiting for the connection to close (the text is typed meanwhile)."""
        with self._cond:
            if self._closing:
                return
            self._closing = True
            self._cond.notify_all()
        self._bill()
        self._frames.put(_END)
        ws = self._ws
        if ws is not None:
            threading.Thread(target=lambda: _quietly(ws.close), name="gemini-stream-close", daemon=True).start()

    # -- the connection

    def _run(self) -> None:
        try:
            ws = self._open()
            if self._closing:  # given up while connecting
                _quietly(ws.close)
                return
            threading.Thread(target=self._receive, args=(ws,), name="gemini-stream-in", daemon=True).start()
            self._send(ws)
        except Exception as e:
            if not self._closing:
                self._fail(self._describe(e))
                if self._opened == 0.0:
                    self._on_down()  # couldn't connect: the next dictations don't try for a minute

    def _open(self):
        ws = self._connect(f"{self.url}?key={self.key}")
        self._ws = ws
        config: dict = {"languageCodes": self.languages, "mode": "VERBATIM"}  # the voice pipeline does its own cleanup
        if self.words:
            config["customVocabulary"] = self.words
        ws.send(_json({"setup": {"model": f"models/{self.model}", "generationConfig": {"responseModalities": ["TEXT"]},
                                 "inputAudioTranscription": config}}))
        reply = _parse(ws.recv(timeout=CONNECT_WAIT))
        if "setupComplete" not in reply:
            raise ConnectionError(f"setup not accepted: {str(reply)[:200]}")
        self._opened = self._clock()
        return ws

    def _send(self, ws) -> None:
        while not self._closing and not self.failed:
            try:
                frame = self._frames.get(timeout=0.5)
            except queue.Empty:
                frame = None
            if self._clock() - self._opened > SESSION_LIMIT:
                raise TimeoutError("the dictation outlasted a Live session")
            if frame is _END:
                if self._ended and not self._closing:
                    ws.send(_json({"realtimeInput": {"audioStreamEnd": True}}))
                return
            if frame:
                ws.send(_json({"realtimeInput": {"audio": {"data": base64.b64encode(frame).decode("ascii"),
                                                           "mimeType": f"audio/pcm;rate={RATE}"}}}))
                self.sent += len(frame) // 2

    def _receive(self, ws) -> None:
        try:
            for raw in ws:
                self._handle(_parse(raw))
        except Exception as e:  # closed by Google (a refused key or model arrives this way), or by us
            if not self._closing:
                self._fail(self._describe(e))
            return
        if not self._closing:
            self._fail("Google closed the connection")

    def _handle(self, message: dict) -> None:
        if "goAway" in message:
            self._fail("Google ends the session")  # before the release: the batch model takes over, nothing is lost
            return
        content = message.get("serverContent") or {}
        now = self._clock()
        with self._cond:
            if isinstance(message.get("usageMetadata"), dict):
                self._usage = message["usageMetadata"]
            self._last_message = now
            final = content.get("inputTranscription") or {}
            if isinstance(final, dict) and isinstance(final.get("text"), str) and final["text"]:
                self._finals.append(final["text"])
                self._interim, self._last_final = False, now
                self._finals_after_end += bool(self._ended)
            interim = content.get("interimInputTranscription") or {}
            if isinstance(interim, dict) and interim.get("text"):
                self._interim = True  # only a sign that more words are coming: the guesses themselves are not used
            if (content.get("turnComplete") or content.get("generationComplete")) and self._ended:
                self._turn_done = True
            self._cond.notify_all()

    # -- under the lock

    def _done(self) -> bool:
        """Whether every word is final: Google said the turn is complete; or no guess is pending and either a final piece
        came after the release and nothing since, or the last final piece came after the voice had stopped a while."""
        if self._turn_done:
            return True
        if self._interim or not self._finals:
            return False
        now = self._clock()
        if self._finals_after_end and now - self._last_message >= SETTLE:
            return True
        return now - self._ended >= SETTLE and self._last_final > self._voice_at and now - self._voice_at >= COVERED

    def _wait(self, seconds: float) -> bool:
        if seconds <= 0:
            return False
        self._cond.wait(seconds)
        return True

    def _fail(self, reason: str) -> None:
        with self._cond:
            if self.failed:
                return
            self.failed = self._scrub(reason)
            self._cond.notify_all()
        log.warning("Gemini Live: %s; the recording goes to the batch model", self.failed)
        self._bill()  # the audio sent is billed all the same; the stream sends no more

    def _bill(self) -> None:
        """The stream into the cost tracker (sst.usage), once: the seconds sent at the Live API's 25 tokens a second,
        and the text written, by Google's count when it gave one, else by its length. Google's own count of the audio
        is only logged, to compare."""
        with self._cond:
            if self._billed or not self.sent:
                return
            self._billed = True
            seconds, meta, text = self.sent / RATE, dict(self._usage), _joined(self._finals)
        written = int(meta.get("responseTokenCount") or 0) or len(text) // usage.CHARS_PER_TOKEN
        if meta:
            log.info("Gemini Live: Google counted %s prompt and %s response tokens for %.1f s",
                     meta.get("promptTokenCount"), meta.get("responseTokenCount"), seconds)
        usage.record("live_dictation", self.model, audio_in=round(seconds * costs.LIVE_AUDIO), out=written,
                     seconds=seconds)

    def _scrub(self, text: str) -> str:
        return text.replace(self.key, "***") if self.key else text

    def _describe(self, error: Exception) -> str:
        close = getattr(error, "rcvd", None)  # Google closed it: its code and reason
        if close is not None:
            return self._scrub(f"Google closed the connection ({close.code}): {close.reason}")
        return self._scrub(f"{type(error).__name__}: {error}")


def _joined(pieces: list[str]) -> str:
    """The final pieces as one text: a space between them, none between Japanese or Chinese ones."""
    text = ""
    for piece in (p.strip() for p in pieces):
        if piece:
            text += piece if not text or (_CJK.match(piece[0]) and _CJK.match(text[-1])) else f" {piece}"
    return text


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _parse(raw) -> dict:
    data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    return data if isinstance(data, dict) else {}


def _quietly(close: Callable[[], object]) -> None:
    with contextlib.suppress(Exception):
        close()


# -- the WebSocket, as sst.live.gemini opens it (copied: dictation never imports the live pipeline)

ADDRESS_WAIT = 2.0  # seconds for each of Google's addresses to answer before the next one is tried


def _connect(url: str):
    from websockets.sync.client import connect
    parts = urlsplit(url)
    secure = parts.scheme == "wss"
    sock = None if urllib.request.getproxies() else _socket(parts.hostname, parts.port or (443 if secure else 80))
    return connect(url, sock=sock, open_timeout=CONNECT_WAIT, close_timeout=2, max_size=None, legacy=True, logger=_WS_LOG)


def _socket(host: str, port: int, each: float = ADDRESS_WAIT) -> socket.socket:
    """A connection to the first of the host's addresses that answers: from the dev laptop's network the first of
    Google's addresses never answers, and Python would wait out the whole timeout on it (sst.live.gemini)."""
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
