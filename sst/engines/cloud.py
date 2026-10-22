"""Speech recognition by a cloud provider (OpenAI, Groq, Google Gemini) with the user's own key, or by the user's own
server (vLLM, a company AI gateway, any server with OpenAI's transcription API).

The voice is sent to the provider: the window asks before a cloud model is used. OpenAI, Groq and own servers share
OpenAI's transcription API (a multipart upload of the recording); Gemini's Flash models get the recording inline in a
generateContent request, with an instruction to write down exactly what was said; Gemini 3's are asked to think as little
as they allow (sst.modelrules). Your words and the chosen language go along as hints.

Gemini Transcribe (gemini-3.5-transcribe), Google's dedicated speech-to-text model, gets no instruction: the recording
and an audioTranscriptionConfig. Always VERBATIM (fillers and false starts kept: the voice pipeline does its own
cleanup), with either word times ("timestamps", the default: merging overlapping chunks needs them) or Your words as
custom vocabulary ("vocabulary"): Google offers only one of the two at a time. A recording too big to send inline goes
through the Files API and is deleted afterwards. Whisper models (OpenAI's whisper-1, Groq's) give word times too;
OpenAI's gpt-transcribe, which replaces its older models (RETIRING), gives none.

Built never to lose a dictation: a connection is opened while the user speaks (prepare), within a few seconds whatever
the network does (sst.netconnect: the first address that answers), and when the provider can't be reached or fails, the
recording is transcribed on this computer instead (Parakeet), with a note saying so (`last_error`). A provider that
couldn't be reached is skipped for a minute, so being offline costs one wait, not one per dictation; one that refused
the key (no credit, quota used up, key not accepted) is skipped for a few minutes. The voice pipeline's chunks
(transcribe_chunk) never fall back here: the pipeline decides, and an error marked `permanent` isn't worth a retry.
They can be sent two or three at once, each on a kept-alive connection of its own.
"""
import base64
import difflib
import hashlib
import http.client
import io
import json
import logging
import re
import threading
import time
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np

from sst import netconnect
from sst.audio import TARGET_RATE, condition, resample
from sst.engines.whisper import LANGUAGES
from sst.modelrules import gemini_takes_temperature, gemini_thinking
from sst.pipeline.contracts import RawTranscript, WordInfo
from sst.pipeline.dictionary import speech_hints

CONNECT_BUDGET = netconnect.CONNECT_BUDGET  # seconds to connect, whatever the network does (name lookup included)
ANSWER_TIMEOUT = 10.0  # seconds, plus ANSWER_PER_SECOND for each second of audio (the upload is 32 KB a second)
ANSWER_PER_SECOND = 0.5
DOWN_FOR = 60.0  # after the provider couldn't be reached, go straight to Parakeet this long
RATE_LIMIT_WAITS = (10, 20, 30, 60)  # seconds; scoring reading tests waits out a provider's per-minute limit
IDLE_RECONNECT = 30.0  # servers drop idle connections; an older one is replaced, not used
POOL_SIZE = 4  # idle connections kept: the pipeline sends 2-3 chunks at once (ASRConfig.max_in_flight)
# A WAV up to this size goes inside the request (Gemini takes 20 MB, and base64 adds a third); a bigger one through the
# Files API. A 20 s chunk is 640 KB: only a recording of over 7 minutes transcribed whole is uploaded.
INLINE_LIMIT = 14_000_000
VOCABULARY_LIMIT = 100  # Your words given to Gemini Transcribe as custom vocabulary
FILE_POLL = 0.5  # seconds between looks at an uploaded file Google is still processing (audio is normally ready at once)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudProvider:
    key: str  # also its sst.gateway.PROVIDERS key: the API key is shared with AI cleanup
    name: str
    url: str
    api: str  # "openai": /audio/transcriptions (multipart); "gemini": generateContent with the audio inline
    models: tuple[str, ...]  # the first is the default
    key_page: str


CLOUD = {p.key: p for p in [
    CloudProvider("openai", "OpenAI", "https://api.openai.com/v1", "openai",
                  ("gpt-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"),
                  "https://platform.openai.com/api-keys"),
    CloudProvider("groq", "Groq", "https://api.groq.com/openai/v1", "openai",
                  ("whisper-large-v3-turbo", "whisper-large-v3"), "https://console.groq.com/keys"),
    # Explicit ids, not the -latest aliases: gemini-flash-latest moved to 3.5 Flash, the dearest Flash, and
    # gemini-flash-lite-latest's model isn't documented. A name typed in still works.
    CloudProvider("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta", "gemini",
                  ("gemini-3.5-transcribe", "gemini-3.5-flash-lite", "gemini-3.6-flash"), "https://aistudio.google.com/apikey"),
]}

# Models a provider is shutting down, and the day: still offered, and working, until then (retirement()).
RETIRING = {"openai": dict.fromkeys(("gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"), "26 February 2027")}
# Models that take the language as a list (languages[]), never along with the older single field.
LANGUAGE_LIST = re.compile(r"(^|/)gpt-transcribe", re.IGNORECASE)

# Your own server: its address (and a key, if it needs one) is the user's; the model is whatever the server offers.
SERVER = CloudProvider("server", "Your server", "", "openai", (), "")
REMOTE = {**CLOUD, SERVER.key: SERVER}  # every speech model that runs somewhere else

SPEECH = re.compile(r"whisper|transcri|speech|asr|parakeet|canary|voxtral|stt|audio", re.IGNORECASE)  # "Load models"

INSTRUCTION = ("Transcribe the speech in this recording exactly, word for word, with punctuation and capital letters. "
               "Output only the words that are spoken: no introduction, no notes, no translation. If nothing is spoken, "
               "output nothing.")
# The hints follow as a spelling reference. Phrased as plain data ("these may occur") a Flash model has written the
# list itself into the transcript (the owner's log, 2026-10-02), most often for a part with little speech.
HINTS = ("Spelling reference only, not part of the recording: if the speaker says one of these names or terms, spell it "
         "like this: {terms}. Never write any of them unless it is clearly spoken in the recording.")
ECHO_MIN_WORDS = 3  # this many hint words in a row, in the hints' own order, look like the hint list written out


class CloudError(Exception):
    def __init__(self, message: str, status: int = 0, permanent: bool = False, refusal: netconnect.Refusal | None = None,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status = status  # the HTTP status, when the provider answered
        self.permanent = permanent or refusal is not None  # asking again now won't help (the pipeline falls back at once)
        self.refusal = refusal  # no credit, quota used up, key not accepted: remembered (sst.netconnect)
        self.retry_after = retry_after  # seconds the provider asked to wait before a retry


class Unreachable(CloudError):
    pass


def retirement(provider: str, model: str) -> str:
    """What to tell the user about a model its provider is shutting down, or "" (whisper-1 on an own server is the
    server's own). The provider's default model replaces it."""
    day = RETIRING.get(provider, {}).get(model)
    return f"{CLOUD[provider].name} shuts {model} down on {day}: choose {CLOUD[provider].models[0]}" if day else ""


def wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    """16-bit mono WAV, what every provider accepts."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


class CloudEngine:
    def __init__(self, provider: str, api_key: str, model: str = "", language: str = "",
                 fallback: Callable[[], object] | None = None, url: str | None = None):
        if provider not in REMOTE:
            raise ValueError(f"Unknown cloud speech provider '{provider}'")
        if not api_key and provider in CLOUD:  # an own server may need none
            raise ValueError(f"{CLOUD[provider].name} needs an API key: enter it on the Speech recognition page.")
        if not (url or REMOTE[provider].url):
            raise ValueError("Your own server needs its address: enter it on the Speech recognition page.")
        self.provider = REMOTE[provider]
        self.name = provider
        # The app keeps the model, the key and a server's address up to date.
        self.model = model or (self.provider.models[0] if self.provider.models else "")
        self.api_key = api_key
        self.language = language  # "" = the provider detects it; the app keeps it up to date
        self.words: list[str] = []  # Your words, sent as a hint; the app keeps it up to date
        self.timestamp_mode = "timestamps"  # Gemini Transcribe (TIMESTAMP_MODES); the voice pipeline sets it
        self.last_error = ""  # set when the provider failed and the recording was transcribed on this computer
        self._fallback_loader, self._fallback = fallback, None
        self._down_until = 0.0
        self._idle: list[tuple[http.client.HTTPConnection, tuple, float]] = []  # (connection, target, last used)
        self._connecting = 0  # connections being opened ahead of time (prepare)
        self._pool = threading.Condition()  # guards the two above; requests themselves run without it
        self._plain_json: set[tuple[str, str]] = set()  # (server, model) that refused verbose_json: not asked again
        self.url = ""
        self.set_url(url or self.provider.url)

    def set_url(self, url: str) -> None:
        """Where requests go: an own server's address can change while it is in use. Taken from the next request on,
        so the window never waits for a dictation being transcribed; connections to the old address aren't reused."""
        url = url.strip().rstrip("/")
        if url != self.url:
            parts = urlsplit(url)
            self._target = (parts.scheme == "https", parts.hostname, parts.port, parts.path)
            self.url, self._down_until = url, 0.0

    def __repr__(self) -> str:  # never let the key reach a log
        return f"CloudEngine({self.name!r}, {self.model!r})"

    @property
    def title(self) -> str:
        return f"{self.provider.name} {self.model}"

    @property
    def biased(self) -> bool:
        """Whether Your words reach the model (sst eval says so): Gemini Transcribe takes them only as vocabulary."""
        return not self._transcribe_model() or self._mode == "vocabulary"

    @property
    def signature(self) -> str:
        words = hashlib.sha1("/".join(self.words).encode("utf-8")).hexdigest()[:10] if self.words else "none"
        where = self.name if self.name in CLOUD else f"{self.name}@{hashlib.sha1(self.url.encode()).hexdigest()[:8]}"
        if self._transcribe_model():
            mode = f"|verbatim:{self._mode}"  # the vocabulary changes the text
        elif self.provider.api == "gemini" and not gemini_takes_temperature(self.model):  # Gemini 3 asked to think less
            mode = f"|think:{gemini_thinking(self.model) or 'default'}"
        else:
            mode = ""  # requests as they always were
        return f"cloud|{where}|{self.model}|lang:{self.language or 'auto'}|words:{words}{mode}|peak-1"

    @property
    def _mode(self) -> str:
        return "vocabulary" if self.timestamp_mode == "vocabulary" else "timestamps"

    @property
    def _where(self) -> str:
        """The provider's host, as sst.netconnect remembers refusals (AI cleanup on the same host and key shares them)."""
        return f"{self._target[1]}:{self._target[2] or ''}"

    def _refused(self) -> str:
        """Why the provider refused this key or model a moment ago (sst.netconnect), or ""."""
        reason = netconnect.refused(self._where, self.api_key, self.model)
        return f"{reason} (a moment ago)" if reason else ""

    def _transcribe_model(self) -> bool:
        """Gemini Transcribe, Google's speech-to-text model (OpenAI's *-transcribe go through OpenAI's transcription API)."""
        return self.provider.api == "gemini" and "transcribe" in self.model

    def prepare(self) -> None:
        """Open a connection in the background while the user is still speaking, unless a fresh one is waiting."""
        target = self._target
        with self._pool:
            self._prune()
            if time.monotonic() < self._down_until or self._connecting or self._idle or self._refused():
                return
            self._connecting += 1  # counted now: a request made before the thread runs waits for it
        threading.Thread(target=self._prepare, args=(target,), name="cloud-connect", daemon=True).start()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """The provider's text; if it can't be had, Parakeet's on this computer, and `last_error` says why."""
        self.last_error = ""
        audio16 = resample(condition(audio), sample_rate, TARGET_RATE)
        try:
            if time.monotonic() < self._down_until:
                raise CloudError("it couldn't be reached a moment ago")
            if self._fallback_loader is not None and (refused := self._refused()):  # scoring asks anyway, and waits
                raise CloudError(refused)
            return self._ask_patiently(wav_bytes(audio16, TARGET_RATE), len(audio16) / TARGET_RATE)
        except TimeoutError:
            reason = "no answer in time"
        except Unreachable as e:
            self._down_until = time.monotonic() + DOWN_FOR
            reason = str(e)
        except (CloudError, OSError, http.client.HTTPException) as e:
            reason = str(e) or type(e).__name__
        log.warning("%s failed (%s); transcribing on this computer", self.title, reason)
        if self._fallback_loader is None:
            raise CloudError(f"{self.provider.name}: {reason}")
        try:
            if self._fallback is None:
                self._fallback = self._fallback_loader()
        except Exception as e:  # e.g. Parakeet isn't downloaded (a cloud-only user)
            log.warning("No speech model to fall back on: %s", e)
            raise CloudError(f"{self.provider.name}: {reason} ({e})") from None
        if hasattr(self._fallback, "words"):
            self._fallback.words = speech_hints(self.words)
        self.last_error = f"{self.provider.name}: {reason}"
        return self._fallback.transcribe(audio, sample_rate)

    def transcribe_chunk(self, audio: np.ndarray, sample_rate: int) -> RawTranscript:
        """One chunk of the voice pipeline, with word times (seconds from the chunk's start) when the model gives them.
        Never falls back on Parakeet and never waits out a rate limit: the pipeline retries and decides. Raises
        CloudError (.status: the HTTP status, 0 without an answer), OSError or TimeoutError. Thread-safe: chunks may be
        sent at the same time."""
        if time.monotonic() < self._down_until:
            raise CloudError("it couldn't be reached a moment ago", permanent=True)  # retrying would fail the same way
        if refused := self._refused():
            raise CloudError(refused, permanent=True)
        audio16 = resample(condition(audio), sample_rate, TARGET_RATE)
        try:
            return self._ask(wav_bytes(audio16, TARGET_RATE), len(audio16) / TARGET_RATE, timed=True)
        except Unreachable:
            self._down_until = time.monotonic() + DOWN_FOR  # the next chunks fail at once, as dictation does
            raise
        except http.client.HTTPException as e:  # a broken answer (not an OSError): readable, like the rest
            raise CloudError(f"a broken answer ({type(e).__name__})") from None

    def check(self, audio: np.ndarray, sample_rate: int) -> str:
        """For the Test button: one recording through the provider, never Parakeet. Raises with a readable reason. A
        model being retired says so."""
        audio16 = resample(condition(audio), sample_rate, TARGET_RATE)
        t0 = time.perf_counter()
        try:
            text = self.ask(wav_bytes(audio16, TARGET_RATE), len(audio16) / TARGET_RATE)
        except TimeoutError:
            raise CloudError("no answer in time") from None
        except (OSError, http.client.HTTPException) as e:
            raise CloudError(f"could not reach {self.provider.name}: {e}") from None
        note = retirement(self.name, self.model)
        return (f"{self.model} answered in {time.perf_counter() - t0:.1f} s: {text or '(nothing recognised)'}"
                + (f" ({note})" if note else ""))

    def _ask_patiently(self, wav: bytes, seconds: float) -> str:
        """ask(), waiting out a rate limit (HTTP 429) when nothing falls back: scoring reading tests sends many
        recordings in a row. Dictation doesn't wait; Parakeet types at once."""
        waits = list(RATE_LIMIT_WAITS) if self._fallback_loader is None else []
        while True:
            try:
                return self.ask(wav, seconds)
            except CloudError as e:
                if e.status != 429 or not waits or (e.refusal and not e.refusal.rate_limit):  # a used-up quota: no use
                    raise
                wait = waits.pop(0)
                log.info("%s is rate limited; trying again in %d s", self.title, wait)
                time.sleep(wait)

    def ask(self, wav: bytes, seconds: float) -> str:
        """The provider's transcript of one WAV recording. Raises CloudError with a readable reason, or OSError."""
        return self._ask(wav, seconds).text

    def _ask(self, wav: bytes, seconds: float, timed: bool = False) -> RawTranscript:
        """The provider's transcript. `timed` asks a Whisper model for word times too (verbose_json: the same text,
        a bigger answer). Gemini Transcribe's request is the same either way, so `sst eval` scores what dictation gets.

        If the text holds the hint list itself (hint_echo), the recording is transcribed once more without hints: when
        those words are gone, the model had copied them from the prompt, and the plain transcript is used."""
        hints = speech_hints(self.words)
        raw = self._ask_once(wav, seconds, timed, hints)
        echo = hint_echo(raw.text, hints)
        if echo is None:
            return raw
        copied = raw.text[echo[0]:echo[1]]
        try:
            plain = self._ask_once(wav, seconds, timed, [])
        except (CloudError, OSError, http.client.HTTPException) as e:  # no second opinion: drop the run itself
            log.warning("%s wrote the hint list into a transcript; removed it (no check possible: %s)", self.title, e)
            raw.text, raw.words = without(raw.text, echo), []
            raw.diagnostics["hint_echo"] = "removed"
            return raw
        if still_there(copied, plain.text):  # said for real: keep the transcript made with the spelling help
            raw.diagnostics["hint_echo"] = "spoken"
            return raw
        log.warning("%s wrote the hint list into a transcript; used the transcript made without hints", self.title)
        plain.diagnostics["hint_echo"] = "replaced"
        return plain

    def _ask_once(self, wav: bytes, seconds: float, timed: bool, hints: list[str]) -> RawTranscript:
        timeout = ANSWER_TIMEOUT + ANSWER_PER_SECOND * seconds
        backend, model, key, where = self.title, self.model, self.api_key, self._where
        try:
            if self._transcribe_model():
                raw = self._ask_transcribe(wav, timeout, hints)
            elif self.provider.api == "gemini":
                raw = self._ask_gemini(wav, timeout, hints)
            else:
                raw = self._ask_openai(wav, timeout, timed and self._asks_word_times(), hints)
        except CloudError as e:
            if e.refusal is not None:  # the next chunks and dictations go straight to Parakeet (and AI cleanup's backup)
                netconnect.refuse(where, key, model, e.refusal)
                log.warning("%s refused: %s; not asked again for %.0f s", backend, e.refusal.reason, e.refusal.seconds)
            raise
        netconnect.forgive(where, key, model)
        raw.text, raw.backend = " ".join(raw.text.split()), backend
        raw.language = raw.language or self.language
        return raw

    def _ask_gemini(self, wav: bytes, timeout: float, hints: list[str]) -> RawTranscript:
        """A Flash model: the recording inline, with an instruction (and the hints, as a spelling reference)."""
        instruction = INSTRUCTION
        if self.language:
            instruction += f" The speech is in {LANGUAGES.get(self.language, self.language)}."
        if hints:
            instruction += " " + HINTS.format(terms=", ".join(f'"{h}"' for h in hints))
        body = json.dumps({"contents": [{"parts": [
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav).decode("ascii")}},
            {"text": instruction}]}], "generationConfig": self._flash_config()}).encode("utf-8")
        status, data, reply = self._request(f"/models/{self.model}:generateContent", body, self._google(), timeout)
        answer = _json(status, data, reply)
        parts = ((answer.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        return RawTranscript("".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")))

    def _flash_config(self) -> dict:
        """A Flash model's generationConfig: Gemini 3 thinks as little as it allows ("medium" unless told: slower, and
        3-4 times the cost) at its own temperature; an older model writes at temperature 0 (sst.modelrules)."""
        config = {"temperature": 0} if gemini_takes_temperature(self.model) else {}
        if thinking := gemini_thinking(self.model):
            config["thinkingConfig"] = {"thinkingLevel": thinking}
        return config

    # ---- Gemini Transcribe (the owner's plan, sections 18-21): upload, transcribe and delete kept apart

    def _ask_transcribe(self, wav: bytes, timeout: float, hints: list[str] | None = None) -> RawTranscript:
        mode = self._mode
        if len(wav) <= INLINE_LIMIT:
            transport = "inline"
            answer = self.transcribe_audio(
                {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav).decode("ascii")}}, timeout, mode,
                hints)
        else:
            transport = "files"
            file = self.upload_audio(wav, timeout)
            try:
                answer = self.transcribe_audio({"fileData": {"mimeType": "audio/wav", "fileUri": file["uri"]}}, timeout,
                                               mode, hints)
            finally:
                self.delete_remote_file(str(file.get("name") or ""))
        text, words, language = gemini_transcript(answer)
        return RawTranscript(text, words, language, diagnostics={"mode": mode, "transport": transport})

    def transcribe_audio(self, audio: dict, timeout: float, mode: str | None = None, hints: list[str] | None = None) -> dict:
        """Gemini Transcribe's answer for one recording, given inline ({"inlineData": ...}) or uploaded
        ({"fileData": ...}). No instruction: the config says what to do. VERBATIM, since SMART mode (cleaned up)
        allows no word times and the pipeline cleans up itself."""
        config = {"mode": "VERBATIM", "languageCodes": [self.language] if self.language else [], "wordTimestamp": True}
        if (mode or self._mode) == "vocabulary":  # Google: custom vocabulary works only without word times
            config["wordTimestamp"] = False
            if vocabulary := vocabulary_list(speech_hints(self.words) if hints is None else hints):
                config["customVocabulary"] = vocabulary
        body = json.dumps({"contents": [{"parts": [audio]}], "generationConfig": {"audioTranscriptionConfig": config}})
        status, data, reply = self._request(f"/models/{self.model}:generateContent", body.encode("utf-8"), self._google(),
                                            timeout)
        return _json(status, data, reply)

    def upload_audio(self, wav: bytes, timeout: float) -> dict:
        """The recording into Gemini's Files API (a resumable upload, sent in one piece): the file's {"name", "uri"}."""
        base = urlsplit(self.url)
        start = f"{base.scheme}://{base.netloc}/upload{base.path}/files"  # .../upload/v1beta/files
        headers = {**self._google(), "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                   "X-Goog-Upload-Header-Content-Length": str(len(wav)), "X-Goog-Upload-Header-Content-Type": "audio/wav"}
        status, data, reply = self._request(start, json.dumps({"file": {"display_name": "rflow-chunk"}}).encode("utf-8"),
                                            headers, timeout)
        if status != 200:
            _json(status, data, reply)  # raises with the provider's reason
        address = reply.get("x-goog-upload-url")
        if not address:
            raise CloudError("the Files API gave no upload address", status)
        # The upload address carries its own authorisation: the key isn't sent again.
        status, data, _ = self._request(address, wav, {"X-Goog-Upload-Command": "upload, finalize",
                                                       "X-Goog-Upload-Offset": "0"}, timeout)
        file = _json(status, data).get("file")
        if not isinstance(file, dict) or not file.get("uri"):
            raise CloudError("the Files API kept no file", status)
        deadline = time.monotonic() + timeout
        while file.get("state") == "PROCESSING" and file.get("name") and time.monotonic() < deadline:
            time.sleep(FILE_POLL)
            status, data, _ = self._request(f"/{file['name']}", None, self._google(), timeout, method="GET")
            file = {**file, **_json(status, data)}
        return file

    def delete_remote_file(self, name: str) -> None:
        """Remove an uploaded recording from Google at once (it would stay 48 hours). Best effort: never raises."""
        if not name:
            return
        try:
            status, data, _ = self._request(f"/{name}", None, self._google(), ANSWER_TIMEOUT, method="DELETE")
            if status not in (200, 204):
                log.info("%s kept %s: HTTP %d %s", self.provider.name, name, status, data[:120])
        except Exception as e:
            log.info("Could not delete %s from %s: %s", name, self.provider.name, e)

    # ---- OpenAI's transcription API (OpenAI, Groq, own servers)

    def _asks_word_times(self) -> bool:
        """Whisper models give word times; gpt-transcribe and gpt-4o-*-transcribe only plain JSON. An own server is asked
        only for a Whisper model, and no more once it refused."""
        return "whisper" in self.model.lower() and (self.url, self.model) not in self._plain_json

    def _ask_openai(self, wav: bytes, timeout: float, timed: bool, hints: list[str]) -> RawTranscript:
        words = ", ".join(hints)
        fields = {"model": self.model} if self.model else {}
        fields["response_format"] = "verbose_json" if timed else "json"
        if timed:
            fields["timestamp_granularities[]"] = "word"
        if self.language:  # gpt-transcribe takes a list, and OpenAI says not to send both
            fields["languages[]" if LANGUAGE_LIST.search(self.model) else "language"] = self.language
        if words:
            fields["prompt"] = f"Names and terms: {words}."
        body, content_type = _multipart(fields, "recording.wav", wav)
        status, data, reply = self._request("/audio/transcriptions", body, {"Content-Type": content_type, **self._auth()},
                                            timeout)
        if timed and self.name == SERVER.key and status in (400, 422):  # FastAPI-based servers say 422
            # A server that answers only plain JSON: ask again without word times. Once that works, this server and
            # model aren't asked for them again (a 400 for another reason fails the plain request too).
            raw = self._ask_openai(wav, timeout, False, hints)
            self._plain_json.add((self.url, self.model))
            log.info("%s gives no word times (HTTP %d); asking for plain JSON from now on", self.title, status)
            return raw
        answer = _json(status, data, reply)
        return RawTranscript(str(answer.get("text") or ""), _words(answer.get("words")) if timed else [],
                             _language_code(answer.get("language")), diagnostics={"format": fields["response_format"]})

    def models(self) -> list[str]:
        """For "Load models": the server's models, the speech ones first (a gateway lists its chat models too).
        Raises CloudError with a readable reason."""
        try:
            status, data, _ = self._request("/models", None, self._auth(), 10.0, method="GET")
        except TimeoutError:
            raise CloudError("no answer in time") from None
        except (OSError, http.client.HTTPException) as e:
            raise CloudError(f"could not reach the server: {e}") from None
        answer = _json(status, data)
        ids = [m["id"].removeprefix("models/") for m in answer.get("data") or []
               if isinstance(m, dict) and isinstance(m.get("id"), str)]
        return sorted(ids, key=lambda name: (not SPEECH.search(name), name.lower()))

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}  # an own server may need no key

    def _google(self) -> dict:
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

    # ---- HTTP: a small pool of kept-alive connections, so two or three chunks can be transcribed at once

    def _request(self, path: str, body: bytes | None, headers: dict, timeout: float,
                 method: str = "POST") -> tuple[int, str, http.client.HTTPMessage]:
        """One request: `path` is under the base URL ("/models"), or a whole URL (the Files API's). A connection is
        taken from the pool and given back after the answer: only that takes the lock, never the request."""
        target = self._target
        if path.startswith(("http://", "https://")):
            parts = urlsplit(path)
            where = (parts.scheme == "https", parts.hostname, parts.port)
            path = parts.path + (f"?{parts.query}" if parts.query else "")
        else:
            where, path = target[:3], target[3] + path
        pooled = where == target[:3]  # an address on another host gets a connection of its own, closed afterwards
        for attempt in (1, 2):
            conn = self._checkout(target) if pooled else None
            if conn is None:
                conn = self._connect(target if pooled else (*where, ""))
            conn.sock.settimeout(timeout)
            try:
                conn.request(method, path, body=body, headers=headers)
                response = conn.getresponse()
                data = response.read().decode("utf-8", "replace")
            except (http.client.RemoteDisconnected, ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                conn.close()  # the server dropped a kept-alive connection: reconnect once
                if attempt == 2:
                    raise
                continue
            except BaseException:
                conn.close()
                raise
            if pooled and not response.will_close:
                self._checkin(conn, target)
            else:
                conn.close()
            return response.status, data, response.headers
        raise AssertionError("unreachable")

    def _checkout(self, target: tuple) -> http.client.HTTPConnection | None:
        """An idle connection to `target`, or None (the caller opens one). One being opened ahead of time is waited
        for rather than opening a second, as when a single connection was kept."""
        deadline = time.monotonic() + CONNECT_BUDGET + netconnect.TLS_LEAST
        with self._pool:
            while True:
                self._prune()
                if target != self._target:
                    return None
                if self._idle:
                    return self._idle.pop()[0]  # the most recently used: the least likely to have been dropped
                left = deadline - time.monotonic()
                if not self._connecting or left <= 0:
                    return None
                self._pool.wait(left)

    def _checkin(self, conn: http.client.HTTPConnection, target: tuple) -> None:
        with self._pool:
            if target == self._target and len(self._idle) < POOL_SIZE:
                self._idle.append((conn, target, time.monotonic()))
                self._pool.notify()
                return
        conn.close()

    def _prune(self) -> None:
        """(Under the lock.) Drop idle connections the server may have closed, or to an address no longer chosen."""
        now, keep = time.monotonic(), []
        for conn, target, used in self._idle:
            if target == self._target and now - used <= IDLE_RECONNECT:
                keep.append((conn, target, used))
            else:
                conn.close()
        self._idle = keep

    def _prepare(self, target: tuple) -> None:
        conn = None
        try:
            conn = self._connect(target)
        except Unreachable as e:  # the dictation finds out too, and uses Parakeet at once
            self._down_until = time.monotonic() + DOWN_FOR
            log.info("Could not connect to %s ahead of time: %s", self.provider.name, e)
        finally:
            with self._pool:  # given back and counted off together, so a waiting request finds it
                self._connecting -= 1
                if conn is not None and target == self._target:
                    self._idle.append((conn, target, time.monotonic()))
                    conn = None
                self._pool.notify_all()
            if conn is not None:
                conn.close()

    def _connect(self, target: tuple) -> http.client.HTTPConnection:
        https, host, port, _ = target
        try:
            return netconnect.http_connection(https, host, port, CONNECT_BUDGET)
        except OSError as e:  # every address was tried within the budget: asking again now won't do better
            raise Unreachable(f"cannot reach {host}: {e}", permanent=True) from None


def _norm(word: str) -> str:
    return re.sub(r"\W", "", word.replace("\u2019", "'").casefold())


_WORD = re.compile(r"[\w'\u2019-]+")


def hint_echo(text: str, hints: list[str]) -> tuple[int, int] | None:
    """Where a transcript holds the hint list itself: the character span of the longest run of ECHO_MIN_WORDS or more
    words in a row (punctuation between them allowed) that are hint words in the hints' own order, or None. A person
    rarely says their dictionary's terms one after another in its order; a model copying its prompt does exactly
    that. A plural or singular counts as the same word ("commits" for "commit")."""
    order: dict[str, list[int]] = {}
    for n, word in enumerate(w for term in hints for w in term.split() if _norm(w)):
        order.setdefault(_norm(word), []).append(n)
    if sum(len(v) for v in order.values()) < ECHO_MIN_WORDS:
        return None
    tokens = list(_WORD.finditer(text))
    best, start, last, count = None, 0, -1, 0
    for i, token in enumerate(tokens):
        key = _norm(token.group())
        alternative = key[:-1] if key.endswith("s") else key + "s"
        positions = sorted(set(order.get(key, []) + order.get(alternative, [])))
        following = next((p for p in positions if p > last), None) if count else None
        if following is not None:
            last, count = following, count + 1
        elif positions:  # a new run starts here
            start, last, count = i, positions[0], 1
        else:
            count = 0
        if count >= ECHO_MIN_WORDS and (best is None or count > best[2]):
            best = (start, i, count)
    return (tokens[best[0]].start(), tokens[best[1]].end()) if best else None


def still_there(copied: str, text: str) -> bool:
    """Whether most of a suspected copied run is in a transcript made without hints, in the same order: then it was
    really said (spelled however the model hears it without help)."""
    want = [_norm(w) for w in _WORD.findall(copied)]
    have = [_norm(w) for w in _WORD.findall(text)]
    matched = sum(block.size for block in difflib.SequenceMatcher(None, want, have, autojunk=False).get_matching_blocks())
    return bool(want) and matched / len(want) >= 0.6


def without(text: str, span: tuple[int, int]) -> str:
    """The text with a span taken out and the punctuation around the gap tidied ("it, . Then" -> "it. Then")."""
    text = text[:span[0]] + " " + text[span[1]:]
    text = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split()))
    text = re.sub(r"([,;:])+\s*([.!?])", r"\2", text)
    text = re.sub(r"([.!?])(\s*[.,;:])+", r"\1", text)
    return text.strip(" ,;:")


def vocabulary_list(words: list[str]) -> list[str]:
    """Your words as Gemini Transcribe's custom vocabulary: each once (whatever its case), at most VOCABULARY_LIMIT."""
    unique: dict[str, str] = {}
    for word in words:
        word = " ".join(str(word).split())
        if word:
            unique.setdefault(word.casefold(), word)
    return list(unique.values())[:VOCABULARY_LIMIT]


def gemini_transcript(answer: dict) -> tuple[str, list[WordInfo], str]:
    """Text, word times and language from a Gemini Transcribe answer. Its REST shape isn't fully documented, so this
    reads it leniently: the text from the candidate's non-thought text parts (else an audioTranscription's own text,
    else its words), the words from any part's audioTranscription. A blocked, empty or cut-off answer raises
    CloudError; a finished one with nothing in it is silence ("")."""
    candidates = answer.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        feedback = answer.get("promptFeedback") or answer.get("prompt_feedback") or {}
        reason = (feedback.get("blockReason") or feedback.get("block_reason")) if isinstance(feedback, dict) else None
        raise CloudError(f"Gemini blocked the recording ({reason})" if reason else "Gemini gave no transcript at all")
    candidate = candidates[0]
    finish = str(candidate.get("finishReason") or candidate.get("finish_reason") or "STOP")
    if finish not in ("STOP", "FINISH_REASON_UNSPECIFIED"):  # SAFETY, RECITATION... or MAX_TOKENS (cut off)
        raise CloudError(f"Gemini stopped transcribing ({finish})")
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    texts, others, words, language = [], [], [], ""
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict) or part.get("thought"):
            continue
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
        found = part.get("audioTranscription") or part.get("audio_transcription")
        if isinstance(found, dict):
            other = found.get("text") or found.get("transcript")
            if isinstance(other, str):
                others.append(other)
            words += _words(found.get("words"))
            language = language or str(found.get("languageCode") or found.get("language_code") or "")
    text = "".join(texts).strip() or " ".join(others).strip() or " ".join(w.text for w in words)
    return text, words, language


def _seconds(value) -> float | None:
    """A time as the providers write it: 1.5, "1.5", "1.500s" (Google's Duration), or {"seconds": 1, "nanos": 5e8}."""
    try:
        if isinstance(value, dict):
            return float(value.get("seconds") or 0) + float(value.get("nanos") or 0) / 1e9
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            return float(value.strip().removesuffix("s"))
    except (TypeError, ValueError):
        pass
    return None


def _first(item: dict, *keys: str):
    return next((item[k] for k in keys if item.get(k) is not None), None)


def _words(items) -> list[WordInfo]:
    """Word times from an answer: OpenAI's [{word, start, end}] or Google's [{word, startOffset, endOffset}] (snake_case
    too). All or nothing: a list with a word lacking its time can't be lined up with the text, so it counts as none."""
    words = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            return []
        text = str(_first(item, "word", "text") or "").strip()
        start = _seconds(_first(item, "start", "startOffset", "start_offset", "startTime", "start_time"))
        end = _seconds(_first(item, "end", "endOffset", "end_offset", "endTime", "end_time"))
        if start is None:
            return []
        if text:
            confidence = _first(item, "confidence", "probability")
            words.append(WordInfo(text, start, max(start, end if end is not None else start),
                                  float(confidence) if isinstance(confidence, int | float) else None))
    return words


def _language_code(name) -> str:
    """OpenAI's verbose_json names the language ("english"); Rflow uses codes ("en"). Anything else is kept."""
    if not isinstance(name, str):
        return ""
    codes = {title.lower(): code for code, title in LANGUAGES.items() if code}
    return codes.get(name.strip().lower(), name.strip())


def _multipart(fields: dict[str, str], filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
              "Content-Type: audio/wav\r\n\r\n".encode())
    out.write(data)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def _json(status: int, data: str, headers=None) -> dict:
    """The answer, or CloudError: one that says no credit, quota used up or key not accepted says so plainly and carries
    the refusal (sst.netconnect); a short rate limit carries the wait the provider asked for."""
    found = netconnect.refusal(status, data, headers) if status != 200 else None
    try:
        answer = json.loads(data)
    except ValueError:
        if found:
            raise CloudError(f"{found.reason} (HTTP {status})", status, refusal=found) from None
        raise CloudError(f"HTTP {status}, not JSON") from None
    if isinstance(answer, list) and answer and isinstance(answer[0], dict) and status != 200:
        answer = answer[0]  # an error wrapped in a list (Google's OpenAI-compatible endpoints)
    if not isinstance(answer, dict):
        raise CloudError(f"HTTP {status}, an unexpected answer", refusal=found)
    if status != 200 or "error" in answer:
        error = answer.get("error") or answer.get("message") or data[:120]
        if isinstance(error, dict):
            error = error.get("message") or error
        detail = f"HTTP {status} {str(error)[:200]}"
        if found:  # said plainly first, the provider's own words after
            raise CloudError(f"{found.reason} ({detail})", status, refusal=found)
        raise CloudError(detail, status, retry_after=netconnect.retry_after(data, headers) if status == 429 else None)
    return answer
