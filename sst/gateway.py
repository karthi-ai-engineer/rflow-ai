"""Text cleanup with an AI model from the provider the user chooses: OpenAI, Anthropic, Google Gemini, Groq, Ollama,
vLLM or any other OpenAI-compatible server (a company AI gateway, LM Studio...). The provider, address, key and models
are the user's settings; each provider gets the request format it understands (PROVIDERS).

Parakeet transcribes on the laptop; only the finished text goes to the endpoint. Built never to hold up typing:
if the chosen model fails or is slow the backup model is tried (Polisher.routes), an unreachable endpoint gives the text
as heard within a few seconds (sst.netconnect) and is skipped for a minute, a provider that refused the key (no credit,
quota used up, key not accepted) is skipped for a few minutes, and the connection is opened while the user is still
speaking. Standard library only; connections go direct (no proxy).
"""
import base64
import ctypes
import dataclasses
import http.client
import json
import logging
import re
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from sst import modelrules, netconnect, usage
from sst.settings import CONFIG_DIR

GATEWAY_FILE = CONFIG_DIR / "gateway.json"  # the API keys, encrypted for the Windows user; never in git
DEFAULT_URL = ""  # the user chooses a provider, or enters their own server's address
# The speech model on the user's own server (sst.engines.cloud.SERVER) keeps its address and key under this name,
# apart from AI cleanup's own server: the two may differ, and changing one must never break the other.
SPEECH_SERVER = "speech-server"
ANTHROPIC_VERSION = "2023-06-01"  # the Messages API version header Anthropic requires

CONNECT_BUDGET = netconnect.CONNECT_BUDGET  # seconds to connect, whatever the network does (name lookup included)
MAX_ASKED = 2            # models asked for one text: the chosen one and one backup (one that refused isn't asked)
ANSWER_TIMEOUT = 2.0     # seconds, plus ANSWER_PER_WORD for each word; a normal sentence takes 0.5-0.7 s
ANSWER_PER_WORD = 0.04
DOWN_FOR = 60.0          # after the gateway couldn't be reached (e.g. off the office network), skip cleanup this long
# Seconds for a Text Transform answer, plus TRANSFORM_PER_WORD for each word: the whole answer is written before the first
# byte comes back, and a 350-word part rewritten as Professional is ~500 tokens (8 s + 0.05 s a word timed out on slow
# answers; the user waits for this one on purpose).
TRANSFORM_TIMEOUT = 15.0
TRANSFORM_PER_WORD = 0.1
IDLE_RECONNECT = 30.0    # servers drop idle connections; refresh an older one while the user is speaking
# The answer's token limit: a cleanup keeps the words; Text Transform and Translate may write more (Tamil or Japanese take
# several tokens a word). An answer that reaches it is cut off, and never used.
ANSWER_TOKENS, ANSWER_TOKENS_PER_WORD = 64, 3
TRANSFORM_TOKENS, TRANSFORM_TOKENS_PER_WORD = 256, 8
THINKING_ROOM = 2000     # tokens added to the limit for a model that thinks first: its thinking counts against it

SYSTEM_PROMPT = (
    "You clean up text from a speech recognizer. Fix recognition errors using the context and the user's vocabulary "
    "(a word that sounds like a vocabulary word usually is that word), add punctuation and capital letters, and remove "
    "filler words such as um, uh and you know. Keep the user's own wording and meaning; do not rephrase, summarize, add "
    "anything or answer questions in the text. Output only the cleaned text.")

# Models in a provider's list that don't write text (speech, images, embeddings...): left out of "Load models".
# Orpheus is Groq's text-to-speech; prompt-guard and gpt-oss-safeguard are classifiers.
NOT_FOR_TEXT = re.compile(r"whisper|tts|orpheus|transcribe|dall-e|image|embed|moderation|realtime|audio|guard|search|"
                          r"babbage|davinci|computer-use|imagen|veo|aqa", re.IGNORECASE)

log = logging.getLogger(__name__)


class Unreachable(OSError):
    pass


class GatewayError(Exception):
    def __init__(self, message: str, refusal: netconnect.Refusal | None = None):
        super().__init__(message)
        self.refusal = refusal  # no credit, quota used up, key not accepted (sst.netconnect)


@dataclass(frozen=True)
class Provider:
    key: str
    name: str
    url: str = ""  # its usual address; "" = the user enters their server's address
    api: str = "openai"  # "openai": OpenAI-compatible /chat/completions; "anthropic": Anthropic's /messages
    needs_key: bool = True
    own_server: bool = False  # run by the user: the address can be changed, and options for self-hosted models are sent
    key_page: str = ""  # where to get an API key
    hint: str = ""  # a model to suggest before the list is loaded


PROVIDERS = {p.key: p for p in [
    Provider("openai", "OpenAI", "https://api.openai.com/v1", key_page="https://platform.openai.com/api-keys",
             hint="e.g. gpt-4o-mini or gpt-4.1-mini (fast chat models)"),
    Provider("anthropic", "Anthropic", "https://api.anthropic.com/v1", api="anthropic",
             key_page="https://console.anthropic.com/settings/keys", hint="e.g. a Haiku model (the fastest)"),
    Provider("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
             key_page="https://aistudio.google.com/apikey", hint="e.g. gemini-3.5-flash-lite (fast and cheap)"),
    Provider("groq", "Groq", "https://api.groq.com/openai/v1", key_page="https://console.groq.com/keys",
             hint="e.g. openai/gpt-oss-20b (fast and cheap)"),
    Provider("ollama", "Ollama (on this computer)", "http://localhost:11434/v1", needs_key=False, own_server=True,
             hint="a model you have pulled, e.g. llama3.2"),
    Provider("vllm", "vLLM or another OpenAI-compatible server", needs_key=False, own_server=True,
             hint="choose after Load models"),
]}
# The providers Rflow offers (the owner's decision of 2026-10-23): Google Gemini, Ollama on this PC and the user's own
# server. OpenAI, Anthropic and Groq stay in PROVIDERS (and their speech models in sst.engines) for a later phase, but
# the window, the setups and the welcome show only these: adding a key here brings a provider back everywhere.
OFFERED = ("gemini", "ollama", "vllm")


def offered(key: str) -> bool:
    """Whether the window may offer a provider, or a speech model by its SPEECH_MODELS key (Parakeet, Whisper and the
    user's own server always: they aren't a provider's)."""
    return key not in PROVIDERS or key in OFFERED


def provider_for(url: str) -> str:
    """The provider an address belongs to, for settings from before the provider choice."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    for key, provider in PROVIDERS.items():
        if provider.url and not provider.own_server and host == urlsplit(provider.url).hostname:
            return key
    if host in ("localhost", "127.0.0.1") and parts.port == 11434:
        return "ollama"
    return "vllm"


@dataclass
class GatewayConfig:
    """The chosen provider, its address and the user's key. On disk the keys are encrypted for the Windows user (DPAPI),
    so only that user on this laptop can read them; a key pasted into the file by hand ("api_key") is encrypted on
    first load. `others` keeps the other providers' address and key, so switching back and forth loses nothing."""

    base_url: str = DEFAULT_URL
    api_key: str = ""
    provider: str = ""  # a PROVIDERS key; "" = worked out from the address (settings from before the provider choice)
    others: dict[str, tuple[str, str]] = field(default_factory=dict)  # provider -> (address, key)

    @property
    def service(self) -> Provider:
        return PROVIDERS.get(self.provider) or PROVIDERS[provider_for(self.base_url)]

    @property
    def address(self) -> str:
        """Where requests go: the address given, or the cloud provider's usual one."""
        return (self.base_url or self.service.url).rstrip("/")

    def __repr__(self) -> str:  # never let a key reach a log
        return (f"GatewayConfig(provider={self.service.key!r}, base_url={self.base_url!r}, "
                f"api_key={'set' if self.api_key else 'missing'})")

    @property
    def chosen(self) -> str:
        """AI cleanup's provider, or "" while none is chosen. A key or server saved before then is kept with the
        others, so saving one never chooses the cleanup's provider (the user testing's M-04)."""
        return self.service.key if self.provider or self.base_url or self.api_key else ""

    def entries(self) -> dict[str, tuple[str, str]]:
        """Every provider's (address, key): the chosen one's and the others'."""
        entries = dict(self.others)
        if self.base_url or self.api_key:
            entries[self.service.key] = (self.base_url, self.api_key)
        return entries

    def key_for(self, provider: str) -> str:
        """The user's key for a provider: one per provider, which speech, AI cleanup and live translation all use."""
        return self.api_key if provider == self.chosen else self.others.get(provider, ("", ""))[1]

    def with_key(self, provider: str, key: str) -> "GatewayConfig":
        """A copy with this provider's key changed (Your API keys, a cloud speech model); AI cleanup's choice of
        provider and address stays as it is."""
        if provider == self.chosen:
            return dataclasses.replace(self, api_key=key)
        return self.with_entry(provider, self.others.get(provider, ("", ""))[0], key)

    def with_entry(self, provider: str, url: str, key: str) -> "GatewayConfig":
        """A copy with this provider's address and key changed (or the speech server's: SPEECH_SERVER)."""
        if provider == self.chosen:
            return dataclasses.replace(self, base_url=url, api_key=key)
        others = {name: entry for name, entry in self.others.items() if name != provider}
        if url or key:
            others[provider] = (url, key)
        return dataclasses.replace(self, others=others)

    def speech_server(self) -> tuple[str, str]:
        """The address and key of the speech model on the user's own server ("" when none is set up)."""
        return self.others.get(SPEECH_SERVER, ("", ""))

    @classmethod
    def load(cls, path: Path = GATEWAY_FILE) -> "GatewayConfig":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            base_url = str(data.get("base_url") or DEFAULT_URL).strip()
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError, AttributeError) as e:
            log.warning("Ignoring unreadable %s: %s", path, e)
            return cls()
        provider = data.get("provider") if data.get("provider") in PROVIDERS else ""
        others = {}
        for key, saved in (data.get("others") or {}).items():
            if (key in PROVIDERS or key == SPEECH_SERVER) and isinstance(saved, dict):
                others[key] = (str(saved.get("base_url") or ""), _read_key(saved, path))
        key = _read_key(data, path)
        config = cls(base_url, key, provider, others)
        if data.get("api_key") and not data.get("api_key_protected"):
            config.save(path)  # a key typed into the file: store it encrypted from now on
        return config

    def save(self, path: Path = GATEWAY_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _write_key(self.base_url, self.api_key)
        if self.provider:
            data = {"provider": self.provider, **data}
        if self.others:
            data["others"] = {key: _write_key(url, secret) for key, (url, secret) in self.others.items()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)


def _read_key(data: dict, path: Path) -> str:
    if data.get("api_key_protected"):
        try:
            return _unprotect(base64.b64decode(data["api_key_protected"])).decode("utf-8")
        except (OSError, ValueError) as e:  # e.g. the file was copied from another user or laptop
            log.warning("Cannot decrypt an API key in %s (%s); enter it again in AI cleanup", path, e)
            return ""
    return str(data.get("api_key") or "").strip()


def _write_key(base_url: str, api_key: str) -> dict:
    data = {"base_url": base_url}
    if api_key:
        data["api_key_protected"] = base64.b64encode(_protect(api_key.encode("utf-8"))).decode("ascii")
    return data


# ---- Windows DPAPI: encryption tied to the signed-in Windows user

class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


_crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
for _f in (_crypt32.CryptProtectData, _crypt32.CryptUnprotectData):
    _f.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   wintypes.DWORD, ctypes.POINTER(_Blob)]
    _f.restype = wintypes.BOOL
_kernel32.LocalFree.argtypes = [ctypes.c_void_p]
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


def _dpapi(function, data: bytes) -> bytes:
    buffer = ctypes.create_string_buffer(data, len(data))
    blob_in, blob_out = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), _Blob()
    if not function(ctypes.byref(blob_in), None, None, None, None, _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(blob_out)):
        raise OSError(ctypes.get_last_error(), "Windows data protection (DPAPI) failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        _kernel32.LocalFree(blob_out.pbData)


def _protect(data: bytes) -> bytes:
    return _dpapi(_crypt32.CryptProtectData, data)


def _unprotect(data: bytes) -> bytes:
    return _dpapi(_crypt32.CryptUnprotectData, data)


def plausible(heard: str, cleaned: str) -> bool:
    """A cleanup keeps roughly the same length; anything else is a model misbehaving (answering, truncating...)."""
    h, c = len(heard.split()), len(cleaned.split())
    return c > 0 and 0.5 * h - 2 <= c <= 1.5 * h + 5  # fillers may go, but a 7-word dictation doesn't become "Yes."


class Polisher:
    """Cleans up dictated text with a model on the endpoint. polish() never raises and never takes much longer than the
    answer timeout; when it can't help, it returns the text unchanged and says why in `last_error`."""

    def __init__(self, config: GatewayConfig, model: str, vocabulary: list[str] = (), fallback: str | None = None,
                 system_prompt: str | None = None):
        self.config, self.model, self.fallback = config, model, fallback
        self.vocabulary = [w.strip() for w in vocabulary if w.strip()]
        self.system_prompt = system_prompt  # None: SYSTEM_PROMPT (the voice pipeline passes its own, stricter one)
        self.last_error = ""
        self.address = config.address
        url = urlsplit(self.address)
        self._https, self._host, self._port, self._path = url.scheme == "https", url.hostname, url.port, url.path
        self._conn: http.client.HTTPConnection | None = None
        self._used = 0.0
        self._down_until = 0.0
        self._lock = threading.Lock()
        self._other: Polisher | None | bool = None  # the backup at another provider; None: not looked for yet

    def routes(self) -> list[tuple["Polisher", str]]:
        """The models to ask, in order: the chosen one; the backup model, never the chosen one again; and, when a cloud
        provider is chosen and the user has a key for another offered one (OFFERED), that provider's model from the
        ready-made setups (sst.setups), so no credit or a used-up quota at one provider doesn't stop the cleanup. While
        Gemini is the only cloud provider offered, that is only for a profile still on a provider no longer offered.
        Without such a key, nothing else is asked."""
        routes = [(self, self.model)]
        if self.fallback and self.fallback != self.model:
            routes.append((self, self.fallback))
        if self._other is None:
            from sst.setups import backup_for  # sst.setups builds on this module
            found = backup_for(self.config) if not self.config.service.own_server else None  # own servers stay private
            self._other = Polisher(*found) if found and found[1] != self.model else False
        if self._other:
            self._other.vocabulary, self._other.system_prompt = self.vocabulary, self.system_prompt
            routes.append((self._other, self._other.model))
        return routes

    # ---- used by the dictation

    def prepare(self) -> None:
        """Open or refresh the connection in the background, while the user is still speaking."""
        threading.Thread(target=self._prepare, name="gateway-connect", daemon=True).start()

    def polish(self, text: str) -> str:
        self.last_error = ""
        if not text.strip() or not self.address or not self.model:
            return text
        if time.monotonic() < self._down_until:
            self.last_error = "the AI endpoint could not be reached a moment ago"
            return text
        asked = 0
        for polisher, model in self.routes():
            if refused := polisher._refused(model):  # asked a moment ago and refused: the next one, without waiting
                self.last_error = self.last_error or refused  # the answer just given says more
                continue
            if asked == MAX_ASKED:
                break
            asked += 1
            t0 = time.perf_counter()
            try:
                cleaned = polisher._ask(model, text)
            except TimeoutError:
                self.last_error = f"{_short(model)} took too long"
                log.warning("Cleanup with %s took too long", model)
                continue  # the backup may answer in time
            except (OSError, http.client.HTTPException) as e:
                if polisher is self:
                    self._mark_down(e)
                else:
                    self.last_error = f"the backup AI could not be reached ({e})"
                return text
            except GatewayError as e:
                self.last_error = str(e)
                log.warning("Cleanup with %s failed: %s", model, e)
                continue
            if not plausible(text, cleaned):
                self.last_error = f"{_short(model)} gave an unusual answer"
                log.warning("Ignoring an implausible cleanup by %s: %r -> %r", model, text, cleaned)
                return text
            self.last_error = ""
            log.info("Cleaned by %s in %.2fs", model, time.perf_counter() - t0)
            return cleaned
        return text

    def complete(self, text: str) -> str:
        """The model's answer to `text` under this polisher's system prompt, without the cleanup's checks: Text
        Transform (sst.transform) may shorten or restructure the text, and checks the answer itself. The backup model is
        tried when the first fails. More time than a cleanup: the user is waiting for it on purpose. Raises GatewayError
        with a readable reason."""
        if not self.address or not self.model:
            raise GatewayError("no AI model is set up: choose one in AI cleanup")
        words, reason, asked = words_in(text), "", 0
        timeout, limit = TRANSFORM_TIMEOUT + TRANSFORM_PER_WORD * words, TRANSFORM_TOKENS + TRANSFORM_TOKENS_PER_WORD * words
        for polisher, model in self.routes():
            if refused := polisher._refused(model):
                reason = reason or refused
                continue
            if asked == MAX_ASKED:
                break
            asked += 1
            try:
                return polisher._ask(model, text, timeout, limit)
            except TimeoutError:
                reason = f"{_short(model)} took too long"
            except (OSError, http.client.HTTPException) as e:
                raise GatewayError(f"could not reach the AI provider: {e}") from None
            except GatewayError as e:
                reason = str(e)
                log.warning("Completion with %s failed: %s", model, e)
        raise GatewayError(reason)

    def check(self) -> str:
        """For the Settings "Test" button: clean one short sentence with the chosen model. Raises with a readable reason."""
        t0 = time.perf_counter()
        try:
            with usage.feature("tests"):
                answer = self._ask(self.model, "so this is a quick test of the dictation clean up")
        except TimeoutError:
            raise GatewayError("no answer in time") from None
        except (OSError, http.client.HTTPException) as e:
            raise GatewayError(f"could not reach the endpoint: {e}") from None
        return f"{_short(self.model)} answered in {time.perf_counter() - t0:.1f} s: {answer}"

    def models(self) -> list[str]:
        """For the "Load models" button: the provider's text models, models on the endpoint's own server first (some
        gateways mark them `"is_cloud": false`). Raises GatewayError with a readable reason."""
        path = "/models?limit=1000" if self.config.service.api == "anthropic" else "/models"
        try:
            with self._lock:
                status, data, _ = self._request(path, None, 10.0, method="GET")
        except TimeoutError:
            raise GatewayError("no answer in time") from None
        except (OSError, http.client.HTTPException) as e:
            raise GatewayError(f"could not reach the endpoint: {e}") from None
        try:
            answer = json.loads(data)
        except ValueError:
            raise GatewayError(f"HTTP {status}, not JSON") from None
        if status != 200:
            raise GatewayError(f"HTTP {status} {_detail(answer, data)}")
        items = answer.get("data", []) if isinstance(answer, dict) else answer
        models = [(m.get("is_cloud") is not False, m["id"].removeprefix("models/"))  # Gemini's ids start "models/"
                  for m in items if isinstance(m, dict) and isinstance(m.get("id"), str)]
        return [name for _, name in sorted(models, key=lambda m: (m[0], m[1].lower())) if not NOT_FOR_TEXT.search(name)]

    # ---- HTTP

    def _body(self, model: str, text: str, limit: int) -> tuple[str, dict]:
        """The request each provider understands: (path, body). `limit`: tokens for the answer, before any thinking."""
        service = self.config.service
        if service.api == "anthropic":
            body = {"model": model, "max_tokens": limit, "system": self._system_prompt(),
                    "messages": [{"role": "user", "content": text}]}
            if modelrules.claude_takes_temperature(model):
                body["temperature"] = 0
            elif modelrules.claude_thinking_off(model):  # Haiku 5.5: no thinking, so no room for it
                body["thinking"] = {"type": "disabled"}
            else:  # Claude 4.7 and later refuse it; room in case the model thinks first
                body["max_tokens"] += THINKING_ROOM
            return "/messages", body
        body = {"model": model,
                "messages": [{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": text}]}
        if service.key == "openai":
            # OpenAI's current name for the limit (its newest models refuse max_tokens). Reasoning models spend tokens on
            # thinking first, as little as they allow, and take only the default temperature.
            reasoning = bool(modelrules.OPENAI_REASONING.match(model))
            body["max_completion_tokens"] = limit + (THINKING_ROOM if reasoning else 0)
            if not reasoning:
                body["temperature"] = 0
            elif effort := modelrules.openai_effort(model):
                body["reasoning_effort"] = effort
        elif service.key == "gemini":
            # No limit: Gemini's thinking counts against it and would cut the answer short. Gemini 3 thinks as little as it
            # allows, at its own temperature.
            if thinking := modelrules.gemini_thinking(model):
                body["reasoning_effort"] = thinking
            if modelrules.gemini_takes_temperature(model):
                body["temperature"] = 0
        elif service.key == "groq":
            body["max_completion_tokens"] = limit  # Groq's current name for the limit (max_tokens is deprecated)
            if effort := modelrules.groq_effort(model):  # gpt-oss: as little thinking as it allows, with room for it
                body.update(reasoning_effort=effort, max_completion_tokens=limit + THINKING_ROOM)
                if "gpt-oss" in model:
                    body["include_reasoning"] = False  # the thinking isn't sent back: only the answer is read
            else:
                body["temperature"] = 0
        else:
            body.update(temperature=0, max_tokens=limit)
        if service.own_server:
            body["chat_template_kwargs"] = {"enable_thinking": False}  # Qwen3 and the like: answer directly
        return "/chat/completions", body

    def _ask(self, model: str, text: str, timeout: float | None = None, limit: int | None = None) -> str:
        words = len(text.split())
        path, body = self._body(model, text, limit or ANSWER_TOKENS + ANSWER_TOKENS_PER_WORD * words)
        sent = json.dumps(body)
        with self._lock:
            status, data, headers = self._request(path, sent, timeout or ANSWER_TIMEOUT + ANSWER_PER_WORD * words)
        try:
            answer = json.loads(data)
        except ValueError:
            answer = None
        if status == 200:
            netconnect.forgive(self._where, self.config.api_key, model)
            if self.config.service.key == "gemini":  # the cost tracker (sst.usage): an answer cut short is billed too
                usage.record("cleanup", model, **usage.chat_counts(answer, sent, _written(answer)))
        elif isinstance(answer, list) and answer and isinstance(answer[0], dict):
            answer = answer[0]  # an error wrapped in a list (Gemini's OpenAI-compatible endpoint)
        if status != 200 and (found := netconnect.refusal(status, data, headers)):
            # Said plainly, and remembered: the next texts go to the backup at once (and cloud speech on the same key
            # to Parakeet).
            netconnect.refuse(self._where, self.config.api_key, model, found)
            raise GatewayError(f"{self.config.service.name}: {found.reason} (HTTP {status} {_detail(answer, data)})",
                               found)
        if answer is None:
            raise GatewayError(f"{_short(model)}: HTTP {status}, not JSON")
        if not isinstance(answer, dict):
            raise GatewayError(f"{_short(model)}: HTTP {status}, an unexpected answer")
        if self.config.service.api == "anthropic":
            if status != 200 or answer.get("type") == "error" or not isinstance(answer.get("content"), list):
                raise GatewayError(f"{_short(model)}: HTTP {status} {_detail(answer, data)}")
            content = "".join(b.get("text", "") for b in answer["content"] if isinstance(b, dict) and b.get("type") == "text")
            cut = answer.get("stop_reason") == "max_tokens"
        else:
            if status != 200 or "error" in answer or not answer.get("choices"):
                raise GatewayError(f"{_short(model)}: HTTP {status} {_detail(answer, data)}")
            choice = answer["choices"][0]
            content, cut = (choice.get("message") or {}).get("content") or "", choice.get("finish_reason") == "length"
            if choice.get("finish_reason") == "content_filter":  # Gemini's SAFETY, OpenAI's filter: maybe half an answer
                raise GatewayError(f"{_short(model)} stopped its answer (the provider's content filter)")
        if cut:  # stopped at the limit: never a text with a hole in it (the backup model, or the text as heard)
            raise GatewayError(f"{_short(model)} ran out of tokens before "
                               + ("the end of its answer" if content.strip() else "answering (thinking used them up)"))
        if not content.strip():  # said plainly, not "the result is empty" from the checks after it
            raise GatewayError(f"{_short(model)} gave an empty answer")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if len(content) > 1 and content[0] == content[-1] == '"' and not text.startswith('"'):
            content = content[1:-1].strip()  # some models wrap the answer in quotes
        return content

    def _request(self, path: str, body: str | None, timeout: float,
                 method: str = "POST") -> tuple[int, str, http.client.HTTPMessage]:
        headers = {"Content-Type": "application/json"}
        if self.config.service.api == "anthropic":
            headers.update({"x-api-key": self.config.api_key, "anthropic-version": ANTHROPIC_VERSION})
        elif self.config.api_key:  # local endpoints such as Ollama need none
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        for attempt in (1, 2):
            if self._conn is None:
                self._connect()
            self._conn.sock.settimeout(timeout)
            try:
                self._conn.request(method, self._path + path, body=body.encode("utf-8") if body else None, headers=headers)
                response = self._conn.getresponse()
                data = response.read().decode("utf-8", "replace")
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                self._close()  # the server dropped a kept-alive connection: reconnect once
                if attempt == 2:
                    raise
                continue
            except BaseException:
                self._close()
                raise
            self._used = time.monotonic()
            return response.status, data, response.headers
        raise AssertionError("unreachable")

    def _prepare(self) -> None:
        with self._lock:
            if time.monotonic() < self._down_until or not self.address:
                return
            if self._conn is None or time.monotonic() - self._used > IDLE_RECONNECT:
                try:
                    self._connect()
                except OSError as e:
                    self._mark_down(e)

    def _connect(self) -> None:
        self._close()
        try:
            self._conn = netconnect.http_connection(self._https, self._host, self._port, CONNECT_BUDGET)
        except OSError as e:  # every address was tried within the budget
            raise Unreachable(f"cannot reach {self._host}: {e}") from None
        self._used = time.monotonic()

    @property
    def _where(self) -> str:
        """The provider's host, as sst.netconnect remembers refusals (cloud speech on the same host and key shares them)."""
        return f"{self._host}:{self._port or ''}"

    def _refused(self, model: str) -> str:
        """Why the provider refused this key or model a moment ago (sst.netconnect), or ""."""
        reason = netconnect.refused(self._where, self.config.api_key, model)
        return f"{self.config.service.name}: {reason} (a moment ago)" if reason else ""

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _mark_down(self, error: Exception) -> None:
        self._down_until = time.monotonic() + DOWN_FOR
        self.last_error = f"the AI endpoint could not be reached ({error})"
        log.warning("Endpoint unreachable, skipping cleanup for %.0fs: %s", DOWN_FOR, error)

    def _system_prompt(self) -> str:
        prompt = SYSTEM_PROMPT if self.system_prompt is None else self.system_prompt
        if not self.vocabulary:
            return prompt
        return prompt + "\nThe user's vocabulary: " + ", ".join(self.vocabulary)


_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def words_in(text: str) -> int:
    """A text's length in words, in any script: Japanese and Chinese have no spaces, so 1,500 characters of Japanese were
    "one word", and got one word's time and tokens. Their letters count 2.5 to a word."""
    return round(len(_CJK.findall(text)) / 2.5) + len(_CJK.sub(" ", text).split())


def _short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def _written(answer) -> str:
    """The text of an OpenAI-compatible answer, whatever its shape ("" when it has none)."""
    try:
        return str(answer["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return ""


def _detail(answer, data: str) -> str:
    """The readable part of an error answer: OpenAI and Anthropic put it in error.message, others in detail."""
    if isinstance(answer, dict):
        error = answer.get("error") or answer.get("detail") or answer.get("message")
        if isinstance(error, dict):
            error = error.get("message") or error
        if error:
            return str(error)[:200]
    return data[:120]
