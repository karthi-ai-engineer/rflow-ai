"""Cloud speech models against a fake provider on this machine (no real provider, no key): each provider's request
format, the hints, the errors, and Parakeet taking over whenever the provider can't help."""
import base64
import io
import json
import socket
import threading
import time
import wave
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from sst import netconnect, usage
from sst.engines import cloud, load_engine, usable
from sst.engines.cloud import CloudEngine, CloudError
from sst.gateway import SPEECH_SERVER, GatewayConfig

RATE = 48_000
AUDIO = (0.1 * np.sin(np.linspace(0, 2000, 2 * RATE))).astype(np.float32)  # two seconds at the microphone's rate


# Gemini Transcribe's documented answer: the text in a text part, the word times in an audioTranscription part.
TRANSCRIBED = {"candidates": [{"content": {"parts": [
    {"text": " Hello  from\nGemini Transcribe. "},
    {"audioTranscription": {"speakerLabel": "spk_1", "words": [
        {"word": "Hello", "startOffset": "0.100s", "endOffset": "0.450s"},
        {"word": "from", "startOffset": "0.450s", "endOffset": "0.700s"},
        {"word": "Gemini", "startOffset": "0.700s", "endOffset": "1.100s"},
        {"word": "Transcribe.", "startOffset": "1.100s", "endOffset": "1.800s"}]}}], "role": "model"},
    "finishReason": "STOP"}]}
VERBOSE = {"task": "transcribe", "language": "english", "duration": 2.0, "text": " Hello  from OpenAI. ",
           "words": [{"word": "Hello", "start": 0.0, "end": 0.42}, {"word": "from", "start": 0.42, "end": 0.7},
                     {"word": "OpenAI.", "start": 0.7, "end": 1.5}]}


class FakeProvider:
    """Answers like OpenAI's /audio/transcriptions (verbose_json too), Gemini's generateContent (Gemini Transcribe
    too) and Google's Files API; `statuses` and `delay` make transcriptions fail or slow, `answer` replaces Gemini
    Transcribe's answer, `drop` closes each connection after one answer without saying so."""

    def __init__(self):
        self.requests, self.connections, self.statuses, self.delay = [], 0, [], 0.0
        self.answer, self.drop = None, False
        self.bodies = {}  # status -> the error answer to give with it (else {"error": {"message": ...}})
        self.reply = None  # (request) -> the text a Flash or OpenAI-format model answers, or None for the default
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive, like the real providers

            def setup(self):
                fake.connections += 1
                super().setup()

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                request = {"path": self.path, "auth": self.headers.get("Authorization"),
                           "x-goog-api-key": self.headers.get("x-goog-api-key"),
                           "headers": {k.lower(): v for k, v in self.headers.items()}}
                fake.requests.append(request)
                if self.path.startswith("/upload/"):  # the Files API: start an upload, then send the bytes
                    request["body"] = body
                    if self.headers.get("X-Goog-Upload-Command") == "start":
                        request["json"] = json.loads(body)
                        return self._reply(200, {}, {"X-Goog-Upload-URL": f"{fake.address}{self.path}?upload_id=u1"})
                    return self._reply(200, {"file": {"name": "files/f1", "uri": f"{fake.address}/v1beta/files/f1",
                                                      "mimeType": "audio/wav", "state": "ACTIVE"}})
                if self.path.endswith(":generateContent"):
                    request["json"] = json.loads(body)
                else:
                    message = BytesParser(policy=HTTP).parsebytes(
                        f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body)
                    request["fields"] = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                                         for part in message.iter_parts()}
                time.sleep(fake.delay)
                status = fake.statuses.pop(0) if fake.statuses else 200
                if status != 200:
                    reason = {401: "Invalid API key", 400: "Unsupported response_format"}.get(status, "Slow down")
                    return self._reply(status, fake.bodies.get(status) or {"error": {"message": reason}})
                if "json" in request and "transcribe" in self.path:
                    return self._reply(200, fake.answer or TRANSCRIBED)
                if fake.reply is not None and (text := fake.reply(request)) is not None:
                    if "json" in request:
                        return self._reply(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})
                    return self._reply(200, {"text": text})
                if "json" in request:
                    return self._reply(200, {"candidates": [{"content": {"parts": [
                        {"text": "Thinking about it...", "thought": True}, {"text": " Hello  from\nGemini. "}]}}]})
                if request["fields"].get("response_format") == b"verbose_json":
                    return self._reply(200, VERBOSE)
                self._reply(200, {"text": " Hello  from OpenAI. "})

            def do_GET(self):
                fake.requests.append({"path": self.path, "auth": self.headers.get("Authorization")})
                self._reply(200, {"object": "list", "data": [{"id": "Qwen/Qwen3-30B"}, {"id": "whisper-1"},
                                                              {"id": "models/gemini-3.5-transcribe"}, {"id": "bge-m3"}]})

            def do_DELETE(self):
                fake.requests.append({"path": self.path, "method": "DELETE",
                                      "x-goog-api-key": self.headers.get("x-goog-api-key")})
                self._reply(200, {})

            def _reply(self, status, payload, headers=None):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = self.close_connection or fake.drop  # gone, without a "Connection: close"

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.address = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def engine(self, provider="openai", model="", language="", fallback=None, path="/v1", key="test-key"):
        return CloudEngine(provider, key, model, language, fallback, url=self.address + path)


@pytest.fixture
def fake():
    provider = FakeProvider()
    yield provider
    provider.server.shutdown()


@pytest.fixture(autouse=True)
def no_refusals():
    netconnect.forget_refusals()  # what one test's fake provider refused never reaches another test
    yield
    netconnect.forget_refusals()


class Parakeet:
    """Stands in for the engine on this computer that a cloud model falls back on."""
    name = "parakeet"

    def __init__(self):
        self.words, self.calls = [], 0

    def transcribe(self, audio, rate):
        self.calls += 1
        return "hello from parakeet"


def _wav(data: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(data)) as w:
        return w.getframerate(), w.getnframes()


def test_openai_and_groq_get_a_multipart_upload_with_the_hints(fake):
    engine = fake.engine("groq", language="ta")
    engine.words = ["Karthi", "Rflow"]
    assert engine.transcribe(AUDIO, RATE) == "Hello from OpenAI." and engine.last_error == ""
    request = fake.requests[0]
    assert request["path"] == "/v1/audio/transcriptions" and request["auth"] == "Bearer test-key"
    fields = request["fields"]
    assert fields["model"] == b"whisper-large-v3-turbo"  # Groq's first model is the default
    assert fields["language"] == b"ta" and fields["response_format"] == b"json"
    assert b"Karthi, Rflow" in fields["prompt"]
    assert _wav(fields["file"]) == (16_000, 32_000)  # resampled to 16 kHz: a third of the upload


def test_no_language_and_no_words_send_no_hints(fake):
    fake.engine("openai").transcribe(AUDIO, RATE)
    assert set(fake.requests[0]["fields"]) == {"model", "response_format", "file"}
    assert fake.requests[0]["fields"]["model"] == b"gpt-transcribe"  # OpenAI's current model is the default


def test_gpt_transcribe_takes_the_language_as_a_list(fake):
    fake.engine("openai", language="ja").transcribe(AUDIO, RATE)
    fields = fake.requests[0]["fields"]
    assert fields["languages[]"] == b"ja" and "language" not in fields  # OpenAI: don't send both
    fake.engine("openai", model="gpt-4o-mini-transcribe", language="ja").transcribe(AUDIO, RATE)
    assert fake.requests[1]["fields"]["language"] == b"ja" and "languages[]" not in fake.requests[1]["fields"]


def test_a_model_being_retired_still_works_and_says_so(fake):
    assert cloud.retirement("openai", "gpt-transcribe") == "" and cloud.retirement("server", "whisper-1") == ""
    for model in ("gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"):
        assert model in cloud.CLOUD["openai"].models  # offered until OpenAI shuts it down
        answer = fake.engine("openai", model=model).check(AUDIO, RATE)
        assert answer.startswith(f"{model} answered in ") and answer.endswith(
            f"(OpenAI shuts {model} down on 26 February 2027: choose gpt-transcribe)")
    assert "(" not in fake.engine("openai").check(AUDIO, RATE)


def test_gemini_gets_the_audio_inline_with_an_instruction(fake):
    engine = fake.engine("gemini", model="gemini-3.6-flash", language="ta", path="/v1beta")
    engine.words = ["Karthi"]
    assert engine.transcribe(AUDIO, RATE) == "Hello from Gemini."  # its thinking is left out
    request = fake.requests[0]
    assert request["path"] == "/v1beta/models/gemini-3.6-flash:generateContent"
    assert request["x-goog-api-key"] == "test-key" and request["auth"] is None
    body = request["json"]
    audio, instruction = body["contents"][0]["parts"]
    assert audio["inline_data"]["mime_type"] == "audio/wav"
    assert _wav(base64.b64decode(audio["inline_data"]["data"])) == (16_000, 32_000)
    assert "word for word" in instruction["text"] and "Tamil" in instruction["text"] and "Karthi" in instruction["text"]
    # Gemini 3 Flash thinks at "medium" unless told, and Google advises its own temperature (1.0) over 0.
    assert body["generationConfig"] == {"thinkingConfig": {"thinkingLevel": "minimal"}}


@pytest.mark.parametrize("model, config", [
    ("gemini-3.5-flash-lite", {"thinkingConfig": {"thinkingLevel": "minimal"}}),
    ("gemini-3.8-flash", {"thinkingConfig": {"thinkingLevel": "low"}}),  # "minimal" is an error on 3.7 and 3.8
    ("gemini-flash-latest", {"thinkingConfig": {"thinkingLevel": "minimal"}}),  # answered by 3.6 Flash (2026-10-23)
    ("gemini-flash-lite-latest", {}),  # its model isn't documented: its own settings, which always work
    ("gemini-2.5-flash-lite", {"temperature": 0}),  # before Gemini 3: as always
])
def test_each_gemini_flash_model_thinks_as_little_as_it_allows(fake, model, config):
    engine = fake.engine("gemini", model=model, path="/v1beta")
    engine.transcribe(AUDIO, RATE)
    assert fake.requests[0]["json"]["generationConfig"] == config
    assert ("|think:" in engine.signature) is ("temperature" not in config)  # cached apart from the old requests' text


def test_a_provider_error_is_typed_by_parakeet_and_said(fake):
    parakeet, loads = Parakeet(), []
    engine = fake.engine(fallback=lambda: loads.append(1) or parakeet)
    engine.words = ["Karthi"]
    fake.statuses = [500, 500]
    assert engine.transcribe(AUDIO, RATE) == "hello from parakeet"
    assert "HTTP 500 Slow down" in engine.last_error and engine.last_error.startswith("OpenAI")
    assert parakeet.words == ["Karthi"]  # Your words go along
    assert engine.transcribe(AUDIO, RATE) == "hello from parakeet" and loads == [1]  # loaded once
    assert engine.transcribe(AUDIO, RATE) == "Hello from OpenAI." and engine.last_error == ""  # the provider is back


QUOTA = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota. Please "
                   "retry in 41.2s.", "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                       {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
                       {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"}]}}


@pytest.mark.parametrize("status, said", [(401, "didn't accept the API key (HTTP 401 Invalid API key)"),
                                          (402, "no credit left on the account (HTTP 402"),
                                          (429, "quota is used up (HTTP 429 You exceeded your current quota")])
def test_a_refusal_is_said_plainly_and_remembered_until_the_key_changes(fake, status, said):
    parakeet = Parakeet()
    engine = fake.engine(fallback=lambda: parakeet)
    fake.statuses, fake.bodies = [status], {429: QUOTA}
    assert engine.transcribe(AUDIO, RATE) == "hello from parakeet" and said in engine.last_error
    t0 = time.perf_counter()
    assert engine.transcribe(AUDIO, RATE) == "hello from parakeet" and "a moment ago" in engine.last_error
    with pytest.raises(CloudError, match="a moment ago") as error:
        engine.transcribe_chunk(AUDIO, RATE)
    assert error.value.permanent and time.perf_counter() - t0 < 0.3 and len(fake.requests) == 1  # not asked again
    engine.api_key = "a-new-key"  # the user entered another key
    assert engine.transcribe(AUDIO, RATE) == "Hello from OpenAI." and engine.last_error == ""


def test_a_used_up_quota_is_the_models_and_a_short_rate_limit_is_retried(fake):
    fake.statuses, fake.bodies = [429], {429: QUOTA}
    with pytest.raises(CloudError):
        fake.engine(model="whisper-1").transcribe_chunk(AUDIO, RATE)
    assert fake.engine(model="gpt-transcribe").transcribe_chunk(AUDIO, RATE).text  # another model on the same key
    fake.statuses, fake.bodies = [429], {429: {"error": {"message": "Rate limit reached. Please try again in 1.5s."}}}
    with pytest.raises(CloudError) as error:
        fake.engine(model="gpt-4o-transcribe").transcribe_chunk(AUDIO, RATE)
    assert not error.value.permanent and error.value.retry_after == 1.5  # worth a retry, after that long


def test_the_pipeline_falls_back_at_once_on_a_refusal_and_skips_the_provider_after(fake):
    from sst.pipeline.asr import ASRScheduler, EngineBackend
    from sst.pipeline.contracts import AudioChunk, RawTranscript

    class Local:
        name = "parakeet"

        def transcribe(self, chunk):
            return RawTranscript("hello from parakeet", backend="parakeet")

    sleeps = []
    fake.statuses, fake.bodies = [429], {429: QUOTA}
    engine = fake.engine("gemini", "gemini-3.5-flash-lite", path="/v1beta")
    with ASRScheduler(EngineBackend(engine), fallback=Local(), sleep=sleeps.append) as scheduler:
        for n in (1, 2):
            session = scheduler.open_session(f"s{n}")
            session.submit(AudioChunk(f"s{n}", 1, AUDIO.copy(), RATE, 0, len(AUDIO), 0, 1.5))
            session.finish(1)
            [result] = session.wait(10)
            assert result.text == "hello from parakeet" and "quota is used up" in result.note
            assert result.diagnostics["attempts"] == 2  # one try of the provider (none for s2), one of Parakeet
    assert sleeps == [] and len(fake.requests) == 1  # no backoff, no retry, and the second dictation didn't ask


def test_cloud_speech_and_ai_cleanup_on_the_same_key_share_a_refusal(fake):
    from sst.gateway import Polisher
    fake.statuses = [402]
    with pytest.raises(CloudError, match="no credit"):
        fake.engine("gemini", "gemini-3.5-flash-lite", path="/v1beta").transcribe_chunk(AUDIO, RATE)
    p = Polisher(GatewayConfig(fake.address + "/v1beta/openai", "test-key", "gemini"), "gemini-3.5-flash-lite")
    assert p.polish("so this is a test") == "so this is a test" and "no credit" in p.last_error
    assert len(fake.requests) == 1  # the cleanup knew without asking


def test_an_unreachable_provider_is_skipped_for_a_minute(monkeypatch):
    monkeypatch.setattr(cloud, "CONNECT_BUDGET", 0.3)  # Windows takes a while to refuse a connection
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    parakeet = Parakeet()
    engine = CloudEngine("openai", "test-key", fallback=lambda: parakeet, url=f"http://127.0.0.1:{port}/v1")
    assert engine.transcribe(AUDIO, RATE) == "hello from parakeet" and "cannot reach 127.0.0.1" in engine.last_error
    connects = []
    monkeypatch.setattr(engine, "_connect", lambda: connects.append(1))
    t0 = time.perf_counter()
    assert engine.transcribe(AUDIO, RATE) == "hello from parakeet" and "a moment ago" in engine.last_error
    assert not connects and time.perf_counter() - t0 < 1.0  # no waiting on the network again


def test_a_slow_answer_gives_way_to_parakeet(fake, monkeypatch):
    monkeypatch.setattr(cloud, "ANSWER_TIMEOUT", 0.2)
    monkeypatch.setattr(cloud, "ANSWER_PER_SECOND", 0.0)
    fake.delay = 1.0
    engine = fake.engine(fallback=Parakeet)
    assert engine.transcribe(AUDIO, RATE) == "hello from parakeet" and "no answer in time" in engine.last_error


def test_without_a_fallback_the_failure_is_raised(fake):
    fake.statuses = [500]
    with pytest.raises(CloudError, match="OpenAI: HTTP 500"):
        fake.engine().transcribe(AUDIO, RATE)


def test_scoring_waits_out_a_rate_limit_but_dictation_does_not(fake, monkeypatch):
    monkeypatch.setattr(cloud, "RATE_LIMIT_WAITS", (0, 0))
    fake.statuses = [429, 429]
    assert fake.engine().transcribe(AUDIO, RATE) == "Hello from OpenAI." and len(fake.requests) == 3
    fake.statuses = [429]
    engine = fake.engine(fallback=Parakeet)
    assert engine.transcribe(AUDIO, RATE) == "hello from parakeet" and "429" in engine.last_error
    assert len(fake.requests) == 4  # not asked again: Parakeet typed at once


def test_the_connection_is_opened_while_the_user_speaks_and_kept(fake):
    engine = fake.engine()
    engine.prepare()
    for _ in range(100):
        if fake.connections:
            break
        time.sleep(0.01)
    engine.transcribe(AUDIO, RATE)
    engine.transcribe(AUDIO, RATE)
    assert fake.connections == 1 and len(fake.requests) == 2


def test_the_test_button_reports_the_time_and_never_falls_back(fake):
    engine = fake.engine(fallback=Parakeet)
    assert engine.check(AUDIO, RATE).startswith("gpt-transcribe answered in ")
    fake.statuses = [401]
    with pytest.raises(CloudError, match="Invalid API key"):
        engine.check(AUDIO, RATE)


def test_a_cloud_model_needs_a_key_which_never_shows():
    with pytest.raises(ValueError, match="needs an API key"):
        CloudEngine("openai", "")
    engine = CloudEngine("openai", "sk-secret")
    assert "sk-secret" not in repr(engine) and engine.title == "OpenAI gpt-transcribe"


def test_the_signature_follows_what_the_text_depends_on():
    engine = CloudEngine("openai", "sk-1")
    first = engine.signature
    engine.api_key = "sk-2"
    assert engine.signature == first  # the key doesn't change the text
    for change in (lambda: setattr(engine, "model", "whisper-1"), lambda: setattr(engine, "language", "ta"),
                   lambda: setattr(engine, "words", ["Karthi"])):
        before = engine.signature
        change()
        assert engine.signature != before


def test_the_catalog_uses_a_cloud_model_only_with_its_key():
    assert usable("openai") == "parakeet" and usable("gemini", GatewayConfig()) == "parakeet"
    keys = GatewayConfig(provider="anthropic", api_key="ant", others={"gemini": ("", "AIza")})
    assert usable("gemini", keys) == "gemini" and usable("openai", keys) == "gemini"  # OpenAI isn't offered: Gemini
    engine = load_engine("gemini", "ja", api_key="g-key")
    assert isinstance(engine, CloudEngine) and engine.model == "gemini-3.5-transcribe" and engine.language == "ja"


def test_an_own_server_needs_an_address_but_maybe_no_key(fake):
    with pytest.raises(ValueError, match="needs its address"):
        CloudEngine("server", "")
    engine = fake.engine("server", model="openai/whisper-large-v3-turbo", key="")
    assert engine.transcribe(AUDIO, RATE) == "Hello from OpenAI." and engine.title == "Your server openai/whisper-large-v3-turbo"
    request = fake.requests[0]
    assert request["path"] == "/v1/audio/transcriptions" and request["auth"] is None  # no key: no header
    assert request["fields"]["model"] == b"openai/whisper-large-v3-turbo"
    fake.engine("server", model="whisper-1", key="gw-key").transcribe(AUDIO, RATE)
    assert fake.requests[1]["auth"] == "Bearer gw-key"


def test_load_models_lists_the_speech_models_first(fake):
    assert fake.engine("server", key="gw-key").models() == ["gemini-3.5-transcribe", "whisper-1", "bge-m3",
                                                              "Qwen/Qwen3-30B"]
    assert fake.requests[0] == {"path": "/v1/models", "auth": "Bearer gw-key"}


def test_a_new_server_address_is_used_from_the_next_request(fake):
    other = FakeProvider()
    try:
        engine = fake.engine("server", model="whisper-1")
        engine.transcribe(AUDIO, RATE)
        before = engine.signature
        engine.set_url(other.address + "/v1/")
        engine.transcribe(AUDIO, RATE)
        assert len(fake.requests) == 1 and len(other.requests) == 1
        assert engine.signature != before  # another server may give other text
    finally:
        other.server.shutdown()


def test_the_catalog_uses_an_own_server_only_with_its_address():
    assert usable("server", GatewayConfig()) == "parakeet"
    keys = GatewayConfig().with_entry(SPEECH_SERVER, "http://10.0.0.5:8000/v1", "")
    assert usable("server", keys) == "server"
    engine = load_engine("server", "ta", model="whisper-1", url="http://10.0.0.5:8000/v1")
    assert engine.url == "http://10.0.0.5:8000/v1" and engine.language == "ta" and engine.api_key == ""


def test_without_parakeet_a_failing_provider_says_why(fake):
    def missing():
        raise RuntimeError("Parakeet isn't downloaded to take over")
    fake.statuses = [503]
    with pytest.raises(CloudError, match=r"OpenAI: HTTP 503 .*\(Parakeet isn't downloaded to take over\)"):
        fake.engine(fallback=missing).transcribe(AUDIO, RATE)


def _times(raw):
    return [(w.text, round(w.start, 3), round(w.end, 3)) for w in raw.words]


# ---- Gemini Transcribe

def test_gemini_transcribe_is_the_default_and_hears_the_audio_alone_with_word_times(fake):
    engine = fake.engine("gemini", language="ta", path="/v1beta")
    engine.words = ["Karthi"]  # not sent: Google takes no custom vocabulary along with word times
    assert engine.model == "gemini-3.5-transcribe" and not engine.biased
    raw = engine.transcribe_chunk(AUDIO, RATE)
    assert raw.text == "Hello from Gemini Transcribe." and raw.backend == "Google Gemini gemini-3.5-transcribe"
    assert _times(raw) == [("Hello", 0.1, 0.45), ("from", 0.45, 0.7), ("Gemini", 0.7, 1.1), ("Transcribe.", 1.1, 1.8)]
    assert raw.language == "ta" and raw.diagnostics == {"mode": "timestamps", "transport": "inline"}
    request = fake.requests[0]
    assert request["path"] == "/v1beta/models/gemini-3.5-transcribe:generateContent"
    assert request["x-goog-api-key"] == "test-key" and request["auth"] is None
    (audio,) = request["json"]["contents"][0]["parts"]  # the recording alone: no instruction
    assert audio["inlineData"]["mimeType"] == "audio/wav"
    assert _wav(base64.b64decode(audio["inlineData"]["data"])) == (16_000, 32_000)
    assert request["json"]["generationConfig"] == {
        "audioTranscriptionConfig": {"mode": "VERBATIM", "languageCodes": ["ta"], "wordTimestamp": True}}
    assert engine.transcribe(AUDIO, RATE) == "Hello from Gemini Transcribe."  # dictation sends the same request
    assert fake.requests[1]["json"] == request["json"]


def test_in_vocabulary_mode_your_words_go_along_without_word_times(fake):
    engine = fake.engine("gemini", path="/v1beta")
    engine.words = ["Karthi", " Rflow ", "karthi", "", *(f"term {i}" for i in range(120))]
    timestamps = engine.signature
    engine.timestamp_mode = "vocabulary"
    assert engine.biased and engine.signature != timestamps  # the vocabulary changes the text
    assert engine.transcribe(AUDIO, RATE) == "Hello from Gemini Transcribe."
    config = fake.requests[0]["json"]["generationConfig"]["audioTranscriptionConfig"]
    assert config["mode"] == "VERBATIM" and config["languageCodes"] == [] and config["wordTimestamp"] is False
    vocabulary = config["customVocabulary"]
    assert vocabulary[:3] == ["Karthi", "Rflow", "term 0"] and len(vocabulary) == 100  # each once, at most 100
    engine.words = []
    engine.transcribe(AUDIO, RATE)
    assert "customVocabulary" not in fake.requests[1]["json"]["generationConfig"]["audioTranscriptionConfig"]


@pytest.mark.parametrize("answer, text, words", [
    ({"candidates": [{"content": {"parts": [{"audioTranscription": {"words": [  # words only, other key and time styles
        {"word": "Hi", "start_offset": {"seconds": 1, "nanos": 500000000}, "end_offset": 2},
        {"text": "there", "startOffset": "2s", "endOffset": "2.5s", "confidence": 0.75}]}}]},
        "finishReason": "STOP"}]}, "Hi there", [("Hi", 1.5, 2.0), ("there", 2.0, 2.5)]),
    ({"candidates": [{"content": {"parts": [{"text": "Thinking...", "thought": True},
                                             {"audio_transcription": {"transcript": "Hi there."}}]}}]}, "Hi there.", []),
    ({"candidates": [{"content": {"parts": [{"audioTranscription": {"text": "Hi.", "words": [
        {"word": "Hi.", "startOffset": "0.2s"}, {"word": "there"}]}}]}}]}, "Hi.", []),  # a word without a time: none
    ({"candidates": [{"finishReason": "STOP"}]}, "", []),  # finished with nothing in it: silence
])
def test_gemini_transcribe_answers_are_read_leniently(answer, text, words):
    got, timed, _ = cloud.gemini_transcript(answer)
    assert got == text and [(w.text, w.start, w.end) for w in timed] == words


@pytest.mark.parametrize("answer, reason", [
    ({"candidates": []}, "no transcript"),
    ({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}, r"blocked the recording \(PROHIBITED_CONTENT\)"),
    ({"candidates": [{"finishReason": "SAFETY"}]}, r"stopped transcribing \(SAFETY\)"),
    ({"candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "MAX_TOKENS"}]}, "MAX_TOKENS"),
])
def test_a_blocked_or_empty_answer_is_an_error(fake, answer, reason):
    fake.answer = answer
    with pytest.raises(CloudError, match=reason):
        fake.engine("gemini", path="/v1beta").transcribe_chunk(AUDIO, RATE)
    engine = fake.engine("gemini", path="/v1beta", fallback=Parakeet)  # dictation: Parakeet types it, and says why
    assert engine.transcribe(AUDIO, RATE) == "hello from parakeet" and "Google Gemini: " in engine.last_error


def test_a_long_recording_goes_through_the_files_api_and_is_deleted(fake, monkeypatch):
    monkeypatch.setattr(cloud, "INLINE_LIMIT", 1000)  # instead of 14 MB
    engine = fake.engine("gemini", path="/v1beta")
    raw = engine.transcribe_chunk(AUDIO, RATE)
    assert raw.text == "Hello from Gemini Transcribe." and raw.diagnostics["transport"] == "files"
    start, upload, transcribe, delete = fake.requests
    assert start["path"] == "/upload/v1beta/files" and start["x-goog-api-key"] == "test-key"
    assert start["json"] == {"file": {"display_name": "rflow-chunk"}}
    headers = start["headers"]
    assert headers["x-goog-upload-protocol"] == "resumable" and headers["x-goog-upload-command"] == "start"
    assert headers["x-goog-upload-header-content-type"] == "audio/wav"
    assert int(headers["x-goog-upload-header-content-length"]) == len(upload["body"])
    assert upload["path"] == "/upload/v1beta/files?upload_id=u1" and upload["x-goog-api-key"] is None  # its own auth
    assert upload["headers"]["x-goog-upload-command"] == "upload, finalize"
    assert upload["headers"]["x-goog-upload-offset"] == "0" and _wav(upload["body"]) == (16_000, 32_000)
    (audio,) = transcribe["json"]["contents"][0]["parts"]
    assert audio == {"fileData": {"mimeType": "audio/wav", "fileUri": f"{fake.address}/v1beta/files/f1"}}
    assert delete == {"path": "/v1beta/files/f1", "method": "DELETE", "x-goog-api-key": "test-key"}
    assert fake.connections == 1  # all on one kept-alive connection
    fake.statuses = [503]
    with pytest.raises(CloudError, match="HTTP 503"):
        engine.transcribe_chunk(AUDIO, RATE)
    assert fake.requests[-1]["method"] == "DELETE"  # a failed transcription removes the upload too


# ---- word times from OpenAI's API

@pytest.mark.parametrize("provider, model", [("openai", "whisper-1"), ("groq", "whisper-large-v3-turbo")])
def test_whisper_models_give_word_times(fake, provider, model):
    raw = fake.engine(provider, model=model).transcribe_chunk(AUDIO, RATE)
    fields = fake.requests[0]["fields"]
    assert fields["response_format"] == b"verbose_json" and fields["timestamp_granularities[]"] == b"word"
    assert raw.text == "Hello from OpenAI." and raw.language == "en"  # "english", as OpenAI writes it
    assert _times(raw) == [("Hello", 0.0, 0.42), ("from", 0.42, 0.7), ("OpenAI.", 0.7, 1.5)]
    assert raw.backend == f"{cloud.CLOUD[provider].name} {model}"


def test_gpt_transcribe_models_and_dictation_keep_plain_json(fake):
    raw = fake.engine("openai").transcribe_chunk(AUDIO, RATE)  # gpt-transcribe gives no word times
    assert raw.text == "Hello from OpenAI." and raw.words == []
    assert set(fake.requests[0]["fields"]) == {"model", "response_format", "file"}
    assert fake.requests[0]["fields"]["response_format"] == b"json"
    fake.engine("groq").transcribe(AUDIO, RATE)  # dictation needs no word times: the text is the same
    assert fake.requests[1]["fields"]["response_format"] == b"json"


def test_a_server_that_refuses_verbose_json_is_asked_for_plain_json_from_then_on(fake):
    engine = fake.engine("server", model="openai/whisper-large-v3", key="")
    fake.statuses = [400]
    raw = engine.transcribe_chunk(AUDIO, RATE)
    assert raw.text == "Hello from OpenAI." and raw.words == []
    assert [r["fields"]["response_format"] for r in fake.requests] == [b"verbose_json", b"json"]
    engine.transcribe_chunk(AUDIO, RATE)
    assert fake.requests[2]["fields"]["response_format"] == b"json"  # remembered: not asked with every chunk
    engine.model = "openai/whisper-large-v3-turbo"  # another model may give them
    assert engine.transcribe_chunk(AUDIO, RATE).words and fake.requests[3]["fields"]["response_format"] == b"verbose_json"


def test_a_server_is_asked_for_word_times_only_for_a_whisper_model(fake):
    engine = fake.engine("server", model="nvidia/canary-1b", key="")
    engine.transcribe_chunk(AUDIO, RATE)
    assert fake.requests[0]["fields"]["response_format"] == b"json"
    engine.model = "whisper-1"
    fake.statuses = [400, 400]  # a 400 for another reason: plain JSON fails too, so it isn't remembered
    with pytest.raises(CloudError, match="HTTP 400") as error:
        engine.transcribe_chunk(AUDIO, RATE)
    assert error.value.status == 400
    assert engine.transcribe_chunk(AUDIO, RATE).words and fake.requests[-1]["fields"]["response_format"] == b"verbose_json"


# ---- the voice pipeline's chunks

def test_chunks_are_transcribed_at_the_same_time(fake):
    engine = fake.engine("groq")
    fake.delay = 0.4
    results = []
    threads = [threading.Thread(target=lambda: results.append(engine.transcribe_chunk(AUDIO, RATE))) for _ in range(2)]
    t0 = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - t0
    assert [r.text for r in results] == ["Hello from OpenAI."] * 2
    assert 0.4 <= elapsed < 0.7, elapsed  # one after the other would take 0.8 s
    assert fake.connections == 2
    engine.transcribe_chunk(AUDIO, RATE)
    assert fake.connections == 2  # both connections were kept for the next chunks


def test_a_chunk_never_falls_back_on_parakeet_nor_waits_out_a_rate_limit(fake):
    loads = []
    engine = fake.engine(fallback=lambda: loads.append(1) or Parakeet())
    fake.statuses = [503]
    with pytest.raises(CloudError, match="HTTP 503") as error:
        engine.transcribe_chunk(AUDIO, RATE)
    assert error.value.status == 503 and loads == [] and engine.last_error == ""
    fake.statuses = [429]  # the pipeline retries, with its own backoff
    with pytest.raises(CloudError) as error:
        fake.engine().transcribe_chunk(AUDIO, RATE)
    assert error.value.status == 429 and len(fake.requests) == 2


def test_an_unreachable_provider_fails_the_next_chunks_at_once(monkeypatch):
    monkeypatch.setattr(cloud, "CONNECT_BUDGET", 0.3)
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    engine = CloudEngine("openai", "test-key", fallback=Parakeet, url=f"http://127.0.0.1:{port}/v1")
    with pytest.raises(cloud.Unreachable, match="cannot reach 127.0.0.1"):
        engine.transcribe_chunk(AUDIO, RATE)
    t0 = time.perf_counter()
    with pytest.raises(CloudError, match="a moment ago") as error:
        engine.transcribe_chunk(AUDIO, RATE)
    assert error.value.status == 0 and time.perf_counter() - t0 < 0.3


def test_an_old_idle_connection_is_replaced_and_a_dropped_one_reconnected(fake, monkeypatch):
    engine = fake.engine()
    engine.transcribe_chunk(AUDIO, RATE)
    monkeypatch.setattr(cloud, "IDLE_RECONNECT", -1.0)  # every kept connection counts as too old
    engine.transcribe_chunk(AUDIO, RATE)
    assert fake.connections == 2
    monkeypatch.setattr(cloud, "IDLE_RECONNECT", 30.0)
    fake.drop = True  # from now on the server closes each connection after answering, without saying so
    for _ in range(2):
        assert engine.transcribe_chunk(AUDIO, RATE).text == "Hello from OpenAI."
    # The dead connection was replaced and nothing was asked twice. Whether the client notices the server's close
    # before reusing the connection (one new connection) or only when it fails (two) depends on timing (seen on CI).
    assert len(fake.requests) == 4 and fake.connections in (3, 4)



# ---- the hint list written into the transcript (the owner's log, 2026-10-02)

OWNER_WORDS = ["version", "installer", "commit", "tray", "app", "pill", "Let's", "move", "stand", "meeting", "Thursday",
               "morning", "commits", "GitHub", "Rflow", "Parakeet", "laptop", "Vercel", "after", "tests"]


def _prompt(request) -> str:
    if "json" in request:
        return " ".join(p.get("text", "") for p in request["json"]["contents"][0]["parts"])
    return (request["fields"].get("prompt") or b"").decode()


def test_only_names_and_terms_go_to_the_speech_model_as_a_spelling_reference(fake):
    for provider, path in (("gemini", "/v1beta"), ("groq", "/v1")):
        engine = fake.engine(provider, model="gemini-3.5-flash-lite" if provider == "gemini" else "", path=path)
        engine.words = OWNER_WORDS
        engine.transcribe(AUDIO, RATE)
        prompt = _prompt(fake.requests[-1])
        assert all(term in prompt for term in ("GitHub", "Rflow", "Parakeet", "Vercel"))
        assert not any(f'"{w}"' in prompt or f" {w}," in prompt for w in ("move", "after", "tests", "laptop", "Let's"))
    assert "Never write any of them unless it is clearly spoken" in _prompt(fake.requests[0])


def test_a_copied_hint_list_is_replaced_by_the_transcript_made_without_hints(fake):
    fake.reply = lambda request: ("What is the problem. GitHub, Rflow, Parakeet, Vercel." if "GitHub" in _prompt(request)
                                  else "What is the problem.")
    engine = fake.engine("gemini", model="gemini-3.5-flash-lite", path="/v1beta")
    engine.words = OWNER_WORDS
    raw = engine.transcribe_chunk(AUDIO, RATE)
    assert raw.text == "What is the problem." and raw.diagnostics["hint_echo"] == "replaced"
    assert "GitHub" in _prompt(fake.requests[0]) and "GitHub" not in _prompt(fake.requests[1])  # asked again, plainly
    assert engine.transcribe(AUDIO, RATE) == "What is the problem."  # the classic path too


def test_terms_really_said_in_their_list_order_are_kept(fake):
    fake.reply = lambda request: ("We ship GitHub, Rflow, Parakeet today." if "GitHub" in _prompt(request)
                                  else "We ship Github, airflow, parakeet today.")
    engine = fake.engine("openai", model="whisper-1")
    engine.words = OWNER_WORDS
    raw = engine.transcribe_chunk(AUDIO, RATE)
    assert raw.text == "We ship GitHub, Rflow, Parakeet today." and raw.diagnostics["hint_echo"] == "spoken"


def test_without_a_second_opinion_the_copied_list_is_taken_out(fake):
    def reply(request):
        if "GitHub" in _prompt(request):
            fake.statuses.append(503)  # the plain request that follows fails
            return "Send it to me, GitHub Rflow Parakeet Vercel."
        return None
    fake.reply = reply
    engine = fake.engine("gemini", model="gemini-3.5-flash-lite", path="/v1beta")
    engine.words = OWNER_WORDS
    raw = engine.transcribe_chunk(AUDIO, RATE)
    assert raw.text == "Send it to me." and raw.diagnostics["hint_echo"] == "removed"


@pytest.mark.parametrize("text, copied", [
    ("Could you tell me what is the problem. version installer. Commit tray app pill. Let's move stand meeting Thursday "
     "morning.", "version installer. Commit tray app pill. Let's move stand meeting Thursday morning"),
    ("this directory, Let's move standard meeting Thursday morning, commits GitHub, Rflow, Parakeet, laptop, Vercel, after "
     "tests.", "meeting Thursday morning, commits GitHub, Rflow, Parakeet, laptop, Vercel, after tests"),
    ("Let's move the meeting to Thursday morning.", None),  # said: not in the list's order, words between
    ("We use GitHub and Vercel every day.", None),
    ("GitHub Rflow", None),  # two in a row is not enough
])
def test_hint_echo_finds_the_list_written_out_in_order(text, copied):
    from sst.engines.cloud import hint_echo
    span = hint_echo(text, OWNER_WORDS)
    assert (text[span[0]:span[1]] if span else None) == copied


def test_taking_out_a_copied_run_tidies_the_punctuation():
    from sst.engines.cloud import without
    text = "Could you tell me the problem, version installer commit tray. Thanks."
    assert without(text, (text.index("version"), text.index("tray") + 4)) == "Could you tell me the problem. Thanks."


# ---- the cost tracker (sst.usage): each Gemini call counted once, for the feature its caller names

@pytest.fixture
def ledger():
    counted = usage.Ledger(None)
    usage.use(counted)  # the conftest puts back none after the test
    return counted


def _entries(ledger) -> dict:
    (day,) = ledger.days.values()
    return day


def test_a_gemini_transcribe_call_is_counted_with_its_usage(fake, ledger):
    fake.answer = {**TRANSCRIBED, "usageMetadata": {
        "promptTokenCount": 60, "candidatesTokenCount": 12, "thoughtsTokenCount": 0, "totalTokenCount": 72,
        "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 50}, {"modality": "TEXT", "tokenCount": 10}]}}
    fake.engine("gemini", path="/v1beta").transcribe_chunk(AUDIO, RATE)
    assert _entries(ledger) == {"dictation|gemini-3.5-transcribe": {
        "audio_in": 50, "text_in": 10, "out": 12, "thoughts": 0, "audio_out": 0, "seconds": 2.0, "calls": 1}}


def test_a_flash_answer_without_usage_is_counted_by_the_audio_length(fake, ledger):
    fake.engine("gemini", model="gemini-3.5-flash-lite", path="/v1beta").transcribe(AUDIO, RATE)
    entry = _entries(ledger)["dictation|gemini-3.5-flash-lite"]
    assert entry["audio_in"] == 64 and entry["calls"] == 1  # 2 s at a Flash model's 32 tokens a second


def test_the_hint_echo_check_is_a_second_call_and_the_test_button_a_test(fake, ledger):
    fake.reply = lambda request: ("What is the problem. GitHub, Rflow, Parakeet, Vercel." if "GitHub" in _prompt(request)
                                  else "What is the problem.")
    engine = fake.engine("gemini", model="gemini-3.5-flash-lite", path="/v1beta")
    engine.words = OWNER_WORDS
    engine.transcribe_chunk(AUDIO, RATE)
    engine.check(AUDIO, RATE)
    entries = _entries(ledger)
    assert entries["dictation|gemini-3.5-flash-lite"]["calls"] == 2 and entries["tests|gemini-3.5-flash-lite"]["calls"] == 2


def test_a_refused_call_and_other_providers_are_not_counted(fake, ledger):
    fake.statuses = [401]
    with pytest.raises(CloudError):
        fake.engine("gemini", path="/v1beta").transcribe_chunk(AUDIO, RATE)
    fake.engine("groq").transcribe(AUDIO, RATE)
    assert ledger.days == {}
