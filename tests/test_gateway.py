"""Text cleanup against a fake gateway on this machine (no company server involved): the answer is used when it's good,
and the text is typed as heard, quickly, whenever the gateway can't help."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sst import gateway, netconnect, usage
from sst.gateway import GatewayConfig, GatewayError, Polisher, plausible

HEARD = "so we merge the five pr's today"
CLEAN = "So we merge the five PRs today."
# Answers that say asking again won't help, as the providers write them (a model by this name gets it)
REFUSALS = {"no-credit": (402, {"error": {"message": "Insufficient balance"}}),
            "bad-key": (401, {"error": {"message": "Invalid API Key"}}),
            "quota": (429, [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current "
                                       "quota", "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                                            "violations": [{"quotaId": "RequestsPerDayPerModel"}]}]}}])}


@pytest.fixture(autouse=True)
def no_refusals():
    netconnect.forget_refusals()  # what one test's fake provider refused never reaches another test
    yield
    netconnect.forget_refusals()


class FakeGateway:
    """Answers like the gateway; what it does depends on the model asked for."""

    def __init__(self):
        self.requests, self.connections = [], 0
        self.models = [{"id": "gpt-cloud", "is_cloud": True}, {"id": "zeta-local", "is_cloud": False},
                       {"id": "Alpha-local", "is_cloud": False}, {"id": "plain-model"}]
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive, like the real gateway

            def setup(self):
                fake.connections += 1
                super().setup()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                      "x-api-key": self.headers.get("x-api-key"),
                                      "anthropic-version": self.headers.get("anthropic-version"), **body})
                anthropic = self.path.endswith("/messages")  # Anthropic's Messages API, not OpenAI's chat/completions
                text = body["messages"][-1]["content"]
                model = body["model"]
                if model in REFUSALS:
                    return self._reply(*REFUSALS[model])
                if model.startswith("error"):
                    if anthropic:
                        return self._reply(400, {"type": "error", "error": {"type": "invalid_request_error",
                                                                             "message": "backend exploded"}})
                    return self._reply(500, {"error": {"message": "backend exploded"}})
                if model == "slow":
                    time.sleep(1.0)
                if model.startswith("cut"):  # stopped at the token limit: part of an answer, or none (thinking used it)
                    part = "So we merge the five" if model == "cut" else ""
                    if anthropic:
                        return self._reply(200, {"type": "message", "role": "assistant", "stop_reason": "max_tokens",
                                                 "content": [{"type": "text", "text": part}] if part else []})
                    return self._reply(200, {"choices": [{"message": {"role": "assistant", "content": part},
                                                          "finish_reason": "length"}]})
                answer = {"long": text + " and then some more words " * 10,
                          "think": "<think>let me see</think> So we merge the five PRs today.",
                          "quoted": '"So we merge the five PRs today."'}.get(model, "So we merge the five PRs today.")
                if anthropic:
                    return self._reply(200, {"type": "message", "role": "assistant",
                                             "content": [{"type": "text", "text": answer}]})
                self._reply(200, {"choices": [{"message": {"role": "assistant", "content": answer}}],
                                  "usage": {"prompt_tokens": 120, "completion_tokens": 9, "total_tokens": 189}})

            def do_GET(self):
                fake.requests.append({"auth": self.headers.get("Authorization"), "x-api-key": self.headers.get("x-api-key"),
                                      "path": self.path})
                if self.path.split("?")[0] != "/v1/models":
                    return self._reply(404, {"detail": "not found"})
                self._reply(200, {"data": fake.models})

            def _reply(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def polisher(self, model="good", fallback=None, vocabulary=(), provider=""):
        return Polisher(GatewayConfig(self.url, "test-key", provider), model, list(vocabulary), fallback)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(gateway, "ANSWER_TIMEOUT", 0.4)
    monkeypatch.setattr(gateway, "ANSWER_PER_WORD", 0.0)
    monkeypatch.setattr(gateway, "CONNECT_BUDGET", 0.3)
    server = FakeGateway()
    yield server
    server.server.shutdown()


def test_a_good_answer_is_used(fake):
    p = fake.polisher(vocabulary=["PRs", "GitHub"])
    assert p.polish(HEARD) == "So we merge the five PRs today." and p.last_error == ""
    sent = fake.requests[0]
    assert sent["auth"] == "Bearer test-key" and sent["model"] == "good"
    assert "PRs, GitHub" in sent["messages"][0]["content"]  # the user's words reach the model
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_connection_is_kept_open_between_dictations(fake):
    p = fake.polisher()
    p.polish(HEARD)
    p.polish(HEARD)
    assert fake.connections == 1


def test_prepare_connects_ahead_of_time(fake):
    p = fake.polisher()
    p.prepare()
    time.sleep(0.3)
    assert fake.connections == 1
    p.polish(HEARD)
    assert fake.connections == 1  # the prepared connection was used


def test_the_other_model_is_tried_when_the_chosen_one_fails(fake):
    p = fake.polisher(model="error", fallback="good")
    assert p.polish(HEARD) == "So we merge the five PRs today."
    assert [r["model"] for r in fake.requests] == ["error", "good"]


def test_both_models_failing_gives_the_heard_text(fake):
    p = fake.polisher(model="error-1", fallback="error-2")
    assert p.polish(HEARD) == HEARD and "HTTP 500" in p.last_error
    assert [r["model"] for r in fake.requests] == ["error-1", "error-2"]


def test_a_slow_answer_gives_the_heard_text_in_time(fake):
    p = fake.polisher(model="slow")
    t0 = time.perf_counter()
    assert p.polish(HEARD) == HEARD
    assert time.perf_counter() - t0 < 0.9  # waited for the answer timeout (0.4 s), not the 1 s answer
    assert "took too long" in p.last_error


def test_a_slow_answer_gives_way_to_the_backup_model(fake):
    p = fake.polisher(model="slow", fallback="good")
    t0 = time.perf_counter()
    assert p.polish(HEARD) == "So we merge the five PRs today." and p.last_error == ""
    assert time.perf_counter() - t0 < 0.9  # the answer timeout (0.4 s), then the backup's quick answer
    assert [r["model"] for r in fake.requests] == ["slow", "good"]


def test_the_backup_is_never_the_chosen_model_again(fake):
    p = fake.polisher(model="error", fallback="error")
    assert p.polish(HEARD) == HEARD and [r["model"] for r in fake.requests] == ["error"]


def test_an_unreachable_gateway_is_skipped_for_a_while(monkeypatch):
    monkeypatch.setattr(gateway, "CONNECT_BUDGET", 0.3)
    p = Polisher(GatewayConfig("http://127.0.0.1:9/v1", "k"), "good")  # nothing listens there
    assert p.polish(HEARD) == HEARD and "could not be reached" in p.last_error
    t0 = time.perf_counter()
    assert p.polish(HEARD) == HEARD  # skipped at once, no new connection attempts
    assert time.perf_counter() - t0 < 0.05


@pytest.mark.parametrize("model, said", [("no-credit", "no credit left on the account"),
                                          ("bad-key", "didn't accept the API key")])
def test_a_refused_key_is_said_plainly_and_not_asked_again(fake, model, said):
    p = fake.polisher(model=model, fallback="good")  # the same key: the backup model is refused too
    assert p.polish(HEARD) == HEARD and said in p.last_error and "HTTP 4" in p.last_error
    t0 = time.perf_counter()
    assert fake.polisher(model=model).polish(HEARD) == HEARD  # a new polisher (Text Transform makes one each time)
    assert time.perf_counter() - t0 < 0.05 and len(fake.requests) == 1  # not asked again
    with pytest.raises(GatewayError, match=said):
        fake.polisher(model=model).complete(HEARD)
    assert len(fake.requests) == 1
    p = Polisher(GatewayConfig(fake.url, "new-key"), model)  # a new key is asked
    p.polish(HEARD)
    assert len(fake.requests) == 2


def test_a_used_up_quota_is_the_models_and_its_backup_answers(fake):
    p = fake.polisher(model="quota", fallback="good")  # Gemini's error, wrapped in a list
    assert p.polish(HEARD) == CLEAN and p.last_error == ""
    assert p.polish(HEARD) == CLEAN
    assert [r["model"] for r in fake.requests] == ["quota", "good", "good"]  # the used-up model wasn't asked again
    p = fake.polisher(model="quota")
    assert p.polish(HEARD) == HEARD and "quota is used up" in p.last_error


def test_the_test_button_asks_anyway_and_an_answer_clears_the_refusal(fake):
    fake.polisher(model="bad-key").polish(HEARD)  # the key is refused...
    p = fake.polisher()
    assert p.polish(HEARD) == HEARD and "a moment ago" in p.last_error  # ...for every model on it
    assert "answered in" in fake.polisher().check()  # the user fixed it and pressed Test
    assert fake.polisher().polish(HEARD) == CLEAN


def test_the_backup_at_another_provider_answers_when_the_chosen_one_refuses(fake):
    # Groq chosen (here, the fake), and a Gemini key saved too: Gemini's setup model stands in.
    others = {"gemini": (fake.url, "gemini-key")}
    p = Polisher(GatewayConfig(fake.url, "groq-key", "groq", others), "no-credit")
    assert p.polish(HEARD) == CLEAN and p.last_error == ""
    assert [(r["model"], r["auth"]) for r in fake.requests] == [("no-credit", "Bearer groq-key"),
                                                               ("gemini-3.5-flash-lite", "Bearer gemini-key")]
    assert p.polish(HEARD) == CLEAN and len(fake.requests) == 3  # straight to the backup now
    # Without another provider's key, nothing else is asked; an own server's text never goes to a cloud provider.
    for config in (GatewayConfig(fake.url, "groq-key-2", "groq"), GatewayConfig(fake.url, "k", "vllm", others)):
        assert Polisher(config, "no-credit").polish(HEARD) == HEARD
    assert len(fake.requests) == 5


def test_gemini_has_no_backup_at_another_provider_while_it_is_the_only_one_offered(fake):
    from sst.setups import backup_for
    assert gateway.OFFERED == ("gemini", "ollama", "vllm")  # OpenAI, Anthropic and Groq kept, not offered
    others = {"groq": (fake.url, "groq-key"), "openai": (fake.url, "sk-key"), "anthropic": (fake.url, "ant-key")}
    assert backup_for(GatewayConfig(fake.url, "gemini-key", "gemini", others)) is None
    p = Polisher(GatewayConfig(fake.url, "gemini-key", "gemini", others), "no-credit")
    assert p.polish(HEARD) == HEARD and [r["model"] for r in fake.requests] == ["no-credit"]  # nothing else asked


@pytest.mark.parametrize("model", ["long"])
def test_an_implausible_answer_is_not_used(fake, model):
    p = fake.polisher(model=model)
    assert p.polish(HEARD) == HEARD and "unusual" in p.last_error


@pytest.mark.parametrize("provider", ["groq", "anthropic"])
def test_an_answer_cut_off_at_the_token_limit_is_never_used(fake, provider):
    # "So we merge the five" would pass as plausible: a text with a hole in it. The backup model's is used instead.
    assert fake.polisher(model="cut", fallback="good", provider=provider).polish(HEARD) == "So we merge the five PRs today."
    p = fake.polisher(model="cut-empty", provider=provider)  # a model whose thinking used up the limit
    assert p.polish(HEARD) == HEARD and "ran out of tokens before answering" in p.last_error
    with pytest.raises(GatewayError, match="ran out of tokens before the end of its answer"):
        fake.polisher(model="cut", provider=provider).complete(HEARD)


def test_text_transform_and_translate_get_a_higher_token_limit(fake):
    polisher = fake.polisher(model="gpt-4.1-mini", provider="openai")
    polisher.polish(HEARD)
    cleanup = _last_post(fake)["max_completion_tokens"]
    polisher.complete(HEARD)  # a translation into Tamil or Japanese takes several tokens a word
    assert _last_post(fake)["max_completion_tokens"] >= cleanup + 200


@pytest.mark.parametrize("model", ["think", "quoted"])
def test_thinking_tags_and_quotes_are_removed(fake, model):
    assert fake.polisher(model=model).polish(HEARD) == "So we merge the five PRs today."


def test_a_local_endpoint_works_without_a_key(fake):
    p = Polisher(GatewayConfig(fake.url, ""), "good")  # e.g. Ollama on localhost
    assert p.polish(HEARD) == "So we merge the five PRs today."
    assert fake.requests[0]["auth"] is None  # no empty "Bearer" header is sent


def test_without_an_endpoint_or_model_nothing_is_sent(fake):
    assert Polisher(GatewayConfig("", "key"), "good").polish(HEARD) == HEARD
    assert fake.polisher(model="").polish(HEARD) == HEARD
    assert fake.requests == []


def test_load_models_lists_local_models_first(fake):
    assert fake.polisher(model="").models() == ["Alpha-local", "zeta-local", "gpt-cloud", "plain-model"]
    assert fake.requests[-1] == {"auth": "Bearer test-key", "x-api-key": None, "path": "/v1/models"}


def test_load_models_reports_an_unreachable_endpoint(monkeypatch):
    monkeypatch.setattr(gateway, "CONNECT_BUDGET", 0.3)
    with pytest.raises(GatewayError, match="could not reach"):
        Polisher(GatewayConfig("http://127.0.0.1:9/v1", "k"), "").models()


def test_check_reports_the_answer_or_a_readable_reason(fake):
    assert "answered in" in fake.polisher().check()
    with pytest.raises(GatewayError, match="HTTP 500"):
        fake.polisher(model="error").check()


def test_the_key_is_stored_encrypted_and_never_shows_in_logs(tmp_path):
    path = tmp_path / "gateway.json"
    config = GatewayConfig("https://gw.example/v1", "secret-key-123")
    assert "secret-key-123" not in repr(config)
    config.save(path)
    assert "secret-key-123" not in path.read_text(encoding="utf-8")  # encrypted for this Windows user (DPAPI)
    assert GatewayConfig.load(path) == config
    assert GatewayConfig.load(tmp_path / "missing.json") == GatewayConfig()


def test_a_key_pasted_into_the_file_by_hand_is_encrypted_on_first_load(tmp_path):
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps({"base_url": "https://gw.example/v1", "api_key": "pasted-key"}), encoding="utf-8")
    assert GatewayConfig.load(path).api_key == "pasted-key"
    assert "pasted-key" not in path.read_text(encoding="utf-8")
    assert GatewayConfig.load(path).api_key == "pasted-key"


def test_a_key_that_cannot_be_decrypted_is_dropped_not_crashed_on(tmp_path):
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps({"base_url": "https://gw.example/v1", "api_key_protected": "bm90IGEgcmVhbCBibG9i"}),
                    encoding="utf-8")  # e.g. copied from another laptop
    assert GatewayConfig.load(path) == GatewayConfig("https://gw.example/v1", "")


@pytest.mark.parametrize("cleaned, ok", [("So we merge the five PRs today.", True), ("", False), ("Yes.", False),
                                         (HEARD + " extra words " * 10, False)])
def test_plausible(cleaned, ok):
    assert plausible(HEARD, cleaned) is ok


def test_removing_fillers_from_a_short_dictation_is_plausible():
    assert plausible("um uh yes okay", "Yes, okay.")


# ---- providers

def _last_post(fake) -> dict:
    return next(r for r in reversed(fake.requests) if "model" in r)


def test_openai_gets_its_current_limit_name_and_no_self_hosted_options(fake):
    assert fake.polisher(provider="openai").polish(HEARD) == "So we merge the five PRs today."
    sent = _last_post(fake)
    assert sent["path"] == "/v1/chat/completions" and sent["auth"] == "Bearer test-key"
    assert sent["temperature"] == 0 and sent["max_completion_tokens"] > 0
    assert "max_tokens" not in sent and "chat_template_kwargs" not in sent  # OpenAI refuses fields it doesn't know


@pytest.mark.parametrize("model, effort", [("gpt-5-mini", "minimal"), ("gpt-6-luna", "none"), ("gpt-6-astra", "low"),
                                           ("o3", None)])
def test_openai_reasoning_models_think_least_with_room_and_no_temperature(fake, model, effort):
    fake.polisher(model=model, provider="openai").polish(HEARD)
    sent = _last_post(fake)
    assert "temperature" not in sent and sent["max_completion_tokens"] > gateway.THINKING_ROOM
    assert sent.get("reasoning_effort") == effort  # the least it allows; a model not known keeps its default


def test_anthropic_gets_its_messages_api(fake):
    polisher = fake.polisher(model="claude-haiku-4-5", provider="anthropic", vocabulary=["PRs"])
    assert polisher.polish(HEARD) == "So we merge the five PRs today."
    sent = _last_post(fake)
    assert sent["path"] == "/v1/messages" and sent["auth"] is None
    assert sent["x-api-key"] == "test-key" and sent["anthropic-version"] == gateway.ANTHROPIC_VERSION
    assert "PRs" in sent["system"] and [m["role"] for m in sent["messages"]] == ["user"]
    assert 0 < sent["max_tokens"] < gateway.THINKING_ROOM and sent["temperature"] == 0  # Haiku 4.5 takes it


def test_claude_haiku_5_5_is_told_not_to_think(fake):
    fake.polisher(model="claude-haiku-5-5", provider="anthropic").polish(HEARD)
    sent = _last_post(fake)
    assert sent["thinking"] == {"type": "disabled"} and "temperature" not in sent  # 4.7 and later refuse it
    assert 0 < sent["max_tokens"] < gateway.THINKING_ROOM  # no room for thinking it won't do
    fake.polisher(model="claude-sonnet-5-5", provider="anthropic").polish(HEARD)
    assert "thinking" not in _last_post(fake)


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-4-7", "a-name-of-its-own"])
def test_claude_4_7_and_later_get_no_temperature_and_room_to_think(fake, model):
    # Anthropic answers HTTP 400 to temperature 0 from Claude 4.7 on, and Opus 5.5 thinks whether asked or not.
    fake.polisher(model=model, provider="anthropic").polish(HEARD)
    sent = _last_post(fake)
    assert "temperature" not in sent and sent["max_tokens"] > gateway.THINKING_ROOM


def test_anthropic_errors_are_readable(fake):
    polisher = fake.polisher(model="error-model", provider="anthropic")
    assert polisher.polish(HEARD) == HEARD and "backend exploded" in polisher.last_error


@pytest.mark.parametrize("model, sent_too", [
    ("gemini-3.5-flash-lite", {"reasoning_effort": "minimal"}), ("gemini-3.8-flash", {"reasoning_effort": "low"}),
    ("gemini-flash-lite-latest", {}), ("gemini-2.5-flash-lite", {"temperature": 0})])
def test_gemini_gets_no_token_limit_and_gemini_3_thinks_least_at_its_own_temperature(fake, model, sent_too):
    fake.polisher(model=model, provider="gemini").polish(HEARD)
    sent = _last_post(fake)
    assert "max_tokens" not in sent and "max_completion_tokens" not in sent and "chat_template_kwargs" not in sent
    assert {k: sent[k] for k in ("reasoning_effort", "temperature") if k in sent} == sent_too


@pytest.mark.parametrize("provider", ["ollama", "vllm"])
def test_self_hosted_servers_get_the_no_thinking_option(fake, provider):
    fake.polisher(provider=provider).polish(HEARD)
    sent = _last_post(fake)
    assert sent["max_tokens"] > 0 and sent["temperature"] == 0 and sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_groq_s_gpt_oss_thinks_little_with_room_for_it(fake):
    # llama-3.1-8b-instant is shut down; gpt-oss-20b thinks at "medium" unless told, and its thinking counts against
    # the limit: without room, the answer could come back empty.
    assert fake.polisher(model="openai/gpt-oss-20b", provider="groq").polish(HEARD) == "So we merge the five PRs today."
    sent = _last_post(fake)
    assert sent["reasoning_effort"] == "low" and sent["include_reasoning"] is False and "temperature" not in sent
    assert sent["max_completion_tokens"] > gateway.THINKING_ROOM and "max_tokens" not in sent  # Groq's current name
    assert "chat_template_kwargs" not in sent
    fake.polisher(model="a-model-that-answers", provider="groq").polish(HEARD)
    sent = _last_post(fake)
    assert sent["temperature"] == 0 and 0 < sent["max_completion_tokens"] < gateway.THINKING_ROOM
    assert "reasoning_effort" not in sent and "include_reasoning" not in sent


def test_load_models_leaves_out_models_that_dont_write_text(fake):
    fake.models = [{"id": "gpt-4o-mini"}, {"id": "whisper-1"}, {"id": "text-embedding-3-small"}, {"id": "tts-1"},
                   {"id": "models/gemini-2.5-flash"}, {"id": "llama-guard-4"}, {"id": "dall-e-3"},
                   {"id": "canopylabs/orpheus-arabic-saudi"}, {"id": "openai/gpt-oss-safeguard-20b"},
                   {"id": "openai/gpt-oss-20b"}]
    assert fake.polisher().models() == ["gemini-2.5-flash", "gpt-4o-mini", "openai/gpt-oss-20b"]


def test_anthropic_models_are_listed_with_its_headers(fake):
    fake.models = [{"type": "model", "id": "claude-model-a", "display_name": "A"}]
    assert fake.polisher(provider="anthropic").models() == ["claude-model-a"]
    listed = fake.requests[-1]
    assert listed["path"] == "/v1/models?limit=1000" and listed["x-api-key"] == "test-key" and listed["auth"] is None


@pytest.mark.parametrize("url, provider", [
    ("https://api.openai.com/v1", "openai"), ("https://api.anthropic.com/v1", "anthropic"),
    ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini"), ("https://api.groq.com/openai/v1", "groq"),
    ("http://localhost:11434/v1", "ollama"), ("http://127.0.0.1:11434/v1", "ollama"),
    ("https://gw.example/v1", "vllm"), ("http://localhost:8000/v1", "vllm"), ("", "vllm")])
def test_settings_from_before_the_provider_choice_get_their_provider(url, provider):
    assert gateway.provider_for(url) == provider
    assert GatewayConfig(url).service.key == provider


def test_a_cloud_provider_uses_its_usual_address():
    assert GatewayConfig("", "key", "openai").address == "https://api.openai.com/v1"
    assert GatewayConfig("http://my-server:8000/v1/", "", "vllm").address == "http://my-server:8000/v1"
    assert GatewayConfig().address == ""  # nothing chosen yet: no requests


def test_every_provider_s_key_is_kept_encrypted_when_switching(tmp_path):
    path = tmp_path / "gateway.json"
    config = GatewayConfig("", "groq-secret", "groq", {"openai": ("", "openai-secret"),
                                                        "vllm": ("https://gw.example/v1", "gw-secret")})
    config.save(path)
    text = path.read_text(encoding="utf-8")
    assert not any(secret in text for secret in ("groq-secret", "openai-secret", "gw-secret"))
    assert GatewayConfig.load(path) == config
    assert "secret" not in repr(config)


def test_cloud_speech_shares_each_providers_key():
    config = GatewayConfig("", "sk-openai", "openai", {"anthropic": ("", "ant-key"), "vllm": ("http://10.0.0.5/v1", "")})
    assert config.key_for("openai") == "sk-openai" and config.key_for("anthropic") == "ant-key"
    assert config.key_for("groq") == ""
    groq = config.with_key("groq", "gsk-key")
    assert groq.key_for("groq") == "gsk-key" and groq.service.key == "openai"  # AI cleanup's choice stays
    assert config.key_for("groq") == ""  # a copy: the original is unchanged
    assert groq.with_key("openai", "sk-new").api_key == "sk-new"
    vllm = config.with_key("vllm", "secret")
    assert vllm.others["vllm"] == ("http://10.0.0.5/v1", "secret")  # its address is kept
    assert "groq" not in groq.with_key("groq", "").others  # a key removed leaves nothing behind
    assert set(groq.entries()) == {"openai", "anthropic", "vllm", "groq"}


def test_saving_a_key_never_chooses_or_changes_the_cleanup_s_provider():
    # The user testing's M-04: a Gemini key added for live translation switched AI cleanup to Gemini, with no model.
    config = GatewayConfig("", "sk-openai", "openai").with_key("gemini", "AIza-live")
    assert (config.chosen, config.api_key, config.key_for("gemini")) == ("openai", "sk-openai", "AIza-live")
    legacy = GatewayConfig("https://api.openai.com/v1", "sk-old")  # from before the provider choice: by its address
    assert legacy.with_key("openai", "sk-new") == GatewayConfig("https://api.openai.com/v1", "sk-new")
    nothing = GatewayConfig()  # a new install: no provider chosen yet
    assert nothing.chosen == "" and nothing.service.key == "vllm"
    server = nothing.with_entry("vllm", "http://localhost:8000/v1", "")
    assert server.chosen == "" and server.address == ""  # kept for later, not made the cleanup's server
    assert server.entries() == {"vllm": ("http://localhost:8000/v1", "")} and server.key_for("vllm") == ""
    assert server.with_key("vllm", "srv-key").others["vllm"] == ("http://localhost:8000/v1", "srv-key")


def test_a_key_saved_for_speech_survives_saving_and_loading(tmp_path):
    path = tmp_path / "gateway.json"
    GatewayConfig().with_key("gemini", "g-key").save(path)
    assert "g-key" not in path.read_text(encoding="utf-8")  # encrypted
    assert GatewayConfig.load(path).key_for("gemini") == "g-key"


def test_the_speech_server_keeps_its_own_address_and_key(tmp_path):
    from sst.gateway import SPEECH_SERVER
    path = tmp_path / "gateway.json"
    config = GatewayConfig("http://gateway.example/v1", "cleanup-key", "vllm")
    speech = config.with_entry(SPEECH_SERVER, "http://localhost:8000/v1", "speech-key")
    assert speech.speech_server() == ("http://localhost:8000/v1", "speech-key")
    assert (speech.base_url, speech.api_key) == ("http://gateway.example/v1", "cleanup-key")  # AI cleanup's untouched
    speech.save(path)
    assert "speech-key" not in path.read_text(encoding="utf-8")  # encrypted
    loaded = GatewayConfig.load(path)
    assert loaded.speech_server() == ("http://localhost:8000/v1", "speech-key") and loaded.service.key == "vllm"
    assert loaded.with_entry(SPEECH_SERVER, "", "").speech_server() == ("", "")
    assert GatewayConfig().speech_server() == ("", "")


# ---- the cost tracker (sst.usage): a Gemini answer counted once, for the feature its caller names

@pytest.fixture
def ledger():
    counted = usage.Ledger(None)
    usage.use(counted)  # the conftest puts back none after the test
    return counted


def _calls(ledger) -> dict:
    (day,) = ledger.days.values()
    return {key: entry["calls"] for key, entry in day.items()}


def test_a_gemini_cleanup_is_counted_with_the_thinking_as_output(fake, ledger):
    assert fake.polisher(provider="gemini").polish(HEARD) == CLEAN
    # total - prompt is billed as output: 9 written and 60 of thinking, which Gemini leaves out of completion_tokens
    assert ledger.days[next(iter(ledger.days))] == {"cleanup|good": {
        "audio_in": 0, "text_in": 120, "out": 9, "thoughts": 60, "audio_out": 0, "seconds": 0, "calls": 1}}


def test_a_backup_model_is_a_call_of_its_own_and_a_failed_one_none(fake, ledger):
    fake.polisher("cut", fallback="good", provider="gemini").polish(HEARD)  # cut short: billed, never used
    assert _calls(ledger) == {"cleanup|cut": 1, "cleanup|good": 1}
    fake.polisher("error", fallback="good", provider="gemini").polish(HEARD)
    assert _calls(ledger) == {"cleanup|cut": 1, "cleanup|good": 2}


def test_the_caller_names_the_feature_and_the_test_button_is_a_test(fake, ledger):
    polisher = fake.polisher(provider="gemini")
    with usage.feature("transform"):
        polisher.complete(HEARD)
    with usage.feature("tests"), usage.feature("translate"):  # the outermost caller says what it is for
        polisher.complete(HEARD)
    polisher.check()
    assert _calls(ledger) == {"transform|good": 1, "tests|good": 2}


def test_only_gemini_is_counted(fake, ledger):
    fake.polisher().polish(HEARD)
    fake.polisher(provider="openai").polish(HEARD)
    assert ledger.days == {}
