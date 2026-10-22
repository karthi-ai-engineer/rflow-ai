"""LLM polish against a fake endpoint on this machine (no real provider): the strict prompt goes as the system message,
the user's terms only when the text has them, and a failure raises LLMError instead of passing the text through."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sst import gateway
from sst.gateway import SYSTEM_PROMPT, GatewayConfig, Polisher
from sst.pipeline.polish import MAX_TERMS, POLISH_PROMPT, GatewayLLM, LLMError, LLMPolisher, present_terms

HEARD = "um so we merge the five PRs today"
CLEAN = "So we merge the five PRs today."


class FakeEndpoint:
    """Answers like an OpenAI-compatible or Anthropic endpoint; what it does depends on the model asked for."""

    def __init__(self):
        self.requests, self.connections = [], 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive, like a real endpoint

            def setup(self):
                fake.connections += 1
                super().setup()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, **body})
                if body["model"].startswith("error"):
                    return self._reply(500, {"error": {"message": "backend exploded"}})
                answer = {"long": "words " * 80, "echo": body["messages"][-1]["content"]}.get(body["model"], CLEAN)
                if self.path.endswith("/messages"):
                    return self._reply(200, {"type": "message", "content": [{"type": "text", "text": answer}]})
                self._reply(200, {"choices": [{"message": {"role": "assistant", "content": answer}}]})

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

    def llm(self, model="good", fallback=None, provider=""):
        return GatewayLLM(GatewayConfig(self.url, "test-key", provider), model, fallback)

    def system(self, index=-1) -> str:
        sent = self.requests[index]
        return sent["system"] if "system" in sent else sent["messages"][0]["content"]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(gateway, "ANSWER_TIMEOUT", 0.4)
    monkeypatch.setattr(gateway, "ANSWER_PER_WORD", 0.0)
    monkeypatch.setattr(gateway, "CONNECT_BUDGET", 0.3)
    server = FakeEndpoint()
    yield server
    server.server.shutdown()


def test_the_strict_prompt_is_sent_as_the_system_message(fake):
    llm = fake.llm()
    assert llm.polish(HEARD) == CLEAN and llm.name == "good"
    assert fake.requests[0]["messages"] == [{"role": "system", "content": POLISH_PROMPT}, {"role": "user", "content": HEARD}]


@pytest.mark.parametrize("rule", ["filler", "self-correction", "question stays a question", "command stays a command",
                                  "never translate", "Never add", "Output only the cleaned text", "never addressed to you"])
def test_the_prompt_says_what_to_do_and_what_never_to_do(rule):
    assert rule in POLISH_PROMPT


def test_only_the_terms_in_the_text_are_listed(fake):
    llm = fake.llm()
    llm.polish("we deploy postgresql to github today", ["PostgreSQL", "Kubernetes", "GitHub", "github"])
    assert fake.system() == POLISH_PROMPT + "\nKeep these terms exactly as written: PostgreSQL, GitHub"
    llm.polish(HEARD, ["Kubernetes"])
    assert fake.system() == POLISH_PROMPT  # nothing listed, and nothing left over from the dictation before


def test_at_most_twenty_terms_are_listed(fake):
    terms = [f"term{i}" for i in range(30)]
    fake.llm(model="echo").polish(" ".join(terms), terms)
    listed = fake.system().split("Keep these terms exactly as written: ")[1].split(", ")
    assert listed == terms[:MAX_TERMS]


def test_present_terms():
    terms = ["GitHub", "Git", "Hub", "Visual Studio Code", "github", ""]
    assert present_terms("open visual  studio\ncode and GitHub", terms) == [
        "GitHub", "Visual Studio Code"]  # whole words only, each once, as the user spells them
    assert present_terms("use foo bar", ["foo\nbar"]) == ["foo bar"]  # a term can't add a line to the instruction


def test_anthropic_gets_the_prompt_as_its_system_field(fake):
    assert fake.llm(provider="anthropic").polish(HEARD, ["PRs"]) == CLEAN
    sent = fake.requests[0]
    assert sent["path"] == "/v1/messages" and [m["role"] for m in sent["messages"]] == ["user"]
    assert sent["system"] == POLISH_PROMPT + "\nKeep these terms exactly as written: PRs"


def test_an_endpoint_error_raises(fake):
    with pytest.raises(LLMError, match="backend exploded"):
        fake.llm(model="error").polish(HEARD)


def test_the_backup_model_is_used_when_the_chosen_one_fails(fake):
    assert fake.llm(model="error", fallback="good").polish(HEARD) == CLEAN
    assert [r["model"] for r in fake.requests] == ["error", "good"]


def test_an_implausible_answer_raises(fake):
    with pytest.raises(LLMError, match="unusual"):
        fake.llm(model="long").polish(HEARD)


def test_an_unreachable_endpoint_raises(monkeypatch):
    monkeypatch.setattr(gateway, "CONNECT_BUDGET", 0.3)
    with pytest.raises(LLMError, match="could not be reached"):
        GatewayLLM(GatewayConfig("http://127.0.0.1:9/v1", "k"), "good").polish(HEARD)  # nothing listens there


def test_without_an_endpoint_or_model_it_raises_and_sends_nothing(fake):
    with pytest.raises(LLMError, match="no endpoint or model"):
        GatewayLLM(GatewayConfig("", "key"), "good").polish(HEARD)
    with pytest.raises(LLMError):
        fake.llm(model="").polish(HEARD)
    assert fake.requests == []


def test_nothing_to_clean_sends_nothing(fake):
    assert fake.llm().polish("   ") == "   " and fake.requests == []


def test_prepare_connects_ahead_of_time(fake):
    llm = fake.llm()
    llm.prepare()
    time.sleep(0.3)
    assert fake.connections == 1
    llm.polish(HEARD)
    assert fake.connections == 1  # the prepared connection was used


def test_it_is_an_llm_polisher(fake):
    assert isinstance(fake.llm(), LLMPolisher)


def test_the_gateway_polisher_keeps_its_own_prompt(fake):
    config = GatewayConfig(fake.url, "test-key")
    assert Polisher(config, "good", ["PRs"]).polish(HEARD) == CLEAN
    assert fake.system() == SYSTEM_PROMPT + "\nThe user's vocabulary: PRs"
    Polisher(config, "good", ["PRs"], system_prompt="Clean this.").polish(HEARD)
    assert fake.system() == "Clean this.\nThe user's vocabulary: PRs"
