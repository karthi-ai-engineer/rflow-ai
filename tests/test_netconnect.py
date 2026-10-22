"""Connecting within a budget and reading refusals (sst.netconnect), on this machine only: local listening sockets, an
address nothing answers from, a name lookup that hangs."""
import socket
import time

import pytest

from sst import netconnect
from sst.netconnect import NO_CREDIT, NO_KEY, QUOTA, open_socket, refusal

DEAD = "10.255.255.1"  # a private address with nothing behind it: a connection there never answers (or fails at once)


@pytest.fixture
def listener():
    server = socket.create_server(("127.0.0.1", 0))
    yield server.getsockname()[1]
    server.close()


def addresses(monkeypatch, *hosts, port):
    infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port)) for host in hosts]
    monkeypatch.setattr(netconnect.socket, "getaddrinfo", lambda *args, **named: infos)


def test_the_first_address_that_answers_wins_when_the_first_one_stalls(monkeypatch, listener):
    addresses(monkeypatch, DEAD, DEAD, "127.0.0.1", port=listener)  # like Google's first address from the dev laptop
    t0 = time.perf_counter()
    with open_socket("api.example.test", listener, budget=3.0, stagger=0.1) as sock:
        assert sock.getpeername() == ("127.0.0.1", listener) and sock.gettimeout() > 0
    assert time.perf_counter() - t0 < 1.0  # two staggers, not a timeout per dead address


def test_offline_gives_up_within_the_budget(monkeypatch):
    addresses(monkeypatch, *[DEAD] * 8, port=443)  # eight addresses, none answering: 36 s before
    t0 = time.perf_counter()
    with pytest.raises(OSError):
        open_socket("api.example.test", 443, budget=0.6)
    assert time.perf_counter() - t0 < 1.0


def test_a_refused_connection_fails_within_the_budget():
    with socket.socket() as s:  # a port nothing listens on (Windows takes about 2 s to say so)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t0 = time.perf_counter()
    with pytest.raises(OSError):
        open_socket("127.0.0.1", port, budget=0.5)
    assert time.perf_counter() - t0 < 1.0


def test_a_name_lookup_that_hangs_is_given_up_on(monkeypatch):
    monkeypatch.setattr(netconnect.socket, "getaddrinfo", lambda *args, **named: time.sleep(3))
    t0 = time.perf_counter()
    with pytest.raises(OSError, match="name server"):
        open_socket("api.example.test", 443, budget=0.3)
    assert time.perf_counter() - t0 < 0.8


def test_a_failed_lookup_is_said(monkeypatch):
    def fail(*args, **named):
        raise socket.gaierror(11001, "getaddrinfo failed")
    monkeypatch.setattr(netconnect.socket, "getaddrinfo", fail)
    with pytest.raises(socket.gaierror):
        open_socket("api.example.test", 443)


def test_one_tls_context_is_shared():
    assert netconnect.ssl_context() is netconnect.ssl_context()


@pytest.mark.parametrize("status, body, headers, reason, model_only", [
    (402, "{}", {}, NO_CREDIT, False),
    (401, '{"error": {"message": "Invalid API Key"}}', {}, NO_KEY, False),
    (403, '{"error": {"status": "PERMISSION_DENIED"}}', {}, NO_KEY, False),
    (429, '{"error": {"code": "insufficient_quota", "message": "check your plan and billing details"}}', {}, NO_CREDIT,
     False),  # OpenAI's way of saying there's no credit
    (400, '{"error": {"message": "Your credit balance is too low to access the Anthropic API"}}', {}, NO_CREDIT, False),
    (429, '{"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"violations": [{"quotaId": "GenerateRequestsPerDay'
          'PerProjectPerModel-FreeTier"}]}]}}', {}, QUOTA, True),
    (429, "Rate limit reached on tokens per day (TPD). Please try again in 7m12s", {"retry-after": "432"}, QUOTA, True),
    (429, '[{"error": {"status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota"}}]', {}, QUOTA, True),
])
def test_refusals(status, body, headers, reason, model_only):
    found = refusal(status, body, headers)
    assert found is not None and found.reason == reason and found.model_only is model_only


def test_a_long_rate_limit_is_refused_for_its_length_and_a_short_one_retried():
    gemini = ('{"error": {"status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota", "details": '
              '[{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"}]}}')
    found = refusal(429, gemini)
    assert found.rate_limit and found.seconds == 41 and found.model_only
    assert refusal(429, "{}", {"retry-after": "2"}) is None and netconnect.retry_after("{}", {"retry-after": "2"}) == 2
    assert refusal(429, '{"error": {"message": "Rate limit reached. Please try again in 450ms."}}') is None
    for status in (400, 404, 408, 500, 503):
        assert refusal(status, '{"error": {"message": "backend exploded"}}') is None


def test_a_refusal_is_remembered_per_key_and_model_until_an_answer():
    netconnect.forget_refusals()
    try:
        netconnect.refuse("host:", "key-1", "model-a", netconnect.Refusal(QUOTA, model_only=True))
        assert netconnect.refused("host:", "key-1", "model-a") == QUOTA
        assert not netconnect.refused("host:", "key-1", "model-b") and not netconnect.refused("host:", "key-2", "model-a")
        netconnect.refuse("host:", "key-1", "model-a", netconnect.Refusal(NO_CREDIT))
        assert netconnect.refused("host:", "key-1", "model-b") == NO_CREDIT  # the account's: every model
        netconnect.forgive("host:", "key-1", "model-b")
        assert not netconnect.refused("host:", "key-1", "model-b")
        assert netconnect.refused("host:", "key-1", "model-a") == QUOTA  # that model's own limit stays
        netconnect.refuse("host:", "key-3", "", netconnect.Refusal(NO_KEY, seconds=-1))
        assert not netconnect.refused("host:", "key-3", "")  # remembered only so long
    finally:
        netconnect.forget_refusals()
