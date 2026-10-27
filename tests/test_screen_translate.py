"""Screen translator's translator (sst.screen.translate): Gemini through a fake post, the cache, the cost."""
import json
import threading

import pytest

from sst import usage
from sst.screen import translate as t


def _answer(texts, usage_meta=True):
    answer = {"candidates": [{"content": {"parts": [{"text": json.dumps(texts, ensure_ascii=False)}]}}]}
    if usage_meta:
        answer["usageMetadata"] = {"promptTokenCount": 150, "candidatesTokenCount": 40, "thoughtsTokenCount": 10}
    return answer


class FakePost:
    def __init__(self, answer=None, error=None):
        self.calls, self.answer, self.error = [], answer, error

    def __call__(self, url, body, key, timeout):
        self.calls.append((url, body, key))
        if self.error:
            raise self.error
        texts = json.loads(body["contents"][0]["parts"][0]["text"].split("Input:\n", 1)[1])
        return self.answer if self.answer is not None else _answer([f"English {len(x)}" for x in texts])


@pytest.fixture
def ledger():
    led = usage.Ledger(None)
    usage.use(led)
    yield led
    usage.use(None)


def test_gemini_gets_the_strings_as_a_json_array_and_each_call_is_counted(ledger):
    post = FakePost()
    gemini = t.Gemini("KEY", post=post)
    assert gemini.translate_batch(["保存する", "送信"]) == ["English 4", "English 2"]
    url, body, key = post.calls[0]
    assert key == "KEY" and url.endswith(f"/models/{t.MODEL}:generateContent")
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["thinkingConfig"] == {"thinkingLevel": "minimal"} and "temperature" not in config
    assert '["保存する", "送信"]' in body["contents"][0]["parts"][0]["text"]
    (line,) = ledger.summary().lines
    assert line.feature == "screen" and line.title == "Screen translator" and line.cost > 0


def test_an_answer_of_the_wrong_length_is_an_error(ledger):
    with pytest.raises(ValueError):
        t.Gemini("KEY", post=FakePost(answer=_answer(["one"]))).translate_batch(["a", "b"])


def _service(backend):
    done = threading.Event()
    svc = t.TranslationService(backend, done.set, workers=1)
    return svc, done


def test_the_glossary_answers_single_ui_words_without_a_call():
    post = FakePost()
    svc, _ = _service(t.Gemini("KEY", post=post))
    svc.request(["保存", "キャンセル"])
    assert svc.get("保存") == "Save" and svc.get("キャンセル") == "Cancel" and post.calls == []
    svc.shutdown()


def test_new_strings_are_translated_once_and_cached(ledger):
    post = FakePost()
    svc, done = _service(t.Gemini("KEY", post=post))
    svc.request(["請求書を送ります", "請求書を送ります"])
    assert done.wait(5)
    svc.pool.shutdown(wait=True)
    assert svc.get("請求書を送ります") == "English 8" and svc.status == "online"
    svc.request(["請求書を送ります"])
    assert len(post.calls) == 1


def test_a_refused_key_stops_asking():
    svc, done = _service(t.Gemini("KEY", post=FakePost(error=t.Refused("Gemini: it didn't accept the API key"))))
    svc.request(["請求書を送ります"])
    assert done.wait(5)
    svc.pool.shutdown(wait=True)
    assert svc.status == "refused" and "API key" in svc.last_error
    svc.request(["別の文です"])
    assert svc.pending == set()


def test_a_network_failure_backs_off():
    svc, done = _service(t.Gemini("KEY", post=FakePost(error=OSError("no network"))))
    svc.request(["請求書を送ります"])
    assert done.wait(5)
    svc.pool.shutdown(wait=True)
    assert svc.status == "offline" and svc._retry_at > 0 and svc.get("請求書を送ります") is None


def test_era_years_bad_translations_and_polish():
    assert t.prep("令和6年4月1日") == "2024年4月1日"
    assert t.prep("令和元年") == "2019年" and t.prep("平成31年") == "2019年"
    assert t.sane("保存", "保存") == "" and t.sane("保存", "保存します") == "" and t.sane("保存", "Save") == "Save"
    assert t.polish("、save the file") == "Save the file"


def test_a_big_screen_goes_in_a_few_requests():
    assert [len(c) for c in t.chunks(["あ" * 10] * 120)] == [50, 50, 20]
    assert [len(c) for c in t.chunks(["あ" * 2000] * 3)] == [1, 1, 1]


def test_a_new_screen_costs_well_under_a_tenth_of_a_cent():
    cost = t.estimate()
    assert 0.0001 < cost < 0.001
    assert t.estimate(0) == 0.0 and t.estimate(model="no-such-model") == 0.0
