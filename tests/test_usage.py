"""The cost tracker (sst.usage): what Gemini calls cost, priced from the tokens and seconds kept, day by day, saved per
profile. The call sites' own tests (test_cloud, test_gateway, test_engine_gemini_stream, test_live_gemini) check that
each call is counted once."""
import ast
import json
import threading
from datetime import date
from pathlib import Path

import pytest

from sst import costs, usage
from sst.usage import Ledger


class Today:
    def __init__(self, day: date):
        self.day = day

    def __call__(self) -> date:
        return self.day


def test_prices_come_from_the_table_and_change_on_their_day():
    transcribe = usage.rates("gemini-3.5-transcribe", "2026-10-23")
    assert (transcribe.audio_in, transcribe.out) == (2.00, 12.00)
    lite = usage.rates("gemini-3.5-flash-lite", "2026-10-23")
    assert (lite.audio_in, lite.text_in, lite.out) == (0.30, 0.30, 2.50)
    assert usage.rates("models/gemini-3.6-flash", "2026-12-31").out == 3.75
    assert usage.rates("gemini-3.6-flash", "2027-01-01").out == 7.50  # Google doubles 3.6 Flash's price
    assert costs.text_price_on("gemini", "gemini-3.8-flash", "2027-03-01").input == 1.50
    assert costs.text_price_on("gemini", "gemini-3.7-flash", "2027-03-01").input == 0.75
    live = usage.rates("gemini-3.5-live-translate-preview", "2026-10-23")
    assert (live.audio_in, live.audio_out) == (3.50, 21.00)
    assert usage.rates("my-own-model", "2026-10-23") is None


def test_the_cost_is_tokens_times_the_price_thinking_billed_as_output():
    counts = {"audio_in": 1_000_000, "text_in": 0, "out": 100_000, "thoughts": 100_000}
    assert usage.cost("gemini-3.5-transcribe", "2026-10-23", counts) == pytest.approx(2.00 + 0.2 * 12.00)
    assert usage.cost("gemini-3.6-flash", "2027-01-02", {"text_in": 1_000_000}) == pytest.approx(1.50)
    # A minute of live translation, 25 tokens a second each way: Google's "approximately $0.0368 per minute"
    minute = {"audio_in": 60 * costs.LIVE_AUDIO, "audio_out": 60 * costs.LIVE_AUDIO}
    assert usage.cost("gemini-3.5-live-translate-preview", "2026-10-23", minute) == pytest.approx(0.0368, abs=1e-4)


def test_answers_are_read_into_counts():
    answer = {"usageMetadata": {"promptTokenCount": 70, "candidatesTokenCount": 20, "thoughtsTokenCount": 30,
                                "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 64},
                                                        {"modality": "TEXT", "tokenCount": 6}]}}
    assert usage.gemini_counts(answer, 2.0, False) == {"audio_in": 64, "text_in": 6, "out": 20, "thoughts": 30,
                                                       "seconds": 2.0}
    assert usage.gemini_counts({}, 2.0, True) == {"audio_in": 50, "seconds": 2.0}  # by its length, 25 tokens a second
    chat = {"usage": {"prompt_tokens": 300, "completion_tokens": 30, "total_tokens": 410}}
    assert usage.chat_counts(chat, "", "") == {"text_in": 300, "out": 30, "thoughts": 80}
    assert usage.chat_counts({"usage": {"prompt_tokens": "x"}}, "", "")["text_in"] == 0
    assert usage.chat_counts({}, "a" * 400, "b" * 40) == {"text_in": 100, "out": 10}


def test_today_and_this_month_by_feature_and_a_new_month_starts_at_nothing():
    today = Today(date(2026, 10, 22))
    ledger = Ledger(None, today=today)
    ledger.record("dictation", "gemini-3.5-transcribe", audio_in=1_000_000, seconds=600)
    today.day = date(2026, 10, 23)
    ledger.record("dictation", "gemini-3.5-transcribe", audio_in=500_000, seconds=300)
    ledger.record("cleanup", "gemini-3.5-flash-lite", text_in=1_000_000)
    ledger.record("cleanup", "gemini-3.5-flash-lite", out=0)
    ledger.record("tests", "someone-elses-model", text_in=5)
    summary = ledger.summary()
    assert summary.today == pytest.approx(1.00 + 0.30) and summary.month == pytest.approx(3.00 + 0.30)
    assert [(line.feature, line.amount) for line in summary.lines] == [
        ("dictation", "15 min"), ("cleanup", "2 calls"), ("tests", "1 call")]
    assert summary.lines[0].cost == pytest.approx(3.00) and summary.unpriced == 1
    today.day = date(2026, 11, 1)
    assert ledger.summary().month == 0 and not ledger.summary().used
    assert ledger.summary(date(2026, 10, 22)).today == pytest.approx(2.00)


def test_saved_a_moment_later_off_the_ui_thread_and_read_back(tmp_path):
    path = tmp_path / "usage.json"
    saved = threading.Event()
    ledger = Ledger(path, today=Today(date(2026, 10, 23)), save_after=0.05)
    real = ledger.flush

    def flush():
        real()
        saved.set()
    ledger.flush = flush
    ledger.record("translate", "gemini-3.5-flash-lite", text_in=10, out=5)
    ledger.record("translate", "gemini-3.5-flash-lite", text_in=10, out=5)
    assert not path.exists()  # not at once: a dictation's few calls are one write
    assert saved.wait(2)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"2026-10-23": {"translate|gemini-3.5-flash-lite": {
        "audio_in": 0, "text_in": 20, "out": 10, "thoughts": 0, "audio_out": 0, "seconds": 0, "calls": 2, "unsent": 0}}}
    again = Ledger(path, today=Today(date(2026, 10, 23)))
    assert again.days == ledger.days and again.summary().lines[0].calls == 2


def test_a_damaged_file_is_kept_aside_and_counting_starts_again(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{not json", encoding="utf-8")
    ledger = Ledger(path, today=Today(date(2026, 10, 23)))
    assert ledger.days == {} and (tmp_path / "usage.damaged.json").exists()
    ledger.record("dictation", "gemini-3.5-transcribe", audio_in=25)
    ledger.flush()
    assert json.loads(path.read_text(encoding="utf-8"))["2026-10-23"]["dictation|gemini-3.5-transcribe"]["calls"] == 1
    # Odd values in a file edited by hand are left out, one by one
    path.write_text(json.dumps({"2026-10-23": {"dictation|m": {"calls": "x", "seconds": -3, "audio_in": 7},
                                               "nokey": {}}, "junk": 5, "2026-10-24": []}), encoding="utf-8")
    assert Ledger(path).days == {"2026-10-23": {"dictation|m": {
        "audio_in": 7, "text_in": 0, "out": 0, "thoughts": 0, "audio_out": 0, "seconds": 0, "calls": 0, "unsent": 0}}}


def test_a_file_from_before_the_cost_saver_loads_with_nothing_unsent(tmp_path):
    path = tmp_path / "usage.json"
    old = {"audio_in": 1500, "text_in": 0, "out": 0, "thoughts": 0, "audio_out": 1500, "seconds": 60, "calls": 0}
    path.write_text(json.dumps({"2026-10-23": {"live_translation|gemini-3.5-live-translate-preview": old}}), encoding="utf-8")
    summary = Ledger(path, today=Today(date(2026, 10, 23))).summary()
    (line,) = summary.lines
    assert (line.unsent, line.saved, summary.saved, line.amount) == (0, 0, 0, "1 min")
    assert line.cost == pytest.approx(0.0368, abs=1e-4)


def test_what_the_cost_saver_didn_t_send_is_not_billed_but_priced_as_saved():
    model = "gemini-3.5-live-translate-preview"
    second_in, second_out = costs.live_rates(model)
    today = Today(date(2026, 10, 22))
    ledger = Ledger(None, today=today)
    # 72 min heard, 42 of them sent (and as much voice back), 30 not sent; and 10 min last month, all unsent
    sent = 42 * 60
    ledger.record("live_translation", model, audio_in=sent * costs.LIVE_AUDIO, audio_out=sent * costs.LIVE_AUDIO,
                  seconds=50 * 60, calls=0, unsent=20 * 60)
    today.day = date(2026, 10, 23)
    ledger.record("live_translation", model, seconds=22 * 60, calls=0, unsent=10 * 60)
    ledger.record("live_translation", model, seconds=0, calls=0, unsent=-5)  # never negative
    ledger.days["2026-09-30"] = {f"live_translation|{model}": dict.fromkeys(usage.COUNTS, 0.0) | {"unsent": 600.0}}
    summary = ledger.summary()
    (line,) = summary.lines
    assert line.cost == pytest.approx(sent * (second_in + second_out)) and summary.month == pytest.approx(line.cost)
    assert summary.today == 0  # today's stretch sent nothing: unsent costs nothing
    assert line.unsent == 30 * 60 and line.saved == pytest.approx(30 * 60 * (second_in + second_out))
    assert summary.saved == pytest.approx(line.saved) and line.saved == pytest.approx(30 * 0.0368, rel=0.01)  # this month only
    assert line.amount == "72 min · 30 min not sent"
    ledger.record("live_translation", model, seconds=0, calls=0, unsent=20)
    assert ledger.summary().lines[0].amount == "72 min · 30 min not sent"
    assert Ledger(None).summary().saved == 0


def test_a_call_is_counted_for_the_feature_its_outermost_caller_names(monkeypatch):
    ledger = Ledger(None)
    monkeypatch.setattr(usage, "_ledger", ledger)
    usage.record("cleanup", "gemini-3.5-flash-lite")
    with usage.feature("transform"):
        usage.record("cleanup", "gemini-3.5-flash-lite")
    usage.within("tests", lambda: usage.within("translate", usage.record, "cleanup", "gemini-3.5-flash-lite"))
    (day,) = ledger.days.values()
    assert {key: entry["calls"] for key, entry in day.items()} == {
        "cleanup|gemini-3.5-flash-lite": 1, "transform|gemini-3.5-flash-lite": 1, "tests|gemini-3.5-flash-lite": 1}


def test_nothing_is_counted_without_a_profile_s_ledger_and_a_failing_listener_never_fails_a_call(monkeypatch):
    usage.record("dictation", "gemini-3.5-transcribe", audio_in=5)  # the CLI, the tests: nowhere to count
    told = []

    def broken():
        told.append(1)
        raise RuntimeError("the window is being rebuilt")
    ledger = Ledger(None, on_change=broken)
    usage.use(ledger)
    usage.record("dictation", "gemini-3.5-transcribe", audio_in=5)
    assert told == [1] and ledger.summary().lines[0].calls == 1


def test_a_new_profile_s_ledger_saves_the_one_before(tmp_path):
    old = Ledger(tmp_path / "a.json", save_after=60)
    usage.use(old)
    usage.record("dictation", "gemini-3.5-transcribe", audio_in=5)
    usage.use(Ledger(tmp_path / "b.json"))
    assert (tmp_path / "a.json").exists() and not (tmp_path / "b.json").exists()


def test_the_tracker_imports_nothing_live_translation_may_not():
    tree = ast.parse(Path(usage.__file__).read_text(encoding="utf-8"))
    names = [a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names]
    names += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert {name for name in names if name.startswith("sst")} == {"sst"}  # from sst import costs


def test_meeting_minutes_have_a_line_of_their_own_counted_in_minutes_of_meeting():
    ledger = Ledger(None, today=Today(date(2026, 10, 27)))
    ledger.record("meeting", "gemini-3.6-flash", audio_in=230_400, seconds=3600)
    ledger.record("meeting", "gemini-3.6-flash", text_in=40_000, out=2_000)  # the minutes: no audio of their own
    (line,) = ledger.summary().lines
    assert (line.title, line.amount, line.calls) == ("Meeting minutes", "60 min", 2)
    assert line.cost == pytest.approx((270_400 * 0.75 + 2_000 * 3.75) / 1e6)
