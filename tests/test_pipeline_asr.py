"""Speech recognition in the voice pipeline (sst.pipeline.asr): ordering, retries, validation, fallback, session
isolation and bounded concurrency. The backends are fakes driven by threading.Event and the backoff sleep is injected,
so the tests are exact and fast and need no network, model or microphone."""
import http.client
import json
import threading
import time
import urllib.error

import numpy as np
import pytest

from sst.pipeline.asr import (
    ASRError,
    ASRScheduler,
    ASRValidator,
    EngineBackend,
    OrderedResultBuffer,
    SessionCancelled,
    SessionFailed,
    classify,
)
from sst.pipeline.contracts import ASRConfig, AudioChunk, ChunkResult, ChunkStatus, ErrorKind, RawTranscript, WordInfo

RATE = 16_000
SESSION = "sess_test_001"
NATO = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
        "romeo sierra tango uniform victor whiskey xray yankee zulu").split()
PARAGRAPH = ("I wanted to follow up on the budget review from yesterday. The numbers for the third quarter look better "
             "than expected, but we still need to check the travel costs before Friday, so please send me the receipts.")


def make_chunk(seq=1, session=SESSION, seconds=2.0, speech=None, start=0.0, overlap=0.0, audio=True):
    begin, n = int(start * RATE), int(seconds * RATE)
    samples = np.full(n, 0.01, dtype=np.float32) if audio else None
    return AudioChunk(session, seq, samples, RATE, begin, begin + n, begin + int(overlap * RATE),
                      seconds * 0.8 if speech is None else speech)


class HTTPFailure(Exception):
    """Like sst.engines.cloud.CloudError: the provider answered with an HTTP status."""

    def __init__(self, status, retry_after=None):
        super().__init__(f"HTTP {status}")
        self.status = status
        if retry_after is not None:
            self.retry_after = retry_after


class Scripted:
    """Answers in order, or per sequence number: text, a RawTranscript, an exception to raise, or a callable(chunk)
    whose return value goes back as it is; then `default`."""

    def __init__(self, *answers, name="fake", default="hello there", by_sequence=None):
        self.name, self.default = name, default
        self.answers, self.by_sequence = list(answers), {k: list(v) for k, v in (by_sequence or {}).items()}
        self.calls = []
        self._lock = threading.Lock()

    def transcribe(self, chunk):
        with self._lock:
            self.calls.append((chunk.session_id, chunk.sequence, chunk.status))
            script = self.by_sequence.get(chunk.sequence) or self.answers
            answer = script.pop(0) if script else self.default
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return answer(chunk)
        return answer if isinstance(answer, RawTranscript) else RawTranscript(answer, backend=self.name)


class Gated:
    """Each chunk waits for its own gate, so the test decides the order in which chunks finish."""
    name = "gated"

    def __init__(self):
        self.started, self._gates, self._lock = [], {}, threading.Lock()

    def gate(self, session, seq):
        with self._lock:
            return self._gates.setdefault((session, seq), threading.Event())

    def release(self, session, seq):
        self.gate(session, seq).set()

    def transcribe(self, chunk):
        with self._lock:
            self.started.append((chunk.session_id, chunk.sequence))
        self.gate(chunk.session_id, chunk.sequence).wait(5)
        return RawTranscript(f"words of part {chunk.sequence}", backend=self.name)


class Counting:
    """Counts the transcriptions running at the same time; all of them wait until `go` is set."""
    name = "counting"

    def __init__(self):
        self.active = self.peak = 0
        self.lock, self.entered, self.go = threading.Lock(), threading.Semaphore(0), threading.Event()

    def transcribe(self, chunk):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        self.entered.release()
        self.go.wait(5)
        with self.lock:
            self.active -= 1
        return RawTranscript("counted words here", backend=self.name)


@pytest.fixture
def make():
    made = []

    def make(backend, fallback=None, rng=lambda: 0.5, **config):
        slept, events = [], []
        scheduler = ASRScheduler(backend, ASRConfig(**config), fallback, on_event=lambda name, data: events.append((name, data)),
                                 sleep=slept.append, rng=rng)
        made.append(scheduler)
        return scheduler, slept, events
    yield make
    for scheduler in made:
        scheduler.close(wait=True)


def transcribe(scheduler, *chunks, keep_audio=False):
    session = scheduler.open_session(chunks[0].session_id, keep_audio=keep_audio)
    for chunk in chunks:
        session.submit(chunk)
    session.finish(len(chunks))
    return session.wait(5)


def until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.002)


def names(events):
    return [name for name, _ in events]


# ---------------------------------------------------------------- ordering (plan §24, §32, §98)

def test_results_come_back_in_sequence_order_whatever_finishes_first(make):
    backend = Gated()
    scheduler, _, events = make(backend, max_in_flight=3)
    delivered = []
    session = scheduler.open_session(SESSION, on_result=lambda r: delivered.append(r.sequence))
    for seq in (1, 2, 3):
        session.submit(make_chunk(seq, start=(seq - 1) * 2.0))
    session.finish(3)
    backend.release(SESSION, 2)  # chunk 1 slow, 2 fast, 3 medium
    until(lambda: session.progress()[0] == 1)
    backend.release(SESSION, 3)
    until(lambda: session.progress()[0] == 2)
    assert delivered == []  # 2 and 3 wait for 1
    backend.release(SESSION, 1)
    results = session.wait(5)
    assert [d["sequence"] for name, d in events if name == "asr_succeeded"] == [2, 3, 1]
    assert delivered == [1, 2, 3]
    assert [r.sequence for r in results] == [1, 2, 3]
    assert [r.text for r in results] == ["words of part 1", "words of part 2", "words of part 3"]
    assert session.progress() == (3, 3)


def result(seq, session=SESSION):
    return ChunkResult(session, seq, 0.0, 1.0, 0.0, f"text {seq}")


def test_ordered_buffer_releases_in_order_and_ignores_repeats_and_other_sessions():
    buffer = OrderedResultBuffer(SESSION)
    assert buffer.push(result(3)) == [] and buffer.push(result(2)) == []
    assert buffer.pending() == 2 and buffer.next_expected == 1
    assert [r.sequence for r in buffer.push(result(1))] == [1, 2, 3]
    assert buffer.next_expected == 4 and buffer.pending() == 0
    assert buffer.push(result(2)) == []  # already released
    assert buffer.push(result(5)) == [] and buffer.push(result(5)) == [] and buffer.pending() == 1
    assert buffer.push(result(4, session="sess_other")) == []
    assert [r.sequence for r in buffer.push(result(4))] == [4, 5]


# ---------------------------------------------------------------- retries (plan §29-30, §99)

def test_two_timeouts_then_success_is_accepted_after_two_retries(make):
    backend = Scripted(TimeoutError(), TimeoutError(), "the meeting is at noon")
    scheduler, slept, events = make(backend)
    [r] = transcribe(scheduler, make_chunk())
    assert r.status is ChunkStatus.ACCEPTED and r.text == "the meeting is at noon"
    assert r.retries == 2 and r.diagnostics["attempts"] == 3 and len(r.diagnostics["durations_ms"]) == 3
    assert r.diagnostics["errors"] == ["transient: fake: no answer in time"] * 2
    assert slept == pytest.approx([0.25, 0.75])
    assert [(d["attempt"], d["delay_ms"]) for name, d in events if name == "asr_retrying"] == [(2, 250), (3, 750)]
    assert names(events) == ["asr_started", "asr_retrying", "asr_retrying", "asr_succeeded"]


@pytest.mark.parametrize("draw", [0.0, 0.37, 1.0])
def test_backoff_jitter_stays_within_its_range(make, draw):
    scheduler, slept, _ = make(Scripted(TimeoutError(), ConnectionResetError(), "fine"), rng=lambda: draw)
    transcribe(scheduler, make_chunk())
    for delay, base in zip(slept, (0.25, 0.75), strict=True):
        assert delay == pytest.approx(base * (1 + 0.2 * (2 * draw - 1)))
        assert base * 0.8 - 1e-9 <= delay <= base * 1.2 + 1e-9


def test_three_failures_fail_the_chunk_and_the_session_fails_closed(make):
    backend = Scripted(by_sequence={2: [TimeoutError(), TimeoutError(), TimeoutError(), "never asked"]})
    scheduler, slept, events = make(backend)
    delivered = []
    session = scheduler.open_session(SESSION, on_result=delivered.append)
    chunks = [make_chunk(seq, start=(seq - 1) * 2.0) for seq in (1, 2, 3)]
    for chunk in chunks:
        session.submit(chunk)
    session.finish(3)
    with pytest.raises(SessionFailed) as caught:
        session.wait(5)
    failure = caught.value
    assert failure.session_id == SESSION
    assert [r.sequence for r in failure.results] == [1, 3]  # never handed out as if the text were complete
    [failed] = failure.failed
    assert failed.sequence == 2 and failed.status is ChunkStatus.FAILED and failed.retries == 2
    assert "no answer in time" in failed.error and failed.text == ""
    assert chunks[1].status is ChunkStatus.FAILED and chunks[1].audio is not None  # kept for a retry
    assert len(slept) == 2 and "asr_failed" in names(events)
    # Progressive text gets the failure and nothing after it: it must not jump the gap.
    assert [(r.sequence, r.status) for r in delivered] == [(1, ChunkStatus.ACCEPTED), (2, ChunkStatus.FAILED)]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_a_permanent_error_is_not_retried(make, status):
    backend = Scripted(HTTPFailure(status), "never asked")
    scheduler, slept, _ = make(backend)
    with pytest.raises(SessionFailed) as caught:
        transcribe(scheduler, make_chunk())
    assert len(backend.calls) == 1 and slept == []
    assert caught.value.failed[0].retries == 0 and f"HTTP {status}" in caught.value.failed[0].error


@pytest.mark.parametrize("status", [0, 408, 429, 500, 503])
def test_rate_limits_and_server_errors_are_retried(make, status):
    backend = Scripted(HTTPFailure(status), "back again")
    scheduler, slept, _ = make(backend)
    [r] = transcribe(scheduler, make_chunk())
    assert r.text == "back again" and r.retries == 1 and len(backend.calls) == 2 and len(slept) == 1


def test_a_servers_retry_after_is_honoured_up_to_a_limit(make):
    backend = Scripted(HTTPFailure(429, retry_after=2.0), HTTPFailure(429, retry_after=60), "ok then")
    scheduler, slept, _ = make(backend)
    transcribe(scheduler, make_chunk())
    assert slept == [2.0, 5.0]


def test_a_backend_that_flags_its_own_answer_gets_one_immediate_retry(make):
    backend = Scripted(ASRError("the model answered with a refusal", ErrorKind.SUSPICIOUS), "the real words")
    scheduler, slept, _ = make(backend)
    [r] = transcribe(scheduler, make_chunk())
    assert r.text == "the real words" and r.retries == 1 and slept == []


@pytest.mark.parametrize("answer", [42, None, RawTranscript(None)])
def test_an_answer_that_is_not_a_transcript_fails_at_once(make, answer):
    backend = Scripted(lambda chunk: answer)
    scheduler, _, _ = make(backend)
    with pytest.raises(SessionFailed) as caught:
        transcribe(scheduler, make_chunk())
    assert len(backend.calls) == 1 and "not a transcript" in caught.value.failed[0].error


def test_a_backend_may_answer_with_plain_text(make):
    scheduler, _, _ = make(Scripted(lambda chunk: "just the text"))
    [r] = transcribe(scheduler, make_chunk())
    assert r.text == "just the text" and r.backend == "fake"


# ---------------------------------------------------------------- validation in the scheduler (plan §27-28, §90)

def test_empty_text_for_real_speech_gets_one_verification_retry_then_fails(make):
    backend = Scripted("", "  ", "never asked")
    scheduler, slept, events = make(backend)
    chunk = make_chunk(seconds=5.0, speech=4.2)
    seen = []
    scheduler.on_event = lambda name, data: seen.append((name, data.get("kind"), chunk.status))
    with pytest.raises(SessionFailed) as caught:
        transcribe(scheduler, chunk)
    assert len(backend.calls) == 2 and slept == []
    assert "nothing recognised in 4.2 s of speech" in caught.value.failed[0].error
    assert ("asr_retrying", "suspicious", ChunkStatus.REJECTED) in seen
    assert chunk.status is ChunkStatus.FAILED and chunk.audio is not None


def test_a_verification_retry_that_answers_properly_is_used(make):
    backend = Scripted("", "it was a long sentence after all")
    scheduler, _, _ = make(backend)
    [r] = transcribe(scheduler, make_chunk(seconds=4.0, speech=3.5))
    assert r.text == "it was a long sentence after all" and r.retries == 1
    assert "suspicious" not in r.diagnostics and r.diagnostics["suspicion"] == 0


def test_a_decoding_loop_fails_the_chunk(make):
    loop = " ".join(["hello"] * 12)
    backend = Scripted(loop, loop)
    scheduler, _, _ = make(backend)
    with pytest.raises(SessionFailed) as caught:
        transcribe(scheduler, make_chunk(seconds=4.0, speech=3.0))
    assert "repeated 12 times" in caught.value.failed[0].error and len(backend.calls) == 2


def test_tolerable_suspicion_keeps_the_less_suspicious_answer_and_flags_it(make):
    many, fewer = " ".join(NATO[:24]), " ".join(NATO[:18])  # both too fast for 2 s of speech, neither impossible
    backend = Scripted(many, fewer)
    scheduler, _, events = make(backend)
    [r] = transcribe(scheduler, make_chunk(seconds=2.0, speech=2.0))
    assert r.status is ChunkStatus.ACCEPTED and r.text == fewer
    assert r.diagnostics["suspicious"] is True and r.diagnostics["validation"]["excessive_rate"] > 0
    assert [d["suspicious"] for name, d in events if name == "asr_succeeded"] == [True]


def test_a_better_first_answer_is_kept_over_a_worse_verification(make):
    first, worse = " ".join(NATO[:18]), " ".join(NATO[:24])
    scheduler, _, _ = make(Scripted(first, worse))
    [r] = transcribe(scheduler, make_chunk(seconds=2.0, speech=2.0))
    assert r.text == first and r.diagnostics["suspicious"] is True


# ---------------------------------------------------------------- fallback

def test_the_fallback_stands_in_when_the_primary_backend_fails(make):
    primary = Scripted(HTTPFailure(503), HTTPFailure(503), HTTPFailure(503), name="gemini")
    primary.title = "Google Gemini"
    fallback = Scripted("typed by the local model", name="parakeet")
    fallback.title = "Parakeet"
    scheduler, _, events = make(primary, fallback)
    [r] = transcribe(scheduler, make_chunk())
    assert r.status is ChunkStatus.ACCEPTED and r.text == "typed by the local model" and r.backend == "parakeet"
    assert r.note == "Parakeet stood in: Google Gemini: HTTP 503"
    assert r.retries == 3 and len(primary.calls) == 3 and len(fallback.calls) == 1
    [(_, data)] = [(n, d) for n, d in events if n == "asr_fallback"]
    assert data["backend"] == "parakeet" and data["reason"] == "Google Gemini: HTTP 503"


def test_the_fallback_stands_in_at_once_after_a_permanent_error(make):
    primary, fallback = Scripted(HTTPFailure(401), name="openai"), Scripted("local words", name="local")
    scheduler, slept, _ = make(primary, fallback)
    [r] = transcribe(scheduler, make_chunk())
    assert len(primary.calls) == 1 and slept == [] and r.note == "local stood in: openai: HTTP 401"


def test_the_fallback_stands_in_for_an_unusable_answer(make):
    primary, fallback = Scripted("", ""), Scripted("what was really said", name="local")
    scheduler, _, _ = make(primary, fallback)
    [r] = transcribe(scheduler, make_chunk(seconds=3.0, speech=2.5))
    assert r.text == "what was really said" and r.note.startswith("local stood in: fake: unusable result (nothing")


def test_when_the_fallback_fails_too_the_chunk_fails_with_both_reasons(make):
    primary = Scripted(HTTPFailure(400))
    fallback = Scripted(RuntimeError("model not downloaded"), name="local")
    scheduler, _, _ = make(primary, fallback)
    with pytest.raises(SessionFailed) as caught:
        transcribe(scheduler, make_chunk())
    error = caught.value.failed[0].error
    assert "fake: HTTP 400" in error and "local: model not downloaded" in error and len(fallback.calls) == 1


def test_an_unusable_fallback_answer_fails_the_chunk(make):
    loop = " ".join(["again"] * 10)
    scheduler, _, _ = make(Scripted("", ""), Scripted(loop, name="local"))
    with pytest.raises(SessionFailed) as caught:
        transcribe(scheduler, make_chunk(seconds=3.0, speech=2.5))
    assert "local: unusable result" in caught.value.failed[0].error


def test_a_chunk_without_audio_fails_without_asking_any_backend(make):
    primary, fallback = Scripted(), Scripted(name="local")
    scheduler, _, events = make(primary, fallback)
    with pytest.raises(SessionFailed) as caught:
        transcribe(scheduler, make_chunk(audio=False))
    assert primary.calls == [] and fallback.calls == []
    assert "audio was already released" in caught.value.failed[0].error and "asr_failed" in names(events)
    empty = make_chunk(seconds=0.0, session="sess_empty")
    with pytest.raises(SessionFailed, match="audio is empty"):
        transcribe(scheduler, empty)


# ---------------------------------------------------------------- sessions (plan §75-77, §89)

def test_a_cancelled_sessions_late_result_is_dropped(make):
    backend = Gated()
    scheduler, _, _ = make(backend)
    seen_a, seen_b = [], []
    a = scheduler.open_session("sess_a", on_result=seen_a.append)
    b = scheduler.open_session("sess_b", on_result=seen_b.append)
    a.submit(make_chunk(1, session="sess_a"))
    until(lambda: ("sess_a", 1) in backend.started)
    a.cancel()
    b.submit(make_chunk(1, session="sess_b"))
    b.finish(1)
    backend.release("sess_b", 1)
    [rb] = b.wait(5)
    backend.release("sess_a", 1)  # the late answer of the cancelled session
    scheduler.close(wait=True)
    assert seen_a == [] and seen_b == [rb] and rb.session_id == "sess_b"
    assert a.progress() == (0, 1) and a.cancelled
    with pytest.raises(SessionCancelled):
        a.wait(1)


def test_cancel_skips_chunks_still_waiting_for_a_worker(make):
    backend = Gated()
    scheduler, _, _ = make(backend, max_in_flight=1)
    session = scheduler.open_session(SESSION)
    chunks = [make_chunk(seq) for seq in (1, 2, 3)]
    for chunk in chunks:
        session.submit(chunk)
    until(lambda: backend.started == [(SESSION, 1)])
    assert chunks[1].status is ChunkStatus.QUEUED
    session.cancel()
    backend.release(SESSION, 1)
    scheduler.close(wait=True)
    assert backend.started == [(SESSION, 1)]


def test_two_sessions_at_once_keep_their_own_results(make):
    backend = Scripted(default=lambda chunk: f"{chunk.session_id[-1]} said part {chunk.sequence}")
    scheduler, _, _ = make(backend, max_in_flight=3)
    a, b = scheduler.open_session("sess_a"), scheduler.open_session("sess_b")
    for seq in (1, 2, 3):
        a.submit(make_chunk(seq, session="sess_a"))
        if seq < 3:
            b.submit(make_chunk(seq, session="sess_b"))
    a.finish(3)
    b.finish(2)
    assert [(r.session_id, r.text) for r in a.wait(5)] == [("sess_a", f"a said part {n}") for n in (1, 2, 3)]
    assert [(r.session_id, r.text) for r in b.wait(5)] == [("sess_b", f"b said part {n}") for n in (1, 2)]


def test_a_chunk_of_another_session_is_refused(make):
    scheduler, _, _ = make(Scripted())
    session = scheduler.open_session(SESSION)
    with pytest.raises(ValueError):
        session.submit(make_chunk(session="sess_other"))
    with pytest.raises(ValueError):
        session.submit(make_chunk(0))
    with pytest.raises(ValueError):
        scheduler.open_session(SESSION)  # still open


def test_a_repeated_chunk_is_transcribed_once(make):
    backend = Scripted()
    scheduler, _, _ = make(backend)
    session = scheduler.open_session(SESSION)
    chunk = make_chunk()
    session.submit(chunk)
    session.submit(chunk)
    session.finish(1)
    assert len(session.wait(5)) == 1 and len(backend.calls) == 1


def test_finish_must_match_what_was_submitted(make):
    scheduler, _, _ = make(Scripted())
    session = scheduler.open_session(SESSION)
    session.submit(make_chunk(2))
    with pytest.raises(ValueError):
        session.finish(1)
    session.finish(2)
    with pytest.raises(ValueError):
        session.finish(3)
    with pytest.raises(ValueError):
        session.submit(make_chunk(3))
    session.submit(make_chunk(1))  # an earlier chunk may still come after finish()
    assert [r.sequence for r in session.wait(5)] == [1, 2]


def test_a_session_without_chunks_is_done_at_once(make):
    scheduler, _, _ = make(Scripted())
    session = scheduler.open_session(SESSION)
    session.finish(0)
    assert session.wait(1) == [] and session.progress() == (0, 0)


def test_wait_times_out_while_chunks_are_still_out(make):
    backend = Gated()
    scheduler, _, _ = make(backend)
    session = scheduler.open_session(SESSION)
    session.submit(make_chunk())
    assert session.progress() == (0, 1)
    session.finish(1)
    with pytest.raises(TimeoutError, match="0 of 1"):
        session.wait(0.05)
    backend.release(SESSION, 1)
    assert len(session.wait(5)) == 1


def test_close_cancels_open_sessions(make):
    backend = Gated()
    scheduler, _, _ = make(backend)
    session = scheduler.open_session(SESSION)
    session.submit(make_chunk())
    scheduler.close()
    backend.release(SESSION, 1)
    with pytest.raises(SessionCancelled):
        session.wait(1)
    with pytest.raises(RuntimeError):
        scheduler.open_session("sess_new")


def test_a_failing_result_handler_or_event_handler_does_not_stop_the_session(make):
    scheduler, _, _ = make(Scripted())
    scheduler.on_event = lambda name, data: 1 / 0

    def explode(result):
        raise RuntimeError("the handler broke")
    session = scheduler.open_session(SESSION, on_result=explode)
    for seq in (1, 2):
        session.submit(make_chunk(seq))
    session.finish(2)
    assert [r.sequence for r in session.wait(5)] == [1, 2]


def test_an_internal_error_fails_the_chunk_instead_of_hanging(make):
    scheduler, _, _ = make(Scripted())

    def broken(chunk, raw):
        raise RuntimeError("a bug")
    scheduler.validator.validate = broken
    with pytest.raises(SessionFailed, match="internal error"):
        transcribe(scheduler, make_chunk())


# ---------------------------------------------------------------- concurrency, time, audio, status (plan §23, §26, §31)

def test_never_more_than_max_in_flight_transcriptions_at_once(make):
    backend = Counting()
    scheduler, _, _ = make(backend, max_in_flight=2)
    sessions = [scheduler.open_session(name) for name in ("sess_a", "sess_b")]
    for session in sessions:
        for seq in (1, 2, 3):
            session.submit(make_chunk(seq, session=session.session_id))
        session.finish(3)
    assert backend.entered.acquire(timeout=5) and backend.entered.acquire(timeout=5)
    assert not backend.entered.acquire(timeout=0.05)  # a third waits for a free worker
    backend.go.set()
    assert all(len(session.wait(5)) == 3 for session in sessions)
    assert backend.peak == 2


def test_word_times_are_converted_to_session_time(make):
    words = [WordInfo("pizza", 0.72, 1.11, 0.9), WordInfo("tonight", 1.2, 1.6)]
    raw = RawTranscript("pizza tonight", words, language="en", backend="gemini-test", diagnostics={"model": "x"})
    scheduler, _, _ = make(Scripted(raw))
    [r] = transcribe(scheduler, make_chunk(start=19.0, overlap=1.0, seconds=3.0))
    assert (r.start, r.end, r.overlap_end) == (19.0, 22.0, 20.0)
    assert [(w.text, w.start, w.end, w.confidence) for w in r.words] == [
        ("pizza", pytest.approx(19.72), pytest.approx(20.11), 0.9), ("tonight", pytest.approx(20.2), pytest.approx(20.6), None)]
    assert words[0].start == 0.72  # the backend's answer is left as it was
    assert r.language == "en" and r.backend == "gemini-test" and r.diagnostics["backend_diagnostics"] == {"model": "x"}


def test_word_times_that_are_not_numbers_are_dropped_but_the_text_kept(make):
    raw = RawTranscript("hello world", [WordInfo("hello", 0.1, 0.4), WordInfo("world", None, float("nan"))])
    scheduler, _, _ = make(Scripted(raw))
    [r] = transcribe(scheduler, make_chunk())
    assert r.text == "hello world" and r.words == [] and "words_dropped" in r.diagnostics


def test_words_without_text_give_the_text(make):
    raw = RawTranscript("", [WordInfo("hello", 0.1, 0.4), WordInfo("again", 0.5, 0.9)])
    scheduler, _, _ = make(Scripted(raw))
    [r] = transcribe(scheduler, make_chunk())
    assert r.text == "hello again" and len(r.words) == 2


def test_audio_is_released_once_accepted_unless_kept(make):
    scheduler, _, _ = make(Scripted())
    chunk, kept = make_chunk(), make_chunk(session="sess_keep")
    transcribe(scheduler, chunk)
    transcribe(scheduler, kept, keep_audio=True)
    assert chunk.status is ChunkStatus.ACCEPTED and chunk.audio is None
    assert kept.status is ChunkStatus.ACCEPTED and kept.audio is not None


def test_a_chunk_goes_through_its_states(make):
    backend = Scripted()
    scheduler, _, events = make(backend)
    chunk = make_chunk()
    [r] = transcribe(scheduler, chunk)
    assert backend.calls == [(SESSION, 1, ChunkStatus.SENT)] and chunk.status is ChunkStatus.ACCEPTED
    assert r.diagnostics["backend"] == "fake" and r.diagnostics["attempts"] == 1 and r.retries == 0
    assert set(r.diagnostics["validation"]) >= {"empty", "repetition", "timestamps"}
    started, succeeded = (d for name, d in events if name in ("asr_started", "asr_succeeded"))
    assert started == {"session_id": SESSION, "sequence": 1, "chunk_id": f"{SESSION}#001", "backend": "fake"}
    assert succeeded["attempts"] == 1 and succeeded["suspicious"] is False and succeeded["note"] == ""


# ---------------------------------------------------------------- EngineBackend

class PlainEngine:
    name, title = "plain", "Plain engine"

    def transcribe(self, audio, rate):
        self.got = (audio.size, rate)
        return "plain text"


class ChunkEngine:
    name = "cloudy"

    def transcribe(self, audio, rate):
        raise AssertionError("transcribe_chunk should be used")

    def transcribe_chunk(self, audio, rate):
        return RawTranscript("hi", [WordInfo("hi", 0.1, 0.3)])


def test_engine_backend_wraps_a_plain_engine():
    engine = PlainEngine()
    backend = EngineBackend(engine)
    raw = backend.transcribe(make_chunk(seconds=1.0))
    assert (raw.text, raw.words, raw.backend) == ("plain text", [], "plain")
    assert engine.got == (RATE, RATE) and backend.name == "plain" and backend.title == "Plain engine"


def test_engine_backend_prefers_transcribe_chunk():
    backend = EngineBackend(ChunkEngine())
    raw = backend.transcribe(make_chunk())
    assert raw.text == "hi" and raw.words[0].start == 0.1 and raw.backend == "cloudy" and backend.title == "cloudy"


def test_engine_backend_refuses_a_chunk_without_audio():
    with pytest.raises(ASRError) as caught:
        EngineBackend(PlainEngine()).transcribe(make_chunk(audio=False))
    assert classify(caught.value) is ErrorKind.PERMANENT


def test_engine_errors_reach_the_scheduler_and_are_retried(make):
    class Flaky(ChunkEngine):
        calls = 0

        def transcribe_chunk(self, audio, rate):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise HTTPFailure(503)
            return super().transcribe_chunk(audio, rate)
    scheduler, slept, _ = make(EngineBackend(Flaky()))
    [r] = transcribe(scheduler, make_chunk())
    assert r.text == "hi" and r.retries == 1 and len(slept) == 1 and r.backend == "cloudy"


# ---------------------------------------------------------------- the validator (plan §27-28, §91)

VALIDATOR = ASRValidator(ASRConfig())


def check(text, seconds=2.0, speech=None, words=()):
    return VALIDATOR.validate(make_chunk(seconds=seconds, speech=speech), RawTranscript(text, list(words)))


@pytest.mark.parametrize("text, seconds, speech", [
    ("yes", 0.8, 0.4),
    ("OK, thanks.", 1.0, 0.6),
    ("Absolutely.", 1.5, 0.9),
    ("no", 0.5, 0.2),
    (PARAGRAPH, 14.0, 11.0),
    ("No, no, no, no.", 2.0, 1.5),  # people do say that
    ("今日は会議の資料を準備してから午後三時に部長と打ち合わせをする予定です", 9.0, 8.0),
    ("", 1.0, 0.3),  # a cough: too little speech to expect words
])
def test_legitimate_answers_pass(text, seconds, speech):
    verdict = check(text, seconds, speech)
    assert verdict.accepted and verdict.suspicion < 1.0 and not VALIDATOR.fatal(verdict)


def test_one_word_for_long_speech_is_only_a_weak_signal():
    verdict = check("yes", 12.0, 10.0)
    assert verdict.accepted and verdict.signals["tiny"] > 0


def test_three_seconds_of_speech_cannot_be_800_words():
    verdict = check(" ".join(f"{NATO[i % 26]}{i}" for i in range(800)), 3.0, 3.0)
    assert not verdict.accepted and VALIDATOR.fatal(verdict)
    assert verdict.signals["excessive_rate"] == 10.0 and verdict.signals["extreme_rate"] == 1.0


def test_a_quiet_voice_the_vad_half_missed_is_suspicious_but_not_fatal():
    verdict = check(" ".join(NATO[:24]), 12.0, 2.0)
    assert not verdict.accepted and not VALIDATOR.fatal(verdict) and verdict.signals["extreme_rate"] == 0


@pytest.mark.parametrize("text", ["", "...", " - "])
def test_nothing_recognised_for_real_speech_is_fatal(text):
    verdict = check(text, 5.0, 4.2)
    assert not verdict.accepted and VALIDATOR.fatal(verdict) and "nothing recognised" in verdict.reason


def test_a_sentence_out_of_silence_is_a_hallucination():
    verdict = check("Thank you for watching.", 2.0, 0.1)
    assert not verdict.accepted and VALIDATOR.fatal(verdict) and verdict.signals["silence_hallucination"] >= 1


@pytest.mark.parametrize("text", [" ".join(["hello"] * 8), "I think " * 4, "see you there " * 5])
def test_a_repeated_word_or_phrase_is_fatal(text):
    verdict = check(text, 6.0, 5.0)
    assert not verdict.accepted and VALIDATOR.fatal(verdict) and "repeated" in verdict.reason


def test_long_text_that_compresses_like_a_loop_is_fatal():
    verdict = check("the cat sat on the mat and looked at the dog " * 6, 20.0, 18.0)
    assert not verdict.accepted and VALIDATOR.fatal(verdict) and "compresses" in verdict.reason


def test_natural_long_text_does_not_look_repetitive():
    assert check(PARAGRAPH, 14.0, 11.0).signals["repetition"] == 0


@pytest.mark.parametrize("times", [
    [(0.5, 0.2)],  # starts after it ends
    [(0.0, 0.5), (0.1, 0.6)],  # goes back more than 0.2 s
    [(0.0, 0.5), (0.6, 3.0)],  # ends well after the 2 s chunk
    [(-1.0, 0.2)],
])
def test_broken_word_times_are_a_medium_signal(times):
    words = [WordInfo(f"w{i}", start, end) for i, (start, end) in enumerate(times)]
    verdict = check(" ".join(w.text for w in words), 2.0, 1.5, words)
    assert 0 < verdict.signals["timestamps"] <= 1 and verdict.accepted  # never enough alone


def test_good_word_times_raise_nothing():
    words = [WordInfo("one", 0.1, 0.4), WordInfo("two", 0.35, 0.7), WordInfo("three", 0.8, 1.2)]
    assert check("one two three", 2.0, 1.5, words).signals["timestamps"] == 0


def test_weak_signals_together_make_a_result_suspicious():
    verdict = check("yes", 20.0, 20.0, [WordInfo("yes", 5.0, 3.0)])  # one word in 20 s, with a broken time
    assert not verdict.accepted and not VALIDATOR.fatal(verdict)


def test_without_a_speech_measurement_only_the_length_is_judged():
    assert check("", 3.0, float("nan")).accepted
    assert check(" ".join(f"{NATO[i % 26]}{i}" for i in range(800)), 3.0, -1.0).signals["extreme_rate"] == 1.0


# ---------------------------------------------------------------- classify (plan §30)

@pytest.mark.parametrize("exc, kind", [
    (ASRError("refusal", ErrorKind.SUSPICIOUS), ErrorKind.SUSPICIOUS),
    (ASRError("bad audio", ErrorKind.PERMANENT, 503), ErrorKind.PERMANENT),  # its own kind wins
    (HTTPFailure(400), ErrorKind.PERMANENT),
    (HTTPFailure(401), ErrorKind.PERMANENT),
    (HTTPFailure(404), ErrorKind.PERMANENT),
    (HTTPFailure(422), ErrorKind.PERMANENT),
    (HTTPFailure(0), ErrorKind.TRANSIENT),
    (HTTPFailure(408), ErrorKind.TRANSIENT),
    (HTTPFailure(425), ErrorKind.TRANSIENT),
    (HTTPFailure(429), ErrorKind.TRANSIENT),
    (HTTPFailure(500), ErrorKind.TRANSIENT),
    (HTTPFailure(504), ErrorKind.TRANSIENT),
    (urllib.error.HTTPError("https://example.invalid", 403, "Forbidden", None, None), ErrorKind.PERMANENT),
    (urllib.error.HTTPError("https://example.invalid", 502, "Bad Gateway", None, None), ErrorKind.TRANSIENT),
    (TimeoutError(), ErrorKind.TRANSIENT),
    (ConnectionResetError(), ErrorKind.TRANSIENT),
    (OSError("network is unreachable"), ErrorKind.TRANSIENT),
    (http.client.RemoteDisconnected("closed"), ErrorKind.TRANSIENT),
    (http.client.IncompleteRead(b""), ErrorKind.TRANSIENT),
    (json.JSONDecodeError("garbled", "<html>", 0), ErrorKind.TRANSIENT),
    (ValueError("unsupported audio"), ErrorKind.PERMANENT),
    (TypeError("bad config"), ErrorKind.PERMANENT),
    (NotImplementedError(), ErrorKind.PERMANENT),
    (RuntimeError("who knows"), ErrorKind.TRANSIENT),
    (KeyboardInterrupt(), ErrorKind.PERMANENT),
])
def test_classify(exc, kind):
    assert classify(exc) is kind


def test_classify_takes_the_backends_word_that_an_error_is_final():
    no_credit, offline = HTTPFailure(429), HTTPFailure(0)  # a used-up quota; "couldn't be reached a moment ago"
    no_credit.permanent = offline.permanent = True
    unsure = HTTPFailure(429)
    unsure.permanent = "yes"  # only True counts
    assert classify(no_credit) is classify(offline) is ErrorKind.PERMANENT
    assert classify(unsure) is ErrorKind.TRANSIENT


def test_classify_reads_status_code_and_ignores_statuses_that_are_not_http():
    requests_like = RuntimeError("Service Unavailable")
    requests_like.status_code = 503
    odd = ValueError("x")
    odd.status = True  # a flag, not an HTTP status
    assert classify(requests_like) is ErrorKind.TRANSIENT and classify(odd) is ErrorKind.PERMANENT
