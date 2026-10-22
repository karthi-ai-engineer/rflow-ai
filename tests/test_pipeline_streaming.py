"""A streaming speech engine in a voice pipeline session (sst.pipeline.streaming): the stream's text when it has it, the
held chunks to the batch model whenever it hasn't, so no word is lost (fake streams and engines, no network)."""
import threading
import time

import numpy as np

from sst.dictate import Dictation
from sst.engines import gemini_stream
from sst.engines.gemini_stream import GeminiEngine
from sst.pipeline import streaming
from sst.pipeline.contracts import Provenance
from sst.pipeline.session import Stages
from sst.pipeline.streaming import StreamedSession
from tests.test_engine_gemini_stream import KEY, LIVE, FakeLive, final, interim
from tests.test_pipeline_session import RATE, FakeLLM, LiveRecorder, ScriptedBackend, pipeline, silence, speech, two_sentences

BATCH = ["I want to order a pizza.", "Please deliver it tonight."]


class FakeStream:
    """A stream that answers `text` once the audio has ended, or fails when told (`fail_after` blocks fed, or at the end)."""

    def __init__(self, text="Streamed words.", fail_after=None, fail_at_end=False):
        self.text, self.fail_after, self.fail_at_end = text, fail_after, fail_at_end
        self.blocks, self.failed, self.closed = [], "", False
        self._ended = threading.Event()

    @property
    def ended(self):
        return self._ended.is_set()

    def feed(self, block):
        self.blocks.append(np.asarray(block))
        if self.fail_after is not None and len(self.blocks) >= self.fail_after:
            self.failed = "the connection broke"

    def end(self):
        self._ended.set()

    def final(self, timeout=120.0):
        self._ended.wait(5)  # as GeminiStream: the text is final only once the audio has ended
        if self.fail_at_end or self.failed:
            self.failed = self.failed or "no final text in time"
            return None
        return self.text

    def close(self):
        self.closed = True


class StreamingEngine:
    name = "fake-stream"

    def __init__(self, stream):
        self.stream, self.rates = stream, []

    def open_stream(self, rate):
        self.rates.append(rate)
        return self.stream


def feed(session, audio, step=RATE // 10):
    for i in range(0, len(audio), step):
        session.feed(audio[i:i + step])
    session.finish()


def test_an_engine_without_a_stream_leaves_the_session_as_it_is():
    session = pipeline(ScriptedBackend(BATCH)).start(RATE)
    assert streaming.attach(session, object()) is session
    assert streaming.attach(session, StreamingEngine(None)) is session


def test_the_streams_text_goes_through_the_text_stages_and_no_chunk_is_sent():
    backend, stream = ScriptedBackend(BATCH), FakeStream("i want to order a pizza please deliver it tonight")
    llm = FakeLLM(answer=lambda text: text.capitalize() + ".")
    session = streaming.attach(pipeline(backend, Stages(llm=llm)).start(RATE), StreamingEngine(stream))
    assert isinstance(session, StreamedSession)
    feed(session, two_sentences())
    result = session.result()
    assert backend.calls == []  # the batch model was never asked
    assert stream.ended and stream.closed is False
    assert result.provenance is Provenance.LLM_POLISHED and result.stages["streamed"] == stream.text
    assert llm.seen[0][0] == stream.text and result.text == "I want to order a pizza please deliver it tonight."
    assert sum(len(b) for b in stream.blocks) == len(two_sentences())  # every sample went to the stream


def test_no_final_text_sends_the_held_chunks_to_the_batch_model():
    backend, stream = ScriptedBackend(BATCH), FakeStream(fail_at_end=True)
    session = streaming.attach(pipeline(backend, max_in_flight=2).start(RATE), StreamingEngine(stream))
    feed(session, two_sentences())
    assert backend.calls == []  # held while the stream could still answer
    result = session.result()
    assert sorted(backend.calls) == [1, 2]
    assert result.text == "I want to order a pizza. Please deliver it tonight." and "streamed" not in result.stages


def test_a_stream_that_breaks_while_the_user_speaks_lets_the_chunks_go_at_once():
    backend, stream = ScriptedBackend(BATCH), FakeStream(fail_after=3)
    session = streaming.attach(pipeline(backend, max_in_flight=2).start(RATE), StreamingEngine(stream))
    audio = two_sentences()
    for i in range(0, 80_000 + 40_000, RATE // 10):  # the first sentence and the pause: its chunk is cut
        session.feed(audio[i:i + RATE // 10])
    deadline = time.monotonic() + 3
    while not backend.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert backend.calls == [1]  # transcribed before the release
    for i in range(120_000, len(audio), RATE // 10):
        session.feed(audio[i:i + RATE // 10])
    session.finish()
    assert len(stream.blocks) == 3  # nothing more went to a broken stream
    assert session.result().text == "I want to order a pizza. Please deliver it tonight."


def test_cancelling_closes_the_stream_and_the_session():
    stream = FakeStream()
    session = streaming.attach(pipeline(ScriptedBackend(BATCH)).start(RATE), StreamingEngine(stream))
    session.feed(speech(1.0))
    session.cancel()
    assert stream.closed and session.state.value == "cancelled"


def test_the_preroll_the_session_keeps_goes_to_the_stream_too():
    stream = FakeStream()
    preroll = np.concatenate([silence(1.5), speech(0.5)])
    session = pipeline(ScriptedBackend(BATCH)).start(RATE, preroll)
    session = streaming.attach(session, StreamingEngine(stream), preroll)
    assert sum(len(b) for b in stream.blocks) == len(preroll) - session.trimmed


def test_end_to_end_with_gemini_live_against_a_fake_server(monkeypatch):
    """GeminiEngine with a live model, the pipeline and a real WebSocket: the text arrives soon after the release."""
    monkeypatch.setattr(gemini_stream.urllib.request, "getproxies", lambda: {})
    server = FakeLive(during={20: [interim("i want to"), final("I want to order a pizza.")]},
                      after_end=[interim("please"), final("Please deliver it tonight.")])
    try:
        backend = ScriptedBackend(["wrong", "wrong"])
        engine = GeminiEngine(KEY, LIVE, "en", url=server.url)
        session = streaming.attach(pipeline(backend).start(RATE), engine)
        audio = two_sentences()
        done = threading.Event()

        def speak():  # in real time's order, a little faster
            for i in range(0, len(audio), RATE // 10):
                session.feed(audio[i:i + RATE // 10])
                time.sleep(0.005)
            session.finish()
            done.set()
        threading.Thread(target=speak).start()
        assert done.wait(10)
        t0 = time.monotonic()
        result = session.result()
        assert time.monotonic() - t0 < 1.0
        assert result.text == "I want to order a pizza. Please deliver it tonight."
        assert backend.calls == []
        assert len(server.frames) == len(range(0, len(audio), RATE // 10)) and server.ended
        assert server.setup["inputAudioTranscription"]["languageCodes"] == ["en-US", "en-GB", "en-IN"]
    finally:
        server.close()


def test_end_to_end_a_refused_stream_falls_back_on_the_batch_model(monkeypatch):
    monkeypatch.setattr(gemini_stream.urllib.request, "getproxies", lambda: {})
    server = FakeLive(close_with=(1008, "models/gemini-3.5-transcribe-live is not found"))
    try:
        backend = ScriptedBackend(BATCH)
        engine = GeminiEngine(KEY, LIVE, url=server.url)
        session = streaming.attach(pipeline(backend, max_in_flight=2).start(RATE), engine)
        feed(session, two_sentences())
        assert session.result().text == "I want to order a pizza. Please deliver it tonight."
        assert engine.open_stream(RATE) is None  # not tried again for a minute
    finally:
        server.close()


def dictate(stream, backend):
    typed, states = [], []
    recorder = LiveRecorder(np.concatenate([silence(1.5), speech(0.3)]))
    d = Dictation(StreamingEngine(stream), recorder, paste=typed.append, sounds=False, save=False)
    d.pipeline = pipeline(backend)
    d.on_state = lambda state, message: states.append(state)
    d.handle("press", 0.0)
    recorder.current_take.chunks += [speech(1.5, seed=5), silence(0.4)]
    d.handle("release", 2.0)
    d.wait()
    return typed, states


def test_dictation_types_the_streams_text():
    backend, stream = ScriptedBackend(["the batch model's words"]), FakeStream("the stream's words")
    typed, states = dictate(stream, backend)
    assert typed == ["the stream's words "] and states[-1] == "typed" and backend.calls == []
    assert stream.ended and len(stream.blocks) > 1


def test_dictation_types_the_batch_models_text_when_the_stream_fails():
    backend, stream = ScriptedBackend(["the batch model's words"]), FakeStream(fail_at_end=True)
    typed, states = dictate(stream, backend)
    assert typed == ["the batch model's words "] and states[-1] == "typed" and backend.calls == [1]
