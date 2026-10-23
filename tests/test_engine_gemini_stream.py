"""Gemini 3.5 Transcribe Live (sst.engines.gemini_stream) against a fake Live API server on 127.0.0.1: a real WebSocket,
no Google, no key, no microphone."""
import base64
import contextlib
import json
import logging
import threading
import time

import numpy as np
import pytest
from websockets.sync.server import serve

from sst.engines import gemini_stream, load_engine
from sst.engines.cloud import CLOUD, CloudEngine
from sst.engines.gemini_stream import GeminiEngine, GeminiStream

KEY = "AIza-secret-key"
LIVE = "gemini-3.5-transcribe-live"
VOICE = (np.sin(np.arange(1600) / 5) * 0.2).astype(np.float32)  # 100 ms with a voice in it
QUIET = np.zeros(1600, np.float32)


def final(text):
    return {"serverContent": {"inputTranscription": {"text": text}}}


def interim(text):
    return {"serverContent": {"interimInputTranscription": {"text": text}}}


class FakeLive:
    """The Live API's WebSocket: records what the client sends and answers as scripted: `during` after that many audio
    frames, `after_end` once the audio has ended (a number there is a pause in seconds)."""

    def __init__(self, during=None, after_end=(), setup_reply=None, close_with=None):
        self.during, self.after_end = during or {}, list(after_end)
        self.setup_reply, self.close_with = setup_reply or {"setupComplete": {}}, close_with
        self.received, self.paths = [], []
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}/ws"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        with contextlib.suppress(OSError):  # shutdown() closes the socket under the selector on Windows
            self.server.serve_forever()

    def _handle(self, conn):
        self.paths.append(conn.request.path)
        self.received.append(json.loads(conn.recv()))
        if self.close_with:
            conn.close(*self.close_with)
            return
        conn.send(json.dumps(self.setup_reply))
        frames = 0
        for raw in conn:
            message = json.loads(raw)
            self.received.append(message)
            realtime = message.get("realtimeInput", {})
            if "audio" in realtime:
                frames += 1
                for answer in self.during.get(frames, []):
                    conn.send(json.dumps(answer))
            if realtime.get("audioStreamEnd"):
                for answer in self.after_end:
                    time.sleep(answer) if isinstance(answer, float) else conn.send(json.dumps(answer))

    @property
    def setup(self):
        return self.received[0]["setup"]

    @property
    def frames(self):
        return [m["realtimeInput"]["audio"] for m in self.received if "audio" in m.get("realtimeInput", {})]

    @property
    def ended(self):
        return any(m.get("realtimeInput", {}).get("audioStreamEnd") for m in self.received)

    def close(self):
        self.server.shutdown()


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(gemini_stream.urllib.request, "getproxies", lambda: {})
    servers = []

    def make(**script):
        servers.append(FakeLive(**script))
        return servers[-1]
    yield make
    for server in servers:
        server.close()


def stream_for(server, languages=(), words=()):
    stream = GeminiStream(KEY, LIVE, list(languages), list(words), url=server.url)
    stream.start()
    return stream


def speak(stream, blocks):
    for block in blocks:
        stream.feed(block)
        time.sleep(0.01)


def wait_until(condition, seconds=3.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_the_setup_asks_for_a_verbatim_transcript_with_the_language_and_your_words(live):
    server = live()
    stream = stream_for(server, ["ja-JP"], ["Rflow", "Parakeet"])
    assert wait_until(lambda: server.received)
    assert server.setup == {"model": f"models/{LIVE}", "generationConfig": {"responseModalities": ["TEXT"]},
                            "inputAudioTranscription": {"languageCodes": ["ja-JP"], "mode": "VERBATIM",
                                                        "customVocabulary": ["Rflow", "Parakeet"]}}
    assert server.paths == [f"/ws?key={KEY}"]
    stream.close()


def test_frames_stream_as_base64_pcm16_and_the_release_ends_the_audio(live):
    server = live()
    stream = stream_for(server)
    speak(stream, [VOICE, QUIET])
    assert wait_until(lambda: len(server.frames) == 2)
    assert server.frames[0]["mimeType"] == "audio/pcm;rate=16000"
    pcm = np.frombuffer(base64.b64decode(server.frames[0]["data"]), "<i2")
    assert len(pcm) == 1600 and np.allclose(pcm / 32767, VOICE, atol=1e-4)
    assert not server.ended
    stream.end()
    assert wait_until(lambda: server.ended)
    stream.close()


def test_final_pieces_are_the_text_and_guesses_are_ignored(live):
    server = live(during={2: [interim("hello th"), final("Hello there.")], 3: [interim("how a")]},
                  after_end=[interim("how are"), final(" How are you?")])
    stream = stream_for(server)
    speak(stream, [VOICE, VOICE, VOICE, QUIET])
    stream.end()
    t0 = time.monotonic()
    assert stream.final() == "Hello there. How are you?"
    assert time.monotonic() - t0 < 1.0 and not stream.failed


def test_japanese_pieces_join_without_spaces(live):
    server = live(after_end=[final("今日は"), final("会議です。")])
    stream = stream_for(server, ["ja-JP"])
    speak(stream, [VOICE, QUIET])
    stream.end()
    assert stream.final() == "今日は会議です。"


def test_the_turn_complete_ends_the_wait_at_once(live, monkeypatch):
    monkeypatch.setattr(gemini_stream, "SETTLE", 5.0)  # only the turn's end can finish it in time
    server = live(after_end=[final("Done."), {"serverContent": {"turnComplete": True}}])
    stream = stream_for(server)
    speak(stream, [VOICE])
    stream.end()
    assert stream.final() == "Done."


def test_words_already_final_when_the_voice_stopped_need_no_answer_after_the_release(live, monkeypatch):
    monkeypatch.setattr(gemini_stream, "COVERED", 0.3)
    server = live(during={4: [final("All of it.")]})  # Google finalised at the pause; nothing comes after the release
    stream = stream_for(server)
    speak(stream, [VOICE, VOICE, QUIET, QUIET])
    assert wait_until(lambda: stream._finals)
    stream.end()
    assert stream.final() == "All of it."


def test_a_guess_still_pending_at_the_release_waits_for_its_final_words(live, monkeypatch):
    monkeypatch.setattr(gemini_stream, "COVERED", 0.0)
    monkeypatch.setattr(gemini_stream, "FINAL_WAIT", 0.6)
    server = live(during={1: [final("First."), interim("and the")]})  # the last words never become final
    stream = stream_for(server)
    speak(stream, [VOICE, QUIET])
    stream.end()
    assert stream.final() is None and stream.failed == "no final text in time"


def test_no_final_text_in_time_gives_none_so_the_batch_model_takes_over(live, monkeypatch):
    monkeypatch.setattr(gemini_stream, "FINAL_WAIT", 0.4)
    server = live(after_end=[1.0, final("Too late.")])
    stream = stream_for(server)
    speak(stream, [VOICE])
    stream.end()
    t0 = time.monotonic()
    assert stream.final() is None
    assert time.monotonic() - t0 < 0.9


def test_nothing_heard_is_left_to_the_batch_model(live, monkeypatch):
    monkeypatch.setattr(gemini_stream, "FINAL_WAIT", 0.4)
    server = live(after_end=[{"serverContent": {"turnComplete": True}}])
    stream = stream_for(server)
    speak(stream, [QUIET])
    stream.end()
    assert stream.final() is None and stream.failed == "no words"


def test_a_refused_key_fails_the_stream_without_the_key_in_the_reason_or_the_log(live, caplog):
    caplog.set_level(logging.DEBUG)
    server = live(close_with=(1008, f"API key not valid: {KEY}"))
    stream = stream_for(server)
    assert wait_until(lambda: stream.failed)
    assert "1008" in stream.failed and KEY not in stream.failed and "***" in stream.failed
    stream.feed(VOICE)
    stream.end()
    assert stream.final() is None
    ours = [r for r in caplog.records if not r.name.startswith("websockets.server")]  # the fake server's own logs
    assert ours and not any(KEY in r.getMessage() for r in ours)


def test_google_ending_the_session_fails_the_stream(live):
    server = live(during={1: [{"goAway": {"timeLeft": "5s"}}]})
    stream = stream_for(server)
    speak(stream, [VOICE])
    assert wait_until(lambda: stream.failed)
    assert "ends the session" in stream.failed


def test_no_server_fails_the_stream_and_the_next_dictations_skip_it_for_a_while(monkeypatch):
    monkeypatch.setattr(gemini_stream.urllib.request, "getproxies", lambda: {})
    engine = GeminiEngine(KEY, LIVE, url="ws://127.0.0.1:9/ws")  # the discard port: nothing listens
    stream = engine.open_stream(16_000)
    assert stream is not None
    assert wait_until(lambda: stream.failed, 6.0)
    stream.end()
    assert stream.final() is None
    assert engine.open_stream(16_000) is None  # down for DOWN_FOR


def test_a_dictation_longer_than_a_session_fails_the_stream(live, monkeypatch):
    monkeypatch.setattr(gemini_stream, "SESSION_LIMIT", 0.2)
    server = live()
    stream = stream_for(server)
    speak(stream, [VOICE])
    time.sleep(0.3)
    stream.feed(VOICE)
    assert wait_until(lambda: stream.failed)
    assert "outlasted" in stream.failed


# -- the engine


def test_gemini_is_a_gemini_engine_whose_live_model_uses_its_batch_twin_for_requests():
    engine = load_engine("gemini", "en", KEY, LIVE)
    assert isinstance(engine, GeminiEngine) and isinstance(engine, CloudEngine)
    assert engine.chosen == LIVE and engine.model == "gemini-3.5-transcribe" and engine.live
    assert engine.title == f"Google Gemini {LIVE}" and KEY not in repr(engine)
    batch = load_engine("gemini", "en", KEY, "gemini-3.5-transcribe")
    assert not batch.live and batch.open_stream(16_000) is None
    assert engine.signature == batch.signature + f"|live:{LIVE}"  # the stream's text can differ from the twin's
    engine.model = "gemini-3.5-flash-lite"  # the app switches models without loading the engine again
    assert engine.model == "gemini-3.5-flash-lite" and not engine.live
    assert LIVE in CLOUD["gemini"].models and CLOUD["gemini"].models[0] != LIVE  # offered, not the default


def test_a_stream_only_where_the_live_model_can(live):
    server = live()
    engine = GeminiEngine(KEY, LIVE, "ta", url=server.url)
    assert engine.open_stream(16_000) is None  # Tamil isn't one of its languages: the batch way
    engine.language = "ja"
    assert engine.open_stream(48_000) is None
    stream = engine.open_stream(16_000)
    assert stream is not None and stream.languages == ["ja-JP"]
    stream.close()


def test_your_words_go_as_custom_vocabulary(live):
    server = live()
    engine = GeminiEngine(KEY, LIVE, url=server.url)
    engine.words = ["Rflow", "Rflow", "Karthi"]
    stream = engine.open_stream(16_000)
    assert wait_until(lambda: server.received)
    assert server.setup["inputAudioTranscription"]["customVocabulary"] == ["Rflow", "Karthi"]
    assert server.setup["inputAudioTranscription"]["languageCodes"] == []  # automatic
    stream.close()
