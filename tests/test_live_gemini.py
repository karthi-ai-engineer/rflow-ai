"""Live captions with Gemini 3.5 Live Translate, against a fake Live API server: no network, no key, no audio."""
import base64
import json
import queue
import socket
import threading
import time

import numpy as np
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from sst.live import gemini
from sst.live.contracts import Kind, LiveConfig
from sst.live.gemini import GeminiLiveTranslate

KEY = "AIza-secret-key"


class FakeServer:
    """One Live API connection: records what the client sends, answers what the test scripts."""

    def __init__(self, setup_reply=None, close_reason=None, refuse_setup=None):
        self.sent, self.closed = [], False
        self.incoming: queue.Queue = queue.Queue()
        self.setup_reply = setup_reply or {"setupComplete": {}}
        self.close_reason = close_reason
        self.refuse_setup = refuse_setup  # an exception: Google closes the connection instead of answering the setup

    def send(self, text):
        self.sent.append(json.loads(text))

    def recv(self, timeout=None):
        if self.refuse_setup:
            raise self.refuse_setup
        return json.dumps(self.setup_reply)

    def __iter__(self):
        while True:
            item = self.incoming.get()
            if isinstance(item, Exception):
                raise item
            if item is None or self.closed:
                if self.close_reason:
                    raise ConnectionError(self.close_reason)
                return
            yield json.dumps(item)

    def say(self, **content):
        self.incoming.put({"serverContent": content})

    def close(self):
        self.closed = True
        self.incoming.put(None)

    @property
    def frames(self):
        return [base64.b64decode(m["realtimeInput"]["audio"]["data"]) for m in self.sent if "audio" in m.get("realtimeInput", {})]


def closed_by_google(code, reason):
    """What websockets raises when Google closes the connection, as in the owner's first run."""
    return ConnectionClosedError(Close(code, reason), Close(code, reason), rcvd_then_sent=True)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def wait_until(condition, seconds=3.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def live():
    made = []

    def make(servers, **config):
        events, clock = [], Clock()
        servers = list(servers)

        def connect(url):
            assert url.endswith(f"?key={KEY}")
            if not servers:
                raise OSError("no more fake servers")
            server = servers.pop(0)
            if isinstance(server, Exception):
                raise server
            return server
        engine = GeminiLiveTranslate(KEY, LiveConfig(**config), events.append, connect=connect, clock=clock)
        made.append(engine)
        return engine, events, clock
    yield make
    for engine in made:
        engine.stop()


def kinds(events, kind):
    return [e for e in events if e.kind is kind]


def test_the_setup_asks_for_the_translation_and_both_transcripts(live):
    server = FakeServer()
    engine, events, _ = live([server], target="ja")
    engine.start()
    assert wait_until(lambda: server.sent)
    setup = server.sent[0]["setup"]
    assert setup["model"] == "models/gemini-3.5-live-translate-preview"
    config = setup["generationConfig"]
    assert config == {"responseModalities": ["AUDIO"],
                      "translationConfig": {"targetLanguageCode": "ja", "echoTargetLanguage": False}}
    # the setup's own fields: inside generationConfig Google closes the connection ("Unknown name")
    assert setup["inputAudioTranscription"] == {} and setup["outputAudioTranscription"] == {}
    assert wait_until(lambda: any(e.text == "Listening" for e in kinds(events, Kind.STATUS)))


def test_frames_stream_as_base64_pcm(live):
    server = FakeServer()
    engine, _, _ = live([server])
    engine.start()
    frame = bytes(range(256)) * 12 + bytes(128)  # 3,200 bytes: 100 ms at 16 kHz
    engine.feed(frame)
    engine.feed(frame)
    assert wait_until(lambda: len(server.frames) == 2) and server.frames[0] == frame
    audio = next(m for m in server.sent if "realtimeInput" in m)["realtimeInput"]["audio"]
    assert audio["mimeType"] == "audio/pcm;rate=16000"


VOICE = (np.sin(np.arange(1600) / 3) * 8000).astype("<i2").tobytes()  # 100 ms of something loud
SILENCE = bytes(3200)


def test_pieces_build_a_line_and_the_turn_finishes_it(live):
    server = FakeServer()
    engine, events, clock = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    engine.feed(VOICE)  # the speaker's last words at 100.0, then silence
    engine.feed(SILENCE)
    server.say(inputTranscription={"text": "今日は", "languageCode": "ja"})
    server.say(inputTranscription={"text": "会議です", "languageCode": "ja"})
    assert wait_until(lambda: len(kinds(events, Kind.SOURCE)) == 2)
    clock.now += 2.5  # the translation is complete 2.5 s after the voice paused
    server.say(outputTranscription={"text": "Today we", "languageCode": "en"})
    server.say(outputTranscription={"text": " have a meeting.", "languageCode": "en"})
    server.say(turnComplete=True)
    assert wait_until(lambda: kinds(events, Kind.LINE))
    assert [e.text for e in kinds(events, Kind.SOURCE)] == ["今日は", "今日は会議です"]
    assert [e.text for e in kinds(events, Kind.TRANSLATION)] == ["Today we", "Today we have a meeting."]
    line = kinds(events, Kind.LINE)[0]
    assert (line.source, line.text, line.language, line.seconds) == ("今日は会議です", "Today we have a meeting.", "ja", 2.5)


def test_a_piece_that_repeats_the_line_replaces_it():
    assert gemini._join("Hello", "Hello world") == "Hello world"
    assert gemini._join("Hello", " world") == "Hello world"
    assert gemini._join("", "今日") == "今日"
    assert gemini._join("at my parents'", "house") == "at my parents' house"  # the owner's run: "parents'house"
    assert gemini._join("I don'", "t know") == "I don't know" and gemini._join("Chris'", " car") == "Chris' car"


def test_a_piece_that_re_sends_the_line_revised_replaces_it_too():
    """In meetings the model re-sends the whole line with a word or a full stop changed: appended, the caption would say
    it twice and the voice would read it again."""
    line = " Yes, we can start now. Next"
    assert gemini._join(line, " Yes, we can start now. Next week.") == " Yes, we can start now. Next week."
    assert gemini._join(line, " Yes. We can start now. Next week.") == " Yes. We can start now. Next week."
    assert gemini._join("We need to finish the report. The deadline", "We have to finish the report. The deadline is "
                        "Friday.") == "We have to finish the report. The deadline is Friday."
    assert gemini._join(line, " week we review it.") == f"{line} week we review it."  # a piece that goes on
    assert gemini._join("Yes, we can", " can we start now?") == "Yes, we can can we start now?"


def test_a_pause_finishes_the_line_when_the_model_doesnt(live):
    server = FakeServer()
    engine, events, clock = live([server], line_pause_s=1.5)
    engine.start()
    assert wait_until(lambda: server.sent)
    server.say(inputTranscription={"text": "Good morning."})
    server.say(outputTranscription={"text": "おはようございます。"})
    assert wait_until(lambda: kinds(events, Kind.TRANSLATION))
    assert not kinds(events, Kind.LINE)
    clock.now += 1.6  # no new words since
    assert wait_until(lambda: kinds(events, Kind.LINE))
    assert kinds(events, Kind.LINE)[0].text == "おはようございます。"


def test_a_long_translation_ends_its_line_at_a_full_stop(live):
    server = FakeServer()
    engine, events, _ = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    server.say(inputTranscription={"text": "まずサーバーは三十五台で、予算は当面二万五千ドル、残りは後で。"})
    long = "We will start with thirty five servers, and the budget should be twenty five thousand dollars for now"
    server.say(outputTranscription={"text": long})
    assert wait_until(lambda: len(kinds(events, Kind.TRANSLATION)) == 1)
    assert not kinds(events, Kind.LINE)  # long, but mid-sentence
    server.say(outputTranscription={"text": " and the rest later."})
    assert wait_until(lambda: kinds(events, Kind.LINE))


def test_a_long_line_waits_for_the_words_heard_to_finish_its_sentence(live):
    """The owner's first run: the translation reached its full stop while the words heard were mid-word. The line ends
    when both are at a sentence end, so each sentence stays with its translation."""
    server = FakeServer()
    engine, events, _ = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    server.say(inputTranscription={"text": "なんと 。そうなんです 。で 、なんかコメントで僕もなんと"
                                           "住んでますみたいなコメントとかが来てびっくり"})
    server.say(outputTranscription={"text": "Oh my god. That's right. And then in the comments, I was like, I actually live "
                                            "there too, or something like that, and I was really surprised."})
    server.say(outputTranscription={"text": " Yeah, yeah."})  # the translation runs on ahead
    assert wait_until(lambda: len(kinds(events, Kind.TRANSLATION)) == 2)
    assert not kinds(events, Kind.LINE)  # the words heard are mid-sentence: wait
    server.say(inputTranscription={"text": "しました 。うん 、はい 。"})
    assert wait_until(lambda: kinds(events, Kind.LINE))
    line = kinds(events, Kind.LINE)[0]
    assert line.source == ("なんと。そうなんです。で、なんかコメントで僕もなんと"
                           "住んでますみたいなコメントとかが来てびっくりしました。うん、はい。")
    assert line.text.endswith("I was really surprised. Yeah, yeah.")


def test_a_translation_behind_the_words_heard_ends_the_line_when_it_catches_up(live):
    """The owner's second run: one Japanese sentence became two English ones, which arrived after it."""
    server = FakeServer()
    engine, events, _ = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    server.say(outputTranscription={"text": "I'm really begging you. Yes. Uh, my hometown is Tokyo, and I moved to Saitama "
                                            "Prefecture. Yeah. I moved there. Well, it's close, but"})
    server.say(inputTranscription={"text": "本当にお願いします。はい。あの、東京、実家東京から埼玉県に。うん。お引越ししました。"
                                           "ま、近いんですけどそれで言うと大体1時間半ぐらいはかかるかなという場所にいます。"})
    server.say(outputTranscription={"text": " so I guess it's about an hour and a half or so."})
    assert wait_until(lambda: len(kinds(events, Kind.TRANSLATION)) == 2)
    assert kinds(events, Kind.LINE)  # both at a sentence end now
    server.say(outputTranscription={"text": " I'm in a place that's about that far."})
    server.say(inputTranscription={"text": "へえ、あ、そっか。"})
    server.say(outputTranscription={"text": " Oh, I see."})
    server.say(turnComplete=True)
    assert wait_until(lambda: len(kinds(events, Kind.LINE)) == 2)
    # a sentence can still land a line late, but the next line doesn't inherit the slip
    assert kinds(events, Kind.LINE)[1].text == "I'm in a place that's about that far. Oh, I see."


def test_a_long_line_ends_anyway_when_the_words_heard_never_catch_up(live):
    server = FakeServer()
    engine, events, _ = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    server.say(inputTranscription={"text": "はい。えーと"})
    server.say(outputTranscription={"text": "A" * 130 + "."})
    server.say(outputTranscription={"text": " " + "B" * 125 + "."})
    assert wait_until(lambda: kinds(events, Kind.LINE))
    line = kinds(events, Kind.LINE)[0]
    assert line.text == "A" * 130 + ". " + "B" * 125 + "." and line.source == "はい。"
    assert wait_until(lambda: kinds(events, Kind.SOURCE)[-1].text == "えーと")  # the rest of the words heard carry on


def test_a_connection_skips_an_address_that_doesnt_answer(monkeypatch):
    """The dev laptop's network: the first of Google's addresses never answers. Here: a closed port, then a real one."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    dead_port = closed.getsockname()[1]
    closed.close()
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
                 for port in (dead_port, listener.getsockname()[1])]
    monkeypatch.setattr(gemini.socket, "getaddrinfo", lambda *args, **named: addresses)
    sock = gemini._socket("generativelanguage.googleapis.com", each=1.0)
    try:
        assert sock.getpeername()[1] == listener.getsockname()[1] and sock.gettimeout() is None
    finally:
        sock.close()
        listener.close()
    monkeypatch.setattr(gemini.socket, "getaddrinfo", lambda *args, **named: addresses[:1])
    with pytest.raises(OSError):
        gemini._socket("generativelanguage.googleapis.com", each=1.0)


def test_japanese_pieces_lose_the_spaces_between_them():
    assert gemini._tidy("なんと 。そうなんです 。で 、なんか") == "なんと。そうなんです。で、なんか"
    assert gemini._tidy("そんな大きい街 ?ですか ?まあ") == "そんな大きい街?ですか?まあ"
    assert gemini._tidy("GitHub のリポジトリ is ready. Yes .") == "GitHub のリポジトリ is ready. Yes ."  # only between Japanese


def test_no_lag_is_claimed_while_the_voice_goes_on(live):
    server = FakeServer()
    engine, events, clock = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    engine.feed(VOICE)
    server.say(inputTranscription={"text": "はい"})
    clock.now += 0.1
    engine.feed(VOICE)  # still speaking when the translation arrives
    server.say(outputTranscription={"text": "Yes."})
    server.say(turnComplete=True)
    assert wait_until(lambda: kinds(events, Kind.LINE))
    assert kinds(events, Kind.LINE)[0].seconds == 0.0


def test_speech_already_in_the_target_language_finishes_as_heard(live):
    server = FakeServer()
    engine, events, clock = live([server], target="en")
    engine.start()
    assert wait_until(lambda: server.sent)
    server.say(inputTranscription={"text": "Let's start.", "languageCode": "en"})  # the model stays silent
    assert wait_until(lambda: kinds(events, Kind.SOURCE))
    clock.now += 2
    assert wait_until(lambda: kinds(events, Kind.LINE))
    line = kinds(events, Kind.LINE)[0]
    assert line.source == "Let's start." and line.text == ""


def test_going_away_opens_a_new_connection_without_losing_audio(live):
    first, second = FakeServer(), FakeServer()
    engine, events, _ = live([first, second])
    engine.start()
    assert wait_until(lambda: first.sent)
    first.incoming.put({"goAway": {"timeLeft": "10s"}})
    assert wait_until(lambda: second.sent)  # set up again on the new one
    engine.feed(bytes(3200))
    assert wait_until(lambda: second.frames) and first.closed
    assert len([e for e in kinds(events, Kind.STATUS) if e.text == "Listening"]) == 2


def test_a_connection_is_renewed_before_googles_ten_minutes(live):
    first, second = FakeServer(), FakeServer()
    engine, _, clock = live([first, second], reconnect_s=540)
    engine.start()
    assert wait_until(lambda: first.sent)
    clock.now += 541
    assert wait_until(lambda: second.sent) and first.closed


def test_a_refused_key_says_so_and_stops_trying(live):
    server = FakeServer(close_reason="API key not valid. Please pass a valid API key.")
    engine, events, _ = live([server, FakeServer()])
    engine.start()
    assert wait_until(lambda: server.sent)
    server.incoming.put(None)  # the server closes the connection with its reason
    assert wait_until(lambda: kinds(events, Kind.ERROR))
    assert kinds(events, Kind.ERROR)[0].text == "Gemini didn't accept the key: check the Gemini key in AI & models."
    assert wait_until(lambda: any(e.text == "Stopped" for e in kinds(events, Kind.STATUS)))


def test_a_request_google_doesnt_accept_stops_at_once_and_is_logged_once(live, caplog):
    reason = 'Invalid JSON payload received. Unknown name "inputAudioTranscription" at \'setup.generation_config\''
    first = FakeServer(refuse_setup=closed_by_google(1007, reason))
    engine, events, _ = live([first, FakeServer()])
    engine.start()
    assert wait_until(lambda: any(e.text == "Stopped" for e in kinds(events, Kind.STATUS)))
    assert [e.text for e in kinds(events, Kind.ERROR)] == [gemini.NOT_ACCEPTED]  # one try, no reconnecting
    assert caplog.text.count("Unknown name") == 1 and "Google closed the connection (1007)" in caplog.text


def test_no_key_at_all_is_a_refused_key(live):
    reason = "Method doesn't allow unregistered callers (callers without established identity)."  # Google, measured
    engine, events, _ = live([FakeServer(refuse_setup=closed_by_google(1008, reason))])
    engine.start()
    assert wait_until(lambda: kinds(events, Kind.ERROR))
    assert kinds(events, Kind.ERROR)[0].text == gemini.REFUSED_KEY


def test_a_model_the_key_cant_use_isnt_called_a_refused_key(live):
    reason = "models/gemini-3.5-live-translate-preview is not found for API version v1beta"
    engine, events, _ = live([FakeServer(refuse_setup=closed_by_google(1008, reason))])
    engine.start()
    assert wait_until(lambda: kinds(events, Kind.ERROR))
    assert kinds(events, Kind.ERROR)[0].text == gemini.NO_MODEL


def test_a_first_connection_that_stalls_is_tried_again_quietly(live):
    server = FakeServer()
    engine, events, _ = live([TimeoutError("timed out"), server])
    engine.start()
    assert wait_until(lambda: server.sent) and not kinds(events, Kind.ERROR)


def test_a_connection_google_closes_while_listening_reconnects(live, monkeypatch):
    first, second = FakeServer(), FakeServer()
    engine, events, _ = live([first, second])
    monkeypatch.setattr(engine._stop, "wait", lambda seconds: False)
    engine.start()
    assert wait_until(lambda: first.sent)
    first.close_with = closed_by_google(1011, "Internal error encountered.")
    first.incoming.put(first.close_with)
    assert wait_until(lambda: second.sent)
    assert kinds(events, Kind.ERROR)[0].text == "The connection to Google broke: captions try again shortly."


def test_no_internet_says_so_and_tries_again(live, monkeypatch):
    server = FakeServer()
    engine, events, _ = live([OSError("getaddrinfo failed"), server])
    monkeypatch.setattr(engine._stop, "wait", lambda seconds: False)  # no real back-off wait in the test
    engine.start()
    assert wait_until(lambda: server.sent)  # the second try connected
    error = kinds(events, Kind.ERROR)[0]
    assert error.text.startswith("Can't reach Google")


def test_the_key_never_appears_in_messages(live, caplog):
    engine, events, _ = live([ConnectionError(f"bad request for ...?key={KEY}"), FakeServer()])
    engine._stop.wait = lambda seconds: False
    engine.start()
    assert wait_until(lambda: kinds(events, Kind.ERROR))
    assert all(KEY not in e.text for e in events) and KEY not in caplog.text


def test_when_the_network_falls_behind_the_oldest_audio_goes(live):
    engine, _, _ = live([])
    for i in range(gemini.QUEUE_FRAMES + 3):
        engine.feed(bytes([i]) * 3200)
    assert engine.dropped == 3 and engine._frames.qsize() == gemini.QUEUE_FRAMES
    assert engine._frames.get_nowait()[0] == 3  # the newest audio stays: captions stay live


def test_stop_finishes_the_line_and_says_stopped(live):
    server = FakeServer()
    engine, events, _ = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    server.say(inputTranscription={"text": "Last words"})
    assert wait_until(lambda: kinds(events, Kind.SOURCE))
    engine.stop()
    assert kinds(events, Kind.LINE)[-1].source == "Last words"
    assert kinds(events, Kind.STATUS)[-1].text == "Stopped" and not threading.current_thread().daemon


# -- Google's refusals, said plainly (#202): they arrive as a close frame's code and reason, or a refused handshake

def stopped(events):
    return any(e.text == "Stopped" for e in kinds(events, Kind.STATUS))


def refused_handshake(status, body=b""):
    return InvalidStatus(Response(status, "refused", Headers(), body))


@pytest.mark.parametrize("refusal, message", [
    (closed_by_google(1011, "Your prepayment credits are depleted. Please go to AI Studio at https://ai.studio/projects "
                            "to manage your project and billing."), gemini.NO_CREDIT),
    (refused_handshake(402, b'{"error": {"message": "Payment required"}}'), gemini.NO_CREDIT),
    (closed_by_google(1008, "Billing account for project is not active"), gemini.NO_CREDIT),
    (refused_handshake(401), gemini.REFUSED_KEY),
    (refused_handshake(403, b'{"error": {"status": "PERMISSION_DENIED"}}'), gemini.REFUSED_KEY),
    (closed_by_google(1008, "API key not valid. Please pass a valid API key."), gemini.REFUSED_KEY),
    (closed_by_google(1011, "Quota exceeded for metric: generativelanguage.googleapis.com/generate_requests_per_model_per_day,"
                            " limit: 50"), gemini.QUOTA_DAY),
])
def test_a_refusal_says_what_it_is_once_and_stops(live, refusal, message):
    engine, events, _ = live([FakeServer(refuse_setup=refusal), FakeServer()])
    engine.start()
    assert wait_until(lambda: stopped(events))
    assert [e.text for e in kinds(events, Kind.ERROR)] == [message]  # no retry, no other words
    assert engine.error == message


def test_a_short_rate_limit_is_waited_out_as_google_asks(live, monkeypatch):
    server = FakeServer()
    engine, events, _ = live([FakeServer(refuse_setup=closed_by_google(1011, "Rate limit hit. Please retry in 7.2s.")),
                              server])
    waits = []
    monkeypatch.setattr(engine._stop, "wait", lambda seconds: waits.append(seconds) or False)
    engine.start()
    assert wait_until(lambda: server.sent)  # tried again
    assert waits == [7.2] and kinds(events, Kind.ERROR)[0].text == gemini.RATE_LIMITED.format(7)


def test_a_quota_that_doesnt_say_when_is_tried_a_few_times_then_stops(live, monkeypatch):
    """Google's quota message mentions billing ("check your plan and billing details"): a quota, not missing credit."""
    reason = "You exceeded your current quota, please check your plan and billing details."
    servers = [FakeServer(refuse_setup=closed_by_google(1011, reason)) for _ in range(gemini.QUOTA_TRIES + 1)]
    engine, events, _ = live(servers)
    waits = []
    monkeypatch.setattr(engine._stop, "wait", lambda seconds: waits.append(seconds) or False)
    engine.start()
    assert wait_until(lambda: stopped(events))
    assert [e.text for e in kinds(events, Kind.ERROR)] == [gemini.QUOTA_NOW] * (gemini.QUOTA_TRIES - 1) + [gemini.QUOTA_STOP]
    assert waits == [gemini.QUOTA_WAIT] * (gemini.QUOTA_TRIES - 1)


# -- silence isn't streamed (#202): Google bills the audio sent

def test_a_long_silence_is_not_sent_and_the_next_sound_comes_with_its_preroll(live):
    server = FakeServer()
    engine, _, _ = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    engine.feed(VOICE)
    for _ in range(30):  # 3 s of digital silence
        engine.feed(SILENCE)
    engine.feed(VOICE)
    assert wait_until(lambda: len(server.frames) == 24)  # the voice, 2 s of silence, 200 ms of it again, the voice
    assert server.frames[0] == VOICE and server.frames[1:21] == [SILENCE] * 20
    assert server.frames[21:] == [SILENCE, SILENCE, VOICE]
    messages = [m["realtimeInput"] for m in server.sent if "realtimeInput" in m]
    assert messages[21] == {"audioStreamEnd": True}  # once, where the silence stopped being sent
    assert sum("audioStreamEnd" in m for m in messages) == 1
    assert engine.sent == 24 and round(engine.silence_s, 1) == 0.8


def test_quiet_sound_is_not_silence():
    hum = (np.sin(np.arange(1600) / 5) * 40).astype("<i2").tobytes()  # -58 dBFS peaks: a quiet room, still sent
    assert gemini._silent(SILENCE) and gemini._silent(b"") and not gemini._silent(hum)
    assert gemini._silent((np.ones(1600) * gemini.SILENT_PEAK).astype("<i2").tobytes())


def test_if_google_refuses_the_stream_end_silence_is_still_left_out_without_it(live, caplog):
    caplog.set_level("INFO", logger="sst.live.gemini")
    first, second = FakeServer(), FakeServer()
    engine, events, _ = live([first, second])
    engine.start()
    assert wait_until(lambda: first.sent)
    for _ in range(25):
        engine.feed(SILENCE)
    assert wait_until(lambda: any("audioStreamEnd" in m.get("realtimeInput", {}) for m in first.sent))
    first.incoming.put(closed_by_google(1007, 'Invalid JSON payload received. Unknown name "audio_stream_end"'))
    assert wait_until(lambda: second.sent)  # at once, quietly
    for _ in range(25):
        engine.feed(SILENCE)
    engine.feed(VOICE)
    assert wait_until(lambda: VOICE in second.frames)
    assert not kinds(events, Kind.ERROR) and not any("audioStreamEnd" in m.get("realtimeInput", {}) for m in second.sent)
    assert "refused audioStreamEnd" in caplog.text


def test_a_connection_google_closes_while_nothing_is_sent_reconnects_without_an_error(live, monkeypatch):
    first, second = FakeServer(), FakeServer()
    engine, events, _ = live([first, second])
    monkeypatch.setattr(engine._stop, "wait", lambda seconds: False)
    engine.start()
    assert wait_until(lambda: first.sent)
    for _ in range(25):
        engine.feed(SILENCE)
    assert wait_until(lambda: engine._skipping)
    first.incoming.put(None)  # closed, no reason given
    assert wait_until(lambda: second.sent) and not kinds(events, Kind.ERROR)
    assert engine.reconnects == 1


# -- the session goes on across connections (#202): Live API session resumption

def test_a_new_connection_takes_up_the_same_session(live):
    first, second = FakeServer(), FakeServer()
    engine, events, _ = live([first, second])
    engine.start()
    assert wait_until(lambda: first.sent)
    assert first.sent[0]["setup"]["sessionResumption"] == {}  # asked for from the start
    first.incoming.put({"sessionResumptionUpdate": {"newHandle": "h-1", "resumable": True}})
    first.incoming.put({"sessionResumptionUpdate": {"newHandle": "", "resumable": False}})  # mid-answer: h-1 kept
    first.say(inputTranscription={"text": "Half a"})
    assert wait_until(lambda: kinds(events, Kind.SOURCE))
    first.incoming.put({"goAway": {"timeLeft": "10s"}})
    assert wait_until(lambda: second.sent)
    assert second.sent[0]["setup"]["sessionResumption"] == {"handle": "h-1"}
    second.say(inputTranscription={"text": " sentence."})
    second.say(turnComplete=True)
    assert wait_until(lambda: kinds(events, Kind.LINE))
    assert [e.source for e in kinds(events, Kind.LINE)] == ["Half a sentence."]  # one line, not cut at the reconnect
    assert engine.reconnects == 1 and engine.resumed == 1


def test_a_setup_google_refuses_with_resumption_is_sent_again_without_it(live, caplog):
    caplog.set_level("INFO", logger="sst.live.gemini")
    refused = closed_by_google(1007, 'Invalid JSON payload received. Unknown name "sessionResumption" at \'setup\'')
    second = FakeServer()
    engine, events, _ = live([FakeServer(refuse_setup=refused), second])
    engine.start()
    assert wait_until(lambda: second.sent)
    assert "sessionResumption" not in second.sent[0]["setup"] and not kinds(events, Kind.ERROR)
    assert "refused session resumption" in caplog.text


def test_a_handle_google_no_longer_knows_starts_a_new_session(live):
    first, second, third = FakeServer(), FakeServer(refuse_setup=closed_by_google(1008, "Session not found")), FakeServer()
    engine, events, _ = live([first, second, third])
    engine.start()
    assert wait_until(lambda: first.sent)
    first.incoming.put({"sessionResumptionUpdate": {"newHandle": "old", "resumable": True}})
    first.incoming.put({"goAway": {}})
    assert wait_until(lambda: third.sent)
    assert second.sent[0]["setup"]["sessionResumption"] == {"handle": "old"}
    assert third.sent[0]["setup"]["sessionResumption"] == {} and not kinds(events, Kind.ERROR)


def test_the_renewal_waits_for_a_pause_but_not_for_ever(live):
    first, second, third = FakeServer(), FakeServer(), FakeServer()
    engine, events, clock = live([first, second, third], reconnect_s=540, line_pause_s=10_000)
    engine.start()
    assert wait_until(lambda: first.sent)
    first.say(inputTranscription={"text": "Mid sentence"})
    assert wait_until(lambda: kinds(events, Kind.SOURCE))
    clock.now += 541  # due, but a line is being built
    time.sleep(0.5)
    assert not second.sent
    first.say(turnComplete=True)  # the line ends: a pause
    assert wait_until(lambda: second.sent) and first.closed
    second.say(inputTranscription={"text": "Talking on"})
    assert wait_until(lambda: len(kinds(events, Kind.SOURCE)) == 2)
    clock.now += 540 + gemini.RECONNECT_GRACE_S + 1  # no pause for too long: renewed anyway, before Google cuts it
    assert wait_until(lambda: third.sent)


def test_the_report_has_numbers_and_no_words(live):
    server = FakeServer()
    engine, events, clock = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    engine.feed(VOICE)
    clock.now += 1.5
    server.say(inputTranscription={"text": "secret words"})
    assert wait_until(lambda: kinds(events, Kind.SOURCE) and server.frames)
    engine.stop()
    report = engine.report()
    assert "first words after 1.5 s" in report and "1 frames sent, 0 dropped" in report and "0 reconnects" in report
    assert "secret" not in report and KEY not in report


# ---- the translation spoken: Gemini's own voice, passed on as it comes

def spoken(*pieces: bytes, mime="audio/pcm;rate=24000") -> dict:
    """A serverContent.modelTurn with the translation spoken, as the Live API sends it (base64 PCM16 in inlineData)."""
    return {"parts": [{"inlineData": {"mimeType": mime, "data": base64.b64encode(p).decode("ascii")}} for p in pieces]}


def captions_of(events):
    return [(e.kind, e.text, e.source) for e in events if e.kind in (Kind.SOURCE, Kind.TRANSLATION, Kind.LINE)]


def test_the_translation_spoken_is_passed_on_in_order_and_the_captions_stay_the_same(live):
    def run(with_voice):
        server = FakeServer()
        engine, events, _ = live([server])
        audio = []
        engine.on_audio = lambda pcm, rate: audio.append((pcm, rate))
        engine.start()
        assert wait_until(lambda: server.sent)
        server.say(inputTranscription={"text": "今日は会議です"})
        if with_voice:  # Google's message may carry the voice and its transcript together
            server.say(modelTurn=spoken(bytes([1, 0]) * 4, bytes([2, 0]) * 4), outputTranscription={"text": "Today we"})
            server.say(modelTurn=spoken(bytes([3, 0]) * 4, mime=""))  # no type said: Google's 24 kHz
            server.say(modelTurn={"parts": [{"text": "not audio"}, {"inlineData": {"mimeType": "image/png", "data": "AA=="}}]})
        else:
            server.say(outputTranscription={"text": "Today we"})
        server.say(outputTranscription={"text": " have a meeting."})
        server.say(turnComplete=True)
        assert wait_until(lambda: kinds(events, Kind.LINE))
        return captions_of(events), audio, engine
    plain, none, _ = run(False)
    shown, audio, engine = run(True)
    assert shown == plain and none == []  # the captions exactly as without the voice
    assert audio == [(bytes([1, 0]) * 4, 24000), (bytes([2, 0]) * 4, 24000), (bytes([3, 0]) * 4, 24000)]
    assert engine.audio_s == pytest.approx(24 / 48000) and "0 s of Gemini's voice" in engine.report()


def test_a_listener_that_fails_never_ends_the_connection(live):
    server = FakeServer()
    engine, events, _ = live([server])
    engine.on_audio = lambda pcm, rate: 1 / 0
    engine.start()
    assert wait_until(lambda: server.sent)
    server.say(modelTurn=spoken(bytes([1, 0]) * 4))
    server.say(outputTranscription={"text": "Still here."}, turnComplete=True)
    assert wait_until(lambda: kinds(events, Kind.LINE)) and not kinds(events, Kind.ERROR)
    assert kinds(events, Kind.LINE)[0].text == "Still here."


def test_without_a_listener_the_voice_is_only_counted(live):
    server = FakeServer()
    engine, events, _ = live([server])
    engine.start()
    assert wait_until(lambda: server.sent)
    server.say(modelTurn=spoken(bytes([1, 0]) * 2400), outputTranscription={"text": "Hello."}, turnComplete=True)
    assert wait_until(lambda: kinds(events, Kind.LINE)) and engine.audio_s == pytest.approx(0.1)
