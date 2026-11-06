"""Live translation spoken aloud: the Speaker (Gemini's own voice played as it streams: its cushion, its pauses,
catching up, muting) and the Player. No sound is played: the player and Windows are faked."""
import threading
import time

import numpy as np
import pytest

from sst.live import speaker as speakers
from sst.live.contracts import MIC, SYSTEM, Kind, LiveConfig
from sst.live.gemini import GeminiLiveTranslate
from sst.live.speaker import Speaker
from sst.live.wasapi import Player

RATE = 24_000  # Gemini's voice


@pytest.fixture(autouse=True)
def _no_voice_left_playing(monkeypatch):
    """Player.play marks Rflow's voice as playing in wasapi's globals, which the capture reads: put back after each test
    (left set, the capture tests run after these failed)."""
    from sst.live import wasapi
    monkeypatch.setattr(wasapi, "_voice_until", 0.0)
    monkeypatch.setattr(wasapi, "_voice_device", "")


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakePlayer:
    """Keeps what it plays, and how many plays ran at once (never more than one)."""

    def __init__(self, private=False, gate=None):
        self.private, self.interrupted, self.closed, self.gate = private, 0, False, gate
        self.pieces, self.rates, self.at_once, self.most, self.started = [], [], 0, 0, 0
        self._count = threading.Lock()

    def play(self, samples, rate):
        with self._count:
            self.at_once, self.started = self.at_once + 1, self.started + 1
            self.most = max(self.most, self.at_once)
        try:
            if self.gate is not None:
                self.gate.wait(5)
            self.pieces.append(np.array(samples, dtype=np.float32))
            self.rates.append(rate)
            time.sleep(0.002)
            return 0.0
        finally:
            with self._count:
                self.at_once -= 1

    def interrupt(self):
        self.interrupted += 1

    def close(self):
        self.closed = True

    @property
    def seconds(self):
        return sum(len(p) / r for p, r in zip(self.pieces, self.rates, strict=True))


def speaker_with(lanes=(SYSTEM,), player=None, clock=None, cushion=0.0, **named):
    """A Speaker on a clock that stands still unless the test moves it; without a cushion unless the test is about it."""
    speaker = Speaker(player or FakePlayer(), lanes, clock=clock or Clock(), **named)
    speaker.cushion = cushion
    return speaker


def wait_until(condition, seconds=3.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        time.sleep(0.01)
    return condition()


def pcm(level: int, seconds: float = 0.1, rate: int = RATE) -> bytes:
    return np.full(round(seconds * rate), level, dtype="<i2").tobytes()


# ---- Gemini's own voice, played as it streams

def test_geminis_voice_is_played_in_order_as_it_came():
    player = FakePlayer()
    speaker = speaker_with(player=player)
    speaker.start()
    pieces = [pcm(1000 * n) for n in range(1, 6)]
    for piece in pieces:
        speaker.hear_audio(SYSTEM, piece, RATE)
    assert wait_until(lambda: player.seconds >= 0.5)
    expected = np.concatenate([np.frombuffer(p, "<i2") for p in pieces]).astype(np.float32) / 32768
    assert np.array_equal(np.concatenate(player.pieces), expected) and set(player.rates) == {RATE}
    assert speaker.sound_s == pytest.approx(0.5) and player.most == 1
    speaker.stop()
    assert player.closed


def test_the_voices_first_piece_waits_for_its_cushion():
    clock, player = Clock(), FakePlayer()
    speaker = speaker_with(player=player, clock=clock, cushion=speakers.CUSHION)
    speaker.start()
    speaker.hear_audio(SYSTEM, pcm(3000, 0.25), RATE)
    time.sleep(0.2)
    assert player.pieces == [] and not speaker.speaking  # held: the pieces after it may come this late
    clock.now += speakers.CUSHION
    assert wait_until(lambda: len(player.pieces) == 1) and speaker.speaking
    speaker.stop()


def test_the_voice_is_speaking_while_it_plays():
    from sst.live import wasapi
    clock = Clock()
    speaker = speaker_with(player=Player(opener=lambda rate: FakeRender(rate)), clock=clock)
    speaker.start()
    assert not speaker.speaking
    before = time.monotonic()
    speaker.hear_audio(SYSTEM, pcm(3000, seconds=1.0), RATE)
    assert wait_until(lambda: speaker.sound_s > 0)
    assert speaker.speaking  # the microphone's way, near speakers, gets silence now
    assert wasapi._voice_until >= before + 1.0  # the capture leaves Rflow's own voice out of the whole mix meanwhile
    clock.now += 1
    assert not speaker.speaking
    speaker.stop()


def test_the_other_apps_are_lowered_while_it_runs_and_put_back_when_it_stops():
    class FakeDucker:
        def __init__(self):
            self.started = self.stopped = False

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True
    clock, ducker, player = Clock(), FakeDucker(), FakePlayer()
    speaker = speaker_with(player=player, clock=clock, ducker=ducker)
    speaker.start()
    assert ducker.started and not ducker.stopped  # lowered from the start, before a word is said
    speaker.hear_audio(SYSTEM, pcm(3000), RATE)
    assert wait_until(lambda: player.pieces)
    clock.now += 60  # a long pause: still lowered
    assert not ducker.stopped
    speaker.stop()
    assert ducker.stopped


def test_muting_and_stopping_silence_the_voice():
    gate = threading.Event()
    player = FakePlayer(gate=gate)
    speaker = speaker_with(player=player)
    speaker.start()
    for n in range(3):
        speaker.hear_audio(SYSTEM, pcm(1000 + n, seconds=0.6), RATE)  # each more than one piece played at once
    assert wait_until(lambda: player.started == 1)
    speaker.set_muted(True)  # the bar's speaker button: what's queued goes, what plays stops
    gate.set()
    speaker.hear_audio(SYSTEM, pcm(1005), RATE)
    time.sleep(0.2)
    assert len(player.pieces) == 1 and player.interrupted
    speaker.set_muted(False)
    speaker.stop()
    speaker.hear_audio(SYSTEM, pcm(1006), RATE)
    time.sleep(0.1)
    assert len(player.pieces) == 1 and player.closed


def test_only_the_ways_spoken_are_played():
    player = FakePlayer()
    speaker = speaker_with(player=player)
    speaker.start()
    speaker.hear_audio(MIC, pcm(1000), RATE)  # with Both, the user's own words are for the others
    time.sleep(0.2)
    assert player.pieces == []
    speaker.set_lanes([MIC])
    speaker.hear_audio(MIC, pcm(1000), RATE)
    assert wait_until(lambda: len(player.pieces) == 1)
    speaker.stop()


def test_which_ways_are_spoken():
    assert LiveConfig(target="en").spoken_lanes() == (SYSTEM,)
    assert LiveConfig(target="ja").spoken_lanes() == (SYSTEM,)  # Gemini speaks any language
    assert LiveConfig(source="microphone", mic_target="en").spoken_lanes() == (MIC,)
    assert LiveConfig(source="both").spoken_lanes() == (SYSTEM,)  # with both, the user's own words are for the others


def test_falling_behind_the_voice_skips_what_is_stale():
    gate, clock = threading.Event(), Clock()
    player = FakePlayer(gate=gate)
    speaker = speaker_with(player=player, clock=clock)
    speaker.start()
    speaker.hear_audio(SYSTEM, pcm(1000, seconds=0.5), RATE)
    assert wait_until(lambda: player.started == 1)
    speaker.hear_audio(SYSTEM, pcm(2000, seconds=0.5), RATE)  # waits behind it, then too long
    clock.now += speakers.STALE + 1
    speaker.hear_audio(SYSTEM, pcm(3000, seconds=0.5), RATE)
    gate.set()
    assert wait_until(lambda: len(player.pieces) == 2)
    assert [int(round(p[0] * 32768)) for p in player.pieces] == [1000, 3000]
    assert speaker.sound_skipped_s == pytest.approx(0.5)
    speaker.stop()


def test_a_player_that_fails_says_so_once_and_the_next_piece_is_tried():
    class Broken(FakePlayer):
        def play(self, samples, rate):
            self.started += 1
            raise OSError("no output device")
    problems, player = [], Broken()
    speaker = speaker_with(player=player, on_problem=problems.append)
    speaker.start()
    for _ in range(3):
        speaker.hear_audio(SYSTEM, pcm(3000, 0.6), RATE)
    assert wait_until(lambda: player.started == 3)
    assert problems == ["The voice couldn't play: no output device"]
    speaker.stop()


class EchoEngine:
    """Like Gemini Live Translate: its voice comes back as long as what it's fed, silence for silence."""

    def __init__(self, on_event):
        self.on_event, self.frames, self.on_audio = on_event, [], None

    def start(self):
        pass

    def feed(self, frame):
        self.frames.append(frame)
        level = int(np.abs(np.frombuffer(frame, "<i2")).max())
        self.on_audio(pcm(level, seconds=0.1), RATE)

    def stop(self):
        pass


def test_geminis_silence_doesnt_keep_the_way_that_hears_the_voice_silenced():
    """The whole mix (Rflow's voice in it) gets silence while the voice speaks, and Gemini answers silence with
    silence: played as voice, it kept the way silenced for good, and nothing more was translated after the first
    sentence."""
    from sst.live.session import LiveSession
    from tests.test_live_session import FakeCapture
    clock, player = Clock(), FakePlayer()
    session = LiveSession(LiveConfig(speak=True))
    capture = FakeCapture()
    capture.hears_self = True
    session.add(SYSTEM, capture, EchoEngine)
    echo = session.lanes[SYSTEM][1]
    speaker = speaker_with(player=player, clock=clock)
    session.set_speaker(speaker)
    talk = np.full(1600, 3000, dtype="<i2").tobytes()  # the meeting goes on, 100 ms at a time
    for _ in range(30):
        capture.on_frame(talk)
        assert wait_until(lambda: not speaker._queue)
        time.sleep(0.01)  # what was taken off the queue is played
        clock.now += 0.1
    heard = [frame for frame in echo.frames[1:] if any(frame)]
    session.stop()
    assert heard  # after the voice said the first, the meeting's words reach Gemini again


def test_geminis_silence_after_its_voice_isnt_played_and_isnt_speaking():
    clock, player = Clock(), FakePlayer()
    speaker = speaker_with(player=Air(clock, player), clock=clock)
    speaker.start()
    speaker.hear_audio(SYSTEM, pcm(0, seconds=0.5), RATE)  # silence, the voice quiet: left out
    assert wait_until(lambda: speaker.quiet_s == pytest.approx(0.5))
    assert player.pieces == [] and not speaker.speaking
    speaker.hear_audio(SYSTEM, pcm(3000, seconds=0.2), RATE)
    assert wait_until(lambda: len(player.pieces) == 1)
    speaker.hear_audio(SYSTEM, pcm(0, seconds=0.2), RATE)  # a pause between its words: played, as it is
    assert wait_until(lambda: len(player.pieces) == 2)
    clock.now += 0.2 + speakers.TAIL + 0.01
    assert not speaker.speaking  # the pause didn't make it last
    speaker.hear_audio(SYSTEM, pcm(0, seconds=0.2), RATE)  # more of it, the voice over: not played
    assert wait_until(lambda: speaker.quiet_s == pytest.approx(0.7)) and len(player.pieces) == 2
    speaker.stop()


class Air:
    """An output in the test's time: what it's given is heard after what was queued before it, for as long as it lasts
    (play() returns the seconds still to be heard, as the real player's)."""

    def __init__(self, clock, player=None):
        self.clock, self.player, self.until, self.private = clock, player or FakePlayer(), 0.0, False

    def play(self, samples, rate):
        self.player.play(samples, rate)
        self.until = max(self.until, self.clock()) + len(samples) / rate
        return self.until - self.clock()

    def interrupt(self):
        self.player.interrupt()

    def close(self):
        self.player.close()


def test_how_long_a_sound_is_heard():
    loud, quiet = np.full(RATE // 10, 0.3, np.float32), np.zeros(RATE // 2, np.float32)
    assert speakers.heard_for(np.concatenate([loud, quiet]), RATE) == pytest.approx(0.1)
    assert speakers.heard_for(quiet, RATE) == 0.0 and speakers.heard_for(np.zeros(0, np.float32), RATE) == 0.0
    assert speakers.heard_for(loud[:1000], RATE) == pytest.approx(1000 / RATE)  # a last block shorter than 20 ms


def test_pauses_are_shortened_to_catch_up_and_the_words_kept_whole():
    loud = np.full(RATE // 5, 0.3, dtype=np.float32)
    samples = np.concatenate([loud, np.zeros(RATE, np.float32), loud])
    shorter = speakers.shorten_pauses(samples, RATE)
    assert len(shorter) == 2 * len(loud) + round(speakers.PAUSE_S * 50) * RATE // 50  # the pause: 1 s -> 0.24 s
    assert np.count_nonzero(shorter) == 2 * len(loud)
    assert np.array_equal(speakers.shorten_pauses(loud, RATE), loud)  # no pause: as it was


def test_geminis_voice_from_the_fake_server_is_heard_through_the_session():
    from sst.live.session import LiveSession
    from tests.test_live_gemini import KEY, FakeServer, spoken
    from tests.test_live_session import FakeCapture
    server, shown, player = FakeServer(), [], FakePlayer()
    config = LiveConfig(speak=True)
    session = LiveSession(config, on_event=shown.append)
    session.add(SYSTEM, FakeCapture(), lambda emit: GeminiLiveTranslate(KEY, config, emit, connect=lambda url: server))
    session.set_speaker(speaker_with(config.spoken_lanes(), player=player))
    assert wait_until(lambda: server.sent)
    server.say(modelTurn=spoken(pcm(1200), pcm(1300)), outputTranscription={"text": "Hello there."})
    server.say(turnComplete=True)
    assert wait_until(lambda: player.seconds >= 0.2 - 1e-6)
    assert [e.text for e in shown if e.kind is Kind.LINE] == ["Hello there."]  # the caption as ever
    server.say(outputTranscription={"text": "No voice for this one."}, turnComplete=True)
    assert wait_until(lambda: [e.text for e in shown if e.kind is Kind.LINE][-1:] == ["No voice for this one."])
    time.sleep(0.1)
    assert player.seconds == pytest.approx(0.2)  # a line Gemini didn't say is shown, and no other voice says it
    session.stop()
    assert player.closed and session.speaker is None


# ---- the cushion, on the arrival times measured (#264): no thread, the test's own time

BUFFER = 0.2  # seconds Windows' output holds (sst.live.wasapi)


class Line:
    """Gemini's voice over a connection, in the test's own time. `pieces`: (when it arrives, loud or a pause), each
    PIECE seconds long. The Speaker's own steps (_take, _play) run here without its thread, against an output that
    plays in that time and takes what it's given as its 0.2 s buffer has room, like Windows'. `gaps`: the holes heard
    inside the voice (seconds of nothing between two loud pieces that followed each other; each loud piece's level
    says which it is)."""
    PIECE = 0.25

    def __init__(self, pieces, cushion=speakers.CUSHION):
        self.clock, self.pieces, self.private = Clock(), [(at, loud, i) for i, (at, loud) in enumerate(pieces)], False
        self.end, self.last, self.gaps, self.played_s = 0.0, None, [], 0.0
        self.speaker = Speaker(self, [SYSTEM], clock=self.clock)
        self.speaker.cushion = cushion
        self.t0 = self.clock.now

    def _arrive(self, until: float) -> None:
        """Every piece due by `until` reaches the speaker, at its own time."""
        while self.pieces and self.t0 + self.pieces[0][0] <= until:
            at, loud, i = self.pieces.pop(0)
            self.clock.now = max(self.clock.now, self.t0 + at)
            self.speaker.hear_audio(SYSTEM, pcm(3000 + i if loud else 0, self.PIECE), RATE)

    def play(self, samples, rate):
        now, seconds = self.clock.now, len(samples) / rate
        first, last = (round(float(x) * 32768) - 3000 if x else None for x in (samples[0], samples[-1]))
        if now > self.end and self.last is not None and first == self.last + 1:  # nothing played inside the voice
            self.gaps.append(now - self.end)
        self.end = max(now, self.end) + seconds
        self.last, self.played_s = last, self.played_s + seconds
        done = max(now, self.end - BUFFER)  # when the last of it has gone into the buffer: play() returns then
        self._arrive(done)
        self.clock.now = max(self.clock.now, done)
        return self.end - self.clock.now

    def interrupt(self):
        pass

    def close(self):
        pass

    def run(self):
        speaker = self.speaker
        while self.pieces or speaker._queue:
            self._arrive(self.clock.now)
            with speaker._lock:
                sound, wait = speaker._take(self.clock.now)
            if sound is not None:
                speaker._play(sound)
            else:
                coming = self.t0 + self.pieces[0][0] if self.pieces else self.clock.now + wait
                self.clock.now = min(coming, self.clock.now + wait)
        return self


def meeting(late, sentences=12, words=14, pause=5):
    """`sentences` of `words` loud pieces with `pause` quiet ones after each, a piece every 250 ms, each as late as
    late(i) says, and never before the one before it (one connection: they come in order)."""
    pieces, before = [], 0.0
    for i in range(sentences * (words + pause)):
        before = max(before, i * Line.PIECE + late(i))
        pieces.append((before, i % (words + pause) < words))
    return pieces


def measured(i: int) -> float:
    """How late a piece came on the owner's laptop (2026-11-06, sessions of a minute with the whole app running): most
    within 30 ms, one in ten up to 85 ms, one in a hundred 160 ms, and now and then 320 ms."""
    rng = np.random.default_rng(i)
    if i % 97 == 50:
        return 0.32 if i % 2 else 0.16
    return float(rng.uniform(0.03, 0.085)) if i % 10 == 3 else float(rng.uniform(0.0, 0.03))


def test_on_the_measured_arrival_times_the_cushion_leaves_no_hole_in_a_word():
    line = Line(meeting(measured)).run()
    assert line.gaps == [] and line.speaker.holes == 0 and line.speaker.cushion == speakers.CUSHION
    assert line.played_s >= 12 * 14 * Line.PIECE  # every word; of the pauses, only their start
    assert line.speaker.quiet_s > 0 and line.speaker.sound_skipped_s == 0


def test_without_a_cushion_the_same_arrival_times_left_holes():
    line = Line(meeting(measured), cushion=0.0).run()  # as it was until 2.8.1: a piece played the moment it came
    assert len(line.gaps) >= 12  # at least one in every sentence
    assert line.speaker.holes == len(line.gaps)


def test_the_voice_stays_a_cushion_behind_its_arrival_and_no_further():
    line = Line(meeting(measured))
    arrived, started = line.t0 + line.pieces[0][0], []
    play = line.play
    line.play = lambda samples, rate: (started.append(line.clock.now), play(samples, rate))[1]
    line.run()
    assert started[0] - arrived == pytest.approx(speakers.CUSHION, abs=0.06)  # the first word: this much later
    last = line.t0 + (12 * 19 - 5) * Line.PIECE  # when the last word's last piece came
    assert line.end - last < speakers.CUSHION + 0.75 + 0.2 + 0.32  # its end: never further behind, sentence by sentence


def test_a_stall_longer_than_the_cushion_makes_one_hole_and_the_cushion_grows():
    late = speakers.CUSHION + 0.3

    def stalled(i):
        return late if i in (6, 44) else 0.0  # the network holds everything that long, twice; then it all comes at once
    line = Line(meeting(stalled, sentences=4)).run()
    assert len(line.gaps) == 1 and line.speaker.holes == 1  # the first time only
    assert line.gaps[0] == pytest.approx(0.3, abs=0.06)  # what it was late by: what came with it is cushion enough
    assert line.speaker.cushion == 2 * speakers.CUSHION  # so the second stall, in a later sentence, was covered


def test_after_a_hole_the_voice_starts_a_cushion_ahead_again_and_the_rest_of_the_sentence_is_whole():
    """Everything a little later from one piece on (no burst to catch up with). As it was at first, played at once
    after the hole: every piece after it came as the one before ended, and the output ran dry for a moment at each (a
    click a piece for the rest of the sentence)."""
    def later(i):
        return 0.0 if i < 3 else speakers.CUSHION + 0.3 + float(np.random.default_rng(i).uniform(0.0, 0.03))
    line = Line(meeting(later, sentences=1, words=40)).run()
    assert len(line.gaps) == 1 and line.speaker.holes == 1
    assert line.gaps[0] == pytest.approx(0.3 + speakers.CUSHION, abs=0.08)  # late by this, and a cushion again


def test_the_cushion_never_grows_past_its_limit():
    line = Line(meeting(lambda i: 2.0 if i % 19 == 6 else 0.0, sentences=6)).run()
    assert line.speaker.holes >= 3 and line.speaker.cushion == speakers.MAX_CUSHION


def test_after_a_stall_the_voice_catches_up_in_its_pauses():
    """The network holds everything for 4 s, then sends it at once: the pauses of what waits are shortened, so the
    voice isn't 4 s behind for the rest of the meeting."""
    def stall(i):
        return max(0.0, 4.0 - (i - 10) * Line.PIECE) if 10 <= i < 26 else 0.0  # pieces 10 to 25 all arrive together
    line = Line(meeting(stall, sentences=8, words=6, pause=4)).run()
    last = line.t0 + (8 * 10 - 4) * Line.PIECE
    assert line.speaker.sound_skipped_s > 0  # pauses left out
    assert line.end - last < 1.5  # back with the speaker by the end
    assert line.played_s >= 8 * 6 * Line.PIECE  # and every word was said


# ---- the player

class FakeRender:
    """An opened output: takes up to `room` frames per write."""

    def __init__(self, rate, room=4000, private=False, changed=False, fail_writes=0):
        self.rate, self.room, self.private, self.changed, self.fail_writes = rate, room, private, changed, fail_writes
        self.written, self.flushed, self.closed = 0, False, False

    def write(self, pcm):
        if self.fail_writes:
            self.fail_writes -= 1
            raise OSError("device invalidated")
        n = min(self.room, len(pcm) // 2)
        self.written += n
        return n

    def queued(self):
        return 1600

    def flush(self):
        self.flushed = True

    def default_changed(self):
        return self.changed

    def close(self):
        self.closed = True


def test_the_player_writes_everything_and_says_what_is_still_to_be_heard():
    opened = []
    player = Player(opener=lambda rate: opened.append(FakeRender(rate)) or opened[-1])
    assert player.play(np.zeros(16000, np.float32), 16000) == pytest.approx(0.1)
    assert opened[0].written == 16000 and not player.private


def test_the_player_follows_a_new_default_output_and_a_lost_one():
    clock, opened = Clock(), []
    renders = iter([FakeRender(16000, changed=True), FakeRender(16000, private=True, fail_writes=1),
                    FakeRender(16000, private=True)])
    player = Player(opener=lambda rate: opened.append(next(renders)) or opened[-1], clock=clock)
    player.play(np.zeros(100, np.float32), 16000)
    clock.now += 3  # time for a look at the default: headphones now
    player.play(np.zeros(100, np.float32), 16000)
    assert opened[0].closed and opened[1].closed and opened[2].written == 100 and player.private


def test_interrupting_the_player_flushes_what_was_queued():
    player = Player(opener=lambda rate: FakeRender(rate, room=0))  # a full buffer: play waits
    done = []
    thread = threading.Thread(target=lambda: done.append(player.play(np.zeros(16000, np.float32), 16000)))
    thread.start()
    time.sleep(0.05)
    player.interrupt()
    thread.join(2)
    assert done == [0.0] and player._stream.flushed
