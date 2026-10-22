"""Live translation spoken aloud: the voice's download and preparation, the Speaker (sentences, catching up, muting),
and the Player. No sound is played, nothing is downloaded: the voice, the player and Windows are faked."""
import json
import threading
import time

import numpy as np
import pytest

from sst import downloads
from sst.live import speaker as speakers
from sst.live import voice as voices
from sst.live.contracts import MIC, SYSTEM, Kind, LiveConfig, LiveEvent
from sst.live.gemini import GeminiLiveTranslate
from sst.live.speaker import Speaker, whole_sentences
from sst.live.voice import PREPARED, TOKENS, Voice, onnx_metadata, prepare
from sst.live.wasapi import Player


@pytest.fixture(autouse=True)
def _no_voice_left_playing(monkeypatch):
    """Player.play marks Rflow's voice as playing in wasapi's globals, which the capture reads: put back after each test
    (left set, the capture tests run after these failed)."""
    from sst.live import wasapi
    monkeypatch.setattr(wasapi, "_voice_until", 0.0)
    monkeypatch.setattr(wasapi, "_voice_device", "")

# ---- the voice: downloaded, then prepared for sherpa-onnx

def read_varint(data: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[i]
        value |= (byte & 0x7F) << shift
        i, shift = i + 1, shift + 7
        if not byte & 0x80:
            return value, i


def metadata_props(data: bytes) -> dict[str, str]:
    """ModelProto's metadata_props (field 14) read back from protobuf bytes."""
    found, i = {}, 0
    while i < len(data):
        tag, i = read_varint(data, i)
        length, i = read_varint(data, i)
        payload, i = data[i:i + length], i + length
        if tag == (14 << 3 | 2):
            entry, j = {}, 0
            while j < len(payload):
                inner, j = read_varint(payload, j)
                size, j = read_varint(payload, j)
                entry[inner >> 3], j = payload[j:j + size].decode(), j + size
            found[entry[1]] = entry[2]
    return found


def test_the_metadata_is_protobuf_that_merges_into_the_model():
    data = onnx_metadata({"model_type": "vits", "sample_rate": 16000, "comment": "piper"})
    assert metadata_props(data) == {"model_type": "vits", "sample_rate": "16000", "comment": "piper"}
    long = onnx_metadata({"k": "v" * 300})  # lengths over 127 take two bytes
    assert metadata_props(long) == {"k": "v" * 300}


VOICE = Voice("test", "Test", "en", downloads.Download("voice", "https://huggingface.co/v/", (
    downloads.ModelFile("test.onnx", 4, "x"), downloads.ModelFile("test.onnx.json", 1, "x"))), "test.onnx",
    downloads.Download("data", "https://huggingface.co/d/", (downloads.ModelFile("phontab", 1, "x"),)))
CONFIG = {"language": {"name_english": "English"}, "espeak": {"voice": "en-us"}, "num_speakers": 1,
          "audio": {"sample_rate": 16000}, "phoneme_id_map": {"_": [0], "^": [1], " ": [3], "a": [14]},
          "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8}}


def downloaded(root):
    folder = VOICE.folder(root)
    folder.mkdir(parents=True)
    model = b"\x08\x07" + b"MODEL"  # any protobuf: ir_version = 7, then the weights
    (folder / "test.onnx").write_bytes(model)
    (folder / "test.onnx.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    return folder, model


def test_preparing_appends_the_settings_writes_the_tokens_and_drops_the_original(tmp_path):
    folder, model = downloaded(tmp_path)
    prepare(VOICE, tmp_path)
    prepared = (folder / PREPARED).read_bytes()
    assert prepared.startswith(model) and metadata_props(prepared[len(model):]) == {
        "model_type": "vits", "comment": "piper", "language": "English", "voice": "en-us", "has_espeak": "1",
        "n_speakers": "1", "sample_rate": "16000"}
    assert (folder / TOKENS).read_text(encoding="utf-8") == "_ 0\n^ 1\n  3\na 14\n"
    assert not (folder / "test.onnx").exists()  # not kept twice
    assert json.loads((folder / voices.MARKER).read_text(encoding="utf-8"))["from"] == "test.onnx"


def test_installing_downloads_what_is_missing_then_prepares_once(tmp_path, monkeypatch):
    fetched = []

    def fake_download(model, progress, cancelled, root):
        fetched.append(model.folder)
        if model is VOICE.model and not (VOICE.folder(root) / "test.onnx").exists():
            downloaded(root)
        (model.path(root)).mkdir(parents=True, exist_ok=True)
        for f in model.files:
            if not (model.path(root) / f.name).exists():
                (model.path(root) / f.name).write_bytes(b"x" * f.size)
        (model.path(root) / downloads.MARKER).write_text("{}", encoding="utf-8")
        progress(model.size, model.size)
        return model.path(root)
    monkeypatch.setattr(voices, "download", fake_download)
    seen = []
    assert not VOICE.ready(tmp_path)
    voices.install(VOICE, lambda done, total: seen.append((done, total)), root=tmp_path)
    assert fetched == ["data", "voice"] and VOICE.ready(tmp_path) and seen[-1] == (VOICE.size, VOICE.size)
    voices.install(VOICE, root=tmp_path)
    assert fetched == ["data", "voice"]  # ready: nothing fetched again


def test_the_real_voices_are_pinned_on_hugging_face():
    for voice in (voices.RYAN, voices.DANNY):
        for download in (voice.model, voice.data):
            assert downloads.allowed(download.base_url) and "/resolve/" in download.base_url
            assert all(len(f.sha256) == 64 for f in download.files)
        assert voice.language == "en" and 60 * 2**20 < voice.size < 70 * 2**20
    assert voices.LOCAL is voices.RYAN and voices.RYAN.data is voices.DANNY.data  # the same English espeak-ng data


def test_an_older_rflows_danny_speaks_until_ryan_is_downloaded(tmp_path, monkeypatch):
    have = set()
    monkeypatch.setattr(Voice, "ready", lambda self, root=None: self.key in have)
    assert voices.local(tmp_path) is voices.RYAN  # nothing yet: Ryan is the one to download
    have.add(voices.DANNY.key)
    assert voices.local(tmp_path) is voices.DANNY  # never left without a voice while Ryan comes
    have.add(voices.RYAN.key)
    assert voices.local(tmp_path) is voices.RYAN


# ---- whole sentences

@pytest.mark.parametrize(("text", "sentences"), [
    ("Hello there. How are", ["Hello there."]),
    ("It costs 3.", []),  # "3.5" may follow: not yet
    ("It costs 3.5 million. Next", ["It costs 3.5 million."]),
    ("Use e.g. this one. And", ["Use e.g. this one."]),
    ("Mr. Smith left. Then", ["Mr. Smith left."]),
    ("Wait... really? Yes! Ok", ["Wait...", "really?", "Yes!"]),
    ("He said “yes.” Then", ["He said “yes.”"]),
])
def test_a_sentence_is_whole_once_the_text_goes_on_past_it(text, sentences):
    assert whole_sentences(text)[0] == sentences


# ---- the speaker

class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeVoice:
    rate = 16000

    def __init__(self, log):
        self.log = log

    def synthesize(self, text, speed=1.0):
        self.log.append((text, round(speed, 2)))
        return np.zeros(int(0.5 * self.rate), dtype=np.float32)  # half a second of speech


class FakePlayer:
    def __init__(self, private=False, gate=None):
        self.private, self.played, self.interrupted, self.closed, self.gate = private, [], 0, False, gate

    def play(self, samples, rate):
        if self.gate is not None:
            self.gate.wait(5)
        self.played.append(len(samples) / rate)
        return 0.0

    def interrupt(self):
        self.interrupted += 1

    def close(self):
        self.closed = True


def speaker_with(lanes=(SYSTEM,), player=None, clock=None, **named):
    said = []
    speaker = Speaker(lambda: FakeVoice(said), player or FakePlayer(), lanes, clock=clock or Clock(), **named)
    return speaker, said


def wait_until(condition, seconds=3.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        time.sleep(0.01)
    return condition()


def translation(text, lane=SYSTEM):
    return LiveEvent(Kind.TRANSLATION, text, lane=lane)


def test_each_sentence_is_said_once_as_soon_as_it_is_whole_and_the_rest_when_the_line_ends():
    speaker, said = speaker_with()
    speaker.start()
    for text in ("Thanks for", "Thanks for joining.", "Thanks for joining. The budget", "Thanks for joining. The budget is"):
        speaker.hear(translation(text))
    assert wait_until(lambda: said == [("Thanks for joining.", 1.0)])
    speaker.hear(LiveEvent(Kind.LINE, "Thanks for joining. The budget is ready.", source="…"))
    assert wait_until(lambda: [text for text, _ in said] == ["Thanks for joining.", "The budget is ready."])
    speaker.stop()
    assert speaker.player.closed


def test_a_last_sentence_is_said_after_a_short_quiet_without_waiting_for_the_line_to_end():
    clock = Clock()
    speaker, said = speaker_with(clock=clock)
    speaker.start()
    speaker.hear(translation("We'll decide next week."))
    time.sleep(0.3)
    assert said == []  # it may still go on ("week.5"? no; but "week. And" could come)
    clock.now += 0.7
    assert wait_until(lambda: said == [("We'll decide next week.", 1.0)])
    speaker.stop()


def test_falling_behind_it_speaks_faster_and_skips_what_is_stale():
    gate, clock = threading.Event(), Clock()
    speaker, said = speaker_with(player=FakePlayer(gate=gate), clock=clock)
    speaker.start()
    speaker.hear(translation("One. Two. Three. Four. Five. And"))
    assert wait_until(lambda: len(said) == 1)  # "One." is playing; four wait behind it
    clock.now += 11  # they've waited too long: the old ones go, the newest is said
    gate.set()
    assert wait_until(lambda: [text for text, _ in said] == ["One.", "Five."])
    assert said[0][1] == 1.6 and speaker.skipped == 3  # "One." was said faster: four were waiting
    speaker.stop()


def test_only_the_ways_it_speaks_are_said_and_untranslated_lines_are_not():
    speaker, said = speaker_with(lanes=(SYSTEM,))
    speaker.start()
    speaker.hear(LiveEvent(Kind.LINE, "はい、そうしましょう。", source="Yes, let's.", lane=MIC))  # the user's own: not said
    speaker.hear(LiveEvent(Kind.LINE, "", source="Let's start."))  # already in English: heard as it is
    speaker.hear(LiveEvent(Kind.LINE, "Good morning."))
    assert wait_until(lambda: [text for text, _ in said] == ["Good morning."])
    speaker.set_lanes([])
    speaker.hear(LiveEvent(Kind.LINE, "Not said."))
    time.sleep(0.2)
    assert [text for text, _ in said] == ["Good morning."]
    speaker.stop()


def test_muting_drops_what_waits_and_stops_what_plays():
    gate = threading.Event()
    speaker, said = speaker_with(player=FakePlayer(gate=gate))
    speaker.start()
    speaker.hear(translation("One. Two. Three"))
    assert wait_until(lambda: len(said) == 1)
    speaker.set_muted(True)
    gate.set()
    speaker.hear(translation("One. Two. Three. Four. Five"))
    time.sleep(0.2)
    assert [text for text, _ in said] == ["One."] and speaker.player.interrupted
    speaker.set_muted(False)
    speaker.hear(LiveEvent(Kind.LINE, "One. Two. Three. Four. Five. Six."))  # what was whole while muted stays unsaid
    assert wait_until(lambda: [text for text, _ in said] == ["One.", "Five. Six."])  # "Five" ended after unmuting
    speaker.stop()


def test_a_rewritten_line_is_not_said_twice():
    speaker, said = speaker_with()
    speaker.start()
    speaker.hear(translation("Hello there. How are"))
    assert wait_until(lambda: len(said) == 1)
    speaker.hear(translation("Hello there. How is it going? Fine"))  # the model rewrote the second sentence
    assert wait_until(lambda: [text for text, _ in said] == ["Hello there.", "How is it going?"])
    speaker.stop()


# ---- Gemini's events in a real meeting: the line's text revised, re-sent and restarted, each sentence said once

class Meeting:
    """Gemini Live Translate's own event making (gemini.py, fed the server's messages by hand) into a Speaker that
    isn't started: what it would say is what waits in its queue."""

    def __init__(self):
        self.clock = Clock()
        self.speaker = Speaker(lambda: None, FakePlayer(), [SYSTEM], clock=self.clock)
        self.engine = GeminiLiveTranslate("key", LiveConfig(), self.speaker.hear, connect=None, clock=self.clock)

    def run(self, steps):
        for step, *value in steps:
            if step == "say":  # a piece of the translation (outputTranscription)
                self.engine._handle({"serverContent": {"outputTranscription": {"text": value[0]}}})
            elif step == "heard":  # a piece of the words heard (inputTranscription)
                self.engine._handle({"serverContent": {"inputTranscription": {"text": value[0]}}})
            elif step == "turn":
                self.engine._handle({"serverContent": {"turnComplete": True}})
            else:  # "quiet": nothing new for this long (a pause ends the line, a last full stop settles)
                self.clock.now += value[0]
                self.engine._check_pause()
                with self.speaker._lock:
                    self.speaker._settle()
        return [sentence for _, _, sentence in self.speaker._queue]


MEETING = {
    "pieces that start with a space, the line ended by the turn": (
        [("say", " Thanks for joining."), ("say", " The budget"), ("say", " is ready."), ("turn",)],
        ["Thanks for joining.", "The budget is ready."]),
    "a last sentence said after a quiet, then its line ends": (
        [("say", " We'll decide"), ("say", " next week."), ("quiet", 0.7), ("turn",)],
        ["We'll decide next week."]),
    "a pause ends the line, then the model re-sends its whole turn": (
        [("heard", "予算は"), ("say", "Thanks for joining."), ("say", " The budget is ready."), ("quiet", 1.6),
         ("say", "Thanks for joining. The budget is ready. We start on Monday."), ("turn",)],
        ["Thanks for joining.", "The budget is ready.", "We start on Monday."]),
    "the line re-sent with an earlier word revised": (
        [("say", "We need"), ("say", " to finish the report."), ("say", " The deadline"),
         ("say", "We have to finish the report. The deadline is Friday."), ("turn",)],
        ["We need to finish the report.", "The deadline is Friday."]),
    "the line re-sent with a comma made a full stop": (
        [("say", " Yes, we can start now."), ("say", " Next"), ("say", " Yes. We can start now. Next week we review."),
         ("turn",)],
        ["Yes, we can start now.", "Next week we review."]),
    "a new turn that starts with the last one's end": (
        [("say", " The budget is ready."), ("say", " We start on Monday."), ("turn",),
         ("say", " We start on Monday."), ("say", " Any questions?"), ("turn",)],
        ["The budget is ready.", "We start on Monday.", "Any questions?"]),
    "a long line cut at a sentence end, the rest re-sent with the next": (
        [("heard", "First point. "), ("say", "The first point is that the budget for the next quarter is ready "
                                            "and approved."),
         ("say", " The second point is that we start hiring in April, as planned."), ("heard", "Second point."),
         ("say", " The second point is that we start hiring in April, as planned. Third, the office moves."),
         ("turn",)],
        ["The first point is that the budget for the next quarter is ready and approved.",
         "The second point is that we start hiring in April, as planned.", "Third, the office moves."]),
}


@pytest.mark.parametrize("name", MEETING)
def test_each_sentence_of_a_meeting_is_said_once_however_gemini_revises_it(name):
    steps, sentences = MEETING[name]
    assert Meeting().run(steps) == sentences


def test_a_revision_of_what_was_said_says_nothing_and_is_counted():
    meeting = Meeting()
    said = meeting.run([("say", "The budget is ready. We"), ("say", "The budgets are ready! We")])
    assert said == ["The budget is ready."] and meeting.speaker.repeats == 1
    assert meeting.run([("say", " start on Monday."), ("turn",)]) == ["The budget is ready.", "We start on Monday."]


def test_what_is_said_again_later_or_only_looks_alike_is_still_said():
    meeting = Meeting()
    first = [("say", "Yes."), ("turn",), ("say", "I think so."), ("turn",)]
    assert meeting.run(first) == ["Yes.", "I think so."]
    meeting.clock.now += 30  # a while later: "Yes." again is a new answer
    assert meeting.run([("say", "Yes."), ("turn",), ("say", "I think not."), ("turn",)])[2:] == ["Yes.", "I think not."]
    sentences = [f"Point number {n} is on the agenda today." for n in range(1, 22)]
    meeting.run([step for sentence in sentences for step in (("say", sentence), ("turn",))])
    assert meeting.run([("say", sentences[0]), ("turn",)])[-1] == sentences[0]  # 20 sentences ago: said again


@pytest.mark.parametrize(("a", "b", "same"), [
    ("We need to finish the report.", "We have to finish the report!", True),
    ("The meeting starts at 3.", "the meeting starts at 3", True),
    ("I think so.", "I think not.", False),  # short: word for word only
    ("We start on Monday.", "We stop on Friday.", False),
    ("It costs 3 million this year.", "It costs 5 million this year.", False),  # another number: another sentence
])
def test_sentences_alike(a, b, same):
    assert speakers.alike(speakers.words(a), speakers.words(b)) is same


def test_speaking_lasts_while_the_voice_can_be_heard():
    clock = Clock()
    speaker, said = speaker_with(clock=clock)
    assert not speaker.speaking
    speaker.start()
    speaker.hear(LiveEvent(Kind.LINE, "Hello."))
    assert wait_until(lambda: said)
    time.sleep(0.05)
    assert speaker.speaking  # just played, and a microphone may still hear the room for a moment
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
    clock, ducker = Clock(), FakeDucker()
    speaker, said = speaker_with(clock=clock, ducker=ducker)
    speaker.start()
    assert ducker.started and not ducker.stopped  # lowered from the start, before a word is said
    speaker.hear(LiveEvent(Kind.LINE, "Hello."))
    assert wait_until(lambda: said)
    clock.now += 60  # a long pause: still lowered
    assert not ducker.stopped
    speaker.stop()
    assert ducker.stopped


def test_a_voice_that_cant_load_says_so_once():
    problems = []

    def broken():
        raise RuntimeError("model.onnx is damaged")
    speaker = Speaker(broken, FakePlayer(), [SYSTEM], on_problem=problems.append)
    speaker.start()
    assert wait_until(lambda: problems == ["The voice couldn't start: model.onnx is damaged"])
    speaker.stop()


def test_which_ways_a_voice_speaks():
    assert LiveConfig(target="en").spoken_lanes("en") == (SYSTEM,)
    assert LiveConfig(target="ja").spoken_lanes("en") == ()
    assert LiveConfig(source="microphone", mic_target="en").spoken_lanes("en") == (MIC,)
    both = LiveConfig(source="both", target="en", mic_target="en")
    assert both.spoken_lanes("en") == (SYSTEM,)  # with both, the user's own words are for the others


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


# ---- Gemini's own voice: played as it streams; the laptop's voice only stands in for it

class RecordingPlayer(FakePlayer):
    """Keeps what it plays, and how many plays ran at once (one voice at a time: never more than one)."""

    def __init__(self, **named):
        super().__init__(**named)
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

    @property
    def seconds(self):
        return sum(len(p) / r for p, r in zip(self.pieces, self.rates, strict=True))


def pcm(level: int, seconds: float = 0.1, rate: int = 24000) -> bytes:
    return np.full(round(seconds * rate), level, dtype="<i2").tobytes()


def test_geminis_voice_is_played_in_order_and_the_sentences_it_says_arent_said_again():
    player = RecordingPlayer()
    speaker, said = speaker_with(player=player, own=(SYSTEM,))
    speaker.start()
    pieces = [pcm(1000 * n) for n in range(1, 6)]
    for piece in pieces:
        speaker.hear_audio(SYSTEM, piece, 24000)
    speaker.hear(LiveEvent(Kind.LINE, "Thanks for joining. The budget is ready."))  # its transcript, after the audio
    assert wait_until(lambda: player.seconds >= 0.5)
    time.sleep(0.1)
    expected = np.concatenate([np.frombuffer(p, "<i2") for p in pieces]).astype(np.float32) / 32768
    assert np.array_equal(np.concatenate(player.pieces), expected) and set(player.rates) == {24000}
    assert said == [] and speaker.voiced == 1 and speaker.said == 0 and speaker.sound_s == pytest.approx(0.5)
    speaker.stop()


def test_a_sentence_geminis_voice_doesnt_say_is_said_by_the_laptops_voice():
    clock = Clock()
    speaker, said = speaker_with(clock=clock, own=(SYSTEM,))
    speaker.start()
    speaker.hear(LiveEvent(Kind.LINE, "Good morning."))
    time.sleep(0.2)
    assert said == []  # Gemini's voice may still come
    clock.now += speakers.AUDIO_WAIT
    assert wait_until(lambda: said == [("Good morning.", 1.0)]) and speaker.said == 1 and speaker.voiced == 0
    speaker.stop()


def test_geminis_voice_coming_in_time_takes_the_sentence_over():
    clock, player = Clock(), RecordingPlayer()
    speaker, said = speaker_with(clock=clock, player=player, own=(SYSTEM,))
    speaker.start()
    speaker.hear(LiveEvent(Kind.LINE, "Good morning."))
    clock.now += 0.5
    speaker.hear_audio(SYSTEM, pcm(2000), 24000)
    clock.now += 5
    assert wait_until(lambda: player.pieces)
    time.sleep(0.2)
    assert said == [] and speaker.voiced == 1
    speaker.hear(LiveEvent(Kind.LINE, "And welcome."))  # Gemini's voice came 5 s ago: still speaking, it says this too
    clock.now += 2
    time.sleep(0.2)
    assert said == [] and speaker.voiced == 2
    clock.now += speakers.OWN_RECENT  # Gemini's voice has stopped coming: the laptop's stands in
    speaker.hear(LiveEvent(Kind.LINE, "Any questions?"))
    clock.now += speakers.AUDIO_WAIT
    assert wait_until(lambda: said == [("Any questions?", 1.0)])
    speaker.stop()


def test_the_voice_is_speaking_while_geminis_voice_plays():
    from sst.live import wasapi
    clock = Clock()
    speaker = Speaker(lambda: None, Player(opener=lambda rate: FakeRender(rate)), [], clock=clock, own=(SYSTEM,))
    speaker.start()
    assert not speaker.speaking
    before = time.monotonic()
    speaker.hear_audio(SYSTEM, pcm(3000, seconds=1.0), 24000)
    assert wait_until(lambda: speaker.sound_s > 0)
    assert speaker.speaking  # the microphone's way, near speakers, gets silence now
    assert wasapi._voice_until >= before + 1.0  # the capture leaves Rflow's own voice out of the whole mix meanwhile
    clock.now += 1
    assert not speaker.speaking
    speaker.stop()


def test_never_two_voices_at_once():
    player = RecordingPlayer()
    speaker, said = speaker_with(lanes=(SYSTEM, MIC), player=player, own=(SYSTEM,))  # the microphone's: the laptop's
    speaker.start()

    def gemini():
        for n in range(20):
            speaker.hear_audio(SYSTEM, pcm(100 + n, seconds=0.05), 24000)
            time.sleep(0.003)
    thread = threading.Thread(target=gemini)
    thread.start()
    for n in range(5):
        speaker.hear(LiveEvent(Kind.LINE, f"Sentence number {n} for the room.", lane=MIC))
    thread.join()
    assert wait_until(lambda: len(said) == 5 and player.seconds >= 1.0 + 5 * 0.5 - 1e-6)
    assert player.most == 1
    speaker.stop()


def test_muting_and_stopping_silence_geminis_voice():
    gate = threading.Event()
    player = RecordingPlayer(gate=gate)
    speaker, _ = speaker_with(player=player, own=(SYSTEM,))
    speaker.start()
    for n in range(3):
        speaker.hear_audio(SYSTEM, pcm(1000 + n, seconds=0.6), 24000)  # each more than one piece played at once
    assert wait_until(lambda: player.started == 1)
    speaker.set_muted(True)  # the bar's speaker button: what's queued goes, what plays stops
    gate.set()
    speaker.hear_audio(SYSTEM, pcm(5), 24000)
    time.sleep(0.2)
    assert len(player.pieces) == 1 and player.interrupted
    speaker.set_muted(False)
    speaker.stop()
    speaker.hear_audio(SYSTEM, pcm(6), 24000)
    time.sleep(0.1)
    assert len(player.pieces) == 1 and player.closed


def test_only_the_ways_spoken_get_geminis_voice():
    player = RecordingPlayer()
    speaker, _ = speaker_with(player=player, own=(SYSTEM,))
    speaker.start()
    speaker.hear_audio(MIC, pcm(10), 24000)  # with Both, the user's own words are for the others
    time.sleep(0.2)
    assert player.pieces == []
    speaker.set_lanes([], own=[MIC])
    speaker.hear_audio(MIC, pcm(10), 24000)
    assert wait_until(lambda: len(player.pieces) == 1)
    speaker.stop()


def test_falling_behind_geminis_voice_skips_what_is_stale():
    gate, clock = threading.Event(), Clock()
    player = RecordingPlayer(gate=gate)
    speaker, _ = speaker_with(player=player, clock=clock, own=(SYSTEM,))
    speaker.start()
    speaker.hear_audio(SYSTEM, pcm(1000, seconds=0.5), 24000)
    assert wait_until(lambda: player.started == 1)
    speaker.hear_audio(SYSTEM, pcm(2000, seconds=0.5), 24000)  # waits behind it, then too long
    clock.now += speakers.STALE + 1
    speaker.hear_audio(SYSTEM, pcm(3000, seconds=0.5), 24000)
    gate.set()
    assert wait_until(lambda: len(player.pieces) == 2)
    assert [int(round(p[0] * 32768)) for p in player.pieces] == [1000, 3000]
    assert speaker.sound_skipped_s == pytest.approx(0.5)
    speaker.stop()


def test_pauses_are_shortened_to_catch_up_and_the_words_kept_whole():
    rate = 24000
    loud = np.full(rate // 5, 0.3, dtype=np.float32)
    samples = np.concatenate([loud, np.zeros(rate, np.float32), loud])
    shorter = speakers.shorten_pauses(samples, rate)
    assert len(shorter) == 2 * len(loud) + round(speakers.PAUSE_S * 50) * rate // 50  # the pause: 1 s -> 0.24 s
    assert np.count_nonzero(shorter) == 2 * len(loud)
    assert np.array_equal(speakers.shorten_pauses(loud, rate), loud)  # no pause: as it was


def test_geminis_voice_from_the_fake_server_is_heard_through_the_session():
    from sst.live.session import LiveSession
    from tests.test_live_gemini import KEY, FakeServer, spoken
    from tests.test_live_session import FakeCapture
    server, shown, said, clock = FakeServer(), [], [], Clock()
    player = RecordingPlayer()
    config = LiveConfig(speak=True)
    session = LiveSession(config, on_event=shown.append)
    session.add(SYSTEM, FakeCapture(), lambda emit: GeminiLiveTranslate(KEY, config, emit, connect=lambda url: server))
    session.set_speaker(Speaker(lambda: FakeVoice(said), player, config.spoken_lanes("en"), clock=clock,
                                own=config.spoken_lanes()))
    assert wait_until(lambda: server.sent)
    server.say(modelTurn=spoken(pcm(1200), pcm(1300)), outputTranscription={"text": "Hello there."})
    server.say(turnComplete=True)
    assert wait_until(lambda: player.seconds >= 0.2 - 1e-6)
    assert [e.text for e in shown if e.kind is Kind.LINE] == ["Hello there."]  # the caption as ever
    clock.now += speakers.OWN_RECENT + 1  # a while later Gemini sends a line without its voice
    server.say(outputTranscription={"text": "No voice for this one."}, turnComplete=True)
    assert wait_until(lambda: [e.text for e in shown if e.kind is Kind.LINE][-1:] == ["No voice for this one."])
    time.sleep(0.1)
    assert said == []  # its voice may still come
    clock.now += speakers.AUDIO_WAIT
    assert wait_until(lambda: said == [("No voice for this one.", 1.0)])
    session.stop()
    assert player.closed and session.speaker is None
