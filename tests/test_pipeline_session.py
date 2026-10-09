"""The voice pipeline end to end, with synthetic speech and fake speech engines: chunks cut while the user speaks,
results ordered and merged, then dictionary, formatting, LLM and guard (no model, no network, no microphone)."""
import threading
import time

import numpy as np
import pytest

from sst.audio import Take
from sst.dictate import Dictation
from sst.pipeline.asr import ASRScheduler, EngineBackend
from sst.pipeline.contracts import Provenance, RawTranscript, VoiceConfig, WordInfo
from sst.pipeline.dictionary import DictionaryEngine, DictionaryStore, TermMode
from sst.pipeline.formatting import Formatter
from sst.pipeline.guard import Guard
from sst.pipeline.session import LazyBackend, Stages, VoicePipeline
from sst.snippets import Snippet

RATE = 16_000


def speech(seconds: float, level: float = 0.05, seed: int = 0) -> np.ndarray:
    """Speech-like audio: noise shaped by a 4.5 Hz syllable rhythm."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.5 * t)
    return (rng.standard_normal(len(t)) * level * envelope).astype(np.float32)


def silence(seconds: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * RATE)) * 0.001).astype(np.float32)


class ScriptedBackend:
    """Answers each chunk with the next scripted text (by sequence), optionally slowly or failing."""
    name = "scripted"

    def __init__(self, texts, delays=None, fail=()):
        self.texts, self.delays, self.fail = texts, delays or {}, set(fail)
        self.calls = []

    def transcribe(self, chunk):
        self.calls.append(chunk.sequence)
        time.sleep(self.delays.get(chunk.sequence, 0))
        if chunk.sequence in self.fail:
            raise ValueError("the provider refused this audio")  # permanent: no retries
        text = self.texts[chunk.sequence - 1]
        n = len(text.split())
        words = [WordInfo(w, i * chunk.duration / max(n, 1), (i + 0.9) * chunk.duration / max(n, 1))
                 for i, w in enumerate(text.split())]
        return RawTranscript(text, words, backend=self.name)


class FakeLLM:
    name = "fake-llm"

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.seen = answer, error, []

    def polish(self, text, protected_terms=()):
        self.seen.append((text, list(protected_terms)))
        if self.error:
            raise RuntimeError(self.error)
        return self.answer(text) if callable(self.answer) else (self.answer if self.answer is not None else text)


def pipeline(backend, stages=None, **asr):
    config = VoiceConfig()
    for key, value in asr.items():
        setattr(config.asr, key, value)
    return VoicePipeline(ASRScheduler(backend, config.asr, sleep=lambda s: None), config, stages or Stages())


def two_sentences():
    # Each long enough to be a part of its own: a pause cuts after 4 s of speech, and a last part needs 2 s
    return np.concatenate([silence(0.3), speech(5.0), silence(1.6), speech(3.0, seed=3), silence(0.5)])


def test_a_pause_splits_the_dictation_and_the_parts_are_joined_in_order():
    backend = ScriptedBackend(["I want to order a pizza.", "Please deliver it tonight."], delays={1: 0.15})
    final = pipeline(backend, max_in_flight=2).process_audio(two_sentences(), RATE)
    assert sorted(backend.calls) == [1, 2]
    assert final.provenance is Provenance.FORMATTED
    assert final.text == "I want to order a pizza. Please deliver it tonight."
    assert final.stages["merged"] == final.text and final.metrics["chunks"] == 2


def test_chunks_go_to_the_engine_while_the_user_is_still_speaking():
    backend = ScriptedBackend(["first part", "second part"])
    session = pipeline(backend).start(RATE)
    audio = two_sentences()
    first_half = int(7.2 * RATE)  # through the pause: the first chunk is cut here
    for i in range(0, first_half, 800):
        session.feed(audio[i:min(i + 800, first_half)])
    for _ in range(100):
        if backend.calls:
            break
        time.sleep(0.01)
    assert backend.calls == [1]  # before the key was let go
    session.feed(audio[first_half:])
    session.finish()
    assert session.result().text == "first part second part"


def test_a_failed_part_fails_the_whole_dictation_rather_than_typing_a_hole():
    backend = ScriptedBackend(["one", "two"], fail={2})
    final = pipeline(backend).process_audio(two_sentences(), RATE)
    assert final.provenance is Provenance.FAILED and final.text == ""
    assert "part 2" in final.error


def test_parakeet_stands_in_for_a_failed_part_and_says_so():
    backend = ScriptedBackend(["one", "two"], fail={2})
    stand_in = ScriptedBackend(["one", "two from parakeet"])
    stand_in.name = "parakeet"
    config = VoiceConfig()
    scheduler = ASRScheduler(backend, config.asr, fallback=LazyBackend("parakeet", lambda: stand_in), sleep=lambda s: None)
    final = VoicePipeline(scheduler, config).process_audio(two_sentences(), RATE)
    assert final.text == "one two from parakeet" and final.notes and "parakeet" in final.notes[0].lower()


def test_the_text_stages_run_in_order_and_each_is_kept():
    store = DictionaryStore()
    store.add_term("PostgreSQL", aliases=["post grass"], mode=TermMode.CAREFUL)
    config = VoiceConfig()
    stages = Stages(dictionary=DictionaryEngine(store, config.dictionary), formatter=Formatter(config.formatting),
                    llm=FakeLLM(lambda text: text.replace("um, ", "").replace("Um, ", "")), guard=Guard(config.guard),
                    terms=store.hint_terms)
    backend = ScriptedBackend(["Um, the post grass upgrade is twenty five percent done."])
    final = pipeline(backend, stages).process_audio(np.concatenate([speech(2.5), silence(0.4)]), RATE)
    assert final.stages["dictionary"] == "Um, the PostgreSQL upgrade is twenty five percent done."
    assert final.stages["formatted"] == "Um, the PostgreSQL upgrade is 25% done."
    assert final.text == "the PostgreSQL upgrade is 25% done." and final.provenance is Provenance.LLM_POLISHED
    assert stages.llm.seen[0][1] == ["PostgreSQL"]  # only the terms in this text are protected


def test_the_guard_keeps_the_text_from_before_the_llm_when_it_changes_a_number():
    config = VoiceConfig()
    llm = FakeLLM(lambda text: text.replace("5", "10"))
    stages = Stages(formatter=Formatter(config.formatting), llm=llm, guard=Guard(config.guard))
    backend = ScriptedBackend(["I need 5 servers by Friday."])
    final = pipeline(backend, stages).process_audio(np.concatenate([speech(2.0), silence(0.4)]), RATE)
    assert final.text == "I need 5 servers by Friday." and final.provenance is Provenance.FORMATTED_FALLBACK
    assert final.guard is not None and not final.guard.accepted


def test_an_unreachable_llm_types_the_trusted_text_and_says_why():
    stages = Stages(llm=FakeLLM(error="the AI endpoint could not be reached"), guard=Guard())
    final = pipeline(ScriptedBackend(["hello there"]), stages).process_audio(
        np.concatenate([speech(1.0), silence(0.4)]), RATE)
    assert final.text == "hello there" and final.provenance is Provenance.FORMATTED_FALLBACK
    assert "could not be reached" in final.error


def test_silence_types_nothing_and_asks_the_engine_nothing():
    backend = ScriptedBackend([])
    final = pipeline(backend).process_audio(silence(3.0), RATE)
    assert final.text == "" and backend.calls == [] and final.provenance is not Provenance.FAILED


def test_the_debug_folder_keeps_every_stage_and_the_audio(tmp_path):
    p = pipeline(ScriptedBackend(["hello there"]))
    p.debug_dir, p.keep_audio = tmp_path, True
    final = p.process_audio(np.concatenate([speech(1.0), silence(0.4)]), RATE)
    folder = tmp_path / final.session_id
    assert (folder / "session.json").exists() and list(folder.glob("chunk_*.wav"))


def test_engine_backend_wraps_the_existing_engines():
    class OldEngine:
        name, title = "parakeet", "Parakeet"

        def transcribe(self, audio, rate):
            return "plain text"
    final = pipeline(EngineBackend(OldEngine())).process_audio(np.concatenate([speech(1.0), silence(0.4)]), RATE)
    assert final.text == "plain text"


# ---- the dictation: the live session fed from the recorder's take

class LiveRecorder:
    """Like the real Recorder: start() opens a take with a pre-roll; the test appends audio while the key is "held"."""
    rate = RATE

    def __init__(self, preroll: np.ndarray):
        self.preroll, self.current_take = preroll, None

    def start(self):
        self.current_take = Take([self.preroll.copy()], self.rate)

    def stop_later(self):
        take, self.current_take = self.current_take, None
        take.done.set()
        return take


def test_dictation_uses_the_pipeline_live_and_types_the_final_text():
    typed, states, results = [], [], []
    recorder = LiveRecorder(np.concatenate([silence(1.5), speech(0.3)]))  # the user started just before the key
    backend = ScriptedBackend(["send the report tomorrow"])
    d = Dictation(None, recorder, paste=typed.append, sounds=False, save=False)
    d.pipeline = pipeline(backend)
    d.on_state = lambda state, message: states.append(state)
    d.on_result = lambda heard, text, seconds: results.append((heard, text))
    d.handle("press", 0.0)
    recorder.current_take.chunks += [speech(1.5, seed=5), silence(0.4)]
    d.handle("release", 2.0)
    d.wait()
    assert typed == ["send the report tomorrow "] and states[-1] == "typed"
    assert results == [("send the report tomorrow", "send the report tomorrow")]


def test_a_failed_dictation_keeps_the_recording_for_a_retry(monkeypatch):
    from sst import dictate
    saved = []
    monkeypatch.setattr(dictate, "save_recording", lambda audio, rate, text: saved.append(len(audio)) or "x")
    typed, states = [], []
    recorder = LiveRecorder(silence(0.2))
    backend = ScriptedBackend(["hello again"], fail={1})
    d = Dictation(None, recorder, paste=typed.append, sounds=False, save=True)
    d.pipeline = pipeline(backend)
    d.on_state = lambda state, message: states.append((state, message))
    d.handle("press", 0.0)
    recorder.current_take.chunks += [speech(1.2), silence(0.4)]
    d.handle("release", 2.0)
    d.wait()
    assert typed == [] and states[-1][0] == "error" and "Nothing was typed" in states[-1][1] and saved
    assert d.last_failed is not None
    backend.fail.clear()
    assert d.retry_last()
    d.wait()
    assert typed == ["hello again "]


def test_overlapping_dictations_never_mix():
    gate = threading.Event()

    class SlowFirst(ScriptedBackend):
        def transcribe(self, chunk):
            if chunk.session_id == first.session_id:
                gate.wait(2)
            return RawTranscript(f"text of {chunk.session_id[-3:]}")
    p = pipeline(SlowFirst([]), max_in_flight=2)
    first = p.start(RATE)
    second = p.start(RATE)
    for session in (first, second):
        session.feed(np.concatenate([speech(1.0), silence(0.4)]))
        session.finish()
    second_text = second.result(5).text
    gate.set()
    first_text = first.result(5).text
    assert second_text.endswith(second.session_id[-3:]) and first_text.endswith(first.session_id[-3:])


@pytest.mark.parametrize("pause, chunks", [(0.3, 1), (1.6, 2)])
def test_only_a_real_pause_splits(pause, chunks):
    backend = ScriptedBackend(["a", "b"])
    audio = np.concatenate([speech(5.0), silence(pause), speech(3.0, seed=7), silence(0.5)])
    final = pipeline(backend).process_audio(audio, RATE)
    assert final.metrics["chunks"] == chunks



# ---- what the owner's log showed (2026-10-02): a short last part, a fragment, Rflow's own beep

def test_a_short_last_part_is_sent_again_with_the_part_before_it():
    backend = ScriptedBackend(["one two three four five six", "never said: version installer commit",
                               "one two three four five six seven"])
    audio = np.concatenate([silence(0.3), speech(6.0), silence(1.6), speech(0.5, seed=4), silence(0.4)])
    final = pipeline(backend).process_audio(audio, RATE)
    assert sorted(backend.calls) == [1, 2]  # the short rest went only as part of chunk 2, which replaces chunk 1
    assert final.text == "never said: version installer commit" and final.stages["merged"] == final.text


def test_a_replaced_part_is_dropped_when_the_result_is_awaited_before_the_last_part():
    # The app's worker asks for the result the moment the recording ends, while the feeder thread is still cutting the
    # last part, the one that replaces the part before it. The owner's dictation of 2026-10-04 came out twice.
    backend = ScriptedBackend(["one two three four five six", "one two three four five six seven"])
    audio = np.concatenate([silence(0.3), speech(6.0), silence(1.6), speech(0.5, seed=4), silence(0.4)])
    session = pipeline(backend).start(RATE)
    session.feed(audio)  # the first part is cut at the pause; the short rest waits for the key release
    waiting, wait = threading.Event(), session._asr.wait

    def asr_wait(timeout=None):
        waiting.set()
        return wait(timeout)

    session._asr.wait = asr_wait
    out = {}
    worker = threading.Thread(target=lambda: out.update(final=session.result()))
    worker.start()
    assert waiting.wait(5)  # result() is already waiting when the key-release part is cut
    session.finish()
    worker.join(5)
    assert sorted(backend.calls) == [1, 2]
    assert out["final"].text == "one two three four five six seven" and out["final"].metrics["chunks"] == 1


def test_a_superseded_part_that_failed_doesnt_fail_the_dictation():
    backend = ScriptedBackend(["", "hello there and more words"], fail={1})
    audio = np.concatenate([silence(0.3), speech(6.0), silence(1.6), speech(0.5, seed=4), silence(0.4)])
    final = pipeline(backend).process_audio(audio, RATE)
    assert final.provenance is not Provenance.FAILED and final.text == "hello there and more words"


def test_rflows_start_beep_is_filtered_out_of_the_recording():
    from sst.pipeline.session import _ToneNotch
    t = np.arange(int(0.6 * RATE)) / RATE
    beep = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    notch = _ToneNotch(RATE, 880, 0, len(beep))
    out = np.concatenate([notch(beep[i:i + 700], i) for i in range(0, len(beep), 700)])  # across blocks
    assert np.sqrt(np.mean(out[RATE // 10:] ** 2)) < 0.01 * np.sqrt(np.mean(beep ** 2))  # 40 dB down after it settles
    voice = speech(0.6)
    kept = _ToneNotch(RATE, 880, 0, len(voice))(voice, 0)
    assert np.sqrt(np.mean(kept ** 2)) > 0.9 * np.sqrt(np.mean(voice ** 2))  # the voice stays
    after = _ToneNotch(RATE, 880, 0, 100)(beep[:500], 1000)
    assert np.array_equal(after, beep[:500])  # outside its span nothing changes


def test_a_voice_command_skips_the_text_stages_and_types_nothing():
    llm = FakeLLM(lambda text: "Polished: " + text)
    stages = Stages(llm=llm, guard=Guard(), command=lambda text: "concise" if "concise" in text.lower() else None)
    backend = ScriptedBackend(["Make it concise."])
    final = pipeline(backend, stages).process_audio(np.concatenate([speech(2.5), silence(0.4)]), RATE)
    assert final.command == "concise" and final.text == "" and not llm.seen  # the LLM never saw it


def test_dictation_hands_a_voice_command_to_the_app():
    typed, states, commands = [], [], []
    recorder = LiveRecorder(silence(0.2))
    d = Dictation(None, recorder, paste=typed.append, sounds=False, save=False)
    d.pipeline = pipeline(ScriptedBackend(["Undo that."]), Stages(command=lambda text: "undo"))
    d.on_state = lambda state, message: states.append(state)
    d.on_command = commands.append
    d.handle("press", 0.0)
    recorder.current_take.chunks += [speech(1.5, seed=5), silence(0.4)]
    d.handle("release", 2.0)
    d.wait()
    assert commands == ["undo"] and typed == [] and states[-1] == "idle"


# ---- snippets (sst.snippets): "my email" types the user's own text

EMAIL = Snippet("my email", "xyz@gmail.com")
ADDRESS = Snippet("insert my address", "1-2-3 Shibuya,\nTokyo 150-0002", anywhere=True)


def snippet_stages(llm=None, **kwargs):
    return Stages(llm=llm, guard=Guard(), snippets=lambda: [EMAIL, ADDRESS], **kwargs)


def run(text, stages):
    return pipeline(ScriptedBackend([text]), stages).process_audio(np.concatenate([speech(2.5), silence(0.4)]), RATE)


def test_a_snippet_said_alone_is_typed_as_entered_and_skips_the_ai():
    llm = FakeLLM(lambda text: "Polished: " + text)
    final = run("My email.", snippet_stages(llm))
    assert final.text == "xyz@gmail.com" and not llm.seen and not final.command


def test_a_snippet_wins_over_a_voice_command_with_the_same_words():
    stages = Stages(command=lambda text: "concise", snippets=lambda: [Snippet("make it concise", "my own text")])
    final = run("Make it concise.", stages)
    assert final.text == "my own text" and final.command == ""


def test_inside_a_sentence_the_ai_sees_a_placeholder_and_the_text_goes_in_last():
    llm = FakeLLM(lambda text: text[0].upper() + text[1:] + ".")
    final = run("send the parcel to insert my address please", snippet_stages(llm))
    assert final.text == "Send the parcel to 1-2-3 Shibuya,\nTokyo 150-0002 please."  # its line break kept
    assert "Shibuya" not in llm.seen[0][0] and "RFSNIP1" in llm.seen[0][0]  # the address never went to the AI
    assert "RFSNIP1" in llm.seen[0][1]  # protected, like the user's terms
    assert final.provenance is Provenance.LLM_POLISHED


def test_an_ai_that_drops_the_placeholder_is_overruled_and_the_snippet_still_typed():
    llm = FakeLLM(lambda text: text.replace("RFSNIP1", "my place"))
    final = run("send the parcel to insert my address", snippet_stages(llm))
    assert final.provenance is Provenance.FORMATTED_FALLBACK  # the guard kept the text from before the AI
    assert final.text == "send the parcel to 1-2-3 Shibuya,\nTokyo 150-0002"


def test_without_snippets_nothing_changes():
    final = run("I checked my email this morning", snippet_stages())
    assert final.text == "I checked my email this morning"


def test_the_pipeline_says_which_microphone_gave_no_sound():
    typed, states = [], []
    recorder = LiveRecorder(np.zeros(RATE // 5, dtype=np.float32))
    recorder.describe = lambda: {"device": "Headset (Buds)"}
    backend = ScriptedBackend(["never asked"])
    d = Dictation(None, recorder, paste=typed.append, sounds=False, save=False)
    d.pipeline = pipeline(backend)
    d.on_state = lambda state, message: states.append((state, message))
    d.handle("press", 0.0)
    recorder.current_take.chunks += [np.zeros(int(1.5 * RATE), dtype=np.float32)]
    d.handle("release", 2.0)
    d.wait()
    assert typed == [] and backend.calls == []
    assert states[-1][0] == "error" and "(Headset (Buds))" in states[-1][1]


# ---- the wake word: its keyword and end phrase are trimmed from the words heard, before everything else

def _wake_trim(text: str) -> str:
    return text.removeprefix("Okay type, ").removesuffix(" Stop it.")


def test_a_wake_word_turn_types_neither_the_keyword_nor_the_end_phrase():
    llm = FakeLLM(lambda text: text[0].upper() + text[1:])
    typed, results = [], []
    recorder = LiveRecorder(np.concatenate([silence(0.2), speech(0.6, seed=7)]))  # the keyword, before the start
    d = Dictation(None, recorder, paste=typed.append, sounds=False, save=False)
    d.pipeline = pipeline(ScriptedBackend(["Okay type, send the report. Stop it."]), Stages(llm=llm, guard=Guard()))
    d.on_result = lambda heard, text, seconds: results.append((heard, text))
    d.trim = _wake_trim
    d.handle("handsfree", 10.0)
    d.trim = None
    recorder.current_take.chunks += [speech(1.5, seed=5), silence(0.4)]
    d.end(keep=True)
    d.wait()
    assert llm.seen[0][0] == "send the report." and typed == ["Send the report. "]
    assert results == [("send the report.", "Send the report.")]  # the history's "heard" has no keyword either


def test_a_command_after_the_keyword_reaches_the_app_and_a_retry_trims_too(monkeypatch):
    from sst import dictate
    monkeypatch.setattr(dictate, "save_recording", lambda audio, rate, text: "x")
    commands, typed = [], []
    recorder = LiveRecorder(silence(0.2))
    backend = ScriptedBackend(["Okay type, make it concise."])
    d = Dictation(None, recorder, paste=typed.append, sounds=False, save=True)
    d.pipeline = pipeline(backend, Stages(command=lambda text: "concise" if text == "make it concise." else None))
    d.on_command = commands.append
    d.trim = _wake_trim
    d.handle("handsfree", 10.0)
    d.trim = None
    recorder.current_take.chunks += [speech(1.5, seed=5), silence(0.4)]
    d.end(keep=True)
    d.wait()
    assert commands == ["concise"] and typed == []
    backend.texts, backend.fail = ["Okay type, hello. Stop it."], {1}  # the next one fails, then is retried
    d.trim = _wake_trim
    d.handle("handsfree", 15.0)
    d.trim = None
    recorder.current_take.chunks += [speech(1.5, seed=5), silence(0.4)]
    d.end(keep=True)
    d.wait()
    assert d.last_failed is not None and typed == []
    backend.fail.clear()
    assert d.retry_last()
    d.wait()
    assert typed == ["hello. "]
