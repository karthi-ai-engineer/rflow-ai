"""The voice pipeline's audio stages: the ring buffer, the VAD, the chunker and the pre-roll trim (plan sections 7-16 and
93-97). Signals are synthetic and seeded: "speech" is band-limited noise plus a gliding harmonic tone whose loudness
swells and falls with syllables (4.5 per second), over a room's white noise; nothing touches a microphone."""
import threading

import numpy as np
import pytest

from sst.pipeline.audio_stream import VAD, Chunker, RingBuffer, VadResult, speech_segments, trim_preroll
from sst.pipeline.contracts import AudioChunk, ChunkingConfig, VadConfig

RATE = 16_000
SYLLABLES = 4.5  # per second
# The cutting mechanics are tested with short sentences: a pause may cut after 0.4 s of speech, and a short rest is
# sent alone. The defaults (4 s, 2 s) are tested in their own tests below.
VAD_CONFIG, CHUNKING = VadConfig(), ChunkingConfig(min_cut_speech_ms=400, min_alone_speech_ms=0)
MAX = CHUNKING.max_duration_ms / 1000
LONGEST = (CHUNKING.max_duration_ms + CHUNKING.boundary_lookahead_ms) / 1000
OVERLAP = CHUNKING.overlap_ms / 1000


def _amplitude(db: float) -> float:
    return 10 ** (db / 20)


def noise(seconds: float, db: float = -60.0, seed: int = 0, rate: int = RATE) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(round(seconds * rate)) * _amplitude(db)).astype(np.float32)


def speech(seconds: float, db: float = -30.0, seed: int = 1, rate: int = RATE, depth: float = 1.0) -> np.ndarray:
    """Whole syllables (so it starts and ends quietly, like words) at `db` dBFS RMS. depth 1: silent between syllables;
    lower: never quiet."""
    n = round(max(1, round(seconds * SYLLABLES)) / SYLLABLES * rate)
    t = np.arange(n) / rate
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / rate)
    spectrum[(freqs < 200) | (freqs > 3500)] = 0
    hiss = np.fft.irfft(spectrum, n)
    phase = 2 * np.pi * np.cumsum(120 + 20 * np.sin(2 * np.pi * 0.7 * t)) / rate
    voice = sum(np.sin(k * phase) / k for k in range(1, 9))
    x = (hiss / hiss.std() + voice / voice.std()) * (1 - depth + depth * np.sin(np.pi * SYLLABLES * t) ** 2)
    return (x / np.sqrt(np.mean(x ** 2)) * _amplitude(db)).astype(np.float32)


def in_room(x: np.ndarray, db: float = -60.0, seed: int = 0) -> np.ndarray:
    return (x + noise(len(x) / RATE, db, seed)).astype(np.float32)


def scene(*parts, floor_db: float = -60.0, rate: int = RATE, seed: int = 0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """("speech", seconds[, db]) and ("pause", seconds) in a row, over a room's noise at floor_db.
    Returns the audio and the speech spans (samples)."""
    pieces, spans, pos = [], [], 0
    for i, (kind, seconds, *db) in enumerate(parts):
        piece = speech(seconds, *db, seed=seed + i + 1, rate=rate) if kind == "speech" else np.zeros(round(seconds * rate))
        if kind == "speech":
            spans.append((pos, pos + len(piece)))
        pieces.append(piece)
        pos += len(piece)
    return (np.concatenate(pieces) + noise(pos / rate, floor_db, seed, rate)).astype(np.float32), spans


def burst(audio: np.ndarray, at: float, seconds: float, db: float, rate: int = RATE, seed: int = 9) -> None:
    """Add a flat burst of noise (a click, a key, a cough) in place."""
    start, n = round(at * rate), max(1, round(seconds * rate))
    audio[start:start + n] += noise(n / rate, db, seed, rate)[:n]


def chunk_all(audio: np.ndarray, rate: int = RATE, blocks=None, config: ChunkingConfig = CHUNKING,
              vad_config: VadConfig = VAD_CONFIG) -> list[AudioChunk]:
    chunker = Chunker("sess_test", rate, config, vad_config)
    out, pos = [], 0
    for size in blocks or [len(audio)]:
        out += chunker.push(audio[pos:pos + size])
        pos += size
    out += chunker.push(audio[pos:])
    return out + chunker.finish()


def describe(chunks: list[AudioChunk]) -> list[tuple]:
    return [(c.sequence, c.start_sample, c.end_sample, c.overlap_end_sample, c.speech_seconds, c.boundary) for c in chunks]


def seconds(spans) -> list[tuple[float, float]]:
    return [(s / RATE, e / RATE) for s, e in spans]


# ---------------------------------------------------------------- ring buffer

def test_ring_buffer_keeps_the_newest_samples_across_wraparound():
    ring = RingBuffer(10)
    ring.write(np.arange(7))
    ring.write(np.arange(7, 14))  # wraps
    np.testing.assert_array_equal(ring.snapshot_last(10), np.arange(4, 14))
    np.testing.assert_array_equal(ring.snapshot_last(3), [11, 12, 13])
    assert (ring.write_index, ring.oldest_index, len(ring)) == (14, 4, 10)
    assert ring.snapshot_last(10).dtype == np.float32


def test_a_write_larger_than_the_buffer_keeps_its_end_and_counts_every_sample():
    ring = RingBuffer(10)
    ring.write(np.arange(3))
    ring.write(np.arange(3, 28))
    assert (ring.write_index, ring.oldest_index) == (28, 18)
    np.testing.assert_array_equal(ring.snapshot_last(100), np.arange(18, 28))
    ring.write(np.arange(28, 33))  # and the next small write still lands in order
    np.testing.assert_array_equal(ring.snapshot_last(10), np.arange(23, 33))


def test_ranges_are_absolute_and_clipped_to_what_is_still_held():
    ring = RingBuffer(8)
    ring.write(np.arange(20))
    np.testing.assert_array_equal(ring.get_range(13, 17), [13, 14, 15, 16])
    np.testing.assert_array_equal(ring.get_range(10, 15), [12, 13, 14])  # 10 and 11 were overwritten
    np.testing.assert_array_equal(ring.get_range(18, 30), [18, 19])  # the future isn't there yet
    np.testing.assert_array_equal(ring.get_range(14, 20), np.arange(14, 20))  # across the wrap
    for start, end in [(0, 5), (25, 30), (15, 15), (17, 12)]:
        assert len(ring.get_range(start, end)) == 0


def test_reads_are_copies_and_empty_reads_are_safe():
    ring = RingBuffer(4)
    assert len(ring.snapshot_last(3)) == 0 and len(ring.get_range(0, 3)) == 0 and ring.write_index == 0
    ring.write([1.0, 2.0])
    snap = ring.snapshot_last(2)
    snap[:] = 0
    np.testing.assert_array_equal(ring.snapshot_last(2), [1.0, 2.0])
    assert len(ring.snapshot_last(0)) == 0 and len(ring.snapshot_last(-5)) == 0
    ring.write(np.zeros(0, np.float32))
    assert ring.write_index == 2


def test_random_writes_match_a_plain_recording():
    rng = np.random.default_rng(5)
    ring, everything, index = RingBuffer(1000), [], 0
    for _ in range(300):
        block = rng.standard_normal(int(rng.integers(0, 2500))).astype(np.float32)
        ring.write(block)
        everything.append(block)
        index += len(block)
        assert ring.write_index == index  # monotonic: it only ever adds
        if index:
            whole = np.concatenate(everything)
            n = int(rng.integers(0, 1200))
            np.testing.assert_array_equal(ring.snapshot_last(n), whole[-min(n, 1000):] if n else whole[:0])
            a, b = sorted(int(v) for v in rng.integers(max(0, index - 1500), index + 10, 2))
            lo = max(a, index - 1000)
            np.testing.assert_array_equal(ring.get_range(a, b), whole[lo:min(b, index)] if b > lo else whole[:0])


def test_a_sounddevice_block_is_taken_as_mono():
    ring = RingBuffer(6)
    ring.write(np.arange(3, dtype=np.float32).reshape(3, 1))  # (frames, 1)
    ring.write(np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32))  # two channels: averaged
    np.testing.assert_array_equal(ring.snapshot_last(5), [0, 1, 2, 2, 6])


def test_readers_on_other_threads_always_see_consistent_audio():
    ring, stop, errors = RingBuffer(4096), threading.Event(), []

    def read():
        while not stop.is_set():
            end = ring.write_index
            for got, first in ((ring.snapshot_last(4096), None), (ring.get_range(end - 3000, end), end - 3000)):
                if len(got) and (np.any(np.diff(got) != 1) or (first is not None and got[0] < first)):
                    errors.append(got[:5])

    readers = [threading.Thread(target=read) for _ in range(2)]
    for r in readers:
        r.start()
    rng, value = np.random.default_rng(2), 0
    for _ in range(3000):  # a ramp: every sample's value is its absolute index
        n = int(rng.integers(1, 1500))
        ring.write(np.arange(value, value + n, dtype=np.float32))
        value += n
    stop.set()
    for r in readers:
        r.join()
    assert not errors and ring.write_index == value
    np.testing.assert_array_equal(ring.snapshot_last(4096), np.arange(value - 4096, value))


def test_a_ring_buffer_needs_room():
    with pytest.raises(ValueError):
        RingBuffer(0)


# ---------------------------------------------------------------- VAD

def test_vad_timeline_follows_speech_short_gaps_pauses_and_long_pauses():
    audio, truth = scene(("speech", 1.0), ("pause", 1.0), ("speech", 1.0), ("pause", 0.1), ("speech", 1.0),
                         ("pause", 0.4), ("speech", 1.0), ("pause", 1.5), ("speech", 0.8), ("pause", 0.5))
    a, b, c, d, e = seconds(truth)
    expected = [a, (b[0], c[1]), d, e]  # the 100 ms gap is bridged, the 0.4 s pause is not
    got = seconds(speech_segments(audio, RATE, VAD_CONFIG))
    assert len(got) == len(expected)
    for (start, end), (true_start, true_end) in zip(got, expected, strict=True):
        assert true_start <= start <= true_start + 0.08 and true_end - 0.08 <= end <= true_end + 0.02


def test_vad_state_holds_through_the_hangover_and_then_lets_go():
    audio, truth = scene(("pause", 0.5), ("speech", 1.0), ("pause", 0.1), ("speech", 1.0), ("pause", 1.0))
    states = VAD(RATE, VAD_CONFIG).process_block(audio)
    at = lambda t: states[int(t / 0.02)]  # noqa: E731
    (_, end1), (start2, end2) = seconds(truth)
    assert not at(0.3) and at(0.8) and at(end1 + 0.05) and at(start2 + 0.3)
    assert at(end2 + 0.1)  # the hangover: word endings are kept...
    assert not at(end2 + 0.35) and not at(end2 + 0.9)  # ...but not for longer than hangover_ms


def test_clicks_and_keys_do_not_start_speech():
    audio = noise(4.0)
    rng = np.random.default_rng(3)
    for k in range(15):  # single-sample clicks, 5 ms and 15 ms knocks at any frame alignment
        start = 0.2 + k * 0.25 + rng.uniform(0, 0.02)
        burst(audio, start, [1 / RATE, 0.005, 0.015][k % 3], -6.0, seed=k)
    vad = VAD(RATE, VAD_CONFIG)
    assert not any(vad.process_block(audio))
    assert speech_segments(audio, RATE, VAD_CONFIG) == []


def test_speech_starts_only_after_min_speech_ms():
    short, long = noise(1.0), noise(1.0)
    burst(short, 0.5, 0.04, -20.0)  # 40 ms, on frame boundaries: two frames
    burst(long, 0.5, 0.1, -20.0)
    assert speech_segments(short, RATE, VAD_CONFIG) == []
    (start, end), = seconds(speech_segments(long, RATE, VAD_CONFIG))
    assert start == pytest.approx(0.5, abs=0.02) and end == pytest.approx(0.6, abs=0.02)  # backdated to its onset


def test_quiet_laptop_speech_over_a_quiet_room_is_heard():
    for db in (-35, -42, -45):
        audio, truth = scene(("pause", 1.0), ("speech", 2.0, db), ("pause", 1.0), ("speech", 1.0, db), ("pause", 0.5))
        got = seconds(speech_segments(audio, RATE, VAD_CONFIG))
        assert len(got) == 2, db
        for (start, end), (true_start, true_end) in zip(got, seconds(truth), strict=True):
            assert abs(start - true_start) < 0.1 and abs(end - true_end) < 0.1, db


def test_a_louder_room_raises_the_floor_without_hiding_speech():
    audio, truth = scene(("pause", 1.0), ("speech", 2.0, -25), ("pause", 1.5), ("speech", 1.0, -25), floor_db=-45)
    got = seconds(speech_segments(audio, RATE, VAD_CONFIG))
    assert len(got) == 2
    for (start, end), (true_start, true_end) in zip(got, seconds(truth), strict=True):
        assert abs(start - true_start) < 0.1 and abs(end - true_end) < 0.1
    vad = VAD(RATE, VAD_CONFIG)
    assert not any(vad.process_block(noise(5.0, -45.0)))  # the noisy room alone, from the very first frame
    assert vad.noise_floor_db == pytest.approx(-45, abs=1.5)


def test_the_noise_floor_follows_a_fan_turned_on_and_off():
    vad = VAD(RATE, VAD_CONFIG)
    assert not any(vad.process_block(noise(2.0, -60, seed=1)))
    loud = vad.process_block(noise(4.0, -40, seed=2))
    assert sum(loud) * 0.02 < 0.8 and not any(loud[50:])  # a moment's doubt at the change, then it is noise
    assert vad.noise_floor_db == pytest.approx(-40, abs=1.5)
    assert any(vad.process_block(in_room(speech(1.0, -22), -40, seed=3)))  # speech over the fan
    vad.process_block(noise(1.0, -40, seed=4))
    assert not any(vad.process_block(noise(0.5, -60, seed=5)))  # fan off: the floor drops at once...
    assert vad.noise_floor_db < -58
    assert any(vad.process_block(in_room(speech(1.0, -45), seed=6)))  # ...and quiet speech counts again


def test_long_speech_without_real_gaps_is_not_learnt_as_noise():
    states = VAD(RATE, VAD_CONFIG).process_block(in_room(speech(30.0, -30, depth=0.75)))  # never 12 dB below its peaks
    assert all(states[10:-15])


def test_dc_offset_changes_nothing():
    audio, _ = scene(("pause", 0.5), ("speech", 2.0, -40), ("pause", 1.0), ("speech", 1.0, -40))
    plain = speech_segments(audio, RATE, VAD_CONFIG)
    assert speech_segments(audio + np.float32(0.05), RATE, VAD_CONFIG) == plain and len(plain) == 2


def test_digital_silence_and_broken_frames_are_safe():
    vad = VAD(RATE, VAD_CONFIG)
    results = [vad.process(np.zeros(320, np.float32)) for _ in range(100)]
    assert all(r == VadResult(0.0, False, -120.0) for r in results)
    assert vad.noise_floor_db == -60.0  # zeros (a device starting up) teach the floor nothing...
    assert not any(vad.process_block(noise(2.0, -60)))  # ...so the room's noise that follows isn't speech
    assert vad.process(np.full(320, np.nan, np.float32)) == VadResult(0.0, False, -120.0)
    assert vad.process(np.zeros(0, np.float32)).level_db == -120.0
    assert any(vad.process_block(in_room(speech(1.0, -40), seed=7)))  # nothing was poisoned
    assert np.isfinite(vad.noise_floor_db)


def test_probability_rises_with_level_and_ignores_anything_below_the_absolute_floor():
    def probability(db, floor):
        return VAD(RATE, VAD_CONFIG, noise_floor_db=floor).process(noise(0.02, db, seed=3)).probability

    levels = [-58, -54, -51, -48, -44, -30]
    p = [probability(db, -60) for db in levels]
    assert all(0 <= a < b <= 1 for a, b in zip(p, p[1:], strict=False))
    assert p[0] < 0.1 and p[2] == pytest.approx(0.5, abs=0.15) and p[-1] > 0.99  # 0.5 at floor + margin
    assert probability(-64, -80) == 0.0  # well above the floor, but quieter than absolute_floor_db


def test_frame_decisions_do_not_depend_on_block_sizes():
    audio, _ = scene(("pause", 0.7), ("speech", 1.5, -40), ("pause", 0.5), ("speech", 1.0))
    whole = VAD(RATE, VAD_CONFIG).process_block(audio)
    vad, pieces, pos, rng = VAD(RATE, VAD_CONFIG), [], 0, np.random.default_rng(4)
    while pos < len(audio):
        n = int(rng.integers(1, 3000))
        pieces += vad.process_block(audio[pos:pos + n])
        pos += n
    assert pieces == whole and len(whole) == len(audio) // 320


@pytest.mark.parametrize("rate", [8_000, 44_100, 48_000])
def test_vad_works_at_any_microphone_rate(rate):
    audio, truth = scene(("pause", 0.5), ("speech", 1.5, -40), ("pause", 1.0), ("speech", 1.0), rate=rate)
    got = speech_segments(audio, rate, VAD_CONFIG)
    assert VAD(rate, VAD_CONFIG).frame == rate // 50 and len(got) == 2
    for (start, end), (true_start, true_end) in zip(got, truth, strict=True):
        assert abs(start - true_start) / rate < 0.1 and abs(end - true_end) / rate < 0.1


# ---------------------------------------------------------------- chunker

def test_45_seconds_of_speech_become_about_three_chunks_cut_between_syllables():
    audio, ((speech_start, _),) = scene(("pause", 0.5), ("speech", 45.0), ("pause", 0.5))
    chunks = chunk_all(audio)
    assert [c.boundary for c in chunks] == ["max", "max", "end"] and [c.sequence for c in chunks] == [1, 2, 3]
    for c in chunks:
        assert c.duration <= LONGEST and c.session_id == "sess_test" and c.rate == RATE
        np.testing.assert_array_equal(c.audio, audio[c.start_sample:c.end_sample])
    for c in chunks[:-1]:
        assert MAX - CHUNKING.boundary_grace_ms / 1000 <= c.duration
        phase = ((c.end_sample - speech_start) / RATE * SYLLABLES) % 1  # 0 or 1 = between two syllables
        assert min(phase, 1 - phase) < 0.1
    assert chunks[-1].end_sample == len(audio)


def test_overlap_positions_chain_the_chunks():
    audio, _ = scene(("pause", 0.5), ("speech", 45.0), ("pause", 0.5))
    first, second, third = chunk_all(audio)
    assert first.start_sample == 0 and first.overlap_end_sample == first.start_sample
    for previous, chunk in ((first, second), (second, third)):
        assert chunk.start_sample == previous.end_sample - round(OVERLAP * RATE)
        assert chunk.overlap_end_sample == previous.end_sample
        np.testing.assert_array_equal(chunk.audio[:chunk.overlap_end_sample - chunk.start_sample],
                                      previous.audio[chunk.start_sample - previous.start_sample:])


def test_a_long_pause_ends_a_chunk_and_the_next_starts_at_once():
    audio, truth = scene(("pause", 0.5), ("speech", 3.0), ("pause", 1.2), ("speech", 3.0), ("pause", 0.5))
    (_, end1), (start2, _) = truth
    chunker = Chunker("sess_test", RATE, CHUNKING, VAD_CONFIG)
    first = chunker.push(audio[:start2])  # the first chunk is ready before the second sentence begins
    rest = chunker.push(audio[start2:]) + chunker.finish()
    assert len(first) == 1 and len(rest) == 1
    (a,), (b,) = first, rest
    assert a.boundary == "pause" and b.boundary == "end" and (a.sequence, b.sequence) == (1, 2)
    assert 0.15 <= (a.end_sample - end1) / RATE <= 0.35  # keep_silence_ms after the last speech
    # After a pause the next chunk starts at the cut: an overlap would reach back into the last word (see _cut)
    assert b.start_sample == a.end_sample and b.overlap_end_sample == b.start_sample
    assert b.end_sample == len(audio)


def test_quiet_speech_is_chunked_like_loud_speech():
    audio, _ = scene(("pause", 0.5), ("speech", 3.0, -42), ("pause", 1.2), ("speech", 3.0, -42), ("pause", 0.5))
    assert [c.boundary for c in chunk_all(audio)] == ["pause", "end"]
    noisy, _ = scene(("pause", 0.5), ("speech", 3.0, -25), ("pause", 1.2), ("speech", 3.0, -25), floor_db=-45)
    assert [c.boundary for c in chunk_all(noisy)] == ["pause", "end"]


@pytest.mark.parametrize("pause", [0.1, 0.3, 0.6, 1.0])
def test_a_pause_shorter_than_the_boundary_keeps_one_chunk(pause):
    audio, _ = scene(("pause", 0.5), ("speech", 3.0), ("pause", pause), ("speech", 3.0), ("pause", 0.5))
    (chunk,) = chunk_all(audio)
    assert (chunk.start_sample, chunk.end_sample, chunk.boundary) == (0, len(audio), "end")


def test_a_lone_hello_is_one_chunk_when_the_key_is_released():
    for db in (-30, -42):
        audio, _ = scene(("pause", 1.0), ("speech", 0.45, db), ("pause", 0.6))
        chunker = Chunker("sess_test", RATE, CHUNKING, VAD_CONFIG)
        assert chunker.push(audio) == []
        (chunk,) = chunker.finish()
        assert chunk.boundary == "end" and chunk.sequence == 1 and chunk.end_sample == len(audio) == chunker.samples_seen
        assert 0.3 <= chunk.speech_seconds <= 0.45


@pytest.mark.parametrize("make", [lambda: noise(30.0), lambda: noise(10.0, -45), lambda: np.zeros(5 * RATE, np.float32),
                                  lambda: noise(10.0, -75)])
def test_silence_alone_sends_nothing(make):
    assert chunk_all(make()) == []
    assert chunk_all(make(), config=ChunkingConfig(min_speech_ms=0)) == []  # even with no minimum, silence isn't speech


def test_a_cough_is_not_sent_and_sequence_numbers_stay_contiguous():
    audio, truth = scene(("pause", 0.5), ("speech", 2.0), ("pause", 3.0), ("pause", 3.0), ("speech", 2.0), ("pause", 0.5))
    burst(audio, 4.6, 0.1, -25.0)  # in the long pause: loud enough to start the VAD, too short to send
    (cough,) = [s for s in seconds(speech_segments(audio, RATE, VAD_CONFIG)) if 4 < s[0] < 5]
    assert cough[1] - cough[0] < CHUNKING.min_speech_ms / 1000
    chunks = chunk_all(audio)
    assert [c.sequence for c in chunks] == [1, 2] and [c.boundary for c in chunks] == ["pause", "end"]
    assert chunks[1].start_sample > round(4.7 * RATE)  # the cough faded out with the silence around it
    assert all(c.speech_seconds >= CHUNKING.min_speech_ms / 1000 for c in chunks)


def test_a_long_pause_does_not_use_up_the_next_chunks_length():
    audio, truth = scene(("pause", 0.5), ("speech", 2.0), ("pause", 8.0), ("speech", 17.0), ("pause", 0.5))
    first, second = chunk_all(audio)
    start2 = truth[1][0] / RATE
    assert first.boundary == "pause" and second.boundary == "end"  # 17 s of speech still fit in one chunk
    assert start2 - 1.25 <= second.start <= start2 - 1.0  # pause_boundary_ms of silence before it, not 8 s
    assert second.overlap_end_sample == second.start_sample  # nothing repeats from the first chunk


def test_every_word_lands_in_a_chunk():
    audio, truth = scene(("pause", 0.3), ("speech", 4.0), ("pause", 1.3), ("speech", 22.0), ("pause", 0.25),
                         ("speech", 1.0), ("pause", 5.0), ("speech", 0.5), ("pause", 2.0), ("speech", 3.0, -42))
    chunks = chunk_all(audio)
    covered = np.zeros(len(audio), bool)
    for c in chunks:
        covered[c.start_sample:c.end_sample] = True
        assert c.duration <= LONGEST
    assert all(covered[s:e].all() for s, e in truth)
    assert [c.sequence for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(b.start_sample > a.start_sample and b.end_sample > a.end_sample for a, b in zip(chunks, chunks[1:], strict=False))


def test_speech_seconds_count_the_speech_heard():
    audio, truth = scene(("pause", 1.0), ("speech", 2.0), ("pause", 0.5), ("speech", 1.0), ("pause", 1.0))
    (chunk,) = chunk_all(audio)
    true_speech = sum(e - s for s, e in truth) / RATE
    assert true_speech - 0.15 <= chunk.speech_seconds <= true_speech + 0.5  # + the short pause the hangover bridges


@pytest.mark.parametrize("gap_at, cut_in_gap", [(17.0, False), (18.5, True), (19.6, True), (20.2, True)])
def test_the_maximum_cut_goes_to_the_quietest_place_in_reach(gap_at, cut_in_gap):
    audio = speech(25.0, depth=0.75)  # never quiet, except one 120 ms gap
    start = round(gap_at * RATE)
    audio[start:start + round(0.12 * RATE)] = 0
    first = chunk_all(in_room(audio))[0]
    assert first.boundary == "max" and MAX - 2.0 <= first.duration <= LONGEST
    assert (gap_at < first.end < gap_at + 0.12) == cut_in_gap


def test_same_audio_in_any_block_sizes_gives_the_same_chunks():
    audio, _ = scene(("pause", 0.4), ("speech", 3.0), ("pause", 1.25), ("speech", 23.0), ("pause", 0.3),
                     ("speech", 2.0, -42), ("pause", 4.0), ("speech", 1.0), ("pause", 0.7))
    burst(audio, 31.0, 0.1, -25.0)
    reference = chunk_all(audio)
    assert [c.boundary for c in reference] == ["pause", "max", "pause", "end"]
    rng = np.random.default_rng(8)
    for blocks in ([320] * (len(audio) // 320), list(rng.integers(1, 30_000, 400)), list(rng.integers(1, 700, 4000))):
        chunks = chunk_all(audio, blocks=[int(b) for b in blocks])
        assert describe(chunks) == describe(reference)
        for a, b in zip(chunks, reference, strict=True):
            np.testing.assert_array_equal(a.audio, b.audio)


def test_one_sample_at_a_time_gives_the_same_chunks():
    audio, _ = scene(("pause", 0.3), ("speech", 1.0), ("pause", 1.3), ("speech", 0.5), ("pause", 0.2))
    reference = chunk_all(audio)
    assert len(reference) == 2
    assert describe(chunk_all(audio, blocks=[1] * len(audio))) == describe(reference)


def test_memory_stays_bounded_in_a_long_session():
    audio, _ = scene(("speech", 50.0), ("pause", 30.0), ("speech", 25.0), ("pause", 15.0))
    chunker = Chunker("sess_test", RATE, CHUNKING, VAD_CONFIG)
    limit = round(LONGEST * RATE) + 320
    block = RATE // 4
    for pos in range(0, len(audio), block):
        chunker.push(audio[pos:pos + block])
        assert chunker.retained <= limit and chunker._buf.size <= 2 * (limit + 640)
    assert chunker.retained <= round(CHUNKING.pause_boundary_ms / 1000 * RATE) + 320  # silence: only the lead-in is kept
    chunker.finish()
    assert chunker._buf.size == 0


def test_a_session_at_48_khz_is_cut_at_the_same_moments():
    parts = (("pause", 0.5), ("speech", 3.0), ("pause", 1.2), ("speech", 22.0), ("pause", 0.5))
    at_16k = chunk_all(scene(*parts)[0])
    at_48k = chunk_all(scene(*parts, rate=48_000)[0], rate=48_000)
    assert [c.boundary for c in at_48k] == [c.boundary for c in at_16k] == ["pause", "max", "end"]
    assert abs(at_16k[0].end - at_48k[0].end) < 0.05  # the pause cut; the maximum's goes to one of many quiet troughs
    for c in at_48k:
        assert c.rate == 48_000 and len(c.audio) == c.end_sample - c.start_sample and c.duration <= LONGEST
    assert at_48k[1].start_sample == at_48k[0].end_sample  # after the pause: no overlap
    assert at_48k[2].start_sample == at_48k[1].end_sample - 48_000  # after the maximum: 1 s of overlap


def test_the_configuration_is_respected():
    config = ChunkingConfig(pause_boundary_ms=600, max_duration_ms=5000, overlap_ms=500, boundary_grace_ms=1000,
                            boundary_lookahead_ms=200, keep_silence_ms=100, min_cut_speech_ms=400, min_alone_speech_ms=0)
    audio, truth = scene(("pause", 0.3), ("speech", 2.0), ("pause", 0.8), ("speech", 12.0), ("pause", 0.3))
    chunks = chunk_all(audio, config=config)
    assert chunks[0].boundary == "pause" and {c.boundary for c in chunks[1:-1]} == {"max"} and len(chunks) >= 4
    assert 0.05 <= (chunks[0].end_sample - truth[0][1]) / RATE <= 0.15  # keep_silence_ms
    for previous, chunk in zip(chunks[1:], chunks[2:], strict=False):
        assert chunk.start_sample == previous.end_sample - RATE // 2
    assert all(c.duration <= 5.2 for c in chunks) and all(4.0 <= c.duration for c in chunks[1:-1])


def test_a_nonsensical_configuration_still_moves_forward():
    config = ChunkingConfig(max_duration_ms=500, overlap_ms=5000, boundary_grace_ms=5000, boundary_lookahead_ms=0,
                            min_alone_speech_ms=0)
    audio, _ = scene(("speech", 5.0))
    chunks = chunk_all(audio, config=config)
    assert len(chunks) >= 9 and all(c.duration <= 0.5 for c in chunks)
    assert all(b.start_sample > a.start_sample for a, b in zip(chunks, chunks[1:], strict=False))


def test_after_finish_the_chunker_takes_no_more_audio():
    chunker = Chunker("sess_test", RATE, CHUNKING, VAD_CONFIG)
    chunker.push(in_room(speech(1.0)))
    assert len(chunker.finish()) == 1 and chunker.finish() == []
    with pytest.raises(RuntimeError):
        chunker.push(noise(0.1))


# ---------------------------------------------------------------- pre-roll

PREROLL = 2 * RATE


def test_speech_begun_before_the_key_press_is_kept():
    audio, ((start, _),) = scene(("pause", 1.7), ("speech", 2.0))  # speaking 300 ms before pressing
    drop = trim_preroll(audio, RATE, PREROLL, VAD_CONFIG)
    assert start - 0.3 * RATE <= drop <= start - 0.15 * RATE


def test_silence_before_the_key_press_keeps_only_its_last_300_ms():
    for floor in (-60, -45):
        audio, _ = scene(("pause", 2.0), ("speech", 2.0), floor_db=floor)
        assert trim_preroll(audio, RATE, PREROLL, VAD_CONFIG) == PREROLL - round(0.3 * RATE)


def test_earlier_unrelated_speech_is_dropped():
    audio, _ = scene(("pause", 0.1), ("speech", 0.4), ("pause", 1.5), ("speech", 1.0))  # ended 1.5 s before the press
    assert trim_preroll(audio, RATE, PREROLL, VAD_CONFIG) == PREROLL - round(0.3 * RATE)


def test_speech_running_through_the_whole_pre_roll_is_kept_whole():
    audio, _ = scene(("speech", 4.0))
    assert trim_preroll(audio, RATE, PREROLL, VAD_CONFIG) == 0


def test_speech_that_just_ended_before_the_press_is_kept_and_older_speech_is_not():
    audio, ((start, end), _) = scene(("pause", 1.133), ("speech", 0.7), ("pause", 0.6), ("speech", 1.0))
    assert end / RATE == pytest.approx(1.8, abs=0.01)  # ended 200 ms before the press
    drop = trim_preroll(audio, RATE, PREROLL, VAD_CONFIG)
    assert start - 0.3 * RATE <= drop <= start - 0.15 * RATE
    audio, ((_, end), _) = scene(("pause", 0.933), ("speech", 0.7), ("pause", 0.8), ("speech", 1.0))
    assert end / RATE == pytest.approx(1.6, abs=0.01)  # 400 ms before the press: the user had stopped
    assert trim_preroll(audio, RATE, PREROLL, VAD_CONFIG) == PREROLL - round(0.3 * RATE)


def test_an_utterance_with_a_short_pause_is_kept_from_its_first_words():
    audio, ((first, _), (second, _)) = scene(("pause", 0.4), ("speech", 0.45), ("pause", 0.4), ("speech", 2.0))
    assert first - 0.3 * RATE <= trim_preroll(audio, RATE, PREROLL, VAD_CONFIG) <= first - 0.15 * RATE
    audio, ((first, _), (second, _)) = scene(("pause", 0.2), ("speech", 0.45), ("pause", 0.9), ("speech", 2.0))
    drop = trim_preroll(audio, RATE, PREROLL, VAD_CONFIG)  # 0.9 s apart: the earlier words are something else
    assert second - 0.3 * RATE <= drop <= second - 0.15 * RATE


def test_nothing_after_the_key_press_is_ever_dropped():
    audio, _ = scene(("pause", 3.0))
    assert trim_preroll(audio, RATE, 0, VAD_CONFIG) == 0
    assert trim_preroll(audio, RATE, round(0.2 * RATE), VAD_CONFIG) == 0  # a pre-roll shorter than what is kept
    assert trim_preroll(audio[:RATE], RATE, PREROLL, VAD_CONFIG) == RATE - round(0.3 * RATE)  # shorter audio than claimed
    assert trim_preroll(np.zeros(0, np.float32), RATE, PREROLL, VAD_CONFIG) == 0
    for drop in (trim_preroll(scene(("speech", 3.0))[0], RATE, PREROLL, VAD_CONFIG),
                 trim_preroll(audio, RATE, PREROLL, VAD_CONFIG)):
        assert 0 <= drop <= PREROLL



# ---- the defaults: enough speech before a pause may cut, and no short rest sent alone (the owner's log, 2026-10-02)

def test_a_pause_cuts_only_after_four_seconds_of_speech():
    short, _ = scene(("pause", 0.3), ("speech", 2.5), ("pause", 1.8), ("speech", 6.0), ("pause", 0.5))
    assert [c.boundary for c in chunk_all(short, config=ChunkingConfig())] == ["end"]  # no 2.5 s fragment alone
    long, _ = scene(("pause", 0.3), ("speech", 5.0), ("pause", 1.8), ("speech", 6.0), ("pause", 0.5))
    assert [c.boundary for c in chunk_all(long, config=ChunkingConfig())] == ["pause", "end"]


@pytest.mark.parametrize("parts, alone_ms", [
    # cut at the maximum (at the quietest point of the last 2 s: the rest has 0.4-2.8 s of speech), then let go
    ((("pause", 0.3), ("speech", 20.8), ("pause", 0.6)), 3000),
    ((("pause", 0.3), ("speech", 6.0), ("pause", 1.6), ("speech", 0.5), ("pause", 0.4)), 2000),  # a word after a pause
])
def test_a_short_rest_goes_again_with_the_part_before_it(parts, alone_ms):
    audio, _ = scene(*parts)
    chunks = chunk_all(audio, config=ChunkingConfig(min_alone_speech_ms=alone_ms))
    first, last = chunks[0], chunks[-1]
    assert len(chunks) == 2 and last.replaces == first.sequence and last.boundary == "end"
    assert (last.start_sample, last.overlap_end_sample) == (first.start_sample, first.overlap_end_sample)
    assert last.end_sample == len(audio) and len(last.audio) == last.end_sample - last.start_sample
    assert np.array_equal(last.audio[:len(first.audio)], first.audio)


def test_a_rest_with_enough_speech_is_sent_alone():
    audio, _ = scene(("pause", 0.3), ("speech", 6.0), ("pause", 1.6), ("speech", 3.0), ("pause", 0.4))
    chunks = chunk_all(audio, config=ChunkingConfig())
    assert [c.boundary for c in chunks] == ["pause", "end"] and chunks[-1].replaces == 0


def finish_timed(audio: np.ndarray, config: ChunkingConfig) -> list[AudioChunk]:
    chunker = Chunker("sess_test", RATE, config, VAD_CONFIG)
    return chunker.push(audio) + chunker.finish(timed=True)


@pytest.mark.parametrize("parts", [
    (("pause", 0.3), ("speech", 20.8), ("pause", 0.6)),  # after a cut at the maximum
    (("pause", 0.3), ("speech", 6.0), ("pause", 1.6), ("speech", 0.5), ("pause", 0.4)),  # a word after a pause
])
def test_with_word_times_a_short_rest_goes_with_only_the_end_of_the_part_before(parts):
    audio, _ = scene(*parts)
    config = ChunkingConfig(min_alone_speech_ms=3000)
    chunks = finish_timed(audio, config)
    first, last = chunks[0], chunks[-1]
    assert len(chunks) == 2 and last.replaces == 0 and last.boundary == "end"
    assert first.start_sample + 2 * RATE <= last.start_sample < first.end_sample  # not the whole part again
    assert last.overlap_end_sample == first.end_sample  # the repeated end, which the merge takes out
    assert last.speech_seconds >= config.join_context_ms / 1000 - 0.05  # enough speech for a good transcript
    assert last.end_sample == len(audio) and len(last.audio) == last.end_sample - last.start_sample
    assert np.array_equal(last.audio[:first.end_sample - last.start_sample], audio[last.start_sample:first.end_sample])


def test_the_rest_starts_between_words_not_inside_one():
    audio, spans = scene(("pause", 0.3), ("speech", 6.0), ("pause", 1.6), ("speech", 0.5), ("pause", 0.4))
    last = finish_timed(audio, ChunkingConfig(min_alone_speech_ms=3000))[-1]
    window = audio[last.start_sample - 400:last.start_sample + 400]
    assert np.sqrt(np.mean(window ** 2)) < 0.3 * np.sqrt(np.mean(audio[spans[0][0]:spans[0][1]] ** 2))


def test_without_word_times_or_much_to_save_the_whole_part_goes_again():
    audio, _ = scene(("pause", 0.3), ("speech", 6.0), ("pause", 1.6), ("speech", 0.5), ("pause", 0.4))
    assert chunk_all(audio, config=ChunkingConfig())[-1].replaces == 1  # untimed (finish() without timed)
    short, _ = scene(("pause", 0.3), ("speech", 4.5), ("pause", 1.6), ("speech", 0.5), ("pause", 0.4))
    assert finish_timed(short, ChunkingConfig(min_cut_speech_ms=4000))[-1].replaces == 1  # under 2 s saved
    assert finish_timed(audio, ChunkingConfig(join_context_ms=0))[-1].replaces == 1  # switched off
