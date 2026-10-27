"""Meeting minutes (sst.meeting.minutes): the transcript and minutes requests, chunks that keep their speakers, minutes.md,
the cost tracker and the estimate, errors in plain words. Fakes only: no network, no key."""
import base64
import json
import wave
from datetime import date

import numpy as np
import pytest

from sst import costs, usage
from sst.meeting import minutes as mm
from sst.meeting.minutes import FALLBACK_MODEL, MINUTES_MODEL, MinutesError, summarise

RATE = 16_000


def write_wav(path, seconds: float, loud: bool = True):
    t = np.arange(int(seconds * RATE)) / RATE
    samples = (np.sin(2 * np.pi * 220 * t) * (3000 if loud else 0)).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())


def meeting(tmp_path, seconds: float = 60, frames: int = 3, mic_loud: bool = True):
    folder = tmp_path / "2026-10-27 1003 Microsoft Teams"
    folder.mkdir()
    write_wav(folder / "you.wav", seconds, mic_loud)
    write_wav(folder / "others.wav", seconds)
    (folder / "frames").mkdir()
    for n in range(frames):
        (folder / "frames" / f"{n + 1:04d} {n:02d}m05s.jpg").write_bytes(b"\xff\xd8jpeg" + bytes([n]))
    (folder / "meeting.json").write_text(json.dumps({"app": "Microsoft Teams", "title": "Weekly sync",
                                                     "started": "2026-10-27T10:03:00", "seconds": seconds,
                                                     "frames": frames}), encoding="utf-8")
    return folder


def answer(data: dict, finish: str = "STOP", audio: int = 100, text: int = 50, out: int = 30) -> dict:
    return {"candidates": [{"finishReason": finish, "content": {"parts": [
        {"text": "thinking...", "thought": True}, {"text": json.dumps(data, ensure_ascii=False)}]}}],
        "usageMetadata": {"promptTokenCount": audio + text, "candidatesTokenCount": out,
                          "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": audio},
                                                  {"modality": "TEXT", "tokenCount": text}]}}


MINUTES = {"title": "Weekly sync", "summary": "The release was discussed.", "decisions": ["Ship on Friday."],
           "actions": [{"task": "Send the notes", "owner": "You", "due": "Friday"},
                       {"task": "Check the build", "owner": "", "due": ""}], "language": "en"}


class Gemini:
    """A fake Gemini API: transcript answers in turn, then the minutes; every request and upload kept."""

    def __init__(self, transcripts=None, minutes=None, fail=None):
        self.transcripts = list(transcripts or [{"language": "en", "names": [], "lines": [
            {"time": "00:03", "speaker": "Speaker 1", "text": "Good morning."},
            {"time": "00:01", "speaker": "You", "text": "Hello everyone."}]}])
        self.minutes = minutes or MINUTES
        self.fail = fail  # (url part, MinutesError) raised once for a matching request
        self.requests, self.uploads, self.deleted = [], [], []

    def post(self, url, payload, key, timeout, method="POST"):
        assert key == "test-key" and timeout > 0
        if method == "DELETE":
            self.deleted.append(url.rsplit("/v1beta/", 1)[1])
            return {}
        self.requests.append((url, payload))
        if self.fail and self.fail[0] in url:
            error, self.fail = self.fail[1], None
            raise error
        schema = payload["generationConfig"]["responseJsonSchema"]
        if schema is mm.TRANSCRIPT_SCHEMA:
            data = self.transcripts.pop(0) if len(self.transcripts) > 1 else self.transcripts[0]
            if isinstance(data, str):
                return answer({}, finish=data)
            return answer(data)
        return answer(self.minutes, audio=0, text=2000)

    def upload(self, data, mime, key, timeout):
        assert mime == "audio/wav" and key == "test-key" and data[:4] == b"RIFF"
        name = f"files/f{len(self.uploads) + 1}"
        self.uploads.append((name, len(data)))
        return {"name": name, "uri": f"https://generativelanguage.googleapis.com/v1beta/{name}", "mimeType": "audio/wav",
                "state": "ACTIVE"}

    def transcript_requests(self):
        return [p for _, p in self.requests if p["generationConfig"]["responseJsonSchema"] is mm.TRANSCRIPT_SCHEMA]

    def minutes_requests(self):
        return [p for _, p in self.requests if p["generationConfig"]["responseJsonSchema"] is mm.MINUTES_SCHEMA]


@pytest.fixture
def ledger(monkeypatch):
    book = usage.Ledger(None, today=lambda: date(2026, 10, 27))
    monkeypatch.setattr(usage, "_ledger", book)
    return book


def run(folder, gemini, **kw):
    return summarise(folder, "test-key", post=gemini.post, upload=gemini.upload, **kw)


def test_the_requests_carry_both_tracks_the_frames_inline_and_the_schemas(tmp_path, ledger):
    folder = meeting(tmp_path)
    gemini = Gemini()
    run(folder, gemini)
    (transcript,), (minutes,) = gemini.transcript_requests(), gemini.minutes_requests()
    assert gemini.requests[0][0].endswith(f"/models/{MINUTES_MODEL}:generateContent")
    parts = transcript["contents"][0]["parts"]
    audio = [p["fileData"] for p in parts if "fileData" in p]
    assert [a["fileUri"].rsplit("/", 1)[1] for a in audio] == ["f1", "f2"] and {a["mimeType"] for a in audio} == {"audio/wav"}
    assert "microphone" in parts[0]["text"] and "You" in parts[0]["text"] and "other people" in parts[2]["text"]
    prompt = parts[-1]["text"]
    assert "Never translate" in prompt and "Tamil script" in prompt and "[inaudible]" in prompt and "Speaker 1" in prompt
    config = transcript["generationConfig"]
    assert config["responseMimeType"] == "application/json" and config["maxOutputTokens"] == mm.MAX_OUTPUT
    assert config["thinkingConfig"] == {"thinkingLevel": "minimal"} and "temperature" not in config  # Gemini 3.6
    assert set(mm.TRANSCRIPT_SCHEMA["required"]) == {"language", "lines", "names"}
    # the minutes: the transcript as text, meeting.json, and every frame inline at high resolution
    parts = minutes["contents"][0]["parts"]
    assert "[00:01] You: Hello everyone." in parts[0]["text"] and "Microsoft Teams" in parts[0]["text"]
    assert "Never invent" in parts[0]["text"] and "only when the meeting said" in parts[0]["text"]
    images = [p for p in parts if "inlineData" in p]
    assert len(images) == 3 and {p["inlineData"]["mimeType"] for p in images} == {"image/jpeg"}
    assert base64.b64decode(images[1]["inlineData"]["data"]) == b"\xff\xd8jpeg\x01"
    assert {p["mediaResolution"]["level"] for p in images} == {"MEDIA_RESOLUTION_HIGH"}
    assert parts[parts.index(images[1]) - 1]["text"] == "Screenshot at 01:05:"
    assert minutes["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert set(minutes["generationConfig"]["responseJsonSchema"]["required"]) >= {"title", "decisions", "actions"}


def test_uploaded_tracks_are_deleted_and_a_silent_microphone_isn_t_sent(tmp_path, ledger):
    gemini = Gemini()
    run(meeting(tmp_path, mic_loud=False), gemini)
    assert len(gemini.uploads) == 1 and gemini.deleted == ["files/f1"]
    parts = gemini.transcript_requests()[0]["contents"][0]["parts"]
    assert "other people" in parts[0]["text"] and sum("fileData" in p for p in parts) == 1


def test_a_long_meeting_is_chunked_and_its_speakers_keep_their_labels_and_names(tmp_path, ledger):
    folder = meeting(tmp_path, seconds=60, frames=0)
    gemini = Gemini(transcripts=[
        {"language": "ja", "names": [], "lines": [{"time": "00:02", "speaker": "Speaker 1", "text": "おはようございます。"},
                                                  {"time": "00:10", "speaker": "You", "text": "Good morning."}]},
        {"language": "ja", "names": [{"label": "Speaker 1", "name": "Tanaka"}], "lines": [
            {"time": "00:05", "speaker": "Speaker 1", "text": "田中です。来週までに資料を送ります。"},
            {"time": "00:09", "speaker": "Speaker 2", "text": "வணக்கம்"}]},
        {"language": "en", "names": [{"label": "Speaker 2", "name": "Tanaka"}], "lines": [
            {"time": "00:01", "speaker": "Speaker 2", "text": "Thanks."}]}])
    result = run(folder, gemini, chunk=20)
    requests = gemini.transcript_requests()
    assert len(requests) == 3 and len(gemini.uploads) == 6 and len(gemini.deleted) == 6
    second = requests[1]["contents"][0]["parts"][-1]["text"]
    assert "00:20 into the meeting" in second and "The people so far: Speaker 1, You" in second
    assert "[00:02] Speaker 1: おはようございます。" in second
    assert "Speaker 1 (Tanaka)" in requests[2]["contents"][0]["parts"][-1]["text"]
    # the name learned in the second part names the first too; a second "Tanaka" for another voice is refused
    assert result.transcript == [("00:02", "Tanaka", "おはようございます。"), ("00:10", "You", "Good morning."),
                                 ("00:25", "Tanaka", "田中です。来週までに資料を送ります。"), ("00:29", "Speaker 2", "வணக்கம்"),
                                 ("00:41", "Speaker 2", "Thanks.")]
    assert result.speakers == ["Tanaka", "You", "Speaker 2"]
    assert "[00:25] Tanaka: 田中です。" in gemini.minutes_requests()[0]["contents"][0]["parts"][0]["text"]
    assert "inlineData" not in json.dumps(gemini.minutes_requests()[0])  # no frames, none sent


def test_a_stretch_that_runs_out_of_room_is_asked_again_in_halves(tmp_path, ledger, monkeypatch):
    monkeypatch.setattr(mm, "MIN_CHUNK_S", 5)
    gemini = Gemini(transcripts=["MAX_TOKENS",
                                 {"language": "en", "names": [], "lines": [{"time": "00:01", "speaker": "You", "text": "A"}]},
                                 {"language": "en", "names": [], "lines": [{"time": "00:01", "speaker": "You", "text": "B"}]}])
    result = run(meeting(tmp_path, seconds=20, frames=0), gemini)
    assert [line for line in result.transcript] == [("00:01", "You", "A"), ("00:11", "You", "B")]
    assert len(gemini.transcript_requests()) == 3 and len(gemini.deleted) == len(gemini.uploads) == 6


def test_minutes_md_and_json_are_written(tmp_path, ledger):
    folder = meeting(tmp_path)
    result = run(folder, Gemini())
    md = (folder / "minutes.md").read_text(encoding="utf-8")
    assert md.startswith("# Weekly sync\n\n27 October 2026, 10:03 · Microsoft Teams · 1 min\n")
    assert "People: You, Speaker 1" in md and "## Summary\n\nThe release was discussed." in md
    assert "- Ship on Friday." in md
    assert "- [ ] Send the notes (You, due Friday)" in md and "- [ ] Check the build\n" in md
    assert md.index("**[00:01] You:** Hello everyone.") < md.index("**[00:03] Speaker 1:** Good morning.")
    saved = json.loads((folder / "minutes.json").read_text(encoding="utf-8"))
    assert saved["title"] == result.title == "Weekly sync" and saved["actions"][0]["due"] == "Friday"
    assert result.actions[1] == mm.ActionItem("Check the build") and result.language == "en"
    empty = mm.render(mm.MinutesResult("Call", ""))
    assert "None recorded." in empty and "Nobody spoke." in empty


def test_every_call_is_counted_as_the_meeting(tmp_path, ledger):
    run(meeting(tmp_path, seconds=60, frames=0), Gemini(), chunk=20)
    (entries,) = ledger.days.values()
    assert list(entries) == ["meeting|gemini-3.6-flash"]
    counts = entries["meeting|gemini-3.6-flash"]
    assert counts["calls"] == 4 and counts["seconds"] == pytest.approx(60) and counts["audio_in"] == 300
    assert usage.FEATURES["meeting"] == ("Meeting minutes", True)
    assert ledger.summary().lines[0].amount == "1 min"


def test_the_estimate_is_about_a_quarter_dollar_an_hour():
    price = costs.speech_price_on("gemini", MINUTES_MODEL, date.today().isoformat())
    # an hour, 20 frames: 3 stretches; both tracks at 32 tokens a second; the transcript at 300 tokens a minute
    audio = 2 * 3600 * 32
    text_in = 3 * 700 + 900 + 18_000 * 0.7 + 20 * 1120
    out = 18_000 + 3 * costs.THINKING["minimal"] + 1500 + costs.THINKING["low"]
    by_hand = (audio * price.audio + text_in * price.text + out * price.output) / 1e6
    assert mm.estimate(3600, 20) == pytest.approx(by_hand)
    if date.today() < date(2027, 1, 1):  # 3.6 Flash doubles then
        assert 0.22 < mm.estimate(3600, 20) < 0.30
    assert mm.estimate(3600, 500) == pytest.approx(mm.estimate(3600, mm.MAX_FRAMES))  # only MAX_FRAMES are sent
    assert mm.estimate(0, 0) == 0 and mm.estimate(3600, 0, model="someone-elses-model") == 0
    assert mm.estimate(1800, 0) < mm.estimate(3600, 0)


def test_errors_come_back_in_plain_words_and_the_recording_is_kept(tmp_path, ledger):
    folder = meeting(tmp_path)
    refused = mm._refused(403, json.dumps({"error": {"message": "API key not valid."}}))
    assert refused.status == 403 and "didn't accept the API key" in str(refused)
    gemini = Gemini(fail=("generateContent", refused))
    with pytest.raises(MinutesError) as e:
        run(folder, gemini)
    assert "API key" in str(e.value) and "The recording is kept" in str(e.value)
    assert (folder / "you.wav").exists() and (folder / "others.wav").exists() and not (folder / "minutes.md").exists()
    assert gemini.deleted == ["files/f1", "files/f2"]  # uploaded files go even when the answer failed
    with pytest.raises(MinutesError, match="need a Gemini API key"):
        summarise(folder, "", post=gemini.post, upload=gemini.upload)
    with pytest.raises(MinutesError, match="no recording"):
        summarise(tmp_path, "test-key", post=gemini.post, upload=gemini.upload)
    gemini = Gemini()
    gemini.post = lambda url, payload, key, timeout, method="POST": {} if method == "DELETE" else {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "not json"}]}}]}
    with pytest.raises(MinutesError, match="couldn't be read"):
        run(folder, gemini)
    assert (folder / "you.wav").exists()


def test_a_model_that_is_gone_or_overloaded_gives_way_to_the_backup(tmp_path, ledger):
    gemini = Gemini(fail=(MINUTES_MODEL, MinutesError("Gemini had trouble (HTTP 503).", 503)))
    run(meeting(tmp_path, frames=0), gemini)
    urls = [url for url, _ in gemini.requests]
    assert urls[0].endswith(f"{MINUTES_MODEL}:generateContent")
    assert all(url.endswith(f"{FALLBACK_MODEL}:generateContent") for url in urls[1:]) and len(urls) == 3
    assert list(next(iter(ledger.days.values()))) == [f"meeting|{FALLBACK_MODEL}"]


def test_frames_go_at_googles_default_resolution_when_their_own_is_refused(tmp_path, ledger):
    gemini = Gemini()
    real_post = gemini.post

    def post(url, payload, key, timeout, method="POST"):
        if "mediaResolution" in json.dumps(payload):
            raise MinutesError("Gemini didn't accept the request (HTTP 400).", 400)
        return real_post(url, payload, key, timeout, method)

    gemini.post = post
    result = run(meeting(tmp_path), gemini)
    (minutes,) = gemini.minutes_requests()
    assert result.title == "Weekly sync" and len([p for p in minutes["contents"][0]["parts"] if "inlineData" in p]) == 3


def test_frames_are_picked_evenly_within_the_budget(tmp_path):
    paths = []
    for n in range(100):
        paths.append(tmp_path / f"{n + 1:04d} {n:02d}m00s.jpg")
        paths[-1].write_bytes(b"x" * 100)
    picked = mm.pick_frames(paths, most=5)
    assert [p.name[:4] for p in picked] == ["0001", "0026", "0051", "0075", "0100"]
    assert len(mm.pick_frames(paths, most=30, budget=1000)) == 10 and mm.pick_frames([], most=5) == []


def test_times_read_and_written():
    assert mm.clock(4565) == "76:05" and mm.clock(-3) == "00:00"
    assert mm._seconds("07:05") == 425 and mm._seconds("1:02:03") == 3723 and mm._seconds("soon") is None
