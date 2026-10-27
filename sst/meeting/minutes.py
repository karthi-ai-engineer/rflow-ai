"""After the meeting: a transcript with speakers, then the minutes (title, summary, decisions, action items), by Gemini.

Two steps, each a generateContent request with a JSON answer (generationConfig.responseMimeType "application/json"
and responseJsonSchema, a JSON Schema whose key order the answer follows):
  1) transcript  you.wav (the microphone: "You") and others.wav (what the laptop played: everyone else) cover the same
                 time, so a stretch of both goes in one request, each as its own audio part after a line saying which
                 it is. A long meeting goes in CHUNK_S stretches, one after another: each is told the people so far and
                 the last lines before it, so the same voice keeps its label. Other people are always "Speaker 1, 2..."
                 in the lines; a name the meeting makes clear comes back apart ("names") and replaces the label all the
                 way through at the end, so a name learned in the second hour also names the first. A stretch whose
                 answer runs out of room (finishReason MAX_TOKENS) is asked again in halves. A silent track (a muted
                 microphone) isn't sent.
  2) minutes     the transcript as text, meeting.json's app, title and date, and up to MAX_FRAMES of the screen's kept
                 frames (evenly picked, inline JPEG, each after its time into the meeting) -> title, summary,
                 decisions, actions (owner and due only when they were said).
The prompts say: words in the language and script they were said in (Japanese, English, Tamil, mixed), never
translated, never guessed; nothing in the minutes that the meeting didn't say. Written into the meeting's folder:
minutes.json (MinutesResult) and minutes.md (readable: title, date, app, people, summary, decisions, the action items as
a checklist, then the transcript). The recording is never deleted, whatever fails: MinutesError says what, plainly.

Each stretch's audio goes through the Files API (a 20-minute track is ~38 MB, over the inline limit), and the uploaded
files are deleted as soon as the answer is in (Google would keep them 48 hours). Every call is counted in the cost
tracker as "meeting" (sst.usage). A model that is gone or overloaded (FALLBACK_STATUSES) gives way to FALLBACK_MODEL for
the rest of the meeting.

MINUTES_MODEL: Gemini 3.6 Flash, Google's multimodal Flash (long audio and images, 1,048,576 tokens in, 65,536 out),
asked to think as little as it allows for the transcript (sst.modelrules) and a little ("low") for the minutes. 3.8
Flash costs the same but can't think less than "low"; 3.5 Flash-Lite costs half but is the weaker listener (the backup).
An hour costs about $0.27 (estimate(); it doubles with 3.6 Flash's price on 2027-01-01).

From Google's Gemini API docs (seen 2026-10-27):
  https://ai.google.dev/gemini-api/docs/audio       WAV among the formats; 32 tokens a second of audio; up to 9.5 hours a
                                                    prompt; "Identify distinct speakers (e.g., Speaker 1, Speaker 2)" by
                                                    prompt; MM:SS to refer to times; inline requests up to 20 MB
  https://ai.google.dev/gemini-api/docs/files       resumable upload to /upload/v1beta/files (X-Goog-Upload-* headers),
                                                    2 GB a file, 20 GB a project, kept 48 hours, DELETE /v1beta/{name}
  https://ai.google.dev/gemini-api/docs/media-resolution  Gemini 3 images: low 280, medium 560, high 1120 tokens
                                                    (the default, recommended for reading images), ultra_high 2240
  https://ai.google.dev/api/generate-content        Part.mediaResolution {"level": "MEDIA_RESOLUTION_HIGH"}, several parts
                                                    of any kind in one request, maxOutputTokens, finishReason
  https://ai.google.dev/gemini-api/docs/structured-output  JSON Schema types and keywords (minItems/maxItems enforced,
                                                    minLength and the like ignored: checked here)
  https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash  1,048,576 tokens in, 65,536 out; audio and image input
Standard library, numpy, sst.costs, sst.usage, sst.modelrules and sst.netconnect only (never sst.pipeline, sst.dictate,
sst.audio: a meeting is not dictation).
"""
import base64
import io
import json
import logging
import re
import time
import urllib.error
import urllib.request
import wave
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from sst import costs, netconnect, usage
from sst.modelrules import gemini_takes_temperature, gemini_thinking

log = logging.getLogger(__name__)

MINUTES_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"
FALLBACK_STATUSES = (404, 429, 500, 502, 503, 504)  # the model gone, overloaded or failing: the backup is worth a try
BASE = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD = "https://generativelanguage.googleapis.com/upload/v1beta/files"
FEATURE = "meeting"  # the cost tracker's line (sst.usage.FEATURES)

YOU, OTHERS = "you.wav", "others.wav"  # sst.meeting.recorder's files
TRACKS = ((YOU, "the microphone: the user of this laptop, speaker \"You\""),
          (OTHERS, "the laptop's sound: the other people in the meeting"))
META = "meeting.json"
FRAMES = "frames"
MINUTES_MD, MINUTES_JSON = "minutes.md", "minutes.json"
YOU_LABEL = "You"

CHUNK_S = 20 * 60  # seconds of the meeting in one transcript request: its answer stays well inside 65,536 tokens
MIN_CHUNK_S = 5 * 60  # a stretch whose answer still runs out of room this short is an error, not split again
CARRY_LINES = 8  # the last lines of the transcript so far, given to the next stretch so the voices keep their labels
SILENT_PEAK = 64  # a track whose loudest sample (of 32,767) is below this has no speech: not sent
MAX_FRAMES = 30  # screen frames given to the minutes, evenly picked
FRAME_BYTES = 12_000_000  # their JPEGs together at most (base64 adds a third): the request stays under 20 MB
FRAME_RESOLUTION = "MEDIA_RESOLUTION_HIGH"  # slides' small print stays readable
MAX_OUTPUT = 65_536  # the models' most; billed only for what is written
ANSWER_TIMEOUT = 120.0  # seconds, plus ANSWER_PER_SECOND for each second of audio in the request
ANSWER_PER_SECOND = 0.25
UPLOAD_TIMEOUT = 120.0  # seconds, plus UPLOAD_PER_MB
UPLOAD_PER_MB = 2.0
MINUTES_TIMEOUT = 300.0
DELETE_TIMEOUT = 20.0
FILE_POLL = 1.0  # seconds between looks at an uploaded file Google is still processing
LEAST = "least"  # as little thinking as the model allows (sst.modelrules), whichever model answers
MINUTES_THINKING = "low"  # the minutes weigh what was decided: a little thinking (Gemini 3), the transcript none

# The estimate (estimate()): Google's 32 audio tokens a second (both tracks, silence included) and 1,120 tokens an image
# at MEDIA_RESOLUTION_HIGH; the rest Rflow's own measures of its prompts and a transcript's length.
AUDIO_TOKENS = costs.FLASH_AUDIO
IMAGE_TOKENS = 1120
TRANSCRIPT_OUT = 300  # tokens a minute of meeting written as transcript JSON (~150 words a minute, the keys)
TRANSCRIPT_TEXT = 0.7  # the same transcript as text for the minutes: the JSON's keys gone
PROMPT_TOKENS = 700  # a transcript request's instruction, the people so far and the carried lines
MINUTES_PROMPT = 900  # the minutes request's instruction and meeting.json
MINUTES_OUT = 1500  # title, summary, decisions and actions of an hour's meeting
KEPT = "The recording is kept in the meeting's folder: the minutes can be written again later."

LANGUAGE_NAMES = {"ja": "Japanese", "en": "English", "ta": "Tamil", "hi": "Hindi", "zh": "Chinese", "ko": "Korean",
                  "fr": "French", "de": "German", "es": "Spanish"}


class MinutesError(Exception):
    """Why the minutes couldn't be written, in plain words. status: the HTTP status (0 without an answer)."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class _TooLong(Exception):
    """An answer that ran out of room (finishReason MAX_TOKENS): its stretch is asked again in halves."""


@dataclass
class ActionItem:
    task: str
    owner: str = ""  # only when it was said
    due: str = ""  # only when it was said, as it was said


@dataclass
class MinutesResult:
    title: str
    summary: str
    decisions: list[str] = field(default_factory=list)
    actions: list[ActionItem] = field(default_factory=list)
    transcript: list[tuple[str, str, str]] = field(default_factory=list)  # (mm:ss, speaker, text)
    speakers: list[str] = field(default_factory=list)  # in the order they first speak
    language: str = ""


# ---------------------------------------------------------------- HTTP (urllib), the fakes' stand-ins in tests

def _open(request: urllib.request.Request, timeout: float) -> tuple[str, dict]:
    """The answer's text and headers; MinutesError in plain words for anything else."""
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=netconnect.ssl_context()) as response:
            return response.read().decode("utf-8", "replace"), {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise _refused(e.code, body, e.headers) from None
    except TimeoutError:
        raise MinutesError("Gemini didn't answer in time.") from None
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise MinutesError("Gemini didn't answer in time.") from None
        raise MinutesError(f"Google couldn't be reached ({e.reason}).") from None
    except OSError as e:
        raise MinutesError(f"The connection to Google broke ({e}).") from None


def _refused(status: int, body: str, headers=None) -> MinutesError:
    """An error answer as a sentence: no credit, quota or key said plainly (sst.netconnect), Google's own words after."""
    try:
        error = json.loads(body)
        error = (error[0] if isinstance(error, list) and error else error).get("error") or {}
        detail = str(error.get("message") or "") if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        detail = body[:200]
    found = netconnect.refusal(status, body, headers)
    plain = {404: f"Gemini doesn't know the model (HTTP {status})", 429: f"Gemini is busy or rate limited (HTTP {status})",
             400: f"Gemini didn't accept the request (HTTP {status})"}.get(
        status, f"Gemini had trouble (HTTP {status})" if status >= 500 else f"Gemini refused (HTTP {status})")
    reason = f"Gemini: {found.reason} (HTTP {status})" if found else plain
    return MinutesError(f"{reason}: {detail[:200]}." if detail else f"{reason}.", status)


def post(url: str, payload: dict | None, key: str, timeout: float, method: str = "POST") -> dict:
    """A JSON request to Gemini's API (POST, GET or DELETE) and its JSON answer ({} for an empty one)."""
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
    text, _ = _open(request, timeout)
    if not text.strip():
        return {}
    try:
        answer = json.loads(text)
    except ValueError:
        raise MinutesError("Gemini's answer couldn't be read.") from None
    return answer if isinstance(answer, dict) else {}


def upload(data: bytes, mime: str, key: str, timeout: float) -> dict:
    """A file into Gemini's Files API (a resumable upload, sent in one piece): its {"name", "uri", "mimeType"...},
    ready to use."""
    start = urllib.request.Request(UPLOAD, data=json.dumps({"file": {"display_name": "rflow-meeting"}}).encode("utf-8"),
                                   method="POST", headers={
                                       "Content-Type": "application/json", "x-goog-api-key": key,
                                       "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                                       "X-Goog-Upload-Header-Content-Length": str(len(data)),
                                       "X-Goog-Upload-Header-Content-Type": mime})
    _, headers = _open(start, timeout)
    address = headers.get("x-goog-upload-url")
    if not address:
        raise MinutesError("Gemini's Files API gave no upload address.")
    # The upload address carries its own authorisation: the key isn't sent again.
    send = urllib.request.Request(address, data=data, method="POST", headers={
        "X-Goog-Upload-Command": "upload, finalize", "X-Goog-Upload-Offset": "0"})
    text, _ = _open(send, timeout)
    try:
        file = json.loads(text).get("file")
    except (ValueError, AttributeError):
        file = None
    if not isinstance(file, dict) or not file.get("uri"):
        raise MinutesError("Gemini's Files API kept no file.")
    deadline = time.monotonic() + timeout
    while file.get("state") == "PROCESSING" and file.get("name") and time.monotonic() < deadline:
        time.sleep(FILE_POLL)
        file = {**file, **post(f"{BASE}/{file['name']}", None, key, timeout, method="GET")}
    if file.get("state") == "FAILED":
        raise MinutesError("Gemini couldn't read the uploaded recording.")
    return file


# ---------------------------------------------------------------- the prompts and the answers' shapes

TRANSCRIPT_PROMPT = """You are writing the transcript of an online meeting from its recording.
{tracks}
The tracks start together and cover the same stretch of time.

Rules:
1. Write down what was said, in the language it was said in: Japanese in Japanese script, Tamil in Tamil script, English \
in English, any other language as spoken. A sentence that mixes languages stays mixed. Never translate, summarise or \
correct what was said; leave out only fillers (um, uh, えーと, あの) and false starts.
2. One line per turn: when the speaker changes, a new line. A long turn may be split at a sentence end, about every 30 \
seconds.
3. time: when the line starts, as MM:SS from the start of these tracks (both start at 00:00). Lines in the order they \
were said, the two tracks interleaved.
4. speaker: "You" for speech on the microphone track. On the laptop track there may be several people: tell them apart \
by their voices and label them "Speaker 1", "Speaker 2"... in the order they first speak, and keep the same label for \
the same voice all the way through. Always use these labels in the lines, even when you learn a name.
5. names: when the meeting makes a person's name clear (they introduce themselves, or are called by name and answer), \
give their label and the name as it was said. Only names that were said; never guess one.
6. If the microphone also picks up the laptop's sound (no headphones), write that speech once, for its speaker on the \
laptop track, never as "You".
7. Only what is in the recording. Never guess at words you can't hear: write [inaudible] for a part that can't be made \
out. Silence, music and noise give no line; if nobody speaks, lines is empty.
8. language: the main language spoken, as a two-letter code (ja, en, ta...).
{languages}{context}"""

MINUTES_PROMPT_TEXT = """You are writing the minutes of an online meeting from its transcript{frames_note}.

Rules:
1. Only what the transcript{frames_only} shows. Never invent a decision, a task, a name, a date or a number.
2. title: a short title for the meeting, under 10 words, from what it was about.
3. summary: what was discussed, in a few short paragraphs (a few sentences for a short meeting).
4. decisions: what was agreed or decided, each in one sentence. Empty when nothing was decided.
5. actions: the tasks someone agreed to do or was asked to do. owner: who will do it, as the transcript names them \
("You" for the user of this laptop), only when the meeting said who; otherwise "". due: when it is due, as it was said \
("Friday", "by 3 November"), only when the meeting said when; otherwise "". Never guess an owner or a date.
6. Write the minutes in the language most of the meeting was spoken in, in its own script; keep names, terms and \
numbers exactly as in the transcript. language: that language's two-letter code.
7. {frames_rule}
8. If the transcript is empty or too short to say anything, say so in the summary and leave the lists empty.

The meeting: {meeting}

Transcript ([MM:SS] speaker: words):
{transcript}"""

FRAMES_RULE = ("A screenshot adds only what is on it (a slide's figures, a document's name), used where the transcript "
               "talks about it; something on screen that nobody spoke about is not a decision or a task.")
NO_FRAMES_RULE = "There are no screenshots: use the transcript alone."

_STRING = {"type": "string"}
TRANSCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {**_STRING, "description": "The main language spoken, a two-letter code."},
        "lines": {"type": "array", "items": {
            "type": "object",
            "properties": {"time": {**_STRING, "description": "MM:SS from the start of the tracks."},
                           "speaker": {**_STRING, "description": "\"You\", or \"Speaker 1\", \"Speaker 2\"..."},
                           "text": {**_STRING, "description": "The words said, in the language and script said."}},
            "required": ["time", "speaker", "text"]}},
        "names": {"type": "array", "items": {
            "type": "object",
            "properties": {"label": {**_STRING, "description": "\"Speaker 1\"..."},
                           "name": {**_STRING, "description": "The name as it was said in the meeting."}},
            "required": ["label", "name"]}},
    },
    "required": ["language", "lines", "names"],
}
MINUTES_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _STRING,
        "summary": _STRING,
        "decisions": {"type": "array", "items": _STRING},
        "actions": {"type": "array", "items": {
            "type": "object",
            "properties": {"task": _STRING,
                           "owner": {**_STRING, "description": "Who will do it, only if said; else empty."},
                           "due": {**_STRING, "description": "When it is due, as said, only if said; else empty."}},
            "required": ["task", "owner", "due"]}},
        "language": {**_STRING, "description": "The minutes' language, a two-letter code."},
    },
    "required": ["title", "summary", "decisions", "actions", "language"],
}


def _config(model: str, schema: dict, thinking: str) -> dict:
    """JSON by the schema; Gemini 3 thinks as asked (as little as it allows for a transcript) at its own temperature,
    an older model writes at temperature 0 (sst.modelrules)."""
    config = {"responseMimeType": "application/json", "responseJsonSchema": schema, "maxOutputTokens": MAX_OUTPUT}
    if gemini_takes_temperature(model):
        config["temperature"] = 0
    if least := gemini_thinking(model):
        config["thinkingConfig"] = {"thinkingLevel": least if thinking == LEAST else thinking}
    return config


def _answer_json(answer: dict) -> dict:
    """A generateContent answer's JSON text, read; _TooLong when it ran out of room, MinutesError when there is none."""
    candidates = answer.get("candidates") or []
    if not candidates:
        reason = (answer.get("promptFeedback") or {}).get("blockReason")
        raise MinutesError(f"Gemini blocked the request ({reason})." if reason else "Gemini gave no answer.")
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    finish = str(candidate.get("finishReason") or "STOP")
    if finish == "MAX_TOKENS":
        raise _TooLong()
    if finish not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
        raise MinutesError(f"Gemini stopped before the end ({finish}).")
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    try:
        data = json.loads(text)
    except ValueError:
        raise MinutesError("Gemini's answer couldn't be read.") from None
    if not isinstance(data, dict):
        raise MinutesError("Gemini's answer couldn't be read.")
    return data


# ---------------------------------------------------------------- times

def clock(seconds: float) -> str:
    """'07:05', '75:10' past an hour: minutes and seconds into the meeting."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _seconds(text: str) -> float | None:
    """'07:05', '1:02:03' or '65' as seconds; None for anything else."""
    parts = str(text).strip().split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return None
    if not numbers or len(numbers) > 3 or any(n < 0 for n in numbers):
        return None
    total = 0.0
    for n in numbers:
        total = total * 60 + n
    return total


# ---------------------------------------------------------------- the recording

def _track_frames(path: Path) -> tuple[int, int, int]:
    """(frames, rate, bytes a frame) of a WAV file; frames 0 for one that is missing or unreadable."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes(), w.getframerate(), w.getsampwidth() * w.getnchannels()
    except (OSError, wave.Error, EOFError):
        return 0, 0, 0


def _clip(path: Path, start: float, end: float) -> bytes | None:
    """The stretch [start, end) seconds of a WAV as a WAV file of its own; None when it's empty or silent."""
    try:
        with wave.open(str(path), "rb") as w:
            rate, frames = w.getframerate(), w.getnframes()
            first, last = int(start * rate), min(frames, int(end * rate))
            if last <= first:
                return None
            w.setpos(first)
            pcm = w.readframes(last - first)
            channels, width = w.getnchannels(), w.getsampwidth()
    except (OSError, wave.Error, EOFError) as e:
        log.warning("Meeting minutes: couldn't read %s: %s", path.name, e)
        return None
    if width == 2 and pcm:
        samples = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2")
        if not samples.size or int(np.abs(samples.astype(np.int32)).max()) < SILENT_PEAK:
            return None  # a muted microphone, a silent stretch: nothing to hear, nothing to pay for
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


def _meta(folder: Path) -> dict:
    try:
        meta = json.loads((folder / META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _frame_time(path: Path) -> float:
    """'0003 12m35s.jpg' -> 755 (sst.meeting.screen.frame_name)."""
    m = re.search(r"(\d+)m(\d+)s", path.stem)
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0.0


def pick_frames(paths: list[Path], most: int = MAX_FRAMES, budget: int = FRAME_BYTES) -> list[Path]:
    """At most `most` frames, evenly spread over the meeting (the first and last kept), fewer if their files together
    pass `budget` bytes."""
    paths = sorted(paths)
    count = min(most, len(paths))
    while count > 0:
        if count == 1:
            chosen = [paths[0]]
        else:
            chosen = [paths[round(i * (len(paths) - 1) / (count - 1))] for i in range(count)]
        chosen = list(dict.fromkeys(chosen))
        if sum(_size(p) for p in chosen) <= budget:
            return chosen
        count -= 1
    return []


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


# ---------------------------------------------------------------- the calls

class _Gemini:
    """The calls of one meeting's minutes: the model (the backup's once the main one failed), every answer counted."""

    def __init__(self, key: str, model: str, post_: Callable, upload_: Callable):
        self.key, self.model, self._post, self._upload = key, model, post_, upload_

    def generate(self, parts: list[dict], schema: dict, thinking: str, timeout: float, seconds: float) -> dict:
        while True:
            body = {"contents": [{"role": "user", "parts": parts}],
                    "generationConfig": _config(self.model, schema, thinking)}
            try:
                answer = self._post(f"{BASE}/models/{self.model}:generateContent", body, self.key, timeout)
            except MinutesError as e:
                if e.status == 400 and any("mediaResolution" in p for p in parts):
                    # a part's own resolution isn't taken by every API version: the frames at Google's default
                    log.info("Meeting minutes: %s refused the frames' resolution, sending them without it", self.model)
                    parts = [{k: v for k, v in p.items() if k != "mediaResolution"} for p in parts]
                    continue
                if e.status in FALLBACK_STATUSES and self.model != FALLBACK_MODEL:
                    log.info("Meeting minutes: %s failed (HTTP %d), trying %s", self.model, e.status, FALLBACK_MODEL)
                    self.model = FALLBACK_MODEL
                    continue
                raise
            usage.record(FEATURE, self.model, **usage.gemini_counts(answer, seconds, transcribe=False))
            return _answer_json(answer)

    def upload(self, wav: bytes) -> dict:
        file = self._upload(wav, "audio/wav", self.key, UPLOAD_TIMEOUT + UPLOAD_PER_MB * len(wav) / 1e6)
        if not isinstance(file, dict) or not file.get("uri"):
            raise MinutesError("Gemini's Files API kept no file.")
        return file

    def delete(self, name: str) -> None:
        """An uploaded track off Google at once (it would stay 48 hours). Best effort: never raises."""
        if not name:
            return
        try:
            self._post(f"{BASE}/{name}", None, self.key, DELETE_TIMEOUT, method="DELETE")
        except Exception as e:  # noqa: BLE001 - Google deletes it within 48 hours anyway
            log.info("Meeting minutes: couldn't delete %s from Google: %s", name, e)


@dataclass
class _Transcript:
    lines: list[tuple[float, str, str]] = field(default_factory=list)  # (seconds into the meeting, label, text)
    labels: list[str] = field(default_factory=list)  # in the order they first speak
    names: dict[str, str] = field(default_factory=dict)  # label -> the name said in the meeting
    languages: Counter = field(default_factory=Counter)  # language -> seconds of meeting in it

    def who(self, label: str) -> str:
        return self.names.get(label, label)


def _languages_line(languages: tuple[str, ...]) -> str:
    if not languages:
        return "The meeting may be in Japanese, English, Tamil or another language, or a mix of them.\n"
    names = [LANGUAGE_NAMES.get(code, code) for code in languages]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
    return f"The meeting is expected to be in {listed}, or a mix of them; write any other language as it is spoken.\n"


def _context(done: _Transcript, start: float) -> str:
    """What a stretch after the first is told: where it starts, the people so far, the last lines before it."""
    if start <= 0:
        return ""
    people = ", ".join(f"{label} ({done.names[label]})" if label in done.names else label for label in done.labels)
    text = (f"\nThis is a later part of the meeting: 00:00 in these tracks is {clock(start)} into the meeting."
            f" Times stay from the start of these tracks.")
    if people:
        text += (f" The people so far: {people}. Keep these labels for the same voices; a new voice gets the next "
                 f"number.")
    if done.lines:
        last = "\n".join(f"[{clock(t)}] {label}: {words}" for t, label, words in done.lines[-CARRY_LINES:])
        text += f"\nThe last lines before this part (context only, not to be written again):\n{last}"
    return text


def _spans(total: float, chunk: float = CHUNK_S) -> list[tuple[float, float]]:
    """The meeting in stretches of `chunk` seconds; a last bit under a minute (a quarter chunk) joins the one before."""
    spans, start = [], 0.0
    while start < total:
        end = min(total, start + chunk)
        if total - end < min(60.0, chunk / 4):
            end = total
        spans.append((start, end))
        start = end
    return spans


def _transcribe(gemini: _Gemini, tracks: list[tuple[Path, str]], total: float, languages: tuple[str, ...],
                chunk: float) -> _Transcript:
    done = _Transcript()
    spans = _spans(total, chunk)
    while spans:
        start, end = spans.pop(0)
        try:
            _transcribe_span(gemini, tracks, start, end, done, languages)
        except _TooLong:
            if end - start < 2 * MIN_CHUNK_S:
                raise MinutesError("The meeting's transcript didn't fit in Gemini's answer.") from None
            middle = (start + end) / 2
            spans[:0] = [(start, middle), (middle, end)]
            log.info("Meeting minutes: %s-%s was too long for one answer, asked in halves", clock(start), clock(end))
    return done


def _transcribe_span(gemini: _Gemini, tracks: list[tuple[Path, str]], start: float, end: float, done: _Transcript,
                     languages: tuple[str, ...]) -> None:
    clips = [(clip, about) for path, about in tracks if (clip := _clip(path, start, end))]
    if not clips:
        return  # nobody spoke: nothing to send
    uploaded: list[dict] = []
    try:
        parts: list[dict] = []
        described = []
        for number, (clip, about) in enumerate(clips, 1):
            file = gemini.upload(clip)
            uploaded.append(file)
            described.append(f"Track {number}: {about}.")
            parts += [{"text": f"Track {number}: {about}."},
                      {"fileData": {"mimeType": file.get("mimeType") or "audio/wav", "fileUri": file["uri"]}}]
        prompt = TRANSCRIPT_PROMPT.format(tracks="\n".join(described), languages=_languages_line(languages),
                                          context=_context(done, start))
        data = gemini.generate([*parts, {"text": prompt}], TRANSCRIPT_SCHEMA, LEAST,
                               ANSWER_TIMEOUT + ANSWER_PER_SECOND * (end - start), end - start)
    finally:
        for file in uploaded:
            gemini.delete(str(file.get("name") or ""))
    _add_lines(done, data, start, end)


def _add_lines(done: _Transcript, data: dict, start: float, end: float) -> None:
    """A stretch's answer into the transcript: times into the meeting, labels in order, names learned."""
    length = end - start
    lines = []
    for item in data.get("lines") or ():
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text") or "").split())
        label = " ".join(str(item.get("speaker") or "").split()) or "Speaker"
        if not text:
            continue
        if label.lower() in ("you", "user", "the user"):
            label = YOU_LABEL
        at = _seconds(item.get("time") or "")
        at = min(max(0.0, at if at is not None else 0.0), length)
        lines.append((start + at, label, text))
    lines.sort(key=lambda line: line[0])  # stable: the answer's order within a second
    for line in lines:
        done.lines.append(line)
        if line[1] not in done.labels:
            done.labels.append(line[1])
    for item in data.get("names") or ():
        if not isinstance(item, dict):
            continue
        label, name = str(item.get("label") or "").strip(), " ".join(str(item.get("name") or "").split())
        taken = {n.casefold() for other, n in done.names.items() if other != label} | {YOU_LABEL.casefold()}
        if label and name and label != YOU_LABEL and label not in done.names and name.casefold() not in taken:
            done.names[label] = name  # the first name heard for a voice stays: a later guess doesn't move it
    if lines and (language := str(data.get("language") or "").strip().lower()):
        done.languages[language] += length


def _minutes(gemini: _Gemini, done: _Transcript, meta: dict, frames: list[Path]) -> dict:
    lines = "\n".join(f"[{clock(t)}] {done.who(label)}: {words}" for t, label, words in done.lines) or "(nobody spoke)"
    about = {"app": meta.get("app", ""), "window title": meta.get("title", ""), "started": meta.get("started", ""),
             "length": f"{round(float(meta.get('seconds') or 0) / 60)} min",
             "people": [done.who(label) for label in done.labels]}
    prompt = MINUTES_PROMPT_TEXT.format(
        frames_note=" and screenshots of the meeting window, taken when the screen changed (shared slides, documents)"
        if frames else "", frames_only=" and the screenshots" if frames else "",
        frames_rule=FRAMES_RULE if frames else NO_FRAMES_RULE,
        meeting=json.dumps(about, ensure_ascii=False), transcript=lines)
    parts: list[dict] = [{"text": prompt}]
    for path in frames:
        try:
            data = path.read_bytes()
        except OSError as e:
            log.info("Meeting minutes: frame %s left out: %s", path.name, e)
            continue
        parts += [{"text": f"Screenshot at {clock(_frame_time(path))}:"},
                  {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(data).decode("ascii")},
                   "mediaResolution": {"level": FRAME_RESOLUTION}}]
    return gemini.generate(parts, MINUTES_SCHEMA, MINUTES_THINKING, MINUTES_TIMEOUT, 0.0)


def _strings(items) -> list[str]:
    return [" ".join(str(item).split()) for item in (items or ()) if isinstance(item, str) and item.strip()]


def summarise(folder: Path, key: str, *, post: Callable = post, upload: Callable = upload, model: str = MINUTES_MODEL,
              languages: tuple[str, ...] = (), chunk: float = CHUNK_S) -> MinutesResult:
    """The meeting's transcript and minutes, written into its folder as minutes.json and minutes.md. Raises
    MinutesError in plain words (the recording always kept). post(url, payload, key, timeout, method="POST") -> dict
    and upload(data, mime, key, timeout) -> file dict are Gemini's API (fakes in tests)."""
    folder = Path(folder)
    try:
        return _summarise(folder, key, post, upload, model, languages, chunk)
    except MinutesError as e:
        raise MinutesError(f"{e} {KEPT}", e.status) from None


def _summarise(folder: Path, key: str, post_: Callable, upload_: Callable, model: str, languages: tuple[str, ...],
               chunk: float) -> MinutesResult:
    if not key:
        raise MinutesError("The minutes need a Gemini API key: enter it on the AI & models page.")
    tracks, total = [], 0.0
    for name, about in TRACKS:
        frames, rate, _ = _track_frames(folder / name)
        if frames and rate:
            tracks.append((folder / name, about))
            total = max(total, frames / rate)
    if not tracks:
        raise MinutesError("There is no recording of the meeting to write minutes from.")
    meta = _meta(folder)
    meta.setdefault("seconds", round(total, 1))
    gemini = _Gemini(key, model, post_, upload_)
    done = _transcribe(gemini, tracks, total, languages, chunk)
    frames = pick_frames(list((folder / FRAMES).glob("*.jpg"))) if (folder / FRAMES).is_dir() else []
    data = _minutes(gemini, done, meta, frames)
    actions = [ActionItem(" ".join(str(a.get("task") or "").split()), " ".join(str(a.get("owner") or "").split()),
                          " ".join(str(a.get("due") or "").split()))
               for a in data.get("actions") or () if isinstance(a, dict) and str(a.get("task") or "").strip()]
    spoken = done.languages.most_common(1)[0][0] if done.languages else ""
    result = MinutesResult(
        title=" ".join(str(data.get("title") or "").split()) or "Meeting",
        summary=str(data.get("summary") or "").strip(),
        decisions=_strings(data.get("decisions")),
        actions=actions,
        transcript=[(clock(t), done.who(label), words) for t, label, words in done.lines],
        speakers=[done.who(label) for label in done.labels],
        language=str(data.get("language") or "").strip().lower() or spoken)
    try:
        (folder / MINUTES_JSON).write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
        (folder / MINUTES_MD).write_text(render(result, meta), encoding="utf-8")
    except OSError as e:
        raise MinutesError(f"The minutes couldn't be saved ({e}).") from None
    return result


# ---------------------------------------------------------------- minutes.md

def _when(started: str) -> str:
    try:
        moment = datetime.fromisoformat(str(started))
    except ValueError:
        return str(started or "")
    return f"{moment.day} {moment:%B %Y, %H:%M}"


def render(result: MinutesResult, meta: dict | None = None) -> str:
    """minutes.md: title, date, app, length, people; summary; decisions; action items as a checklist; the transcript."""
    meta = meta or {}
    about = [part for part in (_when(meta.get("started", "")), str(meta.get("app") or ""),
                               f"{round(float(meta.get('seconds') or 0) / 60)} min" if meta.get("seconds") else "")
             if part]
    out = [f"# {result.title}", ""]
    if about:
        out += [" · ".join(about), ""]
    if result.speakers:
        out += [f"People: {', '.join(result.speakers)}", ""]
    out += ["## Summary", "", result.summary or "No summary.", "", "## Decisions", ""]
    out += [f"- {decision}" for decision in result.decisions] or ["None recorded."]
    out += ["", "## Action items", ""]
    for action in result.actions:
        who = ", ".join(part for part in (action.owner, f"due {action.due}" if action.due else "") if part)
        out.append(f"- [ ] {action.task}" + (f" ({who})" if who else ""))
    if not result.actions:
        out.append("None recorded.")
    out += ["", "## Transcript", ""]
    out += [f"**[{at}] {speaker}:** {text}  " for at, speaker, text in result.transcript] or ["Nobody spoke."]
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------- what the minutes cost

def estimate(seconds: float, frames: int, model: str = MINUTES_MODEL) -> float:
    """What the minutes of a meeting this long, with this many kept frames, cost in USD at today's price (0.0 for a
    model with no price): both tracks' audio, the transcript written and read again, the frames, the minutes."""
    if seconds <= 0:
        return 0.0
    price = costs.speech_price_on("gemini", model, date.today().isoformat())
    if price is None or price.per_minute:
        return 0.0
    chunks = len(_spans(seconds))
    thinking = costs.THINKING.get(gemini_thinking(model), 0)
    written = seconds / 60 * TRANSCRIPT_OUT
    audio = len(TRACKS) * seconds * AUDIO_TOKENS
    text_in = (chunks * PROMPT_TOKENS + MINUTES_PROMPT + written * TRANSCRIPT_TEXT
               + min(max(0, frames), MAX_FRAMES) * IMAGE_TOKENS)
    out = written + chunks * thinking + MINUTES_OUT + (costs.THINKING[MINUTES_THINKING] if gemini_thinking(model) else 0)
    return (audio * price.audio + text_in * (price.text or price.audio) + out * price.output) / 1e6
