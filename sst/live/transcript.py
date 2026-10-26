"""The bilingual transcript of a live translation session: one text file per session in the profile's folder, written
line by line as lines finish, so a crash loses at most the line being spoken."""
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from sst.live.contracts import MIC, SYSTEM, Kind, LiveConfig, LiveEvent, language_name

log = logging.getLogger(__name__)


def heading(c: LiveConfig) -> str:
    """What a session translates, in words: "translated into English", "the microphone, translated into Japanese"."""
    into = {lane: language_name(c.for_lane(lane).target) for lane in (SYSTEM, MIC)}
    if c.source == "microphone":
        return f"the microphone, translated into {into[MIC]}"
    if c.source == "both":
        return f"translated into {into[SYSTEM]}; your own speech into {into[MIC]}"
    return f"translated into {into[SYSTEM]}"


def for_copy(text: str, translation_only: bool = False) -> str:
    """A transcript file's text to paste elsewhere: all of it (both languages), or its heading and each line's
    translation alone, at its time ("[10:02:13] You: ..."; a line already in the target language is its own)."""
    if not translation_only:
        return text.strip() + "\n"
    out, entry = [], ""
    for line in text.splitlines():
        if line.startswith("["):  # a line's time, who and the words heard
            if entry:
                out.append(entry)
            entry = line
        elif line.startswith(" ") and entry:  # its translation, under the words heard
            time, _, heard = entry.partition("] ")
            who = "You: " if heard.startswith("You: ") else ""
            entry = f"{time}] {who}{line.strip()}"
        elif not entry and line.strip():
            out.append(line + "\n")  # the heading
    if entry:
        out.append(entry)
    return "\n".join(out) + "\n"


class Transcript:
    """`config` says what's listened to and into which languages (the heading); with both, the user's own lines say
    "You:". Set it again when the source changes during a session."""

    def __init__(self, folder: Path, config: LiveConfig, now: Callable[[], datetime] = datetime.now):
        self.folder, self.config, self._now = folder, config, now
        self.path: Path | None = None  # made with the first finished line: an empty session leaves no file
        self.lines = 0

    def add(self, event: LiveEvent) -> None:
        if event.kind is not Kind.LINE:
            return
        try:
            if self.path is None:
                self.folder.mkdir(parents=True, exist_ok=True)
                started = self._now()
                self.path = self.folder / f"{started:%Y-%m-%d %H-%M-%S} live captions.txt"
                self.path.write_text(f"Rflow live translation, {started:%Y-%m-%d %H:%M}, {heading(self.config)}\n\n",
                                     encoding="utf-8")
            who = "You: " if event.lane == MIC and self.config.marks_mine else ""
            entry = f"[{self._now():%H:%M:%S}] {who}{event.source or '(not heard)'}\n"
            if event.text and event.text != event.source:
                entry += f"           {event.text}\n"
            with self.path.open("a", encoding="utf-8") as out:
                out.write(entry + "\n")
            self.lines += 1
        except OSError as e:  # never stop the translation over a file
            log.warning("Couldn't write the live translation transcript: %s", e)
