"""The live pipeline's contracts: its settings, and the events every engine produces.

Every engine (Gemini Live Translate now; a streaming recognizer with a clause translator later) turns audio frames into
the same stream of events, so the caption bar, the transcript and the measurements are written once:

    SOURCE      the words heard in the current line so far (they may still change)
    TRANSLATION its translation so far (may still change)
    LINE        the line is finished: its source and translation are final, and the next line starts
    STATUS      connecting, listening, reconnecting, stopped
    ERROR       something went wrong; the text says what, in plain words
    NOTE        why a way shows nothing yet, in plain words (the bar's quiet status line): listening, nothing heard,
                speech already in the target language, a capture's own finding; "" clears it. Made by the session
                (LiveSession.note), never by an engine
"""
import dataclasses
import enum
import time
from dataclasses import dataclass, field

RATE = 16_000  # what the live translation models take: 16 kHz mono, 16-bit
FRAME_MS = 100  # the frame Google recommends for latency: 3,200 bytes at 16 kHz
SYSTEM, MIC = "system", "mic"  # the two ways (lanes): what the laptop plays, and what the microphone hears
# What live translation listens to, as the user chooses it, and the ways each one runs
SOURCES: dict[str, tuple[str, ...]] = {"computer": (SYSTEM,), "microphone": (MIC,), "both": (SYSTEM, MIC)}
# The cost saver (LiveConfig.save): a session with no words this long stops by itself, after a warning this long before
AUTO_STOP_S = 900.0
AUTO_STOP_WARN_S = 120.0


class Kind(enum.Enum):
    SOURCE = "source"
    TRANSLATION = "translation"
    LINE = "line"
    STATUS = "status"
    ERROR = "error"
    NOTE = "note"


@dataclass(frozen=True)
class LiveEvent:
    kind: Kind
    text: str = ""  # SOURCE/TRANSLATION: the whole line so far; LINE: the translation; STATUS/ERROR: the message
    source: str = ""  # LINE: the original words of the finished line
    language: str = ""  # the language the engine says the text is in (BCP-47), if it says
    lane: str = SYSTEM  # SYSTEM: what the laptop plays; MIC: what the microphone hears
    seconds: float = 0.0  # LINE: how long after the voice paused its translation was complete (0 = it didn't pause)
    at: float = field(default_factory=time.monotonic)


@dataclass
class LiveConfig:
    target: str = "en"  # the language captions are translated into (BCP-47)
    echo_target: bool = False  # speech already in the target language: shown as heard, not repeated by the model
    model: str = "gemini-3.5-live-translate-preview"
    line_pause_s: float = 1.5  # a line with no new words for this long is finished (the model may not say so)
    reconnect_s: float = 540.0  # a Live API connection lasts ~10 min: a new one is opened before that
    caption_lines: int = 2  # finished translation lines kept on screen above the one being spoken
    hide_from_capture: bool = True  # the caption bar isn't in screen shares and recordings
    source: str = "computer"  # a key of SOURCES: what's translated (the laptop, the microphone, or both)
    mic_target: str = "ja"  # the language what the microphone hears is translated into
    speak: bool = False  # the translation spoken aloud too, by a voice on the laptop (sst.live.speaker)
    speak_speed: float = 1.0
    duck: float = 0.1  # how loud the other apps stay while live translation speaks (1.0: as they are; sst.live.ducking)
    # The cost saver: audio goes to Google only while someone talks (sst.live.gate; Google bills every second streamed,
    # both ways), and a session with no words for AUTO_STOP_S stops. Off: silence is left out as it always was
    save: bool = False
    # Why the bar is still empty (its status line, sst.live.session), judged on the last note_after_s seconds heard: a
    # frame quieter than quiet_dbfs is near silence (a muted microphone is all zeros, a quiet room about -60 dBFS); with
    # sound in sound_share of those seconds or more, it's sound without speech, else nothing heard
    note_after_s: float = 8.0
    quiet_dbfs: float = -55.0
    sound_share: float = 0.25

    @property
    def lanes(self) -> tuple[str, ...]:
        return SOURCES.get(self.source, SOURCES["computer"])

    @property
    def marks_mine(self) -> bool:
        """With both, the microphone is the user in an online meeting: their lines are marked "You". With the
        microphone alone it hears everyone in the room, so nothing is marked."""
        return self.source == "both"

    def spoken_lanes(self, language: str | None = None) -> tuple[str, ...]:
        """The ways a voice speaking `language` reads aloud: what's translated into that language (None: any, the
        engine's own voice). With both, only the others' words: the user's own are for the people they talk to."""
        lanes = (SYSTEM,) if self.source == "both" else self.lanes
        return tuple(lane for lane in lanes if language is None or self.for_lane(lane).target == language)

    def for_lane(self, lane: str) -> "LiveConfig":
        """The settings one way's engine works with: what the microphone hears goes into mic_target."""
        return dataclasses.replace(self, target=self.mic_target) if lane == MIC else self


# The languages captions can be translated into: the name shown, and the code the model takes. Gemini 3.5 Live
# Translate knows 70+; these are the ones the owner's laptops and colleagues need first, the rest can follow.
LANGUAGES: dict[str, str] = {
    "English": "en", "Japanese": "ja", "Tamil": "ta", "Hindi": "hi", "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW", "Korean": "ko", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Dutch": "nl", "Russian": "ru", "Arabic": "ar", "Telugu": "te", "Malayalam": "ml",
    "Kannada": "kn", "Bengali": "bn", "Marathi": "mr", "Thai": "th", "Vietnamese": "vi", "Indonesian": "id",
    "Turkish": "tr", "Polish": "pl",
}


def language_name(code: str) -> str:
    return next((name for name, c in LANGUAGES.items() if c == code), code)


def already_in(event: LiveEvent, target: str) -> bool:
    """A finished line that came back untranslated because it's in `target` already: words heard, no translation, and
    the engine says they're in the target's language (or doesn't say). Words in another language without a translation
    are a line cut before its translation came, not this."""
    same = not event.language or event.language.split("-")[0].lower() == target.split("-")[0].lower()
    return event.kind is Kind.LINE and not event.text and bool(event.source) and same
