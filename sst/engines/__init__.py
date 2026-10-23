"""Speech recognition: the building block that turns the voice into text, chosen apart from the AI cleanup.

SPEECH_MODELS is the catalog the window shows: each model says where it runs, which languages it knows, its size and
what it is good at. `ready` models can be used in this version; the others are listed as coming next.

An engine has a `name` (its catalog key), a `title` (what reports call it), `transcribe(audio, sample_rate) -> str`,
and a `signature` (what its text depends on: sst.evaluate caches by it). Engines that can use the user's words have a
`words` list, which the app keeps up to date. To add one, create engines/<name>.py and list it below.

Cloud models (sst.engines.cloud) need the user's key for the provider, shared with AI cleanup (sst.gateway). A model on
the user's own server needs its address (GatewayConfig.speech_server).
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sst.downloads import Download
from sst.engines import parakeet, whisper
from sst.engines.cloud import REMOTE, CloudEngine
from sst.gateway import offered

log = logging.getLogger(__name__)


class Engine(Protocol):
    name: str
    title: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


@dataclass(frozen=True)
class SpeechModel:
    key: str
    name: str
    where: str  # a WHERE key
    summary: str  # what it is good at, in one sentence
    languages: str
    size: str
    ready: bool = True  # usable in this version; the others are shown as coming next
    download: Download | None = None  # fetched when chosen (sst.downloads); None = comes with Rflow
    language_choice: bool = False  # the user can choose the language it listens for
    found: Callable[[], bool] | None = None  # also there without the download (Parakeet next to an older Rflow)

    def installed(self) -> bool:
        return self.download is None or self.download.installed() or bool(self.found and self.found())


# Where a speech model runs, in the order the window shows them.
WHERE = {
    "local": "On this computer",
    "cloud": "Cloud",
    "server": "Your own server",
}

SPEECH_MODELS = {m.key: m for m in [
    SpeechModel("parakeet", "NVIDIA Parakeet", "local",
                "Fast and accurate English on any laptop's processor; listens for Your words.",
                "English", f"{parakeet.MODEL.size // 10**6} MB, downloaded when chosen", download=parakeet.MODEL,
                found=lambda: parakeet.find_model() is not None),
    SpeechModel("whisper-turbo", "OpenAI Whisper large-v3 turbo", "local",
                "Many languages, Tamil and Japanese included, and good with names. Without an NVIDIA graphics card "
                "it takes several seconds per sentence (Parakeet about one).",
                "99 languages", "1.6 GB, downloaded when chosen", download=whisper.MODEL, language_choice=True),
    SpeechModel("openai", "OpenAI", "cloud",
                "gpt-transcribe, OpenAI's current transcription model. gpt-4o-mini-transcribe, gpt-4o-transcribe and "
                "whisper-1 are being retired: they work until OpenAI shuts them down on 26 February 2027.",
                "Many languages", "Nothing to download", language_choice=True),
    SpeechModel("groq", "Groq", "cloud",
                "Whisper large-v3 turbo on Groq's servers: the same model as on this computer, back in a fraction of "
                "a second.",
                "99 languages", "Nothing to download", language_choice=True),
    SpeechModel("gemini", "Google Gemini", "cloud",
                "Gemini 3.5 Transcribe writes down exactly what was said, with the time of each word; Gemini's Flash "
                "models are there too. Good with names and mixed languages. Transcribe doesn't know Tamil: for Tamil, "
                "choose gemini-3.5-flash-lite. gemini-3.5-transcribe-live (a preview) transcribes while you speak: the "
                "text is there about half a second after you stop, at about twice the price.",
                "Many languages", "Nothing to download", language_choice=True),
    SpeechModel("server", "Your own server", "server",
                "Whisper or another speech model on a server you run or trust: vLLM, your company's AI gateway, or any "
                "server with OpenAI's transcription API (Speaches, LocalAI...).",
                "Depends on the model", "Nothing to download", language_choice=True),
]}
DEFAULT_MODEL = "parakeet"
ENGINES = [key for key, model in SPEECH_MODELS.items() if model.ready]


def usable(key: str, keys=None) -> str:
    """The model to load for a setting: the chosen one if this version can use it, it is downloaded, a cloud model has
    its key and an own server its address (`keys`: the sst.gateway.GatewayConfig holding them); else Parakeet if it is
    on this computer, so a removed download or a lost key never stops dictation; else "" (no speech model yet: the
    window asks for one). A cloud model of a provider Rflow no longer offers (sst.gateway.OFFERED) gives way to Gemini
    when its key is saved, else to Parakeet."""
    if key in SPEECH_MODELS and not offered(key):
        log.info("Speech model %s is no longer offered: using Gemini or Parakeet instead", key)
        return usable("gemini", keys) if keys and keys.key_for("gemini") else usable(DEFAULT_MODEL, keys)
    model = SPEECH_MODELS.get(key)
    ready = bool(model and model.ready and model.installed())
    if ready and model.where == "cloud":
        ready = bool(keys and keys.key_for(key))
    elif ready and model.where == "server":
        ready = bool(keys and keys.speech_server()[0])
    if ready:
        return key
    return DEFAULT_MODEL if SPEECH_MODELS[DEFAULT_MODEL].installed() else ""


def load_engine(name: str, language: str = "", api_key: str = "", model: str = "",
                fallback: Callable[[], Engine] | None = None, url: str | None = None) -> Engine:
    """`api_key`, `model` and `url` are a cloud model's or an own server's; `fallback` loads the engine used when the
    provider fails."""
    if name == "parakeet":
        from .parakeet import ParakeetEngine

        return ParakeetEngine()
    if name == "whisper-turbo":
        return whisper.WhisperEngine(language=language)
    if name == "gemini":  # streams while a live model is chosen (gemini-3.5-transcribe-live)
        from .gemini_stream import GeminiEngine

        return GeminiEngine(api_key, model, language, fallback)
    if name in REMOTE:
        return CloudEngine(name, api_key, model, language, fallback, url)
    raise ValueError(f"Unknown engine '{name}'. Available: {', '.join(ENGINES)}")
