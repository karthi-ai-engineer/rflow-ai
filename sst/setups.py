"""Ready-made setups: a speech model and an AI model that go together, each the cheapest that does the job well.

Most people don't know which models to use (the owner, 2026-10-06). A setup chooses both parts at once: Recommended,
Fastest, Multilingual, Local, or Custom (the user's own choice of each part). The ids and the estimates come from the
research notes "Rflow model costs and setups" (2026-10-06); sst.costs holds the prices.

  Recommended   Parakeet on this PC                 + Gemini gemini-3.5-flash-lite   a Gemini key   ~$1 a month
  Fastest       Groq whisper-large-v3-turbo          + Groq openai/gpt-oss-20b        a Groq key     ~$0.70
  Multilingual  Gemini gemini-3.5-transcribe         + Gemini gemini-3.5-flash-lite   a Gemini key   ~$3.60
  Local         Parakeet or Whisper on this PC       + AI cleanup off (a model on this PC comes later)   free

No Qt here: apply() works on the TrayApp (sst.app), or the window's PreviewApp, through the same calls the pages use.
A setup's key is saved before (TrayApp.save_key, which changes nothing else); apply() then sets the two parts.
"""
from dataclasses import dataclass

from sst import costs
from sst.engines import SPEECH_MODELS
from sst.engines.cloud import CLOUD
from sst.gateway import PROVIDERS, GatewayConfig


@dataclass(frozen=True)
class Setup:
    key: str
    name: str
    blurb: str  # who it's for, in a sentence
    speech: str = ""  # a SPEECH_MODELS key; "" = the user's choice (Custom)
    speech_model: str = ""  # the cloud model for speech
    provider: str = ""  # the AI cleanup's provider (sst.gateway.PROVIDERS); "" = none (Local) or the user's (Custom)
    model: str = ""  # its model
    needs: str = ""  # the provider whose API key it needs ("" = none)
    free_key: str = ""  # what a free key means, said plainly


SETUPS = {s.key: s for s in [
    Setup("recommended", "Recommended", "Private speech on this PC, polished by a fast, cheap AI.",
          "parakeet", "", "gemini", "gemini-3.5-flash-lite", "gemini",
          "A free Gemini key costs nothing, but Google may use the text to improve its products and people may read "
          "it: use a paid key for work."),
    Setup("fastest", "Fastest", "Speech and AI on Groq's servers: back in a fraction of a second, on any PC.",
          "groq", "whisper-large-v3-turbo", "groq", "openai/gpt-oss-20b", "groq",
          "A free Groq key covers typical use, and Groq doesn't train on what you send, free or paid."),
    Setup("multilingual", "Multilingual", "Gemini Transcribe: 83 languages, Japanese and Hindi too, and mixed ones.",
          "gemini", "gemini-3.5-transcribe", "gemini", "gemini-3.5-flash-lite", "gemini",
          "A free Gemini key costs nothing, but Google may use your voice and text to improve its products and people "
          "may read them: use a paid key for work."),
    Setup("local", "Local", "Everything on this PC: nothing leaves it, no key, no cost.", "parakeet"),
    Setup("custom", "Custom", "Choose the speech model and the AI yourself."),
]}
ORDER = list(SETUPS)  # Recommended first
LOCAL_MODELS = ("parakeet", "whisper-turbo")  # Local's speech: Parakeet (English) or Whisper (99 languages)
# Gemini Transcribe has no Tamil: Multilingual hears Tamil with Gemini 3.5 Flash-Lite (audio in a prompt), same key.
NO_TRANSCRIBE = {"ta": "gemini-3.5-flash-lite"}

# The models' names as people read them
NAMES = {"gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "openai/gpt-oss-20b": "GPT-OSS 20B",
         "whisper-large-v3-turbo": "Whisper large-v3 turbo", "gemini-3.5-transcribe": "Gemini 3.5 Transcribe"}


def speech_model_for(setup: Setup, language: str = "") -> str:
    """The setup's cloud speech model for the language spoken (Multilingual: Flash-Lite for Tamil)."""
    if setup.key == "multilingual":
        return NO_TRANSCRIBE.get(language, setup.speech_model)
    return setup.speech_model


def speech_cost(setup: Setup, language: str = "", local: str = "parakeet") -> costs.Estimate:
    speech = local if setup.key == "local" else setup.speech
    return costs.speech_cost(speech, speech_model_for(setup, language))


def cleanup_cost(setup: Setup) -> costs.Estimate:
    return costs.cleanup_cost(setup.provider, setup.model) if setup.provider else costs.Estimate(0.0, costs.FREE)


def cost(setup: Setup, language: str = "", local: str = "parakeet") -> costs.Estimate:
    """What a setup costs a month for typical use: its speech and its AI together."""
    return costs.total(speech_cost(setup, language, local), cleanup_cost(setup))


def in_use_cost(settings, gateway: GatewayConfig) -> tuple[costs.Estimate, costs.Estimate]:
    """What the speech model and the AI model chosen now cost (Custom shows it): (speech, AI)."""
    speech = settings.speech_model
    model = settings.speech_cloud_models.get(speech) or (CLOUD[speech].models[0] if speech in CLOUD else "")
    ai = costs.cleanup_cost(gateway.chosen, settings.cleanup_model) if settings.cleanup and settings.cleanup_model \
        else costs.Estimate(0.0, costs.FREE)
    return costs.speech_cost(speech, model), ai


def current(settings, gateway: GatewayConfig, pending: str = "") -> str:
    """The setup the settings match, else "custom". `pending`: a speech model being downloaded to be used (a setup
    chosen with Parakeet or Whisper not here yet counts as chosen)."""
    speech = pending or settings.speech_model
    cloud_model = settings.speech_cloud_models.get(speech) or (CLOUD[speech].models[0] if speech in CLOUD else "")
    cleanup = (gateway.chosen, settings.cleanup_model) if settings.cleanup and settings.cleanup_model else None
    for setup in SETUPS.values():
        if setup.key == "custom":
            continue
        if setup.key == "local":
            if speech in LOCAL_MODELS and cleanup is None:
                return "local"
            continue
        if speech != setup.speech or cleanup != (setup.provider, setup.model):
            continue
        if speech in CLOUD and cloud_model != speech_model_for(setup, settings.speech_language):
            continue
        return setup.key
    return "custom"


def backup_for(gateway: GatewayConfig) -> tuple[GatewayConfig, str] | None:
    """AI cleanup's backup at another provider (sst.gateway.Polisher.routes): the first setup's AI whose provider isn't
    the chosen one and whose key the user has, as (that provider's config, its model). None without such a key."""
    for setup in SETUPS.values():
        if setup.provider and setup.provider != gateway.service.key and (key := gateway.key_for(setup.provider)):
            return GatewayConfig(gateway.entries().get(setup.provider, ("", ""))[0], key, setup.provider), setup.model
    return None


def key_missing(setup: Setup, gateway: GatewayConfig) -> str:
    """The provider whose key the setup still needs ("" when it has it, or needs none)."""
    return setup.needs if setup.needs and not gateway.key_for(setup.needs) else ""


def apply(app, setup_key: str, local: str = "parakeet") -> str:
    """Use a setup: its speech model (downloaded first through the usual flow when it isn't on this PC: it's used
    once it's here), then its AI (Local: AI cleanup off, its model kept for later). The key is saved before. Returns
    the speech model being downloaded ("" when none)."""
    setup = SETUPS[setup_key]
    if setup.key == "custom":
        return ""
    s = app.settings
    speech = local if setup.key == "local" else setup.speech
    model = SPEECH_MODELS[speech]
    downloading = ""
    if app.downloading and app.downloading[0] != speech:
        app.cancel_download()  # it would be used once it's here (what is downloaded so far stays for another time)
    if model.where == "cloud":
        app.use_cloud_speech(speech, app.gateway.key_for(speech), speech_model_for(setup, s.speech_language))
    elif model.installed():
        app.choose_speech_model(speech)
    else:
        if not (app.downloading and app.downloading[0] == speech):
            app.download_speech_model(speech)
        downloading = speech
    s, gateway = app.settings, app.gateway
    if setup.provider:
        p = PROVIDERS[setup.provider]
        chosen = GatewayConfig("", gateway.key_for(p.key), p.key,
                               {name: entry for name, entry in gateway.entries().items() if name != p.key})
        fallback = s.cleanup_fallback if gateway.chosen == p.key else ""  # a backup model for another provider goes
        app.save_cleanup(True, setup.model, fallback, chosen)
    elif s.cleanup:
        app.save_cleanup(False, s.cleanup_model, s.cleanup_fallback, gateway)
    return downloading


def model_name(model: str) -> str:
    return NAMES.get(model, model)


# What the last Scan this PC (sst.scan) says about a model on this PC, in plain words
CAN = {"recommended": "runs {name} well", "fast": "runs {name} well", "usable": "runs {name}, with a short wait",
       "slow": "runs {name} slowly", "no": "can't run {name}"}


def pc_check(scan: dict | None, key: str) -> tuple[str, str]:
    """(level, a sentence) for a model on this PC from the last scan's verdict: ("", "") before a scan. The level is
    sst.scan's: recommended, fast, usable, slow or no."""
    for verdict in (scan or {}).get("verdicts", []):
        if verdict.get("key") == key and verdict.get("level") in CAN:
            level, name = verdict["level"], SPEECH_MODELS[key].name
            words = f"This PC {CAN[level].format(name=name)}: {verdict.get('reason', '')}."
            if level == "slow" and key != "parakeet":
                words += " Parakeet is quicker for English."
            return level, words
    return "", ""
