"""Ready-made setups: a speech model and an AI model that go together, each the cheapest that does the job well.

Most people don't know which models to use (the owner, 2026-10-06). A setup chooses both parts at once: Recommended,
Multilingual, Local, or Custom (the user's own choice of each part). The ids and the estimates come from the research
notes "Rflow model costs and setups" (2026-10-06); sst.costs holds the prices.

  Recommended   Parakeet on this PC                 + Gemini gemini-3.5-flash-lite   a Gemini key   ~$1 a month
  Multilingual  Gemini gemini-3.5-transcribe         + Gemini gemini-3.5-flash-lite   a Gemini key   ~$3.60
  Local         Parakeet or Whisper on this PC       + AI cleanup off (a model on this PC comes later)   free
  (Fastest      Groq whisper-large-v3-turbo          + Groq openai/gpt-oss-20b        a Groq key     ~$0.70)

Rflow offers Gemini only for now (the owner, 2026-10-23: sst.gateway.OFFERED): a setup that would use a provider not
offered (Fastest, Groq's) isn't shown, nor chosen by language, until that provider is offered again.

Those are for English, the language Rflow assumes until the user chooses theirs (How Rflow hears you). Recommended
(and Fastest) choose by the languages spoken (resolve(); phase 44, measured 2026-10-22: Parakeet ~0.8 s after the key
is released on a plugged-in laptop, Gemini ~2.2 s, Groq's Whisper ~0.5 s, as accurate on the owner's English):

  English only  Recommended: as above. Fastest: Parakeet + Groq's AI where the last Check this PC found Parakeet
                quick here (it's slow on battery and weak processors), else Groq for both.
  Other         Parakeet hears English only, so the cloud: the fastest offered model that hears every language well
                (Gemini; Groq's Whisper first while Groq is offered, but not for Tamil), with that provider's AI.
                Recommended takes the fastest one whose key the user has (else asks for the fastest's); Fastest always
                the fastest.

Multilingual, Local and Custom don't change with the language. A setup in use is never changed by this: the choice
is made when a setup is applied, and every model a setup may choose counts as that setup (in_use()).

No Qt here: apply() works on the TrayApp (sst.app), or the window's PreviewApp, through the same calls the pages use.
A setup's key is saved before (TrayApp.save_key, which changes nothing else); apply() then sets the two parts.
"""
from collections.abc import Iterable
from dataclasses import dataclass, replace

from sst import costs
from sst.engines import DEFAULT_MODEL, SPEECH_MODELS
from sst.engines.cloud import CLOUD
from sst.engines.whisper import LANGUAGES
from sst.gateway import PROVIDERS, GatewayConfig, offered


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


def _offered(setup: Setup) -> bool:
    """Whether the window may offer a setup: its speech model and its AI's provider are both offered."""
    return offered(setup.speech) and offered(setup.provider)


_ALL = {s.key: s for s in [
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
SETUPS = {key: setup for key, setup in _ALL.items() if _offered(setup)}  # the ones offered: no Fastest while Groq isn't
ORDER = list(SETUPS)  # Recommended first
LOCAL_MODELS = ("parakeet", "whisper-turbo")  # Local's speech: Parakeet (English) or Whisper (99 languages)
# Gemini Transcribe has no Tamil: Multilingual hears Tamil with Gemini 3.5 Flash-Lite (audio in a prompt), same key.
NO_TRANSCRIBE = {"ta": "gemini-3.5-flash-lite"}

# The models' names as people read them
NAMES = {"gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "openai/gpt-oss-20b": "GPT-OSS 20B",
         "whisper-large-v3-turbo": "Whisper large-v3 turbo", "gemini-3.5-transcribe": "Gemini 3.5 Transcribe",
         "gemini-3.5-transcribe-live": "Gemini 3.5 Transcribe Live"}
SHORT = {"whisper-large-v3-turbo": "Whisper turbo"}  # where a line is short


def spoken(languages: str | Iterable[str] = "") -> tuple[str, ...]:
    """The languages the user speaks, as codes: a speech_language setting ("" = not chosen: English, Rflow's default)
    or several codes."""
    codes = [languages] if isinstance(languages, str) else list(languages)
    return tuple(sorted({code for code in codes if code})) or ("en",)


def speech_model_for(setup: Setup, language: str | Iterable[str] = "") -> str:
    """The setup's cloud speech model for the language(s) spoken (Gemini Transcribe: Flash-Lite for Tamil)."""
    if setup.speech == "gemini" and setup.speech_model == "gemini-3.5-transcribe":
        return next((NO_TRANSCRIBE[code] for code in spoken(language) if code in NO_TRANSCRIBE), setup.speech_model)
    return setup.speech_model


# ---- choosing by the languages spoken

# What Recommended and Fastest may use, besides their own (SETUPS). Free-key words: what each provider does with them.
_GROQ_FREE = _ALL["fastest"].free_key
_GEMINI_VOICE_FREE = SETUPS["multilingual"].free_key
PARAKEET_GROQ = Setup("", "", "", "parakeet", "", "groq", "openai/gpt-oss-20b", "groq", _GROQ_FREE)
GROQ = Setup("", "", "", "groq", "whisper-large-v3-turbo", "groq", "openai/gpt-oss-20b", "groq", _GROQ_FREE)
GEMINI = Setup("", "", "", "gemini", "gemini-3.5-transcribe", "gemini", "gemini-3.5-flash-lite", "gemini",
               _GEMINI_VOICE_FREE)
# The cloud's speech, fastest first (measured after the key is released): Groq's Whisper ~0.5 s, Gemini ~2.2 s. The
# languages each hears badly: Whisper's Tamil was weak on the owner's clips. (GEMINI_LIVE, a streaming Gemini engine,
# can go first here once its speed and languages are measured.) Only the offered ones: Gemini alone for now.
CLOUD_SPEED = tuple(option for option in (GROQ, GEMINI) if _offered(option))
WEAK = {"groq": {"ta"}, "gemini": set()}
SECONDS = {"parakeet": "under a second", "groq": "about half a second", "gemini": "about 2 seconds"}
PARAKEET_QUICK = 1.0  # Fastest uses Parakeet where Check this PC found a sentence takes at most this (s): Groq's pace


def _cloud_for(languages: tuple[str, ...]) -> list[Setup]:
    """The cloud setups that hear every language spoken well, fastest first."""
    return [option for option in CLOUD_SPEED if not WEAK[option.speech] & set(languages)]


def parakeet_quick(scan: dict | None) -> bool:
    """The last Check this PC found Parakeet as quick as the cloud here (never before a check)."""
    verdict = next((v for v in (scan or {}).get("verdicts", []) if v.get("key") == "parakeet"), {})
    return verdict.get("level") in ("recommended", "fast") and verdict.get("seconds", 99.0) <= PARAKEET_QUICK


def choices(key: str, languages: str | Iterable[str] = "") -> list[Setup]:
    """What a setup may use for these languages, in the order it prefers (Local and Custom: themselves)."""
    langs = spoken(languages)
    english = langs == ("en",)
    if key == "recommended":
        return [SETUPS[key]] if english else _cloud_for(langs)
    if key == "fastest":
        return [PARAKEET_GROQ, GROQ] if english else _cloud_for(langs)[:1]
    if key == "multilingual":
        return [GEMINI]
    return [SETUPS[key]]


def _names(langs: tuple[str, ...]) -> str:
    names = [LANGUAGES.get(code, code) for code in langs]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def resolve(key: str, languages: str | Iterable[str] = "", gateway: GatewayConfig | None = None,
            scan: dict | None = None) -> Setup:
    """The setup as it would be used now, for the languages spoken: Recommended's cloud model is the fastest whose key
    the user has (else the fastest's key is asked for); Fastest uses Parakeet for English where this PC runs it quickly
    (`scan`: sst.scan's last result). The name and key stay the setup's; the blurb says what it chose."""
    base, langs = SETUPS[key], spoken(languages)
    options = choices(key, langs)
    if key in ("local", "custom", "multilingual"):
        return base
    if key == "fastest" and langs == ("en",):
        chosen = PARAKEET_GROQ if parakeet_quick(scan) else GROQ
    else:
        chosen = next((o for o in options if key == "recommended" and gateway and gateway.key_for(o.needs)), options[0])
    if chosen.speech == "parakeet":
        blurb = base.blurb if key == "recommended" else \
            f"Parakeet on this PC hears English in {SECONDS['parakeet']} here, and Groq's AI polishes it in a fraction of one."
        if not SPEECH_MODELS["parakeet"].installed():
            blurb += f" Parakeet is a one-time download of {SPEECH_MODELS['parakeet'].size.split(',')[0]}."
    elif key == "fastest" and chosen is GROQ and langs == ("en",):
        blurb = base.blurb
    else:
        speed = SECONDS[chosen.speech]
        why = "Parakeet hears English only, so " if key == "recommended" else ""
        groq = any(option.speech == "groq" for option in CLOUD_SPEED)  # said only while Groq is offered
        weak = " (Groq's Whisper is weak at Tamil)" if groq and chosen.speech != "groq" and "ta" in langs else ""
        blurb = (f"{_names(langs)}: {why}{CLOUD[chosen.speech].name.removeprefix('Google ')} hears it in {speed}"
                 f"{weak}, and its AI polishes it.")
    return replace(chosen, key=key, name=base.name, blurb=blurb)


def for_app(app, key: str) -> Setup:
    """A setup as it would be used now by the TrayApp (or PreviewApp): its language, keys and last Check this PC."""
    return resolve(key, app.settings.speech_language, app.gateway, getattr(app, "last_scan", None))


def uses_line(setup: Setup, languages: str | Iterable[str] = "") -> str:
    """What a setup uses, in a line (the welcome's first step)."""
    if setup.key == "local":
        return "Parakeet or Whisper on this PC · AI cleanup coming soon"
    if setup.key == "custom":
        return "Choose the speech model and the AI yourself"
    ai = model_name(setup.model)
    if setup.speech == "parakeet":
        return f"Parakeet on this PC + {ai}"
    model = speech_model_for(setup, languages)
    if model == setup.model:
        return f"{ai} hears you and polishes the text"
    where = "" if setup.speech == "gemini" else f", both on {CLOUD[setup.speech].name}"  # Gemini's names say it
    return f"{SHORT.get(model, model_name(model))} + {ai}{where}"


def speech_cost(setup: Setup, language: str = "", local: str = "parakeet") -> costs.Estimate:
    speech = local if setup.key == "local" else setup.speech
    return costs.speech_cost(speech, speech_model_for(setup, language))


def cleanup_cost(setup: Setup) -> costs.Estimate:
    return costs.cleanup_cost(setup.provider, setup.model) if setup.provider else costs.Estimate(0.0, costs.FREE)


def cost(setup: Setup, language: str = "", local: str = "parakeet") -> costs.Estimate:
    """What a setup costs a month for typical use: its speech and its AI together."""
    return costs.total(speech_cost(setup, language, local), cleanup_cost(setup))


def chosen_speech(settings, gateway: GatewayConfig) -> str:
    """The speech model chosen, as the window shows it: one of a provider no longer offered shows as the one used in its
    place (sst.engines.usable: Gemini when its key is saved, else Parakeet)."""
    if offered(settings.speech_model):
        return settings.speech_model
    return "gemini" if gateway.key_for("gemini") else DEFAULT_MODEL


def in_use_cost(settings, gateway: GatewayConfig) -> tuple[costs.Estimate, costs.Estimate]:
    """What the speech model and the AI model chosen now cost (Custom shows it): (speech, AI)."""
    speech = chosen_speech(settings, gateway)
    model = settings.speech_cloud_models.get(speech) or (CLOUD[speech].models[0] if speech in CLOUD else "")
    ai = costs.cleanup_cost(gateway.chosen, settings.cleanup_model) if settings.cleanup and settings.cleanup_model \
        else costs.Estimate(0.0, costs.FREE)
    return costs.speech_cost(speech, model), ai


def in_use(settings, gateway: GatewayConfig, pending: str = "") -> list[str]:
    """Every setup the settings match, its own models first (["custom"] when none): two can use the same models (for
    Tamil, Recommended chooses what Multilingual uses). A setup matches with any model it may choose for the language
    (choices()), so adding a key or checking this PC never takes it out of use. `pending`: a speech model being
    downloaded to be used (a setup chosen with Parakeet or Whisper not here yet counts as chosen)."""
    speech = pending or chosen_speech(settings, gateway)
    cloud_model = settings.speech_cloud_models.get(speech) or (CLOUD[speech].models[0] if speech in CLOUD else "")
    cleanup = (gateway.chosen, settings.cleanup_model) if settings.cleanup and settings.cleanup_model else None
    language = settings.speech_language

    def uses(option: Setup) -> bool:
        return speech == option.speech and cleanup == (option.provider, option.model) and \
            (speech not in CLOUD or cloud_model == speech_model_for(option, language))
    own = [key for key, setup in SETUPS.items() if key not in ("local", "custom") and uses(setup)]
    chosen = [key for key in SETUPS if key not in ("local", "custom", *own) and any(map(uses, choices(key, language)))]
    local = ["local"] if speech in LOCAL_MODELS and cleanup is None else []
    return own + chosen + local or ["custom"]


def current(settings, gateway: GatewayConfig, pending: str = "") -> str:
    """The setup the settings match, else "custom" (in_use(): the first)."""
    return in_use(settings, gateway, pending)[0]


def backup_for(gateway: GatewayConfig) -> tuple[GatewayConfig, str] | None:
    """AI cleanup's backup at another provider (sst.gateway.Polisher.routes): the first offered setup's AI whose provider
    isn't the chosen one and whose key the user has, as (that provider's config, its model). None without such a key:
    with Gemini the only cloud provider offered, there is none for Gemini (only for a provider no longer offered)."""
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
    the speech model being downloaded ("" when none). What it uses is chosen now, for the user's language (resolve())."""
    setup = for_app(app, setup_key)
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
