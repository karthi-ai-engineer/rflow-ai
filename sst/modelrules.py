"""What the providers' models accept in a request, so that each gets one that works and costs no more than it must.
From the providers' model pages, seen 2026-10-06 (the research notes "Rflow model costs and setups"). For a model these
rules don't know, a value it might refuse is left out (its default always works) or the request stays as it was.

AI cleanup, Text Transform and Translate ask through these (sst.gateway), and so do Gemini's Flash models as speech
models (sst.engines.cloud). Thinking is billed as output and makes the answer slower, and a dictation needs next to none.
Standard library only.
"""
import re

# ---- Google Gemini

# Aliases whose model Google documents, and ids it now answers with a newer model (seen 2026-10-23: gemini-3.5-flash
# and gemini-flash-latest are routed to 3.6 Flash, gemini-3.7-flash to 3.8 Flash). Another alias
# (gemini-flash-lite-latest: its model isn't documented) keeps the model's own thinking and temperature, which always work.
GEMINI_ALIASES = {"gemini-flash-latest": "gemini-3.6-flash", "gemini-3.5-flash": "gemini-3.6-flash",
                  "gemini-3.7-flash": "gemini-3.8-flash"}


def _gemini(model: str) -> tuple[tuple[int, int], str] | None:
    """A Gemini model's version and family, ((3, 5), "flash-lite") for gemini-3.5-flash-lite; None without a version."""
    name = model.strip().lower().removeprefix("models/")
    m = re.fullmatch(r"gemini-(\d+)(?:\.(\d+))?-(.+)", GEMINI_ALIASES.get(name, name))
    return ((int(m.group(1)), int(m.group(2) or 0)), m.group(3)) if m else None


def gemini_thinking(model: str) -> str:
    """How little a Gemini 3 model can be asked to think: "minimal" for its Flash-Lite models and 3.6 Flash, "low" for
    the others ("minimal" is an error on 3.7 and 3.8 Flash); "" for any other model, which keeps its own. Gemini 3's
    Flash models think at "medium" unless told: slower, and 3-4 times the cost."""
    gemini = _gemini(model)
    if gemini is None or gemini[0] < (3, 0):
        return ""
    version, family = gemini
    return "minimal" if family.startswith("flash-lite") or (version == (3, 6) and family.startswith("flash")) else "low"


def gemini_takes_temperature(model: str) -> bool:
    """Whether a model on Gemini's API gets temperature 0, as before: not a Gemini 3 model, nor an alias that may point to
    one. Google advises Gemini 3's default of 1.0, since lower "may lead to unexpected behavior, such as looping"."""
    if not model.strip().lower().removeprefix("models/").startswith("gemini-"):
        return True  # Gemma and the like
    gemini = _gemini(model)
    return gemini is not None and gemini[0] < (3, 0)


# ---- OpenAI (Chat Completions)

# Models that think before answering: o1, o3..., gpt-5 and later. They take only the default temperature (gpt-6's take
# one only at effort "none"), and their thinking counts against the token limit.
OPENAI_REASONING = re.compile(r"^(o\d|gpt-([5-9]|\d{2}))", re.IGNORECASE)
# The least thinking each allows, where it is known. gpt-6-luna, gpt-6-sol and gpt-5.6-luna think at "medium" unless
# told, and allow "none"; gpt-6.1-sol and gpt-6-astra go no lower than "low" (seen 2026-10-23), the first gpt-5 models
# no lower than "minimal". Others keep their default (gpt-5.4-mini and -nano think at "none" anyway; o3 and o4-mini
# aren't documented here; a newer gpt-6 keeps its own rather than be asked for an effort it may refuse).
_OPENAI_LEAST = [(re.compile(pattern, re.IGNORECASE), effort) for pattern, effort in [
    (r"^gpt-6(\.\d+)?-astra", "low"),
    (r"^gpt-6\.1-sol", "low"),
    (r"^gpt-6-(luna|sol)", "none"),
    (r"^gpt-5\.6-luna", "none"),
    (r"^gpt-5(-mini|-nano)?(-\d{4}-\d{2}-\d{2})?$", "minimal"),
]]


def openai_effort(model: str) -> str:
    """The `reasoning_effort` to ask an OpenAI model for: the least it allows, or "" to leave its default."""
    return next((effort for pattern, effort in _OPENAI_LEAST if pattern.search(model.strip())), "")


# ---- Groq: its free and developer keys have no chat model left that doesn't think (llama-3.1-8b-instant and
# llama-3.3-70b-versatile were shut down for them on 2026-08-16)

_GROQ_LEAST = [(re.compile(pattern, re.IGNORECASE), effort) for pattern, effort in [
    (r"gpt-oss", "low"),  # thinks at "medium" unless told, and can't stop: "low" is the least
    (r"qwen", "none"),  # Groq's docs disagree on its default; thinking left on may put <think> text in the answer
]]


def groq_effort(model: str) -> str:
    """The `reasoning_effort` to ask a Groq model for: the least it allows, or "" for a model that doesn't think."""
    return next((effort for pattern, effort in _GROQ_LEAST if pattern.search(model.strip())), "")


# ---- Anthropic

# claude-haiku-4-5, claude-sonnet-4-5-20250929, claude-opus-5-5, claude-fable-5-1, and the older claude-3-5-haiku-20241022
_CLAUDE = re.compile(r"claude-(?:[a-z]+-)?(\d+)(?:-(\d{1,2}))?(?:-[a-z]+)?(?:-\d{8}|-latest)?", re.IGNORECASE)


def claude_version(model: str) -> tuple[int, int] | None:
    """A Claude model's version from its id ((4, 5) for claude-haiku-4-5), or None for an id these rules can't read."""
    m = _CLAUDE.fullmatch(model.strip())
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


# Claude models that answer at once when told to (thinking {"type": "disabled"}): Haiku 5.5 (seen 2026-10-23)
_CLAUDE_THINKING_OFF = re.compile(r"claude-haiku-5-5(?:-\d{8}|-latest)?", re.IGNORECASE)


def claude_thinking_off(model: str) -> bool:
    """Whether a Claude model can be told not to think: then it is, and its answer needs no room for thinking."""
    return bool(_CLAUDE_THINKING_OFF.fullmatch(model.strip()))


def claude_takes_temperature(model: str) -> bool:
    """Whether a Claude model accepts temperature 0. Claude 4.7 and later (Sonnet 5, Opus 4.7 to 5.5, Fable) answer
    HTTP 400 to any sampling value but the default; some of them think whether asked or not (Opus 5.5)."""
    version = claude_version(model)
    return version is not None and version < (4, 7)
