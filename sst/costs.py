"""What the models cost: the price table, the typical user, and an estimate a month with its tier.

Most people don't know which model to use, or what one costs until the bill comes. The window shows "about $X a month
for typical use" next to every speech and AI model it knows, marks the mid-priced ones amber and the expensive ones red,
and the ready-made setups (sst.setups) add their two parts up.

Prices: USD, paid tier, standard (not batch) pricing, as seen on each provider's official page on 2026-10-06 (the
research notes "Rflow model costs and setups"). They change: Gemini 3.6 Flash doubles on 2027-01-01, and OpenAI's
gpt-5.6-sol price is a promotion. A model not in the table has no estimate ("price unknown"), never a red mark.

  Google Gemini  https://ai.google.dev/gemini-api/docs/pricing (models, thinking: .../docs/models, .../docs/thinking)
  OpenAI         https://developers.openai.com/api/docs/pricing (deprecations: .../api/docs/deprecations)
  Groq           https://console.groq.com/docs/models (groq.com/pricing shows no prices any more; the 10 s minimum:
                 https://console.groq.com/docs/speech-to-text)
  Anthropic      https://platform.claude.com/docs/en/about-claude/pricing
"""
import re
from dataclasses import dataclass

SEEN = "2026-10-06"  # the day every price below was read on the provider's own page

# ---------------------------------------------------------------- the typical user


@dataclass(frozen=True)
class Usage:
    """A month of dictation. The typical user (the notes' assumption): 150 dictations of about 8 s a day (20 minutes),
    22 days a month, plus 20 Text Transform or Translate uses a day."""

    dictations: float = 150 * 22
    seconds: float = 8.0  # of speech in a dictation
    sent: float = 9.0  # seconds of audio sent for it: the pipeline's chunks overlap by 1 s
    requests: float = 1.15  # speech requests for it: a long dictation is sent as a few chunks
    tool_uses: float = 20 * 22  # Text Transform and Translate, on the AI model

    @property
    def minutes_sent(self) -> float:
        return self.dictations * self.sent / 60  # 495 for the typical user

    @property
    def speech_requests(self) -> float:
        return self.dictations * self.requests  # 3,795


TYPICAL = Usage()
TYPICAL_WORDS = "typical use: 20 minutes of dictation a day"

# Tokens of the AI model, measured from Rflow's own prompts at about 4 characters a token: a cleanup sends the prompt
# (~260 tokens) and the dictation (~27), and gets the text back; Text Transform and Translate send more and write more
# (rounded up for a repair now and then). Together 1.25M input and 0.19M output tokens a month for the typical user.
CLEANUP_TOKENS = (300, 30)  # (input, output) per dictation
TOOL_TOKENS = (600, 200)  # per Text Transform or Translate use
# Hidden reasoning is billed as output. Extra output tokens per cleanup at each thinking level (twice as many for Text
# Transform and Translate): the notes' assumptions, to be measured with `usage` on real calls.
THINKING = {"": 0, "minimal": 20, "low": 100, "medium": 300, "high": 800}

# Speech through a Gemini model: Transcribe bills 25 audio tokens a second and writes ~175 tokens a minute (Google's
# own figures); a Flash model in a prompt bills 32 audio tokens a second, reads Rflow's instruction and Your words
# (~110 tokens a request) and writes ~200 tokens a minute. These reproduce the notes' estimates within a few cents.
TRANSCRIBE_AUDIO, FLASH_AUDIO = 25, 32  # tokens per second of audio
TRANSCRIBE_OUT, FLASH_OUT = 175, 200  # tokens written per minute of audio
FLASH_PROMPT = 110  # tokens of instruction and hints per request

# ---------------------------------------------------------------- the price table


@dataclass(frozen=True)
class TextPrice:
    """An AI model for cleanup, Text Transform and Translate: USD per 1M tokens."""

    input: float
    output: float  # thinking included
    thinking: str = ""  # how much it reasons as Rflow asks (THINKING); "" = not at all
    tokens: float = 1.0  # its tokenizer: Claude 4.7 and later count ~30% more tokens for the same text


@dataclass(frozen=True)
class SpeechPrice:
    """A cloud speech model: a price a minute (OpenAI, Groq), or by the token (Gemini)."""

    per_minute: float = 0.0
    minimum: float = 0.0  # seconds billed at least per request (Groq: 10 s)
    audio: float = 0.0  # Gemini: USD per 1M audio tokens
    text: float = 0.0  # per 1M text tokens in (the instruction and hints)
    output: float = 0.0  # per 1M tokens out, thinking included
    transcribe: bool = False  # Gemini Transcribe (no instruction, Google's token rates) rather than a Flash model
    thinking: str = ""


_CLAUDE_NEW = 1.3  # Claude 4.7 and later: the same text is ~30% more tokens

# (provider, model) -> price. Thinking as Rflow asks: each model's default, except gpt-oss, which Rflow asks for "low"
# (it can't be switched off; Groq's default is medium).
TEXT: dict[tuple[str, str], TextPrice] = {
    # Google Gemini (thinking billed as output; "Output price (including thinking tokens)")
    ("gemini", "gemini-2.5-flash-lite"): TextPrice(0.10, 0.40),  # limited to existing users since 2026-09-18
    ("gemini", "gemini-3.1-flash-lite"): TextPrice(0.25, 1.50, "minimal"),  # deprecated, shuts down 2027-05-07
    ("gemini", "gemini-3.5-flash-lite"): TextPrice(0.30, 2.50, "minimal"),
    ("gemini", "gemini-2.5-flash"): TextPrice(0.30, 2.50, "low"),  # dynamic thinking: assumed low
    ("gemini", "gemini-3-flash-preview"): TextPrice(0.50, 3.00, "high"),
    ("gemini", "gemini-3.6-flash"): TextPrice(0.75, 3.75, "medium"),  # $1.50 / $7.50 from 2027-01-01
    ("gemini", "gemini-3.7-flash"): TextPrice(0.75, 3.75, "medium"),
    ("gemini", "gemini-3.8-flash"): TextPrice(0.75, 3.75, "medium"),
    ("gemini", "gemini-3.5-flash"): TextPrice(1.50, 9.00, "medium"),
    ("gemini", "gemini-flash-latest"): TextPrice(1.50, 9.00, "medium"),  # an alias: 3.5 Flash today
    ("gemini", "gemini-2.5-pro"): TextPrice(1.25, 10.00, "medium"),  # thinking can't be turned off
    ("gemini", "gemini-3.1-pro-preview"): TextPrice(2.00, 12.00, "high"),  # paid only
    # OpenAI (reasoning tokens billed as output)
    ("openai", "gpt-5-nano"): TextPrice(0.05, 0.40, "medium"),
    ("openai", "gpt-6-luna"): TextPrice(0.10, 0.50, "medium"),
    ("openai", "gpt-4.1-nano"): TextPrice(0.10, 0.40),  # shuts down 2026-10-23
    ("openai", "gpt-4o-mini"): TextPrice(0.15, 0.60),
    ("openai", "gpt-5.6-luna"): TextPrice(0.20, 1.20, "medium"),
    ("openai", "gpt-5.4-nano"): TextPrice(0.20, 1.25),
    ("openai", "gpt-5-mini"): TextPrice(0.25, 2.00, "medium"),
    ("openai", "gpt-4.1-mini"): TextPrice(0.40, 1.60),
    ("openai", "gpt-5.4-mini"): TextPrice(0.75, 4.50),
    ("openai", "gpt-5"): TextPrice(1.25, 10.00, "medium"),
    ("openai", "gpt-5.1"): TextPrice(1.25, 10.00, "medium"),
    ("openai", "gpt-4.1"): TextPrice(2.00, 8.00),
    ("openai", "o3"): TextPrice(2.00, 8.00, "medium"),
    ("openai", "gpt-6-sol"): TextPrice(2.00, 10.00, "medium"),
    ("openai", "gpt-6.1-sol"): TextPrice(2.00, 10.00, "medium"),
    ("openai", "gpt-4o"): TextPrice(2.50, 10.00),
    ("openai", "gpt-5.4"): TextPrice(2.50, 15.00),
    ("openai", "gpt-5.6-sol"): TextPrice(4.00, 20.00, "medium"),  # a promotion
    ("openai", "gpt-5.5"): TextPrice(5.00, 30.00),
    ("openai", "chat-latest"): TextPrice(5.00, 30.00),
    ("openai", "gpt-6-astra"): TextPrice(10.00, 50.00, "medium"),
    ("openai", "o1"): TextPrice(15.00, 60.00, "medium"),  # shuts down 2026-10-23
    # Groq (the console's models page)
    ("groq", "openai/gpt-oss-20b"): TextPrice(0.075, 0.30, "low"),
    ("groq", "openai/gpt-oss-120b"): TextPrice(0.15, 0.60, "low"),
    ("groq", "qwen/qwen3.8-27b"): TextPrice(0.80, 4.00),  # Preview
    # Anthropic
    ("anthropic", "claude-haiku-4-5"): TextPrice(1.00, 5.00),
    ("anthropic", "claude-sonnet-4-5"): TextPrice(3.00, 15.00),  # deprecated, retires 2026-11-30
    ("anthropic", "claude-sonnet-4-6"): TextPrice(3.00, 15.00),
    ("anthropic", "claude-sonnet-5"): TextPrice(2.00, 10.00, "medium", _CLAUDE_NEW),
    ("anthropic", "claude-sonnet-5-5"): TextPrice(2.00, 10.00, "medium", _CLAUDE_NEW),
    ("anthropic", "claude-opus-4-5"): TextPrice(5.00, 25.00),
    ("anthropic", "claude-opus-4-6"): TextPrice(5.00, 25.00),
    ("anthropic", "claude-opus-4-7"): TextPrice(5.00, 25.00, "", _CLAUDE_NEW),
    ("anthropic", "claude-opus-4-8"): TextPrice(5.00, 25.00, "", _CLAUDE_NEW),
    ("anthropic", "claude-opus-5"): TextPrice(5.00, 25.00, "", _CLAUDE_NEW),
    ("anthropic", "claude-opus-5-5"): TextPrice(4.00, 20.00, "medium", _CLAUDE_NEW),  # thinking can't be turned off
    ("anthropic", "claude-fable-5"): TextPrice(10.00, 50.00, "medium", _CLAUDE_NEW),
    ("anthropic", "claude-fable-5-1"): TextPrice(10.00, 50.00, "medium", _CLAUDE_NEW),
}

SPEECH: dict[tuple[str, str], SpeechPrice] = {
    # OpenAI, by the minute (the three older models shut down 2027-02-26; gpt-transcribe replaces them)
    ("openai", "gpt-4o-mini-transcribe"): SpeechPrice(per_minute=0.003),
    ("openai", "gpt-transcribe"): SpeechPrice(per_minute=0.0045),
    ("openai", "gpt-4o-transcribe"): SpeechPrice(per_minute=0.006),
    ("openai", "gpt-4o-transcribe-diarize"): SpeechPrice(per_minute=0.006),
    ("openai", "whisper-1"): SpeechPrice(per_minute=0.006),
    ("openai", "gpt-live-transcribe"): SpeechPrice(per_minute=0.017),  # the Realtime API only
    # Groq: $0.04 and $0.111 an hour, and at least 10 s billed for each request
    ("groq", "whisper-large-v3-turbo"): SpeechPrice(per_minute=0.04 / 60, minimum=10),
    ("groq", "whisper-large-v3"): SpeechPrice(per_minute=0.111 / 60, minimum=10),
    # Google Gemini, by the token
    ("gemini", "gemini-3.5-transcribe"): SpeechPrice(audio=2.00, output=12.00, transcribe=True),
    ("gemini", "gemini-3.5-transcribe-live"): SpeechPrice(audio=3.50, output=21.00, transcribe=True),  # the Live API
    ("gemini", "gemini-2.5-flash-lite"): SpeechPrice(audio=0.30, text=0.10, output=0.40),
    ("gemini", "gemini-3.1-flash-lite"): SpeechPrice(audio=0.50, text=0.25, output=1.50, thinking="minimal"),
    ("gemini", "gemini-3.5-flash-lite"): SpeechPrice(audio=0.30, text=0.30, output=2.50, thinking="minimal"),
    ("gemini", "gemini-3.6-flash"): SpeechPrice(audio=0.75, text=0.75, output=3.75, thinking="medium"),
    ("gemini", "gemini-3.7-flash"): SpeechPrice(audio=0.75, text=0.75, output=3.75, thinking="medium"),
    ("gemini", "gemini-3.8-flash"): SpeechPrice(audio=0.75, text=0.75, output=3.75, thinking="medium"),
    ("gemini", "gemini-3.5-flash"): SpeechPrice(audio=1.50, text=1.50, output=9.00, thinking="medium"),
    ("gemini", "gemini-flash-latest"): SpeechPrice(audio=1.50, text=1.50, output=9.00, thinking="medium"),
    # gemini-flash-lite-latest: Google documents no target for it any more, so no price
}

LOCAL_SPEECH = ("parakeet", "whisper-turbo")  # on this PC: free
LOCAL_AI = ("ollama",)  # an AI model on this PC: free
# The cheapest models that do the job well, for each role: the Fastest setup's two (Groq). A model costing more than
# 10 times as much is marked red, whatever it costs.
CHEAPEST_GOOD = {"speech": ("groq", "whisper-large-v3-turbo"), "cleanup": ("groq", "openai/gpt-oss-20b")}

# ---------------------------------------------------------------- the estimate

FREE, LOW, AMBER, RED, UNKNOWN = "free", "low", "amber", "red", "unknown"
AMBER_FROM, RED_FROM, TIMES = 2.0, 5.0, 10.0  # $2-5 a month amber, over $5 red; or over 10 times the cheapest good one


@dataclass(frozen=True)
class Estimate:
    """What something costs a month for the typical user (None: price unknown), its tier, and how many times the
    cheapest good option for its role that is (0 when free, unknown or not compared)."""

    monthly: float | None
    tier: str
    times: float = 0.0

    @property
    def words(self) -> str:
        """For a line next to a model: "About $1.05 a month", "Free", "Price unknown"."""
        if self.tier == FREE:
            return "Free"
        if self.monthly is None:
            return "Price unknown"
        return f"About {money(self.monthly)} a month"

    @property
    def short(self) -> str:
        """For a model list: "≈ $1.05/mo", "free", "price unknown"."""
        if self.tier == FREE:
            return "free"
        if self.monthly is None:
            return "price unknown"
        return f"≈ {money(self.monthly)}/mo"

    @property
    def warning(self) -> str:
        """The line under an amber or red model ("" for the others)."""
        if self.tier not in (AMBER, RED) or self.monthly is None:
            return ""
        about = f"About {money(self.monthly)} a month for typical use"
        if self.tier == AMBER:
            return f"{about}: a mid-priced model."
        if self.times > TIMES:
            return f"{about}: more than {TIMES:.0f} times the cheapest option."
        return f"{about}: an expensive model, over {money(RED_FROM)} a month."


def money(amount: float) -> str:
    """$0.42, $1.05, $13: cents below $10, whole dollars above."""
    if amount >= 10:
        return f"${amount:.0f}"
    return f"${amount:.2f}"


def tier(monthly: float | None, cheapest: float = 0.0) -> Estimate:
    if monthly is None:
        return Estimate(None, UNKNOWN)
    if monthly <= 0:
        return Estimate(0.0, FREE)
    times = monthly / cheapest if cheapest > 0 else 0.0
    if monthly > RED_FROM or times > TIMES:
        return Estimate(monthly, RED, times)
    return Estimate(monthly, AMBER if monthly >= AMBER_FROM else LOW, times)


_SNAPSHOT = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")  # a dated snapshot: gpt-4.1-mini-2025-04-14, claude-...-20251001


def _find(table: dict, provider: str, model: str):
    """A model's price: by its id as the provider lists it, or as a dated snapshot of a model in the table."""
    name = model.strip().removeprefix("models/").lower()
    return table.get((provider, name)) or table.get((provider, _SNAPSHOT.sub("", name)))


def text_monthly(price: TextPrice, usage: Usage = TYPICAL) -> float:
    extra = THINKING.get(price.thinking, 0)
    tokens_in = usage.dictations * CLEANUP_TOKENS[0] + usage.tool_uses * TOOL_TOKENS[0]
    tokens_out = usage.dictations * (CLEANUP_TOKENS[1] + extra) + usage.tool_uses * (TOOL_TOKENS[1] + 2 * extra)
    return (tokens_in * price.input + tokens_out * price.output) * price.tokens / 1e6


def speech_monthly(price: SpeechPrice, usage: Usage = TYPICAL) -> float:
    minutes, requests = usage.minutes_sent, usage.speech_requests
    if price.per_minute:
        billed = requests * max(usage.sent / usage.requests, price.minimum) / 60 if price.minimum else minutes
        return billed * price.per_minute
    audio_rate, out_rate = (TRANSCRIBE_AUDIO, TRANSCRIBE_OUT) if price.transcribe else (FLASH_AUDIO, FLASH_OUT)
    prompt = 0 if price.transcribe else FLASH_PROMPT
    tokens_out = minutes * out_rate + requests * THINKING.get(price.thinking, 0)
    return (minutes * 60 * audio_rate * price.audio + requests * prompt * price.text + tokens_out * price.output) / 1e6


def _cheapest(role: str) -> float:
    provider, model = CHEAPEST_GOOD[role]
    if role == "speech":
        return speech_monthly(SPEECH[(provider, model)])
    return text_monthly(TEXT[(provider, model)])


def speech_cost(key: str, model: str = "") -> Estimate:
    """A speech model by its sst.engines.SPEECH_MODELS key ("parakeet", "groq"...) and, in the cloud, its model id."""
    if key in LOCAL_SPEECH:
        return Estimate(0.0, FREE)
    price = _find(SPEECH, key, model)
    return tier(speech_monthly(price) if price else None, _cheapest("speech"))


def cleanup_cost(provider: str, model: str) -> Estimate:
    """An AI model for cleanup, Text Transform and Translate, by its sst.gateway.PROVIDERS key and model id."""
    if provider in LOCAL_AI:
        return Estimate(0.0, FREE)
    price = _find(TEXT, provider, model) if model else None
    return tier(text_monthly(price) if price else None, _cheapest("cleanup"))


def total(*parts: Estimate) -> Estimate:
    """Speech and AI together (a setup): unknown if a part is; tiered by the amounts alone (no 10 times rule)."""
    if any(part.monthly is None for part in parts):
        return Estimate(None, UNKNOWN)
    return tier(sum(part.monthly or 0.0 for part in parts))


def cheapest_known(provider: str, models: list[str]) -> str:
    """Of a provider's models, the one with the lowest estimate in the table ("" when none has one)."""
    known = [(text_monthly(price), m) for m in models if (price := _find(TEXT, provider, m))]
    return min(known)[1] if known else ""
