"""What the models cost a month for the typical user (sst.costs), against the research notes of 2026-10-06."""
import pytest

from sst import costs


def _about(estimate: costs.Estimate, amount: float, cents: float = 0.03) -> bool:
    return estimate.monthly is not None and abs(estimate.monthly - amount) <= cents


def test_the_typical_user_sends_what_the_notes_assume():
    usage = costs.TYPICAL
    assert usage.dictations == 3300 and usage.minutes_sent == pytest.approx(495)
    assert usage.speech_requests == pytest.approx(3795)


@pytest.mark.parametrize("provider, model, amount", [
    ("groq", "whisper-large-v3-turbo", 0.42),  # the 10 s minimum per request included
    ("groq", "whisper-large-v3", 1.17),
    ("openai", "gpt-4o-mini-transcribe", 1.49),
    ("openai", "gpt-transcribe", 2.23),
    ("openai", "whisper-1", 2.97),
    ("gemini", "gemini-3.5-transcribe", 2.52),
    ("gemini", "gemini-3.5-flash-lite", 0.85),  # its minimal thinking included
    ("gemini", "gemini-3.6-flash", 5.67),  # medium thinking, as Gemini thinks by default
    ("gemini", "gemini-flash-latest", 5.67),  # answered by 3.6 Flash, at its price, since 2026-10
])
def test_speech_models_cost_what_the_notes_estimate(provider, model, amount):
    assert _about(costs.speech_cost(provider, model), amount, 0.05)


@pytest.mark.parametrize("provider, model, amount", [
    ("gemini", "gemini-3.5-flash-lite", 1.05),
    ("groq", "openai/gpt-oss-20b", 0.28),  # at low reasoning
    ("openai", "gpt-4o-mini", 0.30),
    ("openai", "gpt-4.1-mini", 0.80),
    ("anthropic", "claude-haiku-4-5", 2.19),
    ("openai", "gpt-5-mini", 3.20),
    ("gemini", "gemini-3.6-flash", 6.34),
    ("gemini", "gemini-flash-latest", 6.34),
    ("gemini", "gemini-3.5-flash", 6.34),
    ("anthropic", "claude-haiku-5-5", 0.28),  # asked not to think; ~30% more tokens
    ("anthropic", "claude-sonnet-5-5", 21.99),  # Claude 4.7+ counts ~30% more tokens
    ("anthropic", "claude-fable-5-1", 109.97),
    ("gemini", "gemini-3.1-pro-preview", 44.88),  # high thinking
])
def test_ai_models_cost_what_the_notes_estimate(provider, model, amount):
    assert _about(costs.cleanup_cost(provider, model), amount, 0.05)


def test_models_on_this_pc_are_free_and_unknown_ones_have_no_price():
    assert costs.speech_cost("parakeet").tier == costs.FREE and costs.speech_cost("whisper-turbo").words == "Free"
    assert costs.cleanup_cost("ollama", "llama3.2").tier == costs.FREE
    for unknown in (costs.cleanup_cost("vllm", "Qwen/Qwen3-30B"), costs.cleanup_cost("openai", "gpt-9-imaginary"),
                    costs.speech_cost("server", "whisper-1"), costs.speech_cost("gemini", "gemini-flash-lite-latest"),
                    costs.cleanup_cost("gemini", "")):
        assert unknown.tier == costs.UNKNOWN and unknown.monthly is None  # never red
        assert unknown.words == "Price unknown" and unknown.warning == ""


def test_the_tiers_low_amber_and_red():
    assert costs.cleanup_cost("gemini", "gemini-3.5-flash-lite").tier == costs.LOW
    assert costs.speech_cost("groq", "whisper-large-v3-turbo").tier == costs.LOW
    assert costs.cleanup_cost("anthropic", "claude-haiku-4-5").tier == costs.AMBER  # $2-5, under 10 times
    assert costs.speech_cost("gemini", "gemini-3.5-transcribe").tier == costs.AMBER
    assert costs.speech_cost("gemini", "gemini-3.6-flash").tier == costs.RED  # over $5
    assert costs.cleanup_cost("openai", "gpt-5-mini").tier == costs.AMBER  # $3.20: 3 times Gemini 3.5 Flash-Lite's
    assert costs.cleanup_cost("openai", "gpt-6-astra").tier == costs.RED
    assert costs.cleanup_cost("openai", "gpt-4.1-nano").tier == costs.UNKNOWN  # shut down on 2026-10-23


def test_the_cheapest_option_is_one_offered():
    assert costs.cheapest_good() == {"speech": ("gemini", "gemini-3.5-flash-lite"),
                                     "cleanup": ("gemini", "gemini-3.5-flash-lite")}  # Groq's while it was offered
    assert costs.tier(11.0, 1.0).times == 11.0 and costs.tier(3.0, 0.25).tier == costs.RED  # the 10 times rule stays


def test_the_warning_says_why():
    red = costs.cleanup_cost("gemini", "gemini-3.1-pro-preview")
    assert red.warning == "About $45 a month for typical use: more than 10 times the cheapest option."
    assert costs.speech_cost("gemini", "gemini-3.6-flash").warning == \
        "About $5.67 a month for typical use: an expensive model, over $5.00 a month."
    assert costs.tier(6.0, 1.0).warning == "About $6.00 a month for typical use: an expensive model, over $5.00 a month."
    assert costs.cleanup_cost("anthropic", "claude-haiku-4-5").warning == \
        "About $2.19 a month for typical use: a mid-priced model."
    assert costs.cleanup_cost("gemini", "gemini-3.5-flash-lite").warning == ""  # low: no warning


def test_words_and_money():
    assert costs.cleanup_cost("gemini", "gemini-3.5-flash-lite").words == "About $1.05 a month"
    assert costs.speech_cost("groq", "whisper-large-v3-turbo").short == "≈ $0.42/mo"
    assert costs.money(0.4217) == "$0.42" and costs.money(13.19) == "$13" and costs.money(3.574) == "$3.57"


def test_ids_as_providers_list_them():
    assert costs.cleanup_cost("gemini", "models/gemini-3.5-flash-lite").monthly == \
        costs.cleanup_cost("gemini", "gemini-3.5-flash-lite").monthly  # Gemini's own ids start with models/
    assert costs.cleanup_cost("openai", "gpt-4.1-mini-2025-04-14").monthly == \
        costs.cleanup_cost("openai", "gpt-4.1-mini").monthly  # a dated snapshot
    assert costs.cleanup_cost("anthropic", "claude-haiku-4-5-20251001").tier == costs.AMBER
    assert costs.cleanup_cost("openai", "gpt-4o-mini").monthly < costs.cleanup_cost("openai", "gpt-4o").monthly  # not a prefix


def test_a_total_adds_its_parts_up():
    total = costs.total(costs.speech_cost("groq", "whisper-large-v3-turbo"),
                        costs.cleanup_cost("groq", "openai/gpt-oss-20b"))
    assert _about(total, 0.70) and total.tier == costs.LOW
    assert costs.total(costs.speech_cost("parakeet"), costs.Estimate(0.0, costs.FREE)).tier == costs.FREE
    assert costs.total(costs.speech_cost("server"), costs.cleanup_cost("groq", "openai/gpt-oss-20b")).tier == costs.UNKNOWN


def test_the_cheapest_known_model_of_a_list():
    assert costs.cheapest_known("openai", ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "o3"]) == "gpt-4o-mini"
    assert costs.cheapest_known("groq", ["canopylabs/orpheus-v1-english", "allam-2-7b"]) == ""
