"""What each provider's models accept (sst.modelrules), by model id: no request is sent."""
import pytest

from sst import modelrules


@pytest.mark.parametrize("model, reasoning, effort", [
    ("gpt-6-luna", True, "none"), ("gpt-6-sol", True, "none"), ("gpt-6.1-sol", True, "low"),
    ("gpt-6-astra", True, "low"), ("gpt-5.6-luna", True, "none"), ("gpt-5-nano", True, "minimal"),
    ("gpt-5-mini-2025-08-07", True, "minimal"), ("gpt-5", True, "minimal"), ("gpt-5.4-mini", True, ""),
    ("o4-mini", True, ""), ("gpt-4.1-mini", False, ""), ("gpt-4o-mini", False, ""),
])
def test_openai_reasoning_models_and_the_least_thinking_they_allow(model, reasoning, effort):
    assert bool(modelrules.OPENAI_REASONING.match(model)) is reasoning  # gpt-6 was missed before
    assert modelrules.openai_effort(model) == effort


@pytest.mark.parametrize("model, thinking, temperature", [
    ("gemini-3.5-flash-lite", "minimal", False), ("models/gemini-3.5-flash-lite", "minimal", False),
    ("gemini-3.1-flash-lite", "minimal", False), ("gemini-3.6-flash", "minimal", False),
    ("gemini-3.7-flash", "low", False), ("gemini-3.8-flash", "low", False),  # "minimal" is an error on these
    ("gemini-3.5-flash", "minimal", False), ("gemini-flash-latest", "minimal", False),  # both answered by 3.6 Flash
    ("gemini-3.7-flash", "low", False),  # answered by 3.8 Flash, which refuses "minimal"
    ("gemini-3-flash-preview", "low", False), ("gemini-3.1-pro-preview", "low", False),
    ("gemini-flash-lite-latest", "", False),  # its model isn't documented: its own thinking and temperature
    ("gemini-2.5-flash-lite", "", True), ("gemini-2.5-flash", "", True), ("gemma-3-27b-it", "", True),
])
def test_gemini_3_thinks_least_at_its_own_temperature(model, thinking, temperature):
    assert modelrules.gemini_thinking(model) == thinking
    assert modelrules.gemini_takes_temperature(model) is temperature


@pytest.mark.parametrize("model, effort", [("openai/gpt-oss-20b", "low"), ("openai/gpt-oss-120b", "low"),
                                           ("qwen/qwen3.8-27b", "none"), ("llama-3.1-8b-instant", "")])
def test_groq_models_that_think_are_asked_for_the_least(model, effort):
    assert modelrules.groq_effort(model) == effort


@pytest.mark.parametrize("model, version, temperature", [
    ("claude-haiku-4-5", (4, 5), True), ("claude-haiku-4-5-20251001", (4, 5), True),
    ("claude-sonnet-4-5-20250929", (4, 5), True), ("claude-sonnet-4-6", (4, 6), True), ("claude-opus-4-6", (4, 6), True),
    ("claude-sonnet-4-20250514", (4, 0), True), ("claude-opus-4-1-20250805", (4, 1), True),
    ("claude-3-5-haiku-20241022", (3, 5), True), ("claude-3-7-sonnet-latest", (3, 7), True),
    ("claude-opus-4-7", (4, 7), False), ("claude-opus-4-8", (4, 8), False), ("claude-opus-5", (5, 0), False),
    ("claude-sonnet-5", (5, 0), False), ("claude-sonnet-5-5", (5, 5), False), ("claude-opus-5-5", (5, 5), False),
    ("claude-fable-5", (5, 0), False), ("claude-fable-5-1", (5, 1), False),
    ("my-claude", None, False),  # unknown: the default temperature, which every model takes
])
def test_claude_4_7_and_later_take_no_temperature(model, version, temperature):
    assert modelrules.claude_version(model) == version
    assert modelrules.claude_takes_temperature(model) is temperature
