"""Translate's engine (sst.translate): the language to write in, the prompt, the value check. The model is faked."""
import pytest

from sst.translate import (
    LANGUAGES,
    Translator,
    already_in,
    check,
    choose_target,
    clean,
    detect,
    fallback_second,
    system_language,
    system_prompt,
)


@pytest.mark.parametrize("text, language", [
    ("金曜日までに更新したレポートを送っていただけますか？", "Japanese"),
    ("请在星期五之前把报告发给我。", "Chinese"),
    ("வெள்ளிக்கிழமைக்குள் அறிக்கையை அனுப்ப முடியுமா?", "Tamil"),
    ("Could you send me the updated report by Friday?", "English"),
    ("¿Podrías enviarme el informe actualizado antes del viernes?", ""),  # the Latin alphabet alone can't tell
    ("क्या आप शुक्रवार तक रिपोर्ट भेज सकते हैं?", ""),  # Devanagari: Hindi or Marathi, so no guess
])
def test_the_language_is_named_only_when_the_letters_make_it_certain(text, language):
    assert detect(text) == language


def test_text_already_in_the_language_goes_into_windows_language_or_english():
    assert fallback_second("English", "Japanese") == "Japanese"  # a Japanese Windows
    assert fallback_second("Japanese", "Japanese") == "English"  # never into the same language
    assert fallback_second("English", "English") == ""  # nothing better to offer: the prompt decides
    assert fallback_second("Tamil", "Klingon") == "English"  # a Windows language Translate doesn't offer
    assert system_language() in ("", *LANGUAGES)  # this computer's, whatever it is: one Translate offers, or none


@pytest.mark.parametrize("text, language, expected", [
    ("今日は会議があります。", "Japanese", True),  # kana and kanji
    ("今天有会议", "Japanese", False),  # kanji alone: Chinese
    ("今天有会议", "Chinese (Simplified)", True),
    ("வணக்கம் எல்லோருக்கும்", "Tamil", True),
    ("नमस्ते सबको", "Hindi", True),
    ("Please send the report by Friday.", "English", True),
    ("Bitte schick mir den Bericht.", "English", False),
    ("Hello world", "Japanese", False),
    ("12345", "English", False),
    ("ok", "English", False),  # too little to tell
])
def test_already_in(text, language, expected):
    assert already_in(text, language) is expected


def test_text_already_in_the_language_goes_to_the_second_one():
    assert choose_target("Please send it to me.", "English", "Japanese") == "Japanese"
    assert choose_target("送ってください", "English", "Japanese") == "English"
    assert choose_target("Please send it to me.", "English", "") == "English"  # no second language: as asked


def test_the_prompt_makes_a_translator_not_an_assistant():
    prompt = system_prompt("Japanese")
    assert "into Japanese" in prompt and "never answer" in prompt and "Output only the translation" in prompt
    assert "numbers" in prompt and "links" in prompt


@pytest.mark.parametrize("source, translation, missing", [
    ("Budget $25,000 by 10/15, mail a@b.co, see https://x.io/p", "Presupuesto 25.000 USD antes del 10/15, correo a@b.co, "
                                                                  "ver https://x.io/p", []),
    ("The budget is $25,000.", "Presupuesto de 20.000 dólares.", ["25,000"]),
    ("Retry after 30 seconds.", "Reintentar después de treinta segundos.", ["30"]),
    ("予算は2万5000ドル", "The budget is 25,000 dollars", []),  # Japanese counts in ten-thousands
    ("The budget is 25,000 dollars", "予算は2万5000ドルです", []),
    ("Write to a@b.co", "Escribe a b@a.co", ["a@b.co"]),
    ("No values here.", "Aquí no hay valores.", []),
])
def test_check_names_the_values_the_translation_lost(source, translation, missing):
    assert check(source, translation) == [f"'{value}' isn't in the translation: check it" for value in missing]


@pytest.mark.parametrize("answer, text", [
    ('"Hola"', "Hola"), ("```\nHola\n```", "Hola"), ("「こんにちは」", "「こんにちは」"), ("  Hola  ", "Hola"),
])
def test_clean(answer, text):
    assert clean(answer) == text


def test_translator_asks_the_model_with_the_prompt_and_returns_the_translation():
    asked = []

    def complete(prompt, text):
        asked.append((prompt, text))
        return '"Pouvez-vous envoyer le rapport ?"'
    result = Translator(complete).translate("  Can you send the report?  ", "French", "English")
    assert result.text == "Pouvez-vous envoyer le rapport ?" and result.target == "French" and result.warnings == []
    assert asked == [(system_prompt("French"), "Can you send the report?")]


def test_a_chosen_source_language_is_named_and_the_second_language_is_left_alone():
    asked = []

    def complete(prompt, text):
        asked.append(prompt)
        return "Please send the report."
    result = Translator(complete).translate("Please send the report.", "English", "Japanese", "Spanish")
    assert result.target == "English" and "from Spanish into English" in asked[0]
    assert "from" not in system_prompt("English").split(".")[0]


def test_text_in_the_target_language_is_translated_into_the_second_one():
    result = Translator(lambda prompt, text: "レポートを送ってください。").translate("Please send the report.", "English",
                                                                                "Japanese")
    assert result.target == "Japanese"


def test_an_empty_answer_is_an_error():
    with pytest.raises(ValueError):
        Translator(lambda prompt, text: "  ").translate("Hello there, everyone.", "French")
