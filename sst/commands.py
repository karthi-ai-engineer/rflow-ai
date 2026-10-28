"""Voice commands for Text Transform: hold the dictation key and say "make it concise" instead of a sentence.

A dictation is a command only when the whole of it is one command phrase (with at most a filler or "please" around it):
"make it concise" transforms the last dictation or the selected text, "make it concise and send it to John" is typed as
dictated. Each transform has default phrases; the user can replace them with their own on the Text Transform page
(Settings.command_phrases: transform -> phrases separated by commas or new lines). One word is a command too when it is
the whole dictation ("Professional.", "Concise."): the owner said them that way and saw them typed (2026-10-28), while
"bullet points" alone already worked. A sentence with the word in it is still dictation.

Japanese phrases work the same way ("英語にして", "校正してください"): their polite endings and fillers are said around the
command (ください, お願いします, えっと), like "please". Translate also takes a language and a register in its phrase:
"translate it into English formally", "make it Japanese", "日本語に訳して", "英語で丁寧に" (match_command answers
"translate:English:formal", which sst.transform reads).
"""
import difflib
import re

from sst.transform import translate_key
from sst.translate import LANGUAGES

UNDO = "undo"

DEFAULT_PHRASES: dict[str, tuple[str, ...]] = {
    "concise": ("concise", "shorter", "make it concise", "make it more concise", "make it shorter", "shorten it", "make it short",
                "make it brief", "concise version"),
    "professional": ("professional", "formal", "make it professional", "make it more professional", "make it formal",
                     "make it more formal", "professional version", "formal version"),
    "bullets": ("make it bullet points", "bullet points", "make it a list", "turn it into bullet points",
                "turn it into a list", "bullet list", "make it bullets"),
    "actions": ("action items", "make it action items", "turn it into action items", "make it a to do list",
                "to do list", "make it a task list", "task list"),
    "rewrite": ("rewrite", "rewrite it", "make it clearer", "rewrite it more clearly"),
    "grammar": ("proofread", "fix the grammar", "fix grammar", "fix the typos", "fix typos", "proofread it",
                "correct the grammar", "校正して", "誤字を直して", "文法を直して"),
    "teams": ("make it a teams message", "make it a japanese teams message", "teams message", "japanese teams message",
              "translate to japanese for teams", "translate it to japanese for teams", "make it a teams chat",
              "Teams用にして", "チャット用にして", "社内チャットにして"),
    "email_internal": ("make it an internal email", "make it a japanese email", "make it a japanese internal email",
                       "internal email", "japanese email", "社内メールにして", "メールにして"),
    "email_external": ("make it an external email", "make it a japanese external email", "make it a client email",
                       "make it a japanese client email", "external email", "社外メールにして", "取引先向けにして"),
    "translate": ("translate it", "翻訳して"),
    UNDO: ("undo", "undo that", "undo it", "undo the transform", "restore the original", "put it back", "元に戻して"),
}
MAX_WORDS = 8  # a command is short: anything longer is dictation
FUZZY = 0.9  # a slightly misheard phrase still counts ("make it consise"); a different sentence doesn't

# Said around a command without changing it: "okay, make it concise please", "えっと、英語にしてください".
_BEFORE = {"um", "uh", "er", "okay", "ok", "so", "please", "hey", "now", "rflow", "and", "alright", "right",
           "えっと", "えーと", "えー", "あの", "あのー", "じゃあ", "じゃ", "では", "それでは", "ちょっと", "あ", "ねえ"}
_AFTER = {"please", "thanks", "now", "for", "me", "thank", "you"}
_PRONOUNS = {"this", "that", "it", "them"}  # "make this shorter" is "make it shorter"
_JAPANESE = re.compile(r"[぀-ヿ㐀-鿿]")
# "英語にしてください", "英語にしてくれる?", "校正お願いします", "英語に直して": all the same command as "英語にして".
_JA_ENDINGS = [
    (re.compile(r"[ねよな]+$"), ""),
    (re.compile(r"(?:して|し)(?:ください|下さい|くださいませ|くれる|くれ|くれますか|くれません?か|もらえる|もらえますか|"
                r"もらえません?か|ほしい|欲しい|ちょうだい|いただけますか|いただけません?か)$"), "して"),
    (re.compile(r"(?:を)?(?:お願い|おねがい)(?:します|いたします|できますか|できる)?$"), "して"),
    (re.compile(r"に(?:直|変え|変換)して$"), "にして"),
]

# Translate, with a language and a register: "translate it into English formally", "make it Japanese", "in English".
_REGISTERS = {"formally": "formal", "formal": "formal", "politely": "formal", "polite": "formal", "casually": "casual",
              "casual": "casual", "informally": "casual", "丁寧に": "formal", "丁寧な": "formal", "フォーマルに": "formal",
              "フォーマルな": "formal", "カジュアルに": "casual", "カジュアルな": "casual"}
_LANGUAGE_WORDS = {name.split(" (")[0].casefold(): name for name in LANGUAGES} | {
    "chinese": "Chinese (Simplified)", "simplified chinese": "Chinese (Simplified)", "mandarin": "Chinese (Simplified)",
    "traditional chinese": "Chinese (Traditional)"}
_JA_LANGUAGES = {"英語": "English", "日本語": "Japanese", "中国語": "Chinese (Simplified)", "韓国語": "Korean",
                 "スペイン語": "Spanish", "フランス語": "French", "ドイツ語": "German", "イタリア語": "Italian",
                 "ポルトガル語": "Portuguese", "ロシア語": "Russian", "タミル語": "Tamil", "ヒンディー語": "Hindi",
                 "ベトナム語": "Vietnamese", "タイ語": "Thai", "インドネシア語": "Indonesian"}
_LANGUAGE = "|".join(sorted(map(re.escape, _LANGUAGE_WORDS), key=len, reverse=True))
_REGISTER = "|".join(w for w in _REGISTERS if w.isascii())
_EN_TRANSLATE = re.compile(rf"(?:translate(?: it)?(?: (?:to|into|in))?|make it|in|say it in|put it in) (?P<lang>{_LANGUAGE})"
                           rf"(?: (?P<reg>{_REGISTER}))?")
_JA_LANGUAGE = "|".join(map(re.escape, _JA_LANGUAGES))
_JA_REGISTER = "|".join(w for w in _REGISTERS if not w.isascii())
_JA_TRANSLATE = [  # 英語にして, 日本語に訳して, 丁寧な英語にして / 英語で丁寧に, 英語で訳して (英語でお願いします) / 英訳して
    re.compile(rf"(?P<reg>{_JA_REGISTER})?(?P<lang>{_JA_LANGUAGE})(?:に|へ)(?P<reg2>{_JA_REGISTER})?(?:翻訳|訳)?して"),
    re.compile(rf"(?P<lang>{_JA_LANGUAGE})で(?:(?P<reg2>{_JA_REGISTER})(?:(?:翻訳|訳)?して)?|(?:翻訳|訳)?して)"),
    re.compile(r"(?P<short>英|和)訳して"),
]
_JA_FILLERS = re.compile(r"^(?:えっと|えーと|えー|あのー?|じゃあ|では|それでは|ちょっと)+(?=.)")


def _japanese_ending(word: str) -> str:
    for pattern, repl in _JA_ENDINGS:
        word = pattern.sub(repl, word)
    return word


def normalize(text: str) -> str:
    """Lowercase words without punctuation, "this"/"that" as "it", "to-do" as "to do", fillers around it removed; in
    Japanese, no spaces, and the polite endings as plain "して" ("英語にしてください" as "英語にして")."""
    text = text.casefold().replace("’", "'").replace("-", " ")
    text = re.sub(rf"(?<={_JAPANESE.pattern})\s+|\s+(?={_JAPANESE.pattern})", "", text)  # "Teams 用に して": "teams用にして"
    words = ["it" if w in _PRONOUNS else w for w in re.findall(r"[\w']+", text)]
    while words and words[0] in _BEFORE:
        words.pop(0)
    while words and words[-1] in _AFTER:
        words.pop()
    if words and _JAPANESE.search(words[0]):
        words[0] = _JA_FILLERS.sub("", words[0])
    if words and _JAPANESE.search(words[-1]):
        words[-1] = _japanese_ending(words[-1])
    return " ".join("to do" if w == "todo" else w for w in words)


def parse_phrases(text: str) -> list[str]:
    """The user's phrases as typed in the page's box: separated by commas or new lines, each once."""
    return list(dict.fromkeys(p for part in re.split(r"[,\n;、]", text) if (p := " ".join(part.split()))))


def phrases_for(custom: dict[str, str]) -> dict[str, list[str]]:
    """Every command's phrases: the user's where they set some, else the defaults."""
    return {key: parse_phrases(custom[key]) if key in custom else list(default) for key, default in DEFAULT_PHRASES.items()}


def translate_command(said: str) -> str | None:
    """Translate with a language (and a register) named, from a normalized phrase: "translate it into english formally"
    -> "translate:English:formal", "英訳して" -> "translate:English"; None when it isn't one."""
    if m := _EN_TRANSLATE.fullmatch(said):
        return translate_key(_LANGUAGE_WORDS[m["lang"]], _REGISTERS.get(m["reg"] or "", ""))
    for pattern in _JA_TRANSLATE[:2]:
        if m := pattern.fullmatch(said):
            return translate_key(_JA_LANGUAGES[m["lang"]], _REGISTERS.get(m.groupdict().get("reg") or m["reg2"] or "", ""))
    if m := _JA_TRANSLATE[2].fullmatch(said):
        return translate_key("English" if m["short"] == "英" else "Japanese")
    return None


def match_command(text: str, phrases: dict[str, list[str]]) -> str | None:
    """The command (a transform key, or UNDO) the whole of `text` says, or None for dictation. Translate with a language
    named is "translate:<language>[:<register>]", while Translate has any phrase at all."""
    said = normalize(text)
    if not said or len(said.split()) > MAX_WORDS:
        return None
    best, score = None, 0.0
    for key, options in phrases.items():
        for phrase in options:
            wanted = normalize(phrase)
            if not wanted:
                continue
            if said == wanted:
                return key
            # Close enough only for longer phrases: "make it consise", not "make it" for "make it concise".
            if len(wanted) >= 10 and abs(len(said) - len(wanted)) <= 3:
                ratio = difflib.SequenceMatcher(None, said, wanted).ratio()
                if ratio >= FUZZY and ratio > score:
                    best, score = key, ratio
    if phrases.get("translate") and (command := translate_command(said)):
        return command  # before a near miss: "translate to japanese formally" isn't "...for teams" misheard
    return best
