"""Japanese on the screen into English: Gemini, a cache, and a glossary for single UI words.

A screen's new strings go to Gemini in one generateContent request (a JSON array back, as long as what was sent), so
short labels get the rest of the screen as context; the answers are cached by the text, so a screen read again, or a
page scrolled back, costs nothing. Only the text the OCR read is sent, never the picture. Gemini 3.5 Flash-Lite, asked to
think as little as it can (sst.modelrules), with the profile's Gemini key (GatewayConfig.key_for). Every call is counted
in the cost tracker as "screen" (sst.usage); estimate() is what a new screen costs.

What the owner's own screen translator app learned (2026-10-09):
  * single UI words translate badly without context (保存 -> "keep"): UI_GLOSSARY answers those, and costs nothing
  * the translator gets era years wrong (令和6年 -> "2020"): they are written as Western years first (令和6年 -> 2024年)
  * a translation that is empty, unchanged or still mostly Japanese shows nothing: the original is better than a bad
    sticker
  * a failed request is retried with a backoff (1 s, doubling, at most 30 s); a refused key or used-up credit stops it
"""
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Protocol

from sst import costs, netconnect, usage
from sst.modelrules import gemini_takes_temperature, gemini_thinking

log = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash-lite"
FEATURE = "screen"  # the cost tracker's line (sst.usage.FEATURES)
BASE = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = 25.0
MAX_BATCH, MAX_CHARS = 50, 3500  # strings and characters a request: a big screen goes in a few
PROMPT = (
    "Translate each Japanese string below into natural, concise English as it would appear in a business application's "
    "interface (menus, buttons, form labels, table cells, messages). Keep numbers, product names and Latin text as they "
    "are. Keep proper names as romanised names. Use title case for short labels. If a string is not meaningful Japanese "
    "(an OCR fragment), return it unchanged. Return a JSON array of strings with exactly the same length and order as "
    "the input.\n\nInput:\n")
# What estimate() counts for a new screen: Gemini's own tokens for the prompt, about a token a Japanese character, and
# about four English tokens a string (with the JSON around them).
PROMPT_TOKENS, SCREEN_STRINGS, STRING_CHARS, OUT_PER_STRING = 110, 30, 8, 6


class Refused(Exception):
    """The key isn't accepted, or there is no credit or quota left: asking again won't help."""


class Backend(Protocol):
    name: str

    def translate_batch(self, texts: list[str]) -> list[str]: ...


def _post(url: str, body: dict, key: str, timeout: float) -> dict:
    """A JSON request to Gemini and its JSON answer; Refused when asking again is pointless, OSError for the rest."""
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=netconnect.ssl_context()) as response:
            return json.loads(response.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        found = netconnect.refusal(e.code, text, e.headers)
        if found is not None and not found.rate_limit:
            raise Refused(f"Gemini: {found.reason} (HTTP {e.code})") from None
        raise OSError(f"Gemini HTTP {e.code}") from None
    except ValueError:
        raise OSError("Gemini's answer couldn't be read") from None


class Gemini:
    """translate_batch through Gemini's generateContent, each call counted (sst.usage). post: _post (a fake in tests)."""

    name = "gemini"

    def __init__(self, key: str, model: str = MODEL, post: Callable = _post):
        self.key, self.model, self._post = key, model, post

    def _config(self) -> dict:
        config = {"responseMimeType": "application/json",
                  "responseJsonSchema": {"type": "array", "items": {"type": "string"}}}
        if gemini_takes_temperature(self.model):
            config["temperature"] = 0
        if least := gemini_thinking(self.model):
            config["thinkingConfig"] = {"thinkingLevel": least}
        return config

    def translate_batch(self, texts: list[str]) -> list[str]:
        body = {"contents": [{"role": "user", "parts": [{"text": PROMPT + json.dumps(texts, ensure_ascii=False)}]}],
                "generationConfig": self._config()}
        answer = self._post(f"{BASE}/models/{self.model}:generateContent", body, self.key, TIMEOUT)
        usage.record(FEATURE, self.model, **usage.gemini_counts(answer, 0.0, transcribe=False))
        try:
            parts = answer["candidates"][0]["content"]["parts"]
            out = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise ValueError("Gemini gave no translation") from None
        if not isinstance(out, list) or len(out) != len(texts) or not all(isinstance(o, str) for o in out):
            raise ValueError(f"expected {len(texts)} strings back")
        return out


def estimate(strings: int = SCREEN_STRINGS, chars: int = STRING_CHARS, model: str = MODEL) -> float:
    """What translating a new screen of `strings` strings costs in USD at today's price (0.0 for a model with no
    price); a screen read again costs nothing (the cache)."""
    price = costs.text_price_on("gemini", model, date.today().isoformat())
    if price is None or strings <= 0:
        return 0.0
    tokens_in = PROMPT_TOKENS + strings * (chars + 2)
    tokens_out = strings * OUT_PER_STRING + costs.THINKING.get(gemini_thinking(model), 0)
    return (tokens_in * price.input + tokens_out * price.output) / 1e6


# Single UI words translate badly without context (保存 -> "keep").
UI_GLOSSARY = {
    "ファイル": "File", "編集": "Edit", "表示": "View", "ツール": "Tools", "ヘルプ": "Help",
    "保存": "Save", "上書き保存": "Save", "名前を付けて保存": "Save As", "開く": "Open",
    "閉じる": "Close", "終了": "Exit", "印刷": "Print", "新規": "New", "新規作成": "New",
    "削除": "Delete", "追加": "Add", "変更": "Change", "更新": "Update", "検索": "Search",
    "置換": "Replace", "設定": "Settings", "オプション": "Options", "適用": "Apply",
    "キャンセル": "Cancel", "はい": "Yes", "いいえ": "No", "戻る": "Back", "次へ": "Next",
    "完了": "Done", "送信": "Send", "確認": "Confirm", "登録": "Register", "ログイン": "Log in",
    "ログアウト": "Log out", "元に戻す": "Undo", "やり直し": "Redo", "切り取り": "Cut",
    "コピー": "Copy", "貼り付け": "Paste", "すべて選択": "Select All", "挿入": "Insert",
    "書式": "Format", "ウィンドウ": "Window", "状態": "Status", "件名": "Subject",
    "担当者": "Assignee", "詳細": "Details", "一覧": "List", "ホーム": "Home",
    "通知": "Notifications", "未着手": "Not started", "対応中": "In progress",
    "承認": "Approve", "却下": "Reject", "下書き": "Draft", "添付": "Attach",
    "共有": "Share", "ダウンロード": "Download", "アップロード": "Upload",
    # forms and orders
    "氏名": "Full name", "名前": "Name", "住所": "Address", "電話番号": "Phone number",
    "郵便番号": "Postal code", "メールアドレス": "Email address", "備考": "Notes",
    "必須": "Required", "任意": "Optional", "注文番号": "Order number", "納期": "Delivery date",
    "数量": "Quantity", "単価": "Unit price", "金額": "Amount", "合計": "Total",
    "小計": "Subtotal", "税込": "Tax included", "税抜": "Tax excluded", "御中": "Attn.",
    "請求書": "Invoice", "見積書": "Quote", "領収書": "Receipt", "契約": "Contract",
    "日付": "Date", "期限": "Deadline", "種別": "Type", "区分": "Category", "番号": "No.",
    "性別": "Gender", "生年月日": "Date of birth", "会社名": "Company name", "部署": "Department",
    "役職": "Position", "備考欄": "Notes", "名": "Name", "男": "Male", "女": "Female", "計": "Total",
    # departments
    "営業部": "Sales Dept.", "開発部": "Development Dept.", "人事部": "HR Dept.",
    "総務部": "General Affairs Dept.", "経理部": "Accounting Dept.", "企画部": "Planning Dept.",
    "製造部": "Manufacturing Dept.", "技術部": "Engineering Dept.", "法務部": "Legal Dept.",
    "広報部": "PR Dept.", "管理部": "Administration Dept.", "情報システム部": "IT Dept.",
    # statuses and navigation
    "完了済み": "Completed", "処理中": "Processing", "保留": "On hold", "未処理": "Unprocessed",
    "承認待ち": "Pending approval", "差し戻し": "Returned", "新着": "New", "既読": "Read",
    "未読": "Unread", "すべて": "All", "一括": "Bulk", "選択": "Select", "解除": "Clear",
    "前へ": "Previous", "最新": "Latest", "並び替え": "Sort", "絞り込み": "Filter",
    "エクスポート": "Export", "インポート": "Import", "プロフィール": "Profile",
    "パスワード": "Password", "ユーザー名": "Username", "管理者": "Administrator",
    "メニュー": "Menu", "ダッシュボード": "Dashboard", "レポート": "Report", "履歴": "History",
}

_ERA_BASE = {"令和": 2018, "平成": 1988, "昭和": 1925, "大正": 1911}  # year N of the era = base + N
_ERA = re.compile(r"(令和|平成|昭和|大正)(元|\d{1,2})年")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")


def prep(text: str) -> str:
    """Era years as Western years (令和6年 -> 2024年, 令和元年 -> 2019年)."""
    return _ERA.sub(lambda m: f"{_ERA_BASE[m.group(1)] + (1 if m.group(2) == '元' else int(m.group(2)))}年", text)


def sane(src: str, dst: str) -> str:
    """"" (show nothing) for a translation that is empty, unchanged or still mostly Japanese."""
    if not dst or dst == src or len(_CJK.findall(dst)) > 0.3 * max(1, len(dst)):
        return ""
    return dst


def polish(dst: str) -> str:
    dst = dst.strip().lstrip(",.;:~)]}、。・ ")
    return dst[:1].upper() + dst[1:] if dst[:1].islower() else dst


class TranslationService:
    """Dedupes, batches and caches translation requests off the UI thread; on_update() after each answer or failure."""

    def __init__(self, backend: Backend | None, on_update: Callable[[], None], cache_size: int = 5000, workers: int = 2):
        self.backend, self.on_update, self.cache_size = backend, on_update, cache_size
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.pending: set[str] = set()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="screen-translate")
        self.status, self.last_error = "ready", ""
        self._backoff = self._retry_at = 0.0

    def get(self, text: str) -> str | None:
        """The translation ("" for none worth showing), or None while it's on its way."""
        with self.lock:
            value = self.cache.get(text)
            if value is not None:
                self.cache.move_to_end(text)
            return value

    def request(self, texts: list[str]) -> None:
        if self.backend is None or self.status == "refused" or time.monotonic() < self._retry_at:
            return
        with self.lock:
            todo = []
            for t in dict.fromkeys(texts):
                if t in UI_GLOSSARY:
                    self.cache[t] = UI_GLOSSARY[t]
                elif t not in self.cache and t not in self.pending:
                    todo.append(t)
                    self.pending.add(t)
        for chunk in chunks(todo):
            self.pool.submit(self._run, chunk)

    def _run(self, chunk: list[str]) -> None:
        try:
            out = self.backend.translate_batch([prep(t) for t in chunk])
        except Exception as e:  # the network, a refusal, an answer that couldn't be read
            self.last_error = str(e) or type(e).__name__
            with self.lock:
                self.pending.difference_update(chunk)
            if isinstance(e, Refused):
                self.status = "refused"
            else:
                self.status = "offline"
                self._backoff = min(30.0, max(1.0, self._backoff * 2))
                self._retry_at = time.monotonic() + self._backoff
            log.warning("Screen translator: %s", self.last_error)
            self.on_update()
            return
        self._backoff, self.status, self.last_error = 0.0, "online", ""
        with self.lock:
            for src, dst in zip(chunk, out, strict=True):
                self.cache[src] = sane(src, polish(dst))
                self.pending.discard(src)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.on_update()

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def chunks(texts: list[str], most: int = MAX_BATCH, chars: int = MAX_CHARS):
    chunk, size = [], 0
    for t in texts:
        if chunk and (len(chunk) >= most or size + len(t) > chars):
            yield chunk
            chunk, size = [], 0
        chunk.append(t)
        size += len(t)
    if chunk:
        yield chunk
