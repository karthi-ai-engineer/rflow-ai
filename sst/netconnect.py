"""The network side cloud speech (sst.engines.cloud) and AI cleanup (sst.gateway) share: connecting within a time budget,
and remembering a provider that refused the user's key.

Connecting: Python's own way (socket.create_connection) tries a host's addresses one after another, each with the whole
timeout, and waits on DNS for as long as Windows does. Offline, that made a dictation wait 20-50 s. Here the name is
looked up with a deadline, the addresses are tried a little apart (happy eyeballs, RFC 8305: the next starts when the
one before hasn't answered in STAGGER) and the first to connect wins, all within CONNECT_BUDGET. Google's first address
never answering from some networks (sst.live.gemini) then costs a quarter of a second, not a timeout.

Refusals: an answer that says asking again is pointless (no credit, a quota used up, a key not accepted) is remembered
for the provider and key (or the model, when the limit is the model's), so the next chunks, dictations and cleanups go
straight to their fallback instead of asking again. A success clears it; a new key or model is another entry.
"""
import errno
import hashlib
import http.client
import json
import re
import selectors
import socket
import ssl
import threading
import time
from dataclasses import dataclass

CONNECT_BUDGET = 3.0  # seconds for the name, the connection and the TLS handshake together
STAGGER = 0.25  # seconds an address gets before the next one is tried as well
AGAIN_AFTER = 1.0  # once every address is being tried, one is tried afresh this often (a lost first packet)
MAX_SOCKETS = 12  # connection attempts in flight at once
TLS_LEAST = 1.0  # seconds the TLS handshake always gets, however much of the budget the connection used
REFUSED_FOR = 300.0  # seconds a refusal is remembered (no credit, quota used up, key not accepted)
RATE_LIMIT_WAIT = 5.0  # a rate limit to wait out at most this long is worth a retry; a longer one is a refusal for now

_IN_PROGRESS = {errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, 10035}  # 10035: Windows' WSAEWOULDBLOCK

_context: ssl.SSLContext | None = None
_context_lock = threading.Lock()


def ssl_context() -> ssl.SSLContext:
    """One TLS context for every connection: making one loads the Windows certificate store (~100 ms)."""
    global _context
    with _context_lock:
        if _context is None:
            _context = ssl.create_default_context()
        return _context


def resolve(host: str, port: int, timeout: float) -> list[tuple]:
    """The host's addresses (getaddrinfo), or OSError when the name can't be looked up within `timeout` seconds. The
    lookup runs on a thread of its own: Windows waits on an unanswered DNS server far longer than a dictation can."""
    found: list = []

    def lookup():
        try:
            found.append(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError as e:
            found.append(e)
    thread = threading.Thread(target=lookup, name="dns", daemon=True)
    thread.start()
    thread.join(max(0.0, timeout))
    if not found:
        raise OSError(f"no answer from the name server for {host} in {timeout:.1f} s")
    if isinstance(found[0], OSError):
        raise found[0]
    if not found[0]:
        raise OSError(f"no address for {host}")
    return found[0]


def _interleaved(infos: list[tuple]) -> list[tuple]:
    """IPv6 and IPv4 addresses taking turns (RFC 8305), so one broken family can't hold up the other."""
    families: dict[int, list[tuple]] = {}
    for info in infos:
        families.setdefault(info[0], []).append(info)
    lists, out = list(families.values()), []
    for n in range(max(map(len, lists), default=0)):
        out += [lst[n] for lst in lists if n < len(lst)]
    return out


def open_socket(host: str, port: int, budget: float = CONNECT_BUDGET, stagger: float = STAGGER) -> socket.socket:
    """A connected socket to the first of the host's addresses that answers, within `budget` seconds (the name lookup
    included). Raises OSError (TimeoutError when nothing answered in time). The socket is left in blocking mode with
    the rest of the budget as its timeout, so a TLS handshake after it stays within it too."""
    deadline = time.monotonic() + budget
    infos = _interleaved(resolve(host, port, budget))
    pending: dict[socket.socket, tuple] = {}
    errors: list[OSError] = []
    selector = selectors.DefaultSelector()
    turn, next_start, winner = 0, time.monotonic(), None
    try:
        while winner is None:
            now = time.monotonic()
            if now >= deadline:
                break
            if turn and turn % len(infos) == 0 and not pending and errors:
                break  # every address refused or failed at once: nothing left to wait for
            if (now >= next_start or not pending) and len(pending) < MAX_SOCKETS:
                info = infos[turn % len(infos)]
                turn += 1
                next_start = now + (stagger if turn < len(infos) else AGAIN_AFTER)
                sock = socket.socket(info[0], info[1], info[2])
                sock.setblocking(False)
                code = sock.connect_ex(info[4])
                if code == 0:
                    winner = sock
                elif code in _IN_PROGRESS:
                    pending[sock] = info
                    selector.register(sock, selectors.EVENT_WRITE)
                else:
                    sock.close()
                    errors.append(OSError(code, f"cannot connect to {info[4][0]}"))
                continue
            wait = deadline - now if len(pending) >= MAX_SOCKETS else min(deadline, next_start) - now
            for key, _ in selector.select(max(0.0, wait)):  # Windows reports a failed connect as writable too
                sock = key.fileobj
                selector.unregister(sock)
                info = pending.pop(sock)
                code = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if code == 0 and winner is None:
                    winner = sock
                else:
                    sock.close()
                    if code:
                        errors.append(OSError(code, f"cannot connect to {info[4][0]}"))
                    next_start = time.monotonic()  # one failed: try the next address now (RFC 8305)
    finally:
        for sock in pending:
            sock.close()
        selector.close()
    if winner is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no answer from {host} in {budget:.1f} s")
        raise errors[-1] if errors else OSError(f"cannot connect to {host}")
    winner.setblocking(True)
    winner.settimeout(max(TLS_LEAST, deadline - time.monotonic()))
    return winner


def http_connection(https: bool, host: str, port: int | None, budget: float = CONNECT_BUDGET) -> http.client.HTTPConnection:
    """A connected HTTP(S) connection, the TLS handshake done, within `budget` seconds (plus at least TLS_LEAST for the
    handshake). Raises OSError."""
    if https:
        conn = http.client.HTTPSConnection(host, port, timeout=budget, context=ssl_context())
    else:
        conn = http.client.HTTPConnection(host, port, timeout=budget)
    # http.client connects through socket.create_connection, one address after another: this way instead.
    conn._create_connection = lambda address, timeout, source=None: open_socket(address[0], address[1], budget)
    try:
        conn.connect()
    except BaseException:
        conn.close()
        raise
    return conn


# ---- refusals: answers that say asking again won't help

@dataclass(frozen=True)
class Refusal:
    reason: str  # what to tell the user
    model_only: bool = False  # the model's limit (a quota), not the key's or the account's
    seconds: float = REFUSED_FOR  # how long to remember it
    rate_limit: bool = False  # a per-minute limit: it lifts by itself (scoring reading tests waits it out)


NO_CREDIT = "no credit left on the account"
QUOTA = "its quota is used up"
NO_KEY = "it didn't accept the API key"
_CREDIT = re.compile(r"insufficient[_ ](quota|balance|funds)|credit balance|out of credits?|billing|payment required",
                     re.IGNORECASE)
_DAILY = re.compile(r"per ?day|daily|\bRPD\b|\bTPD\b", re.IGNORECASE)
_QUOTA = re.compile(r"RESOURCE_EXHAUSTED|exceeded your current quota|quota exceeded", re.IGNORECASE)
_RETRY_IN = re.compile(r"(?:retry|try again) in ([\d.]+) ?(ms|s)\b", re.IGNORECASE)


def refusal(status: int, body: str, headers=None) -> Refusal | None:
    """Whether an error answer means asking again (now, or for the next chunk) is pointless: no credit (402, or a 429
    or 400 saying so), a key not accepted (401, 403), a quota used up (Gemini's RESOURCE_EXHAUSTED for the day), or a
    rate limit longer than RATE_LIMIT_WAIT. None for anything worth a retry (a short rate limit, the server's trouble)."""
    if status == 402:
        return Refusal(NO_CREDIT)
    if status in (401, 403):
        return Refusal(NO_KEY)
    if status not in (400, 429):
        return None
    if _CREDIT.search(body or ""):
        return Refusal(NO_CREDIT)
    if status == 400:
        return None
    wait = retry_after(body, headers)
    if _DAILY.search(body or ""):
        return Refusal(QUOTA, model_only=True)
    if wait is not None and wait > RATE_LIMIT_WAIT:  # a per-minute limit: refused until it lifts
        return Refusal(f"it is rate limited for {wait:.0f} s", model_only=True, seconds=min(wait, REFUSED_FOR),
                       rate_limit=True)
    if wait is None and _QUOTA.search(body or ""):  # a quota with no word on when it comes back
        return Refusal(QUOTA, model_only=True, seconds=60.0)
    return None


def retry_after(body: str, headers=None) -> float | None:
    """Seconds the provider asks to wait: the Retry-After header, Gemini's RetryInfo ("41s") or its message ("Please
    retry in 41.2s"); None when it doesn't say."""
    value = headers.get("retry-after") if headers is not None else None
    try:
        if value is not None:
            return max(0.0, float(value))
    except ValueError:
        pass  # an HTTP date: rare, and not worth parsing
    try:
        answer = json.loads(body or "")
    except ValueError:
        answer = None
    if isinstance(answer, list) and answer:  # Gemini's OpenAI-compatible endpoint wraps its error in a list
        answer = answer[0]
    error = answer.get("error") if isinstance(answer, dict) else None
    for detail in (error.get("details") or []) if isinstance(error, dict) else []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            try:
                return float(str(detail.get("retryDelay", "")).removesuffix("s"))
            except ValueError:
                pass
    if found := _RETRY_IN.search(body or ""):
        return float(found.group(1)) / (1000 if found.group(2).lower() == "ms" else 1)
    return None


_refused: dict[tuple[str, str, str], tuple[float, str]] = {}  # (where, key's fingerprint, model or "") -> (until, why)
_refused_lock = threading.Lock()


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]  # the key itself is never kept


def refuse(where: str, key: str, model: str, found: Refusal) -> None:
    """Remember a refusal from `where` (the provider's host) for this key, and for this model only when it's the
    model's limit."""
    with _refused_lock:
        _refused[(where, _fingerprint(key), model if found.model_only else "")] = (time.monotonic() + found.seconds,
                                                                                   found.reason)


def refused(where: str, key: str, model: str) -> str:
    """Why this key (or this model on it) was refused a moment ago, or ""."""
    now, fingerprint = time.monotonic(), _fingerprint(key)
    with _refused_lock:
        for scope in ("", model):
            until, reason = _refused.get((where, fingerprint, scope), (0.0, ""))
            if until > now:
                return reason
    return ""


def forgive(where: str, key: str, model: str) -> None:
    """An answer came: whatever was refused for this key and model is no more (credit added, a new day's quota)."""
    fingerprint = _fingerprint(key)
    with _refused_lock:
        for scope in ("", model):
            _refused.pop((where, fingerprint, scope), None)


def forget_refusals() -> None:
    with _refused_lock:
        _refused.clear()
