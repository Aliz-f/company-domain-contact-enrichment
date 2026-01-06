from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import httpx

from config import PipelineConfig
from schemas import Record

logger = logging.getLogger("Phase 5")

# -----------------------------
# Defaults (used only if assets/phase5/* are missing)
# -----------------------------
DEFAULT_TARGET_PATHS: List[str] = [
    "/",
    "/contact",
    "/about",
    "/legal",
    "/impressum",
    "/imprint",
    "/privacy",
    "/terms",
    "/kontakt",
    "/ueber-uns",
    "/uber-uns",
]

DEFAULT_FREE_EMAIL_PROVIDERS: Set[str] = {
    "gmail.com",
    "googlemail.com",
    "yahoo.com",
    "yahoo.co.uk",
    "hotmail.com",
    "outlook.com",
    "live.com",
    "icloud.com",
    "me.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
    "gmx.de",
    "gmx.net",
    "web.de",
    "mail.ru",
}

DEFAULT_ROLE_LOCAL_PARTS: Set[str] = {
    "info",
    "contact",
    "hello",
    "sales",
    "support",
    "help",
    "office",
    "admin",
    "accounts",
    "billing",
    "jobs",
    "careers",
    "press",
    "media",
    "enquiries",
    "service",
    "customerservice",
}

DEFAULT_NO_REPLY_LOCALS: Set[str] = {
    "noreply",
    "no-reply",
    "donotreply",
    "do-not-reply",
    "mailer-daemon",
    "postmaster",
}

# -----------------------------
# Extraction regex (conservative)
# -----------------------------
EMAIL_REGEX = re.compile(
    r"\b[a-z0-9][a-z0-9._%+\-]{0,63}@[a-z0-9][a-z0-9.\-]{0,253}\.[a-z]{2,24}\b",
    re.IGNORECASE,
)
MAILTO_REGEX = re.compile(r"mailto:([^?\s\"'>]+)", re.IGNORECASE)

PHONE_REGEX = re.compile(
    r"(?:(?:\+|00)\s?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d[\d\s\-\.]{6,}\d"
)

# -----------------------------
# Block / bot / anti-scrape heuristics
# -----------------------------
DEFAULT_BLOCK_STATUSES: Set[int] = {401, 403, 407, 409, 418, 429, 451, 503}
DEFAULT_BLOCK_KEYWORDS: List[str] = [
    "access denied",
    "request blocked",
    "you have been blocked",
    "forbidden",
    "not authorized",
    "captcha",
    "cloudflare",
    "attention required",
    "verify you are human",
    "bot detection",
    "ddos protection",
    "incapsula",
    "akamai",
]


@dataclass(frozen=True)
class _QueuedURL:
    url: str
    depth: int


@dataclass(frozen=True)
class _FetchResult:
    url: str
    depth: int
    status: Optional[int]
    content_type: Optional[str]
    body: str
    bytes_read: int
    error_kind: Optional[str]  # "timeout", "connect", "read", "other", None


# -----------------------------
# Small utilities
# -----------------------------
def _unique_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _read_asset_lines(path: Path) -> List[str]:
    """
    Phase asset reader (line-based).

    Rules:
      - one entry per line
      - ignore empty lines
      - ignore comments starting with '#'
    """
    if not path.exists():
        return []
    try:
        lines: List[str] = []
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
        return lines
    except Exception:
        return []


def _normalize_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return ""
    if not p.startswith("/"):
        p = "/" + p
    return p


def _canonicalize_url(url: str) -> str:
    """
    Stable URL normalization for dedupe and crawl stability:
      - lowercase scheme/host
      - drop fragment
      - drop query (prevents tracking-param crawl explosions)
      - normalize path:
          * ensure '/' for empty path
          * strip trailing '/' except root
      - drop params
      - drop username/password
      - ignore port for canonical form (we keep default behavior via urlparse.hostname)
    """
    try:
        u = urlparse(url)
        if not u.scheme or not u.netloc:
            return (url or "").strip()

        scheme = (u.scheme or "").lower()
        host = (u.hostname or "").lower()
        if not host:
            return (url or "").strip()

        path = u.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        return urlunparse((scheme, host, path, "", "", ""))
    except Exception:
        return (url or "").strip()


def _host_variants(domain: str) -> Set[str]:
    """
    Treat 'domain' and 'www.domain' as the same "official site" bucket
    for link discovery. (Strict: no other subdomains.)
    """
    d = (domain or "").strip().lower()
    if not d:
        return set()
    vars_ = {d}
    if d.startswith("www."):
        vars_.add(d[len("www.") :])
    else:
        vars_.add("www." + d)
    return vars_


def _is_same_domain_or_www(url: str, domain: str) -> bool:
    """Allow link discovery only on domain and www.domain (ignore ports)."""
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower().strip()
        return host in _host_variants(domain)
    except Exception:
        return False


def _extract_links(html_text: str, base_url: str, domain: str) -> List[str]:
    """
    Extract same-site links (domain or www.domain) using a lightweight regex.

    Notes:
      - This is intentionally not a full HTML parser.
      - We prefer determinism and robustness under bad HTML.
    """
    if not html_text:
        return []

    # Unescape HTML entities to reduce missed links (e.g., &amp; in URLs).
    html_text = html.unescape(html_text)

    hrefs = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html_text, flags=re.IGNORECASE)
    out: List[str] = []

    for h in hrefs:
        h = (h or "").strip()
        if not h or h.startswith("#"):
            continue

        hl = h.lower()
        if hl.startswith("mailto:") or hl.startswith("javascript:") or hl.startswith("tel:"):
            continue

        abs_url = urljoin(base_url, h)
        try:
            u = urlparse(abs_url)
            if u.scheme not in ("http", "https"):
                continue
            if not _is_same_domain_or_www(abs_url, domain):
                continue
            out.append(_canonicalize_url(abs_url))
        except Exception:
            continue

    return _unique_preserve_order([x for x in out if x])


def _looks_blocked_by_body(text: str) -> bool:
    """
    Lightweight block-page detection from body text.
    This is intentionally conservative to avoid false positives.
    """
    if not text:
        return False
    t = text.lower()
    hits = 0
    for kw in DEFAULT_BLOCK_KEYWORDS:
        if kw in t:
            hits += 1
            if hits >= 2:
                return True
    return False


class _PerDomainRateLimiter:
    """
    Per-domain delay limiter: ensures >= min_delay_s between requests per domain.

    This reduces the risk of getting blocked and keeps traffic polite.
    """

    def __init__(self, min_delay_s: float) -> None:
        self._min_delay_s = max(0.0, float(min_delay_s))
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_ts: Dict[str, float] = {}

        # Protect first-time lock creation under concurrency (race-safe).
        self._init_lock = asyncio.Lock()

    async def wait(self, domain: str) -> None:
        if self._min_delay_s <= 0:
            return

        d = (domain or "").lower()

        async with self._init_lock:
            lock = self._locks.get(d)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[d] = lock

        async with lock:
            now = time.monotonic()
            last = self._last_ts.get(d, 0.0)
            delta = now - last
            if delta < self._min_delay_s:
                await asyncio.sleep(self._min_delay_s - delta)
            self._last_ts[d] = time.monotonic()


# -----------------------------
# Phase 5 assets loader
# -----------------------------
def _load_phase5_assets(cfg: PipelineConfig) -> Tuple[List[str], Set[str], Set[str], Set[str]]:
    """
    Load Phase 5 assets from assets/phase5/.

    Files:
      - target_paths.txt
      - free_email_providers.txt
      - role_local_parts.txt
      - no_reply_locals.txt

    Falls back to safe defaults if missing.
    """
    phase_dir = cfg.phase_assets_dir(5)

    target_paths = _read_asset_lines(phase_dir / "target_paths.txt")
    free_providers = _read_asset_lines(phase_dir / "free_email_providers.txt")
    role_locals = _read_asset_lines(phase_dir / "role_local_parts.txt")
    no_reply = _read_asset_lines(phase_dir / "no_reply_locals.txt")

    paths = [_normalize_path(p) for p in target_paths] if target_paths else list(DEFAULT_TARGET_PATHS)
    paths = [p for p in paths if p]
    paths = _unique_preserve_order(paths)

    free_set = (
        {x.strip().lower() for x in free_providers if x.strip()}
        if free_providers
        else set(DEFAULT_FREE_EMAIL_PROVIDERS)
    )
    role_set = (
        {x.strip().lower() for x in role_locals if x.strip()}
        if role_locals
        else set(DEFAULT_ROLE_LOCAL_PARTS)
    )
    no_reply_set = (
        {x.strip().lower() for x in no_reply if x.strip()}
        if no_reply
        else set(DEFAULT_NO_REPLY_LOCALS)
    )

    free_set.discard("")
    role_set.discard("")
    no_reply_set.discard("")
    return paths, free_set, role_set, no_reply_set


# -----------------------------
# Task 5.1 Target URL Selection
# -----------------------------
def _select_target_urls(domain: str, target_paths: List[str]) -> List[str]:
    """
    Build a deterministic list of high-yield target URLs under the domain.
    Tries https first, then http.
    """
    d = (domain or "").strip()
    if not d:
        return []

    bases = [f"https://{d}", f"http://{d}"]
    out: List[str] = []
    for base in bases:
        for p in target_paths:
            out.append(_canonicalize_url(base + p))

    return _unique_preserve_order([x for x in out if x])


# -----------------------------
# Task 5.2 Async Fetch (bounded)
# -----------------------------
async def _fetch_limited(
        client: httpx.AsyncClient,
        url: str,
        *,
        timeout_s: float,
        max_bytes: int,
) -> _FetchResult:
    """
    Fetch a single URL with:
      - per-request timeout
      - per-page byte cap via streaming
      - returns decoded text only for text/html/json/xml-ish content types

    Note:
      We do NOT parse arbitrary binaries. If content-type is not text-like,
      we treat it as empty (no extraction).
    """
    try:
        async with client.stream("GET", url, timeout=timeout_s) as resp:
            status = resp.status_code
            ctype = (resp.headers.get("content-type") or "").lower()

            # Keep final URL after redirects for better base_url link resolution/debug.
            final_url = str(resp.url)

            text_like = (
                    ("text" in ctype)
                    or ("html" in ctype)
                    or ("json" in ctype)
                    or ("xml" in ctype)
                    or ("xhtml" in ctype)
                    or (ctype.strip() == "")
            )
            if not text_like:
                return _FetchResult(
                    url=_canonicalize_url(final_url),
                    depth=0,
                    status=status,
                    content_type=ctype,
                    body="",
                    bytes_read=0,
                    error_kind=None,
                )

            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                remaining = max_bytes - len(buf)
                if remaining <= 0:
                    break
                buf.extend(chunk[:remaining])
                if len(buf) >= max_bytes:
                    break

            # Decode conservatively; avoid charset guessing to keep deterministic.
            text = buf.decode("utf-8", errors="ignore")
            return _FetchResult(
                url=_canonicalize_url(final_url),
                depth=0,
                status=status,
                content_type=ctype,
                body=text,
                bytes_read=len(buf),
                error_kind=None,
            )

    except httpx.TimeoutException:
        return _FetchResult(url=url, depth=0, status=None, content_type=None, body="", bytes_read=0, error_kind="timeout")
    except httpx.ConnectError:
        return _FetchResult(url=url, depth=0, status=None, content_type=None, body="", bytes_read=0, error_kind="connect")
    except httpx.ReadError:
        return _FetchResult(url=url, depth=0, status=None, content_type=None, body="", bytes_read=0, error_kind="read")
    except Exception:
        return _FetchResult(url=url, depth=0, status=None, content_type=None, body="", bytes_read=0, error_kind="other")


# -----------------------------
# Task 5.3 Extraction
# -----------------------------
def _extract_emails_from_text(text: str) -> Set[str]:
    out: Set[str] = set()
    if not text:
        return out

    # Basic direct emails
    for m in EMAIL_REGEX.findall(text):
        out.add(m.strip().lower())

    # mailto: may contain URL-encoded strings and extra junk (subject/body)
    for raw in MAILTO_REGEX.findall(text):
        candidate = unquote((raw or "").strip())
        for m in EMAIL_REGEX.findall(candidate):
            out.add(m.strip().lower())

    return out


def _clean_phone(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"[^\d+]", "", s)
    if s.startswith("00"):
        s = "+" + s[2:]
    return s


def _extract_phones_from_text(text: str) -> Set[str]:
    out: Set[str] = set()
    if not text:
        return out

    for m in PHONE_REGEX.findall(text):
        p = _clean_phone(m)
        digits = re.sub(r"\D", "", p)
        if 7 <= len(digits) <= 18:
            out.add(p)

    return out


def _extract_from_jsonld(text: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract emails and phones from JSON-LD blocks.

    We only walk JSON that appears under:
      <script type="application/ld+json"> ... </script>
    """
    emails: Set[str] = set()
    phones: Set[str] = set()
    if not text:
        return emails, phones

    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).strip().lower()
                if lk == "email" and isinstance(v, str):
                    for e in EMAIL_REGEX.findall(v):
                        emails.add(e.strip().lower())
                elif lk in ("telephone", "tel") and isinstance(v, str):
                    for p in PHONE_REGEX.findall(v):
                        phones.add(_clean_phone(p))
                else:
                    walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    for b in blocks:
        b = (b or "").strip()
        if not b:
            continue
        try:
            parsed = json.loads(b)
        except Exception:
            continue
        walk(parsed)

    return emails, phones


# -----------------------------
# Task 5.4 Filtering
# -----------------------------
def _is_free_provider(email: str, free_providers: Set[str]) -> bool:
    try:
        domain = email.split("@", 1)[1].lower().strip()
        return domain in free_providers
    except Exception:
        return False


def _is_no_reply(email: str, no_reply_locals: Set[str]) -> bool:
    try:
        local = email.split("@", 1)[0].lower().strip()

        # Normalize variants like "no-reply+tag"
        local = local.split("+", 1)[0]
        local = re.sub(r"[^a-z0-9\-]", "", local)

        return local in no_reply_locals
    except Exception:
        return False


def _filter_emails(
        emails: Iterable[str],
        official_domain: str,
        *,
        free_providers: Set[str],
        role_locals: Set[str],
        no_reply_locals: Set[str],
) -> List[str]:
    """
    Filtering policy (deterministic):
      - always drop noreply-like mailboxes
      - prefer official-domain emails if present
      - within official-domain, prefer role locals (info@, contact@, ...)
      - drop free-provider emails if any non-free exists
      - otherwise keep the remaining (excluding noreply)
    """
    official_domain = (official_domain or "").lower().strip()

    raw = [e.strip().lower() for e in emails if e and "@" in e]
    raw = _unique_preserve_order(raw)

    non_noreply = [e for e in raw if not _is_no_reply(e, no_reply_locals)]

    official: List[str] = []
    other: List[str] = []
    for e in non_noreply:
        try:
            d = e.split("@", 1)[1].lower().strip()
        except Exception:
            continue
        if official_domain and d == official_domain:
            official.append(e)
        else:
            other.append(e)

    def rank_email(e: str) -> Tuple[int, int, str]:
        """
        Deterministic ranking:
          1) role local preferred
          2) shorter local preferred
          3) lexical tie-break
        """
        local = e.split("@", 1)[0].lower().strip()
        local_base = local.split("+", 1)[0]
        is_role = local_base in role_locals
        return (0 if is_role else 1, len(local_base), e)

    if official:
        return sorted(_unique_preserve_order(official), key=rank_email)

    non_free = [e for e in other if not _is_free_provider(e, free_providers)]
    if non_free:
        return sorted(_unique_preserve_order(non_free), key=rank_email)

    return sorted(_unique_preserve_order(other), key=rank_email)


def _pick_phone(phones: Iterable[str]) -> Optional[str]:
    """
    Deterministic phone selection:
      - prefer +<country>... numbers over local-only
      - prefer shorter digit counts (usually the "base" number)
      - lexical tie-break
    """
    cleaned: List[str] = []
    for p in phones:
        s = _clean_phone(p)
        digits = re.sub(r"\D", "", s)
        if 7 <= len(digits) <= 18:
            cleaned.append(s)

    cleaned = _unique_preserve_order(cleaned)
    if not cleaned:
        return None

    def rank(p: str) -> Tuple[int, int, str]:
        digits = re.sub(r"\D", "", p)
        is_international = p.startswith("+")
        return (0 if is_international else 1, len(digits), p)

    return sorted(cleaned, key=rank)[0]


# -----------------------------
# Domain selection
# -----------------------------
def _pick_official_domain(r: Record) -> Optional[str]:
    """
    Priority:
      1) Phase 4 resolved domain (if present)
      2) first valid domain
      3) None
    """
    rd = getattr(getattr(r, "resolution", None), "resolved_domain", None)
    if isinstance(rd, str) and rd.strip():
        return rd.strip().lower()

    v = getattr(r, "valid_domains", None) or []
    if isinstance(v, list) and v:
        if isinstance(v[0], str) and v[0].strip():
            return v[0].strip().lower()

    return None


# -----------------------------
# Per-record crawl pipeline
# -----------------------------
async def _crawl_record(
        r: Record,
        cfg: PipelineConfig,
        client: httpx.AsyncClient,
        limiter: _PerDomainRateLimiter,
        target_paths: List[str],
        free_providers: Set[str],
        role_locals: Set[str],
        no_reply_locals: Set[str],
) -> None:
    """
    Crawl a resolved domain and extract contacts.

    Idempotency:
      - always resets r.contacts.* and phase5 debug keys up front
      - never uses randomness
      - crawl order is deterministic (seed URLs then discovered links in-order)

    Status/source policy:
      - Only set r.status="success" if we actually found at least one email or phone.
      - Set r.source="crawler" only when we found contacts via crawling.
      - If crawl fails/blocked and no contacts, we do NOT force success.
    """
    # Reset outputs (idempotency)
    r.contacts.emails = []
    r.contacts.phone = None
    r.contacts.crawl_status = "not_started"

    if getattr(r, "debug", None) is None:
        r.debug = {}

    r.debug["phase5_reason"] = "unset"
    r.debug["phase5_domain"] = None
    r.debug["phase5_seed_urls"] = []
    r.debug["phase5_pages_fetched"] = 0
    r.debug["phase5_bytes_total"] = 0
    r.debug["phase5_fetch_log"] = []
    r.debug["phase5_extracted_emails_raw"] = []
    r.debug["phase5_extracted_phones_raw"] = []
    r.debug["phase5_blocked"] = False

    # Only crawl resolved domains
    domain = _pick_official_domain(r)
    r.debug["phase5_domain"] = domain

    if not domain:
        r.contacts.crawl_status = "not_started"
        r.debug["phase5_reason"] = "no_resolved_domain"
        return

    # Skip ambiguous records (Phase 4 sets ambiguity_flag)
    if bool(getattr(getattr(r, "resolution", None), "ambiguity_flag", False)):
        r.contacts.crawl_status = "not_started"
        r.debug["phase5_reason"] = "skipped_ambiguous"
        return

    # Config bounds
    max_pages = max(1, int(getattr(cfg, "crawl_max_pages", 5)))
    max_depth = max(0, int(getattr(cfg, "crawl_max_depth", 2)))
    total_byte_limit = int(float(getattr(cfg, "crawl_total_byte_limit", 2_000_000)))
    max_bytes_per_page = int(float(getattr(cfg, "crawl_max_bytes_per_page", 300_000)))
    timeout_s = float(getattr(cfg, "crawl_timeout_s", 10.0))

    # Seed URLs: high-yield paths under https/http
    seed_urls = _select_target_urls(domain, target_paths)
    r.debug["phase5_seed_urls"] = seed_urls[:]
    if not seed_urls:
        r.contacts.crawl_status = "failed"
        r.debug["phase5_reason"] = "no_seed_urls"
        return

    # Deterministic BFS queue
    q: List[_QueuedURL] = [_QueuedURL(url=u, depth=0) for u in seed_urls]
    seen_urls: Set[str] = set()
    pages_fetched = 0
    bytes_total = 0

    emails_raw: Set[str] = set()
    phones_raw: Set[str] = set()

    blocked = False
    any_timeout = False
    any_connect = False
    any_other_err = False

    while q and pages_fetched < max_pages and bytes_total < total_byte_limit:
        item = q.pop(0)
        url = _canonicalize_url(item.url)
        depth = int(item.depth)

        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        if depth > max_depth:
            continue
        if not _is_same_domain_or_www(url, domain):
            continue

        # Polite per-domain delay (shared limiter across records)
        await limiter.wait(domain)

        remaining_total = max(0, total_byte_limit - bytes_total)
        per_page_cap = min(max_bytes_per_page, remaining_total)
        if per_page_cap <= 0:
            break

        fr = await _fetch_limited(client, url, timeout_s=timeout_s, max_bytes=per_page_cap)
        fr = _FetchResult(
            url=fr.url,
            depth=depth,
            status=fr.status,
            content_type=fr.content_type,
            body=fr.body,
            bytes_read=fr.bytes_read,
            error_kind=fr.error_kind,
        )

        pages_fetched += 1
        bytes_total += int(fr.bytes_read)

        # Log fetch
        r.debug["phase5_fetch_log"].append(
            {
                "url": fr.url,
                "depth": depth,
                "status": fr.status,
                "content_type": fr.content_type,
                "bytes_read": fr.bytes_read,
                "error_kind": fr.error_kind,
            }
        )

        # Basic block detection
        if fr.status in DEFAULT_BLOCK_STATUSES:
            blocked = True
        if fr.body and _looks_blocked_by_body(fr.body):
            blocked = True

        # Track error kinds
        if fr.error_kind == "timeout":
            any_timeout = True
        elif fr.error_kind == "connect":
            any_connect = True
        elif fr.error_kind in ("read", "other"):
            any_other_err = True

        # Extraction (only if we got body)
        if fr.body:
            e1 = _extract_emails_from_text(fr.body)
            p1 = _extract_phones_from_text(fr.body)
            e2, p2 = _extract_from_jsonld(fr.body)

            emails_raw.update(e1)
            emails_raw.update(e2)
            phones_raw.update(p1)
            phones_raw.update(p2)

            # Link discovery (bounded by depth + max_pages)
            if depth < max_depth and pages_fetched < max_pages:
                discovered = _extract_links(fr.body, base_url=fr.url, domain=domain)
                # enqueue in deterministic order
                for u2 in discovered:
                    if u2 not in seen_urls:
                        q.append(_QueuedURL(url=u2, depth=depth + 1))

    # Save crawl stats
    r.debug["phase5_pages_fetched"] = pages_fetched
    r.debug["phase5_bytes_total"] = bytes_total
    r.debug["phase5_blocked"] = bool(blocked)

    r.debug["phase5_extracted_emails_raw"] = sorted(emails_raw)
    r.debug["phase5_extracted_phones_raw"] = sorted(phones_raw)

    # Filtering + outputs
    filtered_emails = _filter_emails(
        emails_raw,
        official_domain=domain,
        free_providers=free_providers,
        role_locals=role_locals,
        no_reply_locals=no_reply_locals,
    )
    picked_phone = _pick_phone(phones_raw)

    r.contacts.emails = filtered_emails
    r.contacts.phone = picked_phone

    found_any = bool(filtered_emails) or bool(picked_phone)

    # Crawl status
    if found_any:
        r.contacts.crawl_status = "success"
        r.status = "success"
        r.source = "crawler"
        r.debug["phase5_reason"] = "contacts_found"
        return

    # No contacts: classify crawl outcome
    if blocked:
        r.contacts.crawl_status = "blocked"
        r.debug["phase5_reason"] = "blocked_no_contacts"
        # Optional: mark failure if you want a terminal-ish state without contacts
        # while still preserving Phase 4's resolution info.
        if r.status == "resolved":
            r.status = "failed"
        return

    if any_timeout:
        r.contacts.crawl_status = "timeout"
        r.debug["phase5_reason"] = "timeout_no_contacts"
        if r.status == "resolved":
            r.status = "failed"
        return

    if any_connect or any_other_err:
        r.contacts.crawl_status = "failed"
        r.debug["phase5_reason"] = "fetch_errors_no_contacts"
        if r.status == "resolved":
            r.status = "failed"
        return

    # Completed but empty
    r.contacts.crawl_status = "failed"
    r.debug["phase5_reason"] = "no_contacts_found"


# -----------------------------
# Async runner
# -----------------------------
async def run_phase_async(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 5: Web Crawling and Extraction (async)

    Inputs:
      - record.resolution.resolved_domain (preferred)
      - record.valid_domains (fallback)
      - Config crawl bounds and http settings

    Output:
      - record.contacts.emails
      - record.contacts.phone
      - record.contacts.crawl_status
      - record.status may become "success" only when contacts exist
      - record.source becomes "crawler" only when contacts were extracted by crawling
      - record.debug["phase5_*"]
    """
    target_paths, free_set, role_set, no_reply_set = _load_phase5_assets(cfg)

    ua = getattr(
        cfg,
        "http_user_agent",
        "Mozilla/5.0 (compatible; CompanyPipeline/1.0; +https://example.invalid)",
    )
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    limits = httpx.Limits(
        max_keepalive_connections=int(getattr(cfg, "http_max_keepalive_connections", 50)),
        max_connections=int(getattr(cfg, "http_max_connections", 200)),
    )

    # Use crawl_timeout_s for per-request GET timeouts in this phase (separate from Phase 3).
    timeout = httpx.Timeout(float(getattr(cfg, "crawl_timeout_s", 10.0)), connect=float(getattr(cfg, "crawl_timeout_s", 10.0)))

    global_conc = max(1, int(getattr(cfg, "crawl_global_concurrency", 15)))
    sem = asyncio.Semaphore(global_conc)

    limiter = _PerDomainRateLimiter(float(getattr(cfg, "crawl_per_domain_delay_s", 0.5)))

    # Idempotency: reset for all records (even skipped)
    for r in records:
        r.contacts.emails = []
        r.contacts.phone = None
        r.contacts.crawl_status = "not_started"
        if getattr(r, "debug", None) is None:
            r.debug = {}
        r.debug["phase5_reason"] = "unset"

    async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            limits=limits,
            headers=headers,
    ) as client:

        async def _guarded(r: Record) -> None:
            # Skip dropped records deterministically
            if (r.status or "").startswith("dropped"):
                r.contacts.crawl_status = "not_started"
                r.debug["phase5_reason"] = f"skipped_status:{r.status}"
                return

            async with sem:
                await _crawl_record(
                    r,
                    cfg,
                    client,
                    limiter,
                    target_paths=target_paths,
                    free_providers=free_set,
                    role_locals=role_set,
                    no_reply_locals=no_reply_set,
                )

        # Batch spawning to avoid huge task lists
        batch_size = max(1, int(getattr(cfg, "phase5_record_batch_size", 250)))
        work = [r for r in records if not (r.status or "").startswith("dropped")]

        for i in range(0, len(work), batch_size):
            chunk = work[i : i + batch_size]
            tasks = [asyncio.create_task(_guarded(r)) for r in chunk]
            if tasks:
                await asyncio.gather(*tasks)

    logger.info("Phase 5 complete: crawled %d records", len(records))
    return records


# -----------------------------
# Sync wrapper (pipeline_runner.py expects this)
# -----------------------------
def run_phase(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 5: Web Crawling and Extraction (sync wrapper)

    Output:
      - record.contacts

    Note:
      If you are running inside an existing event loop (e.g., Jupyter/Colab),
      call run_phase_async(records, cfg) instead.
    """
    try:
        return asyncio.run(run_phase_async(records, cfg))
    except RuntimeError as e:
        msg = str(e).lower()
        if "running event loop" in msg or "asyncio.run()" in msg:
            raise RuntimeError(
                "Phase 5 run_phase() was called from a running event loop. "
                "Use: await run_phase_async(records, cfg) instead."
            ) from e
        raise
