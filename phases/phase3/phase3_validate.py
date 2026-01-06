from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import logging
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx

from config import PipelineConfig
from schemas import Record

logger = logging.getLogger("Phase 3")

# -----------------------------
# Defaults (used only if assets/phase3/* are missing)
# -----------------------------
DEFAULT_PARKED_KEYWORDS: List[str] = [
    "domain for sale",
    "this domain is for sale",
    "buy this domain",
    "purchase this domain",
    "inquire about this domain",
    "make an offer",
    "sedo",
    "afternic",
    "dan.com",
    "godaddy",
    "namecheap",
    "parking",
    "parked",
    "hugedomains",
    "uniregistry",
    "above.com",
    "name.com",
    "register this domain",
]

DEFAULT_DIRECTORY_HOST_HINTS: List[str] = [
    "facebook.com",
    "linkedin.com",
    "google.com",
    "goo.gl",
    "maps.google",
    "yelp.",
    "yellowpages.",
    "companieshouse.",
    "opencorporates.",
    "bloomberg.com",
    "crunchbase.com",
    "dnb.com",
    "zoominfo.com",
]


@dataclass(frozen=True)
class DomainDNSResult:
    """
    Task 3.1 DNS Result
    """
    domain: str
    alive: bool
    wildcard: bool
    ips: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class DomainHTTPResult:
    """
    Task 3.2 HTTP Result
    """
    domain: str
    ok: bool
    reason: str
    scheme_used: Optional[str]
    final_host: Optional[str]
    status_code: Optional[int]
    content_type: Optional[str]


# -----------------------------
# Phase 3 assets
# -----------------------------
def _read_asset_lines(path: Path) -> List[str]:
    """
    Phase asset reader (line-based):
      - one entry per line
      - ignore empty lines
      - ignore comments starting with '#'
    """
    if not path.exists():
        return []
    try:
        out: List[str] = []
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = (raw or "").strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
        return out
    except Exception:
        return []


def _normalize_asset_token(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _load_phase3_assets(cfg: PipelineConfig) -> Tuple[List[str], List[str]]:
    """
    Load Phase 3 assets from assets/phase3/.

    Expected files:
      - parked_keywords.txt
      - directory_host_hints.txt

    Fallback:
      Uses conservative built-in defaults if assets are missing/unreadable.
    """
    parked_path = cfg.phase_asset_path(3, "parked_keywords.txt")
    dir_path = cfg.phase_asset_path(3, "directory_host_hints.txt")

    parked = [_normalize_asset_token(x) for x in _read_asset_lines(parked_path)]
    dirs = [_normalize_asset_token(x) for x in _read_asset_lines(dir_path)]

    if not parked:
        logger.warning("Phase3 parked keywords missing/empty: %s (using defaults)", parked_path)
        parked = [_normalize_asset_token(x) for x in DEFAULT_PARKED_KEYWORDS]

    if not dirs:
        logger.warning("Phase3 directory host hints missing/empty: %s (using defaults)", dir_path)
        dirs = [_normalize_asset_token(x) for x in DEFAULT_DIRECTORY_HOST_HINTS]

    parked = [x for x in parked if x]
    dirs = [x for x in dirs if x]
    return parked, dirs


# -----------------------------
# Shared helpers
# -----------------------------
def _deterministic_label(domain: str, n: int = 12) -> str:
    """
    Deterministic "random" label for wildcard DNS probing.
    """
    d = (domain or "").strip().lower().encode("utf-8", errors="ignore")
    h = hashlib.sha1(d).hexdigest()
    return (h[:n] or "abc123def456")[:n]


def _is_ip_like(host: str) -> bool:
    try:
        socket.inet_pton(socket.AF_INET, host)
        return True
    except OSError:
        pass
    try:
        socket.inet_pton(socket.AF_INET6, host)
        return True
    except OSError:
        return False


# Basic hostname sanity (cheap filter). Accepts multi-label hosts like "a.co.uk".
# FIX: allow hyphens in the final label too (punycode TLDs like "xn--p1ai").
_LABEL_RE = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"

# IMPORTANT:
# This regex is built using an f-string (rf"..."). In f-strings, "{1,253}" would be interpreted
# as a Python expression (a tuple) and would BREAK the regex. So we must escape braces as "{{" and "}}".
_DOMAIN_RE = re.compile(rf"^(?=.{{1,253}}$)({_LABEL_RE}\.)+{_LABEL_RE}$")


def _sanitize_domain(s: str) -> str:
    """
    Normalize a domain-like string to a canonical host:
      - lowercase
      - strip scheme
      - strip path/query/fragment
      - strip port
      - strip trailing dot
      - reject spaces
      - reject raw IPs
      - basic hostname sanity check
    """
    raw = (s or "").strip()
    if not raw:
        return ""

    # Ensure urlparse extracts hostname even if scheme is missing
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", raw):
        raw_for_parse = "http://" + raw
    else:
        raw_for_parse = raw

    try:
        u = urlparse(raw_for_parse)
        host = (u.hostname or "").strip().lower().rstrip(".")
    except Exception:
        host = raw.lower().strip()
        host = re.sub(r"^https?://", "", host).strip()
        host = host.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].strip()
        host = host.split(":", 1)[0].strip()
        host = host.rstrip(".")

    if not host or " " in host:
        return ""

    # IDN support: convert Unicode hostnames to punycode (deterministic)
    try:
        # If host is already ASCII, this is a no-op.
        host = host.encode("idna").decode("ascii")
    except Exception:
        # If IDNA encoding fails, keep original (will likely fail regex).
        pass

    if _is_ip_like(host):
        return ""

    if "." not in host or not _DOMAIN_RE.match(host):
        return ""

    return host


def _coerce_domain_candidates(value: object) -> List[str]:
    """
    IMPORTANT FIX:
    Some pipelines accidentally persist `domain_candidates` as a STRING, e.g.
      "['a.com', 'b.co.uk']"
    or
      '["a.com","b.co.uk"]'
    or even
      'a.com, b.co.uk'

    If we treat that string as an iterable, we iterate over CHARACTERS and
    `_sanitize_domain()` rejects everything -> no_valid_candidates_after_sanitize
    and Phase 3 sends zero DNS/HTTP requests.

    This helper coerces that value into a real List[str] defensively.
    """
    if value is None:
        return []

    # Already list-like
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for v in value:
            if v is None:
                continue
            out.append(v if isinstance(v, str) else str(v))
        return out

    # String inputs (most common failure mode)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []

        # Try JSON first (works for '["a.com","b.co.uk"]')
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple, set)):
                return _coerce_domain_candidates(parsed)
        except Exception:
            pass

        # Try Python literal eval (works for "['a.com', 'b.co.uk']")
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return _coerce_domain_candidates(parsed)
        except Exception:
            pass

        # Fallback: comma/space split
        # Keep only non-empty tokens; sanitize later
        parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
        return parts

    # Any other type -> string it
    return [str(value)]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        d = _sanitize_domain(x)
        if not d or d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


# -----------------------------
# Task 3.1 DNS validation
# -----------------------------
async def _resolve_dns(
        domain: str,
        *,
        timeout_s: float,
        retries_on_timeout: int,
        retry_backoff_s: float,
        enable_wildcard_probe: bool,
) -> DomainDNSResult:
    """
    Resolve A/AAAA via getaddrinfo (system resolver).

    Wildcard probe (optional):
      - resolve a deterministic random subdomain
      - if it also resolves, treat as wildcard DNS
    """
    d = _sanitize_domain(domain)
    if not d:
        return DomainDNSResult(domain=domain, alive=False, wildcard=False, ips=(), reason="invalid_domain")

    async def _get_ips(host: str) -> Tuple[str, ...]:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
        ips: Set[str] = set()
        for info in infos:
            sockaddr = info[4]
            if sockaddr and isinstance(sockaddr, tuple):
                ips.add(sockaddr[0])
        return tuple(sorted(ips))

    attempt = 0
    while True:
        try:
            ips = await asyncio.wait_for(_get_ips(d), timeout=timeout_s)
            break
        except asyncio.TimeoutError:
            if attempt < retries_on_timeout:
                attempt += 1
                await asyncio.sleep(max(0.0, float(retry_backoff_s)))
                continue
            return DomainDNSResult(domain=d, alive=False, wildcard=False, ips=(), reason="dns_timeout")
        except socket.gaierror:
            return DomainDNSResult(domain=d, alive=False, wildcard=False, ips=(), reason="dns_nxdomain")
        except Exception as e:
            return DomainDNSResult(domain=d, alive=False, wildcard=False, ips=(), reason=f"dns_error:{type(e).__name__}")

    if not ips:
        return DomainDNSResult(domain=d, alive=False, wildcard=False, ips=(), reason="dns_no_ips")

    wildcard = False
    if enable_wildcard_probe:
        try:
            probe = f"{_deterministic_label(d)}.{d}"
            probe_ips = await asyncio.wait_for(_get_ips(probe), timeout=max(1.0, timeout_s / 2))
            if probe_ips:
                wildcard = True
        except Exception:
            wildcard = False

    return DomainDNSResult(domain=d, alive=True, wildcard=wildcard, ips=ips, reason="dns_ok")


# -----------------------------
# Task 3.2 HTTP validation
# -----------------------------
def _looks_parked_or_for_sale(html_snippet: str, parked_keywords: List[str]) -> bool:
    if not html_snippet:
        return False
    text = html_snippet.lower()
    return any(k in text for k in parked_keywords)


def _host_matches_hint(host: str, hint: str) -> bool:
    """
    Safer matching than `hint in host`:

    - If hint ends with '.' (like 'yelp.'), treat it as a contains-subdomain-marker match.
    - Else: exact or subdomain match (host == hint OR host endswith '.'+hint)
    """
    h = (host or "").lower().strip(".")
    x = (hint or "").lower().strip()
    if not h or not x:
        return False

    if x.endswith("."):
        return x in (h + ".")
    if h == x:
        return True
    return h.endswith("." + x)


def _host_in_directory_list(host: str, directory_hints: List[str]) -> bool:
    h = (host or "").lower().strip(".")
    return any(_host_matches_hint(h, x) for x in directory_hints)


async def _fetch_snippet(
        client: httpx.AsyncClient,
        url: str,
        *,
        range_end: int,
        snippet_cap: int,
) -> Tuple[Optional[int], Optional[str], bytes, str]:
    """
    Fetch a small snippet of a page, even if Range is ignored.
    Returns: (status_code, final_host, snippet_bytes, content_type)
    """
    headers = {"Range": f"bytes=0-{max(0, int(range_end))}"}
    content = bytearray()
    content_type = ""

    async with client.stream("GET", url, headers=headers) as resp:
        content_type = (resp.headers.get("content-type") or "").lower()
        final_host = resp.url.host or ""
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            take = min(len(chunk), snippet_cap - len(content))
            if take > 0:
                content.extend(chunk[:take])
            if len(content) >= snippet_cap:
                break
        return resp.status_code, final_host, bytes(content), content_type


async def _http_check_domain(
        domain: str,
        client: httpx.AsyncClient,
        cfg: PipelineConfig,
        parked_keywords: List[str],
        directory_hints: List[str],
) -> DomainHTTPResult:
    """
    HTTP reachability check:
      - try https then http
      - do HEAD first (cheap)
      - do GET snippet when:
          - status is blocked/retryable (e.g., 403/405), OR
          - cfg.phase3_snippet_on_success is True (parked detection)
      - reject if redirected to a known directory/listing host
      - optionally treat 401/403 as acceptable (protected but alive)

    IMPORTANT IMPROVEMENT:
      - If https fails (non-allowed status), we try http instead of failing early.
    """
    schemes = ["https", "http"]

    # Config knobs (with safe defaults)
    snippet_cap = int(getattr(cfg, "phase3_http_snippet_bytes", 8192))
    range_end = int(getattr(cfg, "phase3_http_range_end", 4095))
    head_retry_statuses = tuple(getattr(cfg, "phase3_head_retry_statuses", (403, 405)))
    allow_protected = bool(getattr(cfg, "phase3_allow_protected_status", False))
    snippet_on_success = bool(getattr(cfg, "phase3_snippet_on_success", True))

    last_error: Optional[str] = None

    for scheme in schemes:
        url = f"{scheme}://{domain}/"

        try:
            resp = await client.head(url)
            status = resp.status_code
            final_host = resp.url.host or None

            if final_host and _host_in_directory_list(final_host, directory_hints):
                return DomainHTTPResult(
                    domain=domain,
                    ok=False,
                    reason=f"redirected_to_directory:{final_host}",
                    scheme_used=scheme,
                    final_host=final_host,
                    status_code=status,
                    content_type=(resp.headers.get("content-type") or "").lower() or None,
                )

            # If HEAD returned an error and it's not a retryable "blocked" status, try next scheme.
            if status >= 400 and not (allow_protected and status in (401, 403)) and status not in head_retry_statuses:
                last_error = f"http_status_{status}"
                continue

            must_snippet = (status in head_retry_statuses) or snippet_on_success

            if must_snippet:
                status2, final_host2, snippet_bytes, ctype = await _fetch_snippet(
                    client,
                    url,
                    range_end=range_end,
                    snippet_cap=snippet_cap,
                )

                if final_host2 and _host_in_directory_list(final_host2, directory_hints):
                    return DomainHTTPResult(
                        domain=domain,
                        ok=False,
                        reason=f"redirected_to_directory:{final_host2}",
                        scheme_used=scheme,
                        final_host=final_host2,
                        status_code=status2,
                        content_type=ctype or None,
                    )

                if status2 is not None and status2 >= 400 and not (allow_protected and status2 in (401, 403)):
                    last_error = f"http_status_{status2}"
                    continue

                snippet = ""
                if ("text" in ctype) or ("html" in ctype) or (ctype.strip() == ""):
                    snippet = snippet_bytes.decode("utf-8", errors="ignore")

                if _looks_parked_or_for_sale(snippet, parked_keywords):
                    return DomainHTTPResult(
                        domain=domain,
                        ok=False,
                        reason="parked_or_for_sale",
                        scheme_used=scheme,
                        final_host=final_host2 or final_host,
                        status_code=status2,
                        content_type=ctype or None,
                    )

                return DomainHTTPResult(
                    domain=domain,
                    ok=True,
                    reason="http_ok",
                    scheme_used=scheme,
                    final_host=final_host2 or final_host,
                    status_code=status2,
                    content_type=ctype or None,
                )

            # No snippet required; HEAD is enough
            return DomainHTTPResult(
                domain=domain,
                ok=True,
                reason="http_ok_head_only",
                scheme_used=scheme,
                final_host=final_host,
                status_code=status,
                content_type=(resp.headers.get("content-type") or "").lower() or None,
            )

        except (httpx.TimeoutException,) as e:
            last_error = f"http_timeout:{type(e).__name__}"
            continue
        except (httpx.ConnectError, httpx.ReadError) as e:
            last_error = f"http_error:{type(e).__name__}"
            continue
        except Exception as e:
            last_error = f"http_error:{type(e).__name__}"
            continue

    return DomainHTTPResult(
        domain=domain,
        ok=False,
        reason=last_error or "http_unreachable",
        scheme_used=None,
        final_host=None,
        status_code=None,
        content_type=None,
    )


# -----------------------------
# Record validation (DNS + HTTP)
# -----------------------------
async def _validate_record_domains(
        record: Record,
        cfg: PipelineConfig,
        http_sem: asyncio.Semaphore,
        dns_sem: asyncio.Semaphore,
        client: httpx.AsyncClient,
        parked_keywords: List[str],
        directory_hints: List[str],
) -> None:
    """
    Phase 3 per-record pipeline.

    Implements:
      - Task 3.1 DNS validation (alive + optional wildcard detection)
      - Task 3.2 HTTP validation (reachable + parked/directory filtering)
      - Task 3.3 Ordered valid domain list

    Output:
      record.valid_domains (ordered)
      record.debug["phase3_*"]
    """
    # Idempotency: always reset outputs + phase debug keys
    record.valid_domains = []
    if getattr(record, "debug", None) is None:
        record.debug = {}
    record.debug["phase3_reason"] = "unset"
    record.debug["phase3_dns"] = []
    record.debug["phase3_http"] = []
    record.debug["phase3_policy"] = {}
    record.debug["phase3_candidate_input_type"] = type(getattr(record, "domain_candidates", None)).__name__

    # IMPORTANT FIX: coerce domain_candidates into a real List[str]
    candidates = _coerce_domain_candidates(getattr(record, "domain_candidates", None))
    record.debug["phase3_candidate_count_raw"] = len(candidates)

    if not candidates:
        record.debug["phase3_reason"] = "no_candidates"
        return

    ordered = _dedupe_preserve_order(list(candidates))
    record.debug["phase3_candidate_count_after_sanitize"] = len(ordered)
    record.debug["phase3_candidate_sample_sanitized"] = ordered[:10]

    if not ordered:
        # This is the exact failure you reported. With the coercion above,
        # it should only happen when candidates truly contain no valid domains.
        record.debug["phase3_reason"] = "no_valid_candidates_after_sanitize"
        record.debug["phase3_candidate_sample_raw"] = candidates[:10]
        return

    # ---- Task 3.1: DNS check ----
    enable_wildcard_probe = bool(
        getattr(cfg, "phase3_enable_wildcard_probe", getattr(cfg, "enable_wildcard_dns_default", True))
    )
    reject_wildcard = bool(getattr(cfg, "phase3_reject_wildcard_dns", True))

    async def _guarded_dns(d: str) -> DomainDNSResult:
        async with dns_sem:
            return await _resolve_dns(
                d,
                timeout_s=float(cfg.dns_timeout_s),
                retries_on_timeout=int(cfg.dns_retries_on_timeout),
                retry_backoff_s=float(cfg.dns_retry_backoff),
                enable_wildcard_probe=enable_wildcard_probe,
            )

    dns_results = await asyncio.gather(*(asyncio.create_task(_guarded_dns(d)) for d in ordered))

    record.debug["phase3_dns"] = [
        {
            "domain": res.domain,
            "alive": res.alive,
            "wildcard": res.wildcard,
            "ips": list(res.ips),
            "reason": res.reason,
            "rejected_by_policy": bool(reject_wildcard and res.alive and res.wildcard),
        }
        for res in dns_results
    ]

    alive_domains: List[str] = []
    for res in dns_results:
        if not res.alive:
            continue
        if reject_wildcard and res.wildcard:
            continue
        alive_domains.append(res.domain)

    if not alive_domains:
        if reject_wildcard and any(res.alive and res.wildcard for res in dns_results):
            record.debug["phase3_reason"] = "only_wildcard_alive_domains"
        else:
            record.debug["phase3_reason"] = "no_dns_alive_domains_after_policy"
        record.debug["phase3_policy"] = {
            "phase3_enable_wildcard_probe": enable_wildcard_probe,
            "phase3_reject_wildcard_dns": reject_wildcard,
        }
        return

    # ---- Task 3.2: HTTP check ----
    async def _guarded_http(d: str) -> DomainHTTPResult:
        async with http_sem:
            return await _http_check_domain(d, client, cfg, parked_keywords, directory_hints)

    http_results = await asyncio.gather(*(asyncio.create_task(_guarded_http(d)) for d in alive_domains))

    record.debug["phase3_http"] = [
        {
            "domain": hr.domain,
            "ok": hr.ok,
            "reason": hr.reason,
            "scheme_used": hr.scheme_used,
            "final_host": hr.final_host,
            "status_code": hr.status_code,
            "content_type": hr.content_type,
        }
        for hr in http_results
    ]

    valid_set = {hr.domain for hr in http_results if hr.ok}
    record.valid_domains = [d for d in ordered if d in valid_set]

    record.debug["phase3_policy"] = {
        "dns_timeout_s": float(cfg.dns_timeout_s),
        "dns_retries_on_timeout": int(cfg.dns_retries_on_timeout),
        "dns_retry_backoff": float(cfg.dns_retry_backoff),
        "phase3_enable_wildcard_probe": enable_wildcard_probe,
        "phase3_reject_wildcard_dns": reject_wildcard,
        "phase3_allow_protected_status": bool(getattr(cfg, "phase3_allow_protected_status", False)),
        "phase3_snippet_on_success": bool(getattr(cfg, "phase3_snippet_on_success", True)),
        "phase3_head_retry_statuses": list(getattr(cfg, "phase3_head_retry_statuses", (403, 405))),
        "phase3_http_range_end": int(getattr(cfg, "phase3_http_range_end", 4095)),
        "phase3_http_snippet_bytes": int(getattr(cfg, "phase3_http_snippet_bytes", 8192)),
    }

    record.debug["phase3_reason"] = "ok" if record.valid_domains else "no_http_reachable_domains"


# -----------------------------
# Async runner
# -----------------------------
async def run_phase_async(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 3: Fast Domain Validation (async)

    Implements:
      - Task 3.1 DNS checks for candidate domains
      - Task 3.2 HTTP checks for DNS-alive domains
      - Task 3.3 Stores ordered valid domains

    Output:
      record.valid_domains
      record.debug["phase3_*"]
    """
    parked_keywords, directory_hints = _load_phase3_assets(cfg)

    http_conc = int(getattr(cfg, "phase3_http_concurrency", getattr(cfg, "crawl_global_concurrency", 10)))
    dns_conc = int(getattr(cfg, "dns_limit", 25))

    http_sem = asyncio.Semaphore(max(1, http_conc))
    dns_sem = asyncio.Semaphore(max(1, dns_conc))

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
        max_keepalive_connections=int(cfg.http_max_keepalive_connections),
        max_connections=int(cfg.http_max_connections),
    )
    timeout = httpx.Timeout(float(cfg.http_timeout_s), connect=float(cfg.http_timeout_s))

    # Batch records to avoid spawning too many tasks at once.
    batch_size = int(getattr(cfg, "phase3_record_batch_size", 250))
    batch_size = max(1, batch_size)

    # FIX: idempotency for ALL records (including dropped/skipped)
    for r in records:
        r.valid_domains = []
        if getattr(r, "debug", None) is None:
            r.debug = {}
        r.debug["phase3_dns"] = []
        r.debug["phase3_http"] = []
        r.debug["phase3_policy"] = {}
        status = (r.status or "")
        if status.startswith("dropped"):
            r.debug["phase3_reason"] = f"skipped_status:{status}"
        else:
            r.debug["phase3_reason"] = "unset"

    async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            limits=limits,
            headers=headers,
    ) as client:
        work: List[Record] = [r for r in records if not (r.status or "").startswith("dropped")]

        for i in range(0, len(work), batch_size):
            chunk = work[i:i + batch_size]
            tasks = [
                asyncio.create_task(
                    _validate_record_domains(r, cfg, http_sem, dns_sem, client, parked_keywords, directory_hints)
                )
                for r in chunk
            ]
            if tasks:
                await asyncio.gather(*tasks)

    logger.info("Phase 3 complete: validated domains for %d records", len(records))
    return records


# -----------------------------
# Sync wrapper (pipeline_runner.py expects this)
# -----------------------------
def run_phase(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 3: Fast Domain Validation (sync wrapper)

    Output:
      record.valid_domains

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
                "Phase 3 run_phase() was called from a running event loop. "
                "Use: await run_phase_async(records, cfg) instead."
            ) from e
        raise
