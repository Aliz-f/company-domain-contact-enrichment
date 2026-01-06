from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from config import PipelineConfig
from schemas import Record

logger = logging.getLogger("Phase7")


# -----------------------------------------------------------------------------
# 7.1 Email Syntax Validation (RFC-style regex)
# -----------------------------------------------------------------------------
# This regex is similar to what Django uses (RFC 5322-ish) and is widely accepted
# in production validation pipelines. It supports quoted local parts and IDN/punycode.
_EMAIL_RE = re.compile(
    r"(^[-!#$%&'*+/=?^_`{}|~0-9A-Z]+(\.[-!#$%&'*+/=?^_`{}|~0-9A-Z]+)*"
    r'|^"([\001-\010\013\014\016-\037!#-\[\]-\177]|\\[\001-\011\013\014\016-\177])*"'
    r")@((?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
    r"(?:[A-Z]{2,63}|XN--[A-Z0-9]{1,59})|\[(?:IPv6:)?[A-F0-9:.]+\])$",
    re.IGNORECASE,
)


def _is_valid_email_syntax(email: str) -> bool:
    """
    Validates email syntax using an RFC-style regex.
    """
    if not email:
        return False
    e = email.strip()
    if len(e) > 254:
        return False
    return _EMAIL_RE.match(e) is not None


def _email_domain(email: str) -> Optional[str]:
    """
    Extract domain from email; returns lowercased domain.
    """
    if not email or "@" not in email:
        return None
    parts = email.rsplit("@", 1)
    if len(parts) != 2:
        return None
    dom = parts[1].strip().lower()
    return dom or None


def _normalize_email(email: str) -> str:
    """
    Normalize email for comparison/dedup:
      - strip whitespace
      - lowercase
      - remove trailing punctuation commonly captured from HTML text
    """
    e = (email or "").strip().lower()
    e = e.strip(" ,.;:()[]<>\"'")
    return e


# -----------------------------------------------------------------------------
# 7.2 MX Record Check (DNS)
# -----------------------------------------------------------------------------
def _mx_lookup(domain: str, timeout_s: float) -> Optional[bool]:
    """
    Returns:
      True  -> MX exists
      False -> NXDOMAIN / NoAnswer (no MX)
      None  -> Unknown (DNS library missing, timeout, resolver errors)
    """
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain:
        return None

    try:
        import dns.exception
        import dns.resolver
    except Exception:
        logger.warning("dnspython is not installed; MX check will be 'unknown'.")
        return None

    resolver = dns.resolver.Resolver(configure=True)
    resolver.lifetime = float(timeout_s)
    resolver.timeout = float(timeout_s)

    try:
        answers = resolver.resolve(domain, "MX")
        return bool(answers)
    except dns.resolver.NXDOMAIN:
        return False
    except dns.resolver.NoAnswer:
        return False
    except dns.resolver.NoNameservers:
        return None
    except dns.exception.Timeout:
        return None
    except Exception:
        return None


def _mx_exists_with_retries(
        domain: str,
        *,
        timeout_s: float,
        retries: int,
        backoff_s: float,
) -> Optional[bool]:
    """
    Retry MX lookup on timeout/unknown results.
    """
    last: Optional[bool] = None
    for attempt in range(max(0, int(retries)) + 1):
        result = _mx_lookup(domain, timeout_s=timeout_s)
        last = result
        if result in (True, False):
            return result
        if attempt < retries:
            time.sleep(float(backoff_s))
    return last


# -----------------------------------------------------------------------------
# 7.3 Deduplication (stable)
# -----------------------------------------------------------------------------
def _dedupe_emails_preserve_order(emails: List[str]) -> List[str]:
    """
    Stable deduplication by normalized email.
    Keeps first occurrence (deterministic).
    """
    seen: set[str] = set()
    out: List[str] = []
    for e in emails:
        ne = _normalize_email(e)
        if not ne or ne in seen:
            continue
        seen.add(ne)
        out.append(ne)
    return out


# -----------------------------------------------------------------------------
# Main Phase 7 runner
# -----------------------------------------------------------------------------
def run_phase(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 7: Post-validation (no LLM)
      7.1 Email Syntax Validation (RFC-style regex)
      7.2 MX Record existence check
      7.3 Deduplication (per record)

    Input:
      - record.contacts.emails (Phase 5 output)

    Output (schema-aligned):
      - record.validated_contacts.emails (List[str])
      - record.validated_contacts.phone (copied from record.contacts.phone)
      - record.validated_contacts.crawl_status (copied from record.contacts.crawl_status)

    Debug (optional):
      - record.debug["phase7_validated_emails_n"]
      - record.debug["phase7_email_details"] = [{email, domain, syntax_valid, mx}, ...]
    """
    enable_syntax = bool(getattr(cfg, "phase7_enable_syntax_validation", True))
    enable_mx = bool(getattr(cfg, "phase7_enable_mx_check", True))
    enable_dedup = bool(getattr(cfg, "phase7_enable_deduplication", True))
    require_mx = bool(getattr(cfg, "phase7_require_mx", False))

    mx_timeout = float(getattr(cfg, "phase7_mx_timeout_s", getattr(cfg, "dns_timeout_s", 6.0)))
    mx_retries = int(getattr(cfg, "phase7_mx_retries_on_timeout", getattr(cfg, "dns_retries_on_timeout", 1)))
    mx_backoff = float(getattr(cfg, "phase7_mx_retry_backoff", getattr(cfg, "dns_retry_backoff", 0.35)))

    cap = getattr(cfg, "phase7_max_validated_contacts_per_record", None)
    cap_n: Optional[int] = int(cap) if cap is not None else None

    # Cache MX across records within this run (reduces repeated DNS work).
    # Not persisted across runs -> keeps workers stateless.
    mx_cache_global: Dict[str, Optional[bool]] = {}

    for r in records:
        # Idempotency: reset validated_contacts every run
        r.validated_contacts.emails = []
        r.validated_contacts.phone = r.contacts.phone
        r.validated_contacts.crawl_status = r.contacts.crawl_status

        # Skip dropped records
        if (r.status or "").startswith("dropped"):
            if isinstance(r.debug, dict):
                r.debug["phase7_reason"] = f"skipped_status:{r.status}"
                r.debug["phase7_validated_emails_n"] = 0
            continue

        raw_emails = list(r.contacts.emails or [])

        validated_emails: List[str] = []
        details: List[Dict[str, Any]] = []

        for email_raw in raw_emails:
            email = _normalize_email(str(email_raw or ""))
            if not email:
                continue

            syntax_ok = True
            if enable_syntax:
                syntax_ok = _is_valid_email_syntax(email)
            if not syntax_ok:
                details.append(
                    {"email": email, "domain": _email_domain(email), "syntax_valid": False, "mx": None}
                )
                continue

            dom = _email_domain(email)
            mx_ok: Optional[bool] = None

            if enable_mx and dom:
                if dom in mx_cache_global:
                    mx_ok = mx_cache_global[dom]
                else:
                    mx_ok = _mx_exists_with_retries(
                        dom,
                        timeout_s=mx_timeout,
                        retries=mx_retries,
                        backoff_s=mx_backoff,
                    )
                    mx_cache_global[dom] = mx_ok

            if require_mx and mx_ok is not True:
                details.append({"email": email, "domain": dom, "syntax_valid": True, "mx": mx_ok})
                continue

            validated_emails.append(email)
            details.append({"email": email, "domain": dom, "syntax_valid": True, "mx": mx_ok})

            if cap_n is not None and len(validated_emails) >= cap_n:
                break

        if enable_dedup:
            validated_emails = _dedupe_emails_preserve_order(validated_emails)

        r.validated_contacts.emails = validated_emails

        # Debug summary
        if isinstance(getattr(r, "debug", None), dict):
            r.debug["phase7_validated_emails_n"] = len(validated_emails)
            r.debug["phase7_email_details"] = details

    return records
