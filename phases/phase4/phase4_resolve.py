from __future__ import annotations

import logging
import re
import socket
from typing import List, Optional, Set
from urllib.parse import urlparse

from config import PipelineConfig
from schemas import Record

logger = logging.getLogger("Phase 4")

# Basic hostname sanity (cheap filter). Accepts multi-label hosts like "a.co.uk".
# FIX: allow hyphens in the final label too (punycode TLDs like "xn--p1ai").
_LABEL_RE = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_DOMAIN_RE = re.compile(rf"^(?=.{{1,253}}$)({_LABEL_RE}\.)+{_LABEL_RE}$")


# -----------------------------
# Helpers
# -----------------------------
def _is_ip_like(host: str) -> bool:
    """Return True if host is a raw IPv4/IPv6 address."""
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


def _sanitize_domain(s: str) -> str:
    """
    Normalize a domain-like string to a canonical host:
      - lowercase
      - strip scheme
      - strip path/query/fragment
      - strip port
      - strip trailing dot
      - trim spaces
      - reject raw IPs
      - basic hostname sanity check

    Note:
      Phase 3 already sanitizes domains, but Phase 4 re-applies sanitation
      for robustness and idempotency (prevents stale/dirty values on reruns).

    Improvement:
      Supports IDN (internationalized domains) by converting Unicode hostnames
      to ASCII punycode via IDNA.
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
        host = host.encode("idna").decode("ascii")
    except Exception:
        pass

    if _is_ip_like(host):
        return ""

    if "." not in host or not _DOMAIN_RE.match(host):
        return ""

    return host


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """
    Deduplicate while preserving order (deterministic), after sanitation.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        d = _sanitize_domain(x)
        if not d or d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def _reset_resolution(r: Record) -> None:
    """
    Idempotency helper: clear resolution outputs to prevent stale values on reruns.
    """
    if getattr(r, "debug", None) is None:
        r.debug = {}

    r.resolution.resolved_domain = None

    # Reset to a neutral baseline; Phase 4 will set fields deterministically.
    r.resolution.ambiguity_flag = False
    r.resolution.reason = "unset"

    # Also clear mirrored debug values
    r.debug["phase4_resolved_domain"] = None
    r.debug["phase4_ambiguity_flag"] = False
    r.debug["phase4_reason"] = "unset"


# -----------------------------
# Phase runner
# -----------------------------
def run_phase(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 4: Domain Resolution (deterministic, no AI)

    Implements:
      - Task 4.1 Single Domain Resolution
        If exactly one valid domain -> resolved_domain.
      - Task 4.2 Ambiguity Detection
        If more than one valid domain -> ambiguity_flag = True.
      - Task 4.3 Unresolved
        If zero valid domains -> unresolved (not ambiguous).

    Inputs:
      - record.valid_domains (from Phase 3)

    Outputs:
      - record.resolution.resolved_domain
      - record.resolution.ambiguity_flag
      - record.resolution.reason
      - record.debug["phase4_*"]

    Status policy (configurable):
      - IMPORTANT: Phase 4 must NOT set status="success".
        "success" should be reserved for having contacts (Phase 5+).
      - If cfg.phase4_set_status is True:
          resolved   -> status = cfg.phase4_status_resolved   (default "resolved")
          ambiguous  -> status = cfg.phase4_status_ambiguous  (default "ambiguous")
          unresolved -> status = cfg.phase4_status_unresolved (default "unresolved")

    Idempotent:
      - Always resets resolution outputs first.
      - Same input record state + same config => same output.
    """
    set_status = bool(getattr(cfg, "phase4_set_status", True))

    # IMPORTANT: do not default to "success" here
    st_resolved = str(getattr(cfg, "phase4_status_resolved", "resolved"))
    st_ambiguous = str(getattr(cfg, "phase4_status_ambiguous", "ambiguous"))
    st_unresolved = str(getattr(cfg, "phase4_status_unresolved", "unresolved"))

    for r in records:
        status = (r.status or "")

        # Idempotency: clear outputs up-front
        _reset_resolution(r)

        # Skip dropped records (do not overwrite their status or proceed)
        if status.startswith("dropped"):
            r.resolution.reason = "skipped_dropped"
            r.debug["phase4_reason"] = f"skipped_status:{status}"
            r.debug["phase4_resolved_domain"] = None
            r.debug["phase4_ambiguity_flag"] = r.resolution.ambiguity_flag
            continue

        # Normalize valid_domains deterministically
        valid_domains = _dedupe_preserve_order(getattr(r, "valid_domains", []) or [])
        r.valid_domains = valid_domains

        # Task 4.1: Single domain resolved
        if len(valid_domains) == 1:
            resolved = valid_domains[0]
            r.resolution.resolved_domain = resolved
            r.resolution.ambiguity_flag = False
            r.resolution.reason = "single_domain_resolved"

            r.debug["phase4_reason"] = "single_domain_resolved"
            r.debug["phase4_resolved_domain"] = resolved
            r.debug["phase4_ambiguity_flag"] = False

            if set_status:
                r.status = st_resolved

            continue

        # Task 4.2/4.3: Ambiguous vs Unresolved
        r.resolution.resolved_domain = None

        if len(valid_domains) == 0:
            r.resolution.ambiguity_flag = False
            r.resolution.reason = "no_valid_domains"

            r.debug["phase4_reason"] = "unresolved_no_valid_domains"
            r.debug["phase4_resolved_domain"] = None
            r.debug["phase4_ambiguity_flag"] = False

            if set_status:
                r.status = st_unresolved
            continue

        # len(valid_domains) > 1
        r.resolution.ambiguity_flag = True
        r.resolution.reason = "multiple_valid_domains"

        r.debug["phase4_reason"] = f"ambiguous_multiple_valid_domains:{len(valid_domains)}"
        r.debug["phase4_resolved_domain"] = None
        r.debug["phase4_ambiguity_flag"] = True

        if set_status:
            r.status = st_ambiguous

    logger.info("Phase 4 complete: processed %d records", len(records))
    return records
