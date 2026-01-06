from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Set, Tuple

from config import PipelineConfig
from schemas import Record

logger = logging.getLogger("Phase 0")

# -----------------------------
# Defaults (used only if assets are missing)
# -----------------------------
DEFAULT_DUMMY_KEYWORDS: Set[str] = {
    "test",
    "dummy",
    "sample",
    "example",
    "do not use",
    "dont use",
    "do-not-use",
    "donotuse",
    "lorem ipsum",
    "placeholder",
    "asdf",
    "qwerty",
    "xxx",
    "xxxx",
    "demo",
    "trial",
    "fake",
}

DEFAULT_EMPTY_MARKERS: Set[str] = {
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "-",
    "--",
    "???",
    "?",
}


# -----------------------------
# Task 0.1 helpers
# -----------------------------
def _normalize_for_filter(s: str) -> str:
    """
    Normalize strings for robust matching in Phase 0 filters.

    Steps:
      - Unicode normalize (NFKC)
      - lowercase
      - strip
      - replace non-alphanumeric with spaces
      - collapse whitespace
    """
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _read_asset_lines(path: Path) -> List[str]:
    """
    Read a line-based asset file.

    Rules:
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


def _load_phase0_vocab(cfg: PipelineConfig) -> Tuple[Set[str], Set[str]]:
    """
    Load Phase 0 vocabulary from assets/phase0/.

    Expected files:
      - dummy_keywords.txt
      - empty_markers.txt

    Returns:
      (dummy_keywords, empty_markers) as normalized sets.

    Fallback:
      If assets are missing/unreadable, uses conservative built-in defaults
      (also normalized to match filter normalization).
    """
    dummy_path = cfg.phase_asset_path(0, "dummy_keywords.txt")
    empty_path = cfg.phase_asset_path(0, "empty_markers.txt")

    dummy_lines = _read_asset_lines(dummy_path)
    empty_lines = _read_asset_lines(empty_path)

    if dummy_lines:
        dummy = {_normalize_for_filter(x) for x in dummy_lines}
    else:
        dummy = {_normalize_for_filter(x) for x in DEFAULT_DUMMY_KEYWORDS}

    if empty_lines:
        empty = {_normalize_for_filter(x) for x in empty_lines}
    else:
        empty = {_normalize_for_filter(x) for x in DEFAULT_EMPTY_MARKERS}

    dummy.discard("")
    empty.discard("")
    return dummy, empty


def _is_empty_name(raw_name: Optional[str], empty_markers: Set[str]) -> Tuple[bool, str]:
    """
    Decide whether a company name is empty/placeholder.

    Returns:
      (is_empty, reason)
    """
    if raw_name is None:
        return True, "raw_name_is_none"

    s = str(raw_name).strip()
    if not s:
        return True, "raw_name_empty"

    norm = _normalize_for_filter(s)
    if not norm:
        return True, "raw_name_empty_after_normalization"

    if norm in empty_markers:
        return True, "raw_name_empty_marker"

    return False, ""


def _looks_like_dummy(raw_name: str, dummy_keywords: Set[str], min_name_len: int) -> Tuple[bool, str]:
    """
    Deterministic dummy/test detection for Phase 0.

    Rules (conservative):
      - exact match to a dummy keyword
      - contains a dummy keyword as a whole word/phrase
      - common "test company" patterns

    Returns:
      (is_dummy, reason)
    """
    norm = _normalize_for_filter(raw_name)
    if not norm:
        return True, "raw_name_empty_after_normalization"

    if norm in dummy_keywords:
        return True, "dummy_exact_keyword"

    for kw in dummy_keywords:
        if not kw:
            continue
        if re.search(rf"\b{re.escape(kw)}\b", norm):
            return True, f"dummy_contains_keyword:{kw}"

    if re.search(r"\btest\b.*\bcompany\b", norm) or re.search(r"\bcompany\b.*\btest\b", norm):
        return True, "dummy_test_company_pattern"

    # Phase 0 should not drop short names by default; real companies can be very short (BP, 3M, GM).
    _ = min_name_len
    return False, ""


def run_phase(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 0: Pre-filter and Routing (deterministic)

    Implements:
      - Task 0.1 Record Filter:
          Drop records with empty/placeholder names or dummy/test names.
          Tag kept records for downstream routing.

    Outputs (kept records):
      - record.tags.has_address
      - record.tags.has_country
      - record.tags.priority  ("high_priority" | "low_priority")
      - record.debug["phase0_reason"]

    Outputs (dropped records):
      - record.status = "dropped_dummy"
      - record.debug["phase0_reason"] = <reason>

    Notes:
      - This phase returns ONLY kept records (clean stream).
    """
    dummy_keywords, empty_markers = _load_phase0_vocab(cfg)
    min_len = int(getattr(cfg, "phase0_min_name_len", 3))

    kept: List[Record] = []
    dropped_count = 0

    for r in records:
        r.raw_name = (r.raw_name or "").strip()
        r.debug["phase0_reason"] = "unset"

        is_empty, empty_reason = _is_empty_name(r.raw_name, empty_markers)
        if is_empty:
            r.status = "dropped_dummy"
            r.debug["phase0_reason"] = empty_reason
            dropped_count += 1
            continue

        is_dummy, dummy_reason = _looks_like_dummy(r.raw_name, dummy_keywords, min_len)
        if is_dummy:
            r.status = "dropped_dummy"
            r.debug["phase0_reason"] = dummy_reason
            dropped_count += 1
            continue

        has_address = bool(r.address and str(r.address).strip())
        has_country = bool(r.country and str(r.country).strip())

        r.tags.has_address = has_address
        r.tags.has_country = has_country
        r.tags.priority = "high_priority" if (has_address or has_country) else "low_priority"

        r.debug["phase0_reason"] = "kept"
        kept.append(r)

    logger.info("Phase 0 complete: input=%d kept=%d dropped=%d", len(records), len(kept), dropped_count)
    return kept
