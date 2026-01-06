from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Set, Tuple

from config import PipelineConfig
from schemas import Record

logger = logging.getLogger("Phase 1")

# -----------------------------
# Minimal safe fallbacks
# (used only if phase assets are missing)
# -----------------------------
FALLBACK_LEGAL_SUFFIXES: Set[str] = {
    "ltd", "limited",
    "llc", "l l c",
    "inc", "incorporated",
    "corp", "corporation",
    "plc",
    "lp", "llp",
    "gmbh",
    "sarl", "s a r l",
    "bv",
    "oy",
    "kk",
    "spa", "s p a",
    "sa", "s a",
    "ag",
    "ab",
    "as",
    "nv",
    "pte",
    "pte ltd",
    "kg", "kgaa",
}

FALLBACK_STOPWORDS_STRICT: Set[str] = {
    "group",
    "holding",
    "holdings",
    "international",
    "global",
    "services",
    "service",
    "solutions",
    "industries",
    "industry",
    "enterprise",
    "enterprises",
    "systems",
    "system",
}

# -----------------------------
# Task 1.1 helpers
# -----------------------------
def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _strip_diacritics(s: str) -> str:
    """
    Deterministically strip diacritics (accents) to reduce false drops.

    Example:
      "Müller" -> "Muller"
      "Crème"  -> "Creme"
    """
    if not s:
        return ""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _normalize_raw_name(raw: str) -> str:
    """
    Task 1.1 Normalize Raw Name

    Output:
      A string containing only: [a-z0-9], spaces, and single dashes.
    """
    s = unicodedata.normalize("NFKC", raw or "").strip()
    s = _strip_diacritics(s)
    s = s.lower()

    s = s.lstrip("&+@#*~^|!").strip()

    s = re.sub(r"\s*&\s*", " and ", s)
    s = re.sub(r"[$€£¥]+", " ", s)
    s = re.sub(r"[•·]+", " ", s)

    s = re.sub(r"^[^a-z0-9]+", "", s)

    # Keep: a-z, 0-9, space, dash
    s = re.sub(r"[^a-z0-9\s-]", " ", s)

    s = _collapse_ws(s)
    s = re.sub(r"-{2,}", "-", s).strip("-").strip()

    return s


def _has_any_letter(s: str) -> bool:
    return bool(re.search(r"[a-z]", s or ""))


# -----------------------------
# Phase 1 assets
# -----------------------------
def _read_word_list(path: Path) -> Set[str]:
    """
    Read a simple word/phrase list from assets.

    Format:
      - one entry per line
      - ignore empty lines
      - ignore comments starting with '#'
      - returned entries are lowercase (phrases preserved)
    """
    out: Set[str] = set()
    if not path.exists():
        return out
    try:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = (raw or "").strip()
            if not s or s.startswith("#"):
                continue
            out.add(s.lower())
    except Exception:
        return set()
    return out


def _load_phase1_lists(cfg: PipelineConfig) -> Tuple[Set[str], Set[str]]:
    legal_path = cfg.phase_asset_path(1, "legal_suffixes.txt")
    stop_path = cfg.phase_asset_path(1, "stopwords_strict.txt")

    legal_suffixes = _read_word_list(legal_path)
    if not legal_suffixes:
        logger.warning("Phase 1 legal suffix list missing/empty: %s (using fallback)", legal_path)
        legal_suffixes = set(FALLBACK_LEGAL_SUFFIXES)

    stopwords = _read_word_list(stop_path)
    if not stopwords:
        logger.warning("Phase 1 stopwords list missing/empty: %s (using fallback)", stop_path)
        stopwords = set(FALLBACK_STOPWORDS_STRICT)

    legal_suffixes.discard("")
    stopwords.discard("")
    return legal_suffixes, stopwords


# -----------------------------
# Task 1.2 helpers
# -----------------------------
def _tokenize_spaces_and_dashes(s: str) -> List[str]:
    if not s:
        return []
    toks = re.split(r"[\s-]+", s)
    return [t for t in toks if t]


def _strip_legal_suffixes_end(tokens: List[str], legal_suffixes: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Remove legal suffix tokens only from the END, possibly multiple suffixes.
    Supports multi-token suffixes such as:
      - "pte ltd"
      - "l l c"
      - "s a r l"
    """
    if not tokens:
        return [], []

    base = tokens[:]
    removed: List[str] = []

    def tokens_to_phrase(toks: List[str]) -> str:
        return " ".join(toks).strip()

    while base:
        matched = False

        if len(base) >= 3:
            tri = tokens_to_phrase(base[-3:])
            if tri in legal_suffixes:
                removed = base[-3:] + removed
                base = base[:-3]
                matched = True
        if matched:
            continue

        if len(base) >= 2:
            duo = tokens_to_phrase(base[-2:])
            if duo in legal_suffixes:
                removed = base[-2:] + removed
                base = base[:-2]
                matched = True
        if matched:
            continue

        last = base[-1]
        if last in legal_suffixes:
            removed = [last] + removed
            base = base[:-1]
            matched = True

        if not matched:
            break

    return (base, removed) if base else ([], removed)


# -----------------------------
# Task 1.3 helpers
# -----------------------------
def _remove_stopwords_strict(tokens: List[str], stopwords: Set[str]) -> List[str]:
    return [t for t in tokens if t and t not in stopwords]


# -----------------------------
# Phase runner
# -----------------------------
def run_phase(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 1: Name Normalization (deterministic + retryable)

    Notes on retryability:
      - We do NOT permanently skip records dropped by Phase 1 itself.
      - Only records dropped in Phase 0 ("dropped_dummy") are skipped (and normally won't appear anyway).
    """
    legal_suffixes, stopwords = _load_phase1_lists(cfg)

    for r in records:
        status = (r.status or "")

        # Always reset outputs (prevents stale data on reruns)
        r.normalized.normalized_raw = ""
        r.normalized.strict_name = ""
        r.normalized.relaxed_name = ""
        r.normalized.tokens = []

        r.debug["phase1_reason"] = "unset"
        r.debug["phase1_raw_tokens"] = []
        r.debug["phase1_removed_suffix"] = []
        r.debug["phase1_strict_tokens_before_stopwords"] = []

        # Skip only Phase-0 dropped records (they should not be present in the stream anyway)
        if status == "dropped_dummy":
            r.debug["phase1_reason"] = f"skipped_status:{status}"
            continue

        # If Phase 1 previously dropped this record, allow reprocessing (retryable).
        if status.startswith("dropped"):
            r.status = "new"

        normalized_raw = _normalize_raw_name(r.raw_name or "")
        r.normalized.normalized_raw = normalized_raw

        if not normalized_raw or not _has_any_letter(normalized_raw):
            r.status = "dropped_empty"
            r.debug["phase1_reason"] = "no_letters_after_normalization"
            continue

        raw_tokens = _tokenize_spaces_and_dashes(normalized_raw)
        r.debug["phase1_raw_tokens"] = raw_tokens

        base_tokens, removed_suffix_tokens = _strip_legal_suffixes_end(raw_tokens, legal_suffixes)
        r.debug["phase1_removed_suffix"] = removed_suffix_tokens

        if not base_tokens:
            r.status = "dropped_empty"
            r.debug["phase1_reason"] = "only_legal_suffixes"
            continue

        r.debug["phase1_strict_tokens_before_stopwords"] = base_tokens[:]
        strict_tokens = _remove_stopwords_strict(base_tokens, stopwords)

        if not strict_tokens:
            r.status = "dropped_empty"
            r.debug["phase1_reason"] = "only_stopwords_after_removal"
            continue

        strict_name = " ".join(strict_tokens).strip()
        if not strict_name or not _has_any_letter(strict_name):
            r.status = "dropped_empty"
            r.debug["phase1_reason"] = "empty_after_suffix_or_stopwords"
            continue

        r.normalized.strict_name = strict_name
        r.normalized.tokens = strict_tokens
        r.normalized.relaxed_name = strict_tokens[0]

        r.debug["phase1_reason"] = "ok"

    logger.info("Phase 1 complete: processed %d records", len(records))
    return records
