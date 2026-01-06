from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config import PipelineConfig
from schemas import Record

logger = logging.getLogger("Phase 2")

# -----------------------------
# Minimal safe fallbacks
# (used only if phase assets are missing)
# -----------------------------
FALLBACK_COUNTRY_TO_CCTLD: Dict[str, str] = {
    "united kingdom": "uk",
    "uk": "uk",
    "great britain": "uk",
    "england": "uk",
    "scotland": "uk",
    "wales": "uk",
    "northern ireland": "uk",
    "gb": "uk",
    "germany": "de",
    "deutschland": "de",
    "france": "fr",
    "italy": "it",
    "spain": "es",
    "netherlands": "nl",
    "belgium": "be",
    "switzerland": "ch",
    "austria": "at",
    "sweden": "se",
    "norway": "no",
    "denmark": "dk",
    "finland": "fi",
    "ireland": "ie",
    "poland": "pl",
    "portugal": "pt",
    "czech republic": "cz",
    "czechia": "cz",
    "romania": "ro",
    "greece": "gr",
    "hungary": "hu",
    "united states": "us",
    "usa": "us",
    "us": "us",
    "canada": "ca",
    "mexico": "mx",
    "brazil": "br",
    "united arab emirates": "ae",
    "uae": "ae",
    "saudi arabia": "sa",
    "qatar": "qa",
    "israel": "il",
    "turkey": "tr",
    "iran": "ir",
    "india": "in",
    "china": "cn",
    "japan": "jp",
    "south korea": "kr",
    "korea": "kr",
    "singapore": "sg",
    "hong kong": "hk",
    "taiwan": "tw",
    "australia": "au",
    "new zealand": "nz",
}

FALLBACK_GLOBAL_TLDS: List[str] = ["com", "net", "org", "biz"]

FALLBACK_UK_PUBLIC_SUFFIXES: List[str] = [
    "co.uk",
    "org.uk",
    "ac.uk",
    "gov.uk",
    "ltd.uk",
    "plc.uk",
    "me.uk",
]

FALLBACK_BAD_STEMS: Set[str] = {
    "a", "an", "and", "the", "of", "for", "to", "in", "on", "at", "by", "with", "from",
    "company", "co", "group", "services", "service", "solutions", "global", "international",
    "holding", "holdings", "enterprise", "industries", "industry", "systems", "system",
    "support", "consulting", "advisory", "studio", "labs", "lab",
}

FALLBACK_PATTERN_EXTRAS: List[str] = ["company", "group", "hq", "services"]


# -----------------------------
# Small normalization utilities
# -----------------------------
def _strip_diacritics(s: str) -> str:
    """
    Remove diacritics deterministically (ASCII-ish).
    Example: "Müller" -> "Muller".
    """
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -----------------------------
# Candidate validation (prevents Phase 3 from sanitizing everything away)
# -----------------------------
_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def _is_valid_domain_candidate(domain: str) -> bool:
    """
    Conservative DNS hostname validation:
      - lowercase a-z0-9.- only
      - total length <= 253
      - labels 1..63, no leading/trailing '-'
      - must contain at least one dot
      - no empty labels ("..")
    """
    d = (domain or "").strip().lower()
    if not d or "." not in d:
        return False
    if len(d) > 253:
        return False
    if not re.fullmatch(r"[a-z0-9.-]+", d):
        return False
    if d.startswith(".") or d.endswith(".") or ".." in d:
        return False

    labels = d.split(".")
    if any(not lab for lab in labels):
        return False
    for lab in labels:
        if len(lab) > 63:
            return False
        if not _LABEL_RE.fullmatch(lab):
            return False
    return True


# -----------------------------
# Asset readers
# -----------------------------
def _read_asset_lines(path: Path) -> List[str]:
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


def _read_json_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(obj, dict):
            return {}
        out: Dict[str, str] = {}
        for k, v in obj.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            kk = k.strip().lower()
            vv = v.strip().lower().lstrip(".")
            if len(vv) == 2 and vv.isalpha():
                out[kk] = vv
        return out
    except Exception:
        return {}


def _clean_tld_token(t: str) -> Optional[str]:
    t = (t or "").strip().lower().lstrip(".")
    if not t:
        return None
    if not re.fullmatch(r"[a-z0-9]+(\.[a-z0-9]+)*", t):
        return None
    return t


def _load_phase2_assets(cfg: PipelineConfig) -> Tuple[Dict[str, str], List[str], List[str], Set[str], List[str], int]:
    phase = 2
    country_map_path = cfg.phase_asset_path(phase, "country_to_cctld.json")
    global_tlds_path = cfg.phase_asset_path(phase, "global_tlds.txt")
    uk_suffixes_path = cfg.phase_asset_path(phase, "uk_public_suffixes.txt")
    bad_stems_path = cfg.phase_asset_path(phase, "bad_stems.txt")
    extras_path = cfg.phase_asset_path(phase, "pattern_extras.txt")

    country_map = _read_json_map(country_map_path) or {}

    global_lines = _read_asset_lines(global_tlds_path)
    global_tlds = [_clean_tld_token(s) for s in global_lines]
    global_tlds = [t for t in global_tlds if t] or [_clean_tld_token(t) for t in FALLBACK_GLOBAL_TLDS if _clean_tld_token(t)]

    uk_lines = _read_asset_lines(uk_suffixes_path)
    uk_suffixes = [_clean_tld_token(s) for s in uk_lines]
    uk_suffixes = [t for t in uk_suffixes if t] or [_clean_tld_token(t) for t in FALLBACK_UK_PUBLIC_SUFFIXES if _clean_tld_token(t)]

    bad_lines = _read_asset_lines(bad_stems_path)
    bad_stems = {s.strip().lower() for s in bad_lines if s.strip()} or set(FALLBACK_BAD_STEMS)
    bad_stems.discard("")

    extras_lines = _read_asset_lines(extras_path)
    pattern_extras = [s.strip().lower() for s in extras_lines if s.strip()] or list(FALLBACK_PATTERN_EXTRAS)

    min_stem_len = int(getattr(cfg, "phase2_min_stem_len", 3))

    # Defensive: remove empty + keep deterministic order
    global_tlds = _unique_preserve_order([t for t in global_tlds if t])
    uk_suffixes = _unique_preserve_order([t for t in uk_suffixes if t])
    pattern_extras = _unique_preserve_order([e for e in pattern_extras if e])

    return country_map, global_tlds, uk_suffixes, bad_stems, pattern_extras, min_stem_len


# -----------------------------
# Task 2.1 helpers (TLD selection)
# -----------------------------
def _clean_country_value(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    s = unicodedata.normalize("NFKC", str(country)).strip()
    s = _strip_diacritics(s).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _infer_cctld(country: Optional[str], country_map: Dict[str, str]) -> Optional[str]:
    if not country:
        return None

    raw = str(country).strip()
    if len(raw) == 2 and raw.isalpha():
        return raw.lower()

    c = _clean_country_value(raw)
    if not c:
        return None

    if c in country_map:
        return country_map[c]
    if c in FALLBACK_COUNTRY_TO_CCTLD:
        return FALLBACK_COUNTRY_TO_CCTLD[c]

    def try_partial(m: Dict[str, str]) -> Optional[str]:
        for k in sorted(m.keys(), key=len, reverse=True):
            kk = (k or "").strip().lower()
            if len(kk) < 4:
                continue
            if " " in kk and kk in c:
                return m[k]
            if " " not in kk and re.search(rf"\b{re.escape(kk)}\b", c):
                return m[k]
        return None

    hit = try_partial(country_map)
    if hit:
        return hit
    hit = try_partial(FALLBACK_COUNTRY_TO_CCTLD)
    if hit:
        return hit

    return None


def _select_tlds(
        record_country: Optional[str],
        country_map: Dict[str, str],
        global_tlds: List[str],
        uk_public_suffixes: List[str],
) -> Tuple[List[str], Optional[str]]:
    cc = _infer_cctld(record_country, country_map) if record_country else None
    globals_ = list(global_tlds)

    if not cc:
        return globals_, None

    if cc == "uk":
        globals_primary = [t for t in ("com", "net", "org", "biz") if t in globals_]
        globals_rest = [t for t in globals_ if t not in globals_primary and t != "uk"]

        tlds = _unique_preserve_order(
            uk_public_suffixes + globals_primary + ["uk"] + globals_rest
        )
        return tlds, cc

    tlds = [cc] + [t for t in globals_ if t != cc]
    return tlds, cc


# -----------------------------
# Task 2.2 helpers (stems + patterns)
# -----------------------------
def _slugify_domain_stem(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").strip()
    s = _strip_diacritics(s).lower()
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")

    # Avoid generating labels that Phase 3 will discard.
    # We still validate full domains later; this just prevents obvious overflows.
    if len(s) > 63:
        s = s[:63].strip("-")
    return s


def _tokenize_fallback_text(text: str) -> List[str]:
    base = unicodedata.normalize("NFKC", (text or "")).strip()
    base = _strip_diacritics(base).lower()
    base = re.sub(r"[^a-z0-9\s-]", " ", base)
    return [t for t in re.split(r"[\s-]+", base) if t]


def _build_stems(
        strict_tokens: List[str],
        relaxed_name: str,
        fallback_text: str,
        bad_stems: Set[str],
        min_stem_len: int,
) -> Dict[str, List[str]]:
    """
    Build candidate stems from Phase 1 name signals.

    Improvement:
      - allow 2-char real-world stems (BP, GM, 3M) if they contain at least one letter.
    """

    def good_token(t: str) -> bool:
        if not t:
            return False
        if t in bad_stems:
            return False
        if t.isdigit():
            return False

        # normal rule
        if len(t) >= min_stem_len:
            return True

        # exception: allow 2-char stems if they look real (has at least one letter)
        if len(t) == 2 and re.search(r"[a-z]", t):
            return True

        return False

    toks = [t for t in (strict_tokens or []) if t]
    good_toks = [t for t in toks if good_token(t)]

    # Acronym join: ["h","s","b","c"] -> "hsbc"
    if not good_toks and toks and all(len(t) == 1 for t in toks) and len(toks) >= 2:
        joined = _slugify_domain_stem("".join(toks))
        if joined and good_token(joined):
            good_toks = [joined]

    if not good_toks and fallback_text:
        fb = _tokenize_fallback_text(fallback_text)
        good_toks = [t for t in fb if good_token(t)]

    stems_strict: List[str] = []
    stems_relaxed: List[str] = []

    def good_stem(s: str) -> bool:
        if not s:
            return False
        if s.isdigit():
            return False
        if s in bad_stems:
            return False
        if len(s) >= min_stem_len:
            return True
        if len(s) == 2 and re.search(r"[a-z]", s):
            return True
        return False

    if good_toks:
        first = _slugify_domain_stem(good_toks[0])
        compact = _slugify_domain_stem("".join(good_toks))
        dashed = _slugify_domain_stem("-".join(good_toks))

        for v in (first, compact, dashed):
            if good_stem(v) and v not in stems_strict:
                stems_strict.append(v)

        if len(good_toks) >= 2:
            first2_compact = _slugify_domain_stem(good_toks[0] + good_toks[1])
            first2_dash = _slugify_domain_stem(good_toks[0] + "-" + good_toks[1])
            for v in (first2_compact, first2_dash):
                if good_stem(v) and v not in stems_strict:
                    stems_strict.append(v)

    rel = _slugify_domain_stem(relaxed_name or "")
    if rel and good_stem(rel):
        stems_relaxed.append(rel)

    return {
        "strict": _unique_preserve_order(stems_strict),
        "relaxed": _unique_preserve_order(stems_relaxed),
    }


def _expand_patterns_bounded(
        stems: List[str],
        tlds: List[str],
        extras: List[str],
        limit: int,
) -> List[str]:
    out: List[str] = []
    extras = [e for e in extras if e]

    for stem in stems:
        if not stem:
            continue
        for tld in tlds:
            out.append(f"{stem}.{tld}")
            if len(out) >= limit:
                return out

            for e in extras:
                out.append(f"{stem}{e}.{tld}")
                if len(out) >= limit:
                    return out

            for e in extras:
                out.append(f"{stem}-{e}.{tld}")
                if len(out) >= limit:
                    return out

    return out


# -----------------------------
# Task 2.3 helpers (tiered ordering)
# -----------------------------
def _split_candidate_by_known_tlds(cand: str, known_tlds: List[str]) -> Tuple[Optional[str], Optional[str]]:
    c = (cand or "").strip().lower()
    if not c or "." not in c:
        return None, None

    for tld in sorted(known_tlds, key=len, reverse=True):
        suffix = "." + tld
        if c.endswith(suffix) and len(c) > len(suffix):
            stem = c[: -len(suffix)]
            if stem.endswith("."):
                stem = stem[:-1]
            if stem:
                return stem, tld

    stem, tld = c.rsplit(".", 1)
    return (stem or None), (tld or None)


def _candidate_order_key(
        cand: str,
        cc_tld: Optional[str],
        strict_stems_ordered: List[str],
        relaxed_stems_ordered: List[str],
        known_tlds: List[str],
        uk_public_suffixes: List[str],
        pattern_extras: List[str],
        global_tlds: List[str],
) -> Tuple[int, int, int, int, int, int, str]:
    stem, tld = _split_candidate_by_known_tlds(cand, known_tlds)
    if not stem or not tld:
        return (999, 99, 99, 999, 9999, 9999, cand)

    strict_set = set(strict_stems_ordered)
    relaxed_set = set(relaxed_stems_ordered)

    base = None
    pattern_rank = 0
    extra_rank = 999

    def _extra_index(ex: str) -> int:
        try:
            return pattern_extras.index(ex)
        except Exception:
            return 999

    if stem in strict_set or stem in relaxed_set:
        base = stem
        pattern_rank = 0
        extra_rank = 999
    else:
        for e in pattern_extras:
            e = (e or "").strip().lower()
            if not e:
                continue
            suffix = f"-{e}"
            if stem.endswith(suffix):
                maybe = stem[: -len(suffix)]
                if maybe in strict_set or maybe in relaxed_set:
                    base = maybe
                    pattern_rank = 2
                    extra_rank = _extra_index(e)
                    break

        if base is None:
            for e in pattern_extras:
                e = (e or "").strip().lower()
                if not e:
                    continue
                if stem.endswith(e) and len(stem) > len(e):
                    maybe = stem[: -len(e)]
                    if maybe in strict_set or maybe in relaxed_set:
                        base = maybe
                        pattern_rank = 1
                        extra_rank = _extra_index(e)
                        break

    if base in strict_set:
        stem_rank = strict_stems_ordered.index(base)
    elif base in relaxed_set:
        stem_rank = 100 + relaxed_stems_ordered.index(base)
    else:
        stem_rank = 999

    is_cc = False
    if cc_tld:
        if cc_tld == "uk":
            if tld == "uk" or tld in uk_public_suffixes or tld.endswith(".uk"):
                is_cc = True
        else:
            if tld == cc_tld or tld.endswith(f".{cc_tld}"):
                is_cc = True

    global_set = set(global_tlds)

    if is_cc:
        tld_rank = 0
    elif tld == "com":
        tld_rank = 1
    elif tld in global_set:
        tld_rank = 2
    else:
        tld_rank = 3

    stem_len = len(stem)
    dash_count = stem.count("-")

    return (stem_rank, tld_rank, pattern_rank, extra_rank, stem_len, dash_count, cand)


# -----------------------------
# Phase runner
# -----------------------------
def run_phase(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    (
        country_map,
        global_tlds,
        uk_public_suffixes,
        bad_stems,
        pattern_extras,
        min_stem_len,
    ) = _load_phase2_assets(cfg)

    cap = max(1, int(getattr(cfg, "max_domain_candidates", 15)))
    superset_cap = max(cap * 8, cap)

    for r in records:
        status = (r.status or "")

        r.domain_candidates = []
        r.debug["phase2_reason"] = "unset"
        r.debug["phase2_selected_tlds"] = []
        r.debug["phase2_cc_tld"] = None
        r.debug["phase2_strict_stems"] = []
        r.debug["phase2_relaxed_stems"] = []
        r.debug["phase2_candidates_generated"] = 0
        r.debug["phase2_candidates_kept_after_validation"] = 0

        if status.startswith("dropped"):
            r.debug["phase2_reason"] = f"skipped_status:{status}"
            continue

        strict_tokens = list(r.normalized.tokens or [])
        relaxed_name = r.normalized.relaxed_name or ""
        normalized_raw = r.normalized.normalized_raw or (r.raw_name or "")

        if not strict_tokens:
            strict_tokens = _tokenize_fallback_text(normalized_raw)

        tlds, cc_tld = _select_tlds(r.country, country_map, global_tlds, uk_public_suffixes)
        r.debug["phase2_selected_tlds"] = tlds
        r.debug["phase2_cc_tld"] = cc_tld

        stems = _build_stems(
            strict_tokens=strict_tokens,
            relaxed_name=relaxed_name,
            fallback_text=normalized_raw,
            bad_stems=bad_stems,
            min_stem_len=min_stem_len,
        )
        strict_stems = stems["strict"]
        relaxed_stems = stems["relaxed"]

        r.debug["phase2_strict_stems"] = strict_stems
        r.debug["phase2_relaxed_stems"] = relaxed_stems

        if not strict_stems and not relaxed_stems:
            r.debug["phase2_reason"] = "no_valid_stems_after_filtering"
            continue

        candidates: List[str] = []
        candidates += _expand_patterns_bounded(strict_stems, tlds, pattern_extras, limit=superset_cap)

        if len(candidates) < superset_cap and relaxed_stems:
            candidates += _expand_patterns_bounded(
                relaxed_stems, tlds, pattern_extras, limit=(superset_cap - len(candidates))
            )

        candidates = _unique_preserve_order([c for c in candidates if c])
        r.debug["phase2_candidates_generated"] = len(candidates)

        # Validate candidates here so Phase 3 doesn't "sanitize everything away"
        candidates = [c for c in candidates if _is_valid_domain_candidate(c)]
        r.debug["phase2_candidates_kept_after_validation"] = len(candidates)

        if not candidates:
            r.debug["phase2_reason"] = "no_valid_candidates_after_validation"
            continue

        known_tlds = _unique_preserve_order(tlds + global_tlds + uk_public_suffixes)

        ordered = sorted(
            candidates,
            key=lambda c: _candidate_order_key(
                c,
                cc_tld=cc_tld,
                strict_stems_ordered=strict_stems,
                relaxed_stems_ordered=relaxed_stems,
                known_tlds=known_tlds,
                uk_public_suffixes=uk_public_suffixes,
                pattern_extras=pattern_extras,
                global_tlds=global_tlds,
            ),
        )

        r.domain_candidates = ordered[:cap]
        r.debug["phase2_reason"] = "ok"

    logger.info("Phase 2 complete: generated candidates for %d records", len(records))
    return records
