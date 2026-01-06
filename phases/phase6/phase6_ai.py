from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import httpx

from config import PipelineConfig
from schemas import Record

logger = logging.getLogger("Phase6")


# -----------------------------
# Gemini REST client (Flash)
# -----------------------------

@dataclass(frozen=True)
class GeminiConfig:
    """
    Minimal configuration for Gemini REST calls.

    Notes:
      - generateContent is a POST endpoint.
      - Endpoint:
          POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
      - We implement throttling + retries because 429/503 are common under load.
      - We use deterministic-ish generation settings (temperature=0) to reduce output variance.
    """
    api_key: str
    models: Tuple[str, ...]
    base_url: str = "https://generativelanguage.googleapis.com"
    api_version: str = "v1beta"
    timeout_s: float = 30.0

    # Throttle / retry
    min_interval_s: float = 0.25
    max_retries: int = 6
    backoff_base_s: float = 1.0
    backoff_max_s: float = 20.0
    backoff_jitter_s: float = 0.4

    # Generation knobs (keep short + stable)
    temperature: float = 0.0
    top_p: float = 0.1
    max_output_tokens: int = 24


class _RateLimiter:
    """Ensures at least min_interval_s between requests."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._next_ok = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next_ok:
            time.sleep(self._next_ok - now)
        self._next_ok = time.monotonic() + self.min_interval_s


class GeminiClient:
    """
    Tiny REST client for Gemini models.generateContent.

    Behavior:
      - Tries models in order (fallbacks).
      - Retries 429/503 with exponential backoff + jitter.
      - Treats 404 as "model not available" and moves to next model.
    """

    def __init__(self, cfg: GeminiConfig) -> None:
        self.cfg = cfg
        self._client = httpx.Client(
            timeout=float(cfg.timeout_s),
            http2=True,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": cfg.api_key,
            },
        )
        self._limiter = _RateLimiter(cfg.min_interval_s)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _endpoint(self, model: str) -> str:
        model = (model or "").strip()
        if model.startswith("models/"):
            model = model.split("/", 1)[1]
        return f"{self.cfg.base_url}/{self.cfg.api_version}/models/{model}:generateContent"

    @staticmethod
    def _retry_after_seconds(resp: httpx.Response) -> Optional[float]:
        ra = resp.headers.get("Retry-After")
        if not ra:
            return None
        try:
            return float(ra)
        except Exception:
            return None

    def _sleep_backoff(self, attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None and retry_after > 0:
            time.sleep(min(retry_after, self.cfg.backoff_max_s))
            return
        base = self.cfg.backoff_base_s * (2 ** max(0, attempt))
        base = min(base, self.cfg.backoff_max_s)
        jitter = random.random() * self.cfg.backoff_jitter_s
        time.sleep(base + jitter)

    def generate_text(self, prompt: str) -> Tuple[str, str]:
        """
        Returns (text, used_model).
        """
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(self.cfg.temperature),
                "topP": float(self.cfg.top_p),
                "maxOutputTokens": int(self.cfg.max_output_tokens),
                "candidateCount": 1,
            },
        }
        content = json.dumps(payload).encode("utf-8")

        last_exc: Optional[Exception] = None

        for model in self.cfg.models:
            url = self._endpoint(model)

            for attempt in range(self.cfg.max_retries + 1):
                self._limiter.wait()

                try:
                    resp = self._client.post(url, params={"key": self.cfg.api_key}, content=content)
                except Exception as e:
                    last_exc = e
                    logger.warning("Gemini request failed (model=%s, attempt=%d): %s", model, attempt, e)
                    if attempt < self.cfg.max_retries:
                        self._sleep_backoff(attempt, None)
                        continue
                    break

                if 200 <= resp.status_code < 300:
                    data = resp.json()
                    try:
                        text = (data["candidates"][0]["content"]["parts"][0]["text"]).strip()
                        return text, model
                    except Exception:
                        logger.debug("Unexpected Gemini response shape (model=%s): %s", model, data)
                        return "", model

                if resp.status_code == 404:
                    logger.warning("Gemini model not found/unsupported (model=%s).", model)
                    last_exc = httpx.HTTPStatusError("404 model not found", request=resp.request, response=resp)
                    break

                if resp.status_code in (429, 503):
                    logger.warning("Gemini retryable error (model=%s, status=%s, attempt=%d).", model, resp.status_code, attempt)
                    last_exc = httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                    if attempt < self.cfg.max_retries:
                        self._sleep_backoff(attempt, self._retry_after_seconds(resp))
                        continue
                    break

                logger.error("Gemini non-retryable error (model=%s, status=%s).", model, resp.status_code)
                last_exc = httpx.HTTPStatusError("non-retryable", request=resp.request, response=resp)
                break

        if last_exc:
            raise last_exc
        return "", (self.cfg.models[0] if self.cfg.models else "unknown")


# -----------------------------
# Prompt building + parsing
# -----------------------------

_DOMAIN_RE = re.compile(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)


def _speak_domain(domain: str) -> str:
    d = domain.strip().lower()
    d = d.replace("-", " dash ").replace(".", " dot ")
    d = re.sub(r"\s+", " ", d).strip()
    return d


def _build_prompt(company_name: str, address: Optional[str], candidates: List[str]) -> str:
    candidates_clean = [c.strip().lower() for c in candidates if c and c.strip()]
    address_part = f"Address: {address.strip()}.\n" if address and address.strip() else ""

    cand_lines = [f"- {c} (spoken: {_speak_domain(c)})" for c in candidates_clean]
    cands_block = "\n".join(cand_lines) if cand_lines else "- (none)"

    return (
        "You are helping with company domain ambiguity resolution.\n"
        "Your job is NOT to crawl and NOT to guess emails.\n"
        "You MUST choose ONLY from the provided candidate domains.\n"
        "If none are likely official, answer exactly: not_found\n\n"
        f"Company: {company_name.strip()}\n"
        f"{address_part}"
        "Candidate domains:\n"
        f"{cands_block}\n\n"
        "Return ONLY one of the following:\n"
        "1) One exact candidate domain (copy it exactly)\n"
        "2) not_found\n"
    )


def _extract_choice(text: str, candidates: List[str]) -> str:
    if not text:
        return "not_found"

    raw = text.strip().lower()

    if raw in ("not_found", "not found"):
        return "not_found"

    candidates_norm = [c.strip().lower() for c in candidates if c and c.strip()]

    for c in candidates_norm:
        if raw == c:
            return c

    m = _DOMAIN_RE.search(raw)
    if m:
        dom = m.group(0).lower()
        for c in candidates_norm:
            if dom == c:
                return c

    for c in candidates_norm:
        if c in raw:
            return c

    return "not_found"


# -----------------------------
# Record helpers (exact schema)
# -----------------------------

def _ensure_debug(r: Record) -> Dict[str, object]:
    if r.debug is None or not isinstance(r.debug, dict):
        r.debug = {}
    return r.debug


def _eligible_ambiguous_records(records: List[Record]) -> List[Record]:
    out: List[Record] = []
    for r in records:
        if (r.status or "").startswith("dropped"):
            continue
        if r.resolution.ambiguity_flag is True:
            out.append(r)
    return out


def _chunked(items: List[Record], chunk_size: int) -> Iterable[List[Record]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _apply_ai_quota(all_records: List[Record], ambiguous: List[Record], cfg: PipelineConfig) -> Tuple[List[Record], List[Record]]:
    """
    Enforce cfg.max_ai_fraction (strict):
      allowed_n = int(max_ai_fraction * total_records)
    """
    max_frac = getattr(cfg, "max_ai_fraction", None)
    if max_frac is None:
        return ambiguous, []

    try:
        max_frac_f = float(max_frac)
    except Exception:
        return ambiguous, []

    if max_frac_f <= 0:
        return [], ambiguous

    allowed_n = int(max_frac_f * len(all_records))
    if allowed_n <= 0:
        return [], ambiguous

    allowed = ambiguous[:allowed_n]
    skipped = ambiguous[allowed_n:]
    return allowed, skipped


def _get_gemini_api_key(cfg: PipelineConfig) -> Optional[str]:
    k = getattr(cfg, "gemini_api_key", None)
    if k:
        return str(k).strip() or None
    return os.getenv("GEMINI_API_KEY", "").strip() or None


def _get_model_list(cfg: PipelineConfig) -> Tuple[str, ...]:
    models = getattr(cfg, "gemini_models", None)
    if models:
        cleaned = tuple(str(m).strip() for m in models if str(m).strip())
        if cleaned:
            return cleaned

    single = (getattr(cfg, "gemini_model", None) or os.getenv("GEMINI_MODEL", "") or "gemini-2.0-flash").strip()
    return (single,)


def _pick_ai_candidates(r: Record) -> List[str]:
    """
    Candidate priority (matches your spec):
      1) valid_domains
      2) domain_candidates
      3) []
    """
    if r.valid_domains:
        return [c.strip().lower() for c in r.valid_domains if isinstance(c, str) and c.strip()]
    if r.domain_candidates:
        return [c.strip().lower() for c in r.domain_candidates if isinstance(c, str) and c.strip()]
    return []


def _company_label(r: Record) -> str:
    """
    Uses Phase 1 outputs first (more stable than raw).
    """
    if r.normalized.strict_name.strip():
        return r.normalized.strict_name.strip()
    if r.normalized.normalized_raw.strip():
        return r.normalized.normalized_raw.strip()
    if r.raw_name.strip():
        return r.raw_name.strip()
    return r.company_id


def _import_phase5_runner() -> Optional[callable]:
    """
    Try common Phase 5 module paths. Must expose run_phase(records, cfg).
    """
    module_candidates = (
        "phases.phase5.phase5_crawl",
        "phases.phase5.phase5",
        "phases.phase5",
        "phase5",
    )
    for mod_name in module_candidates:
        try:
            mod = __import__(mod_name, fromlist=["run_phase"])
            fn = getattr(mod, "run_phase", None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None


# -----------------------------
# Phase runner
# -----------------------------

def run_phase(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 6: AI-assisted Ambiguity Resolution.

    Entry condition:
      - record.resolution.ambiguity_flag == True

    Behavior:
      - Builds constrained prompt (company name + optional address + candidates)
      - Gemini picks one candidate domain OR not_found
      - If domain picked: set resolved_domain, clear ambiguity, set ai_used=True, source="ai"
        then re-run Phase 5 on that subset only
      - If not_found: clear ambiguity, set status="not_found", set ai_used=True, source="ai"
      - If AI fails: keep record ambiguous (retryable)
      - If skipped due to quota: keep ambiguous unchanged
    """
    ambiguous = _eligible_ambiguous_records(records)
    if not ambiguous:
        logger.info("No ambiguous records to resolve in Phase 6.")
        return records

    allowed, skipped = _apply_ai_quota(records, ambiguous, cfg)

    # Skipped due to quota (no state change besides debug)
    for r in skipped:
        dbg = _ensure_debug(r)
        dbg["phase6_reason"] = "skipped_ai_quota"
        dbg["phase6_decision"] = "skipped"
        dbg["phase6_ai_allowed"] = False

    if not allowed:
        logger.info("Phase 6 quota allows 0 records (ambiguous=%d).", len(ambiguous))
        return records

    api_key = _get_gemini_api_key(cfg)
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; Phase 6 cannot run.")
        for r in allowed:
            dbg = _ensure_debug(r)
            dbg["phase6_reason"] = "missing_api_key"
            dbg["phase6_decision"] = "skipped"
            dbg["phase6_ai_allowed"] = True
        return records

    models = _get_model_list(cfg)
    batch_size = int(getattr(cfg, "gemini_batch_size", 50) or 50)

    # Status policy (domain resolution != contact success)
    set_status = bool(getattr(cfg, "phase6_set_status", True))
    st_resolved = str(getattr(cfg, "phase6_status_resolved", "resolved"))
    st_not_found = str(getattr(cfg, "phase6_status_not_found", "not_found"))

    gem_cfg = GeminiConfig(
        api_key=api_key,
        models=models,
        timeout_s=float(getattr(cfg, "gemini_timeout_s", 30.0) or 30.0),
        min_interval_s=float(getattr(cfg, "gemini_min_interval_s", 0.25) or 0.25),
        max_retries=int(getattr(cfg, "gemini_max_retries", 6) or 6),
        backoff_base_s=float(getattr(cfg, "gemini_backoff_base_s", 1.0) or 1.0),
        backoff_max_s=float(getattr(cfg, "gemini_backoff_max_s", 20.0) or 20.0),
        backoff_jitter_s=float(getattr(cfg, "gemini_backoff_jitter_s", 0.4) or 0.4),
        temperature=float(getattr(cfg, "gemini_temperature", 0.0) or 0.0),
        top_p=float(getattr(cfg, "gemini_top_p", 0.1) or 0.1),
        max_output_tokens=int(getattr(cfg, "gemini_max_output_tokens", 24) or 24),
    )

    phase5_runner = _import_phase5_runner()
    if phase5_runner is None:
        logger.warning("Phase 6 cannot import Phase 5 run_phase() for recrawl. Will not recrawl inside Phase 6.")

    client = GeminiClient(gem_cfg)
    try:
        resolved_for_recrawl: List[Record] = []

        for batch in _chunked(allowed, batch_size):
            for r in batch:
                dbg = _ensure_debug(r)
                dbg["phase6_ai_allowed"] = True
                dbg["phase6_reason"] = "unset"
                dbg["phase6_decision"] = "unset"
                dbg["phase6_used_model"] = None

                candidates = _pick_ai_candidates(r)
                dbg["phase6_candidate_count"] = len(candidates)
                dbg["phase6_candidates_sample"] = candidates[:10]

                # Store prompt/response for audit (schema supports it)
                company = _company_label(r)
                address = r.address
                prompt = _build_prompt(company, address, candidates)

                r.resolution.ai_prompt = prompt
                r.resolution.ai_response = None

                if not candidates:
                    # Spec: no candidates -> not_found
                    r.ai_used = True
                    r.source = "ai"
                    r.resolution.resolved_domain = None
                    r.resolution.ambiguity_flag = False
                    r.resolution.reason = "ai_not_found_no_candidates"
                    if set_status:
                        r.status = st_not_found
                    dbg["phase6_reason"] = "no_candidates"
                    dbg["phase6_decision"] = "not_found"
                    continue

                try:
                    answer, used_model = client.generate_text(prompt)
                    dbg["phase6_used_model"] = used_model
                    r.resolution.ai_response = answer
                except Exception as e:
                    # Retryable: keep ambiguous
                    logger.warning("Gemini call failed for company_id=%s: %s", r.company_id, e)
                    dbg["phase6_reason"] = "gemini_call_failed"
                    dbg["phase6_error"] = str(e)
                    continue

                choice = _extract_choice(answer, candidates)

                if choice == "not_found":
                    r.ai_used = True
                    r.source = "ai"
                    r.resolution.resolved_domain = None
                    r.resolution.ambiguity_flag = False
                    r.resolution.reason = "ai_not_found"
                    if set_status:
                        r.status = st_not_found
                    dbg["phase6_reason"] = "ai_not_found"
                    dbg["phase6_decision"] = "not_found"
                    continue

                # AI selected a domain candidate
                r.ai_used = True
                r.source = "ai"  # source = how official website was resolved
                r.resolution.resolved_domain = choice
                r.resolution.ambiguity_flag = False
                r.resolution.reason = "ai_selected_domain"
                if set_status and r.status == "ambiguous":
                    r.status = st_resolved

                dbg["phase6_reason"] = "ai_selected_domain"
                dbg["phase6_decision"] = choice

                resolved_for_recrawl.append(r)

        # Go back to Phase 5 on resolved subset
        if resolved_for_recrawl and phase5_runner is not None:
            logger.info("Phase 6 resolved %d records; re-running Phase 5 on them.", len(resolved_for_recrawl))
            updated_subset = phase5_runner(resolved_for_recrawl, cfg)

            by_id = {x.company_id: x for x in updated_subset}

            merged: List[Record] = []
            for r in records:
                merged.append(by_id.get(r.company_id, r))
            return merged

        return records

    finally:
        client.close()
