from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import PipelineConfig
from schemas import Contacts, Record

logger = logging.getLogger("Phase8")

_LAT_RE = re.compile(r"^phase(\d+)_latency_s$")


def _safe_get(obj: Any, dotted: str, default=None):
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return default
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _jsonl_dump(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")


def _normalize_email(e: str) -> str:
    e = (e or "").strip().lower()
    e = e.strip(" ,.;:()[]<>\"'")
    return e


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _extract_emails_from_contacts_obj(obj: Any) -> List[str]:
    """
    Schema-first: Contacts.emails is List[str].
    Also supports legacy list-of-dict formats if present.
    """
    if obj is None:
        return []

    # Schema path: Contacts object
    if isinstance(obj, Contacts):
        return [_normalize_email(e) for e in (obj.emails or []) if isinstance(e, str) and e.strip()]

    # Pydantic model or object with .emails
    emails_attr = getattr(obj, "emails", None)
    if isinstance(emails_attr, list):
        return [_normalize_email(e) for e in emails_attr if isinstance(e, str) and e.strip()]

    # Legacy: list of dicts
    if isinstance(obj, list):
        out: List[str] = []
        for item in obj:
            if isinstance(item, dict):
                v = item.get("email")
                if isinstance(v, str) and v.strip():
                    out.append(_normalize_email(v))
            elif isinstance(item, str) and "@" in item:
                out.append(_normalize_email(item))
        return out

    return []


def _extract_emails(record: Record, cfg: PipelineConfig) -> List[str]:
    """
    Prefer Phase 7 validated emails if enabled; otherwise fallback to Phase 5 extracted emails.
    """
    prefer_validated = bool(getattr(cfg, "phase8_prefer_validated_contacts", True))

    validated_emails = _extract_emails_from_contacts_obj(getattr(record, "validated_contacts", None))
    validated_emails = _dedup_preserve_order([e for e in validated_emails if e])

    if validated_emails and prefer_validated:
        return validated_emails

    contacts_emails = _extract_emails_from_contacts_obj(getattr(record, "contacts", None))
    contacts_emails = _dedup_preserve_order([e for e in contacts_emails if e])

    # If prefer_validated=False and validated exists, combine (validated first, stable)
    if validated_emails and not prefer_validated:
        combined = _dedup_preserve_order(validated_emails + contacts_emails)
        return combined

    return contacts_emails


def _extract_phone(record: Record) -> Optional[str]:
    """
    Schema-first:
      - validated_contacts.phone
      - contacts.phone
    """
    vc = getattr(record, "validated_contacts", None)
    if vc is not None:
        p = getattr(vc, "phone", None)
        if isinstance(p, str) and p.strip():
            return p.strip()

    c = getattr(record, "contacts", None)
    if c is not None:
        p = getattr(c, "phone", None)
        if isinstance(p, str) and p.strip():
            return p.strip()

    # Legacy fallbacks (keep robust)
    for field in ("phone", "phones", "extracted_phone", "extracted_phones", "contact_phone", "contact_phones"):
        v = getattr(record, field, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            for x in v:
                if isinstance(x, str) and x.strip():
                    return x.strip()

    return None


def _extract_website(record: Record) -> Optional[str]:
    # Schema path
    v = _safe_get(record, "resolution.resolved_domain", None)
    if isinstance(v, str) and v.strip():
        return v.strip().lower()

    # Legacy paths
    for dotted in ("resolution.selected_domain", "resolution.chosen_domain"):
        vv = _safe_get(record, dotted, None)
        if isinstance(vv, str) and vv.strip():
            return vv.strip().lower()

    for field in ("resolved_domain", "domain", "website", "site"):
        vv = getattr(record, field, None)
        if isinstance(vv, str) and vv.strip():
            return vv.strip().lower()

    return None


def _extract_company_name(record: Record) -> str:
    for field in ("company_name", "raw_name", "name"):
        v = getattr(record, field, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _final_source_for_output(record: Record, emails: List[str], phone: Optional[str]) -> str:
    """
    Output semantics (matches your Phase 8 spec examples):
      - ai       -> if record.ai_used is True
      - crawler  -> if contacts exist and AI was not used
      - none     -> otherwise
    """
    ai_used = bool(getattr(record, "ai_used", False))
    if ai_used:
        return "ai"

    if emails or (phone and phone.strip()):
        return "crawler"

    # If user explicitly set a valid source, keep it (optional)
    v = getattr(record, "source", None)
    if isinstance(v, str) and v.strip().lower() in ("ai", "crawler", "none"):
        return v.strip().lower()

    return "none"


def _final_status_for_output(record: Record, emails: List[str], phone: Optional[str]) -> str:
    """
    Final status policy for reporting:
      - keep dropped_* and not_found as-is
      - success iff we have at least one validated email or a phone
      - otherwise keep existing status (resolved/ambiguous/failed/new/...)
    """
    st = getattr(record, "status", None)
    st = st.strip().lower() if isinstance(st, str) and st.strip() else "unknown"

    if st.startswith("dropped"):
        return st
    if st == "not_found":
        return "not_found"

    if emails or (phone and phone.strip()):
        return "success"

    return st


def _row_from_record(record: Record, cfg: PipelineConfig) -> Dict[str, Any]:
    emails = _extract_emails(record, cfg)
    phone = _extract_phone(record)

    return {
        "company_id": getattr(record, "company_id", None),
        "company_name": _extract_company_name(record),
        "website": _extract_website(record),
        "emails": emails,
        "phone": phone,
        "status": _final_status_for_output(record, emails, phone),
        "source": _final_source_for_output(record, emails, phone),
    }


def _compute_metrics(rows: List[Dict[str, Any]], records: List[Record]) -> Dict[str, Any]:
    """
    Metrics per spec:
      - success_rate
      - ai_usage_rate (based on record.ai_used)
      - avg_latency_per_phase_s (from record.debug)
    """
    n = len(records)
    if n == 0:
        return {
            "n_records": 0,
            "success_rate": 0.0,
            "ai_usage_rate": 0.0,
            "avg_latency_per_phase_s": {},
            "status_counts": {},
            "source_counts": {},
        }

    status_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    success_n = 0

    # AI usage must use explicit flag
    ai_n = sum(1 for r in records if bool(getattr(r, "ai_used", False)))

    # Status/source counts should match the stored output
    for row in rows:
        st = str(row.get("status") or "unknown").lower()
        src = str(row.get("source") or "none").lower()
        status_counts[st] = status_counts.get(st, 0) + 1
        source_counts[src] = source_counts.get(src, 0) + 1
        if st == "success":
            success_n += 1

    # Latency aggregation from record.debug (phaseX_latency_s)
    lat_sum: Dict[str, float] = {}
    lat_cnt: Dict[str, int] = {}

    for r in records:
        dbg = getattr(r, "debug", None)
        if not isinstance(dbg, dict):
            continue
        for k, v in dbg.items():
            if not isinstance(k, str):
                continue
            if not _LAT_RE.match(k):
                continue
            try:
                val = float(v)
            except Exception:
                continue
            lat_sum[k] = lat_sum.get(k, 0.0) + val
            lat_cnt[k] = lat_cnt.get(k, 0) + 1

    avg_latency: Dict[str, float] = {}
    for k, s in lat_sum.items():
        c = lat_cnt.get(k, 0) or 0
        if c > 0:
            avg_latency[k] = s / c

    return {
        "n_records": n,
        "success_rate": success_n / n,
        "ai_usage_rate": ai_n / n,
        "avg_latency_per_phase_s": avg_latency,
        "status_counts": status_counts,
        "source_counts": source_counts,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    fieldnames = ["company_id", "company_name", "website", "emails", "phone", "status", "source"]

    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["emails"] = "; ".join(rr.get("emails") or [])
            w.writerow(rr)

    os.replace(tmp, path)


def run_phase(records: List[Record], cfg: PipelineConfig) -> List[Record]:
    """
    Phase 8: Persistence and Reporting

    Task 8.1: Store Result
      - Writes final_results.jsonl (and optional CSV/compact JSON)
    Task 8.2: Metrics
      - Writes metrics.json
    """
    output_dir = Path(getattr(cfg, "output_dir", Path("output")))
    results_name = str(getattr(cfg, "phase8_results_filename", "final_results.jsonl"))
    metrics_name = str(getattr(cfg, "phase8_metrics_filename", "metrics.json"))

    results_path = output_dir / results_name
    metrics_path = output_dir / metrics_name

    rows = [_row_from_record(r, cfg) for r in records]

    # Task 8.1: Store Result (idempotent overwrite)
    _atomic_write_text(results_path, _jsonl_dump(rows))

    # Optional CSV
    if bool(getattr(cfg, "phase8_write_csv", True)):
        csv_name = str(getattr(cfg, "phase8_csv_filename", "final_results.csv"))
        _write_csv(output_dir / csv_name, rows)

    # Optional compact JSON (single file)
    if bool(getattr(cfg, "phase8_write_compact_json", False)):
        compact_name = str(getattr(cfg, "phase8_compact_filename", "final_results_compact.json"))
        _atomic_write_text(output_dir / compact_name, json.dumps(rows, ensure_ascii=False, indent=2))

    # Task 8.2: Metrics (deterministic)
    metrics = _compute_metrics(rows, records)
    _atomic_write_text(metrics_path, json.dumps(metrics, ensure_ascii=False, indent=2))

    logger.info(
        "Phase 8 wrote results=%s metrics=%s success_rate=%.3f ai_usage_rate=%.3f",
        str(results_path),
        str(metrics_path),
        float(metrics.get("success_rate", 0.0)),
        float(metrics.get("ai_usage_rate", 0.0)),
    )

    return records
