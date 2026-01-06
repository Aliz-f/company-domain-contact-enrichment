from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from config import (
    ADDRESS_COLS,
    COL_ADDR_COUNTRY,
    COL_COMPANY_CATEGORY,
    COL_COMPANY_NAME,
    COL_COMPANY_NUMBER,
    COL_COMPANY_STATUS,
    COL_COUNTRY_OF_ORIGIN,
    COL_DISSOLUTION_DATE,
    COL_INCORPORATION_DATE,
    COL_URI,
    PREV_NAME_COL_TEMPLATE,
    PREV_NAME_DATE_TEMPLATE,
    PREV_NAME_MAX,
    SIC_COLS,
    PipelineConfig,
)
from schemas import CompanyMeta, Record


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (the dataset has leading spaces)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _clean_str(x: object) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def _build_address(row: pd.Series) -> Optional[str]:
    """
    Build address string from ADDRESS_COLS.
    Country is intentionally NOT included here to avoid duplication; it is stored separately in record.country.
    """
    parts: List[str] = []
    for c in ADDRESS_COLS:
        v = _clean_str(row.get(c))
        if v:
            parts.append(v)
    addr = ", ".join(parts).strip()
    return addr if addr else None


def _collect_previous_names(row: pd.Series) -> Tuple[List[str], List[Optional[str]]]:
    """
    Collect previous names and change dates while keeping alignment:
    - If a name exists, we append it
    - We append the corresponding date (or None if missing)
    """
    names: List[str] = []
    dates: List[Optional[str]] = []
    for i in range(1, PREV_NAME_MAX + 1):
        name_col = PREV_NAME_COL_TEMPLATE.format(i=i)
        date_col = PREV_NAME_DATE_TEMPLATE.format(i=i)
        n = _clean_str(row.get(name_col))
        d = _clean_str(row.get(date_col))
        if n:
            names.append(n)
            dates.append(d)
    return names, dates


def _collect_sic_codes(row: pd.Series) -> List[str]:
    sics: List[str] = []
    for c in SIC_COLS:
        v = _clean_str(row.get(c))
        if v:
            sics.append(v)
    return sics


def load_input_csv(path: str | Path, cfg: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = _strip_columns(df)
    if cfg.max_rows is not None:
        df = df.head(cfg.max_rows)
    return df


def records_from_dataframe(df: pd.DataFrame) -> List[Record]:
    """
    Convert the raw dataframe into a list[Record].

    - company_id uses CompanyNumber when available (stable)
    - otherwise falls back to "raw_name::{row_index}" to avoid collisions
    - raw_name uses CompanyName
    - address is built from RegAddress.* fields (excluding country)
    - country uses RegAddress.Country (fallback CountryOfOrigin)
    """
    df = _strip_columns(df)

    records: List[Record] = []
    for idx, row in df.iterrows():
        raw_name = _clean_str(row.get(COL_COMPANY_NAME)) or ""
        company_number = _clean_str(row.get(COL_COMPANY_NUMBER)) or ""

        company_id = company_number if company_number else f"{raw_name}::{idx}"

        address = _build_address(row)
        country = _clean_str(row.get(COL_ADDR_COUNTRY)) or _clean_str(row.get(COL_COUNTRY_OF_ORIGIN))

        prev_names, prev_dates = _collect_previous_names(row)
        sic_codes = _collect_sic_codes(row)

        meta = CompanyMeta(
            company_number=company_number or None,
            company_category=_clean_str(row.get(COL_COMPANY_CATEGORY)),
            company_status=_clean_str(row.get(COL_COMPANY_STATUS)),
            country_of_origin=_clean_str(row.get(COL_COUNTRY_OF_ORIGIN)),
            incorporation_date=_clean_str(row.get(COL_INCORPORATION_DATE)),
            dissolution_date=_clean_str(row.get(COL_DISSOLUTION_DATE)),
            sic_codes=sic_codes,
            uri=_clean_str(row.get(COL_URI)),
            previous_names=prev_names,
            previous_name_change_dates=prev_dates,
        )

        rec = Record(
            company_id=str(company_id),
            raw_name=raw_name,
            address=address,
            country=country,
            meta=meta,
        )
        records.append(rec)

    return records


def records_to_dataframe(records: Iterable[Record]) -> pd.DataFrame:
    return pd.json_normalize([r.model_dump() for r in records])


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent), suffix=".tmp") as tf:
        tf.write(data)
        tmp_name = tf.name
    os.replace(tmp_name, dst)


def _atomic_write_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(dst.parent), suffix=".tmp", encoding="utf-8") as tf:
        tf.write(text)
        tmp_name = tf.name
    os.replace(tmp_name, dst)


def save_phase(records: List[Record], output_dir: str | Path, phase: int) -> Path:
    """
    Save checkpoints as JSONL to preserve nested structure.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"phase{phase}.jsonl"

    lines: List[str] = []
    for r in records:
        lines.append(json.dumps(r.model_dump(), ensure_ascii=False))
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))

    return path


def load_phase(output_dir: str | Path, phase: int) -> List[Record]:
    """
    Load JSONL checkpoint and reconstruct list[Record] with full nested fields.
    """
    path = Path(output_dir) / f"phase{phase}.jsonl"
    records: List[Record] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(Record.model_validate(obj))

    return records


def save_final_csv(records: List[Record], output_path: str | Path) -> Path:
    """
    Write a human-friendly flat CSV final output (runner convenience output).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records)
    df.to_csv(output_path, index=False)
    return output_path


def save_metrics(metrics: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metrics, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(output_path, payload)
    return output_path
