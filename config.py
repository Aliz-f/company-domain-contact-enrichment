from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------------------------
# Column handling (storage.py)
# -------------------------
# The CSV headers contain leading spaces in some names (e.g., " CompanyNumber").
# We STRIP whitespace from all column names at load time.
# These constants are the canonical (stripped) names.

COL_COMPANY_NAME = "CompanyName"
COL_COMPANY_NUMBER = "CompanyNumber"

# Address fields (canonical, stripped)
COL_ADDR_CAREOF = "RegAddress.CareOf"
COL_ADDR_POBOX = "RegAddress.POBox"
COL_ADDR_LINE1 = "RegAddress.AddressLine1"
COL_ADDR_LINE2 = "RegAddress.AddressLine2"
COL_ADDR_TOWN = "RegAddress.PostTown"
COL_ADDR_COUNTY = "RegAddress.County"
COL_ADDR_COUNTRY = "RegAddress.Country"
COL_ADDR_POSTCODE = "RegAddress.PostCode"

COL_COUNTRY_OF_ORIGIN = "CountryOfOrigin"
COL_URI = "URI"

COL_COMPANY_CATEGORY = "CompanyCategory"
COL_COMPANY_STATUS = "CompanyStatus"
COL_INCORPORATION_DATE = "IncorporationDate"
COL_DISSOLUTION_DATE = "DissolutionDate"

# SIC fields
SIC_COLS = [
    "SICCode.SicText_1",
    "SICCode.SicText_2",
    "SICCode.SicText_3",
    "SICCode.SicText_4",
]

# Previous name columns pattern (canonical, stripped)
PREV_NAME_COL_TEMPLATE = "PreviousName_{i}.CompanyName"
PREV_NAME_DATE_TEMPLATE = "PreviousName_{i}.CONDATE"
PREV_NAME_MAX = 10

# -------------------------
# Pipeline phases (module names) (pipeline_runner.py)
# -------------------------
PHASE_MODULES: Dict[int, str] = {
    0: "phases.phase0.phase0_filter",
    1: "phases.phase1.phase1_normalize",
    2: "phases.phase2.phase2_candidates",
    3: "phases.phase3.phase3_validate",
    4: "phases.phase4.phase4_resolve",
    5: "phases.phase5.phase5_crawl",
    6: "phases.phase6.phase6_ai",
    7: "phases.phase7.phase7_postvalidate",
    8: "phases.phase8.phase8_persist_metrics",
}

PHASE_NAMES: Dict[int, str] = {
    0: "pre_filter_and_routing",
    1: "name_normalization",
    2: "domain_candidate_generation",
    3: "fast_domain_validation",
    4: "domain_resolution",
    5: "web_crawling_and_extraction",
    6: "ai_assisted_resolution",
    7: "post_validation",
    8: "persistence_and_reporting",
}


@dataclass(frozen=True)
class PipelineConfig:
    # -------------------------
    # Storage / runner (storage.py, pipeline_runner.py)
    # -------------------------
    max_rows: Optional[int] = 500
    output_dir: Path = Path("output")

    # -------------------------
    # Phase 0 (pre-filter and routing)
    # -------------------------
    phase0_min_name_len: int = 3

    # -------------------------
    # Phase 2 (domain candidate generation)
    # -------------------------
    max_domain_candidates: int = 15
    phase2_min_stem_len: int = 3

    # -------------------------
    # Phase 3 (fast domain validation)
    # -------------------------
    phase3_http_snippet_bytes: int = 8192
    phase3_http_range_end: int = 4095
    phase3_head_retry_statuses: Tuple[int, ...] = (403, 405)
    phase3_reject_wildcard_dns: bool = True
    phase3_allow_protected_status: bool = False

    # -------------------------
    # Phase 4 (domain resolution; deterministic, no AI)
    # -------------------------
    phase4_set_status: bool = True
    # IMPORTANT: Phase 4 only resolves the domain. It should not claim final "success".
    phase4_status_resolved: str = "resolved"
    phase4_status_ambiguous: str = "ambiguous"

    # -------------------------
    # Networking (used by phases 3 and 5)
    # -------------------------
    dns_timeout_s: float = 6.0
    enable_wildcard_dns_default: bool = True
    dns_retries_on_timeout: int = 1
    dns_retry_backoff: float = 0.35
    dns_limit: int = 25

    http_timeout_s: float = 10.0
    http_max_keepalive_connections: int = 50
    http_max_connections: int = 200

    http_user_agent: str = "Mozilla/5.0 (compatible; CompanyPipeline/1.0; +https://example.invalid)"

    # -------------------------
    # Phase 5 (web crawling and extraction)
    # -------------------------
    crawl_max_pages: int = 5
    crawl_max_depth: int = 2
    crawl_total_byte_limit: float = 2_000_000
    crawl_max_bytes_per_page: float = 300_000
    crawl_per_domain_concurrency: int = 2
    crawl_global_concurrency: int = 15
    crawl_per_domain_delay_s: float = 0.5
    crawl_timeout_s: float = 10.0
    crawl_extract_phone: bool = True

    # -------------------------
    # Phase 6 (AI guardrail)
    # -------------------------
    max_ai_fraction: float = 0.30
    gemini_api_key: str | None = None

    gemini_models: tuple[str, ...] | None = ("gemini-2.0-flash", "gemini-2.5-flash")
    gemini_model: str = "gemini-2.0-flash"
    gemini_batch_size: int = 50
    gemini_timeout_s: float = 30.0

    gemini_min_interval_s: float = 0.25
    gemini_max_retries: int = 6
    gemini_backoff_base_s: float = 1.0
    gemini_backoff_max_s: float = 20.0
    gemini_backoff_jitter_s: float = 0.4

    # -------------------------
    # Phase 7 (post-validation, no AI)
    # -------------------------
    phase7_enable_syntax_validation: bool = True
    phase7_enable_mx_check: bool = True
    phase7_enable_deduplication: bool = True
    phase7_require_mx: bool = False

    phase7_mx_timeout_s: float = 6.0
    phase7_mx_retries_on_timeout: int = 1
    phase7_mx_retry_backoff: float = 0.35

    phase7_max_validated_contacts_per_record: int | None = None

    # -------------------------
    # Phase 8 (persistence + reporting)
    # -------------------------
    phase8_results_filename: str = "final_results.jsonl"
    phase8_metrics_filename: str = "metrics.json"
    phase8_write_csv: bool = True
    phase8_csv_filename: str = "final_results.csv"

    phase8_write_compact_json: bool = False
    phase8_compact_filename: str = "final_results_compact.json"
    phase8_prefer_validated_contacts: bool = True

    # -------------------------
    # Assets
    # -------------------------
    assets_dir: Path = Path("assets")

    def phase_assets_dir(self, phase: int) -> Path:
        return self.assets_dir / f"phase{int(phase)}"

    def phase_asset_path(self, phase: int, filename: str) -> Path:
        return self.phase_assets_dir(phase) / filename


# Address columns considered for Phase 0 tagging (storage.py -> Record.address)
ADDRESS_COLS: List[str] = [
    COL_ADDR_CAREOF,
    COL_ADDR_POBOX,
    COL_ADDR_LINE1,
    COL_ADDR_LINE2,
    COL_ADDR_TOWN,
    COL_ADDR_COUNTY,
    COL_ADDR_POSTCODE,
]
