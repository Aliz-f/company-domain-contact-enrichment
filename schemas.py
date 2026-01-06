from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Tags(BaseModel):
    has_address: bool = False
    has_country: bool = False
    priority: str = "low_priority"  # "high_priority" or "low_priority"


class NormalizedName(BaseModel):
    normalized_raw: str = ""
    strict_name: str = ""
    relaxed_name: str = ""
    tokens: List[str] = Field(default_factory=list)


class Resolution(BaseModel):
    resolved_domain: Optional[str] = None
    ambiguity_flag: bool = False
    reason: Optional[str] = None  # e.g., "no_live_domains" / "multiple_live_domains"
    ai_prompt: str | None = None
    ai_response: str | None = None


class Contacts(BaseModel):
    emails: List[str] = Field(default_factory=list)
    phone: Optional[str] = None
    crawl_status: str = "not_started"  # success/failed/blocked/timeout/not_started


class CompanyMeta(BaseModel):
    company_number: Optional[str] = None
    company_category: Optional[str] = None
    company_status: Optional[str] = None
    country_of_origin: Optional[str] = None
    incorporation_date: Optional[str] = None
    dissolution_date: Optional[str] = None
    sic_codes: List[str] = Field(default_factory=list)
    uri: Optional[str] = None
    previous_names: List[str] = Field(default_factory=list)
    # Keep aligned with previous_names; allow None when a date is missing.
    previous_name_change_dates: List[Optional[str]] = Field(default_factory=list)


class Record(BaseModel):
    """
    Record is the evolving state object that each phase reads and writes.

    Idempotency: phases must set fields deterministically from current state + their own inputs,
    not from wall-clock randomness.
    """

    company_id: str
    raw_name: str
    address: Optional[str] = None
    country: Optional[str] = None

    meta: CompanyMeta = Field(default_factory=CompanyMeta)

    # Pipeline state
    # Recommended lifecycle:
    # new -> dropped_dummy | resolved/ambiguous/not_found/failed -> success
    status: str = "new"

    # source is kept for final reporting; you can interpret it as "how the official website was resolved":
    # ai / crawler / none
    source: str = "none"

    # Explicit AI usage flag (preferred for guardrails/metrics).
    # Phase 6 should set ai_used=True when it makes a decision.
    ai_used: bool = False

    tags: Tags = Field(default_factory=Tags)
    normalized: NormalizedName = Field(default_factory=NormalizedName)

    domain_candidates: List[str] = Field(default_factory=list)
    valid_domains: List[str] = Field(default_factory=list)
    resolution: Resolution = Field(default_factory=Resolution)

    contacts: Contacts = Field(default_factory=Contacts)
    validated_contacts: Contacts = Field(default_factory=Contacts)

    debug: Dict[str, Any] = Field(default_factory=dict)
