# Phase 8 — Persistence and Reporting

## Goal
Persist the final pipeline outputs to disk and compute deterministic summary metrics for reporting.

Phase 8 is the final stage of the pipeline. It does **not** perform crawling or AI. It only reads the final `Record` state produced by earlier phases and writes:

- A **JSONL** results file (one row per record)
- Optional **CSV** (human-readable)
- Optional **compact JSON** (pretty-printed)
- A **metrics.json** summary

---

## Inputs (Read)
Phase 8 reads the final state of each `Record`, primarily:

- `record.company_id`
- `record.raw_name`
- `record.resolution.resolved_domain`
- `record.contacts.emails`, `record.contacts.phone`
- `record.validated_contacts.emails`, `record.validated_contacts.phone` (preferred if enabled)
- `record.status`
- `record.source`
- `record.ai_used`
- `record.debug["phase{X}_latency_s"]` (optional per-phase latencies)

---

## Tasks

### Task 8.1 — Store Result
Phase 8 creates a final output row per record with the following fields:

- `company_id`
- `company_name`
- `website`
- `emails` (list of strings)
- `phone` (string or null)
- `status`
- `source`

#### Website selection
- Uses `record.resolution.resolved_domain` if present (lowercased).

#### Email selection
Phase 8 can prefer validated emails (Phase 7 output):

- If `cfg.phase8_prefer_validated_contacts = True` (default):
  - Uses `record.validated_contacts.emails` if non-empty.
  - Otherwise falls back to `record.contacts.emails`.

- If `cfg.phase8_prefer_validated_contacts = False`:
  - Combines validated and raw crawler emails (validated first), then de-duplicates.

All email outputs are:
- normalized (trimmed, lowercased, trailing punctuation removed)
- de-duplicated while preserving order

#### Phone selection
- Prefers `record.validated_contacts.phone`, then `record.contacts.phone`.

#### Final `status` (reporting)
Phase 8 computes a final reporting status:

- Keeps `dropped_*` unchanged
- Keeps `not_found` unchanged
- Sets `success` **iff** at least one contact signal exists:
  - at least one email, **or**
  - a non-empty phone
- Otherwise keeps the record’s current status (e.g., `resolved`, `ambiguous`, `failed`, `new`)

This matches the pipeline meaning of success: **a record is successful only if it produced usable contact output**.

#### Final `source` (reporting)
Phase 8 uses the following reporting semantics:

- `ai` if `record.ai_used == True`
- `crawler` if any contact signal exists (emails or phone) and AI was not used
- `none` otherwise

> Note: `record.source` may still be preserved in earlier phases for internal meaning, but Phase 8 output `source` is intended to reflect the **final resolution/extraction driver**.

#### Output files
Phase 8 writes into `cfg.output_dir`:

- JSONL (always): `cfg.phase8_results_filename` (default `final_results.jsonl`)
- CSV (optional): `cfg.phase8_csv_filename` (default `final_results.csv`)
- Compact JSON (optional): `cfg.phase8_compact_filename` (default `final_results_compact.json`)

All writes are **atomic** (write to `.tmp`, then `os.replace`) to support retry safety.

---

### Task 8.2 — Metrics
Phase 8 computes deterministic summary metrics:

- `success_rate`
  - fraction of records whose **final output status** is `success`

- `ai_usage_rate`
  - fraction of records with `record.ai_used == True`
  - This enforces the project guardrail that AI should be used on a limited fraction of records.

- `avg_latency_per_phase_s`
  - averages of per-phase latencies found in `record.debug` under keys like:
    - `phase0_latency_s`, `phase1_latency_s`, …, `phase8_latency_s`
  - computed by summing per key and dividing by count (only numeric values included)

- `status_counts`
  - counts of final output statuses from stored rows

- `source_counts`
  - counts of final output sources from stored rows

The metrics file is written to:

- `cfg.phase8_metrics_filename` (default `metrics.json`)

---

## Outputs (Written)

### Results row schema
Each JSONL line is a dictionary:

```json
{
  "company_id": "12345",
  "company_name": "ABC Company Ltd",
  "website": "abc.de",
  "emails": ["info@abc.de", "sales@abc.de"],
  "phone": "+4930123456",
  "status": "success",
  "source": "crawler"
}
````

### Metrics schema

`metrics.json` structure:

```json
{
  "n_records": 1000,
  "success_rate": 0.85,
  "ai_usage_rate": 0.25,
  "avg_latency_per_phase_s": {
    "phase0_latency_s": 0.01,
    "phase1_latency_s": 0.02
  },
  "status_counts": {
    "success": 850,
    "not_found": 100,
    "ambiguous": 50
  },
  "source_counts": {
    "crawler": 800,
    "ai": 200,
    "none": 0
  }
}
```

---

## PipelineConfig Settings Used

* `output_dir: str | Path`
* `phase8_results_filename: str` (default `final_results.jsonl`)
* `phase8_metrics_filename: str` (default `metrics.json`)

Optional:

* `phase8_write_csv: bool` (default `True`)
* `phase8_csv_filename: str` (default `final_results.csv`)
* `phase8_write_compact_json: bool` (default `False`)
* `phase8_compact_filename: str` (default `final_results_compact.json`)
* `phase8_prefer_validated_contacts: bool` (default `True`)

---

## Determinism and Idempotency

Phase 8 is designed to be safe to rerun:

* Writes are atomic and overwrite prior outputs deterministically.
* Email normalization and de-duplication are stable.
* Metrics are computed deterministically from current record states.
* No network I/O and no AI calls occur in Phase 8.

---

## Notes

* Phase 8 should be run only after Phase 7 if you want validated contacts preferred.
* The definition of `success` in Phase 8 is based on **having contact output**, not only on intermediate statuses like `resolved`.

