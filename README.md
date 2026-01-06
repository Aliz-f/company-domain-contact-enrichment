# Company Domain + Contact Enrichment Pipeline

This project enriches Companies House style company records by:

1. Cleaning and normalizing company names
2. Generating candidate domains deterministically
3. Validating domains with lightweight DNS + HTTP checks
4. Resolving to a single best domain (or marking ambiguity)
5. Crawling the resolved website to extract emails and phones
6. Optionally using AI to resolve ambiguous cases (with a strict usage cap)
7. Optionally post validating contacts
8. Persisting final outputs and metrics (Phase 8)

The pipeline is designed to be **retryable** and **idempotent**:

- **Retryable:** rerun a single phase without rerunning earlier phases (checkpoints are saved per phase).
- **Idempotent:** rerunning the same phase with the same inputs produces the same outputs.

---

## Key Outputs and Ownership

**Phase 8 is the single owner of final reporting files:**

- `output/final_results.jsonl`
- `output/final_results.csv`
- `output/metrics.json`

The runner also writes operational artifacts that do **not** collide with Phase 8:

- `output/phase{p}.jsonl` checkpoints
- `output/metrics_by_phase.json` (when running `--phase all`)
- `output/metrics_phase{p}.json` (when running a single phase)
- `output/results.csv` (runner flat CSV snapshot of the final `Record` list)

This avoids the common bug where both Phase 8 and the runner try to write the same filenames.

---

## Features

- 8-phase pipeline with deterministic behavior and checkpoints (`output/phase{p}.jsonl`)
- Deterministic candidate generation and domain resolution
- Fast validation using DNS + HTTP heuristics (parked/directory detection)
- Crawling with strict limits and filtering to extract contact info
- AI usage hard cap (default 30%) enforced in the runner
- Phase 8 persistence + metrics with clear semantics:
    - `status="success"` only when at least one email or phone exists
    - `source="ai"` if `record.ai_used=True`, else `crawler` if contacts exist, else `none`

---

## Project Structure

```text
.
├── assets/                           # Phase-specific rules, wordlists, patterns
│   ├── phase0/ dummy_keywords.txt empty_markers.txt
│   ├── phase1/ legal_suffixes.txt stopwords_strict.txt
│   ├── phase2/ bad_stems.txt country_to_cctld.json global_tlds.txt pattern_extras.txt uk_public_suffixes.txt
│   ├── phase3/ directory_host_hints.txt parked_keywords.txt
│   └── phase5/ free_email_providers.txt no_reply_locals.txt role_local_parts.txt target_paths.txt
├── data/                             # Input data
│   ├── basic_company_data.csv
│   └── BasicCompanyDataAsOneFile-2025-12-01.zip
├── phases/                           # Pipeline phases (0..8)
│   ├── phase0/ phase0_filter.py
│   ├── phase1/ phase1_normalize.py
│   ├── phase2/ phase2_candidates.py
│   ├── phase3/ phase3_validate.py
│   ├── phase4/ phase4_resolve.py
│   ├── phase5/ phase5_crawl.py
│   ├── phase6/ phase6_ai.py
│   ├── phase7/ phase7_postvalidate.py
│   └── phase8/ phase8_persist_metrics.py
├── output/                           # Checkpoints + outputs
│   ├── phase0.jsonl ... phase8.jsonl
│   ├── final_results.jsonl final_results.csv metrics.json
│   ├── results.csv
│   ├── metrics_by_phase.json
│   └── metrics_phase*.json
├── config.py                         # PipelineConfig + constants + PHASE_MODULES mapping
├── schemas.py                        # Pydantic schemas (Record, Contacts, Resolution, etc.)
├── storage.py                        # CSV loader, JSONL checkpoint I/O, metrics writers
├── pipeline_runner.py                # Main entrypoint CLI
└── test.py                           # local experiments / scratch
````

---

## Requirements

* Conda (Miniconda or Anaconda)
* Python version defined by `environment.yml`
* Network access for:

    * Phase 3 (DNS + HTTP)
    * Phase 5 (crawling)
    * Phase 6 (if AI is enabled)

---

## Installation (Conda)

This project uses a conda environment defined in `environment.yml`.

### Create the environment

```bash
conda env create -f environment.yml
conda activate scrap
```

### Update the environment (later)

```bash
conda env update -f environment.yml --prune
conda activate scrap
```

---

## Quick Start

```bash
conda activate scrap

python pipeline_runner.py \
  --input data/basic_company_data.csv \
  --output-dir output \
  --phase all \
  --log-level INFO
```

---

## Configuration

### `config.py`

`PipelineConfig` controls the pipeline behavior such as:

* `max_rows` (limit how many input rows to process)
* candidate caps, DNS/HTTP timeouts, concurrency limits
* crawl limits and allowed paths
* `max_ai_fraction` (AI usage cap, default `0.30`)
* Phase 8 output filenames and toggles:

    * `phase8_results_filename`, `phase8_metrics_filename`, `phase8_write_csv`, etc.

### Environment variables (`.env`)

If Phase 6 uses an AI provider, create a `.env` in the project root:

```text
AI_API_KEY=...
AI_MODEL=...
```

The runner tries to load `.env` if `python-dotenv` is installed.

> Tip: If you want to fully disable AI, set `max_ai_fraction = 0.0` in config
> or skip Phase 6 entirely when running specific phases.

---

## Input

The pipeline expects an input CSV such as:

* `data/basic_company_data.csv`

The expected Companies House style fields are defined in `config.py` and parsed in `storage.py`.

Phase 0 drops obvious dummy or empty names using:

* `assets/phase0/dummy_keywords.txt`
* `assets/phase0/empty_markers.txt`

---

## Running the Pipeline

### Run all phases (0 → 8)

```bash
python pipeline_runner.py \
  --input data/basic_company_data.csv \
  --output-dir output \
  --phase all \
  --log-level INFO
```

### Resume from last completed checkpoint

```bash
python pipeline_runner.py \
  --input data/basic_company_data.csv \
  --output-dir output \
  --phase all \
  --resume \
  --log-level INFO
```

### Start from a specific phase

Requires `output/phase{start-1}.jsonl` to exist:

```bash
python pipeline_runner.py \
  --input data/basic_company_data.csv \
  --output-dir output \
  --phase all \
  --start-phase 4 \
  --log-level INFO
```

### Run a single phase

Example: run only Phase 5 using the Phase 4 checkpoint:

```bash
python pipeline_runner.py \
  --input data/basic_company_data.csv \
  --output-dir output \
  --phase 5 \
  --log-level INFO
```

When running a single phase, the runner also writes:

* `output/metrics_phase5.json`

---

## Outputs

### Checkpoints (retryability)

After each phase:

* `output/phase{p}.jsonl`

This stores the list of `Record` objects after that phase.

### Final reporting outputs (Phase 8)

Phase 8 writes:

* `output/final_results.jsonl`
  One JSON object per record with:
  `company_id, company_name, website, emails, phone, status, source`

* `output/final_results.csv`
  Flat CSV where emails are joined with `; `

* `output/metrics.json`
  Includes:

    * `success_rate`
    * `ai_usage_rate` (based on explicit `record.ai_used`)
    * `avg_latency_per_phase_s` (from `record.debug["phaseX_latency_s"]`)
    * `status_counts`, `source_counts`

### Runner outputs (operational)

The runner may also write:

* `output/metrics_by_phase.json` (all phases timing + counts)
* `output/results.csv` (flat dump of `Record` state)

These are intentionally separate from Phase 8 outputs.

---

## Pipeline Phases

### Phase 0 — Filter / Tag

* Loads raw records from CSV
* Drops dummy or empty names using `assets/phase0/*`
* Initializes tags and debug

### Phase 1 — Normalize Company Name

* Strips legal suffixes (`assets/phase1/legal_suffixes.txt`)
* Generates normalized name tokens

### Phase 2 — Candidate Domain Generation

* Generates candidate domains using TLD lists, UK public suffixes, ccTLD mapping, and pattern extras
* Filters bad stems (`assets/phase2/bad_stems.txt`)
* Produces deterministic ordering and applies caps

### Phase 3 — Fast Domain Validation

* Sanitizes candidates
* DNS resolution checks (`dnspython`)
* HTTP(S) probes with short timeouts
* Parked domain detection (`assets/phase3/parked_keywords.txt`)
* Directory host hints (`assets/phase3/directory_host_hints.txt`)
* Outputs `valid_domains`

### Phase 4 — Deterministic Resolution

* Picks a single resolved domain when confident
* Otherwise marks ambiguity and preserves candidates for AI

### Phase 5 — Crawl for Contacts

* Crawls a limited set of target paths (`assets/phase5/target_paths.txt`)
* Extracts emails and phones
* Filters out no-reply locals, role locals, and free email providers
* Stores results in `record.contacts`

### Phase 6 — AI Resolve

* Runs only for ambiguous cases
* Must respect AI usage cap (`cfg.max_ai_fraction`)
* Sets `record.ai_used=True` when used

### Phase 7 — Post validate

* Validates emails and phones depending on config
* Stores results in `record.validated_contacts`

### Phase 8 — Persist + Metrics

* Writes `final_results.jsonl`, optional CSV and compact JSON
* Writes `metrics.json`
* Defines final reporting semantics for `status` and `source`

---

## Semantics

### Status

* `success` means: at least one usable email or phone exists
* Domain-only resolution does not count as success by itself
* Dropped statuses remain unchanged (e.g., `dropped_*`)

### Source

* `ai` if `record.ai_used=True`
* `crawler` if contacts exist and AI was not used
* `none` otherwise

---

## Debugging

### Use DEBUG logs

```bash
python pipeline_runner.py \
  --input data/basic_company_data.csv \
  --output-dir output \
  --phase all \
  --log-level DEBUG
```

### Common issues

* **Too many requests / slow crawl:** reduce crawl paths, timeouts, and concurrency in `PipelineConfig`.
* **AI cap exceeded:** runner raises an error if AI usage exceeds `cfg.max_ai_fraction`.

---

## Clean Restart

To restart from scratch, delete checkpoints:

```bash
rm -f output/phase*.jsonl
```

(Optional) delete final outputs too:

```bash
rm -f output/final_results.* output/metrics.json output/results.csv output/metrics_*.json
```
