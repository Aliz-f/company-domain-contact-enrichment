# Phase 2 — Domain Candidate Generation (Deterministic)

## Goal
Generate a small ordered list of plausible website domains for each company, without AI.

## Inputs
- `record.normalized.tokens` (strict tokens from Phase 1)
- `record.normalized.relaxed_name` (weak signal from Phase 1)
- `record.normalized.normalized_raw` (fallback text)
- `record.country` (optional)

## Tasks

### Task 2.1 — Country-aware TLD Selection
Select TLDs based on country:
- If country is known, prioritize its ccTLD first
- Special-case UK with public suffixes (`co.uk`, `org.uk`, ...)
- Always include global TLDs as fallback

**Outputs**
- `record.debug["phase2_selected_tlds"]`
- `record.debug["phase2_cc_tld"]`

### Task 2.2 — Candidate Stem Construction and Pattern Expansion
Build stems from strict tokens (filtered by `bad_stems` and minimum length) and generate domain patterns:
- `stem.tld`
- `stem-extra.tld`
- `stemextra.tld`

Expansion is bounded to avoid large intermediate lists.

**Outputs**
- `record.debug["phase2_strict_stems"]`
- `record.debug["phase2_relaxed_stems"]`

### Task 2.3 — Scoring, Ordering, and Truncation
Score candidates deterministically using:
- ccTLD preference
- global TLD preference (`com` > `net/org/biz` > other)
- strict vs relaxed stem matches
- simplicity penalties (shorter stems, fewer dashes)

Then sort and keep up to `cfg.max_domain_candidates` (default 15).

**Output**
- `record.domain_candidates` (ordered)

## Outputs
- `record.domain_candidates`
- Debug keys:
    - `phase2_reason`
    - `phase2_selected_tlds`, `phase2_cc_tld`
    - `phase2_strict_stems`, `phase2_relaxed_stems`

## Assets (assets/phase2/)
- `country_to_cctld.json`
- `global_tlds.txt`
- `uk_public_suffixes.txt`
- `bad_stems.txt`
- `pattern_extras.txt`

If assets are missing, Phase 2 uses safe built-in fallbacks.

## Idempotency
Phase 2 resets `record.domain_candidates` and Phase-2 debug keys each run, so rerunning produces consistent results.
