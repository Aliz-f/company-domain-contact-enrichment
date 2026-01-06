# Phase 1 — Name Normalization

## Purpose
Phase 1 converts each kept company name into a deterministic, comparable normalized form that supports:
- stable tokenization
- legal suffix removal (e.g., ltd, gmbh, llc)
- generation of strict and relaxed name signals used downstream for domain candidate generation

This phase is purely local (no network) and is designed to be retryable and idempotent.

---

## Tasks Implemented

### Task 1.1 — Normalize Raw Name
Produces a normalized string that:
- is Unicode normalized
- has diacritics removed for robustness (e.g., ü → u)
- is lowercase
- retains word boundaries using spaces and single dashes
- keeps only `[a-z0-9]`, spaces, and dashes

Output:
- `record.normalized.normalized_raw`

### Task 1.2 — Legal Suffix Handling
Removes legal suffixes only from the end of the token list.
Supports multi-token suffix phrases such as:
- `l l c`
- `pte ltd`
- `s a r l`

Outputs:
- suffix-stripped token stream used for strict token construction
- debug list of removed suffix tokens

### Task 1.3 — Tokenization
Creates:
- strict tokens: base tokens with strict stopwords removed
- relaxed name: a weak signal (first strict token)

Outputs:
- `record.normalized.tokens`
- `record.normalized.strict_name`
- `record.normalized.relaxed_name`

---

## Inputs
- `record.raw_name`

---

## Outputs
Phase 1 always rewrites the following fields (idempotent behavior):
- `record.normalized.normalized_raw`
- `record.normalized.strict_name`
- `record.normalized.relaxed_name`
- `record.normalized.tokens`

Debug keys:
- `record.debug["phase1_reason"]`
- `record.debug["phase1_raw_tokens"]`
- `record.debug["phase1_removed_suffix"]`
- `record.debug["phase1_strict_tokens_before_stopwords"]`

Drop policy:
- If normalization yields no letters → `record.status = "dropped_empty"`
- If the name reduces to only legal suffixes → `record.status = "dropped_empty"`
- If strict name becomes empty → `record.status = "dropped_empty"`

---

## Assets

### Required (recommended)
Place assets in:
- `assets/phase1/legal_suffixes.txt`
- `assets/phase1/stopwords_strict.txt`

Format:
- one entry per line
- comments allowed with `#`
- entries are treated case-insensitively

### Fallback behavior
If assets are missing or empty, Phase 1 uses conservative built-in fallback lists.

---

## Determinism and Idempotency
- Records dropped by Phase 1 (dropped_empty) are still reprocessable when Phase 1 is rerun
- No randomness.
- No network access.
- Outputs are recomputed each run to avoid stale state from previous attempts.
