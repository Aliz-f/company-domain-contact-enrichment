# Phase 0 — Pre-filter and Routing (Deterministic)

## Goal
Remove unusable input records early (empty or dummy/test names) and attach simple routing tags for downstream phases.

## Inputs
- `record.raw_name` (required)
- `record.address` (optional)
- `record.country` (optional)

## Tasks

### Task 0.1 — Record Filter
Drop records with:
- empty or placeholder names (e.g., `N/A`, `null`, `-`)
- dummy/test names (e.g., `test`, `dummy`, `lorem ipsum`)
- and remove the short-name bullet

Vocabulary is loaded from `assets/phase0/` when available and falls back to safe defaults otherwise.

## Outputs

### Kept records
- `record.tags.has_address` (bool)
- `record.tags.has_country` (bool)
- `record.tags.priority` (`"high_priority"` if address or country exists, else `"low_priority"`)
- `record.debug["phase0_reason"] = "kept"`

### Dropped records
- `record.status = "dropped_dummy"`
- `record.debug["phase0_reason"]` is set to a deterministic reason string

## Assets (assets/phase0/)
- `dummy_keywords.txt`
- `empty_markers.txt`

If assets are missing/unreadable, Phase 0 uses conservative built-in defaults.

## Idempotency
This phase:
- normalizes whitespace deterministically
- re-writes `record.debug["phase0_reason"]` every run
- never uses randomness or network calls

Running Phase 0 multiple times on the same input produces the same keep/drop decision and tags.
