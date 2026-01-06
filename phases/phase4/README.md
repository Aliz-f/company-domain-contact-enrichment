# Phase 4 — Domain Resolution (Deterministic)

## Goal
Convert the list of validated domains from Phase 3 into a single resolved website when possible, and flag ambiguous cases deterministically.

## Inputs
- `record.valid_domains` (ordered list produced by Phase 3)

## Tasks

### Task 4.1 — Single Domain Resolution
If exactly one valid domain exists:
- Set `record.resolution.resolved_domain` to that domain
- Set `record.resolution.ambiguity_flag = False`
- Set `record.resolution.reason = "single_domain_resolved"`

### Task 4.2 — Ambiguity Detection
If there are zero or multiple valid domains:
- Set `record.resolution.resolved_domain = None`
- Set `record.resolution.ambiguity_flag = True`
- Set reason:
    - `no_valid_domains` if list is empty
    - `multiple_valid_domains` if list size > 1

## Outputs
- `record.resolution.resolved_domain`
- `record.resolution.ambiguity_flag`
- `record.resolution.reason`
- Debug keys:
    - `record.debug["phase4_reason"]`
    - `record.debug["phase4_resolved_domain"]`
    - `record.debug["phase4_ambiguity_flag"]`

## Status Policy (Configurable)
If `cfg.phase4_set_status` is enabled:
- Resolved records get `cfg.phase4_status_resolved` (default: `resolved`)
- Unresolved records get `cfg.phase4_status_unresolved` (default: `unresolved`)
- Ambiguous records get `cfg.phase4_status_ambiguous` (default: `ambiguous`)

Dropped records are skipped (their status is not changed).

## Determinism and Idempotency
- Phase 4 always re-sanitizes and de-duplicates `valid_domains` while preserving order.
- Phase 4 resets all resolution fields before computing outputs to prevent stale values on reruns.
- For the same inputs and config, the outputs are identical.

## Notes
- Phase 4 does not perform any network I/O and does not use AI.
- Phase 5 uses `resolved_domain` (if present) as the primary crawl target.
