# Phase 6 — AI-assisted Ambiguity Resolution (Gemini Flash)

## Goal
Resolve domain deadlocks **only when deterministic logic cannot pick a single official domain**.

Phase 6 is a guardrail phase:
- it does **not** crawl the web
- it does **not** guess emails
- it chooses **only** from already-generated candidate domains

---

## Entry Condition
A record runs in Phase 6 only if:

- `record.resolution.ambiguity_flag == True`

Dropped records are skipped.

---

## Tasks

### Task 6.1 — Prompt Input Builder
Phase 6 builds a constrained prompt using:

- Company label (prefers `record.normalized.strict_name`, then `record.raw_name`)
- Optional address (`record.address`)
- Candidate domains

Candidate domain list priority (deterministic):
1. `record.valid_domains` (multiple live domains from Phase 3)
2. `record.domain_candidates` (guesses from Phase 2, when no live domains exist)

The prompt forces the model to return **only**:
- one **exact candidate domain** (copy exactly), OR
- `not_found`

Phase 6 stores:
- `record.resolution.ai_prompt`
- `record.resolution.ai_response`

### Task 6.2 — Gemini Flash Call
- Calls Gemini `models.generateContent` via REST
- Processes records in batches of 50 (`cfg.gemini_batch_size`)
- Sends **one request per record**
- Uses throttling + retries for 429/503
- Uses deterministic generation settings (temperature = 0)

### Task 6.3 — Decision Handling (“Go back to Phase Five”)
If Gemini selects a domain candidate:
- `record.resolution.resolved_domain = "<candidate>"`
- `record.resolution.ambiguity_flag = False`
- `record.resolution.reason = "ai_selected_domain"`
- `record.ai_used = True`
- `record.source = "ai"` (AI resolved the official website)
- Then Phase 6 **re-runs Phase 5** on the resolved subset only

If Gemini returns `not_found`:
- `record.resolution.resolved_domain = None`
- `record.resolution.ambiguity_flag = False`
- `record.resolution.reason = "ai_not_found"`
- `record.status = "not_found"` (configurable)
- `record.ai_used = True`
- `record.source = "ai"`

If the AI call fails (transient/network):
- record remains ambiguous (retryable)
- Phase 6 does not finalize status or ambiguity

If skipped due to quota:
- record remains ambiguous unchanged

---

## Outputs (Schema Fields Written)
Phase 6 writes:

- `record.resolution.resolved_domain`
- `record.resolution.ambiguity_flag`
- `record.resolution.reason`
- `record.resolution.ai_prompt`
- `record.resolution.ai_response`
- `record.ai_used` (True only when AI makes a decision)
- `record.source = "ai"` (when AI makes a decision)
- `record.status` (only for `not_found`, and optionally for resolved status)

---

## Status Policy (Configurable)
Domain resolution is **not** contact success.

If `cfg.phase6_set_status` is enabled:
- If AI selects a domain: set `record.status = cfg.phase6_status_resolved` (default `resolved`) when prior status was `ambiguous`
- If AI returns `not_found`: set `record.status = cfg.phase6_status_not_found` (default `not_found`)

Final `success` should be decided after crawling and validation (later phases).

---

## AI Quota (max_ai_fraction)
To keep AI usage below the cap, Phase 6 enforces:
```bash
allowed_n = int(cfg.max_ai_fraction * total_records)
```
Only the first `allowed_n` ambiguous records are sent to AI (deterministic order).
Skipped records remain ambiguous and unchanged.

---

## Environment Variables
Phase 6 reads the API key from:
1. `PipelineConfig.gemini_api_key`, or
2. `GEMINI_API_KEY`

Optional:
- `GEMINI_MODEL` (fallback model name when cfg.gemini_models is not set)

---

## Debug Keys
Phase 6 writes:
- `phase6_ai_allowed`
- `phase6_reason`
- `phase6_decision`
- `phase6_used_model`
- `phase6_candidate_count`
- `phase6_candidates_sample`
- `phase6_error` (only on failures)