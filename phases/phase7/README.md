# Phase 7 — Post-validation (No LLM)

## Goal
Increase output quality and reduce false positives by validating extracted contacts without using AI.

Phase 7 validates and cleans the outputs of Phase 5 (and Phase 6 → Phase 5 recrawl), then writes the final validated contacts into `record.validated_contacts`.

---

## Inputs
Primary inputs (from earlier phases):

- `record.contacts.emails` (Phase 5 extracted emails, ordered list)
- `record.contacts.phone` (optional)
- `record.contacts.crawl_status` (`success`, `failed`, `blocked`, `timeout`, `not_started`)

---

## Tasks

### Task 7.1 — Email Syntax Validation
Validates email format using an RFC-style regex (similar to production validators such as Django).

Rules:
- normalize email (trim, lowercase, remove trailing punctuation)
- reject invalid syntax
- reject emails longer than 254 characters

### Task 7.2 — MX Record Check
Checks whether the email domain has MX records to confirm the domain can receive email.

Behavior:
- optional (controlled by config)
- uses timeouts and limited retries
- returns one of:
  - `True`  → MX exists
  - `False` → no MX / NXDOMAIN / NoAnswer
  - `None`  → unknown (dnspython missing, timeout, resolver failure)

Optional strict mode:
- if `cfg.phase7_require_mx = True`, Phase 7 keeps only emails with `mx == True`.

### Task 7.3 — Deduplication
Removes duplicate emails per record deterministically.

Rule:
- stable dedupe (keeps the first occurrence)
- dedupe key is normalized email

---

## Outputs (Schema-aligned)
Phase 7 writes into the `Contacts` object:

- `record.validated_contacts.emails` (List[str], ordered)
- `record.validated_contacts.phone` (copied from `record.contacts.phone`)
- `record.validated_contacts.crawl_status` (copied from `record.contacts.crawl_status`)

Dropped records (`status` starts with `dropped`) are skipped.

---

## Debug Keys
If `record.debug` exists, Phase 7 adds:

- `phase7_validated_emails_n` — number of validated emails kept
- `phase7_email_details` — list of per-email checks:

```json
  [
    {"email": "info@example.com", "domain": "example.com", "syntax_valid": true, "mx": true},
    {"email": "bad@", "domain": null, "syntax_valid": false, "mx": null}
  ]
````

* `phase7_reason` — set when skipped (e.g., dropped records)

---

## PipelineConfig Settings Used

* `phase7_enable_syntax_validation: bool` (default `True`)
* `phase7_enable_mx_check: bool` (default `True`)
* `phase7_enable_deduplication: bool` (default `True`)
* `phase7_require_mx: bool` (default `False`)

MX behavior:

* `phase7_mx_timeout_s: float` (fallback: `cfg.dns_timeout_s`, default ~6.0)
* `phase7_mx_retries_on_timeout: int` (fallback: `cfg.dns_retries_on_timeout`)
* `phase7_mx_retry_backoff: float` (fallback: `cfg.dns_retry_backoff`)

Output cap:

* `phase7_max_validated_contacts_per_record: int | None`

    * if set, Phase 7 stops after collecting this many validated emails (before dedupe)

---

## Determinism and Idempotency

* Phase 7 resets `record.validated_contacts` on every run to avoid stale values.
* Email normalization and dedupe are deterministic.
* MX results may vary if DNS conditions change, but the logic is deterministic for the same DNS responses.
* No AI is used.
