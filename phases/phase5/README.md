# Phase 5 — Web Crawling and Data Extraction

## Goal
Extract contact signals (emails and optionally phone) from the official website for each record, using a small, deterministic, and resource-bounded crawler.

## Inputs
- Primary: `record.resolution.resolved_domain` (from Phase 4)
- Fallback: `record.valid_domains[0]` (from Phase 3), only when Phase 4 is not ambiguous

## Tasks

### Task 5.1 — Target URL Selection
Build a deterministic set of high-yield URLs under the domain:
- `/`
- `/contact`, `/about`, `/legal`, `/impressum`
- plus common variants (configurable via assets)

The crawler always tries:
- `https://<domain>` first, then `http://<domain>`

### Task 5.2 — Async Fetch (Bounded)
Fetch pages with strict caps and safety limits:
- maximum pages: `cfg.crawl_max_pages` (hard-capped to 5)
- maximum depth: `cfg.crawl_max_depth` (hard-capped to 2)
- per-page byte cap: `cfg.crawl_max_bytes_per_page`
- total byte cap per domain: `cfg.crawl_total_byte_limit`
- global concurrency: `cfg.crawl_global_concurrency`
- per-domain concurrency: `cfg.crawl_per_domain_concurrency`
- per-domain delay: `cfg.crawl_per_domain_delay_s`
- request timeout: `cfg.crawl_timeout_s`

### Task 5.3 — Extraction
Extract from each fetched page:
- emails via regex and `mailto:` links
- emails and phones via JSON-LD (`application/ld+json`)
- optional phones via regex if enabled: `cfg.crawl_extract_phone`

### Task 5.4 — Filtering
Filtering policy (deterministic):
- always remove `noreply`-like mailboxes
- prefer official-domain emails if any exist
- within official-domain, prefer role-based locals (info, contact, sales, ...)
- drop free-provider emails (gmail/yahoo/...) if any non-free exists
- otherwise keep the remaining emails

## Outputs
- `record.contacts.emails` (ordered, filtered)
- `record.contacts.phone` (optional; selected deterministically)
- `record.contacts.crawl_status` in:
    - `not_started`, `success`, `blocked`, `timeout`, `failed`

## Status and Source Policy
Phase 5 is the first phase allowed to mark a record as a contact-level success.

- Phase 5 sets `record.status = cfg.phase5_status_success` (default: `success`) **only if** at least one contact was extracted:
    - `len(record.contacts.emails) > 0` **or**
    - `record.contacts.phone` is not `None`

- Phase 5 sets `record.source = "crawler"` **only if** contacts were extracted (same condition as above).

Notes:
- Phase 4 should not mark domain-only resolution as `success`. Domain resolution is not equivalent to successfully finding contacts.
- Phase 5 does not change status for dropped records.
- If `cfg.phase5_set_status` is disabled, Phase 5 will not modify `record.status`.

## Debug Keys
- `phase5_policy` (audit of key knobs used)
- `phase5_selected_domain`
- `phase5_seed_urls`
- `phase5_fetched_pages`
- `phase5_pages_debug` (per-page status/bytes/error)
- `phase5_total_bytes_read`
- `phase5_emails_raw_count`, `phase5_emails_filtered_count`
- `phase5_phones_raw_count`
- `phase5_reason`

## Assets (assets/phase5/)
- `target_paths.txt`
- `free_email_providers.txt`
- `role_local_parts.txt`
- `no_reply_locals.txt`

If assets are missing, the phase uses safe built-in defaults.

## Determinism and Idempotency
- The crawl queue order is deterministic (seed order + deterministic link insertion).
- URL canonicalization drops query strings and fragments to prevent crawl explosions.
- Contacts outputs are reset at the start of each run to avoid stale results.
- For the same input state, network conditions, and config, the crawl behavior is stable.
