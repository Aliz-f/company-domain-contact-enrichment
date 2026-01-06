# Phase 3 — Fast Domain Validation (DNS + HTTP)

## Goal
Validate candidate domains from Phase 2 using fast network checks, and keep an ordered list of domains that look like real company websites.

## Inputs
- `record.domain_candidates` (ordered list from Phase 2)
- Config timeouts and concurrency from `PipelineConfig`

## Tasks

### Task 3.1 — DNS Validation
For each candidate domain:
- Sanitize and validate domain syntax
- Resolve A/AAAA via `getaddrinfo`
- Optional wildcard probe using a deterministic random subdomain
- Apply policy: reject wildcard DNS domains if configured

**Outputs**
- `record.debug["phase3_dns"]`
- Intermediate set of `dns_alive_domains`

### Task 3.2 — HTTP Validation
For each DNS-alive domain:
- Try `https://` then `http://`
- Use `HEAD` first (cheap)
- Optionally fetch a small `GET` snippet for parked-page detection
- Reject domains that redirect to directory/listing hosts (LinkedIn, Facebook, etc.)
- Optionally allow protected sites (401/403) as “valid” if configured

**Outputs**
- `record.debug["phase3_http"]`

### Task 3.3 — Produce Ordered Valid Domains
Keep only domains that pass both DNS and HTTP validation, preserving the original candidate order.

**Output**
- `record.valid_domains` (ordered)

## Outputs
- `record.valid_domains`
- Debug keys:
    - `phase3_reason`
    - `phase3_dns`
    - `phase3_http`
    - `phase3_policy`

## Assets (assets/phase3/)
- `parked_keywords.txt`
- `directory_host_hints.txt`

If assets are missing, Phase 3 uses safe built-in fallbacks.

## Config Knobs (common)
- DNS:
    - `dns_timeout_s`
    - `dns_retries_on_timeout`
    - `dns_retry_backoff`
    - `dns_limit`
    - `phase3_enable_wildcard_probe` (fallback: `enable_wildcard_dns_default`)
    - `phase3_reject_wildcard_dns`
- HTTP:
    - `http_timeout_s`
    - `http_user_agent`
    - `http_max_connections`, `http_max_keepalive_connections`
    - `phase3_http_concurrency`
    - `phase3_allow_protected_status`
    - `phase3_head_retry_statuses`
    - `phase3_snippet_on_success`
    - `phase3_http_range_end`
    - `phase3_http_snippet_bytes`
- Runner:
    - `phase3_record_batch_size`

## Idempotency
Phase 3 resets `record.valid_domains` and Phase-3 debug keys each run, so rerunning produces consistent results for the same inputs and network conditions.

## Notes
- This phase performs network I/O, so results can vary if a site changes, rate-limits, or is temporarily down.
- Phase 4 handles deterministic “single vs ambiguous” resolution based on `valid_domains`.
