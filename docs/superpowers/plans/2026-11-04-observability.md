# AstroGraph P0/P1 Implementation Plan

**Goal:** Restore a common source baseline and ship one usable observability increment.
**Architecture:** stdlib JSON run journal under each job; context-bound transport observer; immutable per-run report copies; existing workbench API and six-page frontend.
**Spec:** ../specs/2026-11-04-observability-approved.md

## Global constraints
Keep full repository __init__ exports; only staged distributions use reduced exports. Common code targets Python 3.8 syntax/runtime where in portable closure. No credentials, raw prompts/answers, provider exception messages in journal. Retain input SHA binding and no arbitrary HTTP destinations. Existing user packages remain unchanged.

## Task 1: canonical baseline
- Add imported regression cases first; demonstrate failures against repository baseline.
- Reconcile V2.7.1 portable UI / parser fixes and Win7 compatibility without overwriting full modules. Preserve latest repository improvements.
- Run portable suite plus full repository tests, record baseline/optional dependency failures. Commit.

## Task 2: durable per-run telemetry
- Add tests for event redaction, actual HTTP attempts including failed/format-repair retries, nullable usage, source inventories, safe artifacts, restart interruption, legacy reads, concurrency/task separation. Observe RED.
- Implement integrations/execution_observability.py; instrument llm/openai_compatible._post only when a run context is active. Integrate workbench analysis/follow-up and long-report stage events.
- Preserve each run report and receipt; retry creates a new run, not a rewrite of old artifacts.
- Run suites, commit.

## Task 3: API and result view
- Add /api/jobs/{id}/execution safe summary and allowlisted per-run artifact downloads, using existing session/origin guards.
- Add material inventory, execution timeline, per-attempt usage and immutable report download cards to existing results page. Escape all external text; no fake placeholders.
- Add JS/browser/API regressions; validate old jobs and stale response isolation. Commit.

## Task 4: delivery and verification
- Self-review whole diff; repair important findings with regressions.
- Verify common source staging with Win10 and Win7 dependency profiles, final tests, source ZIP, git patch/bundle and inventory hashes.
- Attempt authorized remote submission only through available supported tools; never claim remote write/CI/Windows runtime without evidence.

## Review focus
Full disk or crash during journal writes; delayed response after task selection; untrusted provider text; malformed restored run artifacts/path aliases; partial usage and unsuccessful generation with successful HTTP.

## Execution status (2026-11-04)
Tasks 1–3 implemented and tested. Task 4 verified locally; artifact publication and remote submission are tracked in PORTABLE_OBSERVABILITY_VERIFICATION.md. P2/P3 are not included. One atomic P0/P1 commit is used because the profile/renderer/source assembly changes must travel together.
