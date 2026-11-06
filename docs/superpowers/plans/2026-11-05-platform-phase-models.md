# Platform P3 Explicit Follow-up Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Allow users of the full AstroGraph Mission platform to pin a distinct, approved follow-up LLM profile to each imported external-diagnosis analysis job, preserving current defaults and tenant boundaries.

**Architecture:** Persist profile IDs and a digest of the profile's non-secret execution identity with the job at creation. On any model use, validate the identity and build a StageRouter around the existing provider backends without modifying shared service configuration. Mission UI offers an optional follow-up model, and preflight invalidates stale selections.

**Tech Stack:** Python 3.13, FastAPI, existing Pydantic + PlatformStateRepository, StageRouter, native JavaScript and HTML. No Node runtime requirement, no new third-party libraries.

**Spec:** User-approved P3 stage-model configuration following P0/P1, P2; original diagnosis results remain immutable.

## Global Constraints
- Only the external diagnosis → LLM analysis → report/follow-up workflow; no diagnostic computation or automatic model fallback.
- Preserve the existing single-model default.
- Model selection uses existing administrator-approved LLM profiles; no arbitrary URL, API key, or client-owned permissions.
- Never persist endpoint, credentials or raw request payload in job stage bindings. Detect changed/missing/disabled stage profiles before invocation.
- Do not claim the complete multi-process task queue is delivered by this focused change.

## Review Focus
- Profile edited after job creation: fail closed, do not change models silently.
- Disabled/unvalidated profile: reject job creation and/or invocation.
- Existing single-model jobs: output and follow-up behavior unchanged.
- Stale UI selection after an engine list refresh: clear selection and invalidate preflight.
- Full job report must not leak stage profile secrets.

### Task 1: Persist validated stage binding
**Files:** `src/astrograph/integrations/upload_report.py`; `tests/test_p3_platform_phase_models.py`.
- [x] RED: tests for accepted follow-up profile, rejected disabled/unknown profile, source-identity hash binding.
- [x] GREEN: add strict per-job phase map and identity hash; legacy defaults empty.
- [x] Verify tests and commit.

### Task 2: Resolve router for analysis and follow-up
**Files:** `src/astrograph/integrations/upload_report.py`, `src/astrograph/api.py`, `tests/test_p3_platform_phase_models.py`.
- [x] RED: full flow uses base analysis backend and selected follow-up backend; changed profile fails without fallback.
- [x] GREEN: wrap per-job backend with existing StageRouter and bind at API create.
- [x] Verify regression suite and commit.

### Task 3: Wire Mission UI and regression
**Files:** `src/astrograph/resources/mission_console.html`, `src/astrograph/resources/mission_console.js`, `src/astrograph/resources/mission_console.css`, `tests/ui/test_p3_platform_phase_ui.py`.
- [x] RED: optional select present, choices from known available profiles, changes invalidate preflight, creation request includes selected ID.
- [x] GREEN: implement accessible optional follow-up selector, no unimplemented menus.
- [x] Run targeted tests, core external diagnosis regression, and syntax check; commit.

### Task 4: Reproducible handoff
**Files:** `docs/PORTABLE_OBSERVABILITY.md`, current plan and tests.
- [ ] Run focused platform and portable regressions; capture real outcomes and limits.
- [ ] Produce source patch and snapshot with checksums; attempt remote sync via supported connection only.
- [ ] Clearly state whether native Win7/Win10 or production-model tests were run.
