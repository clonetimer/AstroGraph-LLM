# AstroGraph full-platform P3: explicitly selected follow-up model

Version scope: 2026-11-05 feature increment on top of `2.8.0-dev.2`.

## User workflow

1. In the Mission external-diagnosis workbench, select the analysis engine as before.
2. Optionally choose a second **validated** LLM profile under “证据追问使用的模型”. Only already-approved profiles appear.
3. Prepare the diagnosis report and submit the analysis. The initial structured analysis remains on the primary engine. When answering a follow-up, the system routes just that call to the selected follow-up profile.
4. If you leave the optional selector blank, both phases continue using the same model, exactly as before. No automatic provider fallback or model judgment call is introduced.

Each new task stores a phase-engine ID and digests of the selected profile's non-secret execution identity. It does **not** store the newly selected stage endpoint, key, or request payload. Existing platform metadata may already contain a primary-model endpoint; this change does not retrospectively redact that legacy metadata.

If an administrator disables, removes, or changes either pinned profile, preflight blocks the job or follow-up returns `MODEL_CONFIGURATION_UNAVAILABLE`. The user must create a new task with a current approved configuration. Past original uploads and successful reports remain unchanged.

## Scope and limitations

- This release implements a distinct **follow-up** model in the complete Mission platform. The portable app's previously implemented chunk/synthesis/follow-up stage configuration remains separate.
- The complete platform does **not** yet offer separate chunk/synthesis model selectors, a durable cross-process bounded queue, safe cross-process cancellation, or automatic recovery after service crashes.
- The complete platform's existing rule engine, telemetry workflows, account permissions, and external diagnosis sources are unaffected.
- Invalid profile selections return a sanitized HTTP 422 rather than raw provider configuration errors.
- Existing jobs (no `phase_engine_ids`) retain the previous single-model behavior.

## Verification commands

```bash
PYTHONPATH=src python -B -m pytest -q --astrograph-test-profile source-only tests/test_p3_platform_phase_models.py tests/ui/test_p3_platform_phase_ui.py
node --check src/astrograph/resources/mission_console.js
```

The wider source-only regression and portable regression are listed in the accompanying delivery report. No native Windows 7/10 hardware verification or real provider billing/cost verification is implied.

## 2026-11-05 continuation: explicit long-report stages

The Mission workbench additionally exposes **optional** verified LLM profiles
for `chunk` (per-piece review of original text) and `synthesis` (global
comparison of validated piece summaries). `follow_up` remains independently
selectable. All three selectors default to the same primary analysis model;
there is no automatic provider fallback or model-quality judge.

For **external-diagnosis text totaling more than 60,000 characters** the full
platform now shares the portable pipeline's bounded source-aware chunk
analysis and hierarchical synthesis. For short reports, the primary model is
called only once, including when the optional long-stage selectors are filled
in. A long report can make multiple paid requests and the number of chunks,
overlap, source coverage and bound citations are kept in the report. The current
limit is 180,000 normalized source characters, up to 24 chunks.

For new jobs, all selected stage IDs and non-secret execution fingerprints
are pinned before the first model call. Before any stage runs, a profile that
has changed, been disabled, removed or lost verification causes a **blocked
preflight**. Existing jobs without stage pins use their previous behavior.
Long-report checkpoints include the full staged model identity and original
input/context fingerprints. The operator may explicitly retry; the platform
must never silently reuse checkpoints from a changed model or switch a local
model to an external service.

The full platform still has **no durable multi-process queue**, coordinated
cross-process cancellation, or automatic recovery of paid requests after a
worker crash. CI and source-only tests are not native Win7/Win10 acceptance;
the already published portable ZIPs are unchanged.
