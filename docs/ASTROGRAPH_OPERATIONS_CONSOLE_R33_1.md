# AstroGraph Operations Console R33.1

Status: `CURRENT_R33_1_STABLE_UI_CLOSEOUT`

R33.1 is a stabilization patch for Platform UI `v0.13.1`. It preserves the R32 spacecraft diagnosis core and the R33 authority boundaries.

## Corrected behavior

- Server-recognized input values preserve boolean `false`; missing inputs render as missing instead of falling through to another field or becoming an object string.
- A blocked Preflight remains on Wizard step 2. Step 3 becomes active only when the server returns `can_start=true` for the current file signature.
- The Agent page renders one detailed investigation list. The compact timeline remains in the result-page Agent tab.
- User-facing modes are `标准诊断`, `增强诊断`, and `扩展排查（高级）`. The creative backend mode is available only inside advanced settings.
- Direct navigation to `#knowledge` loads the current job, report, and maintenance contract before rendering graph and simulation material.
- Initial capability loading primes the local platform repository before parallel secondary settings requests, preventing first-page SQLite initialization races.
- `platform_report.js` imports every core symbol it uses, so behavior is identical under real ES Modules and the legacy classic-script browser harness.

## Real HTTP release gate

`tests/ui/test_platform_operations_console_http_e2e.py` starts the actual FastAPI application through Uvicorn on localhost and opens `/platform` with Chromium. It verifies real overview/tool/capability requests, uploads a minimal truth-blind public case, runs Preflight, starts and polls the task, renders the result, and opens `#knowledge` in a new page that retrieves the actual maintenance contract.

The public case is a transport and UI integration fixture. Its diagnostic result is not research performance evidence.

## Authority boundary

- Pipeline execution success is not diagnosis success.
- Agent Probe output is not formal EvidenceFact.
- Simulation fit is not fault probability.
- The browser still submits only `{probe_id}` to the governed Probe endpoint.
- Graph access remains read-only and offers no arbitrary Cypher input.
- Simulation offers no arbitrary function, channel, window, threshold, or algorithm execution.

## Core freeze

R33.1 changes Platform resources, the UI asset registration, UI tests, and current documentation. It does not change diagnosis, evidence, evaluation, fusion, simulation, orchestration, or `investigation/diagnostic_probes.py` business logic.

