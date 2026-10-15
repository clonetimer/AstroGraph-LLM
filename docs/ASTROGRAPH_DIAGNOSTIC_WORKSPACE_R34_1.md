# AstroGraph R34.1 Diagnostic Workspace Production Contract Reachability

Status: `CURRENT_DIAGNOSTIC_WORKSPACE_R34_1_COMPLETE`

## Goal

R34.1 closes the production-contract gap between the R34 browser workspace and existing governed graph/simulation outputs. Platform UI version is `v0.14.1`.

The new read-only endpoint is:

```text
GET /v1/platform/jobs/{job_id}/maintenance/diagnostic-workspace
```

It returns `astrograph-diagnostic-workspace-presentation.v1`. This is a presentation projection, not a diagnosis, Evidence, or release contract.

## Authority snapshot

Every projection carries the current `diagnosis_disposition`, `release_status`, and canonical formal EvidenceBundle SHA256. A retained source bundle is bound to that snapshot. If authority changes after retention, projection fails closed instead of displaying stale graph or simulation context.

Workspace interaction remains ephemeral. Candidate, Evidence, graph-node, observed-trace, simulation-trace, and Agent-step selection never writes to the report, SQLite, EvidenceBundle, formal diagnosis, release state, or Agent trajectory.

## Graph reachability

An existing validated `SystemModelGraph` is retained by exact bytes and projected around the current diagnosis component using exact node identity. The bounded projection includes only explicit `CONNECTED_TO`, `COMMANDS`, `HAS_FAULT_MODE`, and `INSTANCE_OF` edges.

There is no fuzzy component match, LLM edge completion, Cypher input, graph mutation, or click-to-diagnose action. If the exact diagnosis component is absent, the graph status is `UNAVAILABLE`.

## Simulation reachability

R34.1 does not rerun a simulator. A producer may retain the exact `SimulationTrace` at the time an existing R29/R31 provider runs, together with its validated decision receipt. Retention verifies:

- the approved provider ID;
- candidate and diagnosis identity;
- decision-to-trace SHA256 equality;
- provider-specific trace provenance;
- exact channel/unit coverage;
- source-bundle self-hash.

The endpoint then normalizes those already-produced samples for display with semantics `SIMULATED_TRAJECTORY_NOT_PROBABILITY`.

Historical decision receipts that contain only a trace SHA256 cannot reconstruct numeric samples. They remain unavailable unless the exact original trace bytes were retained. The endpoint never reruns the provider and never fabricates a curve from fit or distance.

## Overlay states

The frontend distinguishes three states:

```text
no numerical simulation traces
-> 当前诊断没有可叠加的数值仿真轨迹

traces exist but none match the selected observed channel
-> 当前遥测通道没有匹配的仿真轨迹

compatible trace exists
-> show the Simulation selector and overlay
```

The selector is therefore controlled by channel-compatible traces rather than the global simulation-trace count.

## Positive-path verification

The R34.1 contract tests construct a content-bound ADAPT `SystemModelGraph` and run the existing `ADAPTWaterPumpReducedOrderProvider` to produce a real R31 `SimulationTrace`. The decision receipt is bound to that exact trace hash. Tests cover producer retention, endpoint projection, native SVG rendering, numerical overlay selection, tamper rejection, authority-drift rejection, and unchanged authority before/after browser interaction.

The localhost browser gate uses FastAPI, Uvicorn, Chromium, the production asset route, the production projection endpoint, and the production workspace module. These tests are transport and UI-contract evidence, not diagnostic-performance or external-validation evidence.

## Frozen boundary

R34.1 adds no diagnosis algorithm, physical model, Agent Probe, Tool, simulator parameter, EvidenceFact, release mutation, Bench feature, or external-validation claim. The R32 diagnosis, Evidence, evaluation, fusion, simulation, orchestration, and approved Probe core remain unchanged.

## Known historical limitation

R29/R31 decisions created before this retention contract store a trace digest but not the numeric trace samples. Those old receipts prove trace identity but are insufficient to render an overlay. R34.1 records this limitation instead of regenerating historical simulation output.

## Verification record

- focused static/browser/real-HTTP suite: 40 passed;
- full collection: 3635 tests, 0 collection errors;
- production JavaScript: 12 modules passed `node --check`;
- pre-package source integrity: 2854 files, 0 errors;
- documentation freshness: PASS;
- source-only delivery: `PASS_WITH_EXTERNAL_DEPENDENCIES`;
- strict delivery: expected FAIL only for the inherited eight historical external archives;
- R32 frozen-core diff: empty.

The full pytest suite was not executed for R34.1.