# AstroGraph R34 Diagnostic Workspace

Status: `CURRENT_DIAGNOSTIC_WORKSPACE_R34_PHASE1`

## Goal

R34 turns the stable R33.2 result screen into one read-only diagnostic workspace. An operator can move from a frozen candidate to formal Evidence, a structured propagation graph, observed-versus-simulated telemetry, and recorded Agent investigation steps without changing diagnostic authority.

Platform UI version: `v0.14.0`.

## Workspace layout

The workspace keeps three linked regions inside the existing result page:

- fault context: subsystem, component, fault mode and candidates;
- analysis surface: observed/simulated traces or the structured mechanism graph;
- support context: formal supporting, contradicting and missing Evidence plus recorded Agent steps.

No new primary navigation route is introduced. The existing result tabs and technical report remain available.

## Ephemeral selection contract

The browser owns only this transient selection state:

```text
candidateId
componentId
evidenceId
graphNodeId
observedTraceId
simulationTraceId
```

Selection is discarded on report reload. It is not written to SQLite, the report, `EpisodeEvidenceBundle`, diagnosis, release state or Agent trajectory. Candidate selection therefore never promotes a candidate to the formal diagnosis.

## Graph boundary

The renderer accepts only structured `mechanism_graph.nodes` and `mechanism_graph.edges` already present in the maintenance presentation. When nodes are absent or the graph is unavailable, the UI states that no interactive graph data exists.

The graph is read-only. R34 adds no Cypher input, graph editing, LLM edge completion or click-to-diagnose action.

## Observed and simulated telemetry boundary

Observed curves come from the existing visualization response. Simulated curves are rendered only when the governed simulation review already contains explicit numeric `numerical_traces` or `traces` samples. The browser never executes a simulator and never synthesizes a curve from a fit score.

When no numeric simulation trace exists, the UI states that only a mechanism-consistency result is available. It also preserves the explicit semantic boundary:

```text
simulation fit / distance ≠ fault probability
```

## Evidence and Agent boundary

The support pane uses only the formal Evidence summary supplied by the maintenance contract. Missing Evidence remains “正式证据摘要不可用”; waveform statistics, graph content, simulation output and Agent findings do not manufacture Evidence.

Agent steps are links to already recorded material. Clicking a step changes workspace context only and does not call a Probe. A new Probe still requires the existing explicit operator action and approved `{probe_id}` request contract.

## Implementation surface

The new `platform_diagnostic_workspace.js` module owns selection state, the native SVG graph, trace overlay and cross-panel links. `platform_operations.js` exposes a hook from the existing result projection, while `platform_report.js` passes the visualization object already loaded for that report. No new API or backend mode was added.

The R32 frozen diagnosis, Evidence, evaluation, fusion, simulation, orchestration and approved Probe code remain unchanged.

## Verification contract

R34 tests require:

- candidate, Evidence, graph and Agent clicks to leave diagnosis, release and Evidence content unchanged;
- missing structured graph data and missing numeric simulation traces to fail closed;
- no probability presentation for simulation fit/distance;
- no Cypher or simulator-parameter editor;
- Agent-step navigation to make no Probe request;
- 1366×768 at 160% font scale to have no page-level horizontal overflow;
- a real FastAPI/Uvicorn/Chromium upload → Preflight → diagnosis → result → workspace → graph path;
- production JavaScript syntax, documentation freshness, source integrity and frozen-core diff gates.

## Claim boundary

R34 is an operations-experience release. It adds no diagnostic algorithm, mechanism model, Evidence authority, Agent Probe, Tool, calibration, benchmark result or external-validation claim.
