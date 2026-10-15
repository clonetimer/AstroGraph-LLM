# AstroGraph R35 Product Experience — Diagnostic Review Readability

Status: `CURRENT_R35_PRODUCT_EXPERIENCE_PHASE1_COMPLETE`

Platform UI asset version: `v0.15.0`.

## Goal

R35 begins the post-R34 product-experience phase without reopening the frozen spacecraft diagnosis core. Phase 1 makes the existing read-only Diagnostic Workspace easier to review by exposing what governed material is available, what transient context the operator has selected, and what each graph/trace view actually contains.

R35 does not introduce a diagnosis algorithm, EvidenceFact, Agent Probe, Tool, graph query, simulation provider, release action, benchmark result, or persistence contract.

## Operator experience changes

The Diagnostic Workspace now adds a projection-derived material overview for:

- structured mechanism graph availability and node count;
- observed telemetry count;
- retained numeric simulation-trace count;
- formal Evidence link count;
- recorded Agent-step count.

The overview does not infer missing content from the maintenance page. If the production workspace projection is unavailable, it states that the workspace material is unavailable and does not reconstruct graph, simulation, Evidence, or Agent context in the browser.

A selection summary now exposes the current candidate, Evidence item, graph node, observed trace, and simulation trace. `恢复默认上下文` and the `Escape` key reset only the in-page selection object.

## Graph readability

Selecting a graph node now displays:

- the projected node label and exact node identity;
- the number of explicit edges in the current bounded projection that touch the node;
- any `related_channels` already present on that projected node.

These values are derived only from the existing `mechanism_graph.nodes` and `mechanism_graph.edges` presentation fields. No graph traversal service, Cypher execution, edge completion, or diagnosis mutation is introduced.

## Trace readability

The observed/simulated panel now displays the selected observed channel, unit, time range, axis labels, and the existing channel-compatible simulation selector.

The browser continues to render only numeric samples already present in the verified workspace projection. It does not run a simulator and does not derive a curve from fit/distance. The semantic boundary remains:

```text
simulation fit / distance != fault probability
```

## Accessibility and navigation

The two main workspace views expose tab roles and selected state. Candidate, Evidence, and graph-node selection expose pressed/selected state. The existing graph keyboard activation remains, and `Escape` restores the default transient context.

At 1366x768 with the existing 160% font-scale band, the workspace must remain free of page-level horizontal overflow.

## Frozen boundary

R35 Phase 1 changes only Platform UI resources, UI tests, current documentation, and UI asset registration. The following R32 frozen areas remain unchanged:

```text
src/astrograph/diagnosis/
src/astrograph/evidence/
src/astrograph/evaluation/
src/astrograph/fusion/
src/astrograph/simulation/
src/astrograph/orchestration/
src/astrograph/investigation/diagnostic_probes.py
```

The public workspace schema remains `astrograph-diagnostic-workspace-presentation.v1`. No backend endpoint or retained-source schema change is required.

## Verification contract

R35 Phase 1 requires:

- existing R34.2 production-contract tests to remain green;
- material overview to reflect only the supplied workspace projection;
- context reset to leave maintenance/formal authority objects byte-for-byte unchanged in the browser fixture;
- graph detail to use only explicit projected edges and related-channel references;
- trace context to show channel/unit/time range without probability semantics;
- workspace code to contain no `fetch`, platform `api`, storage write, Probe execution, or simulator execution;
- production JavaScript syntax, documentation freshness, source integrity, and R32 frozen-core diff gates to remain green.

## Claim boundary

R35 Phase 1 is a product-experience/readability increment over already-governed R34.2 material. It provides no new diagnostic-performance evidence, external validation, model qualification, probability calibration, or flight qualification.
