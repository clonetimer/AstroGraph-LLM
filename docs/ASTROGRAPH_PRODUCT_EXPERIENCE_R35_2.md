# AstroGraph R35.2 Product Experience — Review Session and Discrepancy-Aware UX

Status: `CURRENT_R35_2_REVIEW_SESSION_AND_DISCREPANCY_AWARE_UX_COMPLETE`

Platform UI asset version: `v0.15.2`.

## Goal

R35.2 extends the read-only Diagnostic Workspace with a browser-local review session that helps the next operator distinguish reviewed areas, unreviewed areas, and explicit material signals that may need attention. It does not create a new diagnostic discrepancy model and does not reinterpret the governed projection.

R35.2 does not add diagnosis logic, EvidenceFact production, Agent Probe execution, Tool execution, graph query capability, simulation execution, persistence, release actions, approval/signature semantics, or a backend endpoint.

## Ephemeral review session

Six review areas are exposed:

- review identity;
- candidate context;
- formal Evidence;
- mechanism graph;
- observed/simulated traces;
- Agent receipts.

An operator may mark an area reviewed. The marks are held only in JavaScript memory for the current rendered workspace. They are not written to localStorage/sessionStorage, the report, maintenance output, retained source bytes, EvidenceBundle, diagnosis, or release state. Rendering a new workspace resets the review session.

## Explicit attention signals

The workspace derives attention signals only from fields already present in `astrograph-diagnostic-workspace-presentation.v1` plus the current transient browser selection. Current signal classes are:

```text
SOURCE_BUNDLE_NOT_VERIFIED
CONTRADICTING_EVIDENCE_PRESENT
MISSING_EVIDENCE_DECLARED
AGENT_STEP_INCOMPLETE
MECHANISM_GRAPH_UNAVAILABLE
OBSERVED_TELEMETRY_UNAVAILABLE
NUMERIC_SIMULATION_UNAVAILABLE
TRANSIENT_CANDIDATE_BROWSE
SIMULATION_CANDIDATE_IDENTITY_MISMATCH
```

These are review-navigation signals, not fault classifications, probabilities, quality scores, release decisions, or qualification results. `CONTRADICTING` and `MISSING` are displayed only when those explicit Evidence roles are already present in the projection. Candidate-browse drift is explicitly identified as transient UI context.

## Handoff extension

The R35.1 copyable handoff now also records:

- review areas marked reviewed in the current page session;
- review areas still unreviewed;
- the current explicit attention-signal codes and their projection-derived descriptions.

The existing separation between `FROZEN_REPORT_CANDIDATE` and `TRANSIENT_ALTERNATIVE_BROWSE_ONLY` remains unchanged.

## Frozen boundary

The public workspace schema remains `astrograph-diagnostic-workspace-presentation.v1`; the retained-source bundle remains the R34.2 v2 contract. `src/astrograph/integrations/diagnostic_workspace_presentation.py` is unchanged.

The R32 frozen areas remain unchanged:

```text
src/astrograph/diagnosis/
src/astrograph/evidence/
src/astrograph/evaluation/
src/astrograph/fusion/
src/astrograph/simulation/
src/astrograph/orchestration/
src/astrograph/investigation/diagnostic_probes.py
```

## Verification contract

R35.2 requires:

- review marks to be page-memory-only and reset on new workspace render;
- explicit attention signals to be derived only from projection fields and transient selection;
- handoff text to expose reviewed/unreviewed areas and attention signals without mutating formal authority;
- no `fetch`, platform `api`, localStorage/sessionStorage, Probe execution, or simulator execution in the workspace module;
- R35.1 handoff identity and frozen/transient candidate separation to remain intact;
- existing production-contract, Operations Console, production JavaScript, documentation freshness, source-integrity, and R32 frozen-core gates to remain green.

## Claim boundary

R35.2 is an operator-review usability increment. It does not provide new diagnostic-performance evidence, discrepancy inference, statistical confidence, provenance guarantee beyond existing contracts, external validation, model qualification, HIL, safety, flight, or qualification evidence.
