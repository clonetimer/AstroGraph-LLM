# AstroGraph R35.1 Product Experience — Review Identity and Operator Handoff

Status: `CURRENT_R35_1_REVIEW_IDENTITY_AND_OPERATOR_HANDOFF_COMPLETE`

Platform UI asset version: `v0.15.1`.

## Goal

R35.1 extends the R35 read-only Diagnostic Workspace with explicit review identity and a copyable operator-handoff summary. It makes already-present provenance and authority identifiers visible without changing the production projection schema, retained source schema, or any diagnostic authority.

R35.1 does not add diagnosis logic, EvidenceFact production, Agent Probe execution, Tool execution, graph query capability, simulation execution, persistence, release actions, or a new backend endpoint.

## Review identity

The workspace now exposes only fields already carried by `astrograph-diagnostic-workspace-presentation.v1`:

- `job_id`;
- `presentation_sha256`;
- `source_bundle_status` and `source_bundle_sha256`;
- `mechanism_graph.provenance_sha256`;
- `authority_snapshot.evidence_bundle_sha256`;
- frozen selected candidate, diagnosis disposition and release status.

If the production workspace projection itself is unavailable, the browser does not reconstruct these identities from maintenance/report content.

## Simulation-source identity

The existing browser normalization now preserves, for display/handoff only, the already-projected simulation trace fields `provider_id`, `diagnosis_id`, `candidate_id`, and `source_trace_sha256`. Numeric sample rendering is unchanged. No hash is treated as a probability, quality score, or qualification result.

## Operator handoff

A local text summary is generated from the current read-only projection and ephemeral UI selection. It includes production identity, formal authority snapshot state, frozen candidate, effective observed/simulated trace identity, and current browsing context.

The summary explicitly distinguishes:

```text
FROZEN_REPORT_CANDIDATE
TRANSIENT_ALTERNATIVE_BROWSE_ONLY
```

Copy uses the browser clipboard only. It does not call AstroGraph APIs, write local/session storage, mutate report/maintenance objects, or persist selection state.

## Frozen boundary

R35.1 changes only Platform UI resources, UI tests, current documentation, and UI asset registration. The following R32 frozen areas remain unchanged:

```text
src/astrograph/diagnosis/
src/astrograph/evidence/
src/astrograph/evaluation/
src/astrograph/fusion/
src/astrograph/simulation/
src/astrograph/orchestration/
src/astrograph/investigation/diagnostic_probes.py
```

The public workspace schema remains `astrograph-diagnostic-workspace-presentation.v1`; the internal retained-source bundle remains the R34.2 v2 contract.

## Verification contract

R35.1 requires:

- review identity to use only projection fields and fail closed when the projection is unavailable;
- handoff text to keep frozen candidate identity separate from transient alternative browsing;
- simulation-source identity to originate only from already-projected fields;
- copying the handoff to leave maintenance/formal authority objects unchanged and issue no network request;
- no `fetch`, platform `api`, localStorage/sessionStorage, Probe execution or simulator execution in the workspace module;
- existing R35 workspace, Operations Console and production-contract tests to remain green;
- production JavaScript syntax, documentation freshness, source integrity and R32 frozen-core diff gates to remain green.

## Claim boundary

R35.1 is an operator-review and handoff usability increment. It creates no new diagnostic-performance evidence, provenance guarantee beyond the existing hashes/contracts, external validation, model qualification, probability calibration, HIL, safety, flight or qualification claim.
