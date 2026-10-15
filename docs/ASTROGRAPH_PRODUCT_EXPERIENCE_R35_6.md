# AstroGraph R35.6 — Incoming Handoff Coverage Replay

Status: `CURRENT_R35_6_INCOMING_HANDOFF_COVERAGE_REPLAY_COMPLETE`

Platform UI asset version: `v0.15.6`.

R35.6 extends the browser-local R35.5 review-coverage provenance into an incoming-session comparison workflow. It does not add a backend review record, audit trail, approval workflow, diagnosis rule, EvidenceFact, Agent action, simulator execution, graph query, release gate, or persistent operator state.

## Problem being closed

R35.5 can record what material scope an operator had visible when each of six review areas was marked reviewed, and can place those records in the local handoff. A receiving operator still had to compare that handoff manually against the current Diagnostic Workspace.

R35.6 adds an explicit browser-local replay comparison. The receiving operator can paste either:

1. an R35.6 `BROWSER_REVIEW_COVERAGE_REPLAY_V1` manifest; or
2. an R35.5-style handoff containing the per-area `Coverage <area> (...)` lines.

The browser parses the supplied text locally and compares only the prior captured material scope with the current read-only `presentation.v1` projection and current transient UI scope.

## Replay manifest

The browser-local replay contract is:

```text
BROWSER_REVIEW_COVERAGE_REPLAY_V1
```

The canonical handoff block is delimited by:

```text
BEGIN_REVIEW_COVERAGE_REPLAY_V1
...
END_REVIEW_COVERAGE_REPLAY_V1
```

The payload contains:

- source `job_id`;
- source `presentation_sha256`;
- source `source_bundle_sha256`;
- one nullable record for each of the six review areas;
- each captured record's deterministic `basis` and `coverage_key`.

The manifest contains no reviewer identity, timestamp, signature, free-form diagnosis, approval state, or mutation request. Input is limited to 100000 characters. A manifest whose supplied `coverage_key` does not equal the deterministic join of its `basis` is rejected fail closed.

## Incoming comparison states

Each review area is classified independently as one of:

```text
MATCHED
CHANGED
NOT_REPRODUCED
NOT_PROVIDED
```

Semantics:

- `MATCHED`: the incoming captured deterministic coverage key exactly equals the current deterministic coverage key.
- `CHANGED`: the incoming and current scopes are both comparable, but their deterministic coverage keys differ.
- `NOT_REPRODUCED`: the incoming record names explicit material that has no comparable current projected object for that area, such as an incoming available graph when the current projection has no graph nodes.
- `NOT_PROVIDED`: the incoming manifest has no captured coverage record for that area.

These states compare material scope only. They do not state that a prior review was correct, sufficient, complete, valid, invalid, accepted, rejected, or resolved.

## Backward-compatible R35.5 handoff import

If pasted text is not valid replay JSON, the browser accepts the R35.5 handoff coverage lines only when at least one recognized `Coverage <area> (...)` line is present. It reconstructs deterministic basis arrays from those lines and records the import format as:

```text
R35_5_HANDOFF_COVERAGE_LINES
```

Unrecognized text is rejected rather than inferred.

## UI behavior

The Review Session now contains an `交接范围复现对照` panel. It provides:

- a textarea for the incoming handoff or replay manifest;
- `对照当前工作区` to parse and compare locally;
- `复制当前 coverage manifest` to create an outgoing R35.6 replay block;
- `清除对照` to discard incoming replay state;
- per-area status rows and navigation to the corresponding current workspace area.

Navigation reuses the R35.4 read-only review navigation. It can change focus, main view, or existing transient selection only. It does not mark an area reviewed and does not resolve a comparison state.

## Independence from current review state

Incoming replay state is separate from:

- `workspaceReviewSession` review marks;
- R35.5 `workspaceReviewCoverage` records;
- R35.3 closure status;
- R35.2 attention signals.

Loading a replay manifest does not add an attention signal and does not change `REVIEW_INCOMPLETE`, `REVIEWED_WITH_ATTENTION`, or `REVIEWED_NO_EXPLICIT_ATTENTION`. It also does not automatically restore a previous candidate, Evidence selection, graph node, trace, or Agent step.

Incoming replay data exists only in current JavaScript memory. It is cleared when a new Diagnostic Workspace is rendered and is not written to `localStorage`, `sessionStorage`, report, maintenance, EvidenceBundle, retained source, diagnosis, Agent trajectory, or release state.

## Handoff extension

The R35.6 handoff includes:

- the outgoing delimited replay manifest;
- whether an incoming replay is loaded;
- per-area incoming scope comparison states when loaded;
- the boundary marker:

```text
INCOMING_SCOPE_COMPARISON_ONLY_NOT_AUDIT_OR_APPROVAL
```

This is designed for manual operator continuity, not for machine authorization.

## Frozen boundary

R35.6 does not modify:

```text
src/astrograph/diagnosis/
src/astrograph/evidence/
src/astrograph/evaluation/
src/astrograph/fusion/
src/astrograph/simulation/
src/astrograph/orchestration/
src/astrograph/investigation/diagnostic_probes.py
src/astrograph/integrations/diagnostic_workspace_presentation.py
```

The public Diagnostic Workspace schema remains `astrograph-diagnostic-workspace-presentation.v1` and the retained source schema remains `astrograph-diagnostic-workspace-source-bundle.v2`.

## Verification additions

R35.6 adds explicit checks for:

- current coverage replay manifest generation and delimiter output;
- replay import with captured areas producing `MATCHED` and uncaptured areas producing `NOT_PROVIDED`;
- R35.5 handoff-line backward-compatible import;
- transient candidate scope producing `CHANGED` without creating a new attention signal;
- missing current graph material producing `NOT_REPRODUCED`;
- replay-area navigation remaining read-only;
- tampered `coverage_key` rejection;
- replay clear removing browser-local input/state;
- no fetch, storage, maintenance, diagnosis, EvidenceFact, retained-source, or release mutation from replay comparison.

## Claim boundary

R35.6 proves only deterministic browser-local comparison between an operator-supplied prior material-scope record and the currently visible read-only Diagnostic Workspace scope. It does not prove reviewer identity, continuity of personnel, integrity of copied text outside the application, substantive review equivalence, discrepancy resolution, audit admissibility, approval, release authorization, diagnosis correctness, or qualification of any algorithm, model, dataset, software, HIL setup, or flight system.
