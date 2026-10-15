# AstroGraph R35.3 Product Experience — Review Session Closure and Operator Traceability

Status: `CURRENT_R35_3_REVIEW_SESSION_CLOSURE_AND_OPERATOR_TRACEABILITY_COMPLETE`

Platform UI asset version: `v0.15.3`.

## Goal

R35.3 closes the browser-local review-session loop introduced in R35.2. It makes review coverage and still-visible attention items explicit in the handoff surface without creating an approval workflow, operator decision record, release gate, or backend persistence.

R35.3 does not add diagnosis logic, EvidenceFact production, Agent Probe execution, Tool execution, graph query capability, simulation execution, persistence, signature/approval semantics, release actions, or a backend endpoint.

## Derived review-closure state

The Diagnostic Workspace derives one of three page-session states from the six existing review marks and the R35.2 attention signals:

```text
REVIEW_INCOMPLETE
REVIEWED_WITH_ATTENTION
REVIEWED_NO_EXPLICIT_ATTENTION
```

`REVIEW_INCOMPLETE` means at least one of the six browser-local review areas is not marked reviewed. `REVIEWED_WITH_ATTENTION` means all six areas are marked reviewed while one or more explicit attention signals remain visible. `REVIEWED_NO_EXPLICIT_ATTENTION` means all six areas are marked reviewed and no R35.2 attention signal is currently derived from the projection/transient selection.

These values describe review-session coverage only. They are not diagnostic dispositions, evidence conclusions, approvals, signatures, release decisions, quality scores, or statements that an attention item is resolved.

## Closure visibility

The review-session panel now shows:

- the derived session-closure code;
- a plain-language coverage description;
- the remaining unreviewed areas when review is incomplete;
- whether explicit attention signals remain after all six areas have been reviewed.

There is intentionally no “approve”, “close case”, “sign”, “release”, or persisted completion action. Removing a review mark immediately returns the session to `REVIEW_INCOMPLETE`. A transient candidate browse or other explicit R35.2 signal can move a fully reviewed session to `REVIEWED_WITH_ATTENTION`.

## Handoff traceability

The copyable handoff is extended with:

- `Review closure status`;
- `Open review areas`;
- `Open attention signal codes`;
- the fixed semantic marker `SESSION_TRACE_ONLY_NOT_APPROVAL`.

This makes the local review coverage reproducible in copied text while preserving the existing distinction between frozen report identity and transient browser context. The handoff is still generated locally and is not written back to AstroGraph.

## Frozen boundary

The public workspace schema remains `astrograph-diagnostic-workspace-presentation.v1`; the retained-source bundle remains `astrograph-diagnostic-workspace-source-bundle.v2`. `src/astrograph/integrations/diagnostic_workspace_presentation.py` is unchanged.

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

R35.3 requires:

- closure state to be derived only from ephemeral R35.2 review marks plus existing explicit attention signals;
- no persisted or backend review-completion record;
- a fully reviewed session with no attention signal to report `REVIEWED_NO_EXPLICIT_ATTENTION`;
- a fully reviewed session with any attention signal to report `REVIEWED_WITH_ATTENTION`;
- any unreviewed area to report `REVIEW_INCOMPLETE`;
- copied handoff text to carry closure status/open items without mutating maintenance/formal authority;
- no `fetch`, platform `api`, localStorage/sessionStorage, Probe execution, or simulator execution in the workspace module;
- existing R34.2 production-contract, R35/R35.1/R35.2 workspace, Operations Console, production JavaScript, documentation freshness, source-integrity, and R32 frozen-core gates to remain green.

## Claim boundary

R35.3 is a read-only operator-review traceability increment. It does not prove that a human review was substantively sufficient, that attention items were resolved, that a diagnosis is correct, that release is authorized, or that any model/dataset/system is qualified.
