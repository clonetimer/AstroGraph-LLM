# AstroGraph R35.5 Product Experience — Review Coverage Provenance and Session Reproducibility

Status: `CURRENT_R35_5_REVIEW_COVERAGE_PROVENANCE_AND_SESSION_REPRODUCIBILITY_COMPLETE`

Platform UI asset version: `v0.15.5`.

## Goal

R35.5 makes the browser-local R35.2–R35.4 review session reproducible at the material-scope level. Marking a review area as reviewed now captures the explicit projected identities and transient view scope that were present for that area at that moment. The resulting coverage record is included in the local handoff text so another operator can manually reconstruct what was in scope.

R35.5 does not create a persisted audit record, approval record, signature, release action, diagnosis conclusion, discrepancy-resolution result, EvidenceFact, Agent execution, Tool execution, graph query, simulation execution, or backend endpoint.

## Coverage-record contract

Coverage records use the browser-local marker `BROWSER_SESSION_SCOPE_V1` and exist only while the current page session exists. Each of the six R35.2 review areas has either no captured record or one captured record created when the operator marks that area reviewed.

Every captured record contains the current `presentation_sha256` and `source_bundle_sha256` plus area-specific explicit scope:

| Review area | Captured explicit material scope |
| --- | --- |
| `identity` | job identity, presentation identity, source-bundle identity/status, graph provenance, EvidenceBundle identity |
| `candidate` | frozen candidate, component, currently viewed candidate, frozen/transient candidate mode |
| `evidence` | EvidenceBundle identity, projected Evidence IDs/roles, current Evidence selection |
| `graph` | graph status/provenance, projected node identities, edge count, current node selection |
| `traces` | projected observed trace/channel identities, projected simulation trace/provider/candidate/source-trace identities, current observed/simulation view |
| `agent` | projected Agent receipt identities, Probe IDs, and receipt statuses |

The coverage key is a deterministic textual scope trace assembled only from those already-projected fields and the current transient UI selection. It is not a cryptographic signature and does not certify that the review was sufficient or correct.

## Coverage-context drift

For each area already marked reviewed, the browser recomputes the current material scope from the same explicit fields and compares it with the captured coverage record. A difference produces the session-only signal:

```text
REVIEW_COVERAGE_CONTEXT_CHANGED
```

The signal identifies that one or more reviewed areas now display a different scope from the scope that was captured when they were marked reviewed. Typical examples include browsing an alternative candidate after candidate-context review or changing a reviewed selection-sensitive view.

Coverage drift does not automatically clear a review mark, does not claim that prior review is invalid, and does not resolve or rank any diagnostic discrepancy. Returning the transient UI context to the captured scope removes the signal when the deterministic scope record matches again.

## Handoff reproducibility

The R35.1 handoff now includes a `Review coverage provenance (ephemeral)` section with:

- `Coverage contract: BROWSER_SESSION_SCOPE_V1`;
- one line for every review area, either the captured explicit scope or `NOT_CAPTURED`;
- the currently drifted review areas;
- `Coverage meaning: MATERIAL_SCOPE_TRACE_ONLY_NOT_AUDIT_OR_APPROVAL`.

This is intended for manual operator handoff and scope reconstruction. There is no import/restore API, no server-side session, and no claim that copied text is a signed or authoritative audit artifact.

## Review-session interaction

R35.5 preserves the R35.3 closure states:

```text
REVIEW_INCOMPLETE
REVIEWED_WITH_ATTENTION
REVIEWED_NO_EXPLICIT_ATTENTION
```

`REVIEW_COVERAGE_CONTEXT_CHANGED` participates only as another explicit browser-session attention signal. Therefore a fully reviewed session whose current scope differs from a captured reviewed scope is shown as `REVIEWED_WITH_ATTENTION`. The operator may use the existing R35.4 unresolved-item navigation to focus the first drifted coverage record; navigation itself still does not change review marks or resolve the signal.

Clearing the current review session clears both the six ephemeral review marks and all six coverage records. Re-rendering a new Diagnostic Workspace presentation also starts a new empty browser review session.

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

R35.5 requires:

- marking an area reviewed to capture a deterministic area-specific scope record from existing `presentation.v1` fields and transient selection only;
- unreviewed areas to remain `NOT_CAPTURED` in the handoff coverage section;
- changing reviewed candidate context to preserve the review mark while producing `REVIEW_COVERAGE_CONTEXT_CHANGED`;
- returning to the captured candidate context to remove the drift signal without mutating authority;
- unresolved-item navigation for the coverage signal to focus the recorded drift surface without clearing the signal or changing review state;
- no network call, browser persistence, Probe execution, Tool execution, simulator execution, approval, signature, release, or backend write to be introduced by the workspace module;
- maintenance/formal authority to remain unchanged after coverage capture and drift navigation;
- existing R34.2 production-contract, R35–R35.4 workspace, Operations Console, production JavaScript, documentation freshness, source-integrity, and R32 frozen-core gates to remain green.

## Claim boundary

R35.5 is a browser-local material-scope provenance and handoff-reproducibility increment. It does not prove substantive review sufficiency, reviewer identity, review time, audit integrity, discrepancy resolution, diagnosis correctness, release authorization, or qualification of any algorithm, model, dataset, software, HIL setup, or flight system.
