# AstroGraph R35.4 Product Experience — Review Navigation and Unresolved-Item Drill-down

Status: `CURRENT_R35_4_REVIEW_NAVIGATION_AND_UNRESOLVED_ITEM_DRILLDOWN_COMPLETE`

Platform UI asset version: `v0.15.4`.

## Goal

R35.4 makes the R35.2/R35.3 browser-local review state actionable for navigation without converting it into an approval or remediation workflow. Unreviewed review areas and explicit attention signals can now locate the already-rendered read-only material that caused or contextualizes the item.

R35.4 does not add diagnosis logic, discrepancy inference, EvidenceFact production, Agent Probe execution, Tool execution, graph query capability, simulation execution, persistence, signature/approval semantics, release actions, or a backend endpoint.

## Review navigation contract

The review-session panel adds a `未关闭项定位` surface. It contains navigation entries for:

- each review area that is not marked reviewed in the current page session;
- each explicit R35.2 attention signal currently derived from the production projection and transient UI context.

Navigation is deliberately weaker than resolution. Activating a navigation entry may move focus, switch the existing graph/overlay main view, or select an already-present Evidence item for inspection. It does not mark a review area as reviewed, clear an attention signal, change the R35.3 closure state by itself, or persist any state.

## Explicit drill-down mapping

R35.4 uses a fixed mapping from existing review/signal identities to existing UI material:

| Review/signal identity | Navigation target |
| --- | --- |
| `identity` / `SOURCE_BUNDLE_NOT_VERIFIED` | review identity / provenance block |
| `candidate` / `TRANSIENT_CANDIDATE_BROWSE` | candidate-context control |
| `evidence` | first projected Evidence item, or Evidence support panel when unavailable |
| `CONTRADICTING_EVIDENCE_PRESENT` | first projected `CONTRADICTING` Evidence item |
| `MISSING_EVIDENCE_DECLARED` | first projected `MISSING` Evidence item |
| `graph` / `MECHANISM_GRAPH_UNAVAILABLE` | graph main view |
| `traces` / telemetry/simulation attention signals | telemetry overlay main view |
| `agent` | first projected Agent receipt, or support panel when unavailable |
| `AGENT_STEP_INCOMPLETE` | first Agent receipt whose status is not `COMPLETED` |

This mapping does not rank Evidence, infer the cause of a discrepancy, or choose a diagnostic action. When a signal refers to explicit Evidence or an Agent receipt, the target is selected only to make the existing projected material visible.

## Focus and accessibility behavior

The context, main, and support regions are programmatically focusable for drill-down. After a navigation action, the browser moves focus to the corresponding existing control or region. The navigation surface remains keyboard-operable and uses normal buttons; no hidden execution control is introduced.

## Review-state invariants

R35.4 preserves the R35.3 closure semantics:

```text
REVIEW_INCOMPLETE
REVIEWED_WITH_ATTENTION
REVIEWED_NO_EXPLICIT_ATTENTION
```

Navigation does not itself modify any of the six ephemeral review marks. An attention signal remains visible after drill-down unless the underlying transient UI context itself changes through an existing user selection. There is no “resolve”, “acknowledge”, “approve”, “close case”, “sign”, or “release” action.

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

R35.4 requires:

- navigation from an unreviewed area to leave that area unreviewed;
- Evidence attention signals to drill down only to matching explicit projected Evidence roles;
- incomplete-Agent attention to drill down only to an existing projected incomplete Agent receipt;
- transient candidate navigation to preserve the frozen-report/transient-browse distinction;
- no navigation action to remove an attention signal by itself;
- no `fetch`, platform `api`, localStorage/sessionStorage, Probe execution, or simulator execution in the workspace module;
- maintenance/formal authority to remain unchanged after review navigation;
- existing R34.2 production-contract, R35–R35.3 workspace, Operations Console, production JavaScript, documentation freshness, source-integrity, and R32 frozen-core gates to remain green.

## Claim boundary

R35.4 is a read-only navigation/usability increment over already-projected review material. It does not prove that an unreviewed or attention item has been resolved, that review was substantively sufficient, that a diagnosis is correct, that release is authorized, or that any model, dataset, software, HIL setup, or flight system is qualified.
