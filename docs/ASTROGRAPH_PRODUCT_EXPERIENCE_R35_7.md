# AstroGraph R35.7 — Incoming Replay Delta Queue and Guided Re-review

Status: `CURRENT_R35_7_INCOMING_REPLAY_DELTA_QUEUE_COMPLETE`

Platform UI asset version: `v0.15.7`.

R35.7 extends the browser-local R35.6 incoming handoff coverage replay with a guided re-review queue. It does not add a backend review record, discrepancy-resolution workflow, audit trail, reviewer identity, approval, diagnosis rule, EvidenceFact, Agent action, simulator execution, graph query, release gate, or persistent operator state.

## Problem being closed

R35.6 can compare the previous operator's captured coverage scope with the current Diagnostic Workspace and classify each review area as `MATCHED`, `CHANGED`, `NOT_REPRODUCED`, or `NOT_PROVIDED`. A receiving operator could navigate each row, but there was no explicit browser-local queue for working through the non-matching areas without conflating navigation with resolution.

R35.7 adds a guided queue over the existing comparison. The queue contains only:

```text
CHANGED
NOT_REPRODUCED
NOT_PROVIDED
```

`MATCHED` areas do not enter the queue.

## Queue semantics

Each queue item keeps its R35.6 comparison state and adds only a browser-session revisit trace:

```text
PENDING
REVISITED
```

`PENDING` means the current queue item has not been explicitly marked as re-viewed at its current comparison fingerprint.

`REVISITED` means the receiving operator explicitly marked that exact current comparison fingerprint as re-viewed during the current browser session.

A `REVISITED` marker does **not**:

- turn `CHANGED`, `NOT_REPRODUCED`, or `NOT_PROVIDED` into `MATCHED`;
- state that a discrepancy was resolved;
- state that the prior or current review is correct, sufficient, accepted, or rejected;
- change the current R35.2 review mark;
- change R35.3 closure status;
- clear an R35.2/R35.5 attention signal;
- create approval, signature, release, or diagnosis authority.

## Comparison fingerprint binding

A re-review acknowledgement is bound to the exact browser-local comparison fingerprint:

```text
<area>|<comparison-status>|incoming=<incoming-coverage-key>|current=<current-coverage-key>
```

The fingerprint is deterministic from the existing R35.6 comparison inputs. If the current projected/transient material scope changes and therefore produces a different fingerprint, the old acknowledgement is not applied to the new scope. The previous acknowledgement may remain in browser memory for the exact fingerprint it was created against, but only an exact fingerprint match renders the item `REVISITED`.

This prevents a prior click from being silently reused after a candidate, graph node, trace, Evidence selection, Agent scope, or other coverage basis changes.

## Guided navigation

The `接手重新查看队列` panel shows:

- pending count and total queue size;
- queue items in canonical six-area review order;
- the original R35.6 comparison status;
- `PENDING` or `REVISITED` revisit state;
- previous/next guided cursor controls;
- read-only navigation to the current review area;
- an explicit toggle to mark or unmark the exact current fingerprint as re-viewed.

Navigation reuses the R35.4 review-navigation contract. It can change focus, the existing main view, or existing transient selection only. It does not automatically mark the current R35.2 review area as reviewed and does not resolve the replay comparison.

## Independence from current review state

The queue remains separate from:

- `workspaceReviewSession` review marks;
- R35.5 `workspaceReviewCoverage` records;
- R35.3 closure state;
- R35.2/R35.5 attention signals;
- formal diagnosis and release authority.

Loading a new incoming replay resets queue cursor and re-review acknowledgements. Clearing the incoming replay removes all queue state. Rendering a new Diagnostic Workspace also clears replay and queue state.

No queue state is written to `localStorage`, `sessionStorage`, report, maintenance, EvidenceBundle, retained source bytes, diagnosis, Agent trajectory, or release state.

## Handoff extension

The local handoff now includes:

- incoming re-review queue entries with original comparison status and `PENDING`/`REVISITED` state;
- the list of pending re-review areas;
- the boundary marker:

```text
INCOMING_REPLAY_REREVIEW_TRACE_ONLY_NOT_RESOLUTION_OR_APPROVAL
```

This is an operator continuity aid only. It is not an audit record, discrepancy disposition, review-signoff artifact, or authorization token.

## Frozen boundary

R35.7 does not modify:

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

R35.7 adds explicit checks for:

- queue inclusion of only `CHANGED`, `NOT_REPRODUCED`, and `NOT_PROVIDED` areas;
- exclusion of `MATCHED` areas from the queue;
- explicit `PENDING` to `REVISITED` browser-local acknowledgement;
- original R35.6 comparison state remaining unchanged after acknowledgement;
- current R35.2 review marks, closure, attention signals, maintenance authority, storage, and network activity remaining unchanged;
- exact comparison-fingerprint binding of acknowledgements;
- current material-scope changes returning a queue item to `PENDING` when its fingerprint differs;
- previous/next guided queue navigation;
- handoff queue/pending output and explicit non-resolution boundary marker;
- replay clear removing all queue cursor/acknowledgement state.

## Claim boundary

R35.7 proves only deterministic browser-local organization and revisit tracing of non-matching incoming replay comparison items. It does not prove that a discrepancy was investigated adequately, explain why material changed, establish reviewer identity, preserve a legally admissible audit trail, prove substantive equivalence of reviews, authorize diagnosis or release, or qualify any algorithm, model, dataset, software, HIL setup, or flight system.
