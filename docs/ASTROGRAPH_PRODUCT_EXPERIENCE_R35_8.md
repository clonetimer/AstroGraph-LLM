# AstroGraph R35.8 — Replay Delta Basis Diff and Explainable Re-review

Status: `CURRENT_R35_8_REPLAY_BASIS_DIFF_COMPLETE`

Platform UI asset version: `v0.15.8`.

R35.8 extends the browser-local R35.6/R35.7 incoming coverage replay with exact field-level comparison for areas whose overall replay state is `CHANGED`. It does not add causal diagnosis, discrepancy resolution, reviewer identity, persistent audit, approval, release authority, backend review storage, Agent action, Probe execution, simulator execution, graph query, or a new Diagnostic Workspace schema.

## Problem being closed

R35.6 classifies each review area as `MATCHED`, `CHANGED`, `NOT_REPRODUCED`, or `NOT_PROVIDED`, while R35.7 organizes the non-matching areas into a guided re-review queue. A `CHANGED` row, however, still required the operator to inspect the opaque deterministic coverage key to determine which exact basis field differed.

R35.8 exposes that difference mechanically from the already-governed incoming and current coverage `basis` arrays. It does not infer why the value changed.

## Field-level diff contract

For an overall `CHANGED` comparison only, the browser creates an ephemeral field table with four field states:

```text
SAME
VALUE_CHANGED
INCOMING_ONLY
CURRENT_ONLY
```

The comparison key is the exact text before the first `=` in each basis entry. The comparison value is the exact text after that `=`. The UI preserves the incoming/current values as text and does not normalize them into domain claims.

`SAME` is retained in the expanded view so the operator can see which identity dimensions did *not* change. `VALUE_CHANGED`, `INCOMING_ONLY`, and `CURRENT_ONLY` are the only field states counted as changed fields.

## No fabricated diff for non-comparable replay states

R35.8 deliberately does not produce a field-level basis diff for:

```text
NOT_REPRODUCED
NOT_PROVIDED
MATCHED
```

`NOT_REPRODUCED` already states that the prior material cannot be compared against a corresponding current material scope under the R35.6 reproduction rules. Turning that state into a synthetic field explanation would overclaim. `NOT_PROVIDED` has no incoming record. `MATCHED` has no replay delta requiring explanation.

## UI presentation

Each `CHANGED` replay row now includes an expandable `字段级 basis diff` table with:

- exact field name;
- field comparison state;
- previous/incoming value;
- current value.

The R35.7 guided re-review queue also shows only the names of changed fields for `CHANGED` items. This is a navigation aid; the full values remain in the expanded replay diff.

The handoff summary adds `Incoming changed basis fields` plus the explicit boundary marker:

```text
INCOMING_REPLAY_BASIS_DIFF_DISPLAY_ONLY_NOT_CAUSAL_EXPLANATION_OR_RESOLUTION
```

## Semantics and authority boundary

The field-level diff means only that exact imported and current coverage-basis text differs. It does **not** establish:

- why the material changed;
- whether the incoming or current material is correct;
- whether the change is diagnostically significant;
- whether a discrepancy was investigated or resolved;
- whether the review is sufficient;
- whether diagnosis or release should change.

Viewing or expanding the diff does not change R35.2 review marks, R35.3 closure, R35.5 coverage records, R35.6 comparison state, R35.7 `PENDING/REVISITED` trace, attention signals, maintenance authority, or formal diagnosis/release state.

No diff state is written to `localStorage`, `sessionStorage`, report, maintenance, EvidenceBundle, retained source bytes, Agent trajectory, diagnosis, or release state.

## Frozen boundary

R35.8 does not modify:

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

R35.8 adds explicit checks for:

- exact field diff on a candidate replay change;
- unchanged identity fields remaining visible as `SAME`;
- exact candidate view/mode values shown as `VALUE_CHANGED`;
- graph selection changes reporting only `selected_node` as the changed field;
- R35.7 queue rows surfacing changed field names without altering original replay status;
- handoff reporting the changed field set and the non-causal boundary marker;
- `NOT_REPRODUCED` and `NOT_PROVIDED` producing no fabricated field-level diff;
- no network, storage, maintenance-authority, review-state, or replay-state mutation from viewing the diff.

## Claim boundary

R35.8 proves only deterministic browser-local display of exact coverage-basis key/value differences for already-classified `CHANGED` incoming replay areas. It does not explain causality, establish diagnostic significance, resolve discrepancies, validate reviewer actions, create an audit record, authorize diagnosis or release, or qualify any algorithm, model, dataset, software, HIL setup, or flight system.
