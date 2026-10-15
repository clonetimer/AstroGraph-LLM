# AstroGraph Operations Console R33.2

Status: `CURRENT_OPERATIONS_CONSOLE_STABLE_UI_BASELINE`

R33.2 is the semantic closeout for Platform UI `v0.13.2`. It freezes the Operations Console information architecture without changing the R32 diagnosis core, Agent authority, tools, algorithms, or Bench scope.

## Human-review contract

The frontend preserves the backend `requires_human_review` field as a three-state value:

- `true` → `是`
- `false` → `否`
- absent/null → `未提供`

The UI does not infer a formal human-review requirement from `ABSTAIN`, `UNKNOWN`, `INSUFFICIENT`, `AMBIGUOUS`, or a blocked release. Those states may still place a task in the operator-attention queue.

`Operator attention required` and formal `requires_human_review` are separate contracts.

## Verification

Static, offline Chromium, inherited browser behavior, and the real localhost FastAPI/Uvicorn workflow verify the three states. The real HTTP public-case result specifically verifies that an absent field is displayed as `未提供` rather than `否`.

The public case validates transport and UI integration only. It is not diagnostic-performance evidence.

## Frozen boundary

R33.2 changes only the Platform UI status projection, labels, asset version, tests, and current documentation. Diagnosis, Evidence, evaluation, fusion, simulation, orchestration, approved Probe behavior, tool catalog, and Bench behavior remain unchanged.

