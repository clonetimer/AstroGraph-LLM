# R10: Audited phase-aware causal reference chains

**Baseline:** R9 candidate. **Status:** R10 development candidate, not flight-qualified.

R9 required a common intersection of all reviewed physical/operating-mode constraints. That intentionally rejected contradictory reference contexts but could also exclude a legitimate sequence of changing modes or temperatures. R10 provides an **opt-in, independently approved temporal stage annotation on each CAUSES edge**. Any old chain without annotations keeps the R9 static policy. Incomplete or invalid annotations do not bypass R9.

## Privileged review interface

Only an administrator-authorized independent reviewer may call `MaterialCaseMemory.decide_graph(..., decision="CONFIRM", reviewed_applicability=..., reviewed_temporal=...)`. The proposal staging interface rejects `reviewed_temporal`, preventing the LLM from self-approving its time assumptions. The outer reviewed relation still carries the original text, scope, approval version, source hash, truth reference and reviewer.

The following is a **synthetic laboratory pump example**, not evidence about an actual satellite:

```python
reviewed_temporal = {
    "schema_version": "astrograph-reviewed-causal-temporal.v1",
    "equipment_model": "SAT_PUMP_TEST_V1",
    "process_id": "PUMP_BENCH_SEQUENCE",
    "stage_index": 0,
    "pre_context": { ... },   # R9 reviewed_applicability schema, source-stage
    "post_context": { ... },  # R9 reviewed_applicability schema, transition result
    "lag_seconds": {"min": 2, "max": 5},
    "minimum_source_dwell_seconds": 1,
    "review_basis_ref": "independent temporal experimental record"
}
```

Each `pre_context` and `post_context` is a complete R9 context object with independent source-basis traceability. The `pre_context` must **not widen** the approved edge `reviewed_applicability`. All stages on a chain must be annotated, use the same process identifier, advance by exactly one integer stage, and have compatible adjacent `post_context -> pre_context` predicates. Finite nonnegative lag and minimum dwell must satisfy `min <= max` and `dwell <= max`. Bounded cumulative relative delay intervals are returned to help an expert plan follow-up checks; these are **not** flight-observed times.

### Compatibility and strict limits

- With *zero* temporal annotations: R9 same-context intersection and old `INCOMPLETE_REVIEWED_CONTEXT` semantics remain unchanged.
- With *some* temporal annotations: mixed-timeline splices are blocked; unreviewed transitions never waive static constraints.
- With *all* stages independently annotated: each stage is validated against its own reviewed applicability, and successive stage states are checked for overlap. Different stages may legitimately have different modes, environmental categories or physical parameter bounds. A same-time static intersection may be empty and is retained as an audit field, but it does not veto a separately reviewed stage sequence.
- Incompatible states, wrong stage order, different process identifiers, scope widening, unknown units, invalid temporal records or unresolved negative/conditional claims block path approval.
- All outputs retain `current_root_cause_proven=false`, `joint_applicability_verified=false`, `diagnostic_vote_allowed=false` and `automatic_control_authorized=false`. Independent source approvals do not attest spacecraft telemetry, synchronized events, a verified transition, or the actual root cause.
- A reviewer must verify that parameter identifiers refer to the **same physical quantity and sensor location** across source documents. Automatic ontology or timeseries alignment is not implemented.

`evaluate_causal_chains` returns `reviewed_temporal_consistency` including mode, transition receipts, bounded cumulative lag, reasons for rejection and explicit flight-verification flags. `reviewed_context_consistency` is still emitted for backward compatibility; `static_same_context_assumption_not_applied` explains why its *static* conflict does not override a fully reviewed sequence.

Reviewed metadata is stored in the graph approval snapshot, included in checksum/HMAC-protected releases, checked on release verification and optionally published under isolated Neo4j material claim properties. The active flight-diagnosis KG and upstream fault ranking are unchanged.

## Validation

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_foundation_r10_temporal_causal.py tests/test_foundation_r9_reviewed_applicability.py
python scripts/launch_logged_tests.py start --timeout 600 r10-smoke -- python -m pytest -q tests/test_foundation_r10_temporal_causal.py
python scripts/launch_logged_tests.py status r10-smoke
```

For long tests use distinct job names, inspect `validation_logs/<job>/output.log`, then `done.json`. The completion marker can be watched locally or in CI, but does not automatically push a ChatGPT message.

No disclosed real fault's root cause was included in R10 tests, prompts or a retrieval index. Verification uses a generic synthetic pump example with **no diagnosis score claim**. Runtime flight event-alignment and root-cause evaluation remain separate future work.
