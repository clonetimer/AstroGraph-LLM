# R9 — Reviewed causal-context consistency (development candidate)

R8 baseline: independently reviewed `CAUSES` edges could be traversed if model and snapshot matched, but cross-document operating conditions were not executable predicates. R9 attaches *optional expert-reviewed applicability metadata*, intersects those constraints before each causal path extension, and explicitly distinguishes compatibility from in-flight verification.

## Only an independent reviewer can set context

`MaterialCaseMemory.decide_graph(..., reviewed_applicability=...)` accepts a reviewed context only on `CONFIRM`, with a reviewer from the instance's administrator-provided allowlist. Extractor `stage_graph_proposals` rejects `reviewed_applicability`; AI-generated fragment schemas cannot add it. Existing approval `truth_source_ref` remains required. The context adds its own independent `review_basis_ref` and is stored immutably as part of the approved snapshot and release hash. An invalid context must not consume the pending review decision.

```python
reviewed_applicability={
    "schema_version": "astrograph-reviewed-material-applicability.v1",
    "equipment_model": "SAT_PUMP_TEST_V1",
    "operating_modes": ["TRACKING"],
    "environment": {"THERMAL_REGIME": "HOT"},
    "parameter_ranges": [{"name":"SHAFT_TEMP", "unit":"DEGC", "min":10, "max":60}],
    "review_basis_ref":"independent bench report 2026"
}
```

This is an example of the interface, not an approved aircraft or satellite fact. All fields must be supplied for a new context object. The operating mode set must be explicit and nonempty. Environmental keys use a closed vocabulary. Parameter units are checked and normalized to comparable bases (e.g. degC→K, mA→A); unsupported units, NaN, infinities, mixed dimensions, conflicting equipment models and out-of-order ranges are rejected. Parameter names must refer to the *same physical location and measurement definition* across documents; this semantic alignment is still a human-review obligation. A record in which those measurement definitions differ should **not** reuse the same parameter name.

## Causal path behavior

- If two reviewed edges have mutually disjoint mode sets, conflicting environment categories or disjoint same-parameter bounds, the path is **not returned**; `blocked` includes `INCOMPATIBLE_CAUSAL_PATH_CONTEXT` and specific reason codes.
- Compatible reviewed predicates produce `REVIEWED_CONSTRAINTS_MUTUALLY_COMPATIBLE_NOT_FLIGHT_VERIFIED` with the intersection and all hop provenance. `joint_applicability_verified` remains `false` and no diagnostic vote or control is authorized.
- Old edges with no context remain queryable and receive `INCOMPLETE_REVIEWED_CONTEXT`, **not** `true` applicability. These paths must be described as unverified.
- Opposing positive/negative claims with provably disjoint reviewed conditions do not cancel each other; unknown, overlapping, invalid or ambiguous contexts remain conservative.
- Cross-document common context is computed under a *same-context comparison assumption*. If the mechanism describes transitions across different operating modes or evolving temperatures, this simple interval intersection can over-reject. Such stage-dependent/temporal propagation requires a later temporal model and must not be silently accepted as compatible.
- Reviewed `HYPOTHETICAL_OR_CONDITIONAL` cause claims still cannot establish full paths. R9 does not upgrade `condition_quote` into automatically satisfied structured predicates and does not perform free-text semantic entailment.

The new `reviewed_context_consistency` field is included under each eligible `multi_hop.causal_chain_review.chains[]` item and passed to the material explanation task. Neo4j publication (when explicitly enabled) retains a JSON representation of the reviewed context on material claim nodes; it does not change the active telemetry KG.

## Validation

Run the new suite plus R7/R8 and the prior material regression tests with `scripts/launch_logged_tests.py`; inspect `validation_logs/r9-final/output.log` and `done.json`. Fixtures use a generic laboratory pump. The previously disclosed SADA outcome is **not** included in the retrieval corpus, prompts, or expected fault labels.

**Boundary**: structural condition compatibility is not validation of physical causality, sensor measurements, operating history, or current mission fault. This candidate is not a flight-certified diagnosis system and has not undergone real unknown-root-cause accuracy trials.
