# Maintenance Evidence Input Contract R9

## Purpose

R9 converts each currently unresolved formal evidence item into a machine-readable maintenance input contract. The contract is derived only from the serialized `EpisodeEvidenceBundle`, `EvidenceDefinition`, `FeatureDefinition`, `FeatureResult`, and `EvidenceFact` records already present in the governed report.

The contract does not collect telemetry, does not recompute a feature, and does not create a formal `EvidenceFact`. It is an admission/request specification for the next governed data or feature-processing step.

## Contract surface

Maintenance presentation schema:

`astrograph-maintenance-diagnosis.v7`

Machine endpoint:

```text
GET /v1/platform/jobs/{job_id}/maintenance/input-contracts
```

Response schema:

`astrograph-maintenance-input-contracts.v1`

## Per-gap fields

Each `EvidenceInputContract` binds at least:

- `evidence_definition_id`
- `feature_definition_id`
- frozen feature algorithm version
- source-field template
- currently resolved source field, when present
- required unit and observed unit
- feature window scope and governed time window
- required semantics declared by `FeatureDefinition`
- paired nominal requirement and bound nominal run identity when applicable
- input-data SHA256 when a `FeatureResult` already exists
- sample count and quality flags
- current formal fact status
- admission requirements and blockers

The contract is bound to `next_diagnostic_check.input_contract_id` so the operator can tell which structured input requirement corresponds to the current recommended action.

## Admission requirement examples

The R9 maintenance contract can expose requirements such as:

- `FEATURE_DEFINITION_BOUND`
- `SOURCE_FIELD_BOUND`
- `UNIT_MATCH`
- `WINDOW_BOUND`
- `SOURCE_HASH_BOUND`
- `SAMPLE_COVERAGE`
- `FEATURE_ALGORITHM_VERSION`
- `PAIRED_NOMINAL_BOUND`
- `FEATURE_VALUE_AVAILABLE`

Requirement states are explicit (`SATISFIED` / `UNSATISFIED`) and include a human-readable detail string.

## Missing evidence versus contradiction

R9 deliberately separates two operator workflows.

### Missing / invalid evidence

When a critical formal fact is missing or invalid, the contract status is normally:

`INPUT_REQUIRED`

The operator action is to satisfy the contract, rerun the frozen feature computation, and generate a new formal `EvidenceFact` through the governed evidence pipeline.

### Existing contradiction

When a contradiction already has a traceable `FeatureResult` with valid field binding, unit, window, hash, sample coverage, and algorithm version, the contract status can be:

`EXISTING_INPUT_READY_FOR_CONTRACT_REVIEW`

The operator is told to review provenance/unit/window/threshold semantics rather than recollecting the same data or repeatedly executing the same shadow Probe.

If the existing contradiction also has input-contract failures, the status becomes:

`INPUT_CONTRACT_REPAIR_REQUIRED`

## UI behavior

The next-diagnostic-check card shows the bound contract and allows the operator to copy the exact JSON. This is intended for handoff to a telemetry/data-ingest workflow without retyping channel names, units, or time windows.

## Boundary

The R9 input contract is a structured request specification only. It cannot bypass:

- artifact admission;
- telemetry schema/profile checks;
- unit checks;
- model/feature algorithm identity checks;
- source-hash binding;
- deterministic feature/evidence computation;
- evidence semantic compilation;
- release governance.

A contract being fully specified does not itself mean the missing evidence has been obtained or that a fault is supported.
