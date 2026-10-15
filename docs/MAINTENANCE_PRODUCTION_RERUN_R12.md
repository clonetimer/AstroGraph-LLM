# R12 Maintenance Raw-Input Production-Path Rerun

## Purpose

R12 extends the maintenance recovery path only where AstroGraph can prove that supplemental data can be represented as a new governed **raw input artifact**. It does not reinterpret a maintenance FeatureResult capsule as raw telemetry.

The supported executable mode is:

`PUBLIC_CASE_RAW_TELEMETRY_PATCH_RERUN`

A successful child is a new immutable PUBLIC_CASE job with a newly materialized telemetry CSV and manifest. The source job remains frozen.

## PUBLIC_CASE raw-input binding

A PUBLIC_CASE is eligible only when its immutable `manifest.json` explicitly declares:

```json
{
  "maintenance_raw_input_binding": {
    "schema_version": "sat-sim.astrograph-maintenance-raw-input-binding.v1",
    "source_telemetry_sha256": "<sha256 of the manifest-declared telemetry CSV>",
    "time_field": "time_s",
    "full_window_replacement_required": true,
    "patchable_fields": [
      {"field": "eps.battery.soc", "unit": "1"}
    ]
  }
}
```

The binding is opt-in. Older PUBLIC_CASE packages without it remain `BLOCKED` with `RAW_INPUT_BINDING_UNAVAILABLE`.

The server rejects a binding when the source telemetry hash does not match, the time field differs from `time_s`, full-window replacement is not required, a patchable field has no unit, a field is duplicated, or the field is an answer-proxy/prohibited field.

## Patch contract and capsule

For every explicitly bound field the server emits an `astrograph-maintenance-production-raw-input-contract.v1` containing:

- source manifest and telemetry SHA256;
- field and unit;
- exact source timeline identity and sample count;
- replacement scope `FULL_SOURCE_TIMELINE`;
- whether the field already exists in the source CSV.

The operator/client may submit only an `astrograph-maintenance-production-raw-input-patch.v1` capsule with:

- current contract ID;
- exact bound field;
- exact unit;
- the complete source `time_s` vector;
- one finite numeric value for every timestamp;
- an optional source note.

The server does not accept a partial time window, changed timestamps, diagnosis labels, FeatureResults, EvidenceFacts or release decisions.

## Materialization and child creation

After admission the server builds a new immutable PUBLIC_CASE input:

1. re-read and re-verify the original uploaded file receipts;
2. revalidate the PUBLIC_CASE manifest and telemetry hashes;
3. apply the admitted full-timeline field replacement/addition;
4. write a new telemetry CSV;
5. update the child manifest telemetry file/hash/fields/units;
6. carry forward the raw-input binding with the new child telemetry SHA256;
7. add `maintenance_raw_patch_provenance`;
8. run ordinary platform preflight on the child;
9. create a PREPARED production-rerun receipt;
10. execute the child through the normal PUBLIC_CASE diagnosis path when requested.

No source file or source report is rewritten.

## Post-run completion receipt

After the child finishes, R12 rechecks:

- frozen source report identity;
- frozen source manifest SHA256;
- frozen source telemetry SHA256;
- actual child manifest SHA256 versus PREPARED receipt;
- actual child telemetry SHA256 versus PREPARED receipt;
- child artifact kind and execution status.

A trusted `astrograph-maintenance-production-rerun-completion-receipt.v1` is emitted only when every identity still matches.

The completion receipt records `raw_telemetry_rerun_performed=true`, but still keeps `formal_release_eligible=false`. The current PUBLIC_CASE path is a laboratory/simulation diagnosis path and this workflow does not create production flight-release authority.

## Equivalent-rerun idempotency

The complete UI/platform distribution prevents accidental duplicate PUBLIC_CASE raw reruns for the same materialized patch identity:

- readiness blocks an existing equivalent non-failed child;
- child creation takes an atomic content-identity claim before materialization;
- an explicitly `FAILED`/`CANCELLED`/missing previous child may be retried;
- source-side rerun history joins PREPARED and COMPLETED receipts.

## Immutable upload-receipt enforcement

R12 also hardens `_safe_expand()`: every expansion rechecks the current `input/` file set, file sizes and SHA256 values against the immutable upload receipts stored in `job.json`.

Direct filesystem replacement, addition or deletion of an uploaded input therefore fails closed before artifact expansion. This is recorded as `BASE-R12-01` because the missing generic recheck predates R12.

## PIPELINE_RUNTIME boundary

`PIPELINE_RUNTIME` is already a derived upstream runtime artifact. AstroGraph cannot truthfully turn a maintenance feature capsule into new raw telemetry/runtime inputs.

R12 therefore returns `UPSTREAM_RECOMPUTE_ADAPTER_REQUIRED` and a machine-readable `astrograph-maintenance-pipeline-runtime-recompute-request.v1` that freezes:

- source runtime SHA256;
- source mapping-bundle SHA256;
- input/required runtime schema;
- required adapter contract `sat-sim.diagnostic-pipeline-raw-recompute-adapter.v1`;
- current maintenance input-contract IDs;
- `derived_runtime_patch_allowed=false`.

The upstream producer must rebuild runtime/local outputs from governed raw telemetry and resubmit them through the normal PIPELINE_RUNTIME admission path.

## OAG boundary

OAG maintenance reruns continue to use the R11 `OAG_EVIDENCE_BRANCH_REPLAY` workflow. R12 does not conflate OAG evidence-branch replay with raw telemetry rerun.

## Platform endpoints

- `GET /v1/platform/jobs/{job_id}/maintenance/production-rerun/raw-input-contracts`
- `GET /v1/platform/jobs/{job_id}/maintenance/production-rerun/raw-input-contracts/{contract_id}/template`
- `POST /v1/platform/jobs/{job_id}/maintenance/production-rerun/raw-input-contracts/{contract_id}/admit`
- `GET /v1/platform/jobs/{job_id}/maintenance/production-rerun-readiness`
- `POST /v1/platform/jobs/{job_id}/maintenance/production-rerun`
- `GET /v1/platform/jobs/{job_id}/maintenance/production-reruns`

## Claim boundary

R12 can claim that an explicitly bound PUBLIC_CASE field can be replaced over the exact frozen raw-telemetry timeline, materialized as a new immutable PUBLIC_CASE child, and re-executed through the ordinary PUBLIC_CASE path with prepared/completion identity receipts.

R12 does **not** claim that old PUBLIC_CASE packages without raw bindings are patchable, that PIPELINE_RUNTIME can be rebuilt inside AstroGraph without an upstream raw-recompute adapter, that maintenance feature capsules are raw telemetry, or that the resulting child grants formal flight-release authority.
