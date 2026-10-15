# LLM Method Confirmatory R9 Runbook

## Purpose

R9 adds a self-contained materialization step for the frozen 38-case A0–A7 confirmatory denominator. The materialized snapshot is intended for reproducible engineering execution outside the source tree and contains only the frozen case manifest, both dataset sample indices, and payload bytes actually referenced by the selected fault/nominal pairs.

R9 also keeps the existing boundary: content materialization and chmod-based write-bit removal are not substitutes for a kernel-enforced read-only mount. `--require-readonly-datasets` remains the strict admission gate.

## 1. Prepare and freeze the denominator

Use an external experiment directory:

```bash
python scripts/experiments/prepare_llm_method_confirmatory_campaign.py \
  --output-dir /experiments/llm-confirmatory-r9 \
  --adcs-root /datasets/adcs-v5.2.2 \
  --phm-root /datasets/phm-v6.0 \
  --freeze-case-manifest \
  --vllm-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen3.5-9B
```

The preparation directory contains at least:

- `case_manifest.json`
- `readiness.json`
- `dataset_mount_receipt.json`
- `dataset_snapshot_receipt.json`
- `launch_plan.json`

`launch_plan.json` now also contains `snapshot_materialization_argv`, an exact command for creating a self-contained frozen snapshot.

## 2. Materialize the content-addressed snapshot

Run the generated command, or invoke the materializer directly:

```bash
python scripts/experiments/materialize_llm_method_confirmatory_snapshot.py \
  --snapshot-receipt /experiments/llm-confirmatory-r9/dataset_snapshot_receipt.json \
  --case-manifest /experiments/llm-confirmatory-r9/case_manifest.json \
  --output-root /experiments/llm-confirmatory-r9-materialized
```

The output root must be outside the AstroGraph source tree and must not be nested inside either source dataset root.

The materializer fails closed if any frozen source byte changed after the snapshot receipt was created.

Because snapshot identity is path-independent, an identical dataset remounted at a different location can be supplied explicitly with `--adcs-root` and `--phm-root`. The replacement roots must reproduce the same frozen content identity; path relocation never relaxes byte-level verification.

## 3. Materialized directory layout

The resulting directory contains:

```text
case_manifest.json
datasets/
  adcs/
    sample_index.csv
    <selected telemetry/label payloads>
  phm/
    sample_index.csv
    <selected telemetry payloads>
snapshot_metadata/
  source_dataset_snapshot_receipt.json
  materialized_dataset_snapshot_receipt.json
  materialization_receipt.json
```

Only payload paths present in the frozen selected fault/nominal snapshot are copied. Unselected dataset payloads are not required for the confirmatory denominator and are not materialized.

## 4. Identity verification

Before the output is committed, AstroGraph re-runs dataset admission against the copied bytes and requires:

```text
materialized_snapshot_sha256 == source_snapshot_sha256
```

The snapshot SHA256 is path-independent. The copied dataset may therefore live under a different absolute root without changing confirmatory content identity.

The `materialization_receipt.json` records:

- campaign ID;
- source and materialized snapshot SHA256;
- source and materialized snapshot receipt SHA256;
- copied file counts and aggregate identities;
- case-manifest SHA256;
- permission-hardening status;
- explicit claim boundary.

## 5. Permission hardening versus read-only mount

By default the materializer removes write permission bits from copied files/directories after verification.

This protects against common accidental edits but does **not** prove filesystem immutability. In particular, a privileged process or remount can still alter bytes.

For strict engineering execution, expose the materialized dataset roots through a genuine read-only mount and run preparation/runner admission with:

```text
--require-readonly-datasets
```

The formal runner continues to audit `/proc/self/mountinfo` where available and fails closed if a required read-only mount is not observed.

## 6. Running from the materialized snapshot

Use:

- `case_manifest.json` from the materialized root;
- `datasets/adcs` and `datasets/phm` as the dataset roots;
- `snapshot_metadata/materialized_dataset_snapshot_receipt.json` as `--dataset-snapshot-receipt`.

When `prepare_llm_method_confirmatory_campaign.py --materialize-snapshot` is used, `launch_plan.json` records a `materialized_runner_argv` when materialization succeeds.

If strict read-only admission is requested, do not treat chmod-only hardening as satisfying the mount gate. Bind/mount the materialized dataset roots read-only first, then run the formal command.

## 7. Governance boundary

R9 materialization is an experiment-input reproducibility control. It does not:

- produce diagnosis results;
- change promotion thresholds;
- qualify a model;
- authorize Diagnostic Probe output as formal evidence;
- establish flight or safety certification.

Probe promotion remains frozen until the real preregistered 38×8 campaign is executed and independently reviewed.
