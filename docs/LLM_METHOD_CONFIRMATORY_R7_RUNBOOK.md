# LLM Method Confirmatory R7 Runbook

## Purpose

R7 hardens the frozen 38-case A0–A7 engineering confirmatory campaign before any real execution. It does **not** relax the existing qualification boundary: `READY` only admits the experiment; promotion still requires the preregistered paired result gates and does not establish flight qualification.

## 1. Prepare an external working directory

Do not freeze the case manifest into the source delivery. Use an experiment directory outside the source tree:

```bash
python scripts/experiments/prepare_llm_method_confirmatory_campaign.py \
  --output-dir /path/to/confirmatory-run \
  --adcs-root /datasets/whole_spacecraft_adcs_paired_dataset_v5_2_2 \
  --phm-root /datasets/whole_spacecraft_phm_dataset_v6_0 \
  --freeze-case-manifest \
  --vllm-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen3.5-9B
```

The preparation command emits:

- `case_manifest.json` — frozen 38-case denominator when dataset admission succeeds;
- `readiness.json` — design/case/model/dataset admission status;
- `dataset_mount_receipt.json` — selected sample/pair and payload identity receipt;
- `launch_plan.json` — exact formal runner command.

A blocked preparation exits non-zero and does not fabricate an experiment result.

## 2. Dataset mount admission

R7 no longer treats the presence of `sample_index.csv` as sufficient. For each dataset it checks:

1. required sample-index columns;
2. unique sample IDs;
3. every preregistered mapped test row at sample indices 4 and 9;
4. the exact nominal pair identity for each selected fault row;
5. agreement between the frozen case manifest and the mounted sample index;
6. referenced telemetry files, plus ADCS label files;
7. regular-file existence and non-zero length;
8. resolved paths remaining inside the mounted dataset root;
9. a SHA256 aggregate over the actual selected fault + nominal payload files.

The aggregate is reported as `payload_identity_sha256`. A mount that has a valid index but missing/empty/escaping referenced files is `DATASET_MOUNT_INVALID:<dataset>` and cannot start the campaign.

## 3. Pre-execution identity receipt

The formal runner creates `execution_receipt.json` only after readiness is `READY`. R7 freezes, per dataset:

- `sample_index_sha256`;
- `payload_identity_sha256`;
- `payload_file_count`.

It also freezes the design SHA256, case-manifest SHA256, 38-case identity SHA256 and exact requested vLLM model ID.

## 4. Execute only after READY

Either rerun preparation with `--execute-if-ready`, or execute the exact `runner_argv` in `launch_plan.json`.

Do not edit the case manifest, mounted datasets, design or model service while the campaign is running.

## 5. Post-execution identity gate

Before promotion evaluation, the runner probes the environment again and rehashes the selected dataset payloads. Promotion is blocked if any of these drift:

- frozen design;
- case-manifest bytes;
- 38-case identities/count;
- dataset sample index;
- selected telemetry/label payload bytes or payload-file count;
- exact live model identity.

A payload-only change with an unchanged `sample_index.csv` produces `DATASET_PAYLOAD_IDENTITY_DRIFT:<dataset>` and no promotion decision is generated.

## 6. Diagnostic Probe boundary

The confirmatory experiment still evaluates the existing shadow-only Probe mechanism. R7 does not authorize a Probe to create or alter formal `EvidenceFact` records. `formal_evidence_eligible=false` remains mandatory until a separate review is supported by the completed preregistered A0–A7 evidence.
