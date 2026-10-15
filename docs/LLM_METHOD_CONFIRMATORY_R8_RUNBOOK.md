# LLM Method Confirmatory R8 Runbook

## Purpose

R8 strengthens the frozen 38-case A0–A7 engineering confirmatory workflow with a content-addressed dataset snapshot receipt and an optional read-only filesystem gate. It also tightens the operator maintenance loop so a shadow Probe cannot be repeatedly mistaken for formal evidence closure.

R8 does **not** authorize Diagnostic Probe output to create or modify formal `EvidenceFact` records. Promotion remains an engineering review gate only and does not establish flight qualification.

## 1. Prepare in an external experiment directory

Use an experiment work directory outside the source tree:

```bash
python scripts/experiments/prepare_llm_method_confirmatory_campaign.py \
  --output-dir /path/to/confirmatory-run \
  --adcs-root /datasets/whole_spacecraft_adcs_paired_dataset_v5_2_2 \
  --phm-root /datasets/whole_spacecraft_phm_dataset_v6_0 \
  --freeze-case-manifest \
  --vllm-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen3.5-9B
```

The command now emits:

- `case_manifest.json` — frozen 38-case denominator when data admission succeeds;
- `readiness.json` — design/case/model/dataset admission state;
- `dataset_mount_receipt.json` — dataset mount and payload aggregate audit;
- `dataset_snapshot_receipt.json` — content-addressed identity of the exact selected confirmatory bytes;
- `launch_plan.json` — exact formal runner command including the snapshot receipt path.

A blocked preparation exits non-zero and never fabricates a performance report.

## 2. Content-addressed dataset snapshot receipt

`dataset_snapshot_receipt.json` uses schema:

`astrograph-llm-method-confirmatory-dataset-snapshot-receipt.v1`

The snapshot identity binds:

1. campaign ID;
2. frozen case-manifest SHA256;
3. frozen 38-case identity SHA256;
4. each dataset `sample_index.csv` SHA256 and byte length;
5. every selected fault/nominal telemetry payload relative path, size and SHA256;
6. ADCS selected label payloads;
7. a per-dataset snapshot SHA256;
8. a path-independent whole-campaign `snapshot_sha256`.

Absolute mount paths are retained as audit metadata but are deliberately excluded from the content identity. The same bytes mounted at a different root therefore keep the same snapshot identity.

The receipt itself is hash-bound by `receipt_sha256`. Editing receipt metadata without recomputing a valid receipt is rejected by the formal runner.

## 3. Optional strict read-only dataset gate

For the strongest engineering execution mode, mount both datasets read-only and prepare with:

```bash
python scripts/experiments/prepare_llm_method_confirmatory_campaign.py \
  --output-dir /path/to/confirmatory-run \
  --adcs-root /datasets/adcs-ro \
  --phm-root /datasets/phm-ro \
  --freeze-case-manifest \
  --require-readonly-datasets \
  --vllm-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen3.5-9B
```

R8 performs a best-effort Linux mount audit using `/proc/self/mountinfo`. With `--require-readonly-datasets`, any dataset not observed on a `ro` mount blocks readiness with:

- `DATASET_NOT_READ_ONLY:adcs`
- `DATASET_NOT_READ_ONLY:phm`

On systems where mount access cannot be determined, strict read-only admission fails closed rather than guessing from `os.access`.

## 4. Formal runner requires the frozen snapshot receipt

The formal runner now requires:

```text
--dataset-snapshot-receipt /path/to/confirmatory-run/dataset_snapshot_receipt.json
```

`launch_plan.json` contains this argument automatically.

Before creating `execution_receipt.json`, the runner:

1. regenerates the current dataset snapshot identity;
2. validates the snapshot-receipt schema and `receipt_sha256`;
3. checks campaign/case-manifest/case-identity bindings;
4. checks each dataset content snapshot;
5. enforces the read-only policy when requested.

Any mismatch produces a fail-closed `dataset_snapshot_admission.json` and the 38×8 campaign does not start.

## 5. Post-execution dataset snapshot gate

After A0–A7 execution, the runner regenerates the dataset snapshot and writes:

`post_execution_dataset_snapshot.json`

Promotion evaluation is blocked if the snapshot changes. Relevant drift codes include:

- `DATASET_SNAPSHOT_SHA256_DRIFT`
- `DATASET_CONTENT_SNAPSHOT_DRIFT:<dataset>`
- `DATASET_READONLY_POLICY_DRIFT:<dataset>`
- `SNAPSHOT_RECEIPT_SHA256_INVALID`

The execution identity receipt also freezes `dataset_snapshot_sha256`, so the ordinary post-execution identity gate independently detects content snapshot drift.

## 6. TOCTOU boundary

Pre/post hashing alone cannot prove that a writable dataset was not modified and restored to the original bytes during execution.

For a formal engineering confirmatory run, use both:

1. a frozen content-addressed snapshot receipt; and
2. `--require-readonly-datasets` with genuinely read-only dataset mounts.

R8 does not claim stronger immutability than the observed filesystem/mount contract.

## 7. Maintenance Evidence Gap → Probe → Gap loop

Maintenance presentation schema is upgraded to:

`astrograph-maintenance-diagnosis.v6`

When an approved shadow Probe is executed, the result now records a gap reconciliation with:

- Probe ID and result SHA256;
- target `EvidenceDefinition`;
- signed margin when available;
- current formal evidence status;
- whether the formal evidence gap remains open;
- an explicit `gap_closed_by_probe=false` boundary;
- the next governed action required to change the formal state.

The result is persisted under `output/diagnostic_probes/` and appears in `probe_review_history` after refresh.

If the same shadow threshold-margin Probe already reviewed a contradiction and the current formal `EpisodeEvidenceBundle` still contains that gap, R8 no longer offers the same automatic Probe again. The next diagnostic check becomes an action requirement to verify data/units/threshold contracts or provide new governed input and regenerate the formal `EvidenceFact`.

This prevents a UI loop where repeated recomputation is mistaken for evidence acquisition.

## 8. Probe promotion boundary

R8 retains:

`formal_evidence_eligible=false`

for all current v1 Diagnostic Probes. No shadow review can alter diagnosis disposition, release status or formal evidence. Formal-evidence promotion remains frozen until the preregistered real 38×8 A0–A7 campaign is completed and independently reviewed.
