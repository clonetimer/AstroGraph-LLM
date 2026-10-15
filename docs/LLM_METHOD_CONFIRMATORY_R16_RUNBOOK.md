<!-- CURRENT_DOCUMENT: 0.7.1.post49 -->
# LLM Method Confirmatory R16 Runbook

Status: **CURRENT_CONFIRMATORY_RUNBOOK**  
Date: 2026-09-18

This runbook prepares the R16 reproducibility identity artifacts. It does not manufacture confirmatory performance results.

## 1. Freeze the AstroGraph orchestrator environment

Run from the exact Python environment that will launch the campaign:

```bash
python scripts/experiments/freeze_llm_method_confirmatory_environment.py \
  --output /external/confirmatory/environment_lock.json
```

Do not regenerate this lock after observing campaign performance and call the new environment the same experiment.

## 2. Freeze the model-server launch configuration

Start from:

`configs/evaluation/vllm_server_launch_config.template.v1.json`

Fill the exact served model ID and exact launch argv. R16 requires:

```text
generation_config_policy = vllm
--generation-config vllm
```

The vLLM service used for the campaign must actually be launched according to this frozen contract.

## 3. Build the model-runtime spec on the model host

```bash
python scripts/experiments/build_llm_method_confirmatory_model_runtime_spec.py \
  --output /external/model-host/model_runtime_spec.json \
  --producer-id <governed-host-id> \
  --model-id <exact-model-id> \
  --model-revision <exact-revision> \
  --model-root <model-artifact-root> \
  --tokenizer-revision <exact-tokenizer-revision> \
  --tokenizer-root <tokenizer-artifact-root> \
  --server-launch-config /external/model-host/vllm_launch.json \
  --server-artifact-root <vllm-artifact-root> \
  --deployment-kind BARE_METAL
```

For container deployment use `--deployment-kind CONTAINER` and provide `--container-image-digest`.

`--require-readonly-artifacts` is optional but recommended when the deployment can provide genuinely read-only model/tokenizer mounts.

## 4A. Same-host/local recomputation mode

Pass the spec directly to preparation/formal execution:

```bash
python scripts/experiments/prepare_llm_method_confirmatory_campaign.py \
  --output-dir /external/confirmatory/work \
  --adcs-root <frozen-adcs-root> \
  --phm-root <frozen-phm-root> \
  --vllm-url http://127.0.0.1:8000/v1 \
  --vllm-model <exact-model-id> \
  --environment-lock /external/confirmatory/environment_lock.json \
  --model-runtime-spec /external/model-host/model_runtime_spec.json \
  --freeze-case-manifest
```

## 4B. Remote model-host sidecar mode

On the model host:

```bash
python scripts/experiments/serve_llm_method_confirmatory_runtime_attestation.py \
  --spec /external/model-host/model_runtime_spec.json \
  --host 127.0.0.1 \
  --port 8765
```

Expose it through an independently governed authenticated tunnel/proxy if the orchestrator is remote. R16 itself does not authenticate a non-loopback producer.

Then prepare using:

```bash
python scripts/experiments/prepare_llm_method_confirmatory_campaign.py \
  ... \
  --environment-lock /external/confirmatory/environment_lock.json \
  --model-runtime-attestation-url https://<governed-endpoint>/attestation
```

## 5. Required preparation outputs

A valid R16 preparation should contain at least:

- `readiness.json`;
- `runtime_identity_admission.json`;
- `model_runtime_attestation.json` when acquisition succeeded;
- `dataset_mount_receipt.json`;
- `dataset_snapshot_receipt.json`;
- `launch_plan.json`.

`launch_plan.json` is schema v6 and contains a formal runner argv only when both reusable runtime-identity inputs are present.

## 6. Formal execution

Use `launch_plan.json.runner_argv`, or let preparation invoke it with `--execute-if-ready`.

The formal runner rejects an output directory inside the AstroGraph source tree and independently redoes all readiness/runtime/snapshot checks.

Before execution it writes:

- `model_runtime_attestation_pre.json`;
- `runtime_identity_admission_pre.json`;
- `dataset_snapshot_admission.json`;
- `execution_receipt.json`;
- `command_receipt.json`.

After execution it writes:

- `model_runtime_attestation_post.json`;
- `runtime_identity_admission_post.json`;
- `post_execution_readiness.json`;
- `post_execution_dataset_snapshot.json`;
- `post_execution_identity.json`.

Promotion evaluation is forbidden unless `post_execution_identity.status == STABLE`.

## 7. Required claim boundary

Do not state that R16 identity closure proves model quality, remote-host trust, flight qualification or deterministic floating-point replay.

Do not treat a synthetic fixture, local unit test or dry-run receipt as a real A0–A7 result.
