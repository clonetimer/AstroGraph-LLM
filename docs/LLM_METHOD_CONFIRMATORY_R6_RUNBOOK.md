# LLM Method Confirmatory Campaign — R6 Runbook

This runbook covers the frozen 38-case, A0–A7 engineering confirmatory campaign. It does not qualify a model or runtime for flight use.

## 1. Required external inputs

Mount both frozen dataset roots without copying them into the source delivery:

- ADCS v5.2.2 root containing `sample_index.csv`
- PHM v6.0 root containing `sample_index.csv`

Start an OpenAI-compatible vLLM endpoint and choose the exact advertised model ID to be frozen into the execution receipt.

## 2. Prepare in an external work directory

Recommended command:

```bash
python scripts/experiments/prepare_llm_method_confirmatory_campaign.py \
  --output-dir /path/to/confirmatory-work \
  --adcs-root /path/to/whole_spacecraft_adcs_paired_dataset_v5_2_2 \
  --phm-root /path/to/whole_spacecraft_phm_dataset_v6_0 \
  --vllm-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen3.5-9B \
  --freeze-case-manifest
```

The preparation command freezes `case_manifest.json` into the external work directory, probes the live model identity, and writes:

- `readiness.json`
- `launch_plan.json`
- `case_manifest.json` when freezing succeeds

The source tree is not modified when `--output-dir` is used.

## 3. Interpret readiness

`READY` means only that the preregistered design, exact 38-case denominator, dataset sample-index fingerprints, and requested model identity are present. It is not a performance result.

`BLOCKED` is fail-closed. Resolve every blocker before execution. Typical blockers are missing dataset roots, a missing/invalid 38-case manifest, an unreachable model endpoint, or the requested model ID not being advertised.

## 4. Execute the frozen campaign

Use the exact `runner_argv` written to `launch_plan.json`, or prepare and execute in one explicit command by adding:

```bash
--execute-if-ready
```

The formal runner independently repeats readiness admission. It then writes an execution receipt before A0–A7 starts and performs a second identity observation after execution.

Promotion evaluation is forbidden when any of the following drift:

- frozen design SHA256
- case manifest SHA256
- case identity SHA256 / denominator
- ADCS or PHM `sample_index.csv` SHA256
- requested live-model identity

## 5. Promotion boundary

A completed campaign produces a promotion decision only after post-execution identity is `STABLE` and the report case IDs exactly match the frozen 38-case manifest.

Passing the engineering gate authorizes only a governed promotion review. Active diagnostic probes remain shadow-only unless a separate promotion decision explicitly changes their formal-evidence eligibility.
