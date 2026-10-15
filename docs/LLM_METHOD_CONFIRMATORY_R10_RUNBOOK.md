# LLM Method Confirmatory R10 Runbook

R10 adds an archive/copy integrity layer to the existing frozen 38-case confirmatory workflow. It does not create a performance result and does not relax the READY, exact-model, dataset-snapshot, read-only-mount, or post-execution identity gates.

## Intended sequence

1. Prepare/freeze the external experiment work directory using `prepare_llm_method_confirmatory_campaign.py`.
2. Materialize the admitted snapshot with `materialize_llm_method_confirmatory_snapshot.py`.
3. Generate the path-independent export manifest:

```bash
python scripts/experiments/export_llm_method_confirmatory_snapshot.py \
  --snapshot-root /path/to/materialized_snapshot
```

4. Archive/copy/remount the self-contained snapshot as needed.
5. Verify every exported byte before execution or archival use:

```bash
python scripts/experiments/verify_llm_method_confirmatory_snapshot_export.py \
  --snapshot-root /path/to/materialized_snapshot
```

A stable verification reports `STABLE`. Missing files, unexpected files, changed bytes, manifest self-tampering, or symlinks produce `DRIFTED` and fail closed.

## Launch-plan contract

`launch_plan.json` uses schema `astrograph-llm-method-confirmatory-launch-plan.v5` and now contains exact:

- `snapshot_materialization_argv`
- `snapshot_export_argv`
- `snapshot_verify_argv`
- `runner_argv`
- `materialized_runner_argv` when a valid materialization exists

The export manifest is path-independent for an already materialized tree: copying that tree to another root does not change the manifest identity. It is not a substitute for dataset-snapshot identity or model identity.

## Security / qualification boundary

- Symlinks are forbidden in the exported snapshot tree.
- The verifier detects missing, unexpected, or changed files and manifest self-hash drift.
- `chmod` hardening is not equivalent to a true read-only mount.
- A stable export is not evidence that the model is accurate or qualified.
- Formal A0–A7 promotion remains unavailable until the frozen 38-case campaign executes against the admitted real datasets and exact live model identity.
