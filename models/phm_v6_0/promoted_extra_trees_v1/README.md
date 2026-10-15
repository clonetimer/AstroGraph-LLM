# PHM v6.0 promoted ExtraTrees retrospective screening bundle

Bundle ID: `phm-v6.0-extra-trees-retrospective-screening-v1`

This bundle is the AstroGraph post48.dev13 default for **known-activation post-activation trace-summary replay** only.

- Detection: ExtraTrees-500, Validation-selected threshold `0.5`.
- Localization: ExtraTrees-500.
- Classification: ExtraTrees-500, `raw` route.
- Input protocol: per-channel mean/std/min/max/delta/rms after the known activation boundary.
- Dataset identity and file hashes are pinned in `manifest.json`.
- Unknown-onset causal-window PHM models are unchanged.
- No flight, safety or hardware qualification is claimed.

Verify locally:

```bash
PYTHONPATH=src python scripts/integration/verify_phm_v60_promoted_bundle.py \
  --dataset-root /path/to/whole_spacecraft_phm_dataset_v6_0 \
  --output ../AstroGraph-local/phm-dev13-bundle-verify.json
```
