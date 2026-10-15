# R11 Supplemental Input Provenance Boundary

## Purpose

R11 adds a content-bound provenance receipt for each maintenance supplemental input admission. The receipt freezes the exact source report, original EpisodeEvidenceBundle, EvidenceInputContract, admitted capsule values, window/field/unit semantics, admission receipt, actor and parent admission chain.

Schema: `astrograph-maintenance-supplemental-input-provenance.v1`.

## Authority boundary

The provenance receipt proves lineage only. It is not a signature, release receipt, calibrated diagnosis, raw-telemetry replay receipt, or authority to mutate a frozen report. `formal_release_eligible` is structurally fixed to `false`.

A consumer must verify `receipt_sha256` before using the receipt as an audit link. Altering any content-bound field invalidates the receipt.

## Separation from FeatureResult quality

Supplemental provenance is not encoded in `FeatureResult.quality_flags`. Quality flags describe data/observation validity and are consumed by the causal-evidence validator as such. R11 keeps the supplemental origin in the admission/provenance receipts and maintenance overlay `analysis_metadata`, preventing provenance markers from being mistaken for invalid observations.

## Governed rerun note

The complete UI/platform distribution additionally uses these receipts to gate a narrow OAG evidence-branch child replay. That platform workflow requires a verified provenance receipt for every supplemental admission in the current overlay chain. Raw PUBLIC_CASE and PIPELINE_RUNTIME inputs still require their original governed telemetry/runtime rerun paths.
