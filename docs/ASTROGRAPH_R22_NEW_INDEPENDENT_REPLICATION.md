# AstroGraph R22 — CURRENT New Independent Replication Protocol

Document status: `CURRENT_R22_REPLICATION_PROTOCOL`.

Status: **CURRENT / READY_FOR_NEW_HOLDOUT_BLOCKED_DATASET**.

R22 starts only from the packaged PATCH-R21 unified source and does not reopen R20/R21 seen-evaluation data for fitting, tuning, threshold selection, calibration, or optimization. Its purpose is to freeze the next replication protocol before any new holdout result is observed.

## 1. First-contact admission

A new source artifact is hashed before truth inspection or prediction generation. R22 rejects exact source-artifact reuse and exact member-byte reuse against the known R20/R21 dataset identities. The operator must explicitly attest that intake is occurring before truth inspection; this attestation is recorded but is not presented as independent cryptographic proof of human non-observation.

After intake, group IDs may be admitted without labels. R22 derives canonical group identities from source-content identity plus raw group ID. This proves identity novelty only; it does **not** prove physical/statistical independence of trajectories, missions, spacecraft, channels, or operating modes.

## 2. OPS-SAT frozen comparison

R22 freezes the R20 raw-segment baseline at threshold 100 and the R21-selected remediation `LENGTH_AWARE_FALLBACK_FEATURE_SET`.

The remediation is now an executable source artifact rather than a development callback:

- if the original R20 raw detector can score the segment, the unchanged raw score and threshold 100 are used;
- only the original insufficient-baseline/window failure states fall back to the short-segment path;
- the short path computes mean, span, **sample** standard deviation, and slope against sample index when each operator's preregistered minimum sample count is met;
- each feature is converted to absolute robust-z against the frozen R21 MODEL_TRAIN channel reference;
- the short score is the maximum available robust-z;
- short threshold is frozen at 12;
- no R22 threshold refit or candidate search is allowed.

The exact R22 implementation was checked on all 227 R21 development-selection segments and reproduces the complete frozen selected confusion matrix and all reported aggregate metrics exactly. R21 did not persist a per-segment selected-candidate score map, so R22 does **not** claim independently auditable score-by-score identity to an R21 score artifact. R22 freezes its own reproduced per-segment mapping identity for future implementation-drift checks.

Primary R22 OPS endpoint is `coverage × covered F0.5`. Coverage, covered F0.5, false-positive/false-negative counts and descriptive paired corrected/induced cases are reported separately. No inferential p-value is created merely from segment IDs because segment identity does not itself establish independent sampling.

## 3. ESA transport replication only

R21 selected `MEDIAN_5MIN_THRESHOLD_6`. This is the same 5-minute-median, per-channel frozen reference, score threshold 6 baseline already used by R20. R22 therefore explicitly forbids a baseline-vs-remediation superiority claim for ESA.

ESA R22 can only answer whether the **unchanged protocol transports** to a genuinely new holdout. The replication input is a governed six-channel non-overlapping 5-minute-median representation with a separate preprocessing receipt. The receipt binds both the upstream file bytes and the exact `time_ns/value/observed` arrays consumed by the scorer. R22 does not infer that a supplied preprocessed artifact is equivalent to raw telemetry merely because its schema matches.

ESA execution is explicitly two-stage. `run_r22_esa_replication.py` writes a sealed pre-truth prediction receipt first; only then may an optional truth JSON be read and joined to produce the event-metric replication report. The truth join records no baseline-vs-remediation delta because the two implementations are identical by construction.

Primary ESA endpoint remains corrected event-wise F0.5, accompanied by event precision/recall, false-alarm duration, first/confirmed detection delay, and rare-nominal/communication-gap false alarms. No threshold retuning is allowed on the new holdout.

## 4. Claim boundary

Without a new admitted source R22 remains `READY_FOR_NEW_HOLDOUT_BLOCKED_DATASET`. OPS and ESA are independent replication tracks: admission/readiness for one track does not require a simultaneous holdout for the other track. Preparation or synthetic smoke does not create external-validation evidence, probability calibration, flight qualification, or a new `astrograph-external-validation-receipt`.

If a new holdout is later run, the resulting R22 replication report is a new-holdout replication artifact. Any stronger statistical claim requires independent-group provenance beyond byte/group identity novelty.
