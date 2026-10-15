# AstroGraph PATCH-R25 — DXC hybrid diagnosis / ESA high-resolution remediation

Status: **CURRENT_R25_DEVELOPMENT_REMEDIATION_ONLY**  
Date: 2026-09-19

## Scope

R25 follows the R24 benchmark without reopening R20/R21 seen external partitions for fitting, tuning, selection or prediction regeneration. It adds two deterministic cores below the Agent layer:

1. an ESA high-resolution fast-transient branch based on robust lag-one telemetry deltas and multichannel coincidence;
2. a DXC command-aware one-step residual model coupled to the R23 engineering system graph and bounded minimal-diagnosis engine.

Neither output is a calibrated probability or a formal EvidenceFact. The existing R20/R21 external negative results remain frozen.

## ESA high-resolution branch

`src/astrograph/evaluation/esa_high_resolution.py` preserves the raw Mission1 timestamp grid. It fits per-channel robust first-difference references on admitted development windows and scores coincident transient changes across lightweight channels 41–46.

The R25 development experiment uses only `MODEL_TRAIN` for the reference and `DEVELOPMENT_SELECTION` for candidate selection. The engineering selection metric combines alarm-group precision with anomaly-ID recall inside event-centered windows; it is **not** the ESA-ADB corrected event-wise metric.

Selected development configuration:

- robust delta-z threshold: `8`;
- minimum coincident channels: `2`;
- alarm grouping gap: `40 s`.

Development result:

- anomaly IDs: 3;
- hit anomaly IDs: 3;
- alarm groups: 30;
- false alarm groups: 3;
- alarm-group precision: `0.9000`;
- anomaly-ID recall: `1.0000`;
- engineering F0.5: `0.918367`.

A read-only audit of the four frozen R21 confirmatory misses (`id_107`, `id_109`, `id_110`, `id_114`) confirms durations of approximately 3.9–9.9 seconds and two raw samples inside every event annotation on **each** lightweight channel 41–46. R25 does not run the new detector on those seen confirmatory events. The frozen AstroGraph external result therefore remains 0 alarm / 4 FN.

## DXC command-aware residual / graph diagnosis

`src/astrograph/evaluation/dxc_hybrid_model_based.py` introduces a serializable deterministic development model:

- previous numeric telemetry + current explicit command-instance state -> one-step Ridge forecast;
- robust standardized forecast residual;
- scenario startup residual normalization;
- anomalous sensor projection through the ADAPT engineering graph;
- bounded structural conflicts -> subset-minimal/minimum-cardinality diagnosis audit;
- a separate deterministic graph-support ranker for the current engineering hypothesis set.

The development model artifact is content-bound. It records scaler parameters, Ridge coefficients/intercept, residual reference and model SHA256; prediction does not depend on mutable sklearn estimator state.

The 74 DXC09 injected Sample cases are deterministically split by the already-registered scenario SHA256: 49 training, 11 selection, 14 internal replay. This is **not** an independent validation split because the Sample family was already development-exposed before R25.

On the 14-case internal replay:

- R20 baseline: 14/14 first alarms occur before injection;
- R25 hybrid: 1 early alarm, 12 first post-injection alarms, 1 no alarm;
- post-injection first-alarm rate: `0.857143`;
- median post-injection delay: `0.4455 s`;
- fault-element micro F1: `0.095238` vs baseline `0.025381`;
- component micro F1: `0.190476` vs baseline `0.025381`;
- exact fault-set rate remains `0`;
- final predicted-fault any-overlap falls to `0.142857` from the broad baseline `0.357143`.

The bounded minimal-diagnosis candidate set contains some true fault element in `0.357143` of internal-replay cases and contains the exact truth set as a candidate in `0.142857`. This is useful evidence that conflict generation can retain explanations not selected by the final ranker, but it is not a satisfactory isolation result.

A direct experiment that expanded every residual conflict with recent command targets caused hitting-set combinatorics and did not improve the replay isolation signal. The selected path therefore keeps command state in the forecast model and preserves recent command IDs as audit context, while **not** automatically adding a command target to every structural conflict. A future command-response consistency model must decide when a particular command implies an expected downstream transition instead of treating command adjacency as causality.

## Engineering conclusion

R25 substantially strengthens the ESA representation layer and DXC alarm timing, but the two tracks have different maturity:

- **ESA:** high-resolution remediation is ready to freeze for a genuinely new holdout, subject to a formal event metric and false-alarm exposure contract.
- **DXC:** detector timing is improved, but localization/classification remains unstable between the 11-case selection slice and 14-case replay. The next work item is a hybrid component-mode transition/expected-response model and a diagnosis ranker that evaluates candidate consistency, not merely topology support.

R25 does not change Agent authority. These algorithms remain deterministic cores that can later back bounded probes; Agent invocation is not required for primary detection.

## Claim boundary

R25 does not claim new external validation, public SOTA, flight/HIL qualification, calibrated probability, formal-EvidenceFact promotion, or resolution of the frozen R20/R21 ESA/DXC failures. Competition/Rematch DXC and ESA confirmatory/external predictions are not regenerated.
