# CURRENT_SPACECRAFT_SCOPE — AstroGraph R32 Spacecraft Engineering Convergence

## Decision

PATCH-R32 stops horizontal AstroGraph core-feature expansion and evaluates the current system strictly as a **spacecraft engineering diagnosis platform**.

The convergence decision deliberately separates:

1. **software/function convergence** — whether the spacecraft-domain functions needed by AstroGraph are implemented, governed, testable and stable enough to freeze; and
2. **diagnostic external qualification** — whether every algorithm branch has genuinely independent external evidence sufficient for a broad performance/generalization claim.

R32 permits the first to pass while the second remains blocked. Missing new holdout data is an evidence gap, not a reason to keep adding unrelated core features indefinitely.

## Spacecraft-domain capability scope

The frozen core scope includes:

- deterministic anomaly/fault detection;
- component localization and fault classification for admitted profiles;
- knowledge-graph construction and graph reasoning;
- open-set/candidate escape and abstention;
- evidence lineage and deterministic Evidence/Hypothesis governance;
- model-based conflict/minimal-diagnosis reasoning;
- mechanism-model diagnosis and bounded numerical counterfactual simulation;
- controlled Agent investigation with machine-readable tool authority;
- source/data exposure, reproducibility and release-claim governance.

Agent execution remains optional. Core detection/diagnosis must execute without Agent-selected probes.

## Evidence matrix

### ADCS v5.2.2

R24 complete test: detection macro-F1 0.924, localization 0.977, predicted-component classification 0.956, end-to-end 0.916, fault-event recall 1.0. The remaining material operational limitation is the high false-alarm episode burden (~79.22/hour in the R24 test contract). This blocks autonomous operational-FDIR claims, not software feature freeze.

### PHM v6.0

R24 complete test: detection macro-F1 0.899, localization 1.0, classification 1.0, end-to-end macro-F1 0.967. The main boundary is protocol scope: trace-summary/known-activation evidence does not establish unknown-onset continuous FDIR.

### OPS-SAT

The frozen 18-feature RF candidate has strong development evidence (confirmatory-development F0.5 0.902, MCC 0.858, AUROC 0.988, AUPR 0.943). The already-exposed official 529-segment test is not reopened for selection. A genuinely new OPS-like holdout is required before external-performance promotion.

### ESA Mission1 lightweight

The preserved legacy external result remains 0/4 event detection. R25/R26 add a frozen high-resolution fast branch with development event recall 1.0, alarm-group precision 0.9 and engineering F0.5 0.918. Because no genuinely new Mission1-like holdout has been supplied, the old negative external result is not overwritten.

### DXC / ADAPT

The legacy Competition/Rematch negative results remain part of the evidence base. R25-R31 add command-aware residuals, discrete-state consistency, model-based conflicts/minimal diagnoses, expected-response reasoning, qualitative forward simulation and reduced-order physics. The strongest later values are post-contact development/synthetic evidence and do not overwrite R20 external failures. A genuinely new ADAPT/DXC holdout is required for promotion.

## R32 convergence result

R32 classifies the current state as:

- **spacecraft software/function feature freeze: PASS**;
- **spacecraft engineering stable software candidate: PASS**;
- **broad independent external diagnostic qualification: BLOCKED**;
- **flight/safety/HIL qualification: NOT CLAIMED**.

This means active core development now changes direction. New mechanism families, non-spacecraft datasets and new domain-specific capabilities move to the roadmap unless a future spacecraft validation campaign exposes a concrete core defect.

## Product handoff

The next active product phase is **AstroGraph-Bench/UI**:

- benchmark execution and protocol badges;
- stable/current/experimental algorithm selection surfaces;
- qualification and evidence-boundary display;
- dataset/profile admission UX;
- diagnosis timeline and simulation trace visualization;
- candidate/conflict/minimal-diagnosis inspection;
- Agent tool authority inspector;
- comparison dashboards and exportable receipts;
- release-blocker / new-holdout readiness views.

The Bench/UI must not visually collapse development evidence, seen-external audit and independent external validation into one leaderboard.
