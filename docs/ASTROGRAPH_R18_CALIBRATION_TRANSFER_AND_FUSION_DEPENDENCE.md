# AstroGraph R18 — Calibration Transfer and BiRRF Dependence Protocol

Status: **CURRENT_R18_IMPLEMENTED_PROTOCOL / REAL EXTERNAL VALIDATION NOT YET EXECUTED**  
Date: 2026-09-18

## 1. Scope

R18 follows R17's score-semantics repair. It does not rename more scores or grant any new runtime authority. It freezes executable offline research protocols for two open questions:

1. whether a model's probability calibration transfers across independent operating conditions; and
2. how sensitive BiRRF behavior is to inclusion/removal and dependence of its data/KG branches.

The work remains offline and truth-bearing. Runtime diagnosis does not receive ground truth from these modules.

## 2. Calibration campaign partitions

Every sample belongs to exactly one independent `group_id` and one partition:

- `CALIBRATION_FIT` — data that may be used to fit a calibrator;
- `DEVELOPMENT_SELECTION` — data that may select a confidence/abstention threshold;
- `CONFIRMATORY_EVALUATION` — held-out evaluation groups.

A group may not cross partitions. A confidence threshold selected on confirmatory groups is rejected. The threshold receipt is self-hash verified and its `selected_on_groups` must exactly equal the admitted development groups.

This prevents post-hoc threshold selection on the same groups used to report transfer performance.

## 3. Fixed probability universe

Within one calibration campaign, every frozen prediction must expose the exact same class universe, and each truth label must belong to that universe. A missing true class or per-sample class-set drift fails closed rather than silently treating the missing class as probability zero.

This is necessary for interpretable multiclass NLL, Brier score and classwise one-vs-rest calibration metrics.

## 4. Metrics and inferential unit

R18 reports both sample-global metrics and equal-independent-group summaries.

Sample-global:

- accuracy;
- multiclass NLL;
- multiclass Brier score;
- equal-width ECE with frozen bin count;
- AURC.

Independent-group summaries:

- group-macro accuracy;
- group-macro NLL;
- group-macro Brier;
- group-macro ECE;
- classwise one-vs-rest Brier;
- classwise one-vs-rest ECE.

Uncertainty is reported with a deterministic **independent-group percentile bootstrap** using a frozen seed/resample count. The interval is explicitly descriptive. R18 does not claim finite-sample exact coverage from this bootstrap.

ECE is not treated as a unique or sufficient definition of calibration. Its value depends on the estimator/binning scheme, so the report freezes `EQUAL_WIDTH_10_BIN_ECE` and always reports proper scoring rules (NLL/Brier) alongside it.

Methodological context: Guo et al. (2017), *On Calibration of Modern Neural Networks*; Posocco & Bonnefoy (2021), *Estimating Expected Calibration Errors*; Roschewitz et al. (2025), *Where are we with calibration under dataset shift in image classification?*.

## 5. Calibration-transfer strata

The frozen R18 design requires four group-level context dimensions:

- `simulator_version`;
- `mission_mode`;
- `fault_severity`;
- `noise_condition`.

Each independent group must have one stable value for every dimension. The evaluation input declares a baseline value for every dimension.

R18 reports per-stratum calibration metrics and deltas versus the declared baseline. These are **descriptive contrasts**. If another declared dimension changes with the dimension under review, the report marks:

`CONFOUNDED_WITH_OTHER_DECLARED_DIMENSIONS`

and records the co-varying dimensions. R18 therefore does not convert a jointly shifted scenario into a false single-axis causal claim.

## 6. Selective-risk tie repair — BASE-R18-01

The inherited risk-coverage implementation sorted `(confidence, correctness)` tuples one sample at a time. Equal-confidence correct samples could therefore appear before equal-confidence incorrect samples, creating an impossible intermediate threshold point.

R18 groups all equal-confidence samples into one indivisible threshold block. A threshold cannot keep one 0.8-confidence sample while rejecting another 0.8-confidence sample merely because one is correct.

Regression counterexample:

- two samples, both confidence 0.8;
- one correct, one wrong;
- feasible risk-coverage point is only `coverage=1.0, risk=0.5`;
- with `max_selective_risk=0`, no threshold is feasible.

The previous optimistic half-coverage/risk-zero point is no longer representable.

## 7. BiRRF leave-branch-out audit

R18 evaluates these frozen policy variants on identical cases/groups:

- `FULL`;
- `DATA_ONLY`;
- `FORWARD_ONLY`;
- `BACKWARD_ONLY`;
- `NO_DATA`;
- `NO_FORWARD`;
- `NO_BACKWARD`.

For every policy it reports case accuracy and equal-group macro accuracy with group-bootstrap interval. `FULL` is compared with each variant using:

- equal-group accuracy delta;
- positive/negative/tie group counts;
- exact two-sided group-direction sign statistic;
- case-level unique corrections/harms as descriptive counts;
- Top-1 disagreement count.

It additionally reports pairwise data/forward/backward Top-1 disagreement and double-wrong rates at the independent-group level.

These are structural dependence/sensitivity measurements. Removing a branch changes the system and can also change redundancy handling. The result is therefore explicitly:

`STRUCTURAL_SENSITIVITY_NOT_CAUSAL_ATTRIBUTION`

It must not be interpreted as Shapley value, causal information contribution, calibrated probability contribution, or release evidence.

RRF itself is a rank-fusion method rather than a probability model. R17 already froze that semantic boundary. R18 adds empirical dependence analysis without weakening it. Relevant rank-fusion sensitivity context includes Bruch et al. (2022), *An Analysis of Fusion Functions for Hybrid Retrieval*.

## 8. Content identity

The calibration report freezes canonical SHA256 identities for:

- frozen prediction set;
- truth set;
- partition/shift context set.

The BiRRF dependence report freezes a canonical case identity including truth/group identity and the three input rankings. Report self-hashes bind the results.

This makes future real-data reports auditable against the exact admitted records.

## 9. Synthetic smoke boundary

R18 includes synthetic CLI receipts only to prove protocol reachability:

- `governance/evidence/research/r18_calibration_dependence/calibration_transfer_synthetic_receipt.json`;
- `governance/evidence/research/r18_calibration_dependence/birrf_dependence_synthetic_receipt.json`.

They are **not** model calibration evidence, BiRRF performance evidence, OOD validation, or source-utility evidence.

## 10. What R18 does not claim

R18 does not claim:

- that any current AstroGraph model is calibrated on real external data;
- that one ECE value is sufficient evidence of calibration;
- that calibration transfers between simulator versions or mission modes;
- that a selective-risk threshold is safe under OOD shift;
- that BiRRF `1/1/1, k=60` is optimal;
- that a KG branch has causal diagnostic value;
- that a leave-branch-out delta is a probability contribution;
- that HIL, target-compute, operational or flight qualification has occurred;
- that the real 38×8 A0–A7 confirmatory campaign has run.
