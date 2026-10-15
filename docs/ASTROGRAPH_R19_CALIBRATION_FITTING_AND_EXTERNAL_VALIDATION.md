# AstroGraph R19 — Calibration Fitting and External-Validation Admission

## Scope

R19 turns the R18 evaluation-only calibration-transfer protocol into a governed fitting and external-validation workflow. It does **not** claim that any real AstroGraph model is calibrated. The real-data state remains blocked until frozen model predictions, independent-group truth/context assignments, calibration material and a distinct external-validation dataset are admitted.

The frozen sequence is:

`CALIBRATION_FIT -> DEVELOPMENT_SELECTION -> CONFIRMATORY_EVALUATION -> EXTERNAL_VALIDATION`

An independent `group_id` may appear in exactly one partition. The calibrator may use only `CALIBRATION_FIT`; the abstention threshold may use only `DEVELOPMENT_SELECTION`; neither confirmatory nor external groups may retune either object.

## 1. Calibrator implemented in R19

R19 implements one narrow method:

`TEMPERATURE_SCALING_ON_LOG_PROBABILITIES`

For probability vector `p`, the transformed probability is proportional to `exp(log(p) / T)`. When the input probabilities come from a softmax classifier, this is equivalent to ordinary scalar temperature scaling up to the softmax additive-logit invariance. R19 deliberately does not implement Dirichlet/vector/matrix/isotonic calibration yet.

The fit objective is not sample-weighted NLL. It is:

`EQUAL_INDEPENDENT_GROUP_MACRO_NLL`

Each independent group contributes equal weight, regardless of the number of samples/windows inside that group. The deterministic optimizer is a log-space coarse grid followed by golden-section refinement; `T=1` is always an admitted candidate, so the fitted objective cannot become worse than the identity transform merely because the search grid missed exactly 1.0.

The fit receipt content-binds:

- model ID/version and model artifact SHA256;
- calibration dataset ID/SHA256;
- exact CALIBRATION_FIT group IDs;
- frozen prediction/truth/context identities;
- class universe;
- minimum positive independent groups per class;
- optimizer bounds/configuration;
- fitted temperature and before/after equal-group macro NLL.

A reused fit receipt is re-admitted by recomputing those content identities. Matching group names alone are insufficient.

## 2. Per-class admission

A total group count is not enough for multiclass calibration. R19 requires minimum positive independent groups for **every class** in:

- CALIBRATION_FIT;
- CONFIRMATORY_EVALUATION;
- EXTERNAL_VALIDATION.

This prevents a classwise Brier/ECE table with zero positive independent groups from being treated as evidence that calibration transfer was assessed for that class.

## 3. External-validation identity

The final qualification binds both:

- the declared external-validation dataset ID/SHA256; and
- a canonical identity over admitted external prediction, truth, group and strata records.

The external-validation dataset ID must differ from the calibration-fit dataset ID. R19 therefore does not permit relabeling the fit dataset as “external” merely by placing its groups in another JSON section.

The final qualification receipt also binds the exact R19 design-file SHA256 and exact admission-bundle SHA256.

## 4. Metrics and guardrails

R19 carries forward R18's sample-global and equal-independent-group views of:

- accuracy;
- negative log likelihood (NLL);
- multiclass Brier score;
- equal-width ECE;
- classwise one-vs-rest Brier/ECE;
- AURC / selective risk.

NLL and Brier are retained because proper scoring rules evaluate the quality of the full predictive distribution. ECE is kept as a descriptive calibration diagnostic, but it is not the sole admission criterion because finite-sample/binning choices materially affect ECE estimates.

The frozen R19 thresholds are explicitly labeled:

`ENGINEERING_REVIEW_FLOORS_NOT_FLIGHT_CALIBRATED`

A pass means only `PASS_FOR_ENGINEERING_CALIBRATION_REVIEW`.

## 5. Selective-risk guard

R19 does not compare calibrated and raw probabilities at one arbitrarily shared numeric threshold. Two policies are selected separately, but on the **same DEVELOPMENT_SELECTION groups** and under the same maximum development selective-risk constraint:

1. raw-probability baseline policy;
2. calibrated-probability policy.

Both frozen thresholds are then transferred without retuning to EXTERNAL_VALIDATION. The guard checks external selective-risk increase and coverage decrease. This is still an engineering transfer test, not an OOD guarantee.

## 6. Probability qualification receipts

Legacy `astrograph-probability-calibration-receipt.v1` remains readable for backward compatibility and score-semantics auditing. It is now explicitly treated as:

`CONTENT_BOUND_RECEIPT_VERIFIED_LEGACY_NO_EXTERNAL_VALIDATION`

It is **not sufficient** for the RW_JAM formal promotion gate.

R19 adds:

`astrograph-probability-calibration-qualification-receipt.v1`

It binds the fit receipt, design, exact admission bundle, model artifact, calibration dataset, confirmatory/external evaluation report, external dataset identity and development threshold receipt. Only `PASS_FOR_ENGINEERING_CALIBRATION_REVIEW` can satisfy the strengthened promotion gate; a blocked qualification receipt remains auditable but cannot promote the model.

## 7. Synthetic protocol smoke

The synthetic R19 fixture contains:

- 8 fit groups;
- 4 development groups;
- 8 confirmatory groups;
- 8 external-validation groups;
- two classes with 4 positive confirmatory and 4 positive external groups per class.

It produced a deterministic temperature of approximately `1.303429199369053`. A second invocation reused the frozen fit receipt and produced the exact same final qualification receipt.

These synthetic values are **not** calibration evidence, model-quality evidence, OOD evidence, diagnostic-accuracy evidence or flight qualification.

## 8. Research-method rationale

R19 follows the standard post-hoc calibration idea that scalar temperature scaling can improve probability calibration without changing class ranking, while making the experiment stricter in AstroGraph-specific ways: independent-group fitting, locked selection/evaluation partitions, content-addressed receipts and an explicit external-validation partition.

Relevant methodological context:

- Guo, Pleiss, Sun & Weinberger, *On Calibration of Modern Neural Networks* (2017): https://arxiv.org/abs/1706.04599
- Silva Filho et al., *Classifier calibration: a survey on how to assess and improve predicted class probabilities* (Machine Learning, 2023): https://link.springer.com/article/10.1007/s10994-023-06336-7
- Posocco & Bonnefoy, *Estimating Expected Calibration Errors* (2021): https://arxiv.org/abs/2109.03480
- Roschewitz, Mehta, de Sousa Ribeiro & Glocker, *Where are we with calibration under dataset shift in image classification?* (2025): https://arxiv.org/abs/2507.07780

These papers motivate the methodology; they do not validate AstroGraph's specific thresholds or datasets.

## 9. Remaining boundary

R19 still does not establish:

- real calibration on frozen AstroGraph ADCS/PHM data;
- superiority of temperature scaling over Dirichlet/vector/isotonic alternatives;
- finite-sample exact calibration confidence intervals;
- robust calibration under arbitrary covariate/label/concept shift;
- target-hardware/HIL/flight validity;
- probability calibration of BiRRF scores (BiRRF remains rank fusion, not probability inference).
