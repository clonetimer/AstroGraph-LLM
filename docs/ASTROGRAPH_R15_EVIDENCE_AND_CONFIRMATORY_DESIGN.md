# AstroGraph R15 Evidence Validity and Confirmatory Statistical Design

Status: **CURRENT_ENGINEERING_RESEARCH_DESIGN**  
Patch: PATCH-R15  
Date: 2026-09-18

## 1. Scope

R15 deliberately does not add new Agent autonomy, new Probe authority, or new release paths. It closes the five P0 findings identified in the R14 research audit before any real 38×8 A0–A7 confirmatory result may be used for an engineering promotion decision.

The five P0 findings were:

1. clustered confirmatory cases were being treated as independent case-level McNemar pairs;
2. the full-stack A2-vs-A0 estimand was being conflated with the Probe-specific promotion question;
3. the promotion gate lacked a preregistered effect estimate, confidence interval and precision/minimum-effect policy;
4. non-finite or quality-invalid FeatureResult values could enter ordinary threshold truth semantics;
5. mixed Evidence + Knowledge citations could disable Agent measurement grounding.

R15 does **not** run the real confirmatory campaign and does not promote Probe output to formal EvidenceFact production.

## 2. Formal Evidence validity semantics

Formal threshold evaluation now calls one central `classify_feature_result_validity()` policy before comparison.

A FeatureResult can be classified as:

- `VALID`;
- `MISSING_VALUE`;
- `MISSING_WINDOW_DATA`;
- `INSUFFICIENT_SAMPLES`;
- `INVALID_NONFINITE`;
- `INVALID_QUALITY_FLAG`;
- `INVALID_SAMPLE_COUNT`.

Only `VALID` may enter `GREATER_THAN`, `LESS_THAN`, equality, or other formal threshold truth semantics.

Missing-type validity maps to `EvidenceFactStatus.MISSING_DATA`. Invalid-type validity maps to `EvidenceFactStatus.INVALID_DATA`. The observed value is not copied into the EvidenceFact as a valid measurement when comparison was not performed.

The same rule is applied to AND/conjunctive feature conditions. A valid primary feature cannot turn an invalid conjunct into ordinary `NOT_SATISFIED` evidence.

This preserves the distinction:

`bad/unavailable observation != evidence that the condition is false`.

## 3. Numeric grounding with mixed citations

R13 already rejected the direct substitution `E1=0.2 A` -> model text `900 A` when E1 was the only citation.

R15 closes the mixed-source bypass. If a paragraph or hypothesis cites both Evidence and Knowledge, the Evidence subset is still extracted and every explicit measurement in that mixed claim must match a cited frozen Evidence measurement. Adding a Knowledge citation cannot switch off evidence-value validation.

Knowledge-derived quantitative statements should be separated into their own source-scoped claim rather than mixed with a frozen telemetry measurement.

## 4. Confirmatory inferential unit

The frozen case manifest already records `group_id`. R15 makes that field the preregistered inferential unit.

Case-level outcomes remain available as descriptive audit data:

- corrected case count;
- induced-error case count;
- net diagnostic gain;
- legacy exact case-level McNemar p-value.

However the legacy McNemar value is explicitly marked:

`DESCRIPTIVE_ONLY_CLUSTERING_IGNORED`.

It is not used by the R15 promotion gate.

The scientific reason is straightforward: ordinary McNemar assumes independent matched pairs. When multiple paired outcomes occur inside a cluster, ignoring within-cluster dependence can misstate variance and Type-I error. Clustered matched-pair methods are a distinct statistical problem in the literature.

Relevant background:

1. Durkalski VL, Palesch YY, Lipsitz SR, Rust PF. *Analysis of clustered matched-pair data*. Statistics in Medicine 22 (2003), 2417–2428. DOI: https://doi.org/10.1002/sim.1438
2. Yang Z, Sun X, Hardin JW. *Confidence intervals for the difference of marginal probabilities in clustered matched-pair binary data*. Pharmaceutical Statistics 11 (2012), 386–393. DOI: https://doi.org/10.1002/pst.1523

R15 does not claim to reproduce either published adjusted-McNemar procedure exactly.

## 5. R15 group-aware estimand

For each independent `group_id`, R15 computes:

`paired_accuracy_delta_g = (corrected_cases_g - induced_errors_g) / case_count_g`.

The primary effect estimate is the equal-group-weighted mean:

`mean_g(paired_accuracy_delta_g)`.

This prevents a group containing more fault-class cases from automatically becoming more statistically influential simply because it contributes more correlated rows.

Reported inferential fields include:

- `independent_group_count`;
- `positive_group_count`;
- `negative_group_count`;
- `tie_group_count`;
- `group_mean_accuracy_delta`;
- `group_mean_accuracy_delta_ci95_lower`;
- `group_mean_accuracy_delta_ci95_upper`;
- `group_mean_accuracy_delta_ci95_half_width`;
- `group_direction_exact_two_sided_p`.

### 5.1 Exact directional test

A group contributes one direction only:

- positive: net improvement within the group;
- negative: net regression within the group;
- tie: zero net direction.

The exact two-sided sign/binomial test is then applied to positive versus negative independent groups. Ties remain in the effect-size estimate but do not become artificial discordant observations.

This is a deliberately conservative blocked-group design. It requires only the already frozen `group_id`, adds no external statistics dependency, and does not pretend that within-group rows are independent.

### 5.2 Effect interval

The effect interval is a 95% Student-t interval across the independent group-level accuracy deltas. It is clipped to the natural [-1, 1] effect range.

This interval is not represented as an exact clustered-McNemar CI. It is the preregistered engineering effect/precision interval for the equal-group-weighted estimand.

## 6. Frozen R15 engineering thresholds

The R15 v3 confirmatory design freezes before real execution:

- maximum induced errors: 0;
- minimum independent groups: 12;
- minimum equal-group mean paired accuracy gain: 0.15;
- 95% CI lower bound must be > 0;
- 95% CI half-width must be <= 0.20;
- exact group-direction p must be <= 0.05;
- candidate recall must remain 1.0 for the full-stack primary gate;
- evidence citation validity must remain 1.0 for the full-stack primary gate.

The 0.15 floor is an engineering review threshold, not a flight-calibrated utility threshold. It approximately preserves the practical magnitude implied by the previous all-benefit significance-clearing pattern: six one-direction improvements among at most 38 independent groups is 6/38 ≈ 0.158.

R15 explicitly rejects retrospective power justification. If the number of independent groups or achieved CI precision is inadequate, the correct response is a newly preregistered expanded independent-group campaign, not case deletion or post-result threshold adjustment.

## 7. Separate Probe-specific estimand

Full controlled stack:

`A2_CONTROLLED_FULL vs A0_ALGORITHM_ONLY`

answers whether the entire controlled LLM-assisted method improves on the deterministic algorithm baseline under the frozen engineering design.

It does **not** isolate Active Probe contribution.

R15 separately preregisters:

`A2_CONTROLLED_FULL vs A7_NO_ACTIVE_PROBE`.

Only this A2-vs-A7 comparison may inform a future Probe formal-evidence-producer review.

Its gate additionally requires at least six independent `probe_exposed_group_count` groups. A passing result is reported only as:

`ELIGIBLE_FOR_SEPARATE_FORMAL_EVIDENCE_REVIEW`.

It does not change:

- `formal_evidence_eligible=false` for current shadow Probe;
- any EvidenceFact producer registration;
- formal diagnosis/release authority.

An independent governance review would still be required after the real campaign.

## 8. Deterministic counterexample retained as regression

Six corrected cases in six independent groups:

- case-level exact McNemar: 0.03125;
- group-direction exact sign test: 0.03125.

The same six corrected cases placed inside one shared `group_id`:

- case-level exact McNemar remains 0.03125;
- independent-group count becomes 1;
- group-direction exact sign test becomes 1.0;
- 95% group effect interval is unavailable because a one-group sample cannot estimate between-group variance.

This counterexample is frozen in automated tests and demonstrates why case-level significance is no longer used for promotion.

## 9. Remaining research boundary

R15 resolves the R14 P0 design blockers for an **engineering confirmatory** campaign. It does not resolve the remaining R14 P1/P2 findings, including:

- full model/tokenizer/container/GPU/software execution identity;
- HIL / target-hardware / operational-shadow validation;
- OOD/domain-shift qualification;
- heuristic probability semantics/calibration;
- temporal false-alarm/delay calibration;
- supplemental acquisition pedigree;
- BiRRF sensitivity and KG/source-dependence validation;
- multi-fault scope and operational FDIR metrics.

Those remain explicit later gates. Real A0–A7 execution is still blocked until the frozen external datasets, v3 case manifest, exact live model identity and existing snapshot/read-only admission requirements are satisfied.
