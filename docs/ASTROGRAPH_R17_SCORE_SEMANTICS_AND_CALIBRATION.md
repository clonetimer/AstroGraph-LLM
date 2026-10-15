# AstroGraph R17 — Diagnostic Score Semantics and Calibration Audit

Date: 2026-09-18

## Scope

R17 addresses `R14-MOD-001` and begins `R14-FUS-001`. It does **not** fit a new calibrator or claim that BiRRF is calibrated. The objective is to make each numeric quantity state what it means, prevent engineering ranking mass from being represented as probability, require content-bound calibration evidence before a formal model-promotion gate treats a score as calibrated, and provide a frozen BiRRF parameter-sensitivity audit surface.

## Score taxonomy

AstroGraph now distinguishes:

- `CALIBRATED_CLASS_PROBABILITY`: class probability with a calibration identity; formal promotion additionally requires a content-bound calibration receipt.
- `UNCALIBRATED_CLASS_PROBABILITY`: probability-like model output whose empirical correctness calibration has not been qualified.
- `NORMALIZED_RANKING_MASS`: normalized nonnegative/logit/anomaly-derived class mass used to preserve rank; **not a probability claim**.
- `EVIDENCE_STRENGTH` / `EVIDENCE_COVERAGE`: deterministic KG/evidence ranking quantities.
- `RECIPROCAL_RANK_SCORE_NOT_PROBABILITY`: BiRRF fused score.

`FaultRanking` now rejects out-of-range values for probability/normalized-mass semantics. `BiRRFResult` exposes fused-score semantics explicitly. A legacy `normal_probability` field is suppressed when its source is only `NORMALIZED_RANKING_MASS`; the normal rank remains available for nominal ordering logic.

## SAT-SIM engineering screening correction

The built-in RW screening provider previously calculated deterministic engineering strengths, applied softmax, and declared `PROBABILITIES`. R17 changes it to emit `NONNEGATIVE_SCORES`; the common adapter normalizes them only for ranking and marks the result `NORMALIZED_RANKING_MASS`. The hard-coded `uncertainty=0.65` was removed because it had no calibrated uncertainty semantics.

This is a semantic correction, not a new classifier. The monotone class ordering remains governed by the same deterministic engineering response equations.

## Calibration receipt

R17 adds `astrograph-probability-calibration-receipt.v1`, binding:

- calibration ID;
- model ID/version;
- calibration dataset ID;
- calibration split SHA256;
- calibration method;
- calibration report SHA256;
- the validation expected-calibration-error value actually used by the promotion threshold;
- calibration sample and independent-group counts.

A model manifest may retain legacy `calibration_id` without a receipt for compatibility, but the runtime score audit marks that state `LEGACY_ID_ONLY_NOT_CONTENT_BOUND`. The RW_JAM formal promotion gate now requires both the receipt and its SHA256 binding; a bare calibration ID is insufficient. The separately supplied promotion ECE must exactly match the receipt-bound ECE, and the gate evaluates the receipt-bound value, preventing a detached scalar from silently passing a different threshold.

Calibration here means empirical probability calibration on the frozen calibration material. It does not establish transfer under distribution shift, OOD reliability, HIL validity or flight qualification.

## Executable score-semantics audit

`astrograph.evaluation.score_semantics.audit_score_semantics()` cross-checks model manifest, adapted prediction, data ranking and optional BiRRF result. It records whether a score is eligible for a probability-threshold gate and produces a canonical audit hash.

The R17 SAT-SIM receipt demonstrates:

- output: `NONNEGATIVE_SCORES`;
- class values: `NORMALIZED_RANKING_MASS`;
- calibrated: false;
- probability-threshold eligible: false;
- fusion: `RECIPROCAL_RANK_SCORE_NOT_PROBABILITY`.

## BiRRF sensitivity

RRF is rank aggregation, not probability fusion. R17 adds an offline deterministic sensitivity audit plus a frozen engineering grid over branch weights and the RRF smoothing constant. The audit reports top-1/rank-order stability only; it does not choose an optimal parameter or claim performance calibration.

A synthetic counterexample is intentionally retained: changing only the data-branch weight can change the fused top candidate. Therefore the default `1/1/1, k=60` configuration remains an engineering default whose robustness must be evaluated on frozen independent groups before any calibrated/qualified profile is created.

## Research basis

The terminology follows the distinction between confidence values and empirically calibrated correctness probabilities. Guo et al. (2017), *On Calibration of Modern Neural Networks*, showed that modern classifier confidence can be substantially miscalibrated and that post-hoc calibration must be evaluated rather than inferred from softmax alone. RRF originates as a rank-fusion method (Cormack, Clarke & Büttcher, 2009), and later analysis by Bruch, Gai & Ingber (2022) found RRF can be sensitive to fusion parameters. These works motivate the semantic and sensitivity controls; they do not constitute validation of AstroGraph parameters.

References:

1. Guo C, Pleiss G, Sun Y, Weinberger KQ. 2017. *On Calibration of Modern Neural Networks*. https://consensus.app/papers/on-calibration-of-modern-neural-networks-guo-pleiss/a74254376a29509cb850a8feed18b789/
2. Cormack G, Clarke C, Büttcher S. 2009. *Reciprocal rank fusion outperforms condorcet and individual rank learning methods*. https://consensus.app/papers/reciprocal-rank-fusion-outperforms-condorcet-and-cormack-clarke/4c152694303a5ee2963dbb0e2e960b47/
3. Bruch S, Gai S, Ingber A. 2022. *An Analysis of Fusion Functions for Hybrid Retrieval*. https://consensus.app/papers/an-analysis-of-fusion-functions-for-hybrid-retrieval-bruch-gai/36c995fadc625475863f9469fb29f9b1/

## Remaining R17 boundary

R17 does **not** close real probability calibration or fusion-parameter selection. Still required:

- group-aware calibration reliability/CI on frozen external data;
- classwise calibration and Brier/NLL/ECE reporting with sufficient independent groups;
- calibration-transfer testing under simulator/mission-condition shift;
- OOD/selective-risk evaluation;
- preregistered real-dataset BiRRF weight/smoothing sensitivity and simpler baselines;
- KG edge/source dependence and leave-source-out analysis.
