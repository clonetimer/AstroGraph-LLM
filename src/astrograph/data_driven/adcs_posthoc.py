"""Post-hoc calibration and input-novelty qualification for frozen ADCS models.

The protocol is deliberately group-isolated:
- model weights/preprocessing are frozen;
- train windows fit only the input-novelty reference distribution;
- validation trajectory groups are split deterministically into calibration and
  qualification groups;
- no test labels or test windows are required to fit the artifact.

Synthetic perturbations are stress evidence for the novelty score only.  They
are not a substitute for independent real-world OOD datasets.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.data_driven.credibility import CalibrationMetrics, TemperatureScaler, calibration_metrics


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IndependentCalibrationEvidence(_Frozen):
    temperature: float = Field(gt=0.0)
    calibration_before: CalibrationMetrics
    calibration_after: CalibrationMetrics
    qualification_before: CalibrationMetrics
    qualification_after: CalibrationMetrics
    calibration_sample_count: int = Field(ge=1)
    qualification_sample_count: int = Field(ge=1)


class DetectionThresholdEvidence(_Frozen):
    threshold: float = Field(ge=0.0, le=1.0)
    calibration_macro_f1: float = Field(ge=0.0, le=1.0)
    qualification_macro_f1: float = Field(ge=0.0, le=1.0)
    reference_threshold_macro_f1: float = Field(ge=0.0, le=1.0)
    selection_scope: str = "CALIBRATION_GROUPS_ONLY"


class RobustSummaryOODProfile(_Frozen):
    method: str = "ROBUST_DIAGONAL_SUMMARY_DISTANCE_EMPIRICAL_CDF"
    center: tuple[float, ...]
    scale: tuple[float, ...]
    reference_distance_quantiles: tuple[float, ...]
    reference_quantile_levels: tuple[float, ...]
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_quantile: float = Field(gt=0.0, lt=1.0)
    calibration_id_false_reject_rate: float = Field(ge=0.0, le=1.0)
    qualification_id_false_reject_rate: float = Field(ge=0.0, le=1.0)
    synthetic_ood_recall_by_perturbation: dict[str, float]
    train_reference_sample_count: int = Field(ge=100)
    calibration_id_sample_count: int = Field(ge=1)
    qualification_id_sample_count: int = Field(ge=1)
    claim_boundary: str = (
        "Input novelty is calibrated from frozen train/validation ADCS windows. Synthetic perturbation recall is "
        "a deterministic stress test and is not evidence of real-spacecraft or cross-domain OOD generalization."
    )

    @model_validator(mode="after")
    def dimensions(self):
        if not self.center or len(self.center) != len(self.scale):
            raise ValueError("OOD center/scale dimensions must match and be non-empty")
        if len(self.reference_distance_quantiles) != len(self.reference_quantile_levels):
            raise ValueError("OOD quantile levels/values must have equal length")
        if any(b < a for a, b in zip(self.reference_quantile_levels, self.reference_quantile_levels[1:])):
            raise ValueError("OOD quantile levels must be sorted")
        if any(b < a for a, b in zip(self.reference_distance_quantiles, self.reference_distance_quantiles[1:])):
            raise ValueError("OOD quantile values must be sorted")
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("OOD robust scales must be positive")
        return self


class ADCSPosthocQualificationArtifact(_Frozen):
    schema_version: str = "astrograph-adcs-posthoc-qualification.v1"
    qualification_id: str
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_marker_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    channel_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_group_ids: tuple[str, ...]
    qualification_group_ids: tuple[str, ...]
    group_overlap_count: int = Field(ge=0)
    joint_probability_calibration: IndependentCalibrationEvidence
    localization_calibration: IndependentCalibrationEvidence
    detection_threshold: DetectionThresholdEvidence
    ood_profile: RobustSummaryOODProfile
    test_partition_used_for_fitting: bool = False
    weights_modified: bool = False
    claim_boundary: str = (
        "Post-hoc engineering qualification for the frozen ADCS v5.2.2 distribution. It does not repair the "
        "separate five-fold production metric blockers and does not establish flight or cross-satellite validity."
    )

    @model_validator(mode="after")
    def group_isolation(self):
        if set(self.calibration_group_ids) & set(self.qualification_group_ids):
            raise ValueError("calibration and qualification groups must be disjoint")
        if self.group_overlap_count != 0:
            raise ValueError("group overlap must be zero")
        if self.test_partition_used_for_fitting or self.weights_modified:
            raise ValueError("post-hoc qualification cannot use test fitting or modify weights")
        return self


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return TemperatureScaler(float(temperature)).transform(np.asarray(probabilities, dtype=float))


def robust_summary_statistics(summary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(summary, dtype=float)
    if values.ndim != 2 or len(values) < 100 or not np.isfinite(values).all():
        raise ValueError("finite [N,F] train summary with at least 100 rows required")
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
    return center.astype(float), scale.astype(float)


def robust_summary_distance(summary: np.ndarray, center: Sequence[float], scale: Sequence[float]) -> np.ndarray:
    values = np.asarray(summary, dtype=float)
    c = np.asarray(center, dtype=float)
    s = np.asarray(scale, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(c) or c.shape != s.shape:
        raise ValueError("summary/profile dimension mismatch")
    z = (values - c) / s
    # RMS robust z is stable in high dimensions and remains sensitive to distributed drift.
    return np.sqrt(np.mean(np.square(z), axis=1))


def ood_scores_from_profile(summary: np.ndarray, profile: RobustSummaryOODProfile | Mapping[str, Any]) -> np.ndarray:
    profile = profile if isinstance(profile, RobustSummaryOODProfile) else RobustSummaryOODProfile.model_validate(profile)
    distances = robust_summary_distance(summary, profile.center, profile.scale)
    qv = np.asarray(profile.reference_distance_quantiles, dtype=float)
    ql = np.asarray(profile.reference_quantile_levels, dtype=float)
    # Linear interpolation yields a deterministic empirical-CDF-like novelty score.
    return np.clip(np.interp(distances, qv, ql, left=0.0, right=1.0), 0.0, 1.0)


def _macro_f1_binary(truth: np.ndarray, predicted: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int), labels=[0, 1], average="macro", zero_division=0))


def select_detection_threshold(probability: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    probability = np.asarray(probability, dtype=float)
    truth = np.asarray(truth, dtype=int)
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        score = _macro_f1_binary(truth, probability >= threshold)
        candidate = (score, -abs(float(threshold) - 0.5))
        current = (best[0], -abs(best[1] - 0.5))
        if candidate > current:
            best = (score, float(threshold))
    return best[1], best[0]


def _independent_temperature(
    probabilities: np.ndarray,
    truth: np.ndarray,
    calibration_mask: np.ndarray,
    qualification_mask: np.ndarray,
    *,
    calibration_id: str,
) -> tuple[TemperatureScaler, IndependentCalibrationEvidence]:
    scaler, record = TemperatureScaler.fit(
        probabilities[calibration_mask], truth[calibration_mask], calibration_id=calibration_id
    )
    before_q = calibration_metrics(probabilities[qualification_mask], truth[qualification_mask])
    after_q = calibration_metrics(scaler.transform(probabilities[qualification_mask]), truth[qualification_mask])
    return scaler, IndependentCalibrationEvidence(
        temperature=scaler.temperature,
        calibration_before=record.validation_before,
        calibration_after=record.validation_after,
        qualification_before=before_q,
        qualification_after=after_q,
        calibration_sample_count=int(calibration_mask.sum()),
        qualification_sample_count=int(qualification_mask.sum()),
    )


def fit_posthoc_qualification(
    *,
    model_artifact_sha256: str,
    dataset_marker_sha256: str,
    channel_names: Sequence[str],
    validation_group_ids: Sequence[str],
    validation_detection_truth: np.ndarray,
    validation_component_truth: np.ndarray,
    validation_fault_truth: np.ndarray,
    joint_probabilities: np.ndarray,
    component_probabilities: np.ndarray,
    train_summary: np.ndarray,
    validation_summary: np.ndarray,
    synthetic_ood_summaries: Mapping[str, np.ndarray],
    threshold_quantile: float = 0.995,
) -> ADCSPosthocQualificationArtifact:
    groups = np.asarray(list(map(str, validation_group_ids)))
    if len(groups) != len(joint_probabilities):
        raise ValueError("validation group count must match probabilities")
    unique = sorted(set(groups.tolist()))
    if len(unique) < 4:
        raise ValueError("at least four validation trajectory groups are required")
    calibration_groups = tuple(unique[::2])
    qualification_groups = tuple(unique[1::2])
    if not qualification_groups:  # defensive for unusual small inputs
        calibration_groups = tuple(unique[: len(unique)//2])
        qualification_groups = tuple(unique[len(unique)//2 :])
    cmask = np.isin(groups, calibration_groups)
    qmask = np.isin(groups, qualification_groups)
    if np.any(cmask & qmask) or not cmask.any() or not qmask.any():
        raise ValueError("invalid deterministic group split")

    dtruth = np.asarray(validation_detection_truth, dtype=int)
    ctruth = np.asarray(validation_component_truth, dtype=int)
    ftruth = np.asarray(validation_fault_truth, dtype=int)
    joint_truth = np.where(dtruth == 0, 0, ftruth + 1).astype(int)
    if joint_probabilities.shape[1] != int(joint_truth.max()) + 1:
        raise ValueError("joint probability class dimension does not match fault truth")

    joint_scaler, joint_evidence = _independent_temperature(
        np.asarray(joint_probabilities, dtype=float), joint_truth, cmask, qmask,
        calibration_id="adcs-p2-joint-temperature-v1",
    )
    active = dtruth == 1
    local_cmask = cmask & active
    local_qmask = qmask & active
    local_scaler, local_evidence = _independent_temperature(
        np.asarray(component_probabilities, dtype=float), ctruth, local_cmask, local_qmask,
        calibration_id="adcs-p2-localization-temperature-v1",
    )

    calibrated_joint = joint_scaler.transform(joint_probabilities)
    anomaly_probability = 1.0 - calibrated_joint[:, 0]
    threshold, cal_f1 = select_detection_threshold(anomaly_probability[cmask], dtruth[cmask])
    q_f1 = _macro_f1_binary(dtruth[qmask], anomaly_probability[qmask] >= threshold)
    reference_f1 = _macro_f1_binary(dtruth[qmask], anomaly_probability[qmask] >= 0.5)

    center, scale = robust_summary_statistics(train_summary)
    train_distance = robust_summary_distance(train_summary, center, scale)
    levels = np.linspace(0.0, 1.0, 1001)
    quantiles = np.quantile(train_distance, levels)
    provisional = RobustSummaryOODProfile(
        center=tuple(map(float, center)), scale=tuple(map(float, scale)),
        reference_distance_quantiles=tuple(map(float, quantiles)),
        reference_quantile_levels=tuple(map(float, levels)), threshold=0.999,
        threshold_quantile=threshold_quantile, calibration_id_false_reject_rate=0.0,
        qualification_id_false_reject_rate=0.0, synthetic_ood_recall_by_perturbation={},
        train_reference_sample_count=len(train_summary), calibration_id_sample_count=int(cmask.sum()),
        qualification_id_sample_count=int(qmask.sum()),
    )
    id_scores = ood_scores_from_profile(validation_summary, provisional)
    threshold_ood = float(np.quantile(id_scores[cmask], threshold_quantile))
    threshold_ood = min(max(threshold_ood, 0.90), 1.0)
    cal_frr = float(np.mean(id_scores[cmask] > threshold_ood))
    qual_frr = float(np.mean(id_scores[qmask] > threshold_ood))
    perturbation_recall = {}
    for name, values in sorted(synthetic_ood_summaries.items()):
        scores = ood_scores_from_profile(np.asarray(values, dtype=float), provisional)
        perturbation_recall[str(name)] = float(np.mean(scores > threshold_ood))
    ood = provisional.model_copy(update={
        "threshold": threshold_ood,
        "calibration_id_false_reject_rate": cal_frr,
        "qualification_id_false_reject_rate": qual_frr,
        "synthetic_ood_recall_by_perturbation": perturbation_recall,
    })
    qualification_id = "ADCS-POSTHOC-" + _canonical_sha({
        "model": model_artifact_sha256,
        "dataset": dataset_marker_sha256,
        "channels": list(channel_names),
        "cal_groups": calibration_groups,
        "qual_groups": qualification_groups,
        "joint_temperature": joint_scaler.temperature,
        "localization_temperature": local_scaler.temperature,
        "detection_threshold": threshold,
        "ood_threshold": threshold_ood,
    })[:20].upper()
    return ADCSPosthocQualificationArtifact(
        qualification_id=qualification_id,
        model_artifact_sha256=model_artifact_sha256,
        dataset_marker_sha256=dataset_marker_sha256,
        channel_order_sha256=_canonical_sha(list(map(str, channel_names))),
        calibration_group_ids=calibration_groups,
        qualification_group_ids=qualification_groups,
        group_overlap_count=0,
        joint_probability_calibration=joint_evidence,
        localization_calibration=local_evidence,
        detection_threshold=DetectionThresholdEvidence(
            threshold=threshold, calibration_macro_f1=cal_f1,
            qualification_macro_f1=q_f1, reference_threshold_macro_f1=reference_f1,
        ),
        ood_profile=ood,
    )


__all__ = [
    "ADCSPosthocQualificationArtifact", "DetectionThresholdEvidence",
    "IndependentCalibrationEvidence", "RobustSummaryOODProfile",
    "apply_temperature", "fit_posthoc_qualification", "ood_scores_from_profile",
    "robust_summary_distance", "robust_summary_statistics", "select_detection_threshold",
]

# ---------------------------------------------------------------------------
# v2 qualification: group-CV uniform shrinkage + train-only PCA novelty.
# The v1 temperature/robust-diagonal protocol is retained so rejected receipts
# remain readable; v2 is the preferred production-candidate qualification path.
# ---------------------------------------------------------------------------

class GroupCalibrationFold(_Frozen):
    holdout_group_id: str
    fitted_uniform_alpha: float = Field(ge=0.0, le=0.25)
    before: CalibrationMetrics
    after: CalibrationMetrics


class UniformShrinkageCalibrationEvidence(_Frozen):
    method: str = "UNIFORM_CONFIDENCE_SHRINKAGE_SELECTED_BY_NLL"
    final_uniform_alpha: float = Field(ge=0.0, le=0.25)
    folds: tuple[GroupCalibrationFold, ...]
    mean_nll_before: float = Field(ge=0.0)
    mean_nll_after: float = Field(ge=0.0)
    mean_brier_before: float = Field(ge=0.0)
    mean_brier_after: float = Field(ge=0.0)
    mean_ece_before: float = Field(ge=0.0)
    mean_ece_after: float = Field(ge=0.0)
    all_validation_before: CalibrationMetrics
    all_validation_after: CalibrationMetrics
    nll_improved_all_folds: bool
    brier_mean_within_tolerance: bool
    ece_mean_within_tolerance: bool
    qualified: bool
    claim_boundary: str = (
        "Calibration method is selected and independently checked by leave-one-trajectory-group-out validation. "
        "The final alpha is then fitted on the full frozen validation partition; the formal test partition remains untouched."
    )


class PCASummaryOODProfile(_Frozen):
    method: str = "STANDARDIZED_PCA_RECONSTRUCTION_PLUS_LATENT_ENERGY_EMPIRICAL_CDF"
    standard_mean: tuple[float, ...]
    standard_scale: tuple[float, ...]
    pca_mean: tuple[float, ...]
    pca_components: tuple[tuple[float, ...], ...]
    pca_explained_variance: tuple[float, ...]
    reference_distance_quantiles: tuple[float, ...]
    reference_quantile_levels: tuple[float, ...]
    threshold: float = Field(ge=0.0, le=1.0)
    validation_id_false_reject_rate: float = Field(ge=0.0, le=1.0)
    primary_stress_recall: dict[str, float]
    secondary_stress_recall: dict[str, float]
    train_reference_sample_count: int = Field(ge=100)
    validation_sample_count: int = Field(ge=1)
    pca_component_count: int = Field(ge=1)
    qualified: bool
    claim_boundary: str = (
        "OOD_SCORE covers the qualified ADCS window-novelty threat model: hidden channel/value semantic mismatch and "
        "distributed sensor-offset drift. Secondary stress results are diagnostic only. This is not evidence for arbitrary, "
        "cross-satellite, adversarial, or flight OOD detection."
    )

    @model_validator(mode="after")
    def pca_dimensions(self):
        width = len(self.standard_mean)
        if width == 0 or len(self.standard_scale) != width or len(self.pca_mean) != width:
            raise ValueError("PCA OOD standardization dimensions must match")
        if any(value <= 0 for value in self.standard_scale):
            raise ValueError("PCA OOD standard scales must be positive")
        if len(self.pca_components) != len(self.pca_explained_variance) or len(self.pca_components) != self.pca_component_count:
            raise ValueError("PCA OOD component metadata mismatch")
        if any(len(row) != width for row in self.pca_components):
            raise ValueError("PCA OOD component width mismatch")
        if len(self.reference_distance_quantiles) != len(self.reference_quantile_levels):
            raise ValueError("PCA OOD reference quantiles mismatch")
        return self


class ADCSPosthocQualificationArtifactV2(_Frozen):
    schema_version: str = "astrograph-adcs-posthoc-qualification.v2"
    qualification_id: str
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_marker_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    channel_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_group_ids: tuple[str, ...]
    joint_probability_calibration: UniformShrinkageCalibrationEvidence
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    detection_threshold_policy: str = "FROZEN_0P5_GROUP_CV_RETUNING_DID_NOT_IMPROVE"
    ood_profile: PCASummaryOODProfile
    test_partition_used_for_fitting: bool = False
    weights_modified: bool = False
    probability_calibration_qualified: bool
    ood_score_qualified: bool
    formal_probability_and_ood_eligible: bool
    claim_boundary: str = (
        "Engineering post-hoc qualification for the frozen ADCS v5.2.2 model/distribution. Probability calibration and "
        "the bounded OOD threat model are independent of, and do not override, the separate five-fold production metric gate."
    )

    @model_validator(mode="after")
    def consistency(self):
        if self.test_partition_used_for_fitting or self.weights_modified:
            raise ValueError("v2 post-hoc qualification must not fit on test or modify weights")
        if self.probability_calibration_qualified != self.joint_probability_calibration.qualified:
            raise ValueError("probability qualification flag mismatch")
        if self.ood_score_qualified != self.ood_profile.qualified:
            raise ValueError("OOD qualification flag mismatch")
        if self.formal_probability_and_ood_eligible != (self.probability_calibration_qualified and self.ood_score_qualified):
            raise ValueError("formal eligibility flag mismatch")
        return self


def uniform_shrink(probabilities: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not 0.0 <= alpha <= 0.25:
        raise ValueError("uniform shrinkage requires [N,K] probabilities and alpha in [0,0.25]")
    values = np.clip(values, 0.0, 1.0)
    values = values / np.maximum(values.sum(1, keepdims=True), 1e-12)
    return (1.0 - float(alpha)) * values + float(alpha) / values.shape[1]


def _select_uniform_alpha(probabilities: np.ndarray, truth: np.ndarray) -> float:
    best = (float("inf"), 0.0)
    for alpha in np.linspace(0.0, 0.10, 101):
        metric = calibration_metrics(uniform_shrink(probabilities, float(alpha)), truth)
        candidate = (metric.negative_log_likelihood, float(alpha))
        if candidate < best:
            best = candidate
    return best[1]


def fit_group_cv_uniform_calibration(
    probabilities: np.ndarray,
    truth: np.ndarray,
    group_ids: Sequence[str],
    *,
    brier_tolerance: float = 0.001,
    ece_tolerance: float = 0.005,
) -> UniformShrinkageCalibrationEvidence:
    probabilities = np.asarray(probabilities, dtype=float)
    truth = np.asarray(truth, dtype=int)
    groups = np.asarray(list(map(str, group_ids)))
    if len(probabilities) != len(truth) or len(truth) != len(groups):
        raise ValueError("probability/truth/group length mismatch")
    unique = sorted(set(groups.tolist()))
    if len(unique) < 4:
        raise ValueError("at least four validation groups are required")
    folds = []
    for holdout in unique:
        fit = groups != holdout
        test = groups == holdout
        alpha = _select_uniform_alpha(probabilities[fit], truth[fit])
        before = calibration_metrics(probabilities[test], truth[test])
        after = calibration_metrics(uniform_shrink(probabilities[test], alpha), truth[test])
        folds.append(GroupCalibrationFold(holdout_group_id=holdout, fitted_uniform_alpha=alpha, before=before, after=after))
    nll_before = float(np.mean([x.before.negative_log_likelihood for x in folds]))
    nll_after = float(np.mean([x.after.negative_log_likelihood for x in folds]))
    brier_before = float(np.mean([x.before.multiclass_brier_score for x in folds]))
    brier_after = float(np.mean([x.after.multiclass_brier_score for x in folds]))
    ece_before = float(np.mean([x.before.expected_calibration_error for x in folds]))
    ece_after = float(np.mean([x.after.expected_calibration_error for x in folds]))
    nll_all = all(x.after.negative_log_likelihood < x.before.negative_log_likelihood for x in folds)
    brier_ok = brier_after <= brier_before + brier_tolerance
    ece_ok = ece_after <= ece_before + ece_tolerance
    final_alpha = _select_uniform_alpha(probabilities, truth)
    before_all = calibration_metrics(probabilities, truth)
    after_all = calibration_metrics(uniform_shrink(probabilities, final_alpha), truth)
    qualified = bool(nll_all and brier_ok and ece_ok and final_alpha <= 0.05)
    return UniformShrinkageCalibrationEvidence(
        final_uniform_alpha=final_alpha,
        folds=tuple(folds),
        mean_nll_before=nll_before, mean_nll_after=nll_after,
        mean_brier_before=brier_before, mean_brier_after=brier_after,
        mean_ece_before=ece_before, mean_ece_after=ece_after,
        all_validation_before=before_all, all_validation_after=after_all,
        nll_improved_all_folds=nll_all,
        brier_mean_within_tolerance=brier_ok,
        ece_mean_within_tolerance=ece_ok,
        qualified=qualified,
    )


def _pca_distance(summary: np.ndarray, profile: PCASummaryOODProfile) -> np.ndarray:
    values = np.asarray(summary, dtype=float)
    mean = np.asarray(profile.standard_mean, dtype=float)
    scale = np.asarray(profile.standard_scale, dtype=float)
    pca_mean = np.asarray(profile.pca_mean, dtype=float)
    components = np.asarray(profile.pca_components, dtype=float)
    variance = np.asarray(profile.pca_explained_variance, dtype=float)
    standardized = (values - mean) / scale
    centered = standardized - pca_mean
    latent = centered @ components.T
    reconstructed = latent @ components + pca_mean
    reconstruction_error = np.mean(np.square(standardized - reconstructed), axis=1)
    latent_energy = np.mean(np.square(latent / np.sqrt(np.maximum(variance, 1e-9))), axis=1)
    return reconstruction_error + 0.05 * latent_energy


def pca_ood_scores(summary: np.ndarray, profile: PCASummaryOODProfile | Mapping[str, Any]) -> np.ndarray:
    profile = profile if isinstance(profile, PCASummaryOODProfile) else PCASummaryOODProfile.model_validate(profile)
    distances = _pca_distance(summary, profile)
    qv = np.asarray(profile.reference_distance_quantiles, dtype=float)
    ql = np.asarray(profile.reference_quantile_levels, dtype=float)
    return np.clip(np.interp(distances, qv, ql, left=0.0, right=1.0), 0.0, 1.0)


def fit_pca_ood_profile(
    *,
    train_summary: np.ndarray,
    validation_summary: np.ndarray,
    primary_stress_summaries: Mapping[str, np.ndarray],
    secondary_stress_summaries: Mapping[str, np.ndarray] | None = None,
    threshold: float = 0.997,
    explained_variance: float = 0.98,
) -> PCASummaryOODProfile:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    train = np.asarray(train_summary, dtype=float)
    validation = np.asarray(validation_summary, dtype=float)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("train/validation summary dimensions must match")
    scaler = StandardScaler().fit(train)
    scaled = scaler.transform(train)
    pca = PCA(n_components=explained_variance, svd_solver="full", random_state=0).fit(scaled)

    placeholder = PCASummaryOODProfile(
        standard_mean=tuple(map(float, scaler.mean_)),
        standard_scale=tuple(map(float, scaler.scale_)),
        pca_mean=tuple(map(float, pca.mean_)),
        pca_components=tuple(tuple(map(float, row)) for row in pca.components_),
        pca_explained_variance=tuple(map(float, pca.explained_variance_)),
        reference_distance_quantiles=(0.0, 1.0), reference_quantile_levels=(0.0, 1.0),
        threshold=threshold, validation_id_false_reject_rate=0.0,
        primary_stress_recall={}, secondary_stress_recall={},
        train_reference_sample_count=len(train), validation_sample_count=len(validation),
        pca_component_count=int(pca.n_components_), qualified=False,
    )
    distances = _pca_distance(train, placeholder)
    levels = np.linspace(0.0, 1.0, 1001)
    quantiles = np.quantile(distances, levels)
    profile = placeholder.model_copy(update={
        "reference_distance_quantiles": tuple(map(float, quantiles)),
        "reference_quantile_levels": tuple(map(float, levels)),
    })
    id_scores = pca_ood_scores(validation, profile)
    id_frr = float(np.mean(id_scores > threshold))
    primary = {str(k): float(np.mean(pca_ood_scores(np.asarray(v, dtype=float), profile) > threshold)) for k, v in sorted(primary_stress_summaries.items())}
    secondary = {str(k): float(np.mean(pca_ood_scores(np.asarray(v, dtype=float), profile) > threshold)) for k, v in sorted((secondary_stress_summaries or {}).items())}
    qualified = bool(id_frr <= 0.01 and primary and all(value >= 0.95 for value in primary.values()))
    return profile.model_copy(update={
        "validation_id_false_reject_rate": id_frr,
        "primary_stress_recall": primary,
        "secondary_stress_recall": secondary,
        "qualified": qualified,
    })


def fit_posthoc_qualification_v2(
    *,
    model_artifact_sha256: str,
    dataset_marker_sha256: str,
    channel_names: Sequence[str],
    validation_group_ids: Sequence[str],
    validation_detection_truth: np.ndarray,
    validation_fault_truth: np.ndarray,
    joint_probabilities: np.ndarray,
    train_summary: np.ndarray,
    validation_summary: np.ndarray,
    primary_stress_summaries: Mapping[str, np.ndarray],
    secondary_stress_summaries: Mapping[str, np.ndarray] | None = None,
) -> ADCSPosthocQualificationArtifactV2:
    dtruth = np.asarray(validation_detection_truth, dtype=int)
    ftruth = np.asarray(validation_fault_truth, dtype=int)
    joint_truth = np.where(dtruth == 0, 0, ftruth + 1).astype(int)
    calibration = fit_group_cv_uniform_calibration(joint_probabilities, joint_truth, validation_group_ids)
    ood = fit_pca_ood_profile(
        train_summary=train_summary, validation_summary=validation_summary,
        primary_stress_summaries=primary_stress_summaries,
        secondary_stress_summaries=secondary_stress_summaries,
    )
    qualification_id = "ADCS-POSTHOC-V2-" + _canonical_sha({
        "model": model_artifact_sha256,
        "dataset": dataset_marker_sha256,
        "channels": list(channel_names),
        "groups": sorted(set(map(str, validation_group_ids))),
        "alpha": calibration.final_uniform_alpha,
        "ood_threshold": ood.threshold,
        "pca_components": ood.pca_component_count,
    })[:20].upper()
    return ADCSPosthocQualificationArtifactV2(
        qualification_id=qualification_id,
        model_artifact_sha256=model_artifact_sha256,
        dataset_marker_sha256=dataset_marker_sha256,
        channel_order_sha256=_canonical_sha(list(map(str, channel_names))),
        validation_group_ids=tuple(sorted(set(map(str, validation_group_ids)))),
        joint_probability_calibration=calibration,
        ood_profile=ood,
        probability_calibration_qualified=calibration.qualified,
        ood_score_qualified=ood.qualified,
        formal_probability_and_ood_eligible=bool(calibration.qualified and ood.qualified),
    )


__all__ += [
    "ADCSPosthocQualificationArtifactV2", "GroupCalibrationFold", "PCASummaryOODProfile",
    "UniformShrinkageCalibrationEvidence", "fit_group_cv_uniform_calibration",
    "fit_pca_ood_profile", "fit_posthoc_qualification_v2", "pca_ood_scores", "uniform_shrink",
]
