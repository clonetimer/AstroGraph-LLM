"""Temporal operators — zero-fit change detection, missing-run profiles.

All operators are closed-form, no fitting, no thresholds learned from labels.
"""

from __future__ import annotations

import numpy as np

from .contracts import FeatureInput, OperatorResult, nan_safe
from .robust_statistics import _clean_values, _availability


class MultiscaleChangeScoreOperator:
    """MULTISCALE_CHANGE_SCORE — max robust deviation across window scales.

    Computes the largest normalised location shift between left/right halves
    across several scales, using MAD as the scale. Zero-fit, sign-free magnitude.
    """

    operator_id = "multiscale_change_score"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            values = _clean_values(input_)
            avail = _availability(input_)
            n = len(values)
            if n < 8:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=round(avail, 4), quality_flags=["insufficient_samples"],
                    )
                )
                continue

            scales = [int(n * s) for s in (0.1, 0.25, 0.4)]
            scales = [s for s in scales if 2 <= s <= n // 2]
            scores: list[float] = []
            for k in scales:
                left = values[:k]
                right = values[-k:]
                m1 = float(np.median(left))
                m2 = float(np.median(right))
                scale = max(float(np.median(np.abs(left - m1))), float(np.median(np.abs(right - m2))), 1e-12)
                scores.append(abs(m2 - m1) / scale)

            if not scores:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=round(avail, 4), quality_flags=["no_scales"],
                    )
                )
                continue

            best = float(np.max(scores))
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(best),
                    uncertainty=nan_safe(float(np.std(scores))),
                    availability=round(avail, 4),
                    window_descriptor=f"scales={scales}",
                    quality_flags=["change_magnitude"],
                )
            )
        return results


class MissingRunLengthProfileOperator:
    """MISSING_RUN_LENGTH_PROFILE — longest contiguous missing run.

    This is a pure data-quality feature: it must never be caused by label/truth
    presence. It reads only the missing mask.
    """

    operator_id = "missing_run_length_profile"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            mask = np.asarray(input_.missing_mask, dtype=bool)
            n = len(mask)
            avail = _availability(input_)
            if n == 0:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=0.0, quality_flags=["empty"],
                    )
                )
                continue

            longest = 0
            current = 0
            for m in mask:
                if m:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0

            run_count = sum(1 for i in range(1, n) if mask[i] and not mask[i - 1]) + int(mask[0])
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(float(longest)),
                    uncertainty=nan_safe(float(run_count)),
                    availability=round(avail, 4),
                    window_descriptor=f"runs={run_count} longest={longest}",
                    quality_flags=["missing_profile"],
                )
            )
        return results


class AutocorrelationDecayOperator:
    """AUTOCORRELATION_DECAY — robust lag-1 autocorrelation magnitude.

    Used as a thermal inertia / persistence proxy. Zero-fit.
    """

    operator_id = "autocorrelation_decay"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            values = _clean_values(input_)
            avail = _availability(input_)
            n = len(values)
            if n < 4:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=round(avail, 4), quality_flags=["insufficient_samples"],
                    )
                )
                continue
            centered = values - np.mean(values)
            denom = float(np.dot(centered[:-1], centered[:-1]))
            if denom < 1e-12:
                rho = 0.0
            else:
                rho = float(np.dot(centered[:-1], centered[1:]) / denom)
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(rho),
                    uncertainty=0.0,
                    availability=round(avail, 4),
                    window_descriptor=f"n={n}",
                    quality_flags=["lag1"],
                )
            )
        return results


class BurstLossRunProfileOperator:
    """BURST_LOSS_RUN_PROFILE — bursty loss via missing-run length distribution.

    Reports the median run length plus a burstiness proxy (variance of run
    lengths), both zero-fit.
    """

    operator_id = "burst_loss_run_profile"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            mask = np.asarray(input_.missing_mask, dtype=bool)
            avail = _availability(input_)
            runs: list[int] = []
            current = 0
            for m in mask:
                if m:
                    current += 1
                else:
                    if current > 0:
                        runs.append(current)
                    current = 0
            if current > 0:
                runs.append(current)

            if not runs:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=round(avail, 4), quality_flags=["no_loss"],
                    )
                )
                continue

            runs_arr = np.asarray(runs, dtype=float)
            median_run = float(np.median(runs_arr))
            burstiness = float(np.std(runs_arr)) / max(median_run, 1e-12)
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(burstiness),
                    uncertainty=nan_safe(median_run),
                    availability=round(avail, 4),
                    window_descriptor=f"runs={len(runs)} median={median_run:.1f}",
                    quality_flags=["loss_burst"],
                )
            )
        return results


class RegimeChangeProfileOperator:
    """REGIME_CHANGE_PROFILE — coarse regime change via binned level shifts.

    Splits the signal into quarters and reports the max inter-quartile median
    shift, normalised by global MAD. Zero-fit service-regime proxy.
    """

    operator_id = "regime_change_profile"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            values = _clean_values(input_)
            avail = _availability(input_)
            n = len(values)
            if n < 8:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=round(avail, 4), quality_flags=["insufficient_samples"],
                    )
                )
                continue
            quartiles = np.array_split(values, 4)
            medians = [float(np.median(q)) for q in quartiles]
            global_mad = float(np.median(np.abs(values - np.median(values))))
            scale = max(global_mad, 1e-12)
            shift = max(abs(medians[i] - medians[j]) for i in range(4) for j in range(i + 1, 4))
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(shift / scale),
                    uncertainty=nan_safe(global_mad),
                    availability=round(avail, 4),
                    window_descriptor=f"q={[round(m, 3) for m in medians]}",
                    quality_flags=["regime_shift"],
                )
            )
        return results


__all__ = [
    "MultiscaleChangeScoreOperator",
    "MissingRunLengthProfileOperator",
    "AutocorrelationDecayOperator",
    "BurstLossRunProfileOperator",
    "RegimeChangeProfileOperator",
]
