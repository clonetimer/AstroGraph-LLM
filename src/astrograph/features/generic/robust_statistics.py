"""Robust statistics operators — zero-fit (S0), using median/MAD/quantiles.

Every operator here is a closed-form, parameter-free statistic. No fitting, no
thresholds learned from labels, no training. Availability is computed from the
missing mask, never from label information.
"""

from __future__ import annotations

import math

import numpy as np

from .contracts import FeatureInput, OperatorResult, nan_safe


def _clean_values(input_: FeatureInput) -> np.ndarray:
    """Return finite, present values as a float64 array."""
    vals = np.asarray(input_.values, dtype=float)
    mask = np.asarray(input_.missing_mask, dtype=bool)
    present = vals[~mask]
    finite = present[np.isfinite(present)]
    return finite


def _availability(input_: FeatureInput) -> float:
    total = max(len(input_.values), 1)
    present = sum(1 for v, m in zip(input_.values, input_.missing_mask) if not m and v is not None)
    return present / total


def _median_mad(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) if len(values) > 1 else 0.0
    return median, mad


def _iqr(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    q1, q3 = np.percentile(values, [25.0, 75.0])
    return float(q3 - q1)


class RobustLevelDistributionOperator:
    """ROBUST_LEVEL_DISTRIBUTION — median, MAD, IQR of a signal level."""

    operator_id = "robust_level_distribution"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            values = _clean_values(input_)
            median, mad = _median_mad(values)
            avail = _availability(input_)
            base = {
                "operator_id": self.operator_id,
                "availability": round(avail, 4),
                "window_descriptor": f"n={len(values)}",
            }
            results.append(OperatorResult(**base, value=nan_safe(median), uncertainty=nan_safe(mad)))
            results.append(
                OperatorResult(**base, value=nan_safe(_iqr(values)), quality_flags=["interquartile_range"])
            )
        return results


class SignedSlopeDistributionOperator:
    """SIGNED_SLOPE_DISTRIBUTION — robust Theil-Sen slope sign and magnitude.

    Uses median of pairwise slopes on present samples (O(n) via median of
    medians approximation when n is large), signed, zero-fit.
    """

    operator_id = "signed_slope_distribution"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            vals = np.asarray(input_.values, dtype=float)
            times = np.asarray(input_.timestamps_s, dtype=float)
            mask = np.asarray(input_.missing_mask, dtype=bool)
            present_v = vals[~mask]
            present_t = times[~mask]
            finite = np.isfinite(present_v)
            present_v = present_v[finite]
            present_t = present_t[finite]

            avail = _availability(input_)
            if len(present_v) < 3:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=round(avail, 4), window_descriptor="insufficient",
                        quality_flags=["insufficient_samples"],
                    )
                )
                continue

            # Theil-Sen: median slope over sampled pairs (bounded, zero-fit).
            rng = np.random.default_rng(0)  # fixed seed is fine: it is not an identity input
            n = len(present_v)
            max_pairs = 500
            if n * (n - 1) // 2 <= max_pairs:
                slopes: list[float] = []
                for i in range(n):
                    for j in range(i + 1, n):
                        dt = present_t[j] - present_t[i]
                        if abs(dt) < 1e-12:
                            continue
                        slopes.append((present_v[j] - present_v[i]) / dt)
            else:
                idx = rng.choice(n, size=(max_pairs, 2), replace=True)
                slopes = []
                for i, j in idx:
                    dt = present_t[j] - present_t[i]
                    if abs(dt) < 1e-12:
                        continue
                    slopes.append((present_v[j] - present_v[i]) / dt)

            if not slopes:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=round(avail, 4), quality_flags=["zero_dt"],
                    )
                )
                continue

            slope_arr = np.asarray(slopes, dtype=float)
            median_slope = float(np.median(slope_arr))
            mad_slope = float(np.median(np.abs(slope_arr - median_slope))) if len(slope_arr) > 1 else 0.0

            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(median_slope),
                    uncertainty=nan_safe(mad_slope),
                    availability=round(avail, 4),
                    window_descriptor=f"n={len(present_v)}",
                    quality_flags=["theil_sen"],
                )
            )
        return results


class RobustLocalTrendOperator:
    """ROBUST_LOCAL_TREND — sign-consistent drift over halves (no fit).

    Compares robust location of first half vs second half, normalised by MAD.
    """

    operator_id = "robust_local_trend"

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
            half = n // 2
            first = values[:half]
            second = values[half:]
            m1, mad1 = _median_mad(first)
            m2, mad2 = _median_mad(second)
            scale = max(mad1, mad2, 1e-9)
            drift = (m2 - m1) / scale
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(drift),
                    uncertainty=nan_safe(mad1 + mad2),
                    availability=round(avail, 4),
                    window_descriptor=f"halves={half}/{n - half}",
                )
            )
        return results


class BoundaryDwellProfileOperator:
    """BOUNDARY_DWELL_PROFILE — fraction of time dwelling near 0 or 1.

    Requires a bounded signal (fill fraction / state of charge in [0,1]).
    """

    operator_id = "boundary_dwell_profile"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            values = _clean_values(input_)
            avail = _availability(input_)
            if len(values) == 0:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=0.0, quality_flags=["empty"],
                    )
                )
                continue
            frac_low = float(np.mean(values <= 0.02))
            frac_high = float(np.mean(values >= 0.98))
            frac_mid = 1.0 - frac_low - frac_high
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(frac_high - frac_low),
                    uncertainty=nan_safe(math.sqrt(max(frac_mid, 1e-9) / max(len(values), 1))),
                    availability=round(avail, 4),
                    window_descriptor=f"low={frac_low:.3f} high={frac_high:.3f}",
                    quality_flags=["bounded_01"],
                )
            )
        return results


class MultiscaleVariationOperator:
    """MULTISCALE_VARIATION — robust noise scale at coarse vs fine resolution."""

    operator_id = "multiscale_variation"

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
            diff1 = np.diff(values)
            step = max(n // 4, 1)
            diff_step = values[step:] - values[:-step]
            fine_scale = float(np.median(np.abs(diff1)))
            coarse_scale = float(np.median(np.abs(diff_step)))
            ratio = coarse_scale / max(fine_scale, 1e-12)
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(ratio),
                    uncertainty=nan_safe(coarse_scale - fine_scale),
                    availability=round(avail, 4),
                    window_descriptor=f"fine={fine_scale:.3e} coarse={coarse_scale:.3e}",
                    quality_flags=["scale_ratio"],
                )
            )
        return results


class RobustCommandResponseGainOperator:
    """ROBUST_COMMAND_RESPONSE_GAIN — robust gain between command and response.

    Zero-fit: ratio of robust scale of response to robust scale of command,
    signed by the sign of their covariance. Acts as an efficiency/gain-loss
    proxy without fitting a regression.
    """

    operator_id = "robust_command_response_gain"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        if len(inputs) < 2:
            raise ValueError("robust_command_response_gain requires command and response signals")
        cmd = _clean_values(inputs[0])
        rsp = _clean_values(inputs[1])
        avail = min(_availability(inputs[0]), _availability(inputs[1]))
        n = min(len(cmd), len(rsp))
        if n < 4:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["insufficient_samples"],
                )
            ]
        cmd, rsp = cmd[:n], rsp[:n]
        med_cmd = float(np.median(cmd))
        med_rsp = float(np.median(rsp))
        scale_cmd = max(float(np.median(np.abs(cmd - med_cmd))), 1e-12)
        scale_rsp = max(float(np.median(np.abs(rsp - med_rsp))), 1e-12)
        # Sign from rank correlation. Guard against constant inputs: np.corrcoef
        # returns NaN when either series has zero variance, and "NaN >= 0" is
        # False, which would let the sign comparison pick an arbitrary -1.0 and
        # fabricate a confident gain on degenerate data. Mirror the
        # LaggedCoherenceOperator std-guard and fail closed instead.
        std_cmd = float(np.std(cmd))
        std_rsp = float(np.std(rsp))
        if std_cmd <= 1e-12 or std_rsp <= 1e-12:
            return [
                OperatorResult(
                    operator_id=self.operator_id,
                    value=0.0,
                    uncertainty=0.0,
                    availability=round(avail, 4),
                    window_descriptor=f"n={n}",
                    quality_flags=["constant_input"],
                )
            ]
        corr = float(np.corrcoef(cmd, rsp)[0, 1])
        sign = 1.0 if math.isfinite(corr) and corr >= 0.0 else -1.0
        gain = sign * scale_rsp / scale_cmd
        return [
            OperatorResult(
                operator_id=self.operator_id,
                value=nan_safe(gain),
                uncertainty=nan_safe(scale_rsp / scale_cmd),
                availability=round(avail, 4),
                window_descriptor=f"n={n}",
                quality_flags=["gain_proxy"],
            )
        ]


class DynamicConsistencyResidualOperator:
    """DYNAMIC_CONSISTENCY_RESIDUAL — mismatch between command, rate, response.

    For an actuator: expected response direction should track command direction,
    and rate should be consistent with effort. Emits a normalised inconsistency
    score over the window. Zero-fit.
    """

    operator_id = "dynamic_consistency_residual"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        if len(inputs) < 2:
            raise ValueError("dynamic_consistency_residual requires at least two signals")
        first = _clean_values(inputs[0])
        second = _clean_values(inputs[1])
        avail = min(_availability(inputs[0]), _availability(inputs[1]))
        n = min(len(first), len(second))
        if n < 4:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["insufficient_samples"],
                )
            ]
        first, second = first[:n], second[:n]
        # Proportion of time the two signals move in opposite directions.
        d1 = np.diff(first)
        d2 = np.diff(second)
        if len(d1) == 0:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["no_diff"],
                )
            ]
        opposing = float(np.mean(np.sign(d1) * np.sign(d2) < 0))
        results = [opposing]
        if len(inputs) >= 3:
            third = _clean_values(inputs[2])
            m = min(len(third), n - 1)
            d3 = np.diff(third[: m + 1])
            m = min(len(d1), len(d3))
            if m > 0:
                opposing_rate = float(np.mean(np.sign(d1[:m]) * np.sign(d3[:m]) < 0))
                results.append(opposing_rate)
        return [
            OperatorResult(
                operator_id=self.operator_id,
                value=nan_safe(float(np.mean(results))),
                uncertainty=nan_safe(float(np.std(results))) if len(results) > 1 else 0.0,
                availability=round(avail, 4),
                window_descriptor=f"n={n}",
                quality_flags=["consistency"],
            )
        ]


class RegimeConditionalSlopeOperator:
    """REGIME_CONDITIONAL_SLOPE — heating vs cooling slope asymmetry.

    Splits a thermal signal into rising/falling regimes and reports the median
    slope of each. A large asymmetry suggests a one-sided heat path. Zero-fit.
    """

    operator_id = "regime_conditional_slope"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            values = _clean_values(input_)
            avail = _availability(input_)
            n = len(values)
            if n < 5:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=round(avail, 4), quality_flags=["insufficient_samples"],
                    )
                )
                continue
            diffs = np.diff(values)
            rising = diffs[diffs > 1e-12]
            falling = diffs[diffs < -1e-12]
            med_rise = float(np.median(rising)) if len(rising) else 0.0
            med_fall = float(np.median(falling)) if len(falling) else 0.0
            asymmetry = (med_rise + med_fall) / max(abs(med_rise) + abs(med_fall), 1e-12)
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(asymmetry),
                    uncertainty=nan_safe(abs(med_rise) + abs(med_fall)),
                    availability=round(avail, 4),
                    window_descriptor=f"rise={med_rise:.3f} fall={med_fall:.3f}",
                    quality_flags=["heating_cooling"],
                )
            )
        return results


class PlateauAndClippingProfileOperator:
    """PLATEAU_AND_CLIPPING_PROFILE — time spent pinned at signal extremes.

    A converter that clips spends a large fraction of time at its min/max.
    Uses the empirical min/max as pinning bounds (data-only, no threshold fit).
    """

    operator_id = "plateau_and_clipping_profile"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        results: list[OperatorResult] = []
        for input_ in inputs:
            values = _clean_values(input_)
            avail = _availability(input_)
            if len(values) < 5:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                        availability=round(avail, 4), quality_flags=["insufficient_samples"],
                    )
                )
                continue
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            span = vmax - vmin
            if span < 1e-12:
                results.append(
                    OperatorResult(
                        operator_id=self.operator_id, value=1.0, uncertainty=0.0,
                        availability=round(avail, 4), quality_flags=["constant"],
                    )
                )
                continue
            # Pinned: within 2% of either extreme.
            eps = 0.02 * span
            frac_low = float(np.mean(values <= vmin + eps))
            frac_high = float(np.mean(values >= vmax - eps))
            pin_frac = frac_low + frac_high
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(pin_frac),
                    uncertainty=nan_safe(eps / span),
                    availability=round(avail, 4),
                    window_descriptor=f"low={frac_low:.3f} high={frac_high:.3f}",
                    quality_flags=["clipping"],
                )
            )
        return results


__all__ = [
    "RobustLevelDistributionOperator",
    "SignedSlopeDistributionOperator",
    "RobustLocalTrendOperator",
    "BoundaryDwellProfileOperator",
    "MultiscaleVariationOperator",
    "RobustCommandResponseGainOperator",
    "DynamicConsistencyResidualOperator",
    "RegimeConditionalSlopeOperator",
    "PlateauAndClippingProfileOperator",
]
