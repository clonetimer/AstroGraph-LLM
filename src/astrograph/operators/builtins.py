from __future__ import annotations

import numpy as np

from astrograph.operators.models import OperatorInput, OperatorSpec
from astrograph.operators.registry import DeterministicOperatorRegistry, OperatorDefinition


BUILTIN_OPERATOR_VERSION = "0.5.13"


def _none_if_empty(values: np.ndarray, fn) -> float | None:
    return None if values.size == 0 else float(fn(values))


def _max_abs_residual(inputs: OperatorInput) -> float | None:
    return _none_if_empty(inputs.residual(), lambda values: np.max(np.abs(values)))


def _mean_abs_residual(inputs: OperatorInput) -> float | None:
    return _none_if_empty(inputs.residual(), lambda values: np.mean(np.abs(values)))


def _mean_residual(inputs: OperatorInput) -> float | None:
    return _none_if_empty(inputs.residual(), np.mean)


def _last_residual(inputs: OperatorInput) -> float | None:
    residual = inputs.current - inputs.baseline if inputs.baseline is not None else np.array([])
    finite = residual[np.isfinite(residual)]
    return None if finite.size == 0 else float(finite[-1])


def _availability_drop_fraction(inputs: OperatorInput) -> float | None:
    if inputs.baseline is None:
        raise ValueError("availability drop requires baseline")
    valid = np.isfinite(inputs.current) & np.isfinite(inputs.baseline)
    threshold = float(inputs.parameters.get("drop_threshold", 0.5))
    return None if not valid.any() else float(
        np.mean((inputs.baseline[valid] - inputs.current[valid]) > threshold)
    )


def _linear_slope(inputs: OperatorInput) -> float | None:
    if inputs.time_s is None:
        raise ValueError("linear slope requires time_s")
    values = inputs.current
    valid = np.isfinite(values) & np.isfinite(inputs.time_s)
    if int(valid.sum()) < 2:
        return None
    x = inputs.time_s[valid]
    y = values[valid]
    if np.allclose(x, x[0]):
        return None
    return float(np.polyfit(x, y, 1)[0])


def _max_value(inputs: OperatorInput) -> float | None:
    return _none_if_empty(inputs.finite_current(), np.max)


def _min_value(inputs: OperatorInput) -> float | None:
    return _none_if_empty(inputs.finite_current(), np.min)


def _mean_value(inputs: OperatorInput) -> float | None:
    return _none_if_empty(inputs.finite_current(), np.mean)


def _rms_value(inputs: OperatorInput) -> float | None:
    return _none_if_empty(inputs.finite_current(), lambda values: np.sqrt(np.mean(np.square(values))))


def _std_value(inputs: OperatorInput) -> float | None:
    ddof = int(inputs.parameters.get("ddof", 0))
    values = inputs.finite_current()
    if values.size <= ddof:
        return None
    return float(np.std(values, ddof=ddof))


def _range_value(inputs: OperatorInput) -> float | None:
    values = inputs.finite_current()
    return None if values.size == 0 else float(np.max(values) - np.min(values))


def _half_window_mean_difference(inputs: OperatorInput) -> float | None:
    values = inputs.finite_current()
    if values.size < 2:
        return None
    split = values.size // 2
    if split == 0 or split == values.size:
        return None
    return float(np.mean(values[split:]) - np.mean(values[:split]))


def _mean_paired_ratio(inputs: OperatorInput) -> float | None:
    if inputs.baseline is None:
        raise ValueError("paired ratio requires baseline")
    epsilon = float(inputs.parameters.get("epsilon", 1e-12))
    valid = (
        np.isfinite(inputs.current)
        & np.isfinite(inputs.baseline)
        & (np.abs(inputs.baseline) > epsilon)
    )
    return None if not valid.any() else float(np.mean(inputs.current[valid] / inputs.baseline[valid]))


def _weighted_mean(inputs: OperatorInput) -> float | None:
    values = inputs.finite_current()
    if values.size == 0:
        return None
    raw_weights = inputs.parameters.get("weights")
    if raw_weights is None:
        return float(np.mean(values))
    weights = np.asarray(raw_weights, dtype=float).reshape(-1)
    if weights.shape != values.shape:
        raise ValueError("WEIGHTED_MEAN weights must match the finite current input length")
    if not np.isfinite(weights).all() or np.allclose(weights.sum(), 0.0):
        raise ValueError("WEIGHTED_MEAN weights must be finite and have non-zero sum")
    return float(np.average(values, weights=weights))


def _compare(values: np.ndarray, operator: str, threshold: float) -> np.ndarray:
    allowed = {
        "GREATER_THAN": np.greater,
        "GREATER_OR_EQUAL": np.greater_equal,
        "LESS_THAN": np.less,
        "LESS_OR_EQUAL": np.less_equal,
        "EQUAL": np.equal,
        "NOT_EQUAL": np.not_equal,
    }
    try:
        fn = allowed[operator]
    except KeyError as exc:
        raise ValueError(f"unsupported comparison operator: {operator}") from exc
    return fn(values, threshold)


def _joint_threshold_fraction(inputs: OperatorInput) -> float | None:
    """Fraction where two aligned channels satisfy independent conditions."""

    if inputs.baseline is None:
        raise ValueError("joint threshold fraction requires a second aligned input")
    valid = np.isfinite(inputs.current) & np.isfinite(inputs.baseline)
    if not valid.any():
        return None
    current_operator = str(inputs.parameters.get("current_comparison", "GREATER_THAN"))
    current_threshold = float(inputs.parameters.get("current_threshold", 0.5))
    baseline_operator = str(inputs.parameters.get("baseline_comparison", "LESS_OR_EQUAL"))
    baseline_threshold = float(inputs.parameters.get("baseline_threshold", 0.5))
    mask = _compare(inputs.current[valid], current_operator, current_threshold) & _compare(
        inputs.baseline[valid], baseline_operator, baseline_threshold
    )
    return float(np.mean(mask))


def _true_fraction(inputs: OperatorInput) -> float | None:
    values = inputs.finite_current()
    threshold = float(inputs.parameters.get("truth_threshold", 0.5))
    return None if values.size == 0 else float(np.mean(values > threshold))


def _false_fraction(inputs: OperatorInput) -> float | None:
    result = _true_fraction(inputs)
    return None if result is None else float(1.0 - result)


def build_builtin_operator_registry() -> DeterministicOperatorRegistry:
    registry = DeterministicOperatorRegistry()

    def register(
        operator_id: str,
        handler,
        *,
        baseline: bool = False,
        time: bool = False,
        description: str,
    ) -> None:
        registry.register(
            OperatorDefinition(
                spec=OperatorSpec(
                    operator_id=operator_id,
                    operator_version=BUILTIN_OPERATOR_VERSION,
                    requires_baseline=baseline,
                    requires_time=time,
                    description=description,
                ),
                handler=handler,
            )
        )

    register(
        "MAX_ABS_PAIRED_RESIDUAL",
        _max_abs_residual,
        baseline=True,
        description="Maximum absolute current-minus-baseline residual.",
    )
    register(
        "MEAN_ABS_PAIRED_RESIDUAL",
        _mean_abs_residual,
        baseline=True,
        description="Mean absolute current-minus-baseline residual.",
    )
    register(
        "MEAN_PAIRED_RESIDUAL",
        _mean_residual,
        baseline=True,
        description="Mean signed current-minus-baseline residual.",
    )
    register(
        "LAST_PAIRED_RESIDUAL",
        _last_residual,
        baseline=True,
        description="Last finite current-minus-baseline residual.",
    )
    register(
        "MAX_ABS_STATUS_RESIDUAL",
        _max_abs_residual,
        baseline=True,
        description="Maximum absolute residual for status-code channels.",
    )
    register(
        "AVAILABILITY_DROP_FRACTION",
        _availability_drop_fraction,
        baseline=True,
        description="Fraction of samples where baseline availability exceeds current availability.",
    )
    register(
        "LINEAR_SLOPE",
        _linear_slope,
        time=True,
        description="Least-squares slope of the current signal over time.",
    )
    register("MAX_VALUE", _max_value, description="Maximum finite value.")
    register("MIN_VALUE", _min_value, description="Minimum finite value.")
    register("MEAN_VALUE", _mean_value, description="Mean finite value.")
    register("RMS_VALUE", _rms_value, description="Root mean square of finite values.")
    register("STD_VALUE", _std_value, description="Standard deviation of finite values.")
    register("RANGE_VALUE", _range_value, description="Maximum minus minimum finite value.")
    register(
        "HALF_WINDOW_MEAN_DIFFERENCE",
        _half_window_mean_difference,
        description="Second-half mean minus first-half mean.",
    )
    register(
        "MEAN_PAIRED_RATIO",
        _mean_paired_ratio,
        baseline=True,
        description="Mean current-to-baseline ratio over aligned finite samples.",
    )
    register(
        "WEIGHTED_MEAN",
        _weighted_mean,
        description="Weighted mean over scalar or vector inputs.",
    )
    register(
        "JOINT_THRESHOLD_FRACTION",
        _joint_threshold_fraction,
        baseline=True,
        description="Fraction of aligned samples satisfying two configured threshold conditions.",
    )
    register("TRUE_FRACTION", _true_fraction, description="Fraction of values above a truth threshold.")
    register("FALSE_FRACTION", _false_fraction, description="Fraction of values not above a truth threshold.")
    return registry


BUILTIN_OPERATOR_REGISTRY = build_builtin_operator_registry()
