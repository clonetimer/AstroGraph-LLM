"""Train-only channel preprocessing for causal diagnosis windows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

PreprocessingMethod = Literal["standard", "robust_clip"]


@dataclass(frozen=True)
class ChannelPreprocessor:
    """Frozen per-channel preprocessing state learned from Train only."""

    method: PreprocessingMethod
    impute: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    normalized_clip: float | None

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.impute):
            raise ValueError("input channel count does not match preprocessing state")
        filled = np.where(np.isfinite(values), values, self.impute)
        clipped = np.clip(filled, self.lower, self.upper)
        result = (clipped - self.center) / self.scale
        if self.normalized_clip is not None:
            result = np.clip(result, -self.normalized_clip, self.normalized_clip)
        if not np.isfinite(result).all():
            raise ValueError("preprocessing produced a non-finite value")
        return result.astype(np.float32)

    def receipt(self) -> dict[str, object]:
        clipped_channels = int(np.sum(np.isfinite(self.lower) | np.isfinite(self.upper)))
        return {
            "schema_version": "astrograph-channel-preprocessor.v1",
            "method": self.method,
            "fit_scope": "train_only",
            "channel_count": len(self.impute),
            "clipped_channel_count": clipped_channels,
            "normalized_clip": self.normalized_clip,
            "finite_state": bool(
                np.isfinite(self.impute).all()
                and np.isfinite(self.center).all()
                and np.isfinite(self.scale).all()
            ),
        }


def fit_channel_preprocessor(
    train_values: np.ndarray,
    *,
    method: PreprocessingMethod = "standard",
    fallback_impute: np.ndarray | None = None,
    clip_quantiles: tuple[float, float] = (0.005, 0.995),
) -> ChannelPreprocessor:
    """Fit a finite, per-channel state from a Train tensor.

    ``robust_clip`` uses Train quantiles followed by median/MAD scaling.  A
    zero-MAD channel falls back to its interquartile scale and then 1.0; it
    never learns scale from Validation or Test outliers.
    """
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('FIT')

    if train_values.ndim < 2 or train_values.shape[-1] == 0:
        raise ValueError("train_values must have at least two dimensions and one channel")
    if method not in {"standard", "robust_clip"}:
        raise ValueError(f"unsupported preprocessing method: {method}")
    lower_q, upper_q = clip_quantiles
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("clip quantiles must satisfy 0 <= lower < upper <= 1")

    flat = train_values.reshape(-1, train_values.shape[-1]).astype(np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        impute = np.nanmedian(np.where(np.isfinite(flat), flat, np.nan), axis=0)
    if fallback_impute is not None:
        fallback = np.asarray(fallback_impute, dtype=np.float64)
        if fallback.shape != impute.shape:
            raise ValueError("fallback_impute shape does not match channel count")
        impute[~np.isfinite(impute)] = fallback[~np.isfinite(impute)]
    impute[~np.isfinite(impute)] = 0.0
    filled = np.where(np.isfinite(flat), flat, impute)

    if method == "standard":
        lower = np.full(filled.shape[1], -np.inf, dtype=np.float64)
        upper = np.full(filled.shape[1], np.inf, dtype=np.float64)
        center = filled.mean(axis=0)
        scale = filled.std(axis=0)
    else:
        lower = np.quantile(filled, lower_q, axis=0)
        upper = np.quantile(filled, upper_q, axis=0)
        clipped = np.clip(filled, lower, upper)
        center = np.median(clipped, axis=0)
        mad_scale = 1.4826 * np.median(np.abs(clipped - center), axis=0)
        q25, q75 = np.quantile(clipped, (0.25, 0.75), axis=0)
        iqr_scale = (q75 - q25) / 1.349
        scale = np.where(mad_scale >= 1e-9, mad_scale, iqr_scale)

    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    return ChannelPreprocessor(
        method=method,
        impute=impute.astype(np.float32),
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        lower=lower.astype(np.float32),
        upper=upper.astype(np.float32),
        normalized_clip=10.0 if method == "robust_clip" else None,
    )
