"""Shared numerical tools for gray-box state estimation.

These are pure linear-algebra / statistics routines, deliberately separated from
the per-archetype equation families so a third (actuator, queue, …) estimator can
reuse them without importing the whole balance/thermal logic.
"""

from __future__ import annotations

import math

import numpy as np


def linear_covariance(design: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Standard linear-regression covariance from the whitened design matrix."""
    degrees = max(int(design.shape[0] - design.shape[1]), 1)
    sigma2 = float(residuals @ residuals) / degrees
    return sigma2 * np.linalg.pinv(design.T @ design)


def first_difference_cholesky(dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the bidiagonal Cholesky factor of the differenced-noise covariance.

    For independent observation noise, ``diff(noise) / dt`` has a tridiagonal covariance.
    Keeping the actual sampling intervals here avoids assuming a fixed -0.5 correlation when the
    sampling grid changes.
    """
    steps = np.asarray(dt, dtype=float)
    diagonal_covariance = 2.0 / np.square(steps)
    off_diagonal_covariance = -1.0 / (steps[:-1] * steps[1:])
    diagonal = np.empty(len(steps), dtype=float)
    subdiagonal = np.empty(max(len(steps) - 1, 0), dtype=float)
    diagonal[0] = math.sqrt(diagonal_covariance[0])
    for index in range(len(subdiagonal)):
        subdiagonal[index] = off_diagonal_covariance[index] / diagonal[index]
        diagonal[index + 1] = math.sqrt(
            max(
                diagonal_covariance[index + 1] - subdiagonal[index] ** 2,
                1.0e-15,
            )
        )
    return diagonal, subdiagonal


def lower_bidiagonal_solve(
    values: np.ndarray,
    diagonal: np.ndarray,
    subdiagonal: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    vector_input = matrix.ndim == 1
    if vector_input:
        matrix = matrix[:, None]
    solved = np.empty_like(matrix)
    solved[0] = matrix[0] / diagonal[0]
    for index in range(1, len(diagonal)):
        solved[index] = (
            matrix[index] - subdiagonal[index - 1] * solved[index - 1]
        ) / diagonal[index]
    return solved[:, 0] if vector_input else solved


def gls_balance_fit(
    derivative: np.ndarray,
    design: np.ndarray,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generalised least-squares fit of ``derivative ~ design @ coefficients``.

    Returns (coefficients, raw_residual, whitened_innovation, whitened_design,
    cholesky_diagonal, cholesky_subdiagonal).
    """
    diagonal, subdiagonal = first_difference_cholesky(dt)
    whitened_derivative = lower_bidiagonal_solve(derivative, diagonal, subdiagonal)
    whitened_design = lower_bidiagonal_solve(design, diagonal, subdiagonal)
    coefficients = np.linalg.lstsq(
        whitened_design, whitened_derivative, rcond=None
    )[0]
    raw_residual = derivative - design @ coefficients
    whitened_innovation = whitened_derivative - whitened_design @ coefficients
    return (
        coefficients,
        raw_residual,
        whitened_innovation,
        whitened_design,
        diagonal,
        subdiagonal,
    )


def conservation_episode_score(
    whitened_innovation: np.ndarray,
    whitened_design: np.ndarray,
    cholesky_diagonal: np.ndarray,
    cholesky_subdiagonal: np.ndarray,
    change_point_fractions: list[float],
) -> tuple[float, np.ndarray]:
    """Return a GLS score for an unknown persistent balance-law change point."""
    values = np.asarray(whitened_innovation, dtype=float)
    sample_count = len(values)
    parameter_count = whitened_design.shape[1]
    scale = math.sqrt(
        float(values @ values) / max(sample_count - parameter_count, 1)
    )
    if scale <= 1.0e-15:
        return (
            0.0 if float(np.max(np.abs(values))) <= 1.0e-15 else float("inf"),
            np.empty((sample_count, 0), dtype=float),
        )
    indices = sorted(
        {
            min(max(int(round(fraction * sample_count)), 1), sample_count - 1)
            for fraction in change_point_fractions
        }
    )
    steps = (
        np.arange(sample_count)[:, None] >= np.asarray(indices)[None, :]
    ).astype(float)
    whitened_steps = lower_bidiagonal_solve(
        steps, cholesky_diagonal, cholesky_subdiagonal
    )
    design_pseudoinverse = np.linalg.pinv(whitened_design)
    directions = whitened_steps - whitened_design @ (
        design_pseudoinverse @ whitened_steps
    )
    norms = np.linalg.norm(directions, axis=0)
    valid = norms > 1.0e-15
    directions = directions[:, valid] / norms[valid]
    if directions.shape[1] == 0:
        return 0.0, directions
    score = float(np.max(np.abs(values @ directions)) / scale)
    return score, directions


__all__ = [
    "conservation_episode_score",
    "first_difference_cholesky",
    "gls_balance_fit",
    "linear_covariance",
    "lower_bidiagonal_solve",
]
