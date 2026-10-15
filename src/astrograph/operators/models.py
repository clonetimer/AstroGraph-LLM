from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class OperatorSpec(BaseModel):
    """Versioned metadata for one deterministic feature operator."""

    model_config = ConfigDict(extra="forbid")

    operator_id: str = Field(min_length=1)
    operator_version: str = Field(min_length=1)
    requires_baseline: bool = False
    requires_time: bool = False
    deterministic: bool = True
    description: str = ""
    output_semantics: str = "scalar"


class OperatorInput:
    """Numerical input passed to a deterministic operator.

    Arrays are converted to float numpy arrays once at the boundary.  The
    operator layer never reads files, resolves telemetry names, or performs
    unit conversion; those responsibilities remain in the object plugin.
    """

    def __init__(
        self,
        *,
        current: Any,
        baseline: Any | None = None,
        time_s: Any | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.current = np.asarray(current, dtype=float)
        self.baseline = None if baseline is None else np.asarray(baseline, dtype=float)
        self.time_s = None if time_s is None else np.asarray(time_s, dtype=float)
        self.parameters = dict(parameters or {})

        if self.current.ndim != 1:
            raise ValueError("operator current input must be one-dimensional")
        if self.baseline is not None and self.baseline.shape != self.current.shape:
            raise ValueError("operator baseline shape must match current input")
        if self.time_s is not None and self.time_s.shape != self.current.shape:
            raise ValueError("operator time_s shape must match current input")

    def finite_current(self) -> np.ndarray:
        return self.current[np.isfinite(self.current)]

    def residual(self) -> np.ndarray:
        if self.baseline is None:
            raise ValueError("operator requires baseline input")
        residual = self.current - self.baseline
        return residual[np.isfinite(residual)]
