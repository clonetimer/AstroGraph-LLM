"""Balance and coupling operators — zero-fit integral balance residuals.

These encode physical conservation: the change in a state over a window should
equal the time-integrated net flow. Residuals are computed from data only, no
fitting, no learned thresholds.
"""

from __future__ import annotations

import numpy as np

from .contracts import FeatureInput, OperatorResult, nan_safe
from .robust_statistics import _clean_values, _availability


def _period(input_: FeatureInput) -> float:
    """Robust sample period from timestamps."""
    t = np.asarray(input_.timestamps_s, dtype=float)
    if len(t) < 2:
        return 1.0
    diffs = np.diff(t)
    diffs = diffs[diffs > 1e-12]
    return float(np.median(diffs)) if len(diffs) else 1.0


class IntegralBalanceResidualOperator:
    """INTEGRAL_BALANCE_RESIDUAL — scale-normalised state-vs-flow mismatch.

    Requires two inputs: the state signal and the net flow signal, aligned.

    The state and flow may be in *different units* (e.g. SOC dimensionless vs
    power in W). A raw difference would be meaningless, so each term is
    normalised by its own robust scale before subtraction:

        residual = Δstate/σ_state − (Σflow·dt)/σ_flow

    This yields a dimensionless shape-consistency score: does the flow actually
    move the state? Physical capacity identification is the job of the
    state-estimation layer, not this zero-fit screening operator.
    """

    operator_id = "integral_balance_residual"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        if len(inputs) < 2:
            raise ValueError("integral_balance_residual requires a state input and a flow input")
        state_in, flow_in = inputs[0], inputs[1]
        state = _clean_values(state_in)
        flow = _clean_values(flow_in)
        avail = min(_availability(state_in), _availability(flow_in))
        n = min(len(state), len(flow))
        if n < 3:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["insufficient_samples"],
                )
            ]

        state = state[:n]
        flow = flow[:n]
        dt = _period(flow_in)
        delta_state = float(state[-1] - state[0])
        integral_flow = float(np.sum(flow[:-1] + flow[1:]) * 0.5 * dt)
        sigma_state = max(float(np.median(np.abs(state - np.median(state)))), 1e-12)
        sigma_flow = max(float(np.median(np.abs(flow - np.median(flow)))), 1e-12)
        # Soft-normalise each term to [-1, 1] so a near-zero MAD (constant
        # signal) cannot blow the residual up. Bounded, zero-fit, unit-free.
        term_state = (delta_state / sigma_state) / (1.0 + abs(delta_state / sigma_state))
        term_flow = (integral_flow / sigma_flow) / (1.0 + abs(integral_flow / sigma_flow))
        residual = term_state - term_flow
        return [
            OperatorResult(
                operator_id=self.operator_id,
                value=nan_safe(residual),
                uncertainty=nan_safe(1.0 / max(n, 1)),
                availability=round(avail, 4),
                window_descriptor=f"dt={dt:.3f} n={n}",
                quality_flags=["conservation", "scale_normalised"],
            )
        ]


class RobustInputOutputRatioOperator:
    """ROBUST_INPUT_OUTPUT_RATIO — median(output)/median(input) with MAD spread.

    Converter efficiency / delivery ratio proxy. Zero-fit.
    """

    operator_id = "robust_input_output_ratio"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        if len(inputs) < 2:
            raise ValueError("robust_input_output_ratio requires an input and an output signal")
        in_vals = _clean_values(inputs[0])
        out_vals = _clean_values(inputs[1])
        avail = min(_availability(inputs[0]), _availability(inputs[1]))
        n = min(len(in_vals), len(out_vals))
        if n < 3:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["insufficient_samples"],
                )
            ]
        in_vals, out_vals = in_vals[:n], out_vals[:n]
        med_in = float(np.median(in_vals))
        med_out = float(np.median(out_vals))
        denom = max(abs(med_in), 1e-12)
        ratio = med_out / denom
        mad_in = float(np.median(np.abs(in_vals - med_in)))
        mad_out = float(np.median(np.abs(out_vals - med_out)))
        return [
            OperatorResult(
                operator_id=self.operator_id,
                value=nan_safe(ratio),
                uncertainty=nan_safe((mad_out + ratio * mad_in) / max(abs(med_out), 1e-12)),
                availability=round(avail, 4),
                window_descriptor=f"in={med_in:.3e} out={med_out:.3e}",
                quality_flags=["efficiency_ratio"],
            )
        ]


class LaggedCoherenceOperator:
    """LAGGED_COHERENCE — max |correlation| between command and response at lag.

    Uses a robust sign-based correlation (Kendall-tau style) over a bounded set
    of lags. Zero-fit.
    """

    operator_id = "lagged_coherence"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        if len(inputs) < 2:
            raise ValueError("lagged_coherence requires command and response signals")
        cmd = _clean_values(inputs[0])
        rsp = _clean_values(inputs[1])
        avail = min(_availability(inputs[0]), _availability(inputs[1]))
        n = min(len(cmd), len(rsp))
        if n < 8:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["insufficient_samples"],
                )
            ]
        cmd, rsp = cmd[:n], rsp[:n]

        best_rho = 0.0
        best_lag = 0
        max_lag = min(8, n // 4)
        for lag in range(max_lag + 1):
            a = cmd[:-lag] if lag else cmd
            b = rsp[lag:] if lag else rsp
            m = len(a)
            if m < 4:
                continue
            # Sign-based correlation: mean of sign((a-mean)*(b-mean)).
            ca = a - np.mean(a)
            cb = b - np.mean(b)
            rho = float(np.mean(np.sign(ca) * np.sign(cb))) if (np.std(ca) > 1e-12 and np.std(cb) > 1e-12) else 0.0
            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_lag = lag

        return [
            OperatorResult(
                operator_id=self.operator_id,
                value=nan_safe(best_rho),
                uncertainty=nan_safe(float(best_lag)),
                availability=round(avail, 4),
                window_descriptor=f"lag={best_lag}",
                quality_flags=["max_lag_corr"],
            )
        ]


class ConservationResidualOperator:
    """CONSERVATION_RESIDUAL — relative imbalance between input and output.

    (Σinput − Σoutput)/Σinput over the window. Zero-fit.
    """

    operator_id = "conservation_residual"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        if len(inputs) < 2:
            raise ValueError("conservation_residual requires an input and an output signal")
        in_vals = _clean_values(inputs[0])
        out_vals = _clean_values(inputs[1])
        avail = min(_availability(inputs[0]), _availability(inputs[1]))
        n = min(len(in_vals), len(out_vals))
        if n < 3:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["insufficient_samples"],
                )
            ]
        in_vals, out_vals = in_vals[:n], out_vals[:n]
        sum_in = float(np.sum(in_vals))
        sum_out = float(np.sum(out_vals))
        denom = max(abs(sum_in), 1e-12)
        residual = (sum_in - sum_out) / denom
        return [
            OperatorResult(
                operator_id=self.operator_id,
                value=nan_safe(residual),
                uncertainty=nan_safe(denom / max(n, 1)),
                availability=round(avail, 4),
                window_descriptor=f"Σin={sum_in:.3e} Σout={sum_out:.3e}",
                quality_flags=["conservation"],
            )
        ]


class InventoryFlowBalanceOperator:
    """INVENTORY_FLOW_BALANCE — scale-normalised fill change vs net flow.

    Requires state (fill fraction, dimensionless) and flow (fractional flow
    rate, 1/s) inputs. As with the integral-balance residual, each term is
    normalised by its own robust scale so units never mix.
    """

    operator_id = "inventory_flow_balance"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        if len(inputs) < 2:
            raise ValueError("inventory_flow_balance requires fill and flow signals")
        fill = _clean_values(inputs[0])
        flow = _clean_values(inputs[1])
        avail = min(_availability(inputs[0]), _availability(inputs[1]))
        n = min(len(fill), len(flow))
        if n < 3:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["insufficient_samples"],
                )
            ]
        fill, flow = fill[:n], flow[:n]
        dt = _period(inputs[1])
        delta_fill = float(fill[-1] - fill[0])
        integral_flow = float(np.sum(flow[:-1] + flow[1:]) * 0.5 * dt)
        sigma_fill = max(float(np.median(np.abs(fill - np.median(fill)))), 1e-12)
        sigma_flow = max(float(np.median(np.abs(flow - np.median(flow)))), 1e-12)
        # Same bounded soft-normalisation as the integral-balance residual.
        term_fill = (delta_fill / sigma_fill) / (1.0 + abs(delta_fill / sigma_fill))
        term_flow = (integral_flow / sigma_flow) / (1.0 + abs(integral_flow / sigma_flow))
        residual = term_fill - term_flow
        return [
            OperatorResult(
                operator_id=self.operator_id,
                value=nan_safe(residual),
                uncertainty=nan_safe(1.0 / max(n, 1)),
                availability=round(avail, 4),
                window_descriptor=f"dt={dt:.3f} n={n}",
                quality_flags=["inventory_balance", "scale_normalised"],
            )
        ]


class ReferenceInnovationResidualOperator:
    """REFERENCE_INNOVATION_RESIDUAL — deviation of a sensor from a reference.

    For a measurement sensor, the innovation is the difference between the raw
    observation and a robust local reference (median of a trailing window).
    Reports the median absolute innovation scaled by MAD. Zero-fit.
    """

    operator_id = "reference_innovation_residual"

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
            # Robust global location.
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med)))
            scale = max(mad, 1e-12)
            # Median absolute innovation from the global median, normalised.
            innovation = float(np.median(np.abs(values - med))) / scale
            # Drift proxy: late-vs-early location shift.
            half = n // 2
            late_shift = (float(np.median(values[half:])) - float(np.median(values[:half]))) / scale
            results.append(
                OperatorResult(
                    operator_id=self.operator_id,
                    value=nan_safe(innovation),
                    uncertainty=nan_safe(late_shift),
                    availability=round(avail, 4),
                    window_descriptor=f"n={n} med={med:.3f}",
                    quality_flags=["innovation", "drift_proxy"],
                )
            )
        return results


class LumpedBalanceResidualOperator:
    """LUMPED_BALANCE_RESIDUAL — thermal balance residual.

    For a lumped thermal body: dT/dt should track (heat − conductance·(T−T_amb)).
    Emits a normalised mismatch using robust finite differences. Zero-fit.
    """

    operator_id = "lumped_balance_residual"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        # Inputs: [temperature, heat, ambient_temperature] (order per contract).
        if len(inputs) < 3:
            raise ValueError("lumped_balance_residual requires temperature, heat and ambient inputs")
        temp = _clean_values(inputs[0])
        heat = _clean_values(inputs[1])
        amb = _clean_values(inputs[2])
        avail = min(_availability(i) for i in inputs)
        n = min(len(temp), len(heat), len(amb))
        if n < 6:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["insufficient_samples"],
                )
            ]
        temp, heat, amb = temp[:n], heat[:n], amb[:n]
        dt = _period(inputs[0])
        dT = np.diff(temp)
        # Expected: C·dT ≈ (heat + G·(T_amb − T))·dt. Use median ratios as a
        # residual proxy without fitting C and G explicitly.
        tmid = (temp[:-1] + temp[1:]) * 0.5
        amid = (amb[:-1] + amb[1:]) * 0.5
        hmid = (heat[:-1] + heat[1:]) * 0.5
        # Robust scale of dT and of the driving term.
        scale_dt = max(float(np.median(np.abs(dT))), 1e-12)
        drive = hmid + amid - tmid
        scale_drive = max(float(np.median(np.abs(drive))), 1e-12)
        # Normalised correlation of dT with drive → balance consistency.
        if scale_drive < 1e-12 or scale_dt < 1e-12:
            balance = 0.0
        else:
            corr = float(np.mean(np.sign(dT) * np.sign(drive)))
            balance = corr  # +1 = balanced, −1 = inverted.
        return [
            OperatorResult(
                operator_id=self.operator_id,
                value=nan_safe(1.0 - balance),
                uncertainty=nan_safe(scale_drive / scale_dt),
                availability=round(avail, 4),
                window_descriptor=f"dt={dt:.3f} n={n}",
                quality_flags=["thermal_balance"],
            )
        ]


class FlowConservationResidualOperator:
    """FLOW_CONSERVATION_RESIDUAL — input/output flow imbalance for a link.

    For a transport link, delivered should not exceed offered. Reports the
    ratio of (offered − delivered) to offered, with a floor. Zero-fit.
    """

    operator_id = "flow_conservation_residual"

    def compute(self, inputs: list[FeatureInput]) -> list[OperatorResult]:
        if len(inputs) < 2:
            raise ValueError("flow_conservation_residual requires input and output flow signals")
        in_vals = _clean_values(inputs[0])
        out_vals = _clean_values(inputs[1])
        avail = min(_availability(inputs[0]), _availability(inputs[1]))
        n = min(len(in_vals), len(out_vals))
        if n < 3:
            return [
                OperatorResult(
                    operator_id=self.operator_id, value=0.0, uncertainty=0.0,
                    availability=round(avail, 4), quality_flags=["insufficient_samples"],
                )
            ]
        in_vals, out_vals = in_vals[:n], out_vals[:n]
        sum_in = float(np.sum(in_vals))
        sum_out = float(np.sum(out_vals))
        denom = max(abs(sum_in), 1e-12)
        residual = (sum_in - sum_out) / denom
        return [
            OperatorResult(
                operator_id=self.operator_id,
                value=nan_safe(residual),
                uncertainty=nan_safe(abs(residual) / max(n, 1)),
                availability=round(avail, 4),
                window_descriptor=f"Σin={sum_in:.3e} Σout={sum_out:.3e}",
                quality_flags=["flow_conservation"],
            )
        ]


__all__ = [
    "IntegralBalanceResidualOperator",
    "RobustInputOutputRatioOperator",
    "LaggedCoherenceOperator",
    "ConservationResidualOperator",
    "InventoryFlowBalanceOperator",
    "ReferenceInnovationResidualOperator",
    "LumpedBalanceResidualOperator",
    "FlowConservationResidualOperator",
]
