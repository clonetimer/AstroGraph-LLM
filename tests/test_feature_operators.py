"""Tests for the generic zero-fit feature-operator runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrograph.features.generic import (
    IntegralBalanceResidualOperator,
    LaggedCoherenceOperator,
    MissingRunLengthProfileOperator,
    MultiscaleChangeScoreOperator,
    RobustCommandResponseGainOperator,
    RobustInputOutputRatioOperator,
    RobustLevelDistributionOperator,
    SignedSlopeDistributionOperator,
    TokenBuilder,
)
from astrograph.features.generic.contracts import EvidenceToken, FeatureInput
from astrograph.features.generic.registry import FeatureOperatorRegistry, default_registry

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FEATURE_CONTRACT_PATH = CONFIGS / "semantics" / "sat_agent_archetype_feature_contracts.s0.v1.json"


def _signal(
    values,
    *,
    port_role_id="state.energy_fraction",
    quantity_kind_id="energy.state_of_charge",
    semantic_channel_id="asset.test.channel",
    missing=None,
    dt=1.0,
) -> FeatureInput:
    n = len(values)
    missing = missing if missing is not None else [False] * n
    timestamps = [i * dt for i in range(n)]
    return FeatureInput(
        semantic_channel_id=semantic_channel_id,
        quantity_kind_id=quantity_kind_id,
        port_role_id=port_role_id,
        timestamps_s=timestamps,
        values=list(values),
        missing_mask=missing,
    )


class TestRobustLevelDistribution:
    def test_constant_signal_zero_mad(self):
        s = _signal([5.0] * 10)
        results = RobustLevelDistributionOperator().compute([s])
        assert results[0].value == pytest.approx(5.0)
        assert results[0].uncertainty == pytest.approx(0.0)

    def test_missing_values_lower_availability(self):
        s = _signal([1.0, 2.0, 3.0, 4.0], missing=[False, False, True, True])
        results = RobustLevelDistributionOperator().compute([s])
        assert results[0].availability == pytest.approx(0.5)

    def test_median_not_mean(self):
        s = _signal([0.0, 0.0, 0.0, 100.0])
        results = RobustLevelDistributionOperator().compute([s])
        assert results[0].value == pytest.approx(0.0)  # median ignores outlier


class TestSignedSlope:
    def test_positive_slope(self):
        s = _signal([i * 2.0 for i in range(20)])
        results = SignedSlopeDistributionOperator().compute([s])
        assert results[0].value == pytest.approx(2.0, rel=0.2)

    def test_negative_slope(self):
        s = _signal([-i * 3.0 for i in range(20)])
        results = SignedSlopeDistributionOperator().compute([s])
        assert results[0].value == pytest.approx(-3.0, rel=0.2)

    def test_insufficient_samples(self):
        s = _signal([1.0, 2.0])
        results = SignedSlopeDistributionOperator().compute([s])
        assert "insufficient_samples" in results[0].quality_flags


class TestMissingRunLength:
    def test_longest_run(self):
        s = _signal(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            missing=[False, True, True, True, False, False],
        )
        results = MissingRunLengthProfileOperator().compute([s])
        assert results[0].value == pytest.approx(3.0)

    def test_no_missing(self):
        s = _signal([1.0] * 6)
        results = MissingRunLengthProfileOperator().compute([s])
        assert results[0].value == pytest.approx(0.0)


class TestMultiscaleChange:
    def test_step_change_detected(self):
        s = _signal([0.0] * 20 + [10.0] * 20)
        results = MultiscaleChangeScoreOperator().compute([s])
        assert results[0].value > 1.0  # large normalised shift

    def test_flat_signal_small_change(self):
        s = _signal([5.0] * 40)
        results = MultiscaleChangeScoreOperator().compute([s])
        assert results[0].value == pytest.approx(0.0, abs=1.0)


class TestIntegralBalance:
    def test_conservation_holds(self):
        # State increases linearly, flow is constant positive.
        flow = _signal([2.0] * 10, port_role_id="power.exchange", quantity_kind_id="electrical.active_power")
        state = _signal([i * 2.0 for i in range(10)], port_role_id="state.energy_fraction")
        results = IntegralBalanceResidualOperator().compute([state, flow])
        # Δstate = 18, integral flow = 2*9 = 18 → residual ~0.
        assert abs(results[0].value) < 0.5

    def test_conservation_broken(self):
        flow = _signal([2.0] * 10, port_role_id="power.exchange", quantity_kind_id="electrical.active_power")
        state = _signal([0.0] * 10, port_role_id="state.energy_fraction")  # flat, no change
        results = IntegralBalanceResidualOperator().compute([state, flow])
        assert abs(results[0].value) > 0.5  # flow in but no state change


class TestInputOutputRatio:
    def test_efficiency_ratio(self):
        inp = _signal([100.0] * 10, port_role_id="power.input", quantity_kind_id="electrical.active_power")
        out = _signal([80.0] * 10, port_role_id="power.output", quantity_kind_id="electrical.active_power")
        results = RobustInputOutputRatioOperator().compute([inp, out])
        assert results[0].value == pytest.approx(0.8, rel=0.1)


class TestLaggedCoherence:
    def test_aligned_signals_high_correlation(self):
        cmd = _signal([i % 2 for i in range(40)], port_role_id="command.effort", quantity_kind_id="mechanical.torque")
        rsp = _signal([i % 2 for i in range(40)], port_role_id="response.effort", quantity_kind_id="mechanical.torque")
        results = LaggedCoherenceOperator().compute([cmd, rsp])
        assert abs(results[0].value) > 0.5


class TestCommandResponseGain:
    def test_constant_response_is_zero_not_fake_negative(self):
        """Regression: a constant response makes np.corrcoef return NaN, and
        ``NaN >= 0`` is False, which previously produced an arbitrary -1.0
        confidence. The operator must fail closed with 0.0 + constant_input."""
        cmd = _signal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], port_role_id="command.effort", quantity_kind_id="mechanical.torque")
        rsp = _signal([0.5] * 6, port_role_id="response.effort", quantity_kind_id="mechanical.torque")
        results = RobustCommandResponseGainOperator().compute([cmd, rsp])
        assert results[0].value == 0.0
        assert "constant_input" in results[0].quality_flags

    def test_constant_command_is_zero_not_fake_negative(self):
        cmd = _signal([1.0] * 6, port_role_id="command.effort", quantity_kind_id="mechanical.torque")
        rsp = _signal([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], port_role_id="response.effort", quantity_kind_id="mechanical.torque")
        results = RobustCommandResponseGainOperator().compute([cmd, rsp])
        assert results[0].value == 0.0
        assert "constant_input" in results[0].quality_flags

    def test_varying_signals_still_produce_gain(self):
        cmd = _signal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], port_role_id="command.effort", quantity_kind_id="mechanical.torque")
        rsp = _signal([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], port_role_id="response.effort", quantity_kind_id="mechanical.torque")
        results = RobustCommandResponseGainOperator().compute([cmd, rsp])
        assert results[0].value == pytest.approx(2.0, rel=0.2)
        assert "constant_input" not in results[0].quality_flags


class TestRegistry:
    def test_all_contract_families_registered(self):
        reg = default_registry()
        catalog = reg.load_feature_contract_catalog(FEATURE_CONTRACT_PATH)
        # Every archetype's primitives resolve to a registered operator.
        for archetype, primitives in catalog.items():
            assert len(primitives) >= 3, f"{archetype} has too few primitives"
        assert len(catalog) == 9  # all nine source-independent archetypes

    def test_unknown_family_raises(self):
        reg = FeatureOperatorRegistry()
        with pytest.raises(KeyError):
            reg.get("NOT_A_FAMILY")


class TestTokenBuilder:
    def test_tokens_match_contract(self):
        builder = TokenBuilder(archetype="storage.energy_buffer")
        builder.load_contract(FEATURE_CONTRACT_PATH)
        frame = _make_frame()
        result = builder.build(frame, component_archetype_id="storage.energy_buffer")
        assert result.token_count > 0
        for token in result.tokens:
            assert isinstance(token, EvidenceToken)
            assert token.labels_used is False
            assert token.truth_used is False
            assert token.identity_fields_allowed is False
            assert token.component_archetype_id == "storage.energy_buffer"


def _make_frame():
    from astrograph.semantics.contracts import (
        SamplingProfile,
        SemanticSignal,
        SemanticTelemetryFrame,
    )

    return SemanticTelemetryFrame(
        frame_id="test",
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=[0.0, 1.0, 2.0, 3.0, 4.0],
        signals=[
            SemanticSignal(
                semantic_channel_id="asset.eps.storage_0.state_of_charge",
                asset_instance_id="asset-under-test",
                component_archetype_id="storage.energy_buffer",
                port_role_id="state.energy_fraction",
                quantity_kind_id="energy.state_of_charge",
                canonical_unit="1",
                direction=1,
                values=[0.8, 0.79, 0.78, 0.77, 0.76],
                missing_mask=[False] * 5,
            ),
            SemanticSignal(
                semantic_channel_id="asset.eps.storage_0.exchange_power",
                asset_instance_id="asset-under-test",
                component_archetype_id="storage.energy_buffer",
                port_role_id="power.exchange",
                quantity_kind_id="electrical.active_power",
                canonical_unit="W",
                direction=1,
                values=[-10.0, -10.0, -10.0, -10.0, -10.0],
                missing_mask=[False] * 5,
            ),
        ],
        sampling=SamplingProfile(row_count=5, nominal_period_s=1.0, irregular=False, missing_value_count=0),
        semantic_data_sha256="a" * 64,
        labels_used=False,
        truth_used=False,
        claim_boundary="test only",
    )
