from __future__ import annotations

from copy import deepcopy

import pytest

from astrograph.integrations.dor_active_discovery import (
    R5ActiveDiscoveryPayload,
    require_dor_r5_active_discovery_runtime,
)
from astrograph.integrations.dor_runtime import DORRuntimeIdentity, DORStackComponentIdentity


def _payload() -> dict:
    return {
        "structure_posterior": {"friction": 0.6, "authority-loss": 0.3, "jam": 0.1},
        "truth_quarantine": {
            "truth_channels_allowed": False,
            "held_out_truth_loaded": False,
            "true_structure_loaded": False,
        },
        "truth_channels_allowed": False,
        "next_experiment": {
            "experiment_id": "torque-step",
            "execution_mode": "SIMULATION",
            "repeats": 3,
            "cost": 1.0,
            "safety_eligible": True,
        },
        "identifiability": {
            "status": "ALTERNATIVES",
            "remaining_discriminating_experiments": ["torque-step", "coast-down"],
            "certification_allowed": False,
            "truth_channels_allowed": False,
        },
        "hypothesis_space_coverage_complete": False,
    }


def test_strict_r5_contract_accepts_offline_advisory_payload() -> None:
    payload = R5ActiveDiscoveryPayload.model_validate(_payload())
    assert payload.next_experiment is not None
    assert payload.next_experiment.execution_mode == "SIMULATION"
    assert payload.truth_channels_allowed is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("truth_channels_allowed",), "false"),
        (("truth_quarantine", "held_out_truth_loaded"), "false"),
        (("next_experiment", "safety_eligible"), "false"),
    ],
)
def test_string_booleans_fail_closed(path: tuple[str, ...], value: str) -> None:
    payload = deepcopy(_payload())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        R5ActiveDiscoveryPayload.model_validate(payload)


def test_live_experiment_is_forbidden_at_astrograph_boundary() -> None:
    payload = _payload()
    payload["next_experiment"]["execution_mode"] = "LIVE"
    with pytest.raises(ValueError):
        R5ActiveDiscoveryPayload.model_validate(payload)


def test_duplicate_remaining_experiment_ids_fail_closed() -> None:
    payload = _payload()
    payload["identifiability"]["remaining_discriminating_experiments"] = ["probe", "probe"]
    with pytest.raises(ValueError, match="unique"):
        R5ActiveDiscoveryPayload.model_validate(payload)


@pytest.mark.parametrize(
    "posterior",
    [
        {"friction": True},
        {"friction": 0.8, "jam": 0.3},
        {"friction": float("nan")},
        {},
    ],
)
def test_invalid_posterior_fails_closed(posterior: dict) -> None:
    payload = _payload()
    payload["structure_posterior"] = posterior
    with pytest.raises(ValueError):
        R5ActiveDiscoveryPayload.model_validate(payload)


def test_unique_claim_requires_stability_and_complete_coverage() -> None:
    payload = _payload()
    payload["identifiability"]["status"] = "UNIQUE_SUPPORTED"
    with pytest.raises(ValueError, match="STABLE"):
        R5ActiveDiscoveryPayload.model_validate(payload)
    payload["evidence_stability"] = {"status": "STABLE", "truth_channels_allowed": False}
    with pytest.raises(ValueError, match="complete"):
        R5ActiveDiscoveryPayload.model_validate(payload)
    payload["hypothesis_space_coverage_complete"] = True
    assert R5ActiveDiscoveryPayload.model_validate(payload).identifiability is not None


def _runtime(*, dor: str = "1.0.3+timer.1", isgn: str = "2.4.0+timer.1") -> DORRuntimeIdentity:
    versions = {
        "DOR": ("dor-diagnostic-stack", "dor", dor),
        "ISGN": ("isgn-toolkit", "isgn", isgn),
        "M2N": ("m2n-toolkit", "m2n", "2.16.1+timer.1"),
        "FEC": ("fec-toolkit", "fec", "2.4.2+timer.1"),
    }
    components = {
        name: DORStackComponentIdentity(
            distribution=distribution,
            import_name=import_name,
            stack_version=version,
            metadata_version=version,
            module_version=version,
            package_source=f"/fixture/{import_name}",
            package_source_sha256="a" * 64,
        )
        for name, (distribution, import_name, version) in versions.items()
    }
    return DORRuntimeIdentity(
        executable="/fixture/dor",
        executable_sha256="b" * 64,
        dor_version=dor,
        stack_info={},
        components=components,
        stack_fingerprint_sha256="c" * 64,
    )


def test_r5_runtime_gate_accepts_target_and_rejects_older_isgn() -> None:
    require_dor_r5_active_discovery_runtime(_runtime())
    with pytest.raises(ValueError, match="below Active Discovery minimum"):
        require_dor_r5_active_discovery_runtime(_runtime(isgn="2.3.0+timer.1"))
