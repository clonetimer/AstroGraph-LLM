from __future__ import annotations

from pathlib import Path

from astrograph.policy_profiles import FusionPolicyProfile, RuntimePolicyRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_production_strict_profiles_are_registered_but_not_dataset_defaults():
    registry = RuntimePolicyRegistry(ROOT / "configs/runtime_policies/registry.json")
    adcs = registry.resolve(profile_type="FUSION", profile_id="birrf.production-strict.adcs-v5_2_2.v1")
    phm = registry.resolve(profile_type="FUSION", profile_id="birrf.production-strict.phm-v6_0.v1")
    for resolved in (adcs, phm):
        assert isinstance(resolved.profile, FusionPolicyProfile)
        assert resolved.profile.qualification_status == "PRODUCTION_STRICT"
        cfg = resolved.profile.config
        assert cfg.require_complete_evidence_lineage is True
        assert cfg.enforce_kg_fault_universe_coverage is True
        assert cfg.minimum_kg_fault_universe_coverage == 1.0
        assert cfg.kg_partial_coverage_action.value == "SUPPRESS_BRANCH"
        assert cfg.expected_fault_universe
        assert resolved.profile.calibration_receipt is None
        assert "not a performance or flight-qualification claim" in resolved.profile.claim_boundary

    # Existing dataset profiles remain explicitly on the development profile until a reviewed migration.
    import json
    for rel in ("configs/dataset_profiles/adcs_v5_2_2.json", "configs/dataset_profiles/phm_v6_0.json"):
        payload = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert payload["fusion_policy_profile_id"] == "birrf.engineering-default.v1"


def test_adcs_strict_profile_freezes_full_runtime_fault_universe():
    import json
    registry = RuntimePolicyRegistry(ROOT / "configs/runtime_policies/registry.json")
    resolved = registry.resolve(profile_type="FUSION", profile_id="birrf.production-strict.adcs-v5_2_2.v1")
    universe = json.loads((ROOT / "configs/datasets/adcs_v5_2_2/candidate_universe.json").read_text(encoding="utf-8"))
    assert set(resolved.profile.config.expected_fault_universe) == set(universe["runtime_fault_ids"])
