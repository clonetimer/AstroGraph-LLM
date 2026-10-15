from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "datasets" / "phm_v6_0" / "model_policy.json"
BUNDLE = ROOT / "models" / "phm_v6_0" / "promoted_extra_trees_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phm_v60_model_policy_promotes_extra_trees_with_explicit_rollbacks() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert policy["default_replay_strategy"] == "promoted_extra_trees"
    assert policy["rollback_strategy"] == "adaptive"
    assert set(policy["allowed_replay_strategies"]) == {
        "promoted_extra_trees", "adaptive", "legacy_lr"
    }
    assert policy["unknown_onset_runtime_strategy_unchanged"] is True
    assert policy["formal_release_allowed"] is False


def test_promoted_phm_bundle_is_hash_pinned_and_loadable() -> None:
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["strategy"] == "promoted_extra_trees_trace_screening_v1"
    assert manifest["promotion_scope"] == "RETROSPECTIVE_KNOWN_ACTIVATION_TRACE_SCREENING_DEFAULT"
    assert manifest["group_cv_evidence"]["independent_confirmation"] is False
    assert manifest["classification_route"] == "raw"
    for head in ("detection", "localization", "classification"):
        entry = manifest["models"][head]
        path = BUNDLE / entry["path"]
        assert _sha256(path) == entry["sha256"]
        model = joblib.load(path)
        assert model.named_steps["classify"].__class__.__name__ == "ExtraTreesClassifier"
