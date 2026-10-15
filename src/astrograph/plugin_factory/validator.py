from __future__ import annotations

import json
from pathlib import Path

from astrograph.object_plugins.models import ObjectPluginManifest
from astrograph.plugin_factory.graph import OperatorGraphCompiler
from astrograph.plugin_factory.models import ObjectPluginBuildSpec, PluginGenerationPlan


class GeneratedPluginValidator:
    def __init__(self) -> None:
        self.graph_compiler = OperatorGraphCompiler()

    REQUIRED_FILES = {
        "manifest.json",
        "build_spec.json",
        "generation_plan.json",
        "object_profile.json",
        "feature_catalog.json",
        "evidence_catalog.json",
        "plugin.py",
        "test_plugin_contract.py",
        "README.md",
    }

    def validate(self, package_root: str | Path) -> dict[str, object]:
        root = Path(package_root).resolve()
        missing = sorted(name for name in self.REQUIRED_FILES if not (root / name).exists())
        if missing:
            raise ValueError(f"generated plugin package is incomplete: {missing}")
        manifest = ObjectPluginManifest.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))
        spec = ObjectPluginBuildSpec.model_validate_json((root / "build_spec.json").read_text(encoding="utf-8"))
        plan = PluginGenerationPlan.model_validate_json((root / "generation_plan.json").read_text(encoding="utf-8"))
        if len({manifest.plugin_id, spec.plugin_id, plan.plugin_id}) != 1:
            raise ValueError("generated plugin package has inconsistent plugin IDs")
        for feature in spec.features:
            if feature.operator_graph is not None:
                self.graph_compiler.compile(feature.operator_graph)
        profile = json.loads((root / "object_profile.json").read_text(encoding="utf-8"))
        if profile["object_type"] != manifest.object_type:
            raise ValueError("object profile type does not match manifest")
        return {
            "valid": True,
            "plugin_id": manifest.plugin_id,
            "plugin_version": manifest.plugin_version,
            "review_status": manifest.review_status.value,
            "online_registration_allowed": bool(
                manifest.metadata.get("online_registration_allowed", False)
            ),
            "configuration_only_feature_count": plan.configuration_only_feature_count,
            "composition_feature_count": plan.composition_feature_count,
            "custom_code_feature_count": plan.custom_code_feature_count,
            "blocked_feature_count": plan.blocked_feature_count,
        }
