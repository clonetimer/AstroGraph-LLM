from __future__ import annotations

import json
from pathlib import Path

from astrograph.object_plugins.models import ObjectPluginManifest, PluginReviewStatus
from astrograph.plugin_factory.models import ObjectPluginBuildSpec, PluginGenerationPlan


class ObjectPluginScaffolder:
    """Writes a reviewable plugin package without executing generated code."""

    def scaffold(
        self,
        spec: ObjectPluginBuildSpec,
        plan: PluginGenerationPlan,
        output_root: str | Path,
    ) -> Path:
        root = Path(output_root).resolve() / spec.plugin_id.replace(".", "_")
        root.mkdir(parents=True, exist_ok=True)
        manifest = ObjectPluginManifest(
            plugin_id=spec.plugin_id,
            plugin_version=spec.plugin_version,
            object_type=spec.object_type,
            subsystem=spec.subsystem,
            supported_dataset_profiles=spec.dataset_profiles,
            required_operator_ids=plan.required_operator_ids,
            review_status=(
                PluginReviewStatus.REVIEW_REQUIRED
                if plan.requires_human_review
                else PluginReviewStatus.DEVELOPMENT
            ),
            source_refs=spec.source_refs,
            claim_boundary=spec.claim_boundary,
            metadata={
                "generated_by": "astrograph-object-plugin-factory-v0.5.13",
                "online_registration_allowed": False,
                "custom_code_feature_count": plan.custom_code_feature_count,
                "blocked_feature_count": plan.blocked_feature_count,
            },
        )
        self._write_json(root / "manifest.json", manifest.model_dump(mode="json"))
        self._write_json(root / "build_spec.json", spec.model_dump(mode="json"))
        self._write_json(root / "generation_plan.json", plan.model_dump(mode="json"))
        self._write_json(
            root / "object_profile.json",
            {
                "object_type": spec.object_type,
                "subsystem": spec.subsystem,
                "dataset_profiles": spec.dataset_profiles,
                "object_parameters": spec.object_parameters,
                "telemetry_bindings": [item.model_dump(mode="json") for item in spec.telemetry_bindings],
            },
        )
        self._write_json(
            root / "feature_catalog.json",
            {
                "schema_version": "astrograph-generated-feature-catalog.v1",
                "features": [item.model_dump(mode="json") for item in spec.features],
            },
        )
        self._write_json(
            root / "evidence_catalog.json",
            {
                "schema_version": "astrograph-generated-evidence-catalog.v1",
                "evidence": [item.model_dump(mode="json") for item in spec.evidence],
            },
        )
        (root / "plugin.py").write_text(self._plugin_template(spec, plan), encoding="utf-8")
        (root / "test_plugin_contract.py").write_text(self._test_template(spec), encoding="utf-8")
        (root / "README.md").write_text(self._readme(spec, plan), encoding="utf-8")
        if plan.custom_code_feature_count or plan.blocked_feature_count:
            (root / "REVIEW_REQUIRED.md").write_text(
                "# Review required\n\n"
                "This package contains custom-code or blocked feature requests. It must not be "
                "registered in the online runtime until formulas, units, source references, tests, "
                "and generated code have passed human review.\n",
                encoding="utf-8",
            )
        return root

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _plugin_template(spec: ObjectPluginBuildSpec, plan: PluginGenerationPlan) -> str:
        return f'''"""Generated deterministic feature-service scaffold for {spec.plugin_id}.

Direct and DAG-composed registry-backed features are executable after generation. Online plugin
registration remains disabled until the object data adapter, evidence tests,
and approval record are complete. Custom formulas are never executed here.
"""

from pathlib import Path

from astrograph.plugin_factory import ConfigDrivenFeatureService, ObjectPluginBuildSpec

PLUGIN_ID = {spec.plugin_id!r}
PLUGIN_VERSION = {spec.plugin_version!r}
ONLINE_REGISTRATION_ALLOWED = False
REQUIRED_OPERATOR_IDS = {plan.required_operator_ids!r}

_ROOT = Path(__file__).resolve().parent
_SPEC = ObjectPluginBuildSpec.model_validate_json(
    (_ROOT / "build_spec.json").read_text(encoding="utf-8")
)
_FEATURES = {{item.feature_id: item for item in _SPEC.features}}
_SERVICE = ConfigDrivenFeatureService()


def compute_feature(feature_id, *, current_signals, baseline_signals=None, time_s=None):
    """Execute one approved direct or DAG-composed feature deterministically."""
    try:
        feature = _FEATURES[feature_id]
    except KeyError as exc:
        raise KeyError(f"unknown generated feature: {{feature_id}}") from exc
    if feature.custom_formula is not None:
        raise RuntimeError(
            f"feature {{feature_id}} requires reviewed custom code and cannot execute "
            "in the configuration runtime"
        )
    return _SERVICE.compute(
        feature,
        current_signals=current_signals,
        baseline_signals=baseline_signals,
        time_s=time_s,
    ).model_dump(mode="json")


def register(registry):
    raise RuntimeError(
        "Generated feature service is not an approved online object plugin. "
        "Complete the object adapter, evidence tests, and approval record first."
    )
'''

    @staticmethod
    def _test_template(spec: ObjectPluginBuildSpec) -> str:
        return f'''import json
from pathlib import Path


def test_generated_contract_files_are_self_consistent():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    build_spec = json.loads((root / "build_spec.json").read_text(encoding="utf-8"))
    assert manifest["plugin_id"] == {spec.plugin_id!r}
    assert build_spec["plugin_id"] == manifest["plugin_id"]
    assert manifest["metadata"]["online_registration_allowed"] is False
'''

    @staticmethod
    def _readme(spec: ObjectPluginBuildSpec, plan: PluginGenerationPlan) -> str:
        return f"""# {spec.plugin_id}

Generated by AstroGraph Object Plugin Factory V0.5.13.

- Object type: `{spec.object_type}`
- Subsystem: `{spec.subsystem}`
- Configuration-only features: {plan.configuration_only_feature_count}
- Operator-composition features: {plan.composition_feature_count}
- Custom-code features: {plan.custom_code_feature_count}
- Blocked features: {plan.blocked_feature_count}
- Online registration: **disabled**

The Agent produces `build_spec.json`; deterministic factory code produces the
remaining scaffold. Direct and approved DAG-composition features are executable through
`compute_feature`. Custom-code features remain blocked until a reviewed
implementation, static analysis, isolated tests, and approval record are complete.
"""
