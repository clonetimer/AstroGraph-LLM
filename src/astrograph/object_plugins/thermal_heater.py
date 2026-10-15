from __future__ import annotations

from pathlib import Path
from typing import Callable

from astrograph.data.thermal_star_adapter import materialize_d3_thermal_star_delivery
from astrograph.evidence import D3RWExecutableEvidenceCatalog, D3ThermalStarExecutableEvidenceBuilder
from astrograph.evidence.executable import EpisodeEvidenceBundle
from astrograph.object_plugins.base import ObjectDiagnosticPlugin
from astrograph.object_plugins.models import (
    ObjectEvidenceBuildRequest,
    ObjectEvidenceBuildResult,
    ObjectPluginManifest,
    PluginReviewStatus,
)
from astrograph.semantics import TelemetrySemanticsRegistry


BuilderFactory = Callable[[str | Path, str, str], D3ThermalStarExecutableEvidenceBuilder]


class D3ThermalHeaterObjectPlugin(ObjectDiagnosticPlugin):
    """Second built-in object plugin, extracted from the D3 thermal/star service."""

    target_fault_id = "THERMAL_HEATER_STUCK_ON"
    supported_case_ids = {
        "thermal_heater_stuck_on",
        "nominal_heater_on",
        "nominal_heater_off",
    }

    _manifest = ObjectPluginManifest(
        plugin_id="builtin.d3.thermal_heater",
        plugin_version="0.5.13",
        object_type="thermal_heater",
        subsystem="THERMAL",
        supported_dataset_profiles=["PHM_D3_THERMAL_STAR_PRODUCTION"],
        required_operator_ids=[
            "JOINT_THRESHOLD_FRACTION",
            "MEAN_VALUE",
            "MAX_ABS_PAIRED_RESIDUAL",
        ],
        review_status=PluginReviewStatus.DEVELOPMENT,
        source_refs=[
            "configs/thermal_star_evidence.json",
            "docs/source_evidence/THERMAL_HEATER_STUCK_ON.yaml",
            "protocols/phm_project_a",
        ],
        claim_boundary="D3 synthetic thermal-heater engineering evidence; not flight-qualified.",
        metadata={
            "legacy_builder": "D3ThermalStarExecutableEvidenceBuilder",
            "migration_scope": "THERMAL_HEATER_STUCK_ON",
            "feature_runtime": "approved-operator-registry",
        },
    )

    def __init__(self, *, builder_factory: BuilderFactory | None = None) -> None:
        self._builder_factory = builder_factory or self._builder
        self._directory_cache: dict[
            tuple[str, str, str], D3ThermalStarExecutableEvidenceBuilder
        ] = {}

    @property
    def manifest(self) -> ObjectPluginManifest:
        return self._manifest

    def build_evidence(self, request: ObjectEvidenceBuildRequest) -> ObjectEvidenceBuildResult:
        delivery = Path(request.delivery_path).resolve()
        protocol_root = str(Path(request.protocol_root).resolve())
        catalog_path = str(Path(request.evidence_catalog_path).resolve())
        if delivery.is_dir():
            key = (str(delivery), protocol_root, catalog_path)
            builder = self._directory_cache.get(key)
            if builder is None:
                builder = self._builder_factory(delivery, protocol_root, catalog_path)
                self._directory_cache[key] = builder
            return self._result(builder, request.run_id)

        with materialize_d3_thermal_star_delivery(delivery) as package_root:
            builder = self._builder_factory(package_root, protocol_root, catalog_path)
            return self._result(builder, request.run_id)

    @staticmethod
    def _builder(
        package_root: str | Path,
        protocol_root: str,
        catalog_path: str,
    ) -> D3ThermalStarExecutableEvidenceBuilder:
        return D3ThermalStarExecutableEvidenceBuilder(
            package_root,
            semantics_registry=TelemetrySemanticsRegistry(protocol_root),
            catalog=D3RWExecutableEvidenceCatalog(catalog_path),
        )

    def _result(
        self,
        builder: D3ThermalStarExecutableEvidenceBuilder,
        run_id: str,
    ) -> ObjectEvidenceBuildResult:
        bundle = self.filter_bundle(builder.build_run(run_id))
        return ObjectEvidenceBuildResult(
            plugin_id=self.manifest.plugin_id,
            plugin_version=self.manifest.plugin_version,
            builder_id=builder.builder_id,
            builder_version=builder.builder_version,
            bundle=bundle.model_dump(mode="json"),
            core_payload=builder.core_payload(bundle),
            metadata={
                "domain_id": bundle.domain_id,
                "object_type": self.manifest.object_type,
                "subsystem": self.manifest.subsystem,
                "target_fault_id": self.target_fault_id,
                "feature_count": len(bundle.feature_results),
                "evidence_count": len(bundle.evidence_facts),
            },
        )

    def filter_bundle(self, bundle: EpisodeEvidenceBundle) -> EpisodeEvidenceBundle:
        case_id = str(bundle.evaluation_metadata.get("truth_case_id", ""))
        if case_id and case_id not in self.supported_case_ids:
            raise ValueError(
                f"run case {case_id!r} is not supported by {self.manifest.plugin_id}"
            )
        evidence_definitions = [
            item
            for item in bundle.evidence_definitions
            if item.target_fault_id == self.target_fault_id
        ]
        feature_ids = {item.feature_definition_id for item in evidence_definitions}
        feature_definitions = [
            item for item in bundle.feature_definitions if item.feature_definition_id in feature_ids
        ]
        feature_results = [
            item for item in bundle.feature_results if item.feature_definition_id in feature_ids
        ]
        evidence_ids = {item.evidence_definition_id for item in evidence_definitions}
        evidence_facts = [
            item for item in bundle.evidence_facts if item.evidence_definition_id in evidence_ids
        ]
        if not evidence_definitions or not feature_results:
            raise ValueError("thermal-heater evidence catalog is absent from the source bundle")
        mechanism_detected = (
            case_id == "thermal_heater_stuck_on"
            and bundle.episode.source != "full_run_no_mechanism"
        )
        metadata = dict(bundle.evaluation_metadata)
        metadata.update(
            {
                "object_plugin_id": self.manifest.plugin_id,
                "object_plugin_version": self.manifest.plugin_version,
                "source_domain_id": bundle.domain_id,
                "filtered_target_fault_id": self.target_fault_id,
            }
        )
        return bundle.model_copy(
            update={
                "domain_id": "D3_THERMAL_HEATER",
                "analysis_component_instance": "PAYLOAD_HEATER",
                "selected_component_instance": "PAYLOAD_HEATER" if mechanism_detected else None,
                "component_selection_status": (
                    "MECHANISM_DETECTED" if mechanism_detected else "NO_MECHANISM_DETECTED"
                ),
                "feature_definitions": feature_definitions,
                "feature_results": feature_results,
                "evidence_definitions": evidence_definitions,
                "evidence_facts": evidence_facts,
                "evaluation_metadata": metadata,
            }
        )
