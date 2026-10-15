from __future__ import annotations

from pathlib import Path

from astrograph.data.delivery_archive import materialize_d3_delivery
from astrograph.evidence import D3RWExecutableEvidenceBuilder, D3RWExecutableEvidenceCatalog
from astrograph.object_plugins.base import ObjectDiagnosticPlugin
from astrograph.object_plugins.models import (
    ObjectEvidenceBuildRequest,
    ObjectEvidenceBuildResult,
    ObjectPluginManifest,
    PluginReviewStatus,
)
from astrograph.semantics import TelemetrySemanticsRegistry


class D3ReactionWheelObjectPlugin(ObjectDiagnosticPlugin):
    """Compatibility-preserving object plugin for the existing D3 RW service."""

    _manifest = ObjectPluginManifest(
        plugin_id="builtin.d3.reaction_wheel",
        plugin_version="0.5.13",
        object_type="reaction_wheel",
        subsystem="ADCS",
        supported_dataset_profiles=["PHM_D3_RW_PRODUCTION", "fake-d3-rw"],
        required_operator_ids=[
            "MAX_ABS_PAIRED_RESIDUAL",
            "MAX_ABS_STATUS_RESIDUAL",
            "MEAN_ABS_PAIRED_RESIDUAL",
            "LAST_PAIRED_RESIDUAL",
            "AVAILABILITY_DROP_FRACTION",
        ],
        review_status=PluginReviewStatus.DEVELOPMENT,
        source_refs=[
            "configs/reaction_wheel_evidence.json",
            "protocols/phm_project_a",
        ],
        claim_boundary="D3 synthetic reaction-wheel engineering evidence; not flight-qualified.",
        metadata={
            "legacy_builder": "D3RWExecutableEvidenceBuilder",
            "legacy_tool_alias": "d3_rw.evidence.build",
        },
    )

    def __init__(self) -> None:
        self._directory_cache: dict[tuple[str, str, str], D3RWExecutableEvidenceBuilder] = {}

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
                builder = self._builder(delivery, protocol_root, catalog_path)
                self._directory_cache[key] = builder
            return self._result(builder, request.run_id)

        with materialize_d3_delivery(delivery) as package_root:
            builder = self._builder(package_root, protocol_root, catalog_path)
            return self._result(builder, request.run_id)

    @staticmethod
    def _builder(
        package_root: str | Path,
        protocol_root: str,
        catalog_path: str,
    ) -> D3RWExecutableEvidenceBuilder:
        return D3RWExecutableEvidenceBuilder(
            package_root,
            semantics_registry=TelemetrySemanticsRegistry(protocol_root),
            catalog=D3RWExecutableEvidenceCatalog(catalog_path),
        )

    def _result(
        self,
        builder: D3RWExecutableEvidenceBuilder,
        run_id: str,
    ) -> ObjectEvidenceBuildResult:
        bundle = builder.build_run(run_id)
        return ObjectEvidenceBuildResult(
            plugin_id=self.manifest.plugin_id,
            plugin_version=self.manifest.plugin_version,
            builder_id=builder.builder_id,
            builder_version=builder.builder_version,
            bundle=bundle.model_dump(mode="json"),
            core_payload=builder.core_evidence(bundle),
            metadata={
                "domain_id": bundle.domain_id,
                "object_type": self.manifest.object_type,
                "subsystem": self.manifest.subsystem,
            },
        )
