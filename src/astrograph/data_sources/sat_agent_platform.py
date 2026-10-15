from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .file_source import LocalFileDataSource
from .models import DataEntity, DataSourceMetadata, DataSourceType, DatasetSourceConfig


_FORBIDDEN_MODEL_PREFIXES = (
    "bias.",
    "bias_",
    "degradation.",
    "degradation_",
    "drift.",
    "drift_",
    "dropout.",
    "dropout_",
    "effective.",
    "effective_",
    "injected.",
    "injected_",
    "label.",
    "label_",
    "nominal.",
    "nominal_",
    "truth.",
    "truth_",
    "ground_truth.",
    "ground_truth_",
    "annotation.",
    "effect.",
    "event.",
    "modifier.",
    "object.",
    "status.",
    "stuck.",
    "stuck_",
)
_FORBIDDEN_MODEL_NAMES = {
    "anomaly",
    "annotation",
    "case",
    "case_id",
    "case_role",
    "class_id",
    "class",
    "dataset",
    "dataset_id",
    "end_time_s",
    "experiment_id",
    "fault_active",
    "fault",
    "fault_id",
    "fault_severity",
    "ground_truth",
    "health",
    "health_state",
    "is_anomaly",
    "label",
    "modifier",
    "model",
    "nominal",
    "object",
    "effect",
    "event",
    "mode",
    "nominal_case_id",
    "onset_time_s",
    "pair_id",
    "run_id",
    "run",
    "sample_index",
    "scenario_id",
    "scenario",
    "seed",
    "severity",
    "split",
    "status",
    "target",
    "target_level",
    "target_name",
    "task_id",
    "time",
    "time_s",
    "timestamp",
    "datetime",
    "utc",
    "epoch",
    "index",
    "sample_index",
    "truth",
}
_FORBIDDEN_MODEL_SUFFIXES = (
    ".active",
    "_active",
    ".available",
    "_available",
    ".enabled",
    "_enabled",
    ".flag",
    "_flag",
    ".case_id",
    "_case_id",
    ".class_id",
    "_class_id",
    ".dataset_id",
    "_dataset_id",
    ".fault_id",
    "_fault_id",
    ".locked",
    "_locked",
    ".mode",
    "_mode",
    ".model",
    "_model",
    ".pair_id",
    "_pair_id",
    ".run_id",
    "_run_id",
    ".scenario_id",
    "_scenario_id",
    ".seed",
    "_seed",
    ".safe",
    "_safe",
    ".split",
    "_split",
    ".valid",
    "_valid",
)


def _read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # malformed source evidence must remain visible
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return {}, "root is not an object"
    return payload, "ok"


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


class SourcePathEscapeError(ValueError):
    """A discovered source file resolves outside the configured source root."""


def _require_contained_file(root: Path, path: Path) -> None:
    resolved_root = root.resolve()
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SourcePathEscapeError("source file resolves outside configured root") from exc
    if not resolved.is_file():
        raise ValueError("source candidate is not a regular file")


def _time_axis_candidates(columns: list[str], expected: str | None) -> tuple[list[str], list[str]]:
    """Separate explicit time fields from row-index surrogates.

    The adapter never reconstructs time from row order or sampling metadata. An
    alternative explicit field remains structural metadata until a declarative
    profile names it; neither category is exposed as a model feature.
    """

    explicit: list[str] = []
    index_surrogates: list[str] = []
    expected_lower = str(expected or "").strip().lower()
    for column in columns:
        lowered = str(column).strip().lower()
        if not lowered:
            continue
        if lowered in {"index", "sample_index", "row_index"}:
            index_surrogates.append(str(column))
            continue
        if (
            lowered == expected_lower
            or lowered in {"time", "time_s", "timestamp", "datetime", "utc", "epoch"}
            or "timestamp" in lowered
            or "elapsed" in lowered
            or lowered.startswith("time_")
            or lowered.endswith("_time")
        ):
            explicit.append(str(column))
    return sorted(set(explicit)), sorted(set(index_surrogates))


def is_sat_agent_model_channel_safe(name: str) -> bool:
    """Return whether an uncontracted Sat-Agent field is structurally input-safe.

    ``True`` is not a training qualification.  It means only that the field name
    is not an obvious label, truth, split, identity, or simulator-answer proxy.
    A versioned semantic channel contract is still required before model fitting.
    """
    lowered = str(name).strip().lower()
    if not lowered or lowered in _FORBIDDEN_MODEL_NAMES:
        return False
    if lowered.startswith(_FORBIDDEN_MODEL_PREFIXES):
        return False
    if lowered.endswith(_FORBIDDEN_MODEL_SUFFIXES):
        return False
    if any(
        token in lowered
        for token in (
            ".anomaly",
            "_anomaly",
            ".annotation.",
            "_annotation_",
            ".bias",
            "_bias",
            ".case.",
            "_case_",
            ".class.",
            "_class_",
            ".degradation",
            "_degradation",
            ".drift",
            "_drift",
            ".dropout",
            "_dropout",
            ".effect.",
            "_effect_",
            ".effective_",
            "_effective_",
            ".event.",
            "_event_",
            ".failure",
            "_failure",
            ".fault",
            "_fault",
            ".groundtruth",
            "_groundtruth",
            ".ground_truth.",
            "_ground_truth_",
            ".health",
            "_health",
            ".injected",
            "_injected",
            ".label.",
            "_label_",
            ".modifier.",
            "_modifier_",
            ".nominal_",
            "_nominal_",
            ".object.",
            "_object_",
            ".onset",
            "_onset",
            ".provenance",
            "_provenance",
            ".severity",
            "_severity",
            ".split.",
            "_split_",
            ".status.",
            "_status_",
            ".stuck",
            "_stuck",
            ".target.",
            "_target_",
            ".truth.",
            "_truth_",
            ".active_",
            "_active_",
            ".available_",
            "_available_",
            ".enabled_",
            "_enabled_",
            ".reason",
            "_reason",
            ".saturation",
            "_saturation",
            ".valid_",
            "_valid_",
            "active_effects",
            "fault_active",
            "degradation_active",
            "constraint_active",
            "health_state",
            "case_role",
            "effective_capacity",
        )
    ):
        return False
    return True


class SatAgentPlatformDataSource(LocalFileDataSource):
    """Read-only adapter for canonical ``sat-agent-platform`` dataset bundles.

    The platform is one engineering-simulation source even when it contains many
    campaigns, backends, components, and hierarchy levels.  Canonical AstroGraph
    projections are preferred over their source ``trace.csv`` sibling.  A dataset
    contract is authoritative when present; otherwise a conservative name guard
    removes truth, labels, identifiers, split keys, and direct status proxies.

    Truth access is deliberately unavailable through this adapter.  A future
    independent evaluator must use a separately approved truth reader so ordinary
    diagnosis code cannot turn simulator annotations into model inputs.
    """

    adapter_version = "0.7.1-p018.s0"
    supported_formats = ["csv"]

    def __init__(
        self,
        root: str | Path | DatasetSourceConfig,
        *,
        timestamp_column: str = "time_s",
    ) -> None:
        if isinstance(root, DatasetSourceConfig):
            config = root
            if config.source_type != DataSourceType.SAT_AGENT_PLATFORM:
                raise ValueError(
                    "SatAgentPlatformDataSource requires source_type=sat_agent_platform"
                )
        else:
            config = DatasetSourceConfig(
                source_type=DataSourceType.SAT_AGENT_PLATFORM,
                root=str(root),
                timestamp_column=timestamp_column,
                read_only=True,
                lazy=True,
                recursive=True,
            )
        super().__init__(config)
        self._channel_allowlists: dict[str, list[str]] = {}
        self._candidate_file_count = 0
        self._skipped_candidate_errors: Counter[str] = Counter()

    @staticmethod
    def _case_root(path: Path) -> Path:
        if path.name.lower() == "telemetry.csv" and path.parent.name.lower() == "astrograph":
            return path.parent.parent
        return path.parent

    def _candidate_files(self) -> list[Path]:
        root = self.root
        selected: dict[Path, Path] = {}

        # Contracted AstroGraph projections are the most restrictive and win over
        # a sibling source trace for the same case.
        telemetry_paths = root.rglob("telemetry.csv") if self.config.recursive else root.glob("telemetry.csv")
        for path in telemetry_paths:
            case_root = self._case_root(path)
            relative = path.relative_to(root)
            if (
                path.parent.name.lower() == "astrograph"
                and (path.parent / "dataset_contract.json").is_file()
                and self._matches_filters(relative)
            ):
                selected[case_root] = path

        trace_paths = root.rglob("trace.csv") if self.config.recursive else root.glob("trace.csv")
        for path in trace_paths:
            case_root = self._case_root(path)
            relative = path.relative_to(root)
            if (
                case_root not in selected
                and (case_root / "manifest.json").is_file()
                and self._matches_filters(relative)
            ):
                selected[case_root] = path

        candidates = sorted(selected.values(), key=lambda item: item.relative_to(root).as_posix())
        self._candidate_file_count = len(candidates)
        self._skipped_candidate_errors.clear()
        readable: list[Path] = []
        for path in candidates:
            try:
                _require_contained_file(root, path)
                manifest_path, contract_path = self._evidence_paths(path)
                for evidence_path in (manifest_path, contract_path):
                    if evidence_path.exists():
                        _require_contained_file(root, evidence_path)
                if not self._read_columns(path):
                    raise ValueError("no columns")
            except Exception as exc:
                # Keep the aggregate failure count for qualification evidence, but
                # do not let one empty/invalid historical run hide every readable
                # entity in the platform tree.
                self._skipped_candidate_errors[type(exc).__name__] += 1
                continue
            readable.append(path)
        return readable

    @classmethod
    def _entity_id_for(cls, path: Path, root: Path) -> str:
        case_root = cls._case_root(path)
        relative = "." if case_root == root else case_root.relative_to(root).as_posix()
        digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
        return f"sat-entity-{digest}"

    @classmethod
    def _evidence_paths(cls, path: Path) -> tuple[Path, Path]:
        case_root = cls._case_root(path)
        return case_root / "manifest.json", case_root / "astrograph" / "dataset_contract.json"

    def open(self) -> "SatAgentPlatformDataSource":
        if self._opened:
            return self
        super().open()
        enriched: dict[str, DataEntity] = {}
        self._channel_allowlists = {}

        for entity_id, entity in self._entities.items():
            path = self._entity_paths[entity_id]
            columns = self._read_columns(path)
            explicit_time_candidates, index_surrogate_candidates = _time_axis_candidates(
                columns, self.config.timestamp_column
            )
            structural_time_fields = set(explicit_time_candidates) | set(
                index_surrogate_candidates
            )
            expected_time = str(self.config.timestamp_column or "").strip()
            if expected_time and expected_time in columns:
                time_axis_contract_status = "EXPECTED_EXPLICIT_TIME_PRESENT"
            elif explicit_time_candidates:
                time_axis_contract_status = (
                    "ALTERNATIVE_EXPLICIT_TIME_REQUIRES_DECLARATIVE_MAP"
                )
            elif index_surrogate_candidates:
                time_axis_contract_status = "INDEX_SURROGATE_IS_NOT_PHYSICAL_TIME"
            else:
                time_axis_contract_status = "MISSING_EXPLICIT_TIME_AXIS"
            manifest_path, contract_path = self._evidence_paths(path)
            manifest, manifest_status = _read_json_object(manifest_path)
            contract, contract_status = _read_json_object(contract_path)

            contracted_allowlist = _nested(
                contract,
                "telemetry",
                "channel_contract",
                "training_feature_policy",
                "allowed_model_input_channels",
            )
            is_contracted_projection = (
                path.name.lower() == "telemetry.csv"
                and path.parent.name.lower() == "astrograph"
                and contract_path.is_file()
            )
            allowlist_valid = bool(
                isinstance(contracted_allowlist, list)
                and contracted_allowlist
                and all(
                    isinstance(item, str) and bool(item.strip())
                    for item in contracted_allowlist
                )
                and len(contracted_allowlist) == len(set(contracted_allowlist))
            )
            contract_policy_error_codes: list[str] = []
            if is_contracted_projection and not allowlist_valid:
                requested = []
                rejected_prohibited = []
                allowed = []
                missing = []
                policy = "invalid_contract_fail_closed"
                contract_policy_error_codes = [
                    "CONTRACT_PARSE_FAILED"
                    if contract_status != "ok"
                    else "MODEL_CHANNEL_ALLOWLIST_INVALID_OR_MISSING"
                ]
            elif allowlist_valid:
                requested = [str(item) for item in contracted_allowlist]
                rejected_prohibited = [
                    item
                    for item in requested
                    if item in structural_time_fields
                    or not is_sat_agent_model_channel_safe(item)
                ]
                missing = [item for item in requested if item not in columns]
                if is_contracted_projection and (rejected_prohibited or missing):
                    allowed = []
                    policy = "invalid_contract_fail_closed"
                    if rejected_prohibited:
                        contract_policy_error_codes.append(
                            "PROHIBITED_OR_TIMESTAMP_MODEL_CHANNEL"
                        )
                    if missing:
                        contract_policy_error_codes.append("MISSING_MODEL_CHANNEL")
                else:
                    allowed = [
                        item
                        for item in requested
                        if item in columns
                        and item not in structural_time_fields
                        and is_sat_agent_model_channel_safe(item)
                    ]
                    policy = "dataset_contract_allowlist"
            else:
                requested = []
                rejected_prohibited = []
                missing = []
                allowed = [
                    item
                    for item in columns
                    if item not in structural_time_fields
                    and is_sat_agent_model_channel_safe(item)
                ]
                policy = "conservative_uncontracted_guard"
            self._channel_allowlists[entity_id] = allowed

            row_count = _nested(contract, "telemetry", "row_count")
            if not isinstance(row_count, int):
                row_count = _nested(manifest, "quality", "trace_rows")
            if not isinstance(row_count, int):
                row_count = None

            task = manifest.get("task") if isinstance(manifest.get("task"), dict) else {}
            run = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
            simulation = (
                contract.get("simulation") if isinstance(contract.get("simulation"), dict) else {}
            )
            fidelity = (
                simulation.get("fidelity")
                if isinstance(simulation.get("fidelity"), dict)
                else {}
            )
            metadata: dict[str, Any] = {
                **entity.metadata,
                "source_group_id": "SAT_AGENT_PLATFORM",
                "source_family_alias": "sat-agent-platform",
                "manifest_present": manifest_path.is_file(),
                "manifest_parse_status": manifest_status,
                "manifest_version": manifest.get("manifest_version"),
                "contract_present": contract_path.is_file(),
                "contract_parse_status": contract_status,
                "contract_schema_version": contract.get("schema_version"),
                "capability_id": task.get("capability_id") or _nested(contract, "simulation", "capability_id"),
                # ``labels.target_level`` and ``labels.target_name`` are evaluator
                # annotations in Sat-Agent manifests.  Do not surface them from
                # the ordinary model adapter, even as metadata: callers must bind
                # an approved schema/profile explicitly instead of learning or
                # routing from target truth.
                "hierarchy_level": task.get("task_type"),
                "component_archetype": None,
                "duration_s": run.get("duration_s") or simulation.get("duration_s"),
                "sample_s": run.get("sample_s") or simulation.get("sample_s"),
                "backend_family": simulation.get("backend")
                or _nested(manifest, "capability", "source_binding", "mode"),
                "fidelity_level": fidelity.get("fidelity_level"),
                "flight_validated": bool(fidelity.get("flight_validated", False)),
                "model_channel_policy": policy,
                "contract_requested_channel_count": len(requested),
                "contract_missing_model_channels": missing,
                "contract_rejected_prohibited_model_channels": rejected_prohibited,
                "contract_policy_error_codes": contract_policy_error_codes,
                "time_axis_contract_status": time_axis_contract_status,
                "expected_time_axis_field": expected_time or None,
                "explicit_time_axis_candidates": explicit_time_candidates,
                "index_surrogate_candidates": index_surrogate_candidates,
                "time_axis_metadata_is_model_input": False,
                "truth_access": "DENIED_IN_SOURCE_ADAPTER",
                "metadata_is_model_input": False,
            }
            enriched[entity_id] = entity.model_copy(
                update={
                    "relative_path": f"${{ASTROGRAPH_SAT_AGENT_ROOT}}/{entity_id}",
                    "row_count": row_count,
                    "channel_count": len(allowed),
                    "metadata": metadata,
                }
            )

        self._entities = enriched
        return self

    def close(self) -> None:
        super().close()
        self._channel_allowlists.clear()

    def list_channels(self, entity_id: str) -> list[str]:
        self._entity(entity_id)
        return list(self._channel_allowlists[entity_id])

    def read_labels(
        self,
        entity_id: str,
        start: object | None = None,
        end: object | None = None,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        self._entity(entity_id)
        raise PermissionError(
            "Sat-Agent truth is evaluator-only and is not exposed by the model data-source adapter"
        )

    def get_metadata(self) -> DataSourceMetadata:
        metadata = super().get_metadata()
        contracted_count = sum(
            1
            for item in self._entities.values()
            if item.metadata.get("model_channel_policy") == "dataset_contract_allowlist"
        )
        invalid_contract_count = sum(
            1
            for item in self._entities.values()
            if item.metadata.get("model_channel_policy") == "invalid_contract_fail_closed"
        )
        time_axis_status_counts = Counter(
            str(item.metadata.get("time_axis_contract_status", "UNDECLARED"))
            for item in self._entities.values()
        )
        return metadata.model_copy(
            update={
                "source_type": DataSourceType.SAT_AGENT_PLATFORM,
                "root_token": "${ASTROGRAPH_SAT_AGENT_ROOT}",
                "adapter_version": self.adapter_version,
                "metadata": {
                    **metadata.metadata,
                    "layout": "sat-agent-platform-canonical-dataset-bundles",
                    "source_id": "SAT_AGENT_PLATFORM",
                    "source_count": 1,
                    "contracted_entity_count": contracted_count,
                    "invalid_contract_fail_closed_entity_count": invalid_contract_count,
                    "uncontracted_guarded_entity_count": (
                        len(self._entities) - contracted_count - invalid_contract_count
                    ),
                    "truth_reader_available": False,
                    "heterogeneous_component_schemas_preserved": True,
                    "candidate_file_count": self._candidate_file_count,
                    "skipped_unreadable_candidate_count": sum(
                        self._skipped_candidate_errors.values()
                    ),
                    "skipped_unreadable_candidate_error_counts": dict(
                        sorted(self._skipped_candidate_errors.items())
                    ),
                    "time_axis_contract_status_counts": dict(
                        sorted(time_axis_status_counts.items())
                    ),
                },
                "claim_boundary": (
                    "E0 engineering data from one Sat-Agent platform source. Multiple campaigns, "
                    "components, hierarchy levels, configurations, or backends are not independent "
                    "data sources and do not establish cross-source, flight, or real-world generalization."
                ),
            }
        )
