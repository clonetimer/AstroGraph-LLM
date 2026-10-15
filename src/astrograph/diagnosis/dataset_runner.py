from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np
import pandas as pd

from astrograph.audit.store import AuditStore, NullAuditStore, sha256_json
from astrograph.config import DiagnosisPolicy
from astrograph.data.adapter import SIGNAL_TO_ANOMALY
from astrograph.data_sources.base import DatasetSource
from astrograph.diagnosis.workflow import invoke_workflow
from astrograph.knowledge.repository import KnowledgeRepository
from astrograph.schemas import AnomalyEvidence, DiagnosisStatus, WorkflowInput

from .dataset_models import (
    ConclusionLevel,
    DatasetDiagnosisRequest,
    DetectionRecord,
    DiagnosisRunState,
    DiagnosticConclusion,
    LayeredDatasetDiagnosis,
    ObservationFact,
    utc_now,
)


DETECTOR_ID = "astrograph-robust-baseline-deviation.v1"


DatasetDetector = Callable[..., tuple[list[ObservationFact], list[DetectionRecord], list[AnomalyEvidence]]]


def _stable_id(prefix: str, payload: Any, length: int = 24) -> str:
    digest = sha256_json(payload).upper()
    return f"{prefix}-{digest[:length]}"


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": [str(item) for item in frame.columns],
                "dtypes": [str(item) for item in frame.dtypes],
                "shape": list(frame.shape),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    try:
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    except Exception:
        digest.update(frame.to_csv(index=True).encode("utf-8"))
    return digest.hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _observed_time(frame: pd.DataFrame, timestamp_column: str | None, index: int) -> Any:
    if timestamp_column and timestamp_column in frame.columns:
        value = frame.iloc[index][timestamp_column]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float)):
            return value
        return str(value)
    return int(index)


def _mapping_for_signal(
    signal: str,
    explicit: dict[str, dict[str, str]],
) -> tuple[str, str, bool]:
    configured = explicit.get(signal)
    if configured:
        anomaly_id = str(configured.get("anomaly_id", "")).strip()
        name = str(configured.get("name") or configured.get("anomaly_name") or "").strip()
        if not anomaly_id or not name:
            raise ValueError(
                f"signal_mapping for {signal!r} requires anomaly_id and name/anomaly_name"
            )
        return anomaly_id, name, True
    policy = explicit.get("__policy__", {})
    allow_builtin = str(policy.get("allow_builtin_fallback", "true")).strip().lower() not in {"0", "false", "no"}
    if not allow_builtin:
        token = hashlib.sha256(signal.encode("utf-8")).hexdigest()[:16].upper()
        return f"ANOM_UNMAPPED_{token}", f"Unmapped anomaly on {signal}", False
    built_in = SIGNAL_TO_ANOMALY.get(signal)
    if built_in:
        return built_in[0], built_in[1], True
    token = hashlib.sha256(signal.encode("utf-8")).hexdigest()[:16].upper()
    return f"ANOM_UNMAPPED_{token}", f"Unmapped anomaly on {signal}", False


def detect_robust_deviations(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    channels: list[str],
    baseline_fraction: float,
    z_threshold: float,
    minimum_baseline_points: int,
    signal_mapping: dict[str, dict[str, str]] | None = None,
) -> tuple[list[ObservationFact], list[DetectionRecord], list[AnomalyEvidence]]:
    """Detect bounded structural deviations without using labels or truth columns.

    Scores are robust residual strengths, not calibrated probabilities. Unknown
    signals remain explicitly unmapped so the downstream knowledge step can return
    UNKNOWN instead of inventing a fault.
    """

    mapping = signal_mapping or {}
    observations: list[ObservationFact] = []
    detections: list[DetectionRecord] = []
    anomalies: list[AnomalyEvidence] = []
    if frame.empty:
        return observations, detections, anomalies

    baseline_count = max(minimum_baseline_points, int(round(len(frame) * baseline_fraction)))
    baseline_count = min(baseline_count, max(1, len(frame) - 1))

    for signal in channels:
        if signal not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[signal], errors="coerce")
        baseline = numeric.iloc[:baseline_count].dropna()
        missing_count = int(numeric.isna().sum())
        observation_payload: dict[str, Any] = {
            "signal": signal,
            "sample_count": len(frame),
            "missing_count": missing_count,
            "baseline_count": len(baseline),
        }
        if len(baseline) < minimum_baseline_points:
            observations.append(
                ObservationFact(
                    observation_id=_stable_id("OBS", observation_payload),
                    signal=signal,
                    sample_count=len(frame),
                    missing_count=missing_count,
                    metadata={
                        "detector_id": DETECTOR_ID,
                        "status": "INSUFFICIENT_BASELINE",
                        "baseline_count": len(baseline),
                    },
                )
            )
            continue

        median = float(baseline.median())
        mad = float((baseline - median).abs().median())
        robust_scale = 1.4826 * mad
        if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
            std = float(baseline.std(ddof=0))
            robust_scale = std if np.isfinite(std) and std > 1e-12 else max(abs(median) * 1e-6, 1e-9)
        score_series = ((numeric - median).abs() / robust_scale).replace([np.inf, -np.inf], np.nan)
        search = score_series.iloc[baseline_count:]
        maximum_score = float(search.max()) if search.notna().any() else 0.0
        post_missing = numeric.iloc[baseline_count:].isna()
        missing_fraction = float(post_missing.mean()) if len(post_missing) else 0.0
        observation_payload.update(
            {
                "baseline_median": median,
                "baseline_mad": mad,
                "maximum_robust_score": maximum_score,
                "missing_fraction": missing_fraction,
            }
        )
        observations.append(
            ObservationFact(
                observation_id=_stable_id("OBS", observation_payload),
                signal=signal,
                sample_count=len(frame),
                missing_count=missing_count,
                baseline_median=median,
                baseline_mad=mad,
                maximum_robust_score=maximum_score,
                metadata={
                    "detector_id": DETECTOR_ID,
                    "baseline_count": len(baseline),
                    "robust_scale": robust_scale,
                    "score_is_calibrated_probability": False,
                },
            )
        )

        threshold_mask = search >= z_threshold
        has_score_detection = bool(threshold_mask.fillna(False).any())
        has_missing_detection = missing_fraction > 0.0
        if not has_score_detection and not has_missing_detection:
            continue

        if has_score_detection:
            relative_index = int(np.flatnonzero(threshold_mask.fillna(False).to_numpy())[0])
            detected_index = baseline_count + relative_index
        else:
            relative_index = int(np.flatnonzero(post_missing.to_numpy())[0])
            detected_index = baseline_count + relative_index
        anomaly_id, anomaly_name, mapped = _mapping_for_signal(signal, mapping)
        effective_score = max(maximum_score, z_threshold if has_missing_detection else 0.0)
        confidence = min(0.999, effective_score / (effective_score + z_threshold))
        detection_payload = {
            "anomaly_id": anomaly_id,
            "signal": signal,
            "data_index": detected_index,
            "score": effective_score,
            "detector_id": DETECTOR_ID,
        }
        observed_at = _observed_time(frame, timestamp_column, detected_index)
        detection = DetectionRecord(
            detection_id=_stable_id("DET", detection_payload),
            anomaly_id=anomaly_id,
            anomaly_name=anomaly_name,
            signal=signal,
            detector_id=DETECTOR_ID,
            score=effective_score,
            confidence=confidence,
            observed_at=observed_at,
            missing_fraction=missing_fraction,
            mapped=mapped,
            metadata={
                "z_threshold": z_threshold,
                "baseline_count": baseline_count,
                "score_triggered": has_score_detection,
                "missing_triggered": has_missing_detection,
                "score_is_calibrated_probability": False,
            },
        )
        detections.append(detection)
        observed_at_s = _safe_float(observed_at)
        anomalies.append(
            AnomalyEvidence(
                anomaly_id=anomaly_id,
                name=anomaly_name,
                signal=signal,
                confidence=confidence,
                confidence_type="robust_residual_strength_not_calibrated",
                observed_at_s=observed_at_s if observed_at_s is not None and observed_at_s >= 0 else None,
                metadata={
                    "detection_id": detection.detection_id,
                    "detector_id": DETECTOR_ID,
                    "mapped_to_registered_semantics": mapped,
                    "missing_fraction": missing_fraction,
                    "maximum_robust_score": effective_score,
                },
            )
        )

    # Preserve one strongest item for duplicate semantic anomaly IDs.
    strongest: dict[str, tuple[DetectionRecord, AnomalyEvidence]] = {}
    detection_by_id = {item.detection_id: item for item in detections}
    for anomaly in anomalies:
        detection = detection_by_id[str(anomaly.metadata["detection_id"])]
        previous = strongest.get(anomaly.anomaly_id)
        if previous is None or anomaly.confidence > previous[1].confidence:
            strongest[anomaly.anomaly_id] = (detection, anomaly)
    selected_detection_ids = {pair[0].detection_id for pair in strongest.values()}
    detections = [item for item in detections if item.detection_id in selected_detection_ids]
    anomalies = [pair[1] for _, pair in sorted(strongest.items())]
    return observations, detections, anomalies


class JsonDiagnosisRunStateStore:
    """Atomic per-run state store used for idempotent resume on one local node."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._lock = Lock()

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not run_id or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for char in run_id):
            raise ValueError("unsafe run_id")

    def path_for(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.root / f"{run_id}.json"

    def get(self, run_id: str) -> DiagnosisRunState | None:
        path = self.path_for(run_id)
        if not path.is_file():
            return None
        return DiagnosisRunState.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, state: DiagnosisRunState) -> None:
        path = self.path_for(state.run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        state.updated_at = utc_now()
        with self._lock:
            temporary.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
            for attempt in range(5):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if os.name != "nt" or attempt == 4:
                        raise
                    # Antivirus/indexing can briefly hold the existing state
                    # file on Windows. Preserve atomic replace semantics while
                    # bounding the retry interval.
                    time.sleep(0.05 * (attempt + 1))

    def delete(self, run_id: str) -> None:
        self.path_for(run_id).unlink(missing_ok=True)


def _conclusion_from_diagnosis(diagnosis: dict[str, Any]) -> DiagnosticConclusion:
    status = DiagnosisStatus(diagnosis.get("status", DiagnosisStatus.SYSTEM_ERROR.value))
    if status == DiagnosisStatus.DIAGNOSED:
        level = ConclusionLevel.PROBABLE
    elif status == DiagnosisStatus.AMBIGUOUS:
        level = ConclusionLevel.POSSIBLE
    elif status == DiagnosisStatus.UNKNOWN:
        level = ConclusionLevel.UNKNOWN
    else:
        level = ConclusionLevel.SYSTEM_ERROR
    return DiagnosticConclusion(
        level=level,
        fault_id=diagnosis.get("conclusion_fault_id"),
        fault_name=diagnosis.get("conclusion_fault_name"),
        confidence=max(0.0, min(1.0, float(diagnosis.get("top_score") or 0.0))),
        reasons=[str(item) for item in diagnosis.get("reasons", [])],
        claim_boundary=(
            "A DIAGNOSED core status is represented as probable, never confirmed, "
            "because detector scores and E0 knowledge weights are not calibrated "
            "flight probabilities. UNKNOWN is a valid fail-closed result."
        ),
    )


class DatasetDiagnosisRunner:
    """Read -> detect -> retrieve/diagnose -> layered report with resumable state."""

    def __init__(
        self,
        source: DatasetSource,
        repository: KnowledgeRepository,
        *,
        state_store: JsonDiagnosisRunStateStore,
        policy: DiagnosisPolicy | None = None,
        audit_store: AuditStore | None = None,
        detector: DatasetDetector | None = None,
        detector_id: str | None = None,
    ) -> None:
        self.source = source
        self.repository = repository
        self.state_store = state_store
        self.policy = policy or DiagnosisPolicy()
        self.audit_store = audit_store or NullAuditStore()
        self.detector = detector or detect_robust_deviations
        self.detector_id = str(detector_id or DETECTOR_ID).strip()
        if not self.detector_id:
            raise ValueError("detector_id cannot be empty")
        if detector is None and self.detector_id != DETECTOR_ID:
            raise ValueError("a non-default detector_id requires an explicit detector")

    def identity(self, request: DatasetDiagnosisRequest) -> tuple[str, str, str]:
        source_opened_here = not self.source.opened
        if source_opened_here:
            self.source.open()
        try:
            source_metadata = self.source.get_metadata()
            payload = {
                "source_type": source_metadata.source_type.value,
                "root_token": source_metadata.root_token,
                "entity_id": request.entity_id,
                "start": request.start,
                "end": request.end,
                "channels": request.channels,
                "baseline_fraction": request.baseline_fraction,
                "z_threshold": request.z_threshold,
                "minimum_baseline_points": request.minimum_baseline_points,
                "signal_mapping": request.signal_mapping,
                "operating_condition": request.operating_condition,
                "request_id": request.request_id,
            }
            # Preserve historical generic-detector run IDs while isolating custom
            # detector state from the generic pipeline.
            if self.detector_id != DETECTOR_ID:
                payload["detector_id"] = self.detector_id
            request_hash = sha256_json(payload)
            run_id = f"DRUN-{request_hash[:24].upper()}"
            window_id = _stable_id(
                "WIN",
                {
                    "source_type": source_metadata.source_type.value,
                    "entity_id": request.entity_id,
                    "start": request.start,
                    "end": request.end,
                    "channels": request.channels,
                },
            )
            return run_id, window_id, request_hash
        finally:
            if source_opened_here:
                self.source.close()

    def inspect(self, run_id: str) -> dict[str, Any] | None:
        state = self.state_store.get(run_id)
        return None if state is None else state.model_dump(mode="json")

    def run(
        self,
        request: DatasetDiagnosisRequest,
        *,
        output_dir: str | Path | None = None,
        force: bool = False,
    ) -> LayeredDatasetDiagnosis:
        run_id, window_id, request_hash = self.identity(request)
        if force:
            self.state_store.delete(run_id)
        existing = self.state_store.get(run_id)
        if existing and existing.request_hash != request_hash:
            raise ValueError("diagnosis run ID conflicts with a different request")
        if existing and existing.status == "COMPLETED" and existing.result:
            replay = LayeredDatasetDiagnosis.model_validate(existing.result).model_copy(
                update={"replayed": True}
            )
            if output_dir is not None:
                self._write_outputs(replay, existing, output_dir)
            return replay
        state = existing or DiagnosisRunState(
            run_id=run_id,
            request_hash=request_hash,
            status="CREATED",
            window_id=window_id,
            request=request.model_dump(mode="json"),
        )
        saved_detector_id = str(
            (state.stage_outputs.get("RUN_CONFIG", {}) or {}).get("detector_id", "")
        )
        if saved_detector_id and saved_detector_id != self.detector_id:
            raise ValueError(
                f"diagnosis run was created with detector {saved_detector_id}, "
                f"not {self.detector_id}"
            )
        state.stage_outputs["RUN_CONFIG"] = {
            "detector_id": self.detector_id,
            "default_generic_detector": self.detector_id == DETECTOR_ID,
        }
        state.status = "RUNNING"
        state.error = ""
        state.failed_stage = ""
        self.state_store.put(state)

        opened_here = not self.source.opened
        try:
            if opened_here:
                self.source.open()
            entities = {item.entity_id: item for item in self.source.list_entities()}
            if request.entity_id not in entities:
                raise KeyError(f"unknown dataset entity: {request.entity_id}")
            entity = entities[request.entity_id]
            channels = request.channels or self.source.list_channels(request.entity_id)
            frame = self.source.read_range(
                request.entity_id,
                request.start,
                request.end,
                channels,
            )
            data_hash = _frame_hash(frame)
            if state.data_hash and state.data_hash != data_hash:
                raise RuntimeError("source data changed since the saved diagnosis state")
            state.data_hash = data_hash
            state.stage_outputs["DATA_READ"] = {
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "channels": channels,
                "timestamp_column": entity.timestamp_column,
                "data_hash": data_hash,
            }
            if "DATA_READ" not in state.completed_stages:
                state.completed_stages.append("DATA_READ")
            self.state_store.put(state)

            if "DETECTED" in state.completed_stages:
                detected_payload = state.stage_outputs["DETECTED"]
                observations = [
                    ObservationFact.model_validate(item)
                    for item in detected_payload.get("observations", [])
                ]
                detections = [
                    DetectionRecord.model_validate(item)
                    for item in detected_payload.get("detections", [])
                ]
                anomalies = [
                    AnomalyEvidence.model_validate(item)
                    for item in detected_payload.get("anomalies", [])
                ]
            else:
                observations, detections, anomalies = self.detector(
                    frame,
                    timestamp_column=entity.timestamp_column,
                    channels=channels,
                    baseline_fraction=request.baseline_fraction,
                    z_threshold=request.z_threshold,
                    minimum_baseline_points=request.minimum_baseline_points,
                    signal_mapping=request.signal_mapping,
                )
                state.stage_outputs["DETECTED"] = {
                    "observations": [item.model_dump(mode="json") for item in observations],
                    "detections": [item.model_dump(mode="json") for item in detections],
                    "anomalies": [item.model_dump(mode="json") for item in anomalies],
                    "detector_id": self.detector_id,
                }
                state.completed_stages.append("DETECTED")
                self.state_store.put(state)

            if "DIAGNOSED" in state.completed_stages:
                workflow_state = state.stage_outputs["DIAGNOSED"]
            else:
                workflow_request = WorkflowInput(
                    case_id=request.entity_id,
                    request_id=request.request_id,
                    anomalies=anomalies,
                    operating_condition=request.operating_condition,
                    metadata={
                        **request.metadata,
                        "dataset_source": self.source.get_metadata().model_dump(mode="json"),
                        "dataset_run_id": run_id,
                        "window_id": window_id,
                        "data_hash": data_hash,
                        "detector_id": self.detector_id,
                    },
                )
                workflow_state = invoke_workflow(
                    self.repository,
                    workflow_request,
                    policy=self.policy,
                    audit_store=self.audit_store,
                )
                # Persist only JSON-safe workflow fields.
                workflow_state = json.loads(json.dumps(workflow_state, ensure_ascii=False))
                state.stage_outputs["DIAGNOSED"] = workflow_state
                state.completed_stages.append("DIAGNOSED")
                self.state_store.put(state)

            diagnosis = dict(workflow_state["diagnosis"])
            candidates = list(diagnosis.get("candidates", []))
            evidence = list(workflow_state.get("retrieved_edges", []))
            hypotheses = [
                {
                    "rank": index + 1,
                    "fault_id": item.get("fault_id"),
                    "fault_name": item.get("fault_name"),
                    "score": item.get("score", 0.0),
                    "eligible_for_diagnosis": item.get("eligible_for_diagnosis", False),
                    "evidence_count": item.get("evidence_count", 0),
                    "critical_evidence_satisfied": item.get(
                        "critical_evidence_satisfied", True
                    ),
                }
                for index, item in enumerate(candidates)
            ]
            diagnosis_id = _stable_id(
                "DX",
                {
                    "window_id": window_id,
                    "data_hash": data_hash,
                    "policy_version": self.policy.policy_version,
                    "knowledge_version": self.repository.knowledge_version,
                    "anomalies": [item.model_dump(mode="json") for item in anomalies],
                    "diagnosis": diagnosis,
                },
            )
            result = LayeredDatasetDiagnosis(
                diagnosis_id=diagnosis_id,
                run_id=run_id,
                window_id=window_id,
                request_id=request.request_id,
                entity_id=request.entity_id,
                observations=observations,
                detections=detections,
                fault_candidates=candidates,
                evidence=evidence,
                hypotheses=hypotheses,
                conclusion=_conclusion_from_diagnosis(diagnosis),
                provenance={
                    "source": self.source.get_metadata().model_dump(mode="json"),
                    "data_hash": data_hash,
                    "detector_id": self.detector_id,
                    "policy_version": self.policy.policy_version,
                    "knowledge_version": self.repository.knowledge_version,
                    "core_diagnosis_status": diagnosis.get("status"),
                    "core_audit": workflow_state.get("audit_record", {}),
                    "raw_rows_persisted": False,
                    "labels_used_by_detector": False,
                    "truth_channels_used_by_detector": False,
                },
                report_markdown=str(workflow_state.get("report_markdown", "")),
            )
            state.result = result.model_dump(mode="json")
            if "COMPLETED" not in state.completed_stages:
                state.completed_stages.append("COMPLETED")
            state.status = "COMPLETED"
            self.state_store.put(state)
            if output_dir is not None:
                self._write_outputs(result, state, output_dir)
            return result
        except Exception as exc:
            stage = next(
                (
                    name
                    for name in ("DATA_READ", "DETECTED", "DIAGNOSED", "COMPLETED")
                    if name not in state.completed_stages
                ),
                "UNKNOWN",
            )
            state.status = "FAILED"
            state.failed_stage = stage
            state.error = f"{type(exc).__name__}: {exc}"
            self.state_store.put(state)
            raise
        finally:
            if opened_here:
                self.source.close()

    def resume(
        self,
        run_id: str,
        *,
        output_dir: str | Path | None = None,
    ) -> LayeredDatasetDiagnosis:
        state = self.state_store.get(run_id)
        if state is None:
            raise KeyError(f"diagnosis run state not found: {run_id}")
        request = DatasetDiagnosisRequest.model_validate(state.request)
        expected_run_id, _, _ = self.identity(request)
        if expected_run_id != run_id:
            raise ValueError("saved diagnosis state identity is inconsistent")
        return self.run(request, output_dir=output_dir)

    def _write_outputs(
        self,
        result: LayeredDatasetDiagnosis,
        state: DiagnosisRunState,
        output_dir: str | Path,
    ) -> None:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "diagnosis.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (destination / "report.md").write_text(
            result.report_markdown, encoding="utf-8"
        )
        (destination / "run_state.json").write_text(
            state.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
