from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .expanded_single_fault import ExpandedSingleFaultEngine
from .sat_sim_catalog import SatSimCatalogDiagnosisEngine
from .sat_sim_multifault import EFFECT_META as PAIRED_MULTIFault_EFFECTS
from .sat_sim_multifault import diagnose_paired_multifault
from .sat_sim_multifault_single import EFFECT_META as SINGLE_MULTIFault_EFFECTS
from .sat_sim_multifault_single import diagnose_single_trace_multifault
from .sat_sim_open_set import SatSimOpenSetDiagnosisEngine
from .sat_sim_paired import SIGNATURES, SatSimPairedDiagnosisEngine
from .telemetry_contract import normalize_telemetry_frame
from .trusted_single_fault import assess_trusted_single_fault
from .unified_models import (
    DataQualityFinding,
    DiagnosticEvidence,
    EngineCapability,
    EngineExecutionStatus,
    EngineRunRecord,
    UnifiedBackend,
    UnifiedBatchManifest,
    UnifiedBatchResult,
    UnifiedCandidate,
    UnifiedDiagnosisRequest,
    UnifiedDiagnosisResult,
    UnifiedDiagnosisStatus,
    UnifiedInputMode,
)


EXPANDED_SINGLE_FAULT_ENGINE_ID = "expanded_paired_single_fault"
PAIRED_SIGNATURE_ENGINE_ID = "paired_physical_signatures"
PAIRED_MULTIFAULT_ENGINE_ID = "paired_multifault_signatures"
SINGLE_MULTIFAULT_ENGINE_ID = "single_trace_multifault_signatures"
CATALOG_ENGINE_ID = "single_trace_catalog_model"
OPEN_SET_ENGINE_ID = "single_trace_open_set_model"


def engine_capabilities() -> list[EngineCapability]:
    paired_ids = sorted({item.effect_id for item in SIGNATURES})
    expanded_modes = ExpandedSingleFaultEngine().known_modes
    return [
        EngineCapability(
            engine_id=EXPANDED_SINGLE_FAULT_ENGINE_ID,
            input_modes=[UnifiedInputMode.PAIRED],
            backends=["unified", "composite"],
            requires_nominal=True,
            known_effect_ids=expanded_modes,
            claim_boundary=(
                "Development-split paired residual prototypes with frozen observational-equivalence "
                "classes; exact mode claims are made only when available telemetry is identifiable."
            ),
        ),
        EngineCapability(
            engine_id=PAIRED_SIGNATURE_ENGINE_ID,
            input_modes=[UnifiedInputMode.PAIRED],
            backends=["unified", "composite"],
            requires_nominal=True,
            known_effect_ids=paired_ids,
            claim_boundary="Deterministic paired trend signatures; scores are evidence coverage, not probability.",
        ),
        EngineCapability(
            engine_id=PAIRED_MULTIFAULT_ENGINE_ID,
            input_modes=[UnifiedInputMode.PAIRED],
            backends=["composite"],
            requires_nominal=True,
            known_effect_ids=sorted(PAIRED_MULTIFault_EFFECTS),
            claim_boundary="Known-catalog paired physical signatures; no open-set or causal claim.",
        ),
        EngineCapability(
            engine_id=SINGLE_MULTIFAULT_ENGINE_ID,
            input_modes=[UnifiedInputMode.SINGLE_TRACE],
            backends=["composite"],
            requires_nominal=False,
            known_effect_ids=sorted(SINGLE_MULTIFault_EFFECTS),
            claim_boundary="Single-trace laboratory threshold screening; not calibrated probability.",
        ),
        EngineCapability(
            engine_id=CATALOG_ENGINE_ID,
            input_modes=[UnifiedInputMode.SINGLE_TRACE, UnifiedInputMode.PAIRED],
            backends=["unified", "composite"],
            requires_nominal=False,
            requires_model=True,
            claim_boundary="Known-catalog model vote; model vote share is not calibrated probability.",
        ),
        EngineCapability(
            engine_id=OPEN_SET_ENGINE_ID,
            input_modes=[UnifiedInputMode.SINGLE_TRACE, UnifiedInputMode.PAIRED],
            backends=["unified", "composite"],
            requires_nominal=False,
            requires_model=True,
            claim_boundary="Calibrated catalog-membership screening for the model's declared distribution.",
        ),
    ]


def _skipped(engine_id: str, *reason_codes: str) -> EngineRunRecord:
    return EngineRunRecord(
        engine_id=engine_id,
        eligible=False,
        execution_status=EngineExecutionStatus.SKIPPED,
        status="NOT_APPLICABLE",
        reason_codes=list(reason_codes),
    )


def _failed(engine_id: str, exc: Exception) -> EngineRunRecord:
    return EngineRunRecord(
        engine_id=engine_id,
        eligible=True,
        execution_status=EngineExecutionStatus.FAILED,
        status="ENGINE_ERROR",
        reason_codes=[f"{exc.__class__.__name__}:{str(exc)[:240]}"],
    )


def _write_temp_csv(frame: pd.DataFrame, root: Path, name: str) -> Path:
    path = root / name
    frame.to_csv(path, index=False)
    return path


def _expanded_single_fault_record(nominal: pd.DataFrame, effect: pd.DataFrame) -> EngineRunRecord:
    try:
        result = ExpandedSingleFaultEngine().diagnose_frames(nominal=nominal, effect=effect)
    except Exception as exc:
        return _failed(EXPANDED_SINGLE_FAULT_ENGINE_ID, exc)
    detected = result.candidate_modes if result.status == "OBSERVATIONALLY_EQUIVALENT" else (
        [result.primary_mode] if result.primary_mode else []
    )
    return EngineRunRecord(
        engine_id=EXPANDED_SINGLE_FAULT_ENGINE_ID,
        eligible=True,
        execution_status=EngineExecutionStatus.EXECUTED,
        status=result.status,
        reason_codes=result.reason_codes,
        detected_effect_ids=detected,
        primary_effect_id=result.primary_mode,
        score=result.screening_score,
        raw_result=result.to_dict(),
    )


def _paired_signature_record(case_id: str, backend: str, nominal: pd.DataFrame, effect: pd.DataFrame) -> EngineRunRecord:
    try:
        result = SatSimPairedDiagnosisEngine().diagnose_frames(
            case_id=case_id,
            backend=backend,
            nominal=nominal,
            effect=effect,
        )
    except Exception as exc:  # fail one engine without hiding the other engine's result
        return _failed(PAIRED_SIGNATURE_ENGINE_ID, exc)
    detected = [result.conclusion_effect_id] if result.conclusion_effect_id else []
    return EngineRunRecord(
        engine_id=PAIRED_SIGNATURE_ENGINE_ID,
        eligible=True,
        execution_status=EngineExecutionStatus.EXECUTED,
        status=result.status.value,
        reason_codes=[result.resolution_basis],
        detected_effect_ids=detected,
        primary_effect_id=result.conclusion_effect_id,
        score=result.top_score,
        raw_result=result.model_dump(mode="json"),
    )


def _paired_multifault_record(case_id: str, nominal: pd.DataFrame, effect: pd.DataFrame, *, catalog_path: Path | None) -> EngineRunRecord:
    try:
        with tempfile.TemporaryDirectory(prefix="astrograph-unified-") as tmp:
            root = Path(tmp)
            result = diagnose_paired_multifault(
                case_id=case_id,
                nominal_csv=_write_temp_csv(nominal, root, "nominal.csv"),
                effect_csv=_write_temp_csv(effect, root, "effect.csv"),
                catalog_path=catalog_path,
            )
    except Exception as exc:
        return _failed(PAIRED_MULTIFAULT_ENGINE_ID, exc)
    return EngineRunRecord(
        engine_id=PAIRED_MULTIFAULT_ENGINE_ID,
        eligible=True,
        execution_status=EngineExecutionStatus.EXECUTED,
        status=result.status.value,
        reason_codes=[],
        detected_effect_ids=result.detected_effect_ids,
        primary_effect_id=result.primary_effect_id,
        score=None,
        raw_result=result.model_dump(mode="json"),
    )


def _single_multifault_record(case_id: str, effect: pd.DataFrame, *, catalog_path: Path | None) -> EngineRunRecord:
    try:
        with tempfile.TemporaryDirectory(prefix="astrograph-unified-") as tmp:
            root = Path(tmp)
            result = diagnose_single_trace_multifault(
                case_id=case_id,
                telemetry_csv=_write_temp_csv(effect, root, "telemetry.csv"),
                catalog_path=catalog_path,
            )
    except Exception as exc:
        return _failed(SINGLE_MULTIFAULT_ENGINE_ID, exc)
    return EngineRunRecord(
        engine_id=SINGLE_MULTIFAULT_ENGINE_ID,
        eligible=True,
        execution_status=EngineExecutionStatus.EXECUTED,
        status=result.status.value,
        reason_codes=[],
        detected_effect_ids=result.detected_effect_ids,
        primary_effect_id=result.primary_effect_id,
        score=(result.effect_scores.get(result.primary_effect_id) if result.primary_effect_id else None),
        raw_result=result.model_dump(mode="json"),
    )


def _catalog_record(case_id: str, backend: str, effect: pd.DataFrame, *, model_path: Path | None) -> EngineRunRecord:
    if model_path is None:
        return _skipped(CATALOG_ENGINE_ID, "MODEL_NOT_EXPLICITLY_PROVIDED")
    if not model_path.exists():
        return _skipped(CATALOG_ENGINE_ID, "MODEL_PATH_NOT_FOUND")
    try:
        result = SatSimCatalogDiagnosisEngine.from_path(model_path).diagnose_frame(
            case_id=case_id, backend=backend, telemetry=effect
        )
    except Exception as exc:
        return _failed(CATALOG_ENGINE_ID, exc)
    detected = [result.conclusion_effect_id] if result.conclusion_effect_id else []
    return EngineRunRecord(
        engine_id=CATALOG_ENGINE_ID,
        eligible=True,
        execution_status=EngineExecutionStatus.EXECUTED,
        status=result.status,
        reason_codes=[result.resolution_basis],
        detected_effect_ids=detected,
        primary_effect_id=result.conclusion_effect_id,
        score=result.model_vote_share,
        raw_result=result.model_dump(mode="json"),
    )


def _open_set_record(case_id: str, backend: str, effect: pd.DataFrame, *, model_path: Path | None) -> EngineRunRecord:
    if model_path is None:
        return _skipped(OPEN_SET_ENGINE_ID, "MODEL_NOT_EXPLICITLY_PROVIDED")
    if not model_path.exists():
        return _skipped(OPEN_SET_ENGINE_ID, "MODEL_PATH_NOT_FOUND")
    try:
        result = SatSimOpenSetDiagnosisEngine.from_path(model_path).diagnose_frame(
            case_id=case_id, backend=backend, telemetry=effect
        )
    except Exception as exc:
        return _failed(OPEN_SET_ENGINE_ID, exc)
    detected = [result.conclusion_effect_id] if result.conclusion_effect_id else []
    return EngineRunRecord(
        engine_id=OPEN_SET_ENGINE_ID,
        eligible=True,
        execution_status=EngineExecutionStatus.EXECUTED,
        status=result.status,
        reason_codes=result.uncertainty_reason_codes + [result.resolution_basis],
        detected_effect_ids=detected,
        primary_effect_id=result.conclusion_effect_id,
        score=result.diagnostic_confidence,
        raw_result=result.model_dump(mode="json"),
    )


def _candidate_map(records: Iterable[EngineRunRecord]) -> list[UnifiedCandidate]:
    supports: dict[str, set[str]] = defaultdict(set)
    evidence: dict[str, list[str]] = defaultdict(list)
    scores: dict[str, float] = defaultdict(float)
    for record in records:
        for effect_id in record.detected_effect_ids:
            supports[effect_id].add(record.engine_id)
            evidence[effect_id].append(f"{record.engine_id}:{record.status}")
            if record.score is not None:
                scores[effect_id] = max(scores[effect_id], min(1.0, max(0.0, record.score)))
        raw = record.raw_result or {}
        if record.engine_id == PAIRED_SIGNATURE_ENGINE_ID:
            for candidate in raw.get("candidates", []):
                effect_id = candidate.get("effect_id")
                score = candidate.get("score")
                if effect_id and isinstance(score, (int, float)) and score >= 0.55:
                    supports[effect_id].add(record.engine_id)
                    scores[effect_id] = max(scores[effect_id], float(score))
                    evidence[effect_id].append(
                        f"{record.engine_id}:screening_score={float(score):.6f};eligible={bool(candidate.get('eligible'))}"
                    )
    return sorted(
        [
            UnifiedCandidate(
                effect_id=effect_id,
                support_engines=sorted(engine_ids),
                screening_score=scores[effect_id],
                evidence_summary=evidence[effect_id],
            )
            for effect_id, engine_ids in supports.items()
        ],
        key=lambda item: (len(item.support_engines), item.screening_score, item.effect_id),
        reverse=True,
    )


def _fuse_paired(records: list[EngineRunRecord]) -> tuple[UnifiedDiagnosisStatus, list[str], str]:
    by_id = {record.engine_id: record for record in records}
    expanded = by_id.get(EXPANDED_SINGLE_FAULT_ENGINE_ID)
    if expanded is not None and expanded.execution_status == EngineExecutionStatus.EXECUTED:
        if expanded.status == "DIAGNOSED" and expanded.primary_effect_id:
            return UnifiedDiagnosisStatus.DIAGNOSED, [expanded.primary_effect_id], "EXPANDED_SINGLE_FAULT_IDENTIFIABLE"
        if expanded.status == "OBSERVATIONALLY_EQUIVALENT" and expanded.detected_effect_ids:
            return (
                UnifiedDiagnosisStatus.AMBIGUOUS,
                list(expanded.detected_effect_ids),
                "OBSERVATIONALLY_EQUIVALENT_TELEMETRY",
            )
        if expanded.status == "UNKNOWN":
            return UnifiedDiagnosisStatus.UNKNOWN_FAULT, [], "EXPANDED_SINGLE_FAULT_OUT_OF_ENVELOPE"
    by_id = {record.engine_id: record for record in records}
    signature = by_id.get(PAIRED_SIGNATURE_ENGINE_ID)
    multifault = by_id.get(PAIRED_MULTIFAULT_ENGINE_ID)
    if signature is None or signature.execution_status == EngineExecutionStatus.FAILED:
        if multifault and multifault.execution_status == EngineExecutionStatus.EXECUTED:
            if len(multifault.detected_effect_ids) == 1:
                return UnifiedDiagnosisStatus.DIAGNOSED, multifault.detected_effect_ids, "PAIRED_MULTIFAULT_ONLY"
            if multifault.detected_effect_ids:
                return UnifiedDiagnosisStatus.AMBIGUOUS, [], "MULTIPLE_UNCORROBORATED_EFFECTS"
        return UnifiedDiagnosisStatus.DATA_ERROR, [], "NO_PAIRED_ENGINE_RESULT"
    if signature.status == "DATA_ERROR":
        return UnifiedDiagnosisStatus.DATA_ERROR, [], "PAIRED_SIGNATURE_DATA_ERROR"
    if signature.status == "DIAGNOSED" and signature.primary_effect_id:
        if multifault and len(multifault.detected_effect_ids) == 1:
            if multifault.detected_effect_ids[0] != signature.primary_effect_id:
                return UnifiedDiagnosisStatus.CONFLICTING_EVIDENCE, [], "PAIR_ENGINES_DISAGREE"
            return UnifiedDiagnosisStatus.DIAGNOSED, [signature.primary_effect_id], "CROSS_ENGINE_CORROBORATION"
        if multifault and len(multifault.detected_effect_ids) > 1:
            return UnifiedDiagnosisStatus.CONFLICTING_EVIDENCE, [], "MULTIPLE_EFFECTS_CONTRADICT_SINGLE_SIGNATURE"
        return UnifiedDiagnosisStatus.DIAGNOSED, [signature.primary_effect_id], "PAIRED_SIGNATURE_RESOLUTION"
    if multifault and multifault.execution_status == EngineExecutionStatus.EXECUTED:
        detected = multifault.detected_effect_ids
        if len(detected) == 1:
            effect_id = detected[0]
            raw_candidates = (signature.raw_result or {}).get("candidates", [])
            corroborated = any(
                item.get("effect_id") == effect_id and float(item.get("score", 0.0)) >= 0.55
                for item in raw_candidates
            )
            if corroborated:
                return UnifiedDiagnosisStatus.DIAGNOSED, [effect_id], "CROSS_ENGINE_CORROBORATION"
            return UnifiedDiagnosisStatus.CONFLICTING_EVIDENCE, [], "MULTIFAULT_NOT_CORROBORATED"
        if len(detected) > 1:
            return UnifiedDiagnosisStatus.CONFLICTING_EVIDENCE, [], "MULTIPLE_EFFECTS_WITH_AMBIGUOUS_PAIR_SIGNATURE"
    if signature.status == "UNKNOWN":
        return UnifiedDiagnosisStatus.UNKNOWN_FAULT, [], "NO_ELIGIBLE_KNOWN_SIGNATURE"
    return UnifiedDiagnosisStatus.AMBIGUOUS, [], "PARTIAL_PAIRED_EVIDENCE"


def _fuse_single(records: list[EngineRunRecord]) -> tuple[UnifiedDiagnosisStatus, list[str], str]:
    executed = [record for record in records if record.execution_status == EngineExecutionStatus.EXECUTED]
    if not executed:
        return UnifiedDiagnosisStatus.INSUFFICIENT_DATA, [], "NO_ELIGIBLE_SINGLE_TRACE_ENGINE"
    open_set = next((record for record in executed if record.engine_id == OPEN_SET_ENGINE_ID), None)
    catalog = next((record for record in executed if record.engine_id == CATALOG_ENGINE_ID), None)
    physical = next((record for record in executed if record.engine_id == SINGLE_MULTIFAULT_ENGINE_ID), None)
    if open_set:
        if open_set.status == "UNKNOWN_FAULT":
            return UnifiedDiagnosisStatus.UNKNOWN_FAULT, [], "OPEN_SET_REJECTION"
        if open_set.status in {"INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"}:
            return UnifiedDiagnosisStatus.INSUFFICIENT_DATA, [], "OPEN_SET_INSUFFICIENT_OR_OUT_OF_SCOPE"
    if (
        catalog is not None
        and open_set is not None
        and catalog.status == "NORMAL"
        and open_set.status == "NORMAL"
    ):
        return UnifiedDiagnosisStatus.NORMAL, [], "CATALOG_AND_OPEN_SET_NORMAL_AGREEMENT"
    model_conclusions = [
        record.primary_effect_id for record in (catalog, open_set)
        if record is not None and record.primary_effect_id
    ]
    unique_models = sorted(set(model_conclusions))
    if len(unique_models) > 1:
        return UnifiedDiagnosisStatus.CONFLICTING_EVIDENCE, [], "SINGLE_TRACE_MODELS_DISAGREE"
    if len(unique_models) == 1:
        model_effect = unique_models[0]
        if physical and physical.detected_effect_ids and model_effect not in physical.detected_effect_ids:
            return UnifiedDiagnosisStatus.CONFLICTING_EVIDENCE, [], "MODEL_AND_PHYSICAL_SCREENING_DISAGREE"
        return UnifiedDiagnosisStatus.DIAGNOSED, [model_effect], "EXPLICIT_MODEL_WITH_PHYSICAL_COMPATIBILITY"
    if physical:
        if physical.status == "NORMAL":
            return UnifiedDiagnosisStatus.NORMAL, [], "PHYSICAL_SIGNATURE_NORMAL"
        if physical.status == "INSUFFICIENT_EVIDENCE":
            return UnifiedDiagnosisStatus.INSUFFICIENT_DATA, [], "INSUFFICIENT_PHYSICAL_EVIDENCE"
        if physical.detected_effect_ids:
            # One threshold engine alone is screening evidence, not enough for a formal
            # single-trace diagnosis. This prevents a lone false positive from being
            # promoted to DIAGNOSED merely because no other engine was configured.
            return UnifiedDiagnosisStatus.AMBIGUOUS, [], "SINGLE_TRACE_SCREENING_REQUIRES_CORROBORATION"
    if any(record.status == "DATA_ERROR" for record in (catalog, open_set) if record is not None):
        return UnifiedDiagnosisStatus.INSUFFICIENT_DATA, [], "MODEL_INPUT_CONTRACT_NOT_MET"
    return UnifiedDiagnosisStatus.AMBIGUOUS, [], "NO_UNIQUE_SINGLE_TRACE_CONCLUSION"


class UnifiedDiagnosisService:
    """One production orchestration entry over existing deterministic engines.

    This class normalizes inputs and exposes every routing/engine decision. It does
    not silently retrain models or modify the thresholds of the underlying engines.
    """

    def capabilities(self) -> list[EngineCapability]:
        return engine_capabilities()

    def diagnose(self, request: UnifiedDiagnosisRequest) -> UnifiedDiagnosisResult:
        input_mode = UnifiedInputMode.PAIRED if request.nominal_path is not None else UnifiedInputMode.SINGLE_TRACE
        routing: list[str] = [f"INPUT_MODE={input_mode.value}"]
        profiles = []
        try:
            effect_raw = pd.read_csv(request.telemetry_path, low_memory=False)
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            return UnifiedDiagnosisResult(
                case_id=request.case_id,
                input_mode=input_mode,
                backend=request.backend.value,
                status=UnifiedDiagnosisStatus.DATA_ERROR,
                resolution_basis=f"TELEMETRY_READ_ERROR:{exc.__class__.__name__}",
                data_quality=[],
                engine_runs=[],
                routing_decisions=routing,
            )
        effect_norm = normalize_telemetry_frame(effect_raw, requested_backend=request.backend.value)
        profiles.append(effect_norm.profile)
        backend = effect_norm.profile.backend
        nominal_frame: pd.DataFrame | None = None
        if input_mode == UnifiedInputMode.PAIRED:
            try:
                nominal_raw = pd.read_csv(request.nominal_path, low_memory=False)  # type: ignore[arg-type]
            except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
                return UnifiedDiagnosisResult(
                    case_id=request.case_id,
                    input_mode=input_mode,
                    backend=backend,
                    status=UnifiedDiagnosisStatus.DATA_ERROR,
                    resolution_basis=f"NOMINAL_READ_ERROR:{exc.__class__.__name__}",
                    data_quality=profiles,
                    engine_runs=[],
                    routing_decisions=routing,
                )
            nominal_norm = normalize_telemetry_frame(nominal_raw, requested_backend=backend)
            profiles.insert(0, nominal_norm.profile)
            nominal_frame = nominal_norm.frame
            if list(nominal_frame.columns) != list(effect_norm.frame.columns):
                profiles[-1].findings.append(DataQualityFinding(
                    code="PAIRED_COLUMN_MISMATCH", severity="ERROR",
                    message="nominal and effect columns differ"
                ))
                profiles[-1].status = "FAIL"
            if len(nominal_frame) != len(effect_norm.frame):
                profiles[-1].findings.append(DataQualityFinding(
                    code="PAIRED_ROW_COUNT_MISMATCH", severity="ERROR",
                    message="nominal and effect row counts differ"
                ))
                profiles[-1].status = "FAIL"
        if any(profile.status == "FAIL" for profile in profiles):
            return UnifiedDiagnosisResult(
                case_id=request.case_id,
                input_mode=input_mode,
                backend=backend,
                status=UnifiedDiagnosisStatus.DATA_ERROR,
                resolution_basis="DATA_QUALITY_GATE_FAILED",
                data_quality=profiles,
                engine_runs=[],
                routing_decisions=routing + ["ENGINES_NOT_EXECUTED_DUE_TO_DATA_GATE"],
            )
        routing.append(f"BACKEND={backend}")
        records: list[EngineRunRecord] = []
        if input_mode == UnifiedInputMode.PAIRED:
            if nominal_frame is None:
                raise RuntimeError("paired diagnosis reached execution without a nominal frame")
            records.append(_expanded_single_fault_record(nominal_frame, effect_norm.frame))
            records.append(_paired_signature_record(request.case_id, backend, nominal_frame, effect_norm.frame))
            if backend == "composite":
                records.append(_paired_multifault_record(
                    request.case_id, nominal_frame, effect_norm.frame, catalog_path=request.catalog_path
                ))
            else:
                records.append(_skipped(PAIRED_MULTIFAULT_ENGINE_ID, "BACKEND_NOT_SUPPORTED"))
            records.append(_catalog_record(
                request.case_id, backend, effect_norm.frame, model_path=request.catalog_model_path
            ))
            records.append(_open_set_record(
                request.case_id, backend, effect_norm.frame, model_path=request.open_set_model_path
            ))
            status, conclusions, basis = _fuse_paired(records)
        else:
            if backend == "composite":
                records.append(_single_multifault_record(
                    request.case_id, effect_norm.frame, catalog_path=request.catalog_path
                ))
            else:
                records.append(_skipped(SINGLE_MULTIFAULT_ENGINE_ID, "BACKEND_NOT_SUPPORTED"))
            records.append(_catalog_record(
                request.case_id, backend, effect_norm.frame, model_path=request.catalog_model_path
            ))
            records.append(_open_set_record(
                request.case_id, backend, effect_norm.frame, model_path=request.open_set_model_path
            ))
            status, conclusions, basis = _fuse_single(records)
        for record in records:
            routing.append(
                f"{record.engine_id}:{record.execution_status.value}:{record.status}:" + ",".join(record.reason_codes)
            )
        candidates = _candidate_map(records)
        enrichment: dict[str, object] = {}
        expanded_record = next((item for item in records if item.engine_id == EXPANDED_SINGLE_FAULT_ENGINE_ID), None)
        if expanded_record is not None and expanded_record.execution_status == EngineExecutionStatus.EXECUTED:
            expanded_raw = expanded_record.raw_result or {}
            if basis in {"EXPANDED_SINGLE_FAULT_IDENTIFIABLE", "OBSERVATIONALLY_EQUIVALENT_TELEMETRY"}:
                enrichment = {
                    "primary_subsystem": expanded_raw.get("subsystem"),
                    "primary_component": expanded_raw.get("component"),
                    "diagnostic_evidence": [
                        DiagnosticEvidence.model_validate(item) for item in expanded_raw.get("evidence", [])
                    ],
                }
        if (
            status == UnifiedDiagnosisStatus.DIAGNOSED
            and len(conclusions) == 1
            and nominal_frame is not None
        ):
            trusted = assess_trusted_single_fault(
                effect_id=conclusions[0], nominal=nominal_frame, effect=effect_norm.frame
            )
            if trusted:
                existing_evidence = list(enrichment.get("diagnostic_evidence", []))
                trusted_evidence = list(trusted.pop("diagnostic_evidence", []))
                enrichment.update({key: value for key, value in trusted.items() if value is not None})
                enrichment["diagnostic_evidence"] = existing_evidence + trusted_evidence
        return UnifiedDiagnosisResult(
            case_id=request.case_id,
            input_mode=input_mode,
            backend=backend,
            status=status,
            conclusion_effect_ids=conclusions,
            primary_effect_id=conclusions[0] if len(conclusions) == 1 else None,
            resolution_basis=basis,
            data_quality=profiles,
            engine_runs=records,
            candidates=candidates,
            routing_decisions=routing,
            **enrichment,
        )

    def diagnose_paths(
        self,
        *,
        case_id: str,
        telemetry_path: str | Path,
        nominal_path: str | Path | None = None,
        backend: str = "auto",
        catalog_path: str | Path | None = None,
        catalog_model_path: str | Path | None = None,
        open_set_model_path: str | Path | None = None,
    ) -> UnifiedDiagnosisResult:
        return self.diagnose(UnifiedDiagnosisRequest(
            case_id=case_id,
            telemetry_path=Path(telemetry_path),
            nominal_path=Path(nominal_path) if nominal_path is not None else None,
            backend=UnifiedBackend(backend),
            catalog_path=Path(catalog_path) if catalog_path is not None else None,
            catalog_model_path=Path(catalog_model_path) if catalog_model_path is not None else None,
            open_set_model_path=Path(open_set_model_path) if open_set_model_path is not None else None,
        ))

    def diagnose_batch(self, manifest: UnifiedBatchManifest) -> UnifiedBatchResult:
        results = [
            self.diagnose_paths(
                case_id=item.case_id,
                telemetry_path=item.telemetry,
                nominal_path=item.nominal,
                backend=item.backend.value,
            )
            for item in manifest.cases
        ]
        return UnifiedBatchResult(
            results=results,
            status_counts=dict(sorted(Counter(item.status.value for item in results).items())),
        )

    def diagnose_batch_path(self, manifest_path: str | Path) -> UnifiedBatchResult:
        path = Path(manifest_path)
        manifest = UnifiedBatchManifest.model_validate_json(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        resolved_cases = []
        for item in manifest.cases:
            telemetry = item.telemetry if item.telemetry.is_absolute() else base / item.telemetry
            nominal = item.nominal
            if nominal is not None and not nominal.is_absolute():
                nominal = base / nominal
            resolved_cases.append(item.model_copy(update={"telemetry": telemetry, "nominal": nominal}))
        return self.diagnose_batch(manifest.model_copy(update={"cases": resolved_cases}))


__all__ = [
    "EXPANDED_SINGLE_FAULT_ENGINE_ID",
    "PAIRED_SIGNATURE_ENGINE_ID",
    "PAIRED_MULTIFAULT_ENGINE_ID",
    "SINGLE_MULTIFAULT_ENGINE_ID",
    "CATALOG_ENGINE_ID",
    "OPEN_SET_ENGINE_ID",
    "engine_capabilities",
    "UnifiedDiagnosisService",
]
