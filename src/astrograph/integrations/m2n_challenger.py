from __future__ import annotations

"""Offline-only M2N temporal challenger with four-role split discipline."""

from .process_control import run_managed_subprocess
import csv
import json
import math
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence.executable import stable_id

from .dor_runtime import sha256_file
from .external_runtime import PythonCLIRuntimeIdentity, inspect_python_cli_runtime


class M2NChallengerStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class M2NChallengerAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-m2n-challenger-audit-request.v1"
    spec_path: str = Field(min_length=1)
    expected_spec_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    data_path: str = Field(min_length=1)
    expected_data_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    output_dir: str = Field(min_length=1)
    group_column: str = Field(min_length=1)
    split_column: str = Field(min_length=1)
    label_column: str = Field(min_length=1)
    train_value: str = "train"
    validation_value: str = "validation"
    calibration_value: str = "calibration"
    test_value: str = "test"
    calibration_alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    minimum_optimizer_seeds: int = Field(default=3, ge=3)
    executable_path: str = Field(default="m2n", min_length=1)
    timeout_s: float = Field(default=300.0, gt=0.0, le=1800.0)


class M2NChallengerAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-m2n-challenger-audit.v1"
    audit_id: str = Field(min_length=1)
    status: M2NChallengerStatus
    reason: str = Field(min_length=1)
    promotion_allowed: Literal[False] = False
    candidate_bound_to_production: Literal[False] = False
    spec_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    data_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    split_group_counts: dict[str, int] = Field(default_factory=dict)
    optimizer_seeds: list[int] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    simple_baseline_id: str | None = None
    selected_candidate_id: str | None = None
    selection_status: str | None = None
    calibration_alpha: float | None = None
    calibration_nonconformity_threshold: float | None = None
    calibration_context_count: int = 0
    test_context_count: int = 0
    test_argmax_accuracy: float | None = None
    test_macro_f1: float | None = None
    test_prediction_set_coverage: float | None = None
    test_mean_prediction_set_size: float | None = None
    output_manifest_sha256: dict[str, str] = Field(default_factory=dict)
    runtime: PythonCLIRuntimeIdentity | None = None
    design_result: dict[str, Any] | None = None
    claim_boundary: str = (
        "Offline challenger evidence only. Validation selects architecture, calibration sets the "
        "conformal threshold, and test is evaluation-only. No production model is replaced."
    )


class M2NChallengerAuditService:
    bridge_version = "astrograph-m2n-challenger-bridge@1.0.0"

    @staticmethod
    def _bound_file(path_text: str, expected_sha: str, name: str) -> tuple[Path, str]:
        path = Path(path_text).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"M2N challenger {name} not found: {path}")
        actual = sha256_file(path)
        if actual != expected_sha.lower():
            raise ValueError(f"M2N challenger {name} SHA-256 mismatch")
        return path, actual

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("M2N challenger data is empty")
        return rows

    @staticmethod
    def _validate_split(
        rows: list[dict[str, str]],
        request: M2NChallengerAuditRequest,
    ) -> tuple[dict[str, int], dict[str, tuple[str, str]]]:
        required_columns = {request.group_column, request.split_column, request.label_column}
        missing = required_columns - set(rows[0])
        if missing:
            raise ValueError(f"M2N challenger data lacks required columns: {sorted(missing)}")
        allowed = {
            request.train_value,
            request.validation_value,
            request.calibration_value,
            request.test_value,
        }
        groups: dict[str, tuple[str, str]] = {}
        for row in rows:
            group = str(row[request.group_column])
            split = str(row[request.split_column])
            label = str(row[request.label_column])
            if not group or split not in allowed or not label:
                raise ValueError(
                    "M2N challenger group, split, and label must be non-empty and declared"
                )
            prior = groups.get(group)
            if prior is not None and prior != (split, label):
                raise ValueError(
                    f"M2N challenger group {group!r} crosses split or label boundaries"
                )
            groups[group] = (split, label)
        counts = {value: 0 for value in allowed}
        for split, _ in groups.values():
            counts[split] += 1
        if any(counts[value] == 0 for value in allowed):
            raise ValueError(
                "M2N challenger requires non-empty train/validation/calibration/test groups"
            )
        train_labels = {label for split, label in groups.values() if split == request.train_value}
        evaluation_labels = {
            label
            for split, label in groups.values()
            if split in {request.validation_value, request.calibration_value, request.test_value}
        }
        if not evaluation_labels.issubset(train_labels):
            raise ValueError("M2N challenger evaluation roles contain labels unseen in train")
        return counts, groups

    @staticmethod
    def _validate_spec(
        spec: dict[str, Any],
        request: M2NChallengerAuditRequest,
        split_group_counts: dict[str, int],
    ) -> list[int]:
        contract = dict(spec.get("modeling_contract") or {})
        if str(contract.get("objective") or "").upper() != "CATEGORICAL_PREDICTION":
            raise ValueError("M2N challenger currently requires CATEGORICAL_PREDICTION")
        if str(contract.get("input_representation") or "").upper() not in {
            "MULTIVARIATE_SEQUENCE",
            "WAVEFORM_SEQUENCE",
        }:
            raise ValueError("M2N challenger requires a sequence representation")
        if contract.get("label_column") != request.label_column:
            raise ValueError("M2N challenger spec label_column mismatch")
        qualification = dict(spec.get("qualification") or {})
        if qualification.get("independent_context_column") != request.group_column:
            raise ValueError("M2N challenger spec independent_context_column mismatch")
        declared_minimums = {
            request.train_value: qualification.get("min_train_independent_contexts"),
            request.validation_value: qualification.get("min_validation_independent_contexts"),
        }
        if any(not isinstance(value, int) or value <= 0 for value in declared_minimums.values()):
            raise ValueError(
                "M2N challenger requires positive evidence-justified independent-context minimums"
            )
        if any(
            split_group_counts[role] < minimum
            for role, minimum in declared_minimums.items()
        ):
            raise ValueError("M2N challenger split does not satisfy declared context minimums")
        data = dict(spec.get("data") or {})
        if data.get("split_column") != request.split_column:
            raise ValueError("M2N challenger spec split_column mismatch")
        if list(map(str, data.get("train_values") or [])) != [request.train_value]:
            raise ValueError("M2N challenger M2N training role must be train only")
        if list(map(str, data.get("validation_values") or [])) != [request.validation_value]:
            raise ValueError("M2N challenger M2N selection role must be validation only")
        forbidden = {request.calibration_value, request.test_value}
        if forbidden & set(map(str, data.get("train_values") or [])):
            raise ValueError("M2N challenger calibration/test cannot enter training")
        if forbidden & set(map(str, data.get("validation_values") or [])):
            raise ValueError("M2N challenger calibration/test cannot enter model selection")
        training = dict((contract.get("constraints") or {}).get("temporal_training") or {})
        seeds = list(map(int, training.get("seeds") or []))
        if len(seeds) < request.minimum_optimizer_seeds or len(seeds) != len(set(seeds)):
            raise ValueError("M2N challenger requires distinct declared optimizer seeds")
        return seeds

    @staticmethod
    def _prediction_rows(path: Path, group_column: str) -> dict[str, dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        out = {str(row[group_column]): row for row in rows}
        if len(out) != len(rows):
            raise ValueError("M2N challenger prediction output contains duplicate contexts")
        return out

    @staticmethod
    def _macro_f1(labels: list[str], predictions: list[str], classes: list[str]) -> float:
        scores: list[float] = []
        for cls in classes:
            tp = sum(y == cls and p == cls for y, p in zip(labels, predictions, strict=True))
            fp = sum(y != cls and p == cls for y, p in zip(labels, predictions, strict=True))
            fn = sum(y == cls and p != cls for y, p in zip(labels, predictions, strict=True))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            scores.append(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        return sum(scores) / len(scores)

    def audit(self, request: M2NChallengerAuditRequest) -> M2NChallengerAudit:
        runtime: PythonCLIRuntimeIdentity | None = None
        spec_sha: str | None = None
        data_sha: str | None = None
        counts: dict[str, int] = {}
        seeds: list[int] = []
        try:
            spec_path, spec_sha = self._bound_file(
                request.spec_path, request.expected_spec_sha256, "spec"
            )
            data_path, data_sha = self._bound_file(
                request.data_path, request.expected_data_sha256, "data"
            )
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if not isinstance(spec, dict):
                raise ValueError("M2N challenger spec must contain a JSON object")
            rows = self._read_rows(data_path)
            counts, groups = self._validate_split(rows, request)
            seeds = self._validate_spec(spec, request, counts)
            output_dir = Path(request.output_dir).resolve()
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError("M2N challenger output_dir must be absent or empty")
            output_dir.mkdir(parents=True, exist_ok=True)
            runtime = inspect_python_cli_runtime(
                request.executable_path,
                distribution="m2n-toolkit",
                import_name="m2n",
                timeout_s=request.timeout_s,
            )
            completed = run_managed_subprocess(
                [
                    runtime.executable,
                    "temporal-design",
                    "--spec",
                    str(spec_path),
                    "--data",
                    str(data_path),
                    "--out",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "no M2N output").strip()[-2000:]
                raise RuntimeError(f"M2N temporal-design exited {completed.returncode}: {detail}")
            result_path = output_dir / "temporal_design_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            qualification = dict(result.get("qualification") or {})
            if qualification.get("status") != "READY":
                reasons = "; ".join(map(str, qualification.get("reasons") or []))
                raise ValueError(f"M2N challenger temporal qualification is not ready: {reasons}")
            risks = dict(result.get("candidate_validation_risks") or {})
            if "summary_mlp" not in risks:
                raise ValueError("M2N challenger result lacks the required summary_mlp baseline")
            if any(len(values) < request.minimum_optimizer_seeds for values in risks.values()):
                raise ValueError("M2N challenger result lacks required multi-seed risks")
            if any(
                not math.isfinite(float(value))
                for values in risks.values()
                for value in values
            ):
                raise ValueError("M2N challenger result contains non-finite validation risk")
            prediction_path = output_dir / "all-role-predictions.csv"
            completed = run_managed_subprocess(
                [
                    runtime.executable,
                    "predict-temporal",
                    "--design-dir",
                    str(output_dir),
                    "--data",
                    str(data_path),
                    "--out",
                    str(prediction_path),
                ],
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "no M2N output").strip()[-2000:]
                raise RuntimeError(f"M2N predict-temporal exited {completed.returncode}: {detail}")
            predictions = self._prediction_rows(prediction_path, request.group_column)
            classes = sorted(
                key.removeprefix("portfolio_score__")
                for key in next(iter(predictions.values()))
                if key.startswith("portfolio_score__")
            )
            if not classes:
                raise ValueError("M2N challenger predictions lack class scores")
            if set(predictions) != set(groups):
                raise ValueError(
                    "M2N challenger predictions do not exactly cover declared contexts"
                )
            for row in predictions.values():
                scores = [float(row[f"portfolio_score__{cls}"]) for cls in classes]
                if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
                    raise ValueError("M2N challenger predictions contain invalid class scores")

            def score(group: str, cls: str) -> float:
                return float(predictions[group][f"portfolio_score__{cls}"])

            calibration_groups = sorted(
                group
                for group, (split, _) in groups.items()
                if split == request.calibration_value
            )
            nonconformity = sorted(
                1.0 - score(group, groups[group][1]) for group in calibration_groups
            )
            rank = min(
                len(nonconformity) - 1,
                max(0, math.ceil((len(nonconformity) + 1) * (1.0 - request.calibration_alpha)) - 1),
            )
            threshold = float(nonconformity[rank])
            test_groups = sorted(
                group for group, (split, _) in groups.items() if split == request.test_value
            )
            labels = [groups[group][1] for group in test_groups]
            argmax = [
                max(classes, key=lambda cls: score(group, cls)) for group in test_groups
            ]
            prediction_sets = [
                [cls for cls in classes if 1.0 - score(group, cls) <= threshold]
                for group in test_groups
            ]
            accuracy = sum(y == p for y, p in zip(labels, argmax, strict=True)) / len(labels)
            coverage = sum(
                label in values for label, values in zip(labels, prediction_sets, strict=True)
            ) / len(labels)
            mean_size = sum(map(len, prediction_sets)) / len(prediction_sets)
            manifest = {
                str(path.relative_to(output_dir)): sha256_file(path)
                for path in sorted(output_dir.rglob("*"))
                if path.is_file()
            }
            identity = {
                "spec_sha256": spec_sha,
                "data_sha256": data_sha,
                "runtime_source_sha256": runtime.package_source_sha256,
                "output_manifest_sha256": manifest,
                "calibration_threshold": threshold,
                "test_metrics": [accuracy, coverage, mean_size],
            }
            return M2NChallengerAudit(
                audit_id=stable_id("M2N-CH", identity),
                status=M2NChallengerStatus.COMPLETED,
                reason="Offline M2N challenger completed; production promotion remains prohibited.",
                spec_sha256=spec_sha,
                data_sha256=data_sha,
                split_group_counts=dict(sorted(counts.items())),
                optimizer_seeds=seeds,
                candidate_ids=sorted(risks),
                simple_baseline_id="summary_mlp",
                selected_candidate_id=result.get("selected_candidate_id"),
                selection_status=result.get("status"),
                calibration_alpha=request.calibration_alpha,
                calibration_nonconformity_threshold=threshold,
                calibration_context_count=len(calibration_groups),
                test_context_count=len(test_groups),
                test_argmax_accuracy=float(accuracy),
                test_macro_f1=self._macro_f1(labels, argmax, classes),
                test_prediction_set_coverage=float(coverage),
                test_mean_prediction_set_size=float(mean_size),
                output_manifest_sha256=manifest,
                runtime=runtime,
                design_result=result,
            )
        except Exception as exc:
            identity = {
                "spec_sha256": spec_sha,
                "data_sha256": data_sha,
                "runtime_source_sha256": runtime.package_source_sha256 if runtime else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return M2NChallengerAudit(
                audit_id=stable_id("M2N-CH", identity),
                status=M2NChallengerStatus.ERROR,
                reason=f"{type(exc).__name__}: {exc}",
                spec_sha256=spec_sha,
                data_sha256=data_sha,
                split_group_counts=dict(sorted(counts.items())),
                optimizer_seeds=seeds,
                runtime=runtime,
            )
