from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .credibility import IndependentRunRecord, RunProvenanceType, SplitName


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20].upper()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: str, *, field_name: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or ":" in path.parts[0]:
        raise ValueError(f"{field_name} must be a relative POSIX path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} contains an unsafe path segment")
    return path


def _resolved_under(root: Path, relative: str, *, field_name: str) -> Path:
    path = (root / _safe_relative_path(relative, field_name=field_name)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes configured root") from exc
    return path


class CampaignPlanStatus(str, Enum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class CampaignScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    class_ids: list[str]
    object_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_ids")
    @classmethod
    def unique_classes(cls, value: list[str]) -> list[str]:
        if not value or len(value) != len(set(value)):
            raise ValueError("scenario class_ids must be nonempty and unique")
        return value


class CampaignVariant(BaseModel):
    """One explicitly declared simulator factor combination.

    The factory does not invent simulator flags.  Any initial-state, operating-condition,
    noise, or severity variation must be represented by concrete CLI arguments accepted by
    the external registered runtime.  Reserved arguments remain controlled by AstroGraph.
    """

    model_config = ConfigDict(extra="forbid")

    variant_id: str = Field(min_length=1)
    cli_args: list[str] = Field(default_factory=list)
    initial_state_id: str | None = None
    operating_condition_id: str | None = None
    fault_severity_id: str | None = None
    satellite_instance_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cli_args")
    @classmethod
    def protect_reserved_cli_args(cls, value: list[str]) -> list[str]:
        reserved = {"--scenario", "--seed", "--duration-s", "--output", "--smoke-all"}
        for item in value:
            if item in reserved or any(item.startswith(f"{flag}=") for flag in reserved):
                raise ValueError(f"campaign variant cannot override reserved runtime argument: {item}")
            if "\x00" in item:
                raise ValueError("campaign CLI arguments cannot contain NUL")
        return value


class CampaignSplitCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: int = Field(default=1, ge=1)
    validation: int = Field(default=1, ge=1)
    test: int = Field(default=1, ge=1)

    @property
    def total(self) -> int:
        return self.train + self.validation + self.test


class IndependentRunCampaignSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-independent-run-campaign-spec.v1"
    campaign_id: str = Field(min_length=1)
    runtime_script_name: str = Field(default="run_whole_spacecraft_all_faults.py", pattern=r"^[A-Za-z0-9_.-]+\.py$")
    normal_scenario_id: str = Field(default="nominal", min_length=1)
    normal_class_id: str = Field(default="NORMAL", min_length=1)
    scenarios: list[CampaignScenario]
    seeds: list[int]
    variants: list[CampaignVariant] = Field(default_factory=lambda: [CampaignVariant(variant_id="DEFAULT")])
    duration_s: float = Field(default=120.0, gt=0.0)
    split_counts: CampaignSplitCounts = Field(default_factory=CampaignSplitCounts)
    python_executable: str = "python"
    output_layout: str = Field(default="runs/{run_id}", pattern=r"^runs/\{run_id\}$")
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "A campaign plan defines intended independent simulator invocations. Independence is not established "
        "until produced run artifacts pass content, provenance, grouping, and split audits."
    )

    @field_validator("scenarios")
    @classmethod
    def unique_scenarios(cls, value: list[CampaignScenario]) -> list[CampaignScenario]:
        ids = [item.scenario_id for item in value]
        if not value or len(ids) != len(set(ids)):
            raise ValueError("campaign scenario IDs must be nonempty and unique")
        return value

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, value: list[int]) -> list[int]:
        if not value or len(value) != len(set(value)):
            raise ValueError("campaign seeds must be nonempty and unique")
        return value

    @field_validator("variants")
    @classmethod
    def unique_variants(cls, value: list[CampaignVariant]) -> list[CampaignVariant]:
        ids = [item.variant_id for item in value]
        if not value or len(ids) != len(set(ids)):
            raise ValueError("campaign variant IDs must be nonempty and unique")
        return value

    @model_validator(mode="after")
    def reserve_normal_scenario(self) -> "IndependentRunCampaignSpec":
        if self.normal_scenario_id in {item.scenario_id for item in self.scenarios}:
            raise ValueError("normal_scenario_id must not be repeated in fault scenarios")
        return self


class CampaignRunPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    class_ids: list[str]
    split: SplitName
    seed: int
    variant_id: str = Field(min_length=1)
    duration_s: float = Field(gt=0.0)
    independence_group_id: str = Field(min_length=1)
    paired_nominal_run_id: str | None = None
    output_relative_root: str = Field(min_length=1)
    expected_scenario_relative_dir: str = Field(min_length=1)
    # Retained as an audit/display artifact only. Execution reconstructs argv from
    # typed fields and never executes this template directly.
    command_argv_template: list[str]
    runtime_cli_args: list[str] = Field(default_factory=list)
    object_id: str | None = None
    initial_state_id: str | None = None
    operating_condition_id: str | None = None
    fault_severity_id: str | None = None
    satellite_instance_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("runtime_cli_args")
    @classmethod
    def protect_runtime_cli_args(cls, value: list[str]) -> list[str]:
        reserved = {"--scenario", "--seed", "--duration-s", "--output", "--smoke-all"}
        for item in value:
            if item in reserved or any(item.startswith(f"{flag}=") for flag in reserved):
                raise ValueError(f"campaign run cannot override reserved runtime argument: {item}")
            if "\x00" in item:
                raise ValueError("campaign runtime arguments cannot contain NUL")
        return value

    @model_validator(mode="after")
    def validate_output_layout(self) -> "CampaignRunPlan":
        output = _safe_relative_path(self.output_relative_root, field_name="output_relative_root")
        expected = _safe_relative_path(
            self.expected_scenario_relative_dir,
            field_name="expected_scenario_relative_dir",
        )
        if expected.parts[: len(output.parts)] != output.parts:
            raise ValueError("expected_scenario_relative_dir must be under output_relative_root")
        if expected.name != self.scenario_id:
            raise ValueError("expected_scenario_relative_dir must end with scenario_id")
        return self


class IndependentRunCampaignPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-independent-run-campaign-plan.v2"
    campaign_id: str = Field(min_length=1)
    status: CampaignPlanStatus
    runtime_script_name: str = Field(min_length=1)
    runs: list[CampaignRunPlan]
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    varied_factor_dimensions: list[str] = Field(default_factory=list)
    split_counts: CampaignSplitCounts
    factor_combination_count: int = Field(ge=0)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "PASS means the requested run matrix can be scheduled without reusing one factor combination across "
        "train/validation/test. It does not prove the simulator responds materially to those factors."
    )


def campaign_plan_sha256(plan: IndependentRunCampaignPlan) -> str:
    payload = {
        "schema_version": plan.schema_version,
        "campaign_id": plan.campaign_id,
        "status": plan.status.value,
        "runtime_script_name": plan.runtime_script_name,
        "runs": [item.model_dump(mode="json") for item in plan.runs],
        "blocking_reasons": plan.blocking_reasons,
        "warnings": plan.warnings,
        "varied_factor_dimensions": plan.varied_factor_dimensions,
        "split_counts": plan.split_counts.model_dump(mode="json"),
        "factor_combination_count": plan.factor_combination_count,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class CampaignExecutionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str = Field(pattern=r"^(SUCCEEDED|FAILED|SKIPPED)$")
    return_code: int | None = None
    command: list[str]
    output_root: str
    message: str = ""


class CampaignExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-independent-run-campaign-execution.v1"
    campaign_id: str
    status: str = Field(pattern=r"^(PASS|FAILED|BLOCKED)$")
    rows: list[CampaignExecutionRow]
    runtime_script_sha256: str | None = None
    claim_boundary: str = "Execution success does not imply run independence, data quality, or diagnostic validity."


def _split_for_index(index: int, counts: CampaignSplitCounts) -> SplitName:
    if index < counts.train:
        return SplitName.TRAIN
    if index < counts.train + counts.validation:
        return SplitName.VALIDATION
    return SplitName.TEST


def build_independent_run_campaign_plan(spec: IndependentRunCampaignSpec) -> IndependentRunCampaignPlan:
    factors = [(seed, variant) for seed in sorted(spec.seeds) for variant in sorted(spec.variants, key=lambda x: x.variant_id)]
    blocking: list[str] = []
    if len(factors) < spec.split_counts.total:
        blocking.append(
            f"INSUFFICIENT_FACTOR_COMBINATIONS:{len(factors)}<{spec.split_counts.total}"
        )
    selected = factors[: spec.split_counts.total]
    warnings: list[str] = []
    varied_dimensions: list[str] = []
    if len(spec.seeds) > 1:
        varied_dimensions.append("RANDOM_SEED")
    factor_fields = {
        "RUNTIME_VARIANT": [item.variant_id for item in spec.variants],
        "INITIAL_STATE": [item.initial_state_id for item in spec.variants],
        "OPERATING_CONDITION": [item.operating_condition_id for item in spec.variants],
        "FAULT_SEVERITY": [item.fault_severity_id for item in spec.variants],
        "SATELLITE_INSTANCE": [item.satellite_instance_id for item in spec.variants],
    }
    for dimension, values in factor_fields.items():
        unique = {item for item in values if item is not None}
        if len(unique) > 1 or (dimension == "RUNTIME_VARIANT" and len(set(values)) > 1):
            varied_dimensions.append(dimension)
    if varied_dimensions == ["RANDOM_SEED"]:
        warnings.append("ONLY_RANDOM_SEED_VARIES")
    if "OPERATING_CONDITION" not in varied_dimensions:
        warnings.append("OPERATING_CONDITION_NOT_VARIED")
    if "FAULT_SEVERITY" not in varied_dimensions:
        warnings.append("FAULT_SEVERITY_NOT_VARIED")
    if "INITIAL_STATE" not in varied_dimensions:
        warnings.append("INITIAL_STATE_NOT_VARIED")
    runs: list[CampaignRunPlan] = []
    nominal_by_factor: dict[tuple[int, str], str] = {}

    for index, (seed, variant) in enumerate(selected):
        split = _split_for_index(index, spec.split_counts)
        group_id = _stable_id(
            "IG",
            {"campaign_id": spec.campaign_id, "seed": seed, "variant_id": variant.variant_id},
        )
        nominal_id = _stable_id(
            "RUN",
            {
                "campaign_id": spec.campaign_id,
                "scenario_id": spec.normal_scenario_id,
                "seed": seed,
                "variant_id": variant.variant_id,
            },
        )
        nominal_by_factor[(seed, variant.variant_id)] = nominal_id
        output_root = spec.output_layout.format(run_id=nominal_id)
        runs.append(
            CampaignRunPlan(
                run_id=nominal_id,
                scenario_id=spec.normal_scenario_id,
                class_ids=[spec.normal_class_id],
                split=split,
                seed=seed,
                variant_id=variant.variant_id,
                duration_s=spec.duration_s,
                independence_group_id=group_id,
                output_relative_root=output_root,
                expected_scenario_relative_dir=f"{output_root}/{spec.normal_scenario_id}",
                command_argv_template=[
                    spec.python_executable,
                    "{runtime_script}",
                    "--scenario",
                    spec.normal_scenario_id,
                    "--duration-s",
                    str(spec.duration_s),
                    "--seed",
                    str(seed),
                    "--output",
                    "{run_output_root}",
                    *variant.cli_args,
                ],
                runtime_cli_args=list(variant.cli_args),
                initial_state_id=variant.initial_state_id,
                operating_condition_id=variant.operating_condition_id,
                fault_severity_id=None,
                satellite_instance_id=variant.satellite_instance_id,
                metadata={"variant_metadata": variant.metadata, "normal_reference": True},
            )
        )

    for scenario in sorted(spec.scenarios, key=lambda x: x.scenario_id):
        for index, (seed, variant) in enumerate(selected):
            split = _split_for_index(index, spec.split_counts)
            run_id = _stable_id(
                "RUN",
                {
                    "campaign_id": spec.campaign_id,
                    "scenario_id": scenario.scenario_id,
                    "seed": seed,
                    "variant_id": variant.variant_id,
                },
            )
            output_root = spec.output_layout.format(run_id=run_id)
            runs.append(
                CampaignRunPlan(
                    run_id=run_id,
                    scenario_id=scenario.scenario_id,
                    class_ids=scenario.class_ids,
                    split=split,
                    seed=seed,
                    variant_id=variant.variant_id,
                    duration_s=spec.duration_s,
                    independence_group_id=_stable_id(
                        "IG",
                        {
                            "campaign_id": spec.campaign_id,
                            "scenario_id": scenario.scenario_id,
                            "seed": seed,
                            "variant_id": variant.variant_id,
                        },
                    ),
                    paired_nominal_run_id=nominal_by_factor[(seed, variant.variant_id)],
                    output_relative_root=output_root,
                    expected_scenario_relative_dir=f"{output_root}/{scenario.scenario_id}",
                    command_argv_template=[
                        spec.python_executable,
                        "{runtime_script}",
                        "--scenario",
                        scenario.scenario_id,
                        "--duration-s",
                        str(spec.duration_s),
                        "--seed",
                        str(seed),
                        "--output",
                        "{run_output_root}",
                        *variant.cli_args,
                    ],
                    runtime_cli_args=list(variant.cli_args),
                    object_id=scenario.object_id,
                    initial_state_id=variant.initial_state_id,
                    operating_condition_id=variant.operating_condition_id,
                    fault_severity_id=variant.fault_severity_id,
                    satellite_instance_id=variant.satellite_instance_id,
                    metadata={"scenario_metadata": scenario.metadata, "variant_metadata": variant.metadata},
                )
            )

    plan = IndependentRunCampaignPlan(
        campaign_id=spec.campaign_id,
        status=CampaignPlanStatus.BLOCKED if blocking else CampaignPlanStatus.PASS,
        runtime_script_name=spec.runtime_script_name,
        runs=runs,
        blocking_reasons=blocking,
        warnings=warnings,
        varied_factor_dimensions=varied_dimensions,
        split_counts=spec.split_counts,
        factor_combination_count=len(factors),
        plan_sha256="0" * 64,
    )
    return plan.model_copy(update={"plan_sha256": campaign_plan_sha256(plan)})


def execute_independent_run_campaign(
    plan: IndependentRunCampaignPlan,
    *,
    runtime_root: str | Path,
    output_root: str | Path,
    stop_on_failure: bool = True,
) -> CampaignExecutionReport:
    if plan.status != CampaignPlanStatus.PASS:
        return CampaignExecutionReport(campaign_id=plan.campaign_id, status="BLOCKED", rows=[])
    if plan.schema_version != "astrograph-independent-run-campaign-plan.v2":
        raise ValueError("legacy campaign plans must be regenerated before execution")
    expected_plan_hash = campaign_plan_sha256(plan)
    if expected_plan_hash != plan.plan_sha256:
        raise ValueError("campaign plan SHA256 validation failed")

    runtime_root = Path(runtime_root).resolve()
    output_root = Path(output_root).resolve()
    runtime_script = (runtime_root / "scripts" / plan.runtime_script_name).resolve()
    if runtime_script.name != plan.runtime_script_name or not runtime_script.is_file():
        raise FileNotFoundError(f"registered runtime script not found: {runtime_script}")
    if runtime_script.parent != (runtime_root / "scripts").resolve():
        raise ValueError("runtime script must be located directly under runtime_root/scripts")
    script_hash = _sha256_file(runtime_script)
    rows: list[CampaignExecutionRow] = []
    failed = False
    for run in plan.runs:
        run_output_root = _resolved_under(
            output_root,
            run.output_relative_root,
            field_name="output_relative_root",
        )
        expected_scenario_dir = _resolved_under(
            output_root,
            run.expected_scenario_relative_dir,
            field_name="expected_scenario_relative_dir",
        )
        try:
            expected_scenario_dir.relative_to(run_output_root)
        except ValueError as exc:
            raise ValueError("expected scenario directory escapes run output root") from exc

        if failed and stop_on_failure:
            rows.append(
                CampaignExecutionRow(
                    run_id=run.run_id,
                    status="SKIPPED",
                    command=[],
                    output_root=str(run_output_root),
                    message="Skipped after an earlier campaign run failed.",
                )
            )
            continue
        run_output_root.mkdir(parents=True, exist_ok=True)
        # Never execute command_argv_template. It is untrusted plan metadata retained
        # only for audit compatibility. argv is reconstructed from typed, validated
        # fields and the registered script.
        command = [
            sys.executable,
            str(runtime_script),
            "--scenario",
            run.scenario_id,
            "--duration-s",
            str(run.duration_s),
            "--seed",
            str(run.seed),
            "--output",
            str(run_output_root),
            *run.runtime_cli_args,
        ]
        completed = subprocess.run(command, cwd=runtime_root, check=False)
        status = "SUCCEEDED" if completed.returncode == 0 else "FAILED"
        failed = failed or completed.returncode != 0
        rows.append(
            CampaignExecutionRow(
                run_id=run.run_id,
                status=status,
                return_code=completed.returncode,
                command=command,
                output_root=str(run_output_root),
            )
        )
    return CampaignExecutionReport(
        campaign_id=plan.campaign_id,
        status="FAILED" if failed else "PASS",
        rows=rows,
        runtime_script_sha256=script_hash,
    )


def build_inventory_from_campaign_plan(
    plan: IndependentRunCampaignPlan,
    *,
    output_root: str | Path,
) -> tuple[list[IndependentRunRecord], list[str]]:
    """Build a strict inventory from produced artifacts without treating missing runs as evidence.

    The external runtime is expected to write ``telemetry.csv`` and ``run_manifest.json`` under
    ``<output_root>/<run.output_relative_root>/<scenario_id>``. Exact telemetry duplicates and
    split/group leakage are detected later by ``audit_independent_runs(..., strict_provenance=True)``.
    """

    output_root = Path(output_root).resolve()
    records: list[IndependentRunRecord] = []
    missing: list[str] = []
    plan_by_id = {item.run_id: item for item in plan.runs}
    for run in plan.runs:
        scenario_dir = output_root / run.expected_scenario_relative_dir
        telemetry = scenario_dir / "telemetry.csv"
        manifest = scenario_dir / "run_manifest.json"
        if not telemetry.is_file() or not manifest.is_file():
            missing.append(run.run_id)
            continue
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        scenario = str(manifest_payload.get("scenario", ""))
        seed = manifest_payload.get("seed")
        if scenario != run.scenario_id:
            raise ValueError(f"run {run.run_id} manifest scenario mismatch: {scenario!r}")
        if int(seed) != run.seed:
            raise ValueError(f"run {run.run_id} manifest seed mismatch: {seed!r}")
        config_payload = {
            "scenario": scenario,
            "seed": int(seed),
            "duration_s": manifest_payload.get("duration_s"),
            "events": manifest_payload.get("events", []),
            "runtime": manifest_payload.get("runtime"),
            "variant_id": run.variant_id,
            "initial_state_id": run.initial_state_id,
            "operating_condition_id": run.operating_condition_id,
            "fault_severity_id": run.fault_severity_id,
        }
        config_hash = hashlib.sha256(
            json.dumps(config_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        paired = plan_by_id.get(run.paired_nominal_run_id) if run.paired_nominal_run_id else None
        records.append(
            IndependentRunRecord(
                run_id=run.run_id,
                class_ids=run.class_ids,
                split=run.split,
                object_id=run.object_id,
                operating_condition_id=run.operating_condition_id,
                satellite_instance_id=run.satellite_instance_id,
                random_seed=run.seed,
                initial_state_id=run.initial_state_id,
                fault_severity_id=run.fault_severity_id,
                provenance_type=RunProvenanceType.SIMULATION_NATIVE,
                independence_group_id=run.independence_group_id,
                content_sha256=_sha256_file(telemetry),
                configuration_sha256=config_hash,
                parent_run_id=None,
                paired_nominal_run_id=None if paired is None else paired.run_id,
                metadata={
                    "scenario_id": run.scenario_id,
                    "variant_id": run.variant_id,
                    "scenario_dir": Path(run.expected_scenario_relative_dir).as_posix(),
                    "scenario_dir_relative": Path(run.expected_scenario_relative_dir).as_posix(),
                    "scenario_dir_base": "CAMPAIGN_OUTPUT_ROOT",
                    "manifest_sha256": _sha256_file(manifest),
                    "campaign_id": plan.campaign_id,
                    "campaign_plan_sha256": plan.plan_sha256,
                },
            )
        )
    return records, missing
