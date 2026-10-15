from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, model_validator

from astrograph.integrations.dor_active_discovery import R5ActiveDiscoveryPayload


class RWActiveDiscoveryProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    simulation_model: str = Field(min_length=1)
    source_evidence_level: Literal["E0_ENGINEERING_ASSUMPTION"]
    bank_draws_per_particle_experiment: StrictInt = Field(ge=3)
    parameter_particles_per_structure: StrictInt = Field(ge=1)
    held_out_seed_namespace: str = Field(min_length=1)
    adcs_v5_2_2_test_partition_accessed: StrictBool = False
    phm_v6_0_test_partition_accessed: StrictBool = False
    dataset_metric_claimed: StrictBool = False

    @model_validator(mode="after")
    def deny_dataset_claims(self) -> "RWActiveDiscoveryProvenance":
        if any(
            (
                self.adcs_v5_2_2_test_partition_accessed,
                self.phm_v6_0_test_partition_accessed,
                self.dataset_metric_claimed,
            )
        ):
            raise ValueError("E0 pilot cannot access dataset test partitions or claim dataset metrics")
        return self


class RWActiveDiscoveryAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_count: StrictInt = Field(ge=1)
    evaluation_seed_count: StrictInt = Field(ge=2)
    top1_correct: StrictInt = Field(ge=0)
    top1_accuracy: StrictFloat = Field(ge=0.0, le=1.0)
    mean_initial_entropy_bits: StrictFloat = Field(ge=0.0)
    mean_final_entropy_bits: StrictFloat = Field(ge=0.0)
    mean_experiments_used: StrictFloat = Field(gt=0.0)


class RWActiveDiscoveryEpisode(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    evaluation_truth_structure: str = Field(min_length=1)
    evaluation_seed: StrictInt
    rank_input_truth_fields: list[str] = Field(default_factory=list, max_length=0)
    top_structure: str = Field(min_length=1)
    correct: StrictBool


class RWActiveDiscoveryPilotReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dor-r5-rw-pilot-report.v1"]
    claim_boundary: str = Field(min_length=1)
    runtime: dict[str, str]
    policy: Literal["active", "fixed", "random"]
    maximum_experiments: StrictInt = Field(ge=1, le=3)
    provenance: RWActiveDiscoveryProvenance
    feature_names: list[str] = Field(min_length=1)
    candidate_structures: list[str] = Field(min_length=3, max_length=3)
    experiment_catalog: list[dict] = Field(min_length=2)
    episodes: list[RWActiveDiscoveryEpisode] = Field(min_length=6)
    aggregate: RWActiveDiscoveryAggregate
    representative_advisory_payload: R5ActiveDiscoveryPayload

    @model_validator(mode="after")
    def cross_check_report(self) -> "RWActiveDiscoveryPilotReport":
        expected = {"friction_increase", "torque_authority_loss", "jam_output_stagnation"}
        if set(self.candidate_structures) != expected or len(set(self.candidate_structures)) != 3:
            raise ValueError("pilot must contain exactly the governed three-structure RW hypothesis set")
        pairs = {(row.evaluation_truth_structure, row.evaluation_seed) for row in self.episodes}
        if len(pairs) != len(self.episodes):
            raise ValueError("pilot evaluation structure/seed pairs must be unique")
        if {row.evaluation_truth_structure for row in self.episodes} != expected:
            raise ValueError("pilot must evaluate every governed RW structure")
        if self.aggregate.episode_count != len(self.episodes):
            raise ValueError("pilot aggregate episode count mismatch")
        correct = sum(row.correct for row in self.episodes)
        if self.aggregate.top1_correct != correct:
            raise ValueError("pilot aggregate correct count mismatch")
        if abs(float(self.aggregate.top1_accuracy) - correct / len(self.episodes)) > 1e-12:
            raise ValueError("pilot aggregate accuracy mismatch")
        return self


class RWActiveDiscoveryPilotService:
    """Execute the reviewed ISGN runner in a temporary directory and validate its report."""

    def run(
        self,
        *,
        python_executable: str,
        runner_path: str | Path,
        timeout_s: float = 120.0,
        policy: Literal["active", "fixed", "random"] = "active",
        maximum_experiments: int = 3,
        evaluation_seed_count: int = 3,
    ) -> RWActiveDiscoveryPilotReport:
        runner = Path(runner_path).resolve()
        if not runner.is_file():
            raise FileNotFoundError(runner)
        with tempfile.TemporaryDirectory(prefix="astrograph-rw-active-discovery-") as directory:
            output = Path(directory) / "report.json"
            completed = subprocess.run(
                [
                    python_executable,
                    str(runner),
                    "--output",
                    str(output),
                    "--policy",
                    policy,
                    "--max-experiments",
                    str(maximum_experiments),
                    "--evaluation-seed-count",
                    str(evaluation_seed_count),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if completed.returncode != 0 or not output.is_file():
                detail = (completed.stderr or completed.stdout or "no pilot output").strip()[-2000:]
                raise RuntimeError(f"RW Active Discovery pilot failed ({completed.returncode}): {detail}")
            raw = json.loads(output.read_text(encoding="utf-8"))
        return RWActiveDiscoveryPilotReport.model_validate(raw)


class RWActiveDiscoveryPolicyPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Literal["active", "fixed", "random"]
    experiment_budget: StrictInt = Field(ge=1, le=3)
    episode_count: StrictInt = Field(ge=1)
    top1_accuracy: StrictFloat = Field(ge=0.0, le=1.0)
    mean_final_entropy_bits: StrictFloat = Field(ge=0.0)


class RWActiveDiscoveryComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dor-r5-rw-comparison.v1"]
    points: list[RWActiveDiscoveryPolicyPoint] = Field(min_length=9, max_length=9)
    entropy_target_bits: StrictFloat = Field(gt=0.0)
    experiments_to_entropy_target: dict[str, StrictInt | None]
    stop_gate_status: Literal["ADVANCE", "STOP_NO_CLEAR_BENEFIT"]
    reasons: list[str] = Field(min_length=1)
    online_integration_allowed: Literal[False] = False
    kg_update_allowed: Literal[False] = False
    birrf_integration_allowed: Literal[False] = False


class RWActiveDiscoveryComparisonService:
    """Run equal-seed/equal-budget Active, fixed and random offline comparisons."""

    def run(
        self,
        *,
        python_executable: str,
        runner_path: str | Path,
        evaluation_seed_count: int = 5,
        timeout_s: float = 180.0,
    ) -> RWActiveDiscoveryComparisonReport:
        pilot = RWActiveDiscoveryPilotService()
        points: list[RWActiveDiscoveryPolicyPoint] = []
        for policy in ("active", "fixed", "random"):
            for budget in (1, 2, 3):
                report = pilot.run(
                    python_executable=python_executable,
                    runner_path=runner_path,
                    timeout_s=timeout_s,
                    policy=policy,
                    maximum_experiments=budget,
                    evaluation_seed_count=evaluation_seed_count,
                )
                points.append(
                    RWActiveDiscoveryPolicyPoint(
                        policy=policy,
                        experiment_budget=budget,
                        episode_count=report.aggregate.episode_count,
                        top1_accuracy=float(report.aggregate.top1_accuracy),
                        mean_final_entropy_bits=max(
                            float(report.aggregate.mean_final_entropy_bits), 0.0
                        ),
                    )
                )
        target = 0.05
        to_target: dict[str, int | None] = {}
        for policy in ("active", "fixed", "random"):
            eligible = [
                row.experiment_budget
                for row in points
                if row.policy == policy and row.mean_final_entropy_bits <= target
            ]
            to_target[policy] = min(eligible) if eligible else None
        at_one = {row.policy: row for row in points if row.experiment_budget == 1}
        accuracy_gain = at_one["active"].top1_accuracy - max(
            at_one["fixed"].top1_accuracy, at_one["random"].top1_accuracy
        )
        baseline_steps = [to_target[name] for name in ("fixed", "random") if to_target[name] is not None]
        experiment_gain = bool(
            to_target["active"] is not None
            and baseline_steps
            and to_target["active"] < min(baseline_steps)
        )
        advance = accuracy_gain >= 0.02 or experiment_gain
        reasons = [
            f"one-experiment top-1 gain over best baseline: {accuracy_gain:.6f}",
            f"experiments to <= {target:.2f} bits: active={to_target['active']}, fixed={to_target['fixed']}, random={to_target['random']}",
        ]
        if not advance:
            reasons.append("neither the 0.02 accuracy margin nor an experiment-count reduction was achieved")
        return RWActiveDiscoveryComparisonReport(
            schema_version="astrograph-dor-r5-rw-comparison.v1",
            points=points,
            entropy_target_bits=target,
            experiments_to_entropy_target=to_target,
            stop_gate_status="ADVANCE" if advance else "STOP_NO_CLEAR_BENEFIT",
            reasons=reasons,
        )


__all__ = [
    "RWActiveDiscoveryPilotReport",
    "RWActiveDiscoveryPilotService",
    "RWActiveDiscoveryComparisonReport",
    "RWActiveDiscoveryComparisonService",
]
