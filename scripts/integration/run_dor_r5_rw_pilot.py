#!/usr/bin/env python3
"""Run the DOR R5 reaction-wheel Active Discovery E0 pilot.

This runner must be executed with the reviewed DOR environment.  It deliberately
keeps evaluation truth outside every call to ``rank_experiments``: the planner sees
only a prior, a simulation bank and an offline experiment catalogue.  A held-out
synthetic observation is generated only after an experiment has been selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from isgn import __version__ as isgn_version
from isgn.active_discovery import (
    ExperimentCandidate,
    ParameterParticle,
    ParameterParticleSMC,
    ParticleSimulationBank,
    assess_evidence_stability,
    assess_structure_identifiability,
    predictive_adequacy,
    rank_experiments,
)


STRUCTURES = ("friction_increase", "torque_authority_loss", "jam_output_stagnation")
EXPERIMENTS = (
    ExperimentCandidate("low_speed_torque_pulse", repeats=1, unit_cost=1.0, execution_mode="SIMULATION"),
    ExperimentCandidate("high_speed_sweep", repeats=1, unit_cost=1.2, execution_mode="SIMULATION"),
    ExperimentCandidate("commanded_coast_down", repeats=1, unit_cost=0.8, execution_mode="SIMULATION"),
)
FEATURES = (
    "speed_response_ratio",
    "torque_execution_ratio",
    "normalized_motor_current",
    "output_stagnation_ratio",
)


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _mean(structure: str, severity: float, experiment_id: str) -> np.ndarray:
    """Small dimensionless response model derived from RW balance signatures.

    Values are engineering assumptions, not flight-calibrated distributions.  They
    encode only qualitative mechanism effects: friction raises current and coast
    damping, authority loss lowers torque execution, and a jam causes stagnation.
    """
    if structure == "friction_increase":
        table = {
            "low_speed_torque_pulse": (0.76, 0.92, 1.15 + 0.45 * severity, 0.12),
            "high_speed_sweep": (0.62, 0.88, 1.35 + 0.70 * severity, 0.20),
            "commanded_coast_down": (0.48 - 0.18 * severity, 0.96, 0.62, 0.16),
        }
    elif structure == "torque_authority_loss":
        table = {
            "low_speed_torque_pulse": (0.42 - 0.16 * severity, 0.38 - 0.20 * severity, 0.92, 0.24),
            "high_speed_sweep": (0.52 - 0.18 * severity, 0.44 - 0.20 * severity, 0.94, 0.25),
            "commanded_coast_down": (0.83, 0.46 - 0.18 * severity, 0.48, 0.08),
        }
    elif structure == "jam_output_stagnation":
        table = {
            "low_speed_torque_pulse": (0.06 + 0.03 * severity, 0.04, 1.55, 0.94),
            "high_speed_sweep": (0.04 + 0.03 * severity, 0.03, 1.72, 0.97),
            "commanded_coast_down": (0.03, 0.05, 0.72, 0.98),
        }
    else:
        raise KeyError(structure)
    return np.asarray(table[experiment_id], dtype=float)


def _draws(structure: str, severity: float, experiment_id: str, *, seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.asarray((0.045, 0.040, 0.070, 0.035), dtype=float)
    values = rng.normal(_mean(structure, severity, experiment_id), noise, size=(count, len(FEATURES)))
    return np.clip(values, 0.0, None)


def _bank() -> tuple[list[ParameterParticle], ParticleSimulationBank]:
    particles: list[ParameterParticle] = []
    samples: dict[tuple[str, str, int], np.ndarray] = {}
    for structure in STRUCTURES:
        for index, severity in enumerate((0.25, 0.50, 0.75)):
            particle_id = f"{structure}:severity-{index}"
            particles.append(ParameterParticle(particle_id, structure, {"severity": severity}))
            for candidate in EXPERIMENTS:
                samples[(particle_id, candidate.experiment_id, candidate.repeats)] = _draws(
                    structure,
                    severity,
                    candidate.experiment_id,
                    seed=_seed("bank", particle_id, candidate.experiment_id),
                    count=32,
                )
    return particles, ParticleSimulationBank(
        samples,
        shrinkage=0.35,
        ridge=1e-3,
        covariance_floor=(0.025, 0.025, 0.04, 0.02),
        student_df=5.0,
    )


def _episode(
    structure: str,
    evaluation_seed: int,
    *,
    policy: str = "active",
    max_experiments: int = 3,
) -> dict[str, Any]:
    particles, bank = _bank()
    prior = ParameterParticleSMC(particles, seed=evaluation_seed)
    posterior = prior.copy()
    observations: list[tuple[str, np.ndarray, int]] = []
    remaining = list(EXPERIMENTS)
    selected: list[dict[str, Any]] = []
    adequacy: dict[str, Any] | None = None
    initial_entropy = posterior.structure_entropy()
    for step in range(max_experiments):
        if not remaining:
            break
        scores = []
        if policy == "active":
            scores = rank_experiments(
                posterior,
                bank,
                remaining,
                remaining_budget=3.0 - step * 0.8,
                max_source_particles=9,
                predictive_draws_per_particle=2,
            )
            if not scores:
                break
            chosen_candidate = next(
                item for item in remaining if item.experiment_id == scores[0].experiment_id
            )
            expected_gain = scores[0].expected_structure_information_gain_bits
            utility_per_cost = scores[0].utility_per_cost
        elif policy == "fixed":
            chosen_candidate = remaining[0]
            expected_gain = None
            utility_per_cost = None
        elif policy == "random":
            # Selection depends only on the public evaluation seed and step, never on structure truth.
            rng = np.random.default_rng(_seed("random-policy", evaluation_seed, step))
            chosen_candidate = remaining[int(rng.integers(0, len(remaining)))]
            expected_gain = None
            utility_per_cost = None
        else:
            raise ValueError(f"unsupported policy: {policy}")
        # Evaluation truth enters only here, after ranking has returned.
        observation = _draws(
            structure,
            0.50,
            chosen_candidate.experiment_id,
            seed=_seed("held-out", structure, evaluation_seed, chosen_candidate.experiment_id),
            count=1,
        )[0]
        update = posterior.update(
            bank,
            chosen_candidate.experiment_id,
            observation,
            repeats=chosen_candidate.repeats,
        )
        adequacy = predictive_adequacy(
            posterior,
            bank,
            chosen_candidate.experiment_id,
            observation,
        )
        observations.append(
            (chosen_candidate.experiment_id, observation, chosen_candidate.repeats)
        )
        selected.append(
            {
                "step": step + 1,
                "experiment_id": chosen_candidate.experiment_id,
                "expected_information_gain_bits": expected_gain,
                "utility_per_cost": utility_per_cost,
                "posterior_entropy_bits": update.posterior_structure_entropy_bits,
            }
        )
        remaining = [
            item for item in remaining if item.experiment_id != chosen_candidate.experiment_id
        ]
    stability = assess_evidence_stability(posterior, prior, bank, observations)
    identifiability = assess_structure_identifiability(
        posterior,
        bank,
        EXPERIMENTS,
        remaining_budget=0.0,
        executed_experiment_ids=[item[0] for item in observations],
        adequacy_status=(adequacy or {"status": "HYPOTHESIS_CLASS_INADEQUATE"})["status"],
        evidence_stability_status=stability.status,
        hypothesis_space_coverage_complete=True,
    )
    structure_posterior = posterior.structure_posterior()
    top_structure = max(structure_posterior, key=structure_posterior.get)
    return {
        "evaluation_truth_structure": structure,
        "evaluation_seed": evaluation_seed,
        "rank_input_truth_fields": [],
        "selected_experiments": selected,
        "initial_entropy_bits": initial_entropy,
        "final_entropy_bits": posterior.structure_entropy(),
        "structure_posterior": structure_posterior,
        "top_structure": top_structure,
        "correct": top_structure == structure,
        "predictive_adequacy": adequacy,
        "evidence_stability": stability.to_dict(),
        "identifiability": identifiability.to_dict(),
    }


def build_report(
    *,
    policy: str = "active",
    max_experiments: int = 3,
    evaluation_seeds: tuple[int, ...] = (1701, 1702, 1703),
) -> dict[str, Any]:
    episodes = [
        _episode(structure, seed, policy=policy, max_experiments=max_experiments)
        for structure in STRUCTURES
        for seed in evaluation_seeds
    ]
    representative = episodes[0]
    correct = sum(bool(row["correct"]) for row in episodes)
    payload = {
        "schema_version": "astrograph-dor-r5-active-discovery-input.v1",
        "structure_posterior": representative["structure_posterior"],
        "truth_quarantine": {
            "truth_channels_allowed": False,
            "held_out_truth_loaded": False,
            "true_structure_loaded": False,
        },
        "truth_channels_allowed": False,
        "predictive_adequacy": representative["predictive_adequacy"],
        "identifiability": representative["identifiability"],
        "evidence_stability": representative["evidence_stability"],
        "hypothesis_space_coverage_complete": True,
    }
    return {
        "schema_version": "astrograph-dor-r5-rw-pilot-report.v1",
        "claim_boundary": "E0 engineering simulation integration evidence only; not dataset, hardware, flight, root-cause or diagnostic performance evidence.",
        "runtime": {"isgn_version": isgn_version},
        "policy": policy,
        "maximum_experiments": max_experiments,
        "provenance": {
            "simulation_model": "dimensionless reaction-wheel response signatures",
            "source_evidence_level": "E0_ENGINEERING_ASSUMPTION",
            "bank_draws_per_particle_experiment": 32,
            "parameter_particles_per_structure": 3,
            "held_out_seed_namespace": "held-out",
            "adcs_v5_2_2_test_partition_accessed": False,
            "phm_v6_0_test_partition_accessed": False,
            "dataset_metric_claimed": False,
        },
        "feature_names": list(FEATURES),
        "candidate_structures": list(STRUCTURES),
        "experiment_catalog": [candidate.to_dict() for candidate in EXPERIMENTS],
        "episodes": episodes,
        "aggregate": {
            "episode_count": len(episodes),
            "evaluation_seed_count": len(evaluation_seeds),
            "top1_correct": correct,
            "top1_accuracy": correct / len(episodes),
            "mean_initial_entropy_bits": float(np.mean([row["initial_entropy_bits"] for row in episodes])),
            "mean_final_entropy_bits": max(
                float(np.mean([row["final_entropy_bits"] for row in episodes])),
                0.0,
            ),
            "mean_experiments_used": float(np.mean([len(row["selected_experiments"]) for row in episodes])),
        },
        "representative_advisory_payload": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", choices=("active", "fixed", "random"), default="active")
    parser.add_argument("--max-experiments", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--evaluation-seed-count", type=int, choices=range(2, 21), default=3)
    args = parser.parse_args()
    seeds = tuple(range(1701, 1701 + args.evaluation_seed_count))
    report = build_report(
        policy=args.policy,
        max_experiments=args.max_experiments,
        evaluation_seeds=seeds,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
