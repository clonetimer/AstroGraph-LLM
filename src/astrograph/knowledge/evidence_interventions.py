
"""Outcome-blind, provenance-blocked evidence coverage interventions."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import random
from typing import Iterable, Mapping, Sequence

@dataclass(frozen=True)
class EvidenceBlock:
    block_id: str
    fault_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    provenance_independence_group: str

    def __post_init__(self) -> None:
        if not self.block_id or not self.provenance_independence_group:
            raise ValueError("block_id and provenance_independence_group required")
        if not self.fault_ids or not self.evidence_ids:
            raise ValueError("block must contain faults and evidence")

@dataclass(frozen=True)
class InterventionPoint:
    target_fraction: float
    selected_block_ids: tuple[str, ...]
    achieved_fault_fraction: float
    selected_fault_ids: tuple[str, ...]

@dataclass(frozen=True)
class InterventionSchedule:
    schema: str
    universe: tuple[str, ...]
    replicate_id: int
    points: tuple[InterventionPoint, ...]
    schedule_sha256: str


def _canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

def build_nested_schedule(
    universe: Sequence[str],
    blocks: Sequence[EvidenceBlock],
    target_fractions: Sequence[float],
    *,
    replicate_id: int,
    design_seed_sha256: str,
) -> InterventionSchedule:
    universe_tuple = tuple(sorted(set(universe)))
    if not universe_tuple:
        raise ValueError("universe cannot be empty")
    if any(not 0 <= x <= 1 for x in target_fractions):
        raise ValueError("target fractions must be in [0, 1]")
    targets = tuple(sorted(set(float(x) for x in target_fractions)))
    if len(design_seed_sha256) != 64:
        raise ValueError("design_seed_sha256 must be SHA-256 hex")
    # Group entire shared provenance together; a source family is never split into
    # pseudo-independent rules.
    by_group: dict[str, list[EvidenceBlock]] = {}
    for block in blocks:
        by_group.setdefault(block.provenance_independence_group, []).append(block)
    groups = sorted(by_group)
    seed_material = f"{design_seed_sha256}:{replicate_id}".encode()
    rng = random.Random(int(hashlib.sha256(seed_material).hexdigest(), 16))
    rng.shuffle(groups)
    # Deterministic round-robin preference for groups that add currently uncovered
    # faults; no outcomes, labels, accuracy, or test groups are inputs.
    selected_groups: list[str] = []
    selected_faults: set[str] = set()
    remaining = list(groups)
    order: list[str] = []
    while remaining:
        gains = []
        for g in remaining:
            faults = {f for b in by_group[g] for f in b.fault_ids if f in universe_tuple}
            gains.append((len(faults - selected_faults), g, faults))
        max_gain = max(x[0] for x in gains)
        candidates = [x for x in gains if x[0] == max_gain]
        _, chosen, faults = candidates[0]
        order.append(chosen)
        selected_faults |= faults
        remaining.remove(chosen)
    points = []
    for target in targets:
        chosen_groups = []
        covered: set[str] = set()
        required = math.ceil(target * len(universe_tuple) - 1e-12)
        for g in order:
            if len(covered) >= required:
                break
            chosen_groups.append(g)
            covered |= {f for b in by_group[g] for f in b.fault_ids if f in universe_tuple}
        block_ids = sorted(b.block_id for g in chosen_groups for b in by_group[g])
        points.append(InterventionPoint(
            target_fraction=target,
            selected_block_ids=tuple(block_ids),
            achieved_fault_fraction=len(covered) / len(universe_tuple),
            selected_fault_ids=tuple(sorted(covered)),
        ))
    payload = {
        "schema": "astrograph-evidence-intervention-schedule.v1",
        "universe": list(universe_tuple),
        "replicate_id": replicate_id,
        "points": [asdict(p) for p in points],
    }
    return InterventionSchedule(
        schema=payload["schema"], universe=universe_tuple, replicate_id=replicate_id,
        points=tuple(points), schedule_sha256=_canonical_sha(payload),
    )

def validate_nested_schedule(schedule: InterventionSchedule) -> None:
    previous_blocks: set[str] = set()
    previous_faults: set[str] = set()
    previous_target = -1.0
    for point in schedule.points:
        blocks = set(point.selected_block_ids)
        faults = set(point.selected_fault_ids)
        if point.target_fraction < previous_target:
            raise ValueError("targets are not sorted")
        if not previous_blocks.issubset(blocks) or not previous_faults.issubset(faults):
            raise ValueError("intervention schedule is not nested")
        previous_blocks, previous_faults = blocks, faults
        previous_target = point.target_fraction
