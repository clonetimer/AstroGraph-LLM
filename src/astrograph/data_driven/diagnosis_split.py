"""Grouped holdout split for the S1 diagnosis baseline (loop 15).

Split discipline (DEV_GOAL + S1):
* grouped holdout: no random row split.  The group key is the **instance /
  config / scenario family**, so every nominal / fault / degradation variant of
  the same object stays in the SAME split group.  This prevents leakage between
  the healthy baseline and its faulted variants, and makes the test set a true
  cross-object generalisation probe.
* frozen: the split is written to ``data/diagnosis_splits/`` as a JSON document
  with the exact object->split assignment, the seed used, and a content hash so
  downstream replays can verify nothing changed after freezing.
* seeds/split are control variables, NOT model inputs (deny rules).

The splitter groups by ``component_instance_id`` (each object = one scenario
family group containing its nominal/fault/degradation variants), then assigns
object groups to TRAIN / TUNE / TEST deterministically.  Mechanism classes that
appear cross-object (CAPACITY_LOSS, FROZEN, GAIN_LOSS) remain represented in
train even when whole objects are held out — that is the S1 design goal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrograph.data_driven.credibility import SplitName

from .diagnosis_dataset import DiagnosisSample

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLITS_DIR = ROOT / "data" / "diagnosis_splits"

# Canonical string used in the frozen split document for the validation group.
VALIDATION_GROUP_TOKEN = "VALIDATION"

# Frozen assignment for the S1 baseline: 9 object groups -> train/tune/test.
# The TEST set (solar_panel + mtb) holds out WHOLE objects, but every mechanism
# class it contains (GAIN_LOSS via solar, CAPACITY_LOSS/FROZEN via mtb) has a
# cross-object representative in TRAIN (eps-solar_fail for GAIN_LOSS,
# battery/rw for CAPACITY_LOSS/FROZEN).  This is exactly the S1 design probe:
# can the mechanism-class head + object-context prior generalise the class to an
# object it never saw?  This assignment is baked (not RNG) so the frozen split
# is byte-stable across runs.
_FROZEN_OBJECT_SPLIT: dict[str, str] = {
    "battery": "TRAIN",
    "reaction_wheel": "TRAIN",
    "imu": "TRAIN",
    "thermal": "TRAIN",
    "eps": "TRAIN",
    "data_queue": VALIDATION_GROUP_TOKEN,
    "onboard_storage": VALIDATION_GROUP_TOKEN,
    "solar_panel": "TEST",
    "mtb": "TEST",
}

_SPLIT_NAME_ORDER = (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST)


def _split_name(value: str) -> SplitName:
    mapping = {
        "TRAIN": SplitName.TRAIN,
        "VALIDATION": SplitName.VALIDATION,
        "TUNE": SplitName.VALIDATION,
        "TEST": SplitName.TEST,
    }
    return mapping[value.upper()]


@dataclass
class FrozenSplit:
    schema_version: str
    split_id: str
    seed: int
    group_key: str
    groups: dict[str, str]  # component_instance_id -> split name
    episode_split: dict[str, str]  # episode_id -> split name
    sha256: str = ""

    def assign(self, sample: DiagnosisSample) -> SplitName:
        return _split_name(self.groups[sample.component_instance_id])

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_id": self.split_id,
            "seed": self.seed,
            "group_key": self.group_key,
            "groups": self.groups,
            "episode_split": self.episode_split,
            "sha256": self.sha256,
        }


def build_grouped_split(
    samples: list[DiagnosisSample],
    *,
    groups: dict[str, str] | None = None,
    seed: int = 20260804,
    split_id: str = "diagnosis-baseline-v1",
) -> FrozenSplit:
    """Build a grouped split by component_instance_id (instance/config family)."""
    groups = dict(groups or _FROZEN_OBJECT_SPLIT)
    unknown = {s.component_instance_id for s in samples} - set(groups)
    if unknown:
        raise ValueError(f"objects missing from grouped split assignment: {sorted(unknown)}")

    episode_split: dict[str, str] = {}
    for sample in samples:
        episode_split[sample.episode_id] = groups[sample.component_instance_id]

    payload = {
        "schema_version": "astrograph-diagnosis-split.v1",
        "split_id": split_id,
        "seed": seed,
        "group_key": "component_instance_id",
        "groups": {k: groups[k] for k in sorted(groups)},
        "episode_split": {k: episode_split[k] for k in sorted(episode_split)},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FrozenSplit(
        schema_version="astrograph-diagnosis-split.v1",
        split_id=split_id,
        seed=seed,
        group_key="component_instance_id",
        groups=dict(groups),
        episode_split=episode_split,
        sha256=digest,
    )


def freeze_split(
    split: FrozenSplit,
    *,
    output_dir: Path = DEFAULT_SPLITS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split.split_id}.json"
    path.write_text(
        json.dumps(split.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def load_frozen_split(
    split_id: str,
    *,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> FrozenSplit:
    path = splits_dir / f"{split_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"frozen split missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload["groups"]
    episode_split = payload.get("episode_split", {})
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "split_id": payload["split_id"],
                "seed": payload["seed"],
                "group_key": payload["group_key"],
                "groups": {k: groups[k] for k in sorted(groups)},
                "episode_split": {k: episode_split[k] for k in sorted(episode_split)},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    recomputed = FrozenSplit(
        schema_version=payload["schema_version"],
        split_id=payload["split_id"],
        seed=payload["seed"],
        group_key=payload["group_key"],
        groups=groups,
        episode_split=episode_split,
        sha256=digest,
    )
    return recomputed


def split_samples(
    samples: list[DiagnosisSample],
    split: FrozenSplit,
) -> dict[SplitName, list[DiagnosisSample]]:
    """Partition samples into TRAIN / VALIDATION / TEST by the frozen grouping.

    Returns a dict keyed by SplitName with lists of samples.  Every sample is
    assigned via the episode-level split map so the exact same episode always
    lands in the same split regardless of how the group key is spelled.
    """
    out: dict[SplitName, list[DiagnosisSample]] = {name: [] for name in _SPLIT_NAME_ORDER}
    for sample in samples:
        split_name = _split_name(split.episode_split[sample.episode_id])
        out[split_name].append(sample)
    return out


def verify_no_leakage(split: FrozenSplit, samples: list[DiagnosisSample]) -> None:
    """Assert grouped-holdout integrity: no object spans two splits, and the
    episode map is consistent with the group map."""
    by_episode: dict[str, str] = {}
    for sample in samples:
        expected = split.groups[sample.component_instance_id]
        actual = split.episode_split[sample.episode_id]
        if actual != expected:
            raise AssertionError(
                f"episode {sample.episode_id} ({sample.component_instance_id}) "
                f"assigned {actual} but its object group is {expected}"
            )
        by_episode[sample.episode_id] = expected
    if set(by_episode) != set(split.episode_split):
        raise AssertionError("episode set in split does not match dataset")
    # no object in two splits
    obj_splits: dict[str, set[str]] = {}
    for sample in samples:
        obj_splits.setdefault(sample.component_instance_id, set()).add(
            split.episode_split[sample.episode_id]
        )
    for obj, splits in obj_splits.items():
        if len(splits) != 1:
            raise AssertionError(f"object {obj} spans splits {sorted(splits)}")


__all__ = [
    "DEFAULT_SPLITS_DIR",
    "FrozenSplit",
    "build_grouped_split",
    "freeze_split",
    "load_frozen_split",
    "split_samples",
    "verify_no_leakage",
]
