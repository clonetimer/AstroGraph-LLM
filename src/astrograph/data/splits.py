from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _assign(items: list, counts: tuple[int, int, int, int], seed: int) -> dict[str, list]:
    names = ("train", "validation", "test", "holdout")
    rng = random.Random(seed + len(items) * 31 + sum(counts))
    items = list(items)
    rng.shuffle(items)
    result: dict[str, list] = {name: [] for name in names}
    pos = 0
    for name, count in zip(names, counts):
        result[name] = items[pos : pos + count]
        pos += count
    if pos != len(items):
        raise ValueError("Split counts do not consume all items")
    return result


def build_d1_pair_group_split(package_root: str | Path, output: str | Path, *, seed: int = 20260718) -> dict:
    root = Path(package_root)
    audit = root / "03_run_audit"
    manifests: dict[str, dict] = {}
    for directory in audit.iterdir():
        if directory.is_dir() and len(directory.name) == 32:
            manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8-sig"))
            manifests[str(manifest["run_id"])] = manifest

    nominal_by_repeat = {
        int(m["repeat_index"]): run_id for run_id, m in manifests.items() if m["case_type"] == "nominal"
    }
    fault_groups: dict[int, list[str]] = defaultdict(list)
    for run_id, manifest in manifests.items():
        if manifest["case_type"] == "fault":
            fault_groups[int(manifest["repeat_index"])].append(run_id)

    run_split: dict[str, str] = {}
    group_records: list[dict] = []
    paired = _assign(sorted(fault_groups), (19, 4, 4, 3), seed)
    for split, repeat_indices in paired.items():
        for repeat_index in repeat_indices:
            members = [nominal_by_repeat[repeat_index], *sorted(fault_groups[repeat_index])]
            for run_id in members:
                run_split[run_id] = split
            group_records.append(
                {
                    "group_id": f"paired_repeat_{repeat_index:02d}",
                    "split": split,
                    "repeat_index": repeat_index,
                    "member_run_ids": members,
                }
            )

    nominal_only = sorted(set(nominal_by_repeat) - set(fault_groups))
    for split, repeat_indices in _assign(nominal_only, (12, 3, 3, 2), seed).items():
        for repeat_index in repeat_indices:
            run_id = nominal_by_repeat[repeat_index]
            run_split[run_id] = split
            group_records.append(
                {
                    "group_id": f"nominal_only_repeat_{repeat_index:02d}",
                    "split": split,
                    "repeat_index": repeat_index,
                    "member_run_ids": [run_id],
                }
            )

    modes = sorted({m["case_id"] for m in manifests.values() if m["case_type"] == "degradation"})
    for mode in modes:
        run_ids = sorted(
            run_id
            for run_id, manifest in manifests.items()
            if manifest["case_type"] == "degradation" and manifest["case_id"] == mode
        )
        for split, members in _assign(run_ids, (19, 4, 4, 3), seed).items():
            for run_id in members:
                run_split[run_id] = split

    if len(run_split) != len(manifests):
        raise ValueError(f"Expected {len(manifests)} assigned runs, got {len(run_split)}")
    cross_split_pairs = [
        run_id
        for run_id, manifest in manifests.items()
        if manifest["case_type"] == "fault"
        and run_split[run_id] != run_split[str(manifest["paired_nominal_run_id"])]
    ]
    if cross_split_pairs:
        raise ValueError(f"Pair-group split leakage remains: {cross_split_pairs[:5]}")

    document = {
        "schema_version": "astrograph-leak-safe-split.v1",
        "source_dataset_id": "phm_d1_production_v1_20260718",
        "strategy": "paired_nominal_group_stratified_four_way",
        "seed": seed,
        "claim": (
            "AstroGraph-derived split. The producer split is run_id-disjoint but places "
            "91 of 150 fault-paired-nominal groups across different splits."
        ),
        "split_counts": dict(Counter(run_split.values())),
        "run_ids": {
            split: sorted(run_id for run_id, assigned in run_split.items() if assigned == split)
            for split in ("train", "validation", "test", "holdout")
        },
        "paired_groups": group_records,
        "pair_cross_split_count": 0,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document
