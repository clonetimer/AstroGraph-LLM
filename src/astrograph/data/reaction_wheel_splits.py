from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DEFAULT_VALIDATION_GROUPS = (0, 4, 7, 11)
DEFAULT_TEST_GROUPS = (2, 12, 16, 23)
DEFAULT_HOLDOUT_GROUPS = (1, 3, 8)


def build_d3_pair_group_split(
    package_root: str | Path,
    output: str | Path,
    *,
    validation_groups: tuple[int, ...] = DEFAULT_VALIDATION_GROUPS,
    test_groups: tuple[int, ...] = DEFAULT_TEST_GROUPS,
    holdout_groups: tuple[int, ...] = DEFAULT_HOLDOUT_GROUPS,
) -> dict:
    root = Path(package_root)
    audit_root = root / "03_run_audit"
    manifests: dict[str, dict] = {}
    params: dict[str, dict] = {}
    for directory in audit_root.iterdir():
        if directory.is_dir() and (directory / "run_manifest.json").exists():
            manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8-sig"))
            run_id = str(manifest["run_id"])
            manifests[run_id] = manifest
            params[run_id] = json.loads((directory / "parameter_sample.json").read_text(encoding="utf-8-sig"))

    repeats = sorted({int(item["repeat_index"]) for item in manifests.values()})
    reserved = set(validation_groups) | set(test_groups) | set(holdout_groups)
    train_groups = sorted(set(repeats) - reserved)
    group_split = {
        **{index: "train" for index in train_groups},
        **{index: "validation" for index in validation_groups},
        **{index: "test" for index in test_groups},
        **{index: "holdout" for index in holdout_groups},
    }
    if set(group_split) != set(repeats):
        raise ValueError("D3 split repeat groups must cover every repeat exactly once")
    run_split = {run_id: group_split[int(manifest["repeat_index"])] for run_id, manifest in manifests.items()}

    cross_pairs = [
        run_id
        for run_id, manifest in manifests.items()
        if manifest["case_type"] == "fault"
        and run_split[run_id] != run_split[str(manifest["paired_nominal_run_id"])]
    ]
    if cross_pairs:
        raise ValueError(f"D3 pair-group leakage remains: {cross_pairs[:5]}")

    producer = json.loads((root / "02_dataset" / "dataset_split_manifest.json").read_text(encoding="utf-8-sig"))
    producer_lookup: dict[str, str] = {}
    for key, split in (
        ("train_run_ids", "train"),
        ("validation_run_ids", "validation"),
        ("test_run_ids", "test"),
        ("zero_shot_validation_run_ids", "zero_shot_validation"),
    ):
        for run_id in producer.get(key, []):
            producer_lookup[str(run_id)] = split
    producer_cross = sum(
        1
        for run_id, manifest in manifests.items()
        if manifest["case_type"] == "fault"
        and producer_lookup.get(run_id) != producer_lookup.get(str(manifest["paired_nominal_run_id"]))
    )

    friction_by_repeat: dict[int, dict] = {}
    for run_id, manifest in manifests.items():
        if manifest["case_id"] == "rw_friction_increase":
            sampled = params[run_id].get("sampled_values", {})
            friction_by_repeat[int(manifest["repeat_index"])] = {
                "severity_level": str(sampled.get("severity_level", "")),
                "wheel_index": int(sampled.get("wheel_index", -1)),
            }

    def coverage(group_ids: list[int]) -> dict:
        return {
            "repeat_indices": sorted(group_ids),
            "severity_levels": sorted({friction_by_repeat[index]["severity_level"] for index in group_ids}),
            "wheel_indices": sorted({friction_by_repeat[index]["wheel_index"] for index in group_ids}),
            "severity_wheel_pairs": sorted(
                {
                    f"{friction_by_repeat[index]['severity_level']}|{friction_by_repeat[index]['wheel_index']}"
                    for index in group_ids
                }
            ),
        }

    document = {
        "schema_version": "astrograph-d3-pair-group-split.v1",
        "source_dataset_id": "phm_d3_rw_production_v2_20260718",
        "strategy": "paired_repeat_group_coverage_optimized_four_way",
        "seed_reference": 20260718,
        "claim": (
            "AstroGraph-derived split keeps each nominal run and its two paired fault runs in one split. "
            f"Producer split crosses {producer_cross} fault/paired-nominal boundaries."
        ),
        "split_counts": dict(Counter(run_split.values())),
        "run_ids": {
            split: sorted(run_id for run_id, assigned in run_split.items() if assigned == split)
            for split in ("train", "validation", "test", "holdout")
        },
        "repeat_groups": {
            split: sorted(index for index, assigned in group_split.items() if assigned == split)
            for split in ("train", "validation", "test", "holdout")
        },
        "coverage": {
            split: coverage([index for index, assigned in group_split.items() if assigned == split])
            for split in ("train", "validation", "test", "holdout")
        },
        "pair_cross_split_count": 0,
        "producer_pair_cross_split_count": producer_cross,
        "holdout_semantics": "Known-class holdout; not true unseen-fault zero-shot.",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document
