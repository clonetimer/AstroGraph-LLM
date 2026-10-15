#!/usr/bin/env python3
"""Materialize a small, labeled, groupable, replayable episode dataset (G2).

This is the project-owner-approved "物化带标签 episode" work item.  It turns a
curated set of real Sat-Agent source traces into (``EpisodeManifest``,
``LabeledEpisode``) pairs:

* The public ``EpisodeManifest`` is built from the *same* source trace by the
  generic episode builder, which selects ONLY ``time_s`` + channel-map bound
  fields — the manifest stays label-free (``labels_used``/``truth_used`` are
  ``False`` and frozen).
* Labels (health state, fault ids, degradation flags, scenario family) are
  extracted through the independent label adapter into a separate
  ``LabeledEpisode`` record, linked to the manifest by ``episode_id``.  The
  label record is NOT a model input.

S0 discipline: this script performs NO fitting, NO calibration and creates NO
formal train/validation split or seed.  It only materialises episodes so that
grouped holdout can be defined later (an S1-authorized step).

Inputs are read from the Sat-Agent platform (read-only).  Outputs go to::

    data/labeled_episodes/
        catalog.json                # LabeledEpisodeCatalog (dataset index)
        manifests/<episode_id>.json # EpisodeManifest (label-free)
        labels/<episode_id>.json    # LabeledEpisode (labels + grouping)

The curated source map covers five objects spanning five archetypes, each with
a nominal / fault / degradation triple:

    imu (sensor.measurement), mtb (actuator.controlled_effort),
    pdu (converter.power), data-queue (queue.bounded_inventory),
    heater (thermal.lumped_capacity)

Run with the AstroGraph interpreter only::

    /d/Project/test_lunwen/.venv/Scripts/python.exe scripts/materialize_labeled_episodes.py

The source root may be overridden with ``--sat-root`` or the
``ASTROGRAPH_SAT_AGENT_ROOT`` environment variable; the output dir with
``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from astrograph.windowing.label_adapter import materialize_labeled_episodes

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "labeled_episodes"

DEFAULT_SAT_AGENT_ROOT = r"D:/Project/test_sat/sat_agent_platform"

# Curated real source traces.  Each profile_id maps to a scenario family ->
# relative (to the Sat-Agent root) source path.  All paths confirmed present.
DEFAULT_LABELED_SOURCES: dict[str, dict[str, str]] = {
    "sat-agent.component.imu.trace.v1": {
        "nominal": (
            "reports/vllm_nl_e2e_matrix_en_batch_039/evidence/cases/"
            "component_imu__nominal__en/runs/trace.csv"
        ),
        "fault": (
            "reports/vllm_nl_e2e_matrix_alias_batch_039/evidence/cases/"
            "component_imu__fault__alias/runs/trace.csv"
        ),
        "degradation": (
            "reports/vllm_nl_e2e_matrix_alias_batch_039/evidence/cases/"
            "component_imu__degradation__alias/runs/trace.csv"
        ),
    },
    "sat-agent.component.mtb.trace.v1": {
        "nominal": (
            "reports/vllm_nl_e2e_matrix_alias_batch_051/evidence/cases/"
            "component_mtb__nominal__alias/runs/trace.csv"
        ),
        "fault": (
            "reports/vllm_nl_e2e_matrix_alias_batch_051/evidence/cases/"
            "component_mtb__fault__alias/runs/trace.csv"
        ),
        "degradation": (
            "reports/vllm_nl_e2e_matrix_alias_batch_051/evidence/cases/"
            "component_mtb__degradation__alias/runs/trace.csv"
        ),
    },
    "sat-agent.component.pdu.trace.v1": {
        "nominal": (
            "reports/vllm_nl_e2e_matrix_en_batch_063/evidence/cases/"
            "component_pdu__nominal__en/runs/trace.csv"
        ),
        "fault": (
            "reports/vllm_nl_e2e_matrix_alias_batch_063/evidence/cases/"
            "component_pdu__fault__alias/runs/trace.csv"
        ),
        "degradation": (
            "reports/vllm_nl_e2e_matrix_alias_batch_063/evidence/cases/"
            "component_pdu__degradation__alias/runs/trace.csv"
        ),
    },
    "sat-agent.component.data-queue.trace.v1": {
        "nominal": (
            "reports/vllm_nl_e2e_matrix_alias_batch_027/evidence/cases/"
            "component_data_queue__nominal__alias/runs/trace.csv"
        ),
        "fault": (
            "reports/vllm_nl_e2e_matrix_alias_batch_027/evidence/cases/"
            "component_data_queue__fault__alias/runs/trace.csv"
        ),
        "degradation": (
            "reports/vllm_nl_e2e_matrix_alias_batch_027/evidence/cases/"
            "component_data_queue__degradation__alias/runs/trace.csv"
        ),
    },
    "sat-agent.component.heater.trace.v1": {
        "nominal": (
            "reports/vllm_nl_e2e_matrix_alias_batch_039/evidence/cases/"
            "component_heater__nominal__alias/runs/trace.csv"
        ),
        "fault": (
            "reports/vllm_nl_e2e_matrix_alias_batch_039/evidence/cases/"
            "component_heater__fault__alias/runs/trace.csv"
        ),
        "degradation": (
            "reports/vllm_nl_e2e_matrix_alias_batch_039/evidence/cases/"
            "component_heater__degradation__alias/runs/trace.csv"
        ),
    },
}


def _stats(catalog, output_dir: Path) -> dict:
    episodes = catalog.episodes
    by_scenario: dict[str, int] = {}
    by_config: dict[str, list[str]] = {}
    by_archetype: dict[str, list[str]] = {}
    labeled = {"nominal": 0, "fault": 0, "degradation": 0}
    for ep in episodes:
        fam = ep.grouping.scenario_family
        by_scenario[fam] = by_scenario.get(fam, 0) + 1
        by_config.setdefault(ep.grouping.config_id, []).append(ep.episode_id)
        by_archetype.setdefault(ep.grouping.component_archetype_id, []).append(ep.episode_id)
        labeled[fam] = labeled.get(fam, 0) + 1
    manifest_files = len(list((output_dir / "manifests").glob("*.json")))
    label_files = len(list((output_dir / "labels").glob("*.json")))
    return {
        "episode_count": len(episodes),
        "object_count": len({ep.grouping.component_instance_id for ep in episodes}),
        "by_scenario_family": by_scenario,
        "by_config": {k: len(v) for k, v in by_config.items()},
        "by_archetype": {k: len(v) for k, v in by_archetype.items()},
        "manifest_files": manifest_files,
        "label_files": label_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sat-root",
        default=None,
        help="Sat-Agent platform root (read-only). Defaults to $ASTROGRAPH_SAT_AGENT_ROOT or the canonical dev path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dataset directory. Defaults to data/labeled_episodes.",
    )
    parser.add_argument(
        "--source-map",
        default=None,
        help="Optional JSON file overriding the profile->{scenario->relative path} source map.",
    )
    args = parser.parse_args(argv)

    sat_root = Path(args.sat_root) if args.sat_root else Path(
        os.environ.get("ASTROGRAPH_SAT_AGENT_ROOT", DEFAULT_SAT_AGENT_ROOT)
    ).expanduser().resolve()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    source_map = (
        json.loads(Path(args.source_map).read_text(encoding="utf-8"))
        if args.source_map
        else DEFAULT_LABELED_SOURCES
    )

    catalog = materialize_labeled_episodes(
        profile_sources=source_map,
        profile_path=ROOT / "configs/data_sources/sat_agent_platform_schema_profiles.s0.v1.json",
        channel_map_catalog_path=ROOT / "configs/semantics/sat_agent_platform_channel_maps.s0.v1.json",
        semantic_schema_path=ROOT / "configs/semantics/core_semantic_schema.json",
        output_dir=output_dir,
        sat_root=sat_root,
        catalog_id="sat-agent-labeled-episodes-v1",
    )

    report = {
        "catalog_path": str(output_dir / "catalog.json"),
        "stats": _stats(catalog, output_dir),
        "claim": catalog.claim_boundary,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
