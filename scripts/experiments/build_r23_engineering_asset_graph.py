from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from astrograph.knowledge.engineering_asset_graph import build_engineering_asset_graph


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.simulation_package) as archive:
        component_registry = json.loads(archive.read("interfaces/ComponentRegistry.json"))
        fault_catalog = json.loads(archive.read("configuration/FaultCatalog.json"))
        signal_registry = json.loads(archive.read("interfaces/SignalRegistry.json"))
    graph = build_engineering_asset_graph(
        source_artifact_sha256=sha256_file(args.simulation_package),
        component_registry=component_registry,
        fault_catalog=fault_catalog,
        signal_registry=signal_registry,
    )
    payload = graph.model_dump(mode="json")
    payload["summary"] = {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "unresolved_fault_component_label_count": len(graph.unresolved_fault_component_labels),
        "unresolved_signal_group_count": len(graph.unresolved_signal_groups),
        "formal_evidence_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
