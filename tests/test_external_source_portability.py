from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import pytest

from astrograph.data import ExternalSourceRegistry, ExternalSourceResolutionError


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "configs/datasets/adcs_v5_2_2"
REGISTRY_PATH = DATASET_ROOT / "external_source_registry.json"
SOURCE_URI_RE = re.compile(r"^source://([a-z0-9][a-z0-9._-]*)$")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_active_adcs_dataset_configs_do_not_embed_host_absolute_paths() -> None:
    for path in sorted(DATASET_ROOT.glob("*.json")):
        for value in _strings(_load(path)):
            assert not value.startswith("/"), (path, value)
            assert not WINDOWS_ABSOLUTE_RE.match(value), (path, value)


def test_all_external_source_uris_are_registered_and_portable() -> None:
    registry = _load(REGISTRY_PATH)
    records = {item["source_id"]: item for item in registry["sources"]}
    assert records

    referenced_ids: set[str] = set()
    for path in sorted(DATASET_ROOT.glob("*.json")):
        if path == REGISTRY_PATH:
            continue
        for value in _strings(_load(path)):
            match = SOURCE_URI_RE.match(value)
            if match:
                referenced_ids.add(match.group(1))

    assert referenced_ids == set(records)
    for source_id, record in records.items():
        assert record["logical_uri"] == f"source://{source_id}"
        resolver = record["resolver"]
        assert resolver["root_env"].startswith("ASTROGRAPH_")
        relative = Path(resolver["relative_path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        integrity = record["integrity"]
        assert integrity["status"] in {"HASH_FROZEN", "UNVERIFIED_EXTERNAL_SOURCE"}
        if integrity["status"] == "HASH_FROZEN":
            assert re.fullmatch(r"[0-9a-f]{64}", integrity["sha256"] or "")
        else:
            assert integrity["sha256"] is None


def test_registry_resolves_configured_external_source_without_host_path_leak(tmp_path: Path) -> None:
    registry = ExternalSourceRegistry.from_json(REGISTRY_PATH)
    generator = tmp_path / "sat" / "scripts" / "generate_whole_spacecraft_adcs_eps_dataset_v522.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("print('fixture')\n", encoding="utf-8")

    resolved = registry.resolve(
        "source://sat-agent.adcs-v5_2_2-generator",
        environ={"ASTROGRAPH_SAT_AGENT_ROOT": str(tmp_path / "sat")},
    )

    assert resolved.path == generator.resolve()
    assert resolved.hash_verified is False
    assert len(resolved.sha256 or "") == 64


def test_unverified_external_source_cannot_satisfy_hash_frozen_mode(tmp_path: Path) -> None:
    registry = ExternalSourceRegistry.from_json(REGISTRY_PATH)
    generator = tmp_path / "sat" / "scripts" / "generate_whole_spacecraft_adcs_eps_dataset_v522.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("print('fixture')\n", encoding="utf-8")

    with pytest.raises(ExternalSourceResolutionError, match="lacks a frozen integrity receipt"):
        registry.resolve(
            "source://sat-agent.adcs-v5_2_2-generator",
            environ={"ASTROGRAPH_SAT_AGENT_ROOT": str(tmp_path / "sat")},
            require_hash_frozen=True,
        )


def test_external_source_resolution_fails_closed_when_root_is_unconfigured() -> None:
    registry = ExternalSourceRegistry.from_json(REGISTRY_PATH)
    with pytest.raises(ExternalSourceResolutionError, match="ASTROGRAPH_SAT_AGENT_ROOT is required"):
        registry.resolve(
            "source://sat-agent.adcs-v5_2_2-generator",
            environ={},
        )
