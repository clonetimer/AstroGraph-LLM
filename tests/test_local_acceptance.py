from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.evaluation.local_acceptance import (
    materialize_llm_config,
    parse_top_k_values,
    validate_protocol_root,
    validate_simulation_output_root,
)


def test_parse_top_k_values() -> None:
    assert parse_top_k_values("2,4,6") == (2, 4, 6)
    with pytest.raises(ValueError):
        parse_top_k_values("2,2")
    with pytest.raises(ValueError):
        parse_top_k_values("0")


def test_materialize_llm_config_overrides(tmp_path: Path) -> None:
    template = tmp_path / "template.json"
    template.write_text(json.dumps({
        "base_url": "http://old/v1",
        "model": "old",
        "candidate_top_k": 4,
        "tokenizer_path": "${OLD}",
    }), encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    output = tmp_path / "resolved.json"
    config = materialize_llm_config(
        template_path=template,
        output_path=output,
        top_k=6,
        base_url="http://127.0.0.1:9000/v1/",
        model="Qwen-local",
        api_key="EMPTY",
        tokenizer_path=tokenizer,
        timeout_s=240,
    )
    assert config["candidate_top_k"] == 6
    assert config["base_url"] == "http://127.0.0.1:9000/v1"
    assert config["model"] == "Qwen-local"
    assert config["tokenizer_path"] == str(tokenizer.resolve())
    assert json.loads(output.read_text(encoding="utf-8"))["timeout_s"] == 240.0


def test_validate_simulation_and_protocol_roots(tmp_path: Path) -> None:
    simulation_root = tmp_path / "sim"
    for index in range(2):
        scenario = simulation_root / f"s{index}"
        scenario.mkdir(parents=True)
        (scenario / "telemetry.csv").write_text("t\n0\n", encoding="utf-8")
        (scenario / "run_manifest.json").write_text("{}", encoding="utf-8")
    report = validate_simulation_output_root(simulation_root, expected_count=2)
    assert report["valid_count"] == 2

    protocol_root = tmp_path / "protocol"
    protocol_root.mkdir()
    for name in (
        "dataset_contract.json",
        "fault_observability_matrix.csv",
        "physical_semantics_contract.json",
        "telemetry_dictionary.csv",
    ):
        (protocol_root / name).write_text("{}", encoding="utf-8")
    assert validate_protocol_root(protocol_root)["root"] == str(protocol_root.resolve())
