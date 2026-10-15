from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def parse_top_k_values(raw: str | Iterable[int]) -> tuple[int, ...]:
    """Parse and validate a deterministic, duplicate-free Top-K sequence."""
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        try:
            values = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid --top-k value: {raw!r}") from exc
    else:
        values = [int(value) for value in raw]
    if not values:
        raise ValueError("At least one Top-K value is required.")
    if any(value <= 0 for value in values):
        raise ValueError("Top-K values must be positive integers.")
    if len(set(values)) != len(values):
        raise ValueError("Top-K values must not contain duplicates.")
    return tuple(values)


def resolve_tokenizer_path(explicit_path: Path | None) -> Path:
    raw = explicit_path or os.environ.get("ASTROGRAPH_QWEN_TOKENIZER_PATH")
    if not raw:
        raise ValueError(
            "Qwen tokenizer path is required. Pass --tokenizer-path or set "
            "ASTROGRAPH_QWEN_TOKENIZER_PATH."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Tokenizer path does not exist or is not a directory: {path}")
    return path


def validate_simulation_output_root(root: Path, *, expected_count: int = 21) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Simulation output root does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for scenario_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        telemetry = scenario_dir / "telemetry.csv"
        manifest = scenario_dir / "run_manifest.json"
        rows.append({
            "scenario": scenario_dir.name,
            "telemetry_exists": telemetry.is_file(),
            "manifest_exists": manifest.is_file(),
            "valid": telemetry.is_file() and manifest.is_file(),
        })
    valid_count = sum(row["valid"] for row in rows)
    if len(rows) != expected_count or valid_count != expected_count:
        invalid = [row["scenario"] for row in rows if not row["valid"]]
        raise ValueError(
            f"Expected {expected_count} valid scenario directories under {root}, "
            f"found {len(rows)} directories and {valid_count} valid scenarios. "
            f"Invalid scenarios: {invalid}"
        )
    return {
        "root": str(root),
        "scenario_count": len(rows),
        "valid_count": valid_count,
        "rows": rows,
    }


def validate_protocol_root(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    required = (
        "dataset_contract.json",
        "fault_observability_matrix.csv",
        "physical_semantics_contract.json",
        "telemetry_dictionary.csv",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if not root.is_dir() or missing:
        raise ValueError(f"Invalid protocol root {root}; missing files: {missing}")
    return {"root": str(root), "required_files": list(required)}


def materialize_llm_config(
    *,
    template_path: Path,
    output_path: Path,
    top_k: int,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    tokenizer_path: Path | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    template_path = template_path.expanduser().resolve()
    if not template_path.is_file():
        raise ValueError(f"LLM config template does not exist: {template_path}")
    config = json.loads(template_path.read_text(encoding="utf-8-sig"))
    config["candidate_top_k"] = int(top_k)
    if base_url:
        config["base_url"] = base_url.rstrip("/")
    if model:
        config["model"] = model
    if api_key is not None:
        config["api_key"] = api_key
    if tokenizer_path is not None:
        config["tokenizer_path"] = str(tokenizer_path.expanduser().resolve())
        config["tokenizer_mode"] = "required"
        config["tokenizer_local_files_only"] = True
    if timeout_s is not None:
        config["timeout_s"] = float(timeout_s)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config
