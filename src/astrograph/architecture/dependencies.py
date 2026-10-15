from __future__ import annotations

import ast
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class LayerRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1)
    path_prefix: str = Field(min_length=1)
    forbidden_import_prefixes: tuple[str, ...] = ()


class StableLayerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-stable-layer-policy.v1"
    rules: tuple[LayerRule, ...]


class LayerViolation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    path: str
    imported_module: str
    line: int


class StableLayerReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-stable-layer-report.v1"
    status: str
    scanned_file_count: int
    violations: tuple[LayerViolation, ...]


def load_stable_layer_policy(path: str | Path) -> StableLayerPolicy:
    return StableLayerPolicy.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result.append((node.module, node.lineno))
    return result


def scan_stable_layers(root: str | Path, policy: StableLayerPolicy) -> StableLayerReport:
    root = Path(root).resolve()
    violations: list[LayerViolation] = []
    scanned: set[Path] = set()
    for rule in policy.rules:
        base = (root / rule.path_prefix).resolve()
        if not base.exists():
            violations.append(
                LayerViolation(
                    rule_id=rule.rule_id,
                    path=rule.path_prefix,
                    imported_module="<PATH_MISSING>",
                    line=0,
                )
            )
            continue
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for path in files:
            scanned.add(path)
            for imported, line in _imports(path):
                if any(
                    imported == prefix or imported.startswith(prefix + ".")
                    for prefix in rule.forbidden_import_prefixes
                ):
                    violations.append(
                        LayerViolation(
                            rule_id=rule.rule_id,
                            path=path.relative_to(root).as_posix(),
                            imported_module=imported,
                            line=line,
                        )
                    )
    return StableLayerReport(
        status="PASS" if not violations else "FAIL",
        scanned_file_count=len(scanned),
        violations=tuple(violations),
    )
