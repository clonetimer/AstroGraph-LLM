from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpecializationPattern(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern_id: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    description: str = Field(min_length=1)


class CompatibilityPath(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class ArchitectureBoundaryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-architecture-boundary-policy.v1"]
    scan_roots: list[str]
    generic_paths: list[str]
    compatibility_allowlist: list[CompatibilityPath]
    patterns: list[SpecializationPattern]

    @model_validator(mode="after")
    def unique_and_disjoint(self) -> "ArchitectureBoundaryPolicy":
        for field_name in ("scan_roots", "generic_paths"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique")
        compatibility = [item.path for item in self.compatibility_allowlist]
        if len(compatibility) != len(set(compatibility)):
            raise ValueError("compatibility_allowlist paths must be unique")
        if set(compatibility).intersection(self.generic_paths):
            raise ValueError("generic paths cannot be compatibility allowlisted")
        pattern_ids = [item.pattern_id for item in self.patterns]
        if len(pattern_ids) != len(set(pattern_ids)):
            raise ValueError("pattern IDs must be unique")
        for item in self.patterns:
            re.compile(item.expression)
        return self


class BoundaryScanIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    path: str
    pattern_id: str = ""
    value: str = ""
    line: int | None = None


class BoundaryScanReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-architecture-boundary-report.v1"] = (
        "astrograph-architecture-boundary-report.v1"
    )
    status: Literal["PASS", "FAIL"]
    scanned_file_count: int
    generic_match_count: int
    compatibility_match_count: int
    issues: list[BoundaryScanIssue]


def load_boundary_policy(path: str | Path) -> ArchitectureBoundaryPolicy:
    return ArchitectureBoundaryPolicy.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _json_strings(value: Any) -> list[tuple[str, int | None]]:
    found: list[tuple[str, int | None]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.append((str(key), None))
            found.extend(_json_strings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_json_strings(child))
    elif isinstance(value, str):
        found.append((value, None))
    return found


def _file_strings(path: Path) -> list[tuple[str, int | None]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        tree = ast.parse(text, filename=str(path))
        found = [
            (node.value, getattr(node, "lineno", None))
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                found.append((node.id, getattr(node, "lineno", None)))
            elif isinstance(node, ast.Attribute):
                parts = [node.attr]
                value = node.value
                while isinstance(value, ast.Attribute):
                    parts.append(value.attr)
                    value = value.value
                if isinstance(value, ast.Name):
                    parts.append(value.id)
                    found.append((".".join(reversed(parts)), getattr(node, "lineno", None)))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", None)
                if module:
                    found.append((module, getattr(node, "lineno", None)))
                found.extend((alias.name, getattr(node, "lineno", None)) for alias in node.names)
        return found
    if path.suffix == ".json":
        return _json_strings(json.loads(text))
    return []


def scan_architecture_boundaries(
    root: str | Path,
    policy: ArchitectureBoundaryPolicy,
) -> BoundaryScanReport:
    root = Path(root).resolve()
    generic = set(policy.generic_paths)
    compatibility = {item.path for item in policy.compatibility_allowlist}
    compiled = [(item, re.compile(item.expression)) for item in policy.patterns]
    issues: list[BoundaryScanIssue] = []
    generic_matches = 0
    compatibility_matches = 0

    for relative in sorted(generic | compatibility):
        if not (root / relative).is_file():
            issues.append(BoundaryScanIssue(code="DECLARED_BOUNDARY_PATH_MISSING", path=relative))

    files: list[Path] = []
    for scan_root in policy.scan_roots:
        base = root / scan_root
        if not base.is_dir():
            issues.append(BoundaryScanIssue(code="SCAN_ROOT_MISSING", path=scan_root))
            continue
        files.extend(path for path in base.rglob("*") if path.is_file() and path.suffix in {".py", ".json"})

    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix()
        try:
            values = _file_strings(path)
        except (OSError, SyntaxError, json.JSONDecodeError) as exc:
            issues.append(BoundaryScanIssue(code="BOUNDARY_FILE_PARSE_FAILED", path=relative, value=str(exc)))
            continue
        for value, line in values:
            for pattern, expression in compiled:
                match = expression.search(value)
                if match is None:
                    continue
                matched = match.group(0)
                if relative in generic:
                    generic_matches += 1
                    issues.append(BoundaryScanIssue(
                        code="GENERIC_CORE_SPECIALIZATION",
                        path=relative,
                        pattern_id=pattern.pattern_id,
                        value=matched,
                        line=line,
                    ))
                elif relative in compatibility:
                    compatibility_matches += 1
                else:
                    issues.append(BoundaryScanIssue(
                        code="UNDECLARED_SPECIALIZATION",
                        path=relative,
                        pattern_id=pattern.pattern_id,
                        value=matched,
                        line=line,
                    ))

    return BoundaryScanReport(
        status="PASS" if not issues else "FAIL",
        scanned_file_count=len(set(files)),
        generic_match_count=generic_matches,
        compatibility_match_count=compatibility_matches,
        issues=issues,
    )
