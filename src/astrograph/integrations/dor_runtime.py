from __future__ import annotations

"""Shared runtime identity helpers for optional external DOR integrations."""

from .process_control import run_managed_subprocess
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_STACK_COMPONENTS = {
    "DOR": ("dor-diagnostic-stack", "dor"),
    "ISGN": ("isgn-toolkit", "isgn"),
    "M2N": ("m2n-toolkit", "m2n"),
    "FEC": ("fec-toolkit", "fec"),
}


class DORStackComponentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution: str = Field(min_length=1)
    import_name: str = Field(min_length=1)
    stack_version: str = Field(min_length=1)
    metadata_version: str = Field(min_length=1)
    module_version: str = Field(min_length=1)
    package_source: str = Field(min_length=1)
    package_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DORRuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dor-runtime-identity.v2"
    executable: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dor_version: str = Field(min_length=1)
    stack_info: dict[str, Any]
    components: dict[str, DORStackComponentIdentity]
    stack_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    distribution_source: str | None = None
    distribution_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    distribution_source_hash_scope: str | None = None


def _hash_python_package(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"DOR package source contains no Python files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _python_interpreter(executable: Path) -> Path:
    lines = executable.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines or not lines[0].startswith("#!"):
        raise ValueError("DOR executable does not expose a Python interpreter shebang")
    interpreter = Path(lines[0][2:].strip())
    if not interpreter.is_file():
        raise FileNotFoundError(f"DOR Python interpreter not found: {interpreter}")
    return interpreter


def _probe_stack_components(
    executable: Path,
    stack_info: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, DORStackComponentIdentity]:
    interpreter = _python_interpreter(executable)
    probe_spec = {
        name: {"distribution": distribution, "import_name": import_name}
        for name, (distribution, import_name) in _STACK_COMPONENTS.items()
    }
    code = (
        "import importlib,importlib.metadata,json,pathlib;"
        f"spec={probe_spec!r};"
        "out={};"
        "exec(\"for name,item in spec.items():\\n"
        " m=importlib.import_module(item['import_name'])\\n"
        " out[name]={'metadata_version':importlib.metadata.version(item['distribution']),"
        "'module_version':str(getattr(m,'__version__','')),'package_source':str(pathlib.Path(m.__file__).resolve().parent)}\");"
        "print(json.dumps(out))"
    )
    probe = run_managed_subprocess(
        [str(interpreter), "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError((probe.stderr or probe.stdout or "DOR stack probe failed").strip())
    raw = json.loads(probe.stdout)
    components: dict[str, DORStackComponentIdentity] = {}
    for name, (distribution, import_name) in _STACK_COMPONENTS.items():
        stack_component = stack_info.get(name)
        if not isinstance(stack_component, dict):
            raise ValueError(f"DOR stack-info is missing component {name}")
        if stack_component.get("distribution") != distribution:
            raise ValueError(f"DOR stack-info distribution mismatch for {name}")
        if stack_component.get("installed") is not True:
            raise ValueError(f"DOR stack-info reports {name} as not installed")
        info = raw.get(name)
        if not isinstance(info, dict):
            raise ValueError(f"DOR Python probe is missing component {name}")
        stack_version = str(stack_component.get("version") or "")
        metadata_version = str(info.get("metadata_version") or "")
        module_version = str(info.get("module_version") or "")
        if not module_version:
            raise ValueError(f"{name} module does not expose __version__")
        if len({stack_version, metadata_version, module_version}) != 1:
            raise ValueError(
                f"{name} version identity mismatch: stack={stack_version!r}, "
                f"metadata={metadata_version!r}, module={module_version!r}"
            )
        package_source = Path(str(info.get("package_source") or ""))
        if not package_source.is_dir():
            raise ValueError(f"{name} package source is not a directory: {package_source}")
        components[name] = DORStackComponentIdentity(
            distribution=distribution,
            import_name=import_name,
            stack_version=stack_version,
            metadata_version=metadata_version,
            module_version=module_version,
            package_source=str(package_source),
            package_source_sha256=_hash_python_package(package_source),
        )
    return components


def _stack_fingerprint(components: dict[str, DORStackComponentIdentity]) -> str:
    payload = {
        name: component.model_dump(mode="json")
        for name, component in sorted(components.items())
    }
    return sha256_bytes(canonical_json_bytes(payload))


def inspect_dor_runtime(executable: str, *, timeout_s: float) -> DORRuntimeIdentity:
    resolved = shutil.which(executable) if "/" not in executable else executable
    if not resolved or not Path(resolved).is_file():
        raise FileNotFoundError(f"DOR executable not found: {executable}")
    path = Path(resolved).resolve()
    completed = run_managed_subprocess(
        [str(path), "stack-info"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no subprocess output").strip()[-2000:]
        raise RuntimeError(f"DOR stack-info exited {completed.returncode}: {detail}")
    stack_info = json.loads(completed.stdout)
    components = _probe_stack_components(path, stack_info, timeout_s=timeout_s)
    dor = components["DOR"]
    source, source_sha256 = dor.package_source, dor.package_source_sha256
    return DORRuntimeIdentity(
        executable=str(path),
        executable_sha256=sha256_file(path),
        dor_version=dor.stack_version,
        stack_info=stack_info,
        components=components,
        stack_fingerprint_sha256=_stack_fingerprint(components),
        distribution_source=source,
        distribution_source_sha256=source_sha256,
        distribution_source_hash_scope="dor-package-python-files",
    )
