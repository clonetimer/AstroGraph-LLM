from __future__ import annotations

"""Shared identity probe for optional Python CLI distributions."""

from .process_control import run_managed_subprocess
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_python_package(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"Python package source contains no Python files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class PythonCLIRuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-python-cli-runtime-identity.v1"
    executable: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    distribution: str = Field(min_length=1)
    version: str = Field(min_length=1)
    module_version: str = Field(min_length=1)
    import_name: str = Field(min_length=1)
    package_source: str = Field(min_length=1)
    package_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_source_hash_scope: str = "package-python-files"


def inspect_python_cli_runtime(
    executable: str,
    *,
    distribution: str,
    import_name: str,
    timeout_s: float,
) -> PythonCLIRuntimeIdentity:
    resolved = shutil.which(executable) if "/" not in executable else executable
    if not resolved or not Path(resolved).is_file():
        raise FileNotFoundError(f"external executable not found: {executable}")
    path = Path(resolved).resolve()
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines or not lines[0].startswith("#!"):
        raise ValueError("external executable does not expose a Python interpreter shebang")
    interpreter = Path(lines[0][2:].strip())
    code = (
        "import importlib,importlib.metadata,json,pathlib;"
        f"m=importlib.import_module({import_name!r});"
        f"print(json.dumps({{'version':importlib.metadata.version({distribution!r}),"
        "'module_version':str(getattr(m,'__version__','')),'package_source':str(pathlib.Path(m.__file__).resolve().parent)}))"
    )
    probe = run_managed_subprocess(
        [str(interpreter), "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError((probe.stderr or probe.stdout or "runtime probe failed").strip())
    info = json.loads(probe.stdout)
    version = str(info["version"])
    module_version = str(info.get("module_version") or "")
    if not module_version:
        raise ValueError(f"external module {import_name} does not expose __version__")
    if module_version != version:
        raise ValueError(
            f"external runtime version mismatch for {distribution}: "
            f"metadata={version!r}, module={module_version!r}"
        )
    package_source = Path(str(info["package_source"]))
    return PythonCLIRuntimeIdentity(
        executable=str(path),
        executable_sha256=sha256_file(path),
        distribution=distribution,
        version=version,
        module_version=module_version,
        import_name=import_name,
        package_source=str(package_source),
        package_source_sha256=hash_python_package(package_source),
    )
