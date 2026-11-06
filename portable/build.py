"""Build the focused Windows x64 distribution without modifying core modules.

Build-time network preparation is separate. This builder accepts an official
runtime archive and verified wheels, and performs no network access or pip calls.
"""
from __future__ import annotations

import argparse
import ast
import tempfile
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import zipfile

RUNTIME_VERSION = "3.14.10"
RUNTIME_SHA256 = "90905e64a840557d8c1bc4f5b74db3d7ee85e3ffb217611cac1c7813acea4159"
SOURCE_BASE_COMMIT = "5389bfd78f8a9f62117f8804eab181fda7557046"
PINNED = {
    "pydantic": "2.14.2", "pydantic-core": "2.50.2", "annotated-types": "0.8.0",
    "typing-extensions": "4.16.0", "typing-inspection": "0.4.4", "pypdf": "6.21.4",
}
CORE_FILES = (
    "__init__.py", "provider_lifecycle.py",
    "integrations/external_diagnosis_analysis.py",
    "integrations/external_diagnosis_report.py",
    "integrations/external_diagnosis_conversation.py",
    "integrations/long_report_analysis.py", "integrations/execution_observability.py",
    "integrations/reviewed_context.py", "integrations/stage_execution.py",
    "integrations/result_log_review.py", "integrations/execution_budget.py",
    "integrations/bounded_http.py", "integrations/process_control.py",
    "integrations/http_worker.py", "llm/__init__.py", "llm/openai_compatible.py",
    "llm/candidate_selection.py", "llm/token_budget.py", "llm/task_prompts.py",
    "llm/audit.py", "resources/execution_view.js", "resources/execution_view.css",
    "resources/workspace_view.js", "resources/workspace_view.css", "retrieval/structured_document.py", "retrieval/models.py",
    "retrieval/document_governance.py",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_app(repo_root: Path, destination: Path) -> Path:
    """Copy the canonical, small import closure and focused server into destination."""
    repo_root, destination = Path(repo_root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("staging destination must be empty")
    source_root = repo_root / "src" / "astrograph"
    if not source_root.is_dir():
        source_root = repo_root / "app" / "astrograph"
    for relative in CORE_FILES:
        source = source_root / relative
        target = destination / "astrograph" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for package in ("integrations", "retrieval"):
        (destination / "astrograph" / package / "__init__.py").write_text(
            '"""Focused portable subset; unrelated engine exports intentionally omitted."""\n',
            encoding="utf-8",
        )
    portable_root = repo_root / "portable"
    if not portable_root.is_dir():
        portable_root = repo_root / "app"
    shutil.copy2(portable_root / "workbench.py", destination / "workbench.py")
    for directory in ("assets", "examples"):
        shutil.copytree(portable_root / directory, destination / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for asset in ("execution_view.js", "execution_view.css", "workspace_view.js", "workspace_view.css"):
        shutil.copy2(source_root / "resources" / asset, destination / "assets" / asset)
    return destination


def application_version(repo_root: Path) -> str:
    """Read metadata without importing the application or its dependencies."""
    source = Path(repo_root) / "portable" / "workbench.py"
    for statement in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VERSION" for t in statement.targets):
            value = ast.literal_eval(statement.value)
            if isinstance(value, str) and re.fullmatch(r"[0-9A-Za-z.+-]+", value):
                return value
    raise ValueError("application VERSION is missing or invalid")


def stage_bootstrap(repo_root: Path, destination: Path, profile: str = "win10") -> None:
    """Package every file referenced by the selected offline startup scripts."""
    root, target = Path(repo_root), Path(destination)
    if profile not in {"win10", "win7"}:
        raise ValueError("unknown runtime profile")
    scripts = root / "portable" if profile == "win10" else root / "portable/profiles/win7"
    for name in ("CHECK.bat", "START.bat"):
        text = (scripts / name).read_text(encoding="ascii").replace("\r\n", "\n")
        (target / name).write_bytes(text.replace("\n", "\r\n").encode("ascii"))
    for name in ("verify_manifest.py", "win7_environment.py") if profile == "win7" else ("verify_manifest.py",):
        shutil.copy2(root / "portable" / name, target / name)
    guide = (scripts / "README_zh-CN.md").read_text(encoding="utf-8")
    (target / "README_zh-CN.md").write_text(guide, encoding="utf-8")
    (target / "README_zh-CN.txt").write_text(guide, encoding="utf-8-sig")


def stage_source(repo_root: Path, destination: Path) -> None:
    """Export the portable source closure, not all research/diagnostic modules."""
    root, target = Path(repo_root), Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    ignored = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.git', '.pytest_cache',
                                    '_test_artifacts', '*.ttf', '*.otf', '*.woff', '*.woff2')
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / 'app'
        stage_app(root, staged)
        shutil.copytree(staged / 'astrograph', target / 'src/astrograph', ignore=ignored)
    shutil.copytree(root / 'portable', target / 'portable', ignore=ignored)
    (target / 'docs').mkdir()
    for pattern in ('WIN7_*.md', 'PORTABLE_OBSERVABILITY*.md'):
        for doc in (root / 'docs').glob(pattern):
            shutil.copy2(doc, target / 'docs' / doc.name)
    for name in ('LICENSE', 'LICENSE.txt', 'COPYING'):
        if (root / name).is_file():
            shutil.copy2(root / name, target / name)
    (target / 'README.md').write_text(
        '# AstroGraph portable source subset\n\n'
        'Run `python -m unittest discover -s portable -p "test*.py"`.\n'
        'This is the minimal shared source closure, not the full repository. '
        'Do not overwrite an existing repository with it. Runtime materials are separate.\n', encoding='utf-8')


def _extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        failure = bundle.testzip()
        if failure:
            raise ValueError(f"archive CRC failed: {failure}")
        for member in bundle.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename or ":" in member.filename:
                raise ValueError("unsafe archive member")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("archive symlinks are not supported")
        bundle.extractall(destination)


def _verify_amd64(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(2) != b"MZ":
            raise ValueError(f"not a Windows PE file: {path.name}")
        handle.seek(0x3C)
        offset = struct.unpack("<I", handle.read(4))[0]
        handle.seek(offset)
        if handle.read(4) != b"PE\x00\x00" or struct.unpack("<H", handle.read(2))[0] != 0x8664:
            raise ValueError(f"not an AMD64 PE file: {path.name}")


def _verified_wheels(wheels_dir: Path) -> tuple[list[Path], dict]:
    provenance_path = wheels_dir / "wheel-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    receipts = {item["filename"]: item for item in provenance["wheels"]}
    wheels = sorted(wheels_dir.glob("*.whl"))
    found = {}
    for wheel in wheels:
        receipt = receipts.get(wheel.name, {})
        if (receipt.get("sha256") != sha256(wheel)
                or receipt.get("pypi_sha256") != receipt.get("sha256")
                or receipt.get("sha256_matches_pypi") is not True):
            raise ValueError(f"missing or invalid PyPI hash verification: {wheel.name}")
        if not re.fullmatch(r"https://pypi\.org/pypi/[^/]+/[^/]+/json", receipt.get("metadata_url", "")):
            raise ValueError("official PyPI metadata receipt required")
        with zipfile.ZipFile(wheel) as bundle:
            metadata_names = [n for n in bundle.namelist() if n.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise ValueError("wheel metadata missing or ambiguous")
            metadata = BytesParser().parsebytes(bundle.read(metadata_names[0]))
            name = re.sub(r"[-_.]+", "-", metadata["Name"]).lower()
            version = metadata["Version"]
            if name not in PINNED or PINNED[name] != version or name in found:
                raise ValueError(f"unexpected, unpinned or duplicate dependency: {name} {version}")
            expected_tag = "cp314-cp314-win_amd64" if name == "pydantic-core" else "py3-none-any"
            if not wheel.name.endswith(f"-{expected_tag}.whl"):
                raise ValueError(f"unsupported wheel tag: {wheel.name}")
            found[name] = version
    if found != PINNED:
        raise ValueError(f"incomplete wheel closure: {found}")
    return wheels, provenance


def build_distribution(repo_root: Path, runtime: Path, wheels_dir: Path,
                       output_dir: Path, source_commit: str | None = None) -> tuple[Path, Path]:
    repo_root, runtime, wheels_dir, output_dir = map(Path, (repo_root, runtime, wheels_dir, output_dir))
    if sha256(runtime) != RUNTIME_SHA256:
        raise ValueError("official Python runtime SHA256 mismatch")
    wheels, wheel_provenance = _verified_wheels(wheels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    package_name = "AstroGraph-LLM-Win10-x64"
    package = output_dir / package_name
    if package.exists():
        raise FileExistsError(f"refusing to overwrite an existing distribution: {package}")
    package.mkdir()
    _extract_archive(runtime, package / "runtime")
    _verify_amd64(package / "runtime" / "python.exe")
    (package / "runtime" / "python314._pth").write_text(
        "python314.zip\n.\n../packages\n../app\n", encoding="ascii")
    stage_app(repo_root, package / "app")
    (package / "packages").mkdir()
    for wheel in wheels:
        _extract_archive(wheel, package / "packages")
    for extension in (package / "packages").rglob("*.pyd"):
        _verify_amd64(extension)
    (package / "data").mkdir()
    stage_bootstrap(repo_root, package, "win10")
    shutil.copy2(repo_root / "portable/check_installation.py", package / "CHECK_INSTALLATION.py")
    stage_source(repo_root, package / "source")
    for name in ("LICENSE", "LICENSE.txt", "COPYING"):
        if (repo_root / name).is_file():
            shutil.copy2(repo_root / name, package / name)
    provenance_dir = package / "provenance"
    provenance_dir.mkdir()
    runtime_provenance = runtime.with_name(f"python-{RUNTIME_VERSION}-runtime-provenance.json")
    if runtime_provenance.exists():
        shutil.copy2(runtime_provenance, provenance_dir / "python-runtime.json")
    else:
        _json_write(provenance_dir / "python-runtime.json", {
            "version": RUNTIME_VERSION, "sha256": RUNTIME_SHA256,
            "download_url": f"https://www.python.org/ftp/python/{RUNTIME_VERSION}/{runtime.name}",
        })
    _json_write(provenance_dir / "wheel-provenance.json", wheel_provenance)
    if source_commit is None:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    manifest = {
        "schema_version": "astrograph-portable-package.v1", "source_commit": source_commit,
        "source_base_commit": SOURCE_BASE_COMMIT, "readable_application_source_included": True,
        "built_at": datetime.now(timezone.utc).isoformat(), "target": "Windows 10 x64",
        "python_version": RUNTIME_VERSION, "runtime_archive_sha256": RUNTIME_SHA256,
        "dependencies": PINNED, "native_windows_execution_verified": False,
        "package_version": application_version(repo_root),
        "shared_source_closure": list(CORE_FILES),
        "profile": "win10",
        "files": [{"path": p.relative_to(package).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(package.rglob("*")) if p.is_file()],
    }
    _json_write(package / "MANIFEST.json", manifest)
    archive = output_dir / f"{package_name}.zip"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(package.rglob("*")):
            bundle.write(path, path.relative_to(output_dir).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise ValueError("final ZIP CRC verification failed")
    (output_dir / f"{archive.name}.sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
    return package, archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--wheels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    package, archive = build_distribution(Path(__file__).resolve().parents[1], args.runtime,
                                         args.wheels_dir, args.output_dir, args.source_commit)
    print(json.dumps({"package": str(package.resolve()), "zip": str(archive.resolve()), "sha256": sha256(archive)}, indent=2))


if __name__ == "__main__":
    main()
