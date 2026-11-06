"""Explicit build-time downloads and official hash verification; never used at startup."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from build import RUNTIME_SHA256, RUNTIME_VERSION, _verified_wheels, sha256


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "AstroGraph-portable-build/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def prepare_vendor(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"python-{RUNTIME_VERSION}-embed-amd64.zip"
    runtime = output_dir / filename
    runtime_url = f"https://www.python.org/ftp/python/{RUNTIME_VERSION}/{filename}"
    if not runtime.exists() or sha256(runtime) != RUNTIME_SHA256:
        temporary = runtime.with_suffix(".download")
        try:
            with urllib.request.urlopen(runtime_url, timeout=120) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            if sha256(temporary) != RUNTIME_SHA256:
                raise ValueError("official runtime SHA256 mismatch")
            temporary.replace(runtime)
        finally:
            temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(runtime) as archive:
        if archive.testzip():
            raise ValueError("runtime ZIP CRC check failed")
        if "LICENSE.txt" not in archive.namelist():
            raise ValueError("runtime license missing")
    checked_on = datetime.now(timezone.utc).date().isoformat()
    provenance = {
        "checked_on": checked_on, "version": RUNTIME_VERSION, "architecture": "amd64",
        "artifact": filename, "download_url": runtime_url,
        "release_page": "https://www.python.org/downloads/release/python-31410/",
        "supported_windows_documentation": "https://docs.python.org/3.14/using/windows.html#supported-windows-versions",
        "bytes": runtime.stat().st_size, "sha256": sha256(runtime),
        "official_sha256": RUNTIME_SHA256, "sha256_matches_official": True,
        "zip_crc_verified": True, "executed": False,
        "hash_source": "Official Python 3.14.10 release checksum, pinned in portable/build.py",
        "limitations": ["Native execution is verified separately; download alone is not Windows qualification."],
    }
    (output_dir / f"python-{RUNTIME_VERSION}-runtime-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    wheels_dir = output_dir / "wheels-win314"
    wheels_dir.mkdir(exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
        "--index-url", "https://pypi.org/simple", "--only-binary=:all:", "--no-deps",
        "--platform", "win_amd64", "--implementation", "cp", "--python-version", "314",
        "--abi", "cp314", "--requirement", str(Path(__file__).with_name("requirements.txt")),
        "--dest", str(wheels_dir),
    ], check=True)
    receipts = []
    for wheel in sorted(wheels_dir.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError("missing or ambiguous wheel metadata")
            metadata = BytesParser().parsebytes(archive.read(names[0]))
        name, version = metadata["Name"], metadata["Version"]
        metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
        document = fetch_json(metadata_url)
        matches = [item for item in document["urls"] if item["filename"] == wheel.name]
        if len(matches) != 1:
            raise ValueError(f"wheel absent from official PyPI metadata: {wheel.name}")
        remote = matches[0]
        digest = sha256(wheel)
        if digest != remote["digests"]["sha256"]:
            raise ValueError(f"official PyPI SHA256 mismatch: {wheel.name}")
        receipts.append({
            "filename": wheel.name, "name": name, "version": version,
            "bytes": wheel.stat().st_size, "sha256": digest,
            "pypi_sha256": remote["digests"]["sha256"], "sha256_matches_pypi": True,
            "metadata_url": metadata_url, "download_url": remote["url"],
        })
    (wheels_dir / "wheel-provenance.json").write_text(json.dumps({
        "checked_on": checked_on, "target": "cp314-win_amd64", "wheels": receipts,
    }, indent=2) + "\n", encoding="utf-8")
    _verified_wheels(wheels_dir)
    return {"runtime": str(runtime), "wheels_dir": str(wheels_dir), "verified_wheels": len(receipts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Build cache directory; this command explicitly uses the network")
    args = parser.parse_args()
    print(json.dumps(prepare_vendor(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
