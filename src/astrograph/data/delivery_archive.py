from __future__ import annotations

import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .safe_zip import ZipExtractionBudget, extract_selected_member, normalize_zip_member


_REQUIRED_D3_RELATIVE_FILES = {
    "02_dataset/campaign_plan.json",
    "02_dataset/dataset_manifest.json",
    "02_dataset/dataset_split_manifest.json",
    "02_dataset/aggregate/telemetry_adcs_raw.csv",
}


def _locate_delivery_root(root: Path) -> Path:
    if (root / "02_dataset" / "dataset_manifest.json").exists():
        return root
    matches = list(root.rglob("02_dataset/dataset_manifest.json"))
    if len(matches) != 1:
        raise ValueError(f"cannot uniquely locate D3 delivery root under {root}: {len(matches)} candidates")
    return matches[0].parent.parent


def _wanted_member(relative: str) -> bool:
    if relative in _REQUIRED_D3_RELATIVE_FILES:
        return True
    if not relative.startswith("03_run_audit/"):
        return False
    return relative.endswith(("/run_manifest.json", "/parameter_sample.json", "/fault_effect_report.json"))


@contextmanager
def materialize_d3_delivery(path: str | Path) -> Iterator[Path]:
    """Yield a directory-shaped D3 delivery from either a directory or ZIP.

    The producer ZIP stores Windows backslashes in member names. Extraction is
    normalized explicitly and only the dataset/audit files required by AstroGraph
    are materialized; the large source snapshot is intentionally skipped.
    """

    source = Path(path).resolve()
    if source.is_dir():
        yield _locate_delivery_root(source)
        return
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError(f"D3 delivery must be a directory or ZIP: {source}")

    temp_root = Path(tempfile.mkdtemp(prefix="astrograph-d3-delivery-"))
    try:
        with zipfile.ZipFile(source) as archive:
            normalized = [(info, normalize_zip_member(info)) for info in archive.infolist()]
            manifest_members = [name for _, name in normalized if name.endswith("/02_dataset/dataset_manifest.json")]
            if len(manifest_members) != 1:
                raise ValueError(f"D3 ZIP must contain one dataset manifest, got {len(manifest_members)}")
            root_prefix = manifest_members[0][: -len("02_dataset/dataset_manifest.json")].rstrip("/")
            output_root = temp_root / Path(root_prefix).name
            budget = ZipExtractionBudget()
            for info, normalized_name in normalized:
                if info.is_dir() or not normalized_name.startswith(root_prefix + "/"):
                    continue
                relative = normalized_name[len(root_prefix) + 1 :]
                if not _wanted_member(relative):
                    continue
                extract_selected_member(
                    archive,
                    info,
                    output_root=output_root,
                    relative=relative,
                    budget=budget,
                )
        yield _locate_delivery_root(output_root)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
