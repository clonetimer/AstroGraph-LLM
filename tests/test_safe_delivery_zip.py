from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from astrograph.data.thermal_star_adapter import materialize_d3_thermal_star_delivery
from astrograph.data.delivery_archive import materialize_d3_delivery
from astrograph.data.safe_zip import normalize_zip_member


def _write(archive: zipfile.ZipFile, name: str, content: str = "{}") -> None:
    archive.writestr(name, content)


def test_d3_delivery_rejects_parent_path_segments(tmp_path: Path) -> None:
    archive_path = tmp_path / "d3.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        _write(archive, "delivery/02_dataset/dataset_manifest.json")
        _write(archive, "delivery/02_dataset/campaign_plan.json")
        _write(archive, "delivery/02_dataset/dataset_split_manifest.json")
        _write(archive, "delivery/02_dataset/aggregate/telemetry_adcs_raw.csv", "time_s\n0\n")
        _write(archive, "delivery/03_run_audit/../../../escaped/run_manifest.json")

    with pytest.raises(ValueError, match="unsafe path segment"):
        with materialize_d3_delivery(archive_path):
            pass
    assert not (tmp_path / "escaped" / "run_manifest.json").exists()


def test_thermal_star_delivery_rejects_parent_path_segments(tmp_path: Path) -> None:
    archive_path = tmp_path / "thermal.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        _write(archive, "phm_d3_thermal_star_production/dataset_manifest.json")
        _write(archive, "phm_d3_thermal_star_production/dataset_split_manifest.json")
        _write(archive, "phm_d3_thermal_star_production/aggregate/telemetry_raw.csv", "time_s\n0\n")
        _write(archive, "phm_d3_thermal_star_production/runs/../../../escaped/run_manifest.json")

    with pytest.raises(ValueError, match="unsafe path segment"):
        with materialize_d3_thermal_star_delivery(archive_path):
            pass
    assert not (tmp_path / "escaped" / "run_manifest.json").exists()


def test_zip_symbolic_link_members_are_rejected() -> None:
    info = zipfile.ZipInfo("delivery/02_dataset/dataset_manifest.json")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16

    with pytest.raises(ValueError, match="symbolic-link"):
        normalize_zip_member(info)
