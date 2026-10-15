from __future__ import annotations

import zipfile

import pytest

from astrograph.assurance.archive_intake import IntakeStatus, scan_zip_archive


def test_clean_archive_passes(tmp_path) -> None:
    path = tmp_path / "clean.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("data/a.csv", "t,x\n0,1\n")
    result = scan_zip_archive(path, "clean")
    assert result.status is IntakeStatus.PASS
    assert result.bad_crc_member is None
    assert result.member_count == 1


def test_pickle_requires_quarantine_without_deserialising(tmp_path) -> None:
    path = tmp_path / "pickle.zip"
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("data/frame.pkl", b"not deserialised")
    result = scan_zip_archive(path, "pickle")
    assert result.status is IntakeStatus.QUARANTINE_REQUIRED
    assert result.pickle_members == ("data/frame.pkl",)


def test_path_traversal_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    result = scan_zip_archive(path, "bad")
    assert result.status is IntakeStatus.REJECT
    assert result.unsafe_member_names == ("../escape.txt",)


def test_duplicate_member_is_rejected(tmp_path) -> None:
    path = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name: 'x.csv'"):
        with zipfile.ZipFile(path, "w") as handle:
            handle.writestr("x.csv", "1")
            handle.writestr("x.csv", "2")
    result = scan_zip_archive(path, "duplicate")
    assert result.status is IntakeStatus.REJECT
    assert result.duplicate_member_names == ("x.csv",)


def test_scan_is_deterministic(tmp_path) -> None:
    path = tmp_path / "same.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("b.csv", "b")
        handle.writestr("a.csv", "a")
    one = scan_zip_archive(path, "same")
    two = scan_zip_archive(path, "same")
    assert one.to_dict() == two.to_dict()
    assert one.sha256 == two.sha256
