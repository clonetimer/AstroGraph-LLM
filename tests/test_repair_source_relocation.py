from pathlib import Path
import hashlib
import pytest
from astrograph.knowledge.source_paths import resolve_bound_source


def workspace(tmp_path):
    (tmp_path/"pyproject.toml").write_text("[project]")
    (tmp_path/"src/astrograph").mkdir(parents=True)
    (tmp_path/"configs/knowledge").mkdir(parents=True)
    source=tmp_path/"configs/knowledge/example.json"
    source.write_text('{"reviewed":true}')
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("recorded", ["/old/machine/project/configs/knowledge/example.json", r"C:\old\project\configs\knowledge\example.json"])
def test_relocation_is_portable_and_content_bound(tmp_path, recorded):
    source, digest=workspace(tmp_path)
    assert resolve_bound_source(recorded, digest, anchor=tmp_path/"configs/bundle.json") == source


def test_checksum_mismatch_not_silently_accepted(tmp_path):
    source,digest=workspace(tmp_path)
    source.write_text("tampered")
    with pytest.raises(ValueError):
        resolve_bound_source("/old/configs/knowledge/example.json",digest,anchor=tmp_path/"configs/bundle.json")


def test_path_traversal_not_relocated(tmp_path):
    source,digest=workspace(tmp_path)
    with pytest.raises(ValueError, match="traversal"):
        resolve_bound_source("/old/../configs/knowledge/example.json",digest,anchor=tmp_path/"configs/bundle.json")


def test_ambiguous_suffixes_not_accepted(tmp_path):
    source,digest=workspace(tmp_path)
    (tmp_path/"example.json").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="one"):
        resolve_bound_source("/old/configs/knowledge/example.json",digest,anchor=tmp_path/"configs/bundle.json")
