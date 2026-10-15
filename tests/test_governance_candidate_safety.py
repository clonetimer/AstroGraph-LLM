from __future__ import annotations

import json
from pathlib import Path

from astrograph.governance.kernel import _read_candidate_version


def test_candidate_version_reader_never_executes_candidate_source(tmp_path: Path) -> None:
    (tmp_path / "src/astrograph").mkdir(parents=True)
    (tmp_path / "release.json").write_text(json.dumps({"version": "1.2.3"}), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="candidate"\nversion="1.2.3"\n', encoding="utf-8")
    marker = tmp_path / "SHOULD_NOT_EXIST"
    (tmp_path / "src/astrograph/__init__.py").write_text(
        '__version__ = "1.2.3"\n'
        'from pathlib import Path\n'
        f'Path({str(marker)!r}).write_text("executed")\n',
        encoding="utf-8",
    )

    version, errors = _read_candidate_version(tmp_path)

    assert version == "1.2.3"
    assert errors == []
    assert not marker.exists()
