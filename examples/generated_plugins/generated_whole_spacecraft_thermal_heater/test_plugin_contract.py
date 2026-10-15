import json
from pathlib import Path


def test_generated_contract_files_are_self_consistent():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    build_spec = json.loads((root / "build_spec.json").read_text(encoding="utf-8"))
    assert manifest["plugin_id"] == 'generated.whole_spacecraft.thermal_heater'
    assert build_spec["plugin_id"] == manifest["plugin_id"]
    assert manifest["metadata"]["online_registration_allowed"] is False
