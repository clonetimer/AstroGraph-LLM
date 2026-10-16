"""R11-F CLI safeguards: preflight makes no live calls; scoring is post-run only."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrograph.evaluation.material_llm_blind_campaign import (
    ARM_ORDER, BlindMaterialCase, run_blind_material_campaign,
)


ROOT = Path(__file__).resolve().parents[1]


def make_manifest(tmp_path):
    document = tmp_path / "synthetic-reference.txt"
    document.write_text("Test equipment A may overheat under load.", encoding="utf-8")
    payload = {
        "schema_version": "astrograph-blind-inputs.v1",
        "cases": [{
            "case_id": "EXPERIMENT-1", "group_id": "SYNTHETIC-FAMILY",
            "frozen_diagnosis": {
                "candidate_fault_id": "BENCH_A",
                "candidate_fault_ids": ["BENCH_A", "BENCH_B"],
                "material_source_group_id": "EXPERIMENT-1",
                "evidence": [],
            },
            "reviewed_material": None,
            "document_paths": ["synthetic-reference.txt"],
        }],
    }
    p = tmp_path / "inputs.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_cli_default_preflight_does_not_call_llm_or_create_results(tmp_path):
    manifest = make_manifest(tmp_path)
    out = tmp_path / "run"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_r11f_blind_material_campaign.py"),
        "--manifest", str(manifest), "--output", str(out),
    ], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "PREFLIGHT_ONLY_NO_LLM_CALLS"
    assert not out.exists()


@pytest.mark.parametrize("bad", ["truth_field", "path_escape"])
def test_cli_rejects_hidden_truth_and_outside_documents(tmp_path, bad):
    manifest = make_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    if bad == "truth_field":
        payload["cases"][0]["frozen_diagnosis"]["true_fault_id"] = "BENCH_A"
    else:
        outside = tmp_path.parent / "outside-test-asset.txt"
        outside.write_text("not approved input", encoding="utf-8")
        payload["cases"][0]["document_paths"] = ["../outside-test-asset.txt"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    run = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_r11f_blind_material_campaign.py"),
        "--manifest", str(manifest), "--output", str(tmp_path / "refused"),
    ], capture_output=True, text=True)
    assert run.returncode != 0
    assert not (tmp_path / "refused").exists()


def test_scoring_cli_is_separate_from_input_and_never_calls_model(tmp_path):
    manifest = make_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    row = payload["cases"][0]
    case = BlindMaterialCase(
        case_id=row["case_id"], group_id=row["group_id"],
        frozen_diagnosis=row["frozen_diagnosis"], reviewed_material=None,
        document_paths=(tmp_path / row["document_paths"][0],),
    )

    class PassiveService:
        paperseek_client = None
        def __init__(self, arm):
            self.config = SimpleNamespace(
                material_tool_calls_enabled=arm == ARM_ORDER[-1],
                a2o_enabled=False,
            )
        def enrich(self, **kwargs):
            return {"status": "NOT_CONFIGURED",
                    "generation": {"status": "NOT_CONFIGURED", "explanation": None}}

    root = tmp_path / "campaign"
    run_blind_material_campaign([case], service_factory=PassiveService, output_root=root)
    reviews = [
        {"case_id": case.case_id, "arm": arm,
         "evidence_entailment": 0, "counterevidence_recognition": 0,
         "abstention_appropriateness": 0, "review_actionability": 0}
        for arm in ARM_ORDER
    ]
    rating_path = tmp_path / "independent-human-ratings.json"
    rating_path.write_text(json.dumps({
        "schema_version": "astrograph-material-llm-human-ratings.v1",
        "reviews": reviews,
    }), encoding="utf-8")
    script = ROOT / "scripts/score_r11f_blind_material_campaign.py"
    command = [sys.executable, str(script), "--campaign-root", str(root),
               "--ratings", str(rating_path)]
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)["status"] == "HUMAN_SCORES_RECORDED"
    scores = json.loads((root / "human_scores.json").read_text())
    assert scores["diagnostic_accuracy_gain"] is None
    assert scores["rated_cells"] == 4
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0


def test_scoring_cli_detects_manifest_modification(tmp_path):
    manifest = make_manifest(tmp_path)
    data = json.loads(manifest.read_text())["cases"][0]
    case = BlindMaterialCase(
        case_id=data["case_id"], group_id=data["group_id"],
        frozen_diagnosis=data["frozen_diagnosis"], reviewed_material=None,
        document_paths=(tmp_path / data["document_paths"][0],),
    )

    class PassiveService:
        paperseek_client = None
        def __init__(self, arm):
            self.config = SimpleNamespace(material_tool_calls_enabled=(arm == ARM_ORDER[-1]), a2o_enabled=False)
        def enrich(self, **kwargs):
            return {"status": "NOT_CONFIGURED", "generation": {"status": "NOT_CONFIGURED"}}

    root = tmp_path / "campaign"
    run_blind_material_campaign([case], service_factory=PassiveService, output_root=root)
    with (root / "manifest.json").open("a") as handle:
        handle.write(" ")
    rating = tmp_path / "human.json"
    rating.write_text(json.dumps({"schema_version": "astrograph-material-llm-human-ratings.v1", "reviews": []}))
    run = subprocess.run([
        sys.executable, str(ROOT / "scripts/score_r11f_blind_material_campaign.py"),
        "--campaign-root", str(root), "--ratings", str(rating),
    ], capture_output=True, text=True)
    assert run.returncode != 0
    assert not (root / "human_scores.json").exists()
