"""Exercise the Agent on supplied fixtures; this is NOT an accuracy benchmark.

The synthetic trainer really updates a small candidate TCN. Production weights
and the active graph are never replaced. All artifacts are written to a NEW
output directory. Run from a prepared environment with PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile
import io
import numpy as np
import pandas as pd
from astrograph.agents.mission.models import AgentRequest, AgentSettings
from astrograph.agents.mission.service import MissionAgent
from astrograph.agents.mission.storage import file_sha
from astrograph.agents.mission.training import TrainingManifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real", action="store_true", help="Also run one ADCS raw-telemetry smoke slice from an externally supplied dataset ZIP.")
    parser.add_argument("--adcs-dataset-zip", type=Path, help="ADCS v5.2.2 dataset ZIP used only when --real is requested.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protected_paths = [
        "src/astrograph/oag/kernel.py",
        "src/astrograph/fusion/decision.py",
        "src/astrograph/oag/sensor_checks.py",
        "models/adcs_v5_2_2/p2_innovation_v2.pt",
    ]
    protected = {path: file_sha(root / path) for path in protected_paths}
    settings = AgentSettings(project_root=str(root), read_roots=[str(root), str(out)], output_root=str(out / "runs"),
                             approved_model_sha256=[protected["models/adcs_v5_2_2/p2_innovation_v2.pt"]])
    results = {}
    agent = MissionAgent(settings)
    request = AgentRequest(task_id="fixture-diagnosis", artifacts={
        "input": str(root / "examples/core_diagnosis/input.json"),
        "graph_dir": str(root / "examples/core_diagnosis/graph"),
        "measurements": str(root / "examples/core_diagnosis/measurements.json")})
    diagnosis = agent.run(request)
    assert diagnosis.status == "SUCCEEDED", diagnosis.model_dump()
    results["fixture_diagnosis"] = {"status": diagnosis.status, "scope": "SYNTHETIC_TOOL_FIXTURE"}
    research = agent.run(AgentRequest(task_id="fixture-retrieval", task="research", goal="陀螺 gyro 故障 bias",
                        documents=[str(root / "examples/mission_agent/mechanism-note.md")]))
    assert research.status == "SUCCEEDED", research.model_dump()
    results["retrieval"] = {"status": research.status, "source_count": len(research.output["sources"])}

    rng = np.random.default_rng(7)
    arrays = {}
    for split, count in (("train", 24), ("validation", 12)):
        labels = np.arange(count) % 3
        features = rng.normal(0, .1, size=(count, 8, 2)).astype("float32")
        features[labels == 1, :, 0] += 2
        features[labels == 2, :, 1] -= 2
        arrays.update({f"{split}_x": features, f"{split}_anomaly": (labels > 0).astype("int64"),
                       f"{split}_fault": labels.astype("int64") - 1, f"{split}_component": labels.astype("int64") - 1,
                       f"{split}_group": np.array([split + str(i // 3) for i in range(count)])})
    data = out / "synthetic-training.npz"
    np.savez(data, **arrays)
    manifest = TrainingManifest(data_path=str(data), data_sha256=file_sha(data),
        channel_names=["sensor.x", "sensor.y"], channel_units=["1", "1"], fault_ids=["F1", "F2"],
        component_ids=["C1", "C2"], fault_to_component={"F1": "C1", "F2": "C2"}, epochs=2, width=4, batch_size=12)
    mp = out / "training-manifest.json"
    mp.write_text(manifest.model_dump_json(indent=2))
    training_settings = settings.model_copy(update={"approved_training_manifest_sha256": [file_sha(mp)]})
    (out / "settings.json").write_text(training_settings.model_dump_json(indent=2))
    training_request = AgentRequest(task_id="fixture-training", task="train_model", artifacts={"training_manifest": str(mp)})
    (out / "request-training.json").write_text(training_request.model_dump_json(indent=2))
    candidate = MissionAgent(training_settings).run(training_request)
    assert candidate.status == "SUCCEEDED", candidate.model_dump()
    report = candidate.output["task_output"]
    assert report["weights_changed"] and report["group_overlap"] == 0 and not report["active_model_modified"]
    fp = out / "truth-free-features.npz"
    np.savez(fp, x=arrays["validation_x"], channel_names=np.array(manifest.channel_names), channel_units=np.array(manifest.channel_units))
    inference = agent.run(AgentRequest(task_id="fixture-inference", task="infer_candidate", artifacts={
        "candidate_manifest": report["candidate_manifest"], "features": str(fp)}))
    assert inference.status == "SUCCEEDED", inference.model_dump()
    results["training"] = {"scope": "SYNTHETIC_24_TRAIN_12_VALIDATION_NOT_ADCS_RETRAINING", "epochs": 2,
        "weights_changed": report["weights_changed"], "test_partition_used": False, "production_promoted": False}
    results["candidate_inference"] = {"status": inference.status, "n": 12, "scope": "UNCALIBRATED_CANDIDATE_ONLY"}
    if args.real:
        if args.adcs_dataset_zip is None:
            parser.error("--real requires --adcs-dataset-zip because historical telemetry slices are intentionally not bundled in the current-source delivery.")
        dataset_zip = args.adcs_dataset_zip.resolve()
        if not dataset_zip.is_file():
            parser.error(f"ADCS dataset ZIP does not exist: {dataset_zip}")
        dataset = "whole_spacecraft_adcs_paired_dataset_v5_2_2/"
        source_member = dataset + "scenarios/attitude_acquisition_post_settle/adcs_imu_axis_dropout/sample_03/telemetry.csv"
        nominal_member = dataset + "nominal_baseline/attitude_acquisition/sample_03/telemetry.csv"
        real_dir = out / "adcs-smoke-slice"
        real_dir.mkdir()
        with zipfile.ZipFile(dataset_zip) as zf:
            source_df = pd.read_csv(io.BytesIO(zf.read(source_member)), low_memory=False)
            nominal_df = pd.read_csv(io.BytesIO(zf.read(nominal_member)), low_memory=False)
        source_df = source_df[source_df["time_s"] <= 150]
        nominal_df = nominal_df[nominal_df["time_s"] <= 150]
        source_path = real_dir / "source.csv"
        nominal_path = real_dir / "nominal.csv"
        source_df.to_csv(source_path, index=False)
        nominal_df.to_csv(nominal_path, index=False)
        request = AgentRequest(task_id="external-adcs-smoke-slice", artifacts={
            "source": str(source_path),
            "nominal": str(nominal_path),
            "model": str(root / "models/adcs_v5_2_2/p2_innovation_v2.pt")})
        real = agent.run(request)
        assert real.status == "SUCCEEDED", real.model_dump()
        results["real_slice"] = {
            "status": real.status,
            "scope": "EXTERNAL_ADCS_SMOKE_SLICE_NOT_ACCURACY_ESTIMATE",
            "dataset_zip_sha256": file_sha(dataset_zip),
            "source_member": source_member,
            "nominal_member": nominal_member,
            "diagnosis": real.output["diagnosis_result"]["diagnosis"],
        }
    unchanged = {path: file_sha(root / path) == digest for path, digest in protected.items()}
    assert all(unchanged.values()), unchanged
    receipt = {"status": "PASS", "at": datetime.now(timezone.utc).isoformat(), "checks": results,
               "protected_files_unchanged": unchanged, "live_external_models_evaluated": False}
    (out / "RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
