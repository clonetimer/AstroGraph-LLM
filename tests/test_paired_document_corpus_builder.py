import csv
import json
from pathlib import Path

from astrograph.retrieval import DocumentVisibilityClass
from astrograph.retrieval.document_corpus_builder import (
    ANOMALY_CANONICAL_BYTES,
    OBSERVATION_CANONICAL_BYTES,
    audit_paired_document_corpus,
    build_causal_reports,
    build_paired_document_corpus,
    verify_paired_document_corpus_rebuild,
)


def _write_telemetry(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["time_s", "adcs.current_a", "mode"],
        )
        writer.writeheader()
        writer.writerow({"time_s": 0, "adcs.current_a": offset, "mode": "track"})
        writer.writerow({"time_s": 10, "adcs.current_a": offset + 2, "mode": "track"})


def _dataset(root: Path) -> None:
    rows = [
        {
            "sample_id": "fault-name-hidden:mc_04",
            "sample_index": "4",
            "telemetry_file": "missing/posthoc.csv",
            "condition_id": "sealed_fault",
            "diagnostic_class": "sealed_class",
        },
        {
            "sample_id": "nominal:mc_00",
            "sample_index": "0",
            "telemetry_file": "visible/nominal.csv",
            "condition_id": "nominal",
            "diagnostic_class": "nominal",
        },
        {
            "sample_id": "wheel_fault:mc_01",
            "sample_index": "1",
            "telemetry_file": "visible/fault.csv",
            "condition_id": "wheel_fault",
            "diagnostic_class": "wheel_fault",
            "effect": "torque_loss",
        },
        {
            "sample_id": "development_fault:mc_03",
            "sample_index": "3",
            "telemetry_file": "visible/development.csv",
            "condition_id": "development_fault",
            "diagnostic_class": "development_fault",
        },
    ]
    fields = sorted({key for row in rows for key in row})
    root.mkdir(parents=True)
    with (root / "sample_index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_telemetry(root / "visible/nominal.csv", 1.0)
    _write_telemetry(root / "visible/fault.csv", 2.0)
    _write_telemetry(root / "visible/development.csv", 3.0)


def test_builder_skips_posthoc_before_open_and_physically_separates_truth(tmp_path):
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"policy": "test"}), encoding="utf-8")
    generator = tmp_path / "generator.py"
    generator.write_text("CONDITIONS = {}\n", encoding="utf-8")
    output = tmp_path / "corpus"

    manifest, receipt = build_paired_document_corpus(
        datasets=[("fixture", dataset)],
        output_root=output,
        policy_path=policy,
        generator_paths={"fixture": generator},
    )

    assert receipt["bundle_count"] == 3
    assert receipt["skipped_posthoc_row_count"] == 1
    assert receipt["posthoc_telemetry_or_truth_opened"] is False
    assert receipt["split_counts"] == {"DEVELOPMENT": 1, "TRAIN": 2}
    assert receipt["document_counts"] == {
        "OBSERVATION_CAUSAL": 6,
        "RESOLVED_CASE_TRAIN_ONLY": 1,
        "TRUTH_QUARANTINED": 3,
    }
    assert receipt["document_type_counts"] == {
        "ANOMALY_REPORT": 3,
        "CAMPAIGN_REPORT": 3,
        "CLOSURE_REPORT": 1,
        "TEST_LOG": 3,
    }
    assert receipt["runtime_index_audit"]["passed"] is True
    assert len(manifest.index.runtime_document_ids) == 6
    assert len(manifest.index.evaluator_truth_document_ids) == 3
    assert not (
        set(manifest.index.runtime_document_ids)
        & set(manifest.index.evaluator_truth_document_ids)
    )

    all_artifacts = [artifact for bundle in manifest.bundles for artifact in bundle.artifacts]
    assert all(artifact.descriptor.license_id for artifact in all_artifacts)
    assert all(artifact.descriptor.usage_boundary for artifact in all_artifacts)
    assert all(artifact.descriptor.template_version for artifact in all_artifacts)
    runtime = [
        artifact
        for artifact in all_artifacts
        if artifact.descriptor.document_id in manifest.index.runtime_document_ids
    ]
    assert all(
        artifact.descriptor.visibility_class == DocumentVisibilityClass.OBSERVATION_CAUSAL
        for artifact in runtime
    )
    for artifact in runtime:
        text = (output / artifact.relative_path).read_text(encoding="utf-8").casefold()
        assert "wheel_fault" not in text
        assert "development_fault" not in text
        assert "diagnostic_class" not in text
        assert "condition_id" not in text

    closures = [
        artifact
        for artifact in all_artifacts
        if artifact.descriptor.visibility_class
        == DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY
    ]
    assert len(closures) == 1
    assert closures[0].descriptor.contains_episode_truth is True
    assert closures[0].descriptor.document_id not in manifest.index.runtime_document_ids

    audit = audit_paired_document_corpus(
        manifest_path=output / "manifest.json",
        corpus_root=output,
        datasets={"fixture": dataset},
        generator_paths={"fixture": generator},
    )
    assert audit["status"] == "PASS"
    assert audit["artifact_file_count_checked"] == 10
    assert audit["runtime_truth_value_proxy_documents_checked"] == 6
    assert audit["future_cutoff_gate_documents_checked"] == 6
    assert audit["runtime_document_cutoff_counts"] == {"10": 6}

    rebuild = verify_paired_document_corpus_rebuild(
        existing_manifest_path=output / "manifest.json",
        datasets=[("fixture", dataset)],
        policy_path=policy,
        generator_paths={"fixture": generator},
        temporary_parent=tmp_path / "rebuilds",
    )
    assert rebuild["status"] == "PASS"
    assert rebuild["artifact_count"] == 10
    assert rebuild["artifact_hash_mismatch_count"] == 0
    assert not list((tmp_path / "rebuilds").iterdir())


def test_causal_reports_have_independent_fixed_cutoffs_and_no_diagnosis(tmp_path):
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "time_s,current_a,mode\n"
        "0,1,track\n10,1,track\n20,1,track\n30,1,track\n"
        "40,2,track\n60,3,track\n90,4,track\n120,5,track\n180,6,track\n",
        encoding="utf-8",
    )
    reports = build_causal_reports(telemetry)
    assert [item[0] for item in reports] == [60.0, 120.0, 180.0]
    for cutoff, observation, anomaly in reports:
        assert len(observation.encode("utf-8")) == OBSERVATION_CANONICAL_BYTES
        assert len(anomaly.encode("utf-8")) == ANOMALY_CANONICAL_BYTES
        assert f"Cutoff time (s): {cutoff:g}" in observation
        assert f"Observation cutoff (s): {cutoff:g}" in anomaly
        assert "no fault label" in anomaly
        assert "not a diagnosis" in anomaly
