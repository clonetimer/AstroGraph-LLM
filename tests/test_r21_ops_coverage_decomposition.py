from astrograph.evaluation.real_data_failure_analysis import summarize_ops_coverage


def test_ops_coverage_is_truth_and_sampling_stratified():
    rows = [
        {"truth_anomaly": 0, "channel": "A", "sampling": "5", "raw_segment_length": 10, "blocked": True},
        {"truth_anomaly": 0, "channel": "A", "sampling": "1", "raw_segment_length": 40, "blocked": False},
        {"truth_anomaly": 1, "channel": "B", "sampling": "5", "raw_segment_length": 40, "blocked": False},
    ]
    report = summarize_ops_coverage(rows)
    assert report["coverage"] == 2 / 3
    assert report["strata"]["truth"]["0"]["blocked_rate"] == 0.5
    assert report["strata"]["sampling"]["5"]["blocked"] == 1


def test_frozen_r21_report_preserves_422_107_and_non_mcar_composition():
    from pathlib import Path
    import json, os
    path = os.environ.get("ASTROGRAPH_R21_RESULTS")
    if not path:
        path = str(Path(__file__).resolve().parents[2].parents[1] / "results")
    report_path = Path(path) / "ops_failure_mechanism_report.json"
    if not report_path.exists():
        return
    summary = json.loads(report_path.read_text(encoding="utf-8"))["summary"]
    assert (summary["scored_count"], summary["blocked_count"]) == (422, 107)
    assert summary["strata"]["truth"]["1"]["blocked"] == 3
    assert summary["strata"]["truth"]["0"]["blocked"] == 104
