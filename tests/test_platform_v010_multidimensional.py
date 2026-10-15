from __future__ import annotations

from pathlib import Path

from astrograph.integrations.multidimensional_diagnostics import assess_multidimensional_telemetry
from astrograph.integrations.temporal_localization import localize_from_visualization
from astrograph.integrations.upload_report import UploadReportJobStore


def _synthetic_visualization() -> dict:
    a=[]; b=[]; c=[]
    for i in range(121):
        t=float(i)
        # level shift + trend
        va = 0.05 * ((i % 5) - 2) if i < 60 else 6.0 + 0.06*(i-60)
        # oscillation increase with almost unchanged center
        vb = 0.03 * ((i % 4) - 1.5) if i < 60 else (3.0 if i % 2 else -3.0)
        # correlated with a before, decoupled after
        vc = va * 0.8 if i < 60 else ((i % 7) - 3) * 0.4
        a.append({"t":t,"v":va}); b.append({"t":t,"v":vb}); c.append({"t":t,"v":vc})
    return {
        "status":"AVAILABLE",
        "timeline":{"start_s":0.0,"end_s":120.0,"onset_s":None,"onset_source":None},
        "channels":[
            {"name":"a","label":"a","unit":"","points":a},
            {"name":"b","label":"b","unit":"","points":b},
            {"name":"c","label":"c","unit":"","points":c},
        ],
        "truth":{"fault_onset_s":1.0},
        "onset_time_s":1.0,
    }


def test_multidimensional_assessment_detects_multiple_statistical_views_without_truth() -> None:
    viz=_synthetic_visualization()
    temporal=localize_from_visualization(viz, diagnostic_fault_id="TEST")
    result=assess_multidimensional_telemetry(viz, temporal=temporal)
    assert result["status"] == "AVAILABLE"
    assert result["truth_or_labels_used"] is False
    assert result["diagnostic_vote_allowed"] is False
    assert result["probability_claim_allowed"] is False
    dims={item["dimension"]:item for item in result["dimension_summary"]}
    assert dims["LEVEL_SHIFT"]["index"] >= 70
    assert dims["OSCILLATION"]["index"] >= 40
    assert result["channels"]
    assert result["focus_start_s"] != 1.0


def test_multimetric_window_localizer_reports_first_detection_not_truth_onset() -> None:
    result=localize_from_visualization(_synthetic_visualization(), diagnostic_fault_id="TEST")
    assert result["method"] == "WINDOWED_MULTIMETRIC_TELEMETRY"
    assert result["truth_or_labels_used"] is False
    assert result["first_detection_s"] is not None
    assert result["first_detection_s"] > 20.0
    assert result["first_detection_s"] != 1.0


def test_adcs_bundle_maintenance_contains_multidimensional_review_and_curve_linked_mechanism(tmp_path: Path) -> None:
    source=Path('/mnt/data/ADCS_RW_JAM_UI_test_bundle.zip')
    if not source.is_file():
        return
    store=UploadReportJobStore(tmp_path/'platform')
    job=store.create('诊断反作用轮异常','normal')
    store.put_file(job['job_id'],source.name,source.read_bytes())
    report=store.run(job['job_id'])
    assert report['status']=='SUCCEEDED'
    maintenance=store.maintenance(job['job_id'])
    stats=maintenance['multidimensional_assessment']
    assert stats['status']=='AVAILABLE'
    assert stats['diagnostic_vote_allowed'] is False
    assert len(stats['channels']) >= 3
    assert any(item['level'] != 'STABLE' for item in stats['dimension_summary'])
    nodes=maintenance['mechanism_graph']['nodes']
    assert any(node.get('related_channels') for node in nodes)
    viz=store.visualization(job['job_id'])
    assert viz['timeline']['onset_source']=='FIRST_DETECTION'
    assert viz['timeline']['onset_s'] == maintenance['temporal_localization']['first_detection_s']


def test_sqlite_persists_multidimensional_assessment(tmp_path: Path) -> None:
    from astrograph.integrations.platform_state import PlatformStateRepository
    repo=PlatformStateRepository(tmp_path/'control.db')
    payload={"status":"AVAILABLE","truth_or_labels_used":False,"dimension_summary":[]}
    repo.save_multidimensional_assessment('JOB-X',payload)
    assert repo.get_multidimensional_assessment('JOB-X') == payload


def test_multidimensional_assessment_endpoint(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from astrograph.api import create_app
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT', str(tmp_path/'platform'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','local')
    source=Path('/mnt/data/ADCS_RW_JAM_UI_test_bundle.zip')
    if not source.is_file():
        return
    client=TestClient(create_app())
    created=client.post('/v1/platform/jobs',json={"task":"诊断反作用轮异常","work_mode":"normal"})
    assert created.status_code==200, created.text
    job_id=created.json()['job_id']
    uploaded=client.put(f'/v1/platform/jobs/{job_id}/files/{source.name}', content=source.read_bytes())
    assert uploaded.status_code==200, uploaded.text
    started=client.post(f'/v1/platform/jobs/{job_id}/run')
    assert started.status_code==200, started.text
    # Local executor may complete synchronously or shortly after; use direct store status through API polling.
    import time
    for _ in range(100):
        job=client.get(f'/v1/platform/jobs/{job_id}').json()
        if job.get('status') in {'SUCCEEDED','FAILED'}:
            break
        time.sleep(0.02)
    assert job['status']=='SUCCEEDED', job
    result=client.get(f'/v1/platform/jobs/{job_id}/multidimensional-assessment')
    assert result.status_code==200, result.text
    body=result.json()
    assert body['status']=='AVAILABLE'
    assert body['diagnostic_vote_allowed'] is False
