from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from astrograph.integrations.platform_state import PlatformStateRepository
from astrograph.integrations.temporal_localization import localize_adcs_model_csv


def _full_adcs_sample(tmp_path: Path) -> Path | None:
    source = Path('/mnt/data/ADCS.zip')
    if not source.is_file():
        return None
    member = (
        'whole_spacecraft_adcs_paired_dataset_v5_2_2/scenarios/'
        'steady_pointing_payload_settled/adcs_rw_jamming/sample_00/telemetry.csv'
    )
    with zipfile.ZipFile(source) as archive:
        target = tmp_path / 'telemetry.csv'
        target.write_bytes(archive.read(member))
    return target


def test_frozen_adcs_detector_localizes_full_contract_without_truth(tmp_path: Path) -> None:
    telemetry = _full_adcs_sample(tmp_path)
    if telemetry is None:
        pytest.skip('external ADCS.zip regression material is not present')
    result = localize_adcs_model_csv(telemetry, diagnostic_fault_id='ADCS_RW_JAM')
    assert result['status'] == 'DETECTED'
    assert result['truth_or_labels_used'] is False
    assert result['candidate_matches'] is True
    assert result['detected_fault_id'] == 'ADCS_RW_JAM'
    assert result['first_detection_s'] == 96.0
    assert result['confirmed_detection_s'] == 98.0
    assert result['model_artifact_sha256']


def test_frozen_adcs_detector_fails_closed_when_required_channels_are_missing() -> None:
    telemetry = Path('/mnt/data/adcs_v011/telemetry.csv')
    if not telemetry.is_file():
        pytest.skip('external reduced telemetry regression material is not present')
    result = localize_adcs_model_csv(telemetry, diagnostic_fault_id='ADCS_RW_JAM')
    assert result['status'] == 'UNAVAILABLE'
    assert result['truth_or_labels_used'] is False
    assert result['missing_channels']
    assert result['available_required_channels'] < result['required_channel_count']


def test_sqlite_diagnostic_history_normalizes_dimensions_channels_and_temporal(tmp_path: Path) -> None:
    repo = PlatformStateRepository(tmp_path / 'control.db')
    temporal1 = {
        'first_detection_s': 96.0,
        'confirmed_detection_s': 98.0,
        'method': 'MODEL_PLUS_MULTIMETRIC_TELEMETRY',
        'primary_source': 'ADCS_P2_MODEL',
        'agreement': 'AGREE',
    }
    temporal2 = {**temporal1, 'first_detection_s': 100.0, 'confirmed_detection_s': 102.0}
    multidim = {
        'dimension_summary': [
            {'dimension': 'LEVEL_SHIFT', 'label_zh': '幅值偏移', 'level': 'SIGNIFICANT', 'index': 90.0, 'affected_channels': 2},
            {'dimension': 'VARIABILITY', 'label_zh': '波动变化', 'level': 'WATCH', 'index': 55.0, 'affected_channels': 1},
        ],
        'channels': [
            {
                'channel': 'adcs.rw.measured_speed_rad_s_2',
                'label_zh': '反作用轮转速',
                'dimensions': [
                    {'dimension': 'LEVEL_SHIFT', 'level': 'SIGNIFICANT', 'index': 95.0},
                    {'dimension': 'VARIABILITY', 'level': 'WATCH', 'index': 50.0},
                ],
            }
        ],
    }
    repo.index_diagnostic_history(job_id='JOB-A', fault_id='ADCS_RW_JAM', temporal=temporal1, multidimensional=multidim)
    repo.index_diagnostic_history(job_id='JOB-B', fault_id='ADCS_RW_JAM', temporal=temporal2, multidimensional=multidim)
    result = repo.diagnostic_history_summary(fault_id='ADCS_RW_JAM')
    assert result['sample_count'] == 2
    assert result['temporal']['median_s'] == 98.0
    dims = {item['dimension']: item for item in result['dimensions']}
    assert dims['LEVEL_SHIFT']['significant_count'] == 2
    assert result['channels'][0]['channel'] == 'adcs.rw.measured_speed_rad_s_2'


def test_full_telemetry_ui_bundle_uses_model_and_statistical_detection(tmp_path: Path) -> None:
    from astrograph.integrations.upload_report import UploadReportJobStore

    source = Path('/mnt/data/ADCS_RW_JAM_full_telemetry_UI_test_bundle_v011.zip')
    if not source.is_file():
        pytest.skip('external v0.11 full-telemetry UI bundle is not present')
    store = UploadReportJobStore(tmp_path / 'platform')
    job = store.create('诊断反作用轮异常', 'normal')
    store.put_file(job['job_id'], source.name, source.read_bytes())
    report = store.run(job['job_id'])
    assert report['status'] == 'SUCCEEDED'
    temporal = store.maintenance(job['job_id'])['temporal_localization']
    assert temporal['primary_source'] == 'ADCS_P2_MODEL'
    assert temporal['agreement'] == 'AGREE'
    assert temporal['first_detection_s'] == 96.0
    sources = {item['source_id']: item for item in temporal['detection_sources']}
    assert sources['ADCS_P2_MODEL']['status'] == 'DETECTED'
    assert sources['MULTIMETRIC_STATISTICS']['status'] in {'DETECTED', 'SUSPECTED'}
    assert sources['MULTIMETRIC_STATISTICS']['truth_or_labels_used'] is False
    viz = store.visualization(job['job_id'])
    names = [item['name'] for item in viz['channels']]
    assert 'adcs.rw.command_execution_residual_norm_nm' in names
    assert 'adcs.pointing_error_deg' in names


def test_diagnostic_history_api_reads_control_plane(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from astrograph.api import create_app

    root = tmp_path / 'platform'
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT', str(root))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE', 'local')
    repo = PlatformStateRepository(root / 'control_plane.db')
    repo.index_diagnostic_history(
        job_id='JOB-H1', fault_id='ADCS_RW_JAM',
        temporal={'first_detection_s': 96.0, 'confirmed_detection_s': 98.0, 'method': 'MODEL', 'primary_source': 'ADCS_P2_MODEL', 'agreement': 'AGREE'},
        multidimensional={'dimension_summary': [{'dimension': 'LEVEL_SHIFT', 'label_zh': '幅值偏移', 'level': 'SIGNIFICANT', 'index': 88.0, 'affected_channels': 2}], 'channels': []},
    )
    client = TestClient(create_app())
    response = client.get('/v1/platform/diagnostic-history?fault_id=ADCS_RW_JAM')
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['sample_count'] == 1
    assert body['dimensions'][0]['dimension'] == 'LEVEL_SHIFT'
