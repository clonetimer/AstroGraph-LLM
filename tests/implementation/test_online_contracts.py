from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError
from astrograph.semantics.online_contracts import ChannelBinding, DiagnosticRequest, ObservationFrame, ReleaseRef
from astrograph.data_driven.multitask_contract import TaskDistribution, MultiTaskPluginResult

T = datetime(2026, 9, 13, tzinfo=timezone.utc)
C = ChannelBinding(channel_id="speed", semantic_id="rw.speed", unit="rad/s")
PIN = ReleaseRef(release_id="v1", sha256="a" * 64)

def frame(**changes):
    value = dict(artifact_id="obs_001", channels=[C], observed_at=[T, T + timedelta(seconds=1)],
                 available_at=[T, T + timedelta(seconds=1)], values=[[1], [2]], observed_mask=[[True], [True]])
    value.update(changes)
    return ObservationFrame.create(**value)

def request(**changes):
    value = dict(request_id="req", asset_id="asset", operating_mode="tracking", cutoff_at=T+timedelta(seconds=2),
                 observation=frame(), semantic_release=PIN, model_release=PIN, knowledge_release=PIN, plugin_release=PIN)
    value.update(changes)
    return DiagnosticRequest(**value)

def test_valid_truth_free_input_and_dor_sidecar():
    req = request()
    assert req.require_channels((C,)) == ((1.0,), (2.0,))
    assert req.dor_context()["supervision_available_to_runtime"] is False
    assert "train" not in req.model_dump()
    assert DiagnosticRequest.model_validate_json(req.model_dump_json()) == req

@pytest.mark.parametrize("changes", [
    {"observed_at": [T,T]}, {"values": [[1,2],[2,3]]}, {"observed_mask": [[True]]},
    {"observed_at": [T.replace(tzinfo=None), T.replace(tzinfo=None)]},
    {"values": [[None],[2]]}, {"values": [[1],[2]],"observed_mask":[[False],[True]]},
    {"values": [[float('nan')],[2]]}, {"values": [[float('inf')],[2]]},
    {"available_at": [T-timedelta(seconds=1),T]},
])
def test_reject_bad_frame(changes):
    with pytest.raises((ValueError,ValidationError)):frame(**changes)

def test_checksum_detects_altered_payload():
    data=frame().model_dump(mode="json");data["values"][0][0]=8
    with pytest.raises(ValidationError,match="checksum"):ObservationFrame.model_validate(data)

def test_missing_is_explicit_not_filled():
    f=frame(values=[[None],[2]],observed_mask=[[False],[True]])
    with pytest.raises(ValueError,match="complete"):request(observation=f).require_channels((C,))

@pytest.mark.parametrize("field", ["fault_label","diagnostic_class","fault_id","injection_time"])
def test_no_supervision(field):
    with pytest.raises(ValidationError):ChannelBinding(channel_id=field,semantic_id=field,unit="1")
    with pytest.raises(ValidationError):request(**{field:1})

def test_future_and_reference_qualification_rejected():
    with pytest.raises(ValidationError,match="future"):request(cutoff_at=T)
    with pytest.raises(ValidationError,match="qualification"):request(reference=frame(),reference_applicability="QUALIFIED")
    assert request(reference=frame(),reference_applicability="UNVERIFIED").reference is not None

def test_order_unit_mismatch():
    with pytest.raises(ValueError,match="unit"):request().require_channels((C.model_copy(update={"unit":"rpm"}),))

def test_multiclass_and_abstention_retained():
    anomaly=TaskDistribution(status="AVAILABLE",class_ids=("NORMAL","ANOMALY"),values=(0.2,0.8),score_kind="PROBABILITY")
    component=TaskDistribution(status="ABSTAINED",score_kind="PROBABILITY",reason="missing current")
    fault=TaskDistribution(status="AVAILABLE",class_ids=("1","2","3"),values=(0.1,0.2,0.7),score_kind="PROBABILITY")
    result=MultiTaskPluginResult(model_release=PIN,input_sha256="b"*64,anomaly=anomaly,component=component,fault=fault)
    assert result.fault.class_ids == ("1","2","3")
    assert result.component.status=="ABSTAINED"

@pytest.mark.parametrize("scores",[(0.2,0.3),(-0.1,1.1),(float('nan'),0.1)])
def test_probability_validation(scores):
    with pytest.raises(ValidationError):TaskDistribution(status="AVAILABLE",class_ids=("a","b"),values=scores,score_kind="PROBABILITY")


def test_calibration_claim_requires_release():
    with pytest.raises(ValidationError,match="calibration"):
        TaskDistribution(status="AVAILABLE",class_ids=("a","b"),values=(.2,.8),score_kind="CALIBRATED_PROBABILITY")
    assert TaskDistribution(status="AVAILABLE",class_ids=("a","b"),values=(.2,.8),
        score_kind="CALIBRATED_PROBABILITY",calibration_release=PIN).calibration_release==PIN
