import pytest
from astrograph.assurance.feature_semantic_contract_v3 import *
H="a"*64

def channel(**kw):
 d=dict(tensor_index=0,channel_name="x",source_field="x",physical_unit="rad/s",coordinate_frame="BODY",axis_definition="X_AXIS",sign_convention="RH_POSITIVE",sampling_rate_hz=10,timestamp_semantics="sample_time",resampling_method="EXACT",missingness_policy="REJECT",imputation_policy="NONE",normalization={"kind":"NONE"},preprocessing_sha256=H)
 d.update(kw); return ChannelSemanticSpec(**d)

def test_frozen_requires_model_and_resolved_semantics():
 with pytest.raises(ValueError): FeatureSemanticContractV3(contract_id="c",status="FROZEN",channels=[channel()],claim_boundary="x")
 c=FeatureSemanticContractV3(contract_id="c",status="FROZEN",model_package_sha256=H,channels=[channel()],claim_boundary="x")
 assert not c.unresolved_semantics()

def test_contiguous_order_enforced():
 with pytest.raises(ValueError): FeatureSemanticContractV3(contract_id="c",status="DRAFT",channels=[channel(tensor_index=1)],claim_boundary="x")
