import pytest
from astrograph.assurance.analysis_plan import *
from astrograph.assurance.sealed_campaign import SealedCampaignPreregistration
H="c"*64

def test_analysis_plan_blocks_posthoc_selection():
 with pytest.raises(ValueError): PrimaryAnalysisPlan(plan_id="x",primary_endpoint="risk",primary_contrast="A7-A0",independent_unit="MONTE_CARLO_GROUP",development_groups=["3","4"],status="DRAFT",noninferiority_margin=.01,multiplicity_method="HIERARCHICAL")
 p=PrimaryAnalysisPlan(plan_id="x",primary_endpoint="risk",primary_contrast="A7-A0",independent_unit="MONTE_CARLO_GROUP",development_groups=["3","8"],status="FROZEN",noninferiority_margin=.01,multiplicity_method="HIERARCHICAL"); assert len(p.compute_sha256())==64

def test_sealed_campaign_rejects_reused_id():
 kw=dict(campaign_id="NEW",source_tree_sha256=H,generator_sha256=H,model_package_sha256=H,temporal_policy_sha256=H,evidence_package_sha256=H,fusion_policy_sha256=H,rejection_policy_sha256=H,evaluator_sha256=H,statistics_plan_sha256=H,status="FROZEN")
 assert len(SealedCampaignPreregistration(**kw).compute_sha256())==64
 with pytest.raises(ValueError): SealedCampaignPreregistration(**(kw|{"previously_accessed_campaign_ids":["NEW"]}))
