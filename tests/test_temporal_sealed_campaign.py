import json
from pathlib import Path
import pytest
from astrograph.assurance.temporal_sealed_campaign import *

def plan():return TemporalCampaignPrecisionPlan(mission_strata=('attitude_acquisition','steady_pointing_payload','propulsive_maneuver'))
def test_precision_plan_meets_exact_bounds():
 p=plan();assert p.zero_false_alarm_upper_bound<.10;assert p.all_group_success_lower_bound>.90;assert p.minimum_per_class_lower_bound>.80
 assert abs(clopper_pearson_upper_one_sided(0,30)-0.0950338528553)<1e-10
 assert abs(clopper_pearson_lower_one_sided(30,30)-0.9049661471447)<1e-10

def test_27_of_30_is_not_enough_but_28_is():
 assert clopper_pearson_lower_one_sided(27,30)<.8
 assert clopper_pearson_lower_one_sided(28,30)>.8

def test_mapping_never_runtime_exposed():
 with pytest.raises(ValueError):FaultConditionMapping(diagnostic_class='x',generator_condition_id='y',semantic_relation='EXACT',rationale='z',runtime_exposed=True)

def test_blocked_ledger_cannot_consume_authorization():
 with pytest.raises(ValueError):OneTimeExecutionLedger(campaign_id='c',preregistration_sha256='0'*64,authorization_token_sha256='1'*64,state=CampaignState.FROZEN_RUNTIME_BLOCKED,generation_consumed=True)
