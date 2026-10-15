import numpy as np, pytest
from astrograph.experiments.mechanism_evidence_temporal import (
    MECHANISM_CHANNEL_ORDER,MechanismBank,MechanismScoreConfiguration,
    MechanismTemporalPolicy,ScoreAggregation,candidate_score_configurations,
    candidate_policies,local_change_bank_scores,select_scores,
)
def mat(n=50):return np.zeros((n,len(MECHANISM_CHANNEL_ORDER)))
def test_grid_is_conservative():
    assert len(candidate_score_configurations())==24
    p=candidate_policies();assert len(p)==600
    assert all(x.threshold_multiplier>=1 and x.persistence_points>=2 for x in p)
def test_step_detected_without_truth():
    x=mat();x[35:,MECHANISM_CHANNEL_ORDER.index("adcs.innovation.star_gyro_rate_residual_norm_rad_s")]=2
    c=MechanismScoreConfiguration(channel_floor_multiplier=.1,previous_window_spread_multiplier=1,aggregation=ScoreAggregation.MAX)
    s=local_change_bank_scores(x,channel_order=MECHANISM_CHANNEL_ORDER,channel_scale=np.ones(x.shape[1]),configuration=c)
    assert max(s[MechanismBank.GYRO][31:])>0 and max(s[MechanismBank.GYRO][:31])==0
def test_policy_rejects_low_persistence():
    c=MechanismScoreConfiguration(channel_floor_multiplier=.1,previous_window_spread_multiplier=1,aggregation=ScoreAggregation.MAX)
    with pytest.raises(ValueError):
        MechanismTemporalPolicy(policy_id="x",score_configuration_sha256=c.configuration_sha256,threshold_multiplier=1,persistence_points=1)
def test_selection_is_bank_specific_and_causal():
    scores={b:np.zeros(50) for b in MechanismBank};scores[MechanismBank.REACTION_WHEEL][35:37]=2
    c=MechanismScoreConfiguration(channel_floor_multiplier=.1,previous_window_spread_multiplier=1,aggregation=ScoreAggregation.MAX)
    p=MechanismTemporalPolicy(policy_id="x",score_configuration_sha256=c.configuration_sha256,threshold_multiplier=1,persistence_points=2)
    r=select_scores(scores,p,{b:1 for b in MechanismBank})
    assert r["selected_bank"]=="REACTION_WHEEL" and r["selected_index"]==36
