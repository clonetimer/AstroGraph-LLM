
import pytest
from astrograph.fusion.contextual_evidence_policy import (
    ContextualEvidencePolicy, PolicyStatus, ReliabilityAggregator, ReliabilityComponents,
)
W = {k: 1.0 for k in ReliabilityComponents.__dataclass_fields__}

def policy(mode=ReliabilityAggregator.BOTTLENECK):
    return ContextualEvidencePolicy("astrograph-contextual-evidence-policy.v1", "p", PolicyStatus.CANDIDATE, mode, W, 0.0)

def test_bottleneck_and_effective_weight():
    c = ReliabilityComponents(1,.8,.9,.7,.95,.85,.9)
    assert policy().multiplier(c) == .7
    assert policy().effective_weight(2.0, c) == 1.4

def test_truth_metadata_rejected():
    c = ReliabilityComponents(1,1,1,1,1,1,1)
    with pytest.raises(ValueError):
        policy().multiplier(c, {"nested": {"ground_truth": "f"}})

def test_posthoc_groups_cannot_select_policy():
    with pytest.raises(ValueError):
        ContextualEvidencePolicy("astrograph-contextual-evidence-policy.v1", "p", PolicyStatus.DEVELOPMENT_SELECTED, ReliabilityAggregator.BOTTLENECK, W, 0, (3,4), (4,9), "a"*64)

def test_geometric_candidate_is_not_silent_selection():
    p = policy(ReliabilityAggregator.GEOMETRIC_MEAN)
    assert p.status is PolicyStatus.CANDIDATE
    assert 0 < p.multiplier(ReliabilityComponents(.5,.5,.5,.5,.5,.5,.5)) <= .5 + 1e-12
