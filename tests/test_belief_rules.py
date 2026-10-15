import math
import pytest
from astrograph.fusion.belief_rules import er_aggregate,matching_degrees,infer


def test_single_rule_preserves_incomplete_belief():
    r=er_aggregate([1.],[[.6,.2]])
    assert r['beliefs']==pytest.approx([.6,.2]);assert r['ignorance']==pytest.approx(.2)


def test_symmetric_conflict_does_not_select_one_side():
    r=er_aggregate([.5,.5],[[1.,0.],[0.,1.]])
    assert r['beliefs']==pytest.approx([.5,.5]);assert r['ignorance']==pytest.approx(0.)


def test_all_unknown_is_not_normalized_to_certainty():
    r=er_aggregate([.5,.5],[[0.,0.],[0.,0.]])
    assert r['beliefs']==[0.,0.];assert r['ignorance']==pytest.approx(1.)


def test_zero_weight_rule_has_no_effect():
    assert er_aggregate([1.,0.],[[.2,.3],[1.,0.]])['beliefs']==pytest.approx([.2,.3])


@pytest.mark.parametrize('x',[-1.,math.nan,math.inf,1.1])
def test_invalid_antecedent_rejected(x):
    with pytest.raises(ValueError):matching_degrees(x)


def test_incomplete_rules_preserve_mass_over_interpolation():
    for i in range(101):
        r=infer(i/100,'UNKNOWN',[[.2,.4]]*12)
        assert sum(r['beliefs'])+r['ignorance']==pytest.approx(1.)
        assert r['ignorance']>0


def test_rule_order_permutation_invariance():
    a=er_aggregate([.3,.7],[[.1,.7],[.5,.2]])
    b=er_aggregate([.7,.3],[[.5,.2],[.1,.7]])
    assert a['beliefs']==pytest.approx(b['beliefs'])
