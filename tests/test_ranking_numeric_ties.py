import pytest
from astrograph.fusion.models import FaultRanking, RankedFault, RankingSource


def ranking(scores):
    return FaultRanking(ranking_id='numeric-test', episode_id='numeric-episode',
                        source=RankingSource.DATA_DRIVEN, producer_id='numeric-test', producer_version='1',
                        candidates=[RankedFault(fault_id=f'F{i}', rank=i+1, score=p,
                                                source=RankingSource.DATA_DRIVEN) for i, p in enumerate(scores)])


def test_unequal_tiny_scores_can_be_tied_without_failing_order_validation():
    r = ranking([.9, 4e-14, 2e-14, 0.])
    assert [c.rank for c in r.candidates] == [1, 2, 2, 2]
    assert [c.score for c in r.candidates] == [.9, 4e-14, 2e-14, 0.]


def test_tolerance_chain_does_not_join_incompatible_endpoints():
    r = ranking([1.5e-12, .75e-12, 0.])
    assert [c.rank for c in r.candidates] == [1, 1, 3]


def test_genuinely_reversed_scores_are_still_rejected():
    with pytest.raises(ValueError, match='higher source score'):
        ranking([.1, .9])
