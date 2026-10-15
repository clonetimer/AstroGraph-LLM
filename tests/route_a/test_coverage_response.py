
import pytest
from astrograph.experiments.coverage_response import CoveragePoint,summarize_coverage_response
def test_group_macro_response_and_nonmonotonicity_are_reported_not_hidden():
    pts=[]
    for g in (3,8):
        pts += [CoveragePoint("p",0,g,0,0,.8),CoveragePoint("p",0,g,.5,.5,.9),CoveragePoint("p",0,g,1,1,.85)]
    s=summarize_coverage_response(pts)
    assert s.group_count==2 and s.monotonicity_violations==1 and abs(s.net_gain_full_minus_zero-.05)<1e-12
def test_duplicate_group_level_record_rejected():
    p=CoveragePoint("p",0,3,0,0,.8)
    with pytest.raises(ValueError): summarize_coverage_response([p,p])
