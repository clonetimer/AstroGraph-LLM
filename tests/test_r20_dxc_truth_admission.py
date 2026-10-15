import pytest
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario
from astrograph.evaluation.fault_set_metrics import admit_truth

def test_no_injection_is_not_manufactured_nominal_truth():
    _,t=parse_scenario('sensors @0 { E1 = 0 };')
    with pytest.raises(ValueError,match='TRUTH_UNAVAILABLE'):admit_truth(t)
