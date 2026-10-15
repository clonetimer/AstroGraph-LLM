import pytest
from astrograph.evaluation.esa_anomaly_adapter import *

def test_annotations_keep_rare_nominal_and_gap_distinct():
    assert event_category('Rare Event')=='RARE_NOMINAL'
    assert event_category('Communication Gap')=='COMMUNICATION_GAP'
    with pytest.raises(ValueError,match='NOT_FORMAL_FAULT'):require_fault_evidence('RARE_NOMINAL')

def test_unknown_annotation_not_binary_guessed():
    with pytest.raises(ValueError,match='UNADMITTED'):event_category('unknown')
