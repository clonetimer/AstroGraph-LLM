import csv
import pytest
from astrograph.evaluation.ops_sat_adapter import *
from astrograph.evaluation.binary_anomaly_metrics import binary_metrics

def test_raw_export_auxiliary_label_does_not_override_numeric_truth():
    assert truth_row({'segment':'s','label':'anomaly','anomaly':'0','train':'0'})['anomaly']==0

def test_unknown_numeric_truth_is_not_guessed():
    with pytest.raises(ValueError,match='LABEL_SEMANTICS'):truth_row({'segment':'s','anomaly':'nominal','train':'0'})

def test_tied_score_auc_and_average_precision():
    m=binary_metrics([0,1],[5.,5.],5)
    assert m['AUROC']==.5 and m['AUPRC_average_precision']==.5
    assert m['TP']==m['FP']==1

def test_nonfinite_metric_rejected():
    with pytest.raises(ValueError):binary_metrics([1],[float('nan')],1)
