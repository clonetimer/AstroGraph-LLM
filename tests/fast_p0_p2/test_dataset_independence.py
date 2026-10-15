import pytest
from astrograph.assurance.dataset_independence import *
def test_seed_only_is_not_confirmatory_independence():
 r=DatasetIndependenceRecord(dataset_id='x',generator_implementation='SHARED',scenario_design='SHARED',randomization='INDEPENDENT',evaluator_implementation='SHARED',label_source='SHARED',previously_accessed=False,hidden_truth_until_prediction_freeze=True,confirmatory_ready=False)
 assert not r.confirmatory_ready
def test_false_promotion_rejected():
 with pytest.raises(ValueError): DatasetIndependenceRecord(dataset_id='x',generator_implementation='SHARED',scenario_design='SHARED',randomization='INDEPENDENT',evaluator_implementation='SHARED',label_source='SHARED',previously_accessed=False,hidden_truth_until_prediction_freeze=True,confirmatory_ready=True)
