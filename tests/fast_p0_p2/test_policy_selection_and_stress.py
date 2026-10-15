import pytest
from astrograph.fusion.development_policy_selection import *
from astrograph.evaluation.stress_plan import *

def row(pid,g,a,r,n=.2,b=.1,e=.05): return GroupPolicyMetrics(pid,g,a,r,n,b,e)
def test_noninferiority_first_selects_only_feasible():
 rows=[row('A0','3',.9,.1),row('A0','8',.9,.1),row('bad','3',.95,.4),row('bad','8',.95,.4),row('good','3',.92,.09),row('good','8',.91,.1)]
 r=select_policy_noninferiority_first(rows,baseline_policy_id='A0',development_groups={'3','8'},forbidden_groups={'4','9'},max_risk_increase=.02,max_nll_increase=.02,max_brier_increase=.02,max_ece_increase=.02,max_accuracy_drop=.01)
 assert r['selected_policy_id']=='good' and not r['audit']['bad']['feasible']
def test_posthoc_rows_fail():
 with pytest.raises(ValueError): select_policy_noninferiority_first([row('A0','4',.9,.1)],baseline_policy_id='A0',development_groups={'4'},forbidden_groups={'4','9'},max_risk_increase=.1,max_nll_increase=.1,max_brier_increase=.1,max_ece_increase=.1,max_accuracy_drop=.1)
def test_stress_plan_is_outcome_blind_and_hashed():
 p=default_outcome_blind_stress_plan(); assert len(p.cases)>=7 and len(p.compute_sha256())==64
