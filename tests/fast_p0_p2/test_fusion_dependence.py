from astrograph.evaluation.fusion_dependence import *
def test_group_dependence_audit():
 d=[BranchPrediction("1","g1","A"),BranchPrediction("2","g2","A")]
 k=[BranchPrediction("1","g1","A"),BranchPrediction("2","g2","B")]
 t=[TruthRecord("1","g1","A"),TruthRecord("2","g2","A")]
 r=audit_branch_dependence(d,k,t); assert r["group_count"]==2 and r["group_macro_data_accuracy"]==1
