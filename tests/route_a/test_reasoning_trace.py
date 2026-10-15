
import pytest
from astrograph.knowledge.reasoning_trace import MultistageReasoningTrace,ReasoningStage,StageRecord
H="e"*64
def test_trace_is_ordered_label_free_and_explanation_faithful():
    rows=(StageRecord(ReasoningStage.TEMPORAL_LOCALIZATION,(H,),H),StageRecord(ReasoningStage.EVIDENCE_EXECUTION,(H,),H,("ev1",)),StageRecord(ReasoningStage.EXPLANATION,(H,),H,metadata={"cited_evidence_ids":["ev1"]}))
    tr=MultistageReasoningTrace("astrograph-multistage-reasoning-trace.v1","traj",rows)
    assert len(tr.compute_sha256())==64
def test_unexecuted_evidence_citation_rejected():
    rows=(StageRecord(ReasoningStage.EVIDENCE_EXECUTION,(H,),H,("ev1",)),StageRecord(ReasoningStage.EXPLANATION,(H,),H,metadata={"cited_evidence_ids":["ev2"]}))
    with pytest.raises(ValueError): MultistageReasoningTrace("astrograph-multistage-reasoning-trace.v1","t",rows)
def test_truth_field_rejected():
    with pytest.raises(ValueError): StageRecord(ReasoningStage.FUSION,(H,),H,metadata={"true_label":"x"})
