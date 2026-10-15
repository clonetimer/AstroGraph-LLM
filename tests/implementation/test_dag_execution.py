import json
import threading
import pytest
from astrograph.orchestration.dag import DAGPlan,TaskNode,DAGExecutor,ChangeLedger


def test_parallel_branches_join_only_after_all_dependencies(tmp_path):
    barrier=threading.Barrier(2,timeout=3)
    plan=DAGPlan("parallel",(TaskNode("A"),TaskNode("B"),TaskNode("Join",("A","B"))))
    def branch(_):barrier.wait();return 1
    def join(deps):assert set(deps)=={"A","B"};return sum(x.output for x in deps.values())
    ledger=ChangeLedger(tmp_path/"ledger.jsonl")
    result=DAGExecutor(plan,{"A":branch,"B":branch,"Join":join},max_workers=2,ledger=ledger).run(run_id="test")
    assert result["Join"].output==2
    assert result["Join"].started_ns>=max(result[x].finished_ns for x in ("A","B"))
    assert len(ledger.verify(ledger.path))==6


def test_failure_blocks_only_dependents_and_allowed_join_can_degrade():
    plan=DAGPlan("failure",(TaskNode("A"),TaskNode("B"),TaskNode("Blocked",("A",)),
                           TaskNode("Join",("A","B"),allow_failed_dependencies=True)))
    def fail(_):raise ValueError("failure")
    result=DAGExecutor(plan,{"A":fail,"B":lambda _:2,"Blocked":lambda _:99,
                            "Join":lambda deps:deps["B"].output}).run()
    assert result["A"].status=="FAILED"
    assert result["Blocked"].status=="BLOCKED"
    assert result["Join"].output==2

@pytest.mark.parametrize("nodes",[
    (TaskNode("A",("B",)),TaskNode("B",("A",))),
    (TaskNode("A",("missing",)),), (TaskNode("A"),TaskNode("A")),
    (TaskNode("A",writes=("src/tools",)),TaskNode("B",writes=("src/tools/plugin.py",))),
])
def test_invalid_dag_rejected_before_changes(nodes):
    with pytest.raises(ValueError):DAGPlan("bad",nodes)

def test_ordered_shared_writes_and_serialization():
    plan=DAGPlan("valid",(TaskNode("A",writes=("x",)),TaskNode("B",("A",),writes=("x",))))
    assert DAGPlan.from_dict(json.loads(json.dumps(plan.to_dict())))==plan
    assert "A --> B" in plan.mermaid()

def test_hash_chained_ledger_detects_edit(tmp_path):
    ledger=ChangeLedger(tmp_path/"events.jsonl");ledger.append(run_id="r",task_id="A",event="STARTED")
    ledger.path.write_text(ledger.path.read_text(encoding="utf-8").replace("STARTED","CHANGED"))
    with pytest.raises(ValueError):ChangeLedger.verify(ledger.path)

def test_real_langgraph_backend_when_dependencies_present():
    try:
        from langgraph.graph import StateGraph
    except ImportError as exc:
        pytest.skip(f"real LangGraph dependency closure unavailable: {exc.name}")
    from astrograph.orchestration.langgraph_dag import run_langgraph_dag
    plan=DAGPlan("real",(TaskNode("A"),TaskNode("B"),TaskNode("Join",("A","B"))))
    outcomes=run_langgraph_dag(plan,{"A":lambda _:1,"B":lambda _:2,"Join":lambda d:d["A"].output+d["B"].output})
    assert outcomes["Join"].output==3

def test_missing_langgraph_is_explicit_not_silent_fallback():
    try:
        from langgraph.graph import StateGraph
    except ImportError:
        from astrograph.orchestration.langgraph_dag import build_langgraph_dag
        plan=DAGPlan("one",(TaskNode("A"),))
        with pytest.raises(RuntimeError,match="LANGGRAPH_DEPENDENCY_UNAVAILABLE"):
            build_langgraph_dag(plan,{"A":lambda _:1})
