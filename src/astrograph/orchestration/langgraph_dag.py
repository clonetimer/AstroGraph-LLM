"""Optional real LangGraph backend. Missing dependencies never become a mock graph."""
from __future__ import annotations
from dataclasses import asdict
from typing import Annotated, Any, TypedDict
from uuid import uuid4
from .dag import ChangeLedger, DAGPlan, Handler, TaskOutcome, execute_node


def _merge(left: dict, right: dict) -> dict:
    duplicate = set(left) & set(right)
    if any(left[key] != right[key] for key in duplicate):
        raise ValueError("DAG result collision")
    return {**left, **right}


class GraphState(TypedDict):
    outcomes: Annotated[dict[str, dict[str, Any]], _merge]
    run_id: str


def build_langgraph_dag(plan: DAGPlan, handlers: dict[str, Handler], *,
                        ledger: ChangeLedger | None = None, checkpointer=None):
    if set(handlers) != {node.task_id for node in plan.nodes}:
        raise ValueError("one handler per DAG node is required")
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError as exc:
        raise RuntimeError("LANGGRAPH_DEPENDENCY_UNAVAILABLE: install the complete wheel dependency set") from exc
    graph = StateGraph(GraphState)
    for spec in plan.nodes:
        def make_node(node):
            def invoke(state: GraphState):
                parents = {key: TaskOutcome(**state["outcomes"][key]) for key in node.depends_on}
                outcome = execute_node(node, handlers[node.task_id], parents,
                                       run_id=state["run_id"], ledger=ledger)
                return {"outcomes": {node.task_id: asdict(outcome)}}
            return invoke
        graph.add_node(spec.task_id, make_node(spec))
        # List-start edges are AND joins; separate edges would permit early execution.
        if spec.depends_on:
            graph.add_edge(list(spec.depends_on), spec.task_id)
        else:
            graph.add_edge(START, spec.task_id)
    referenced = {parent for node in plan.nodes for parent in node.depends_on}
    for spec in plan.nodes:
        if spec.task_id not in referenced:
            graph.add_edge(spec.task_id, END)
    return graph.compile(checkpointer=checkpointer)


def run_langgraph_dag(plan: DAGPlan, handlers: dict[str, Handler], *, max_workers=4,
                      run_id=None, ledger=None) -> dict[str, TaskOutcome]:
    graph = build_langgraph_dag(plan, handlers, ledger=ledger)
    result = graph.invoke({"outcomes": {}, "run_id": run_id or str(uuid4())},
                          {"max_concurrency": max_workers, "recursion_limit": len(plan.nodes) + 4})
    return {node.task_id: TaskOutcome(**result["outcomes"][node.task_id]) for node in plan.nodes}
