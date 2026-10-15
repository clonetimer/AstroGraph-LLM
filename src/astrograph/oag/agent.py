"""Bounded autonomous diagnosis over the native fusion kernel.

Planning is deterministic and evidence-gap driven in this release, not an LLM
performance claim. There is no online DAG requirement and no DOR dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Iterable

from astrograph.evidence.executable import FeatureResult, evaluate_evidence
from astrograph.memory.service import CaseMemoryService, PendingCase, CheckExperience
from astrograph.orchestration.models import WorkflowContext, PermissionScope
from .contracts import OAGInput, content_hash
from .evidence_tools import EvidenceCheckTool
from .kernel import EvidenceDiagnosisKernel


@dataclass(frozen=True)
class CaseContext:
    case_id: str
    source_group_id: str
    operating_mode: str
    observed_end: datetime

    def __post_init__(self):
        if not all((self.case_id, self.source_group_id, self.operating_mode)):
            raise ValueError("case identity, independent source group and operating mode required")
        if self.observed_end.tzinfo is None or self.observed_end.utcoffset() is None:
            raise ValueError("absolute timezone-aware observation end is required for memory")


def _usable(feature: FeatureResult | None) -> bool:
    return bool(feature is not None and feature.value is not None
                and math.isfinite(feature.value) and feature.sample_count > 0
                and not feature.quality_flags)


def _symptoms(bundle) -> tuple[str, ...]:
    positive = {f.evidence_definition_id for f in bundle.evidence_facts if f.status.value == "SATISFIED"}
    if positive:
        return tuple(sorted(positive))
    return tuple(sorted({f.feature_definition_id for f in bundle.feature_results if _usable(f)})) or ("NO_OBSERVED_SYMPTOM",)


def _apply_features(bundle, tool: EvidenceCheckTool, requested: tuple[str, ...], values: Iterable[FeatureResult]):
    """Validate an entire batch before atomically admitting any observation.

    Thresholds, labels, model output and root input receipts are never tool outputs.
    An existing good measurement cannot be silently overwritten by a new value.
    """
    values = [FeatureResult.model_validate_json(f.model_dump_json()) for f in values]
    ids = [f.feature_definition_id for f in values]
    if len(set(ids)) != len(ids) or not set(ids) <= set(requested):
        raise ValueError("duplicate or unrequested feature result")
    if len({f.feature_result_id for f in values}) != len(values):
        raise ValueError("duplicate feature result identifier")
    definitions = {d.feature_definition_id: d for d in bundle.feature_definitions}
    before = {f.feature_definition_id: f for f in bundle.feature_results}
    changes = []
    for feature in values:
        if feature.feature_definition_id not in tool.feature_definition_ids:
            raise ValueError("feature not registered for this tool")
        definition = definitions.get(feature.feature_definition_id)
        if definition is None:
            raise ValueError("unknown feature definition")
        old = before.get(feature.feature_definition_id)
        fields = {definition.source_field_template}
        if old is not None:
            fields.add(old.resolved_field)
        expected_component = old.component_instance if old else bundle.analysis_component_instance
        ep = bundle.episode
        if (feature.episode_id != ep.episode_id or feature.run_id != ep.run_id
                or feature.paired_nominal_run_id != ep.paired_nominal_run_id
                or feature.component_instance != expected_component):
            raise ValueError("feature belongs to a different episode, reference or component")
        if (feature.window_start_s != ep.start_s or feature.window_end_s != ep.end_s):
            raise ValueError("feature window differs from the admitted episode")
        if (feature.unit != definition.unit or feature.resolved_field not in fields
                or feature.algorithm_version != definition.algorithm_version):
            raise ValueError("feature semantic or algorithm contract mismatch")
        if feature.input_data_sha256 not in tool.input_hashes:
            raise ValueError("unadmitted feature source")
        if not _usable(feature):
            raise ValueError("unusable or nonfinite feature result")
        for existing in bundle.feature_results:
            if existing.feature_result_id == feature.feature_result_id and existing.feature_definition_id != feature.feature_definition_id:
                raise ValueError("feature result identifier collides with another feature")
        if _usable(old):
            if (old.value, old.unit, old.window_start_s, old.window_end_s) != (
                    feature.value, feature.unit, feature.window_start_s, feature.window_end_s):
                raise ValueError("cannot overwrite an already admitted valid observation")
            continue
        changes.append(feature)
    if not changes:
        return bundle, []
    changed = {f.feature_definition_id for f in changes}
    after = {**before, **{f.feature_definition_id: f for f in changes}}
    results = [after[k] for k in sorted(after)]
    affected = {d.evidence_definition_id for d in bundle.evidence_definitions
                if ({d.feature_definition_id} | {c.feature_definition_id for c in d.and_feature_conditions}) & changed}
    previous_facts = {f.evidence_definition_id: f for f in bundle.evidence_facts}
    facts = [f for f in bundle.evidence_facts if f.evidence_definition_id not in affected]
    for d in bundle.evidence_definitions:
        if d.evidence_definition_id not in affected:
            continue
        previous = previous_facts.get(d.evidence_definition_id)
        if previous is not None and previous.status.value == "NOT_APPLICABLE":
            # Acquiring a number cannot change a rule's operating applicability.
            facts.append(previous)
            continue
        if d.applicable_condition_ids:
            active = set(bundle.episode.operating_condition.get("active_condition_ids", []))
            if not set(d.applicable_condition_ids) <= active:
                if previous is not None:
                    facts.append(previous)
                continue
        fact = evaluate_evidence(d, after.get(d.feature_definition_id), all_feature_results=results)
        if fact.episode_id == "UNKNOWN":
            fact = fact.model_copy(update={"episode_id": bundle.episode.episode_id})
        facts.append(fact)
    revised = bundle.model_copy(deep=True, update={"feature_results": results, "evidence_facts": facts})
    return revised, [f.feature_result_id for f in changes]


class AutonomousDiagnosisService:
    def __init__(self, kernel: EvidenceDiagnosisKernel, *, tools: tuple[EvidenceCheckTool, ...] = (),
                 memory: CaseMemoryService | None = None, max_checks: int = 3, plan_selector=None, check_executor=None, progress_callback=None):
        if not 0 <= max_checks <= 20:
            raise ValueError("max_checks must be between zero and twenty")
        if len({t.tool_id for t in tools}) != len(tools):
            raise ValueError("duplicate diagnostic check tool ID")
        self.kernel, self.tools, self.memory, self.max_checks = kernel, tools, memory, max_checks
        self.plan_selector = plan_selector
        self.check_executor = check_executor
        self.progress_callback = progress_callback

    def _plan_options(self, bundle, result, used, memory_hits):
        decision = result["diagnosis"]
        relevant = set(decision["candidate_fault_ids"]) | set(decision["alternatives"]) | set(decision["data_top_fault_ids"])
        # Do not restrict investigation to the data Top-1: executable hypotheses
        # with unresolved requirements are also legitimate competitors.
        relevant |= {c["fault_id"] for c in result["summary"]["candidates"]
                     if not c["contradicting_evidence_ids"]
                     and (c["missing_required_definition_ids"] or c["supporting_evidence_ids"])}
        if not relevant:
            relevant = set(decision["knowledge_top_fault_ids"])
        values = {f.feature_definition_id: f for f in bundle.feature_results}
        # Group the *condition*, not just the feature. Equal predicates shared
        # by candidates are support checks, not advertised as discriminators.
        conditions: dict[tuple, set[str]] = {}
        needed: set[str] = set()
        for d in bundle.evidence_definitions:
            if d.target_fault_id not in relevant:
                continue
            rows = [(d.feature_definition_id, d.comparison.value, d.threshold, d.threshold_unit)]
            rows += [(c.feature_definition_id, c.comparison.value, c.threshold, c.threshold_unit) for c in d.and_feature_conditions]
            key = (d.role.value, tuple(sorted(rows)))
            conditions.setdefault(key, set()).add(d.target_fault_id)
            needed.update(fid for fid, *_ in rows if not _usable(values.get(fid)))
        discriminators = {fid for key, targets in conditions.items() if targets != relevant
                          for fid, *_ in key[1] if fid in needed}
        advised = {check["tool_id"] for hit in memory_hits for check in hit["case"].get("checks", [])
                   if check["outcome"] == "ADMITTED"}
        options = []
        for tool in self.tools:
            request = tuple(sorted(tool.feature_definition_ids & needed))
            if tool.tool_id in used or not request:
                continue
            score = (len(set(request) & discriminators), len(request), int(tool.tool_id in advised), -tool.cost)
            options.append((score, tool.tool_id, tool, request))
        options.sort(key=lambda x: (tuple(-v for v in x[0]), x[1]))
        return [(tool, requested, {
            "goal": "DISTINGUISH_CANDIDATES" if set(requested) & discriminators else "FILL_EVIDENCE_GAP",
            "competing_fault_ids": sorted(relevant),
            "requested_feature_definition_ids": list(requested),
            "memory_advised": tool.tool_id in advised,
            "memory_is_diagnostic_vote": False,
        }) for _, _, tool, requested in options]

    def _plan(self, bundle, result, used, memory_hits, history=()):
        options = self._plan_options(bundle, result, used, memory_hits)
        if not options:
            return None
        if self.plan_selector is not None:
            # The planner can reorder only already eligible checks. It cannot
            # submit measurements, alter evidence rules or authorize new tools.
            from copy import deepcopy
            offered = [{"tool_id":tool.tool_id,"tool_version":tool.tool_version,
                        "feature_definition_ids":list(requested),"objective":rationale}
                       for tool,requested,rationale in options]
            try:
                import inspect
                extra={}
                parameters=inspect.signature(self.plan_selector).parameters
                if "history" in parameters:
                    extra={"history":deepcopy(list(history)),"remaining_checks":self.max_checks-len(used)}
                selected = self.plan_selector(options=deepcopy(offered), diagnosis=deepcopy(result["diagnosis"]),**extra)
                for option in options:
                    if option[0].tool_id == selected:
                        return option
            except Exception:
                pass  # Deterministic fallback never changes diagnostic evidence.
        return options[0]

    def run(self, data: OAGInput, *, context: WorkflowContext, case: CaseContext | None = None) -> dict:
        required = {PermissionScope.RUN_DIAGNOSIS, PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE}
        if not required <= set(context.permissions):
            raise PermissionError("diagnostic read and run permissions required")
        data = OAGInput.model_validate_json(data.model_dump_json())
        effective = data.bundle.model_copy(deep=True)
        root_hash = content_hash(data.model_dump(mode="json"))
        root_receipt = (data.three_task_result.evidence_projection.model_dump(mode="json")
                        if data.three_task_result and data.three_task_result.evidence_projection else None)
        trace, revisions, checks = [], [], []
        hits = []
        memory_status = {"retrieval": "NOT_CONFIGURED", "write": "NOT_CONFIGURED", "snapshot": None}
        if self.memory is not None and case is not None:
            try:
                snapshot = self.memory.snapshot()
                hits = self.memory.retrieve(
                    _symptoms(effective), operating_mode=case.operating_mode, snapshot=snapshot,
                    cutoff_at=case.observed_end, exclude_case_id=case.case_id,
                    exclude_source_group_id=case.source_group_id, context=context,
                    knowledge_version=self.kernel.knowledge_version,
                )
                memory_status.update(retrieval="SUCCEEDED", snapshot=snapshot,
                                     retrieved_case_ids=[x["case"]["case_id"] for x in hits])
            except Exception as exc:
                memory_status.update(retrieval="UNAVAILABLE", retrieval_error=type(exc).__name__)
        initial = self.kernel.evaluate(data, bundle=effective)
        result = initial
        if self.progress_callback:
            from copy import deepcopy
            self.progress_callback(deepcopy(result), [], "INITIAL_DIAGNOSIS")
        used = set()
        stop = "CHECK_BUDGET_EXHAUSTED"
        for index in range(self.max_checks + 1):
            status = result["diagnosis"]["status"]
            if status in {"SUPPORTED_SINGLETON", "NORMAL_REFERENCE_MATCH", "BLOCKED"}:
                stop = "DECISION_REACHED" if status != "BLOCKED" else "HARD_CONFLICT_REQUIRES_REVIEW"
                break
            if index >= self.max_checks:
                break
            plan = self._plan(effective, result, used, hits, trace)
            if plan is None:
                stop = "NO_AVAILABLE_DISCRIMINATING_CHECK"
                break
            tool, requested, rationale = plan
            used.add(tool.tool_id)
            item = {"round": index, "tool_id": tool.tool_id, "tool_version": tool.tool_version,
                    "plan": rationale, "before_decision": result["diagnosis"]["status"],
                    "before_selected": result["diagnosis"]["selected_fault_id"]}
            try:
                before_hash = content_hash(effective.model_dump(mode="json"))
                values = self.check_executor(tool,requested) if self.check_executor else tool.handler(requested)
                updated, admitted = _apply_features(effective, tool, requested, values)
                if not admitted:
                    item.update(outcome="NO_CHANGE", admitted_evidence_ids=[])
                    checks.append(CheckExperience(tool_id=tool.tool_id, feature_definition_ids=requested, outcome="NO_CHANGE"))
                    trace.append(item)
                    if self.progress_callback:
                        from copy import deepcopy
                        self.progress_callback(deepcopy(result),deepcopy(trace),"NO_NEW_EVIDENCE")
                    # An unproductive tool does not prevent trying a different
                    # registered check; every tool can execute at most once.
                    continue
                revised_result = self.kernel.evaluate(data, bundle=updated)
                effective = updated
                revision = {"previous_bundle_sha256": before_hash,
                            "effective_bundle_sha256": content_hash(effective.model_dump(mode="json")),
                            "tool_id": tool.tool_id, "tool_version": tool.tool_version,
                            "admitted_input_sha256": sorted(tool.input_hashes),
                            "feature_result_ids": admitted,
                            "model_prediction_recomputed": False,
                            "original_model_receipt_rewritten": False}
                revisions.append(revision)
                result = revised_result
                item.update(outcome="ADMITTED", admitted_evidence_ids=admitted,
                            after_decision=result["diagnosis"]["status"],
                            after_selected=result["diagnosis"]["selected_fault_id"])
                checks.append(CheckExperience(tool_id=tool.tool_id, feature_definition_ids=requested,
                                              outcome="ADMITTED", evidence_ids=tuple(admitted)))
            except Exception as exc:
                item.update(outcome="FAILED", error_type=type(exc).__name__, error=str(exc))
                checks.append(CheckExperience(tool_id=tool.tool_id, feature_definition_ids=requested, outcome="FAILED"))
            trace.append(item)
            if self.progress_callback:
                from copy import deepcopy
                self.progress_callback(deepcopy(result),deepcopy(trace),"CHECK_COMPLETED")
        if self.memory is not None and case is not None:
            try:
                pending = PendingCase(
                    case_id=case.case_id, source_group_id=case.source_group_id, producer_actor=context.actor,
                    symptom_ids=_symptoms(data.bundle), operating_mode=case.operating_mode,
                    proposed_fault_id=result["diagnosis"]["selected_fault_id"],
                    evidence_ids=tuple(sorted({f.evidence_fact_id for f in effective.evidence_facts})) or (root_hash,),
                    observed_end=case.observed_end, end_reason="completed", output_valid=True,
                    checks=tuple(checks), decision_status=result["diagnosis"]["status"],
                    decision_sha256=content_hash(result["diagnosis"]), knowledge_version=self.kernel.knowledge_version,
                )
                receipt = self.memory.stage_case(pending, event_id="diagnostic-stage:" + case.case_id, context=context)
                memory_status.update(write="STAGED_PENDING_INDEPENDENT_CONFIRMATION", receipt=receipt)
            except Exception as exc:
                memory_status.update(write="UNAVAILABLE", write_error=type(exc).__name__)
        return {"schema_version": "astrograph-autonomous-diagnosis.v1", "status": "SUCCEEDED",
                "planner": ("BOUNDED_ASSISTED_CHECK_SELECTION.v1" if self.plan_selector else "DETERMINISTIC_EVIDENCE_GAP_PLANNER.v1"),
                "request_sha256": root_hash, "knowledge_version": self.kernel.knowledge_version,
                "initial_diagnosis": initial["diagnosis"], **result,
                "checks_executed": len(trace), "planning_trace": trace, "stopping_reason": stop,
                "evidence_revisions": revisions, "effective_evidence_bundle": effective.model_dump(mode="json"),
                "original_model_evidence_projection": root_receipt, "memory": memory_status,
                "confirmed_memory_is_diagnostic_vote": False, "control_actions_authorized": False}
