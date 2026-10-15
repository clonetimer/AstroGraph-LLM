"""Explicit autonomous-diagnosis and independent case-confirmation CLI actions."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from astrograph.memory.service import CaseMemoryService, MemoryPolicy
from astrograph.orchestration.models import WorkflowContext, PermissionScope
from .agent import AutonomousDiagnosisService, CaseContext
from .contracts import OAGInput
from .evidence_tools import MeasurementBank, tools_from_measurements
from .kernel import EvidenceDiagnosisKernel


def add_agent_commands(commands):
    p = commands.add_parser("diagnose-agent", help="Evidence-aware fusion, autonomous read-only checks, and optional pending-case memory")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--graph-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--measurements", type=Path, help="Host-admitted raw series, not answers or evidence statuses")
    p.add_argument("--paired-source", type=Path, help="Raw paired telemetry for the opt-in sensor-check research profile")
    p.add_argument("--paired-nominal", type=Path, help="Matching causal reference, bound to the input receipt")
    p.add_argument("--max-checks", type=int, default=3)
    p.add_argument("--memory-db", type=Path)
    p.add_argument("--memory-policy", type=Path)
    p.add_argument("--case-id")
    p.add_argument("--source-group-id")
    p.add_argument("--operating-mode")
    p.add_argument("--observed-end-utc", help="Absolute timezone-aware end; do not substitute simulation-relative time")
    p.add_argument("--actor", default="diagnostic-agent")
    p = commands.add_parser("memory-confirm", help="Record an independently reviewed truth; never called automatically by the Agent")
    for name in ("case-id", "verified-fault-id", "truth-source-ref", "reviewer", "verified-at", "available-at", "event-id"):
        p.add_argument("--" + name, required=True)
    for name in ("memory-db", "memory-policy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-snapshot", type=int, required=True)


def execute_agent_command(args):
    from .cli import _write
    if args.command == "memory-confirm":
        policy = MemoryPolicy.model_validate_json(args.memory_policy.read_text(encoding="utf-8"))
        service = CaseMemoryService(args.memory_db, policy)
        context = WorkflowContext(workflow_id="memory.independent-confirmation", run_id=args.event_id,
                                  actor=args.reviewer, permissions=[PermissionScope.REVIEW_MEMORY])
        receipt = service.confirm_case(
            case_id=args.case_id, event_id=args.event_id, verified_fault_id=args.verified_fault_id,
            truth_source_ref=args.truth_source_ref, verified_at=datetime.fromisoformat(args.verified_at),
            available_at=datetime.fromisoformat(args.available_at), expected_snapshot=args.expected_snapshot,
            context=context,
        )
        _write(args.output, receipt)
        return 0
    data = OAGInput.model_validate_json(args.input.read_text(encoding="utf-8"))
    tools = ()
    if args.measurements:
        bank = MeasurementBank.model_validate_json(args.measurements.read_text(encoding="utf-8"))
        tools = tools_from_measurements(bank, data.bundle)
    if bool(args.paired_source) != bool(args.paired_nominal):
        raise ValueError("sensor checks require both --paired-source and --paired-nominal")
    if args.paired_source:
        import pandas as pd
        from .sensor_checks import sensor_check_tools
        sensor_tools = sensor_check_tools(data, source=pd.read_csv(args.paired_source, low_memory=False),
                                         nominal=pd.read_csv(args.paired_nominal, low_memory=False))
        if not sensor_tools:
            raise ValueError("input has no published sensor-check definitions; prepare with sensor-checks research profile")
        tools += sensor_tools
    memory, case = None, None
    permissions = [PermissionScope.RUN_DIAGNOSIS, PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE]
    if args.memory_db:
        if not all((args.memory_policy, args.case_id, args.source_group_id, args.operating_mode, args.observed_end_utc)):
            raise ValueError("memory requires policy, case ID, independent source group, mode and absolute observation end")
        policy = MemoryPolicy.model_validate_json(args.memory_policy.read_text(encoding="utf-8"))
        memory = CaseMemoryService(args.memory_db, policy)
        case = CaseContext(args.case_id, args.source_group_id, args.operating_mode,
                           datetime.fromisoformat(args.observed_end_utc))
        permissions.extend([PermissionScope.READ_MEMORY, PermissionScope.WRITE_MEMORY])
    context = WorkflowContext(workflow_id="diagnosis.autonomous", run_id=args.case_id or data.bundle.episode.episode_id,
                              actor=args.actor, permissions=permissions)
    result = AutonomousDiagnosisService(EvidenceDiagnosisKernel(args.graph_dir), tools=tools,
                                       memory=memory, max_checks=args.max_checks).run(data, context=context, case=case)
    _write(args.output, result)
    return 0
