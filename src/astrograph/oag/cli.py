"""Local CLI for the native OAG pipeline and label-isolated evaluator."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.fusion import FaultRanking, RankingSource
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.runtime import RuntimeRequest, RuntimeStatus
from astrograph.benchmark import evaluate
from .contracts import OAGInput, content_hash
from .runtime import OAG_PIPELINE_ID, build_csv_oag_runtime
from .paperseek_http import PaperSeekConfig, PaperSeekHTTPClient


def _write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid corrupt partial output if interrupted during serialization.
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    import tempfile
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(raw)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def demo_input(project_root: Path) -> OAGInput:
    """Replay existing evidence, explicitly removing evaluation-only metadata.

    The upstream model is absent in this demo. The DATA_DRIVEN branch is empty,
    not copied from the KG and not created from the reference fault label.
    """
    path = project_root / "tests/fixtures/evidence/reaction_wheel_friction_increase/diagnosis/episode_evidence_bundle.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["evaluation_metadata"] = {}
    bundle = EpisodeEvidenceBundle.model_validate(data)
    ranking = FaultRanking(ranking_id="UPSTREAM-NOT-CONFIGURED", episode_id=bundle.episode.episode_id,
                           source=RankingSource.DATA_DRIVEN, producer_id="absent-upstream-model",
                           producer_version="1", candidates=[],
                           claim_boundary="Empty branch: demo does not load a diagnosis model.")
    return OAGInput(bundle=bundle, diagnostic_ranking=ranking)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "demo"):
        p = commands.add_parser(name)
        p.add_argument("--project-root", type=Path, default=Path.cwd())
        p.add_argument("--graph-dir", type=Path)
        p.add_argument("--output", type=Path, required=True)
        if name == "run":
            p.add_argument("--input", type=Path, required=True)
        p.add_argument("--paperseek-url")
        p.add_argument("--llm-base-url")
        p.add_argument("--llm-model")
        p.add_argument("--extract-ontology", action="store_true")
        p.add_argument("--require-llm", action="store_true")
        p.add_argument("--require-paperseek", action="store_true")
        p.add_argument("--timeout", type=float, default=120)
        p.add_argument("--harness-dir", type=Path, help="Persist role, handoff and model-budget receipts")
        p.add_argument("--request-id", default=None)
        p.add_argument("--review-evidence", action="store_true")
    p = commands.add_parser("prepare-input", help="Bind a native three-task prediction to an evidence bundle")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("prepare-native", help="Build evidence and three-task output from one paired telemetry window")
    p.add_argument('--mechanism-confirmation', action=argparse.BooleanOptionalAction, default=None,
                   help='Use the version-pinned RW mechanism/classification agreement policy; --no-mechanism-confirmation preserves the detector veto')
    p.add_argument('--domain', choices=['adcs', 'phm'], required=True)
    p.add_argument('--evidence-profile', choices=['rw-window', 'legacy', 'zero-authority', 'terminal8-zero-authority', 'sensor-checks'],
                   help='ADCS defaults to window RW rules plus preserved gyro/star rules; PHM uses legacy.')
    p.add_argument('--graph-output', type=Path)
    p.add_argument('--degraded-evidence-output', type=Path, help='Separate candidate-only receipt on missing ADCS observations; preparation still fails without model input.')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--nominal', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--project-root', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('diagnose-native', help='Paired ADCS model and physical diagnosis, including qualified missing-data inference')
    for name in ['source','nominal','model','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p = commands.add_parser("evaluate")
    p.add_argument("--records", type=Path, required=True, help="JSON array of BenchmarkRecord, truth stays here")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=3)
    from .agent_cli import add_agent_commands, execute_agent_command
    add_agent_commands(commands)
    args = parser.parse_args(argv)
    try:
        if args.command in {'diagnose-agent', 'memory-confirm'}:
            return execute_agent_command(args)
        if args.command == 'diagnose-native':
            from .physical_native import diagnose_files
            result=diagnose_files(source_path=args.source,nominal_path=args.nominal,
                model_path=args.model,project_root=args.project_root,output_path=args.output)
            _write(args.output,result)
            return 0
        if args.command == 'prepare-native':
            if args.mechanism_confirmation and (args.domain != 'adcs' or args.evidence_profile == 'legacy'):
                raise ValueError('mechanism confirmation requires the native ADCS window evidence profile')
            if args.domain != 'adcs' and args.evidence_profile not in (None, 'legacy'):
                raise ValueError('rw-window evidence requires the ADCS domain')
            if args.mechanism_confirmation and args.evidence_profile in ('terminal8-zero-authority', 'sensor-checks'):
                raise ValueError('experimental profiles do not carry the legacy detector-override qualification')
            if args.degraded_evidence_output and (args.domain != 'adcs' or args.evidence_profile == 'legacy'):
                raise ValueError('degraded mechanism evidence requires an ADCS window profile')
            import hashlib
            import pandas as pd
            from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
            from astrograph.simulation import SimulationRunCompleted
            from .native_bridge import from_native_frames
            if args.domain == 'adcs':
                from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package as loader, build_registry_from_adcs_three_task_provider as registry_builder
                package = 'adcs_v5_2_2'
            else:
                from astrograph.integrations.phm_three_task_torch import load_phm_three_task_torch_package as loader, build_registry_from_phm_three_task_provider as registry_builder
                package = 'phm_v6_0'
            filename = ('runtime_window_integrated_evidence.json'
                        if args.domain == 'adcs' and args.evidence_profile != 'legacy'
                        else 'runtime_evidence.json')
            evidence_package = args.project_root / 'configs/datasets' / package / filename
            if args.evidence_profile in ('zero-authority', 'terminal8-zero-authority'):
                evidence_package = args.project_root / 'configs/research' / (args.evidence_profile.replace('-', '_') + '.v1.json')
            if args.evidence_profile == 'sensor-checks':
                if args.degraded_evidence_output:
                    raise ValueError('sensor-check research profile has no qualified degraded-evidence fallback')
                evidence_package = args.project_root / 'configs/datasets/adcs_v5_2_2/runtime_sensor_checks.research.json'
            digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
            provider = loader(args.model, expected_sha256=digest, device='cpu')
            source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
            event = SimulationRunCompleted(event_id='OAG-'+source_hash[:20], run_id='OAG-'+source_hash[:20],
                run_dir=str(args.source.resolve().parent), nominal_run_dir=str(args.nominal.resolve().parent),
                runtime_id='oag-native-paired-frame.v1')
            try:
                data = from_native_frames(builder=ConfiguredRuntimeEvidenceBuilder(evidence_package),
                    event=event, source=pd.read_csv(args.source, low_memory=False), nominal=pd.read_csv(args.nominal, low_memory=False),
                    provider=provider, registry=registry_builder(provider), review_evidence=True,
                    **({'mechanism_confirmation_enabled': args.mechanism_confirmation}
                       if args.mechanism_confirmation is not None else {}))
            except ValueError:
                if args.degraded_evidence_output:
                    from .degraded import mechanism_only_fallback
                    schema=json.loads((args.project_root/'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text(encoding="utf-8"))['canonical_sha256']
                    fallback=mechanism_only_fallback(builder=ConfiguredRuntimeEvidenceBuilder(evidence_package),
                        event=event,source=pd.read_csv(args.source,low_memory=False),nominal=pd.read_csv(args.nominal,low_memory=False),
                        schema_sha256=schema,window_length=int(provider.manifest.input_contract['window_length']))
                    _write(args.degraded_evidence_output, fallback)
                raise
            _write(args.output, data.model_dump(mode='json'))
            if args.graph_output:
                from .native_bridge import write_package_graph
                write_package_graph(evidence_package, args.graph_output)
            return 0
        if args.command == "prepare-input":
            from astrograph.data_driven.models import DataDrivenPrediction
            bundle = EpisodeEvidenceBundle.model_validate_json(args.bundle.read_text(encoding="utf-8"))
            prediction = DataDrivenPrediction.model_validate_json(args.prediction.read_text(encoding="utf-8"))
            data = OAGInput.from_prediction(bundle, prediction, review_evidence=True)
            _write(args.output, data.model_dump(mode="json"))
            return 0
        if args.command == "evaluate":
            records = json.loads(args.records.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError("benchmark records must be an array")
            result = evaluate(records, k=args.top_k)
            _write(args.output, result)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        started = perf_counter()
        root = args.project_root.resolve()
        data = demo_input(root) if args.command == "demo" else OAGInput.model_validate_json(args.input.read_text(encoding="utf-8"))
        data = OAGInput.model_validate({**data.model_dump(),
                "require_llm": data.require_llm or args.require_llm,
                "review_evidence": data.review_evidence or args.review_evidence,
                "require_paperseek": data.require_paperseek or args.require_paperseek,
                "require_ontology_extraction": data.require_ontology_extraction or args.extract_ontology})
        graph = args.graph_dir or root / "benchmarks/knowledge_graph/online_extended"
        if bool(args.llm_base_url) != bool(args.llm_model):
            raise ValueError("--llm-base-url and --llm-model must be supplied together")
        llm = None
        if args.llm_base_url:
            config = OpenAICompatibleConfig(base_url=args.llm_base_url, model=args.llm_model,
                       api_key=os.environ.get("ASTROGRAPH_LLM_API_KEY", "not-set"),
                       timeout_s=args.timeout, tokenizer_mode="disabled")
            llm = OpenAICompatibleStructuredBackend(config)
        papers = PaperSeekHTTPClient(PaperSeekConfig(base_url=args.paperseek_url, timeout_s=args.timeout)) if args.paperseek_url else None
        if args.harness_dir:
            from .harness import execute_harness
            result = execute_harness(
                data, graph_dir=graph, output_root=args.harness_dir,
                request_id=args.request_id or "OAG-" + content_hash(data.model_dump(mode="json"))[:20],
                paperseek_client=papers, llm_backend=llm,
                ontology_backend=llm if args.extract_ontology else None)
            _write(args.output, result)
            print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False))
            return 0 if result["status"] == "SUCCEEDED" else 2
        runtime = build_csv_oag_runtime(graph, paperseek_client=papers, llm_backend=llm,
                                         ontology_backend=llm if args.extract_ontology else None)
        result = runtime.execute(RuntimeRequest(request_id="OAG-" + content_hash(data.model_dump(mode="json"))[:20],
                               pipeline_id=OAG_PIPELINE_ID, payload=data.model_dump(mode="json")))
        output = result.model_dump(mode="json")
        output["wall_latency_s"] = perf_counter() - started
        output["run_mode"] = "FIXTURE_REPLAY_KG_ONLY_NO_PERFORMANCE_CLAIM" if args.command == "demo" else "USER_SUPPLIED_EVIDENCE"
        _write(args.output, output)
        print(json.dumps({"status": result.status.value, "output": str(args.output.resolve()),
                          "summary_status": result.output.get("summary", {}).get("status"),
                          "error": result.error.model_dump() if result.error else None}, ensure_ascii=False))
        return 0 if result.status == RuntimeStatus.SUCCEEDED else 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
