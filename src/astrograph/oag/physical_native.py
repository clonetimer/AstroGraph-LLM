"""Executable paired ADCS diagnosis with independent physical and model routes."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
from astrograph.simulation import SimulationRunCompleted
from astrograph.runtime import RuntimeRequest
from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package,build_registry_from_adcs_three_task_provider
from .native_bridge import from_native_frames,write_package_graph,project_registered_telemetry
from .runtime import build_csv_oag_runtime,OAG_PIPELINE_ID
from .physical_diagnosis import diagnose_physical_complement, RW_FIELDS


def diagnose_files(*,source_path,nominal_path,model_path,project_root,output_path):
    root=Path(project_root);output=Path(output_path)
    if output.exists():raise ValueError('diagnosis output already exists')
    schema=json.loads((root/'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text(encoding="utf-8"))['canonical_sha256']
    source=project_registered_telemetry(pd.read_csv(source_path,low_memory=False),schema_sha256=schema)
    nominal=project_registered_telemetry(pd.read_csv(nominal_path,low_memory=False),schema_sha256=schema)
    model=Path(model_path);digest=hashlib.sha256(model.read_bytes()).hexdigest()
    provider=load_adcs_three_task_torch_package(model,expected_sha256=digest,device='cpu')
    length=int(provider.manifest.input_contract['window_length'])
    if length!=20:raise ValueError('physical complement requires the qualified 20-point model window')
    readiness = paired_route_readiness(source, nominal, list(provider.selected_channels), window_length=length)
    available = readiness['model_inputs_ready'] and readiness['paired_mechanism_ready']
    original=None;summary=None
    if available:
        output.parent.mkdir(parents=True,exist_ok=True)
        package=root/'configs/research/zero_authority.v1.json'
        graph=write_package_graph(package,output.parent/(output.stem+'-evidence-graph'))
        event=SimulationRunCompleted(event_id='PHYSICAL-NATIVE',run_id='PHYSICAL-NATIVE',run_dir='opaque',nominal_run_dir='paired-reference',runtime_id='physical-complement.v1')
        data=from_native_frames(builder=ConfiguredRuntimeEvidenceBuilder(package),event=event,source=source,nominal=nominal,
            provider=provider,registry=build_registry_from_adcs_three_task_provider(provider),review_evidence=True)
        original=build_csv_oag_runtime(graph).execute(RuntimeRequest(request_id='PHYSICAL-NATIVE',pipeline_id=OAG_PIPELINE_ID,payload=data.model_dump(mode='json'))).model_dump(mode='json')
        if original['status']!='SUCCEEDED':raise ValueError('native OAG execution failed')
        summary=original['output']['summary']
    diagnosis=diagnose_physical_complement(source,nominal,summary)
    return {'schema_version':'astrograph-native-physical-diagnosis.v1','diagnosis':diagnosis,
         'model_status': ('EXECUTED' if available else
            'UNAVAILABLE_MISSING_OBSERVATIONS' if not readiness['model_inputs_ready'] else
            'NOT_EXECUTED_EVIDENCE_UNAVAILABLE'),
        'route_readiness':readiness,
        'model_sha256':digest,'original_oag_result':original,'llm_invoked':False,
        'claim_boundary':'Research diagnosis from registered paired observations; physical evidence can operate without model imputation. Model results and physical selections remain separately traceable.'}


def paired_route_readiness(source, nominal, selected_channels, *, window_length=20):
    """Check each route's actual inputs without filling data or catching unrelated errors.

    A complete model tensor does not guarantee that paired mechanism statistics can
    be built. Missing RW observations on either side only disable that combined
    model+mechanism path; the partial-pair physical route remains independently usable.
    Structural/time/schema errors still fail closed.
    """
    if window_length != 20 or len(source) < window_length:
        raise ValueError('twenty observations required')
    window=source.iloc[-window_length:]
    times=window.time_s.to_numpy(float)
    if not np.isfinite(times).all() or not np.all(np.diff(times)>0):
        raise ValueError('ordered finite observation times required')
    paired=nominal.loc[nominal.time_s.between(times[0], times[-1])]
    if not np.array_equal(times, paired.time_s.to_numpy(float)):
        raise ValueError('twenty aligned scheduled observations are required')
    def missing(frame, fields):
        absent=[field for field in fields if field not in frame]
        if absent:
            raise ValueError('registered observation fields are missing: ' + ', '.join(absent))
        return [field for field in fields if not np.isfinite(frame[field].to_numpy(float)).all()]
    model_missing=missing(window, selected_channels)
    source_missing=missing(window, RW_FIELDS)
    reference_missing=missing(paired, RW_FIELDS)
    return {'model_inputs_ready':not model_missing,
            'paired_mechanism_ready':not source_missing and not reference_missing,
            'model_missing_fields':model_missing,
            'mechanism_source_missing_fields':source_missing,
            'mechanism_reference_missing_fields':reference_missing,
            'imputation_performed':False}
