"""Bind native model and evidence projections derived from the same paired window."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from astrograph.data_driven import ModelInferenceRequest, ObservationScope
from astrograph.assurance import canonical_sha256
from .contracts import OAGInput, content_hash
from .three_task import EvidenceProjectionReceipt


def from_native_frames(*, builder, event, source: pd.DataFrame, nominal: pd.DataFrame,
                       provider, registry, **options) -> OAGInput:
    """Build both projections together; never relabel one representation's hash.

    The model hashes a float32 channel matrix. Each feature hashes its own
    semantic-channel/current/nominal merge. Their hashes necessarily differ.
    This derivation receipt binds all of them to one caller-owned window; it is
    an integrity record, not external authentication of uploaded telemetry.
    """
    from .window_features import PACKAGE_ID, TERMINAL_PACKAGE_ID, derive_window_statistics, derive_terminal_statistics
    from .sensor_checks import PACKAGE_ID as SENSOR_PACKAGE_ID, deferred_sensor_fields
    window_mechanisms = builder.package.package_id in {PACKAGE_ID, TERMINAL_PACKAGE_ID, SENSOR_PACKAGE_ID}
    schema = builder.package.source_schema_sha256
    if window_mechanisms:
        import json
        from pathlib import Path
        schema = json.loads((Path(__file__).resolve().parents[3] /
            'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text(encoding="utf-8"))['canonical_sha256']
    source = project_registered_telemetry(source, schema_sha256=schema)
    nominal = project_registered_telemetry(nominal, schema_sha256=schema)
    length = int(provider.manifest.input_contract['window_length'])
    if len(source) < length:
        raise ValueError('source is shorter than the native model window')
    window = source.iloc[-length:].copy()
    start, end = float(window.time_s.iloc[0]), float(window.time_s.iloc[-1])
    paired = nominal.loc[nominal.time_s.between(start, end)].copy()
    statistics = derive_terminal_statistics if builder.package.package_id == TERMINAL_PACKAGE_ID else derive_window_statistics
    evidence_source, evidence_nominal = (statistics(window, paired)
        if window_mechanisms else (window, paired))
    if builder.package.package_id == SENSOR_PACKAGE_ID:
        evidence_source, evidence_nominal = deferred_sensor_fields(evidence_source, evidence_nominal)
    bundle, gate = builder.build_from_frames(event, source=evidence_source, nominal=evidence_nominal)
    if not gate.passed:
        raise ValueError('native paired evidence data gate failed')
    if bundle.evaluation_metadata not in ({}, {'eligible_for_model_input': False}):
        raise ValueError('native builder supplied unexpected evaluation metadata')
    # This builder marker prohibits feeding evidence into the telemetry model.
    # The model above/below receives only registered raw channels; OAG receives
    # a separate projection with the evaluator-only marker removed.
    bundle = bundle.model_copy(update={'evaluation_metadata': {}})
    channels = list(provider.selected_channels)
    values = window[channels].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('native model window contains non-finite values')
    payload = {'channel_order': channels, 'telemetry_window': values.tolist()}
    model_hash = canonical_sha256(payload)
    frame_hash = lambda frame: hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    source_hash, nominal_hash = frame_hash(window), frame_hash(paired)
    request = ModelInferenceRequest(
        episode_id=bundle.episode.episode_id, model_id=provider.manifest.model_id,
        model_version=provider.manifest.model_version,
        feature_schema_id=provider.manifest.feature_schema_ids[0],
        input_data_sha256=model_hash, registered_fault_ids=sorted(provider.runtime_fault_ids),
        payload=payload, observation_scope=ObservationScope(
            scope_id='OAG-WINDOW-'+source_hash[:20], source_id='paired-telemetry-window',
            window_start_s=start, window_end_s=end, source_data_sha256=source_hash))
    prediction = registry.predict(request)
    receipt = EvidenceProjectionReceipt(
        source_window_sha256=source_hash, nominal_window_sha256=nominal_hash,
        model_input_sha256=model_hash, evidence_bundle_sha256=content_hash(bundle.model_dump(mode='json')),
        feature_input_hashes={f.feature_result_id:f.input_data_sha256 for f in bundle.feature_results},
        model_artifact_sha256=provider.model_artifact_sha256,
        evidence_package_sha256=builder.package_sha256,
        window_start_s=start, window_end_s=end)
    if window_mechanisms:
        options.setdefault('resolve_supported_ties', True)
        options.setdefault('resolve_unsupported_top', True)
        options.setdefault('complementary_diagnosis_enabled', True)
        options.setdefault('mechanism_confirmation_enabled',
            options['complementary_diagnosis_enabled'] and builder.package_sha256 in
            {'6ee2ab138a1001e00db509796eef68df6540c93fb457f543aa6bd9a24ca74148', '27c25dcdcf2deb05c475ae32132a6e4c137658c714afd4c22e90262bc7deecef'})
    return OAGInput.from_prediction(bundle, prediction, evidence_projection=receipt,
        registered_fault_ids=sorted(set(provider.runtime_fault_ids) | {d.target_fault_id for d in bundle.evidence_definitions}), **options)


def write_package_graph(package_path, output):
    """Project frozen configured definitions into a session graph, never Active KG."""
    import csv,json
    from pathlib import Path
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
    builder=ConfiguredRuntimeEvidenceBuilder(Path(package_path))
    definitions=builder.package.evidence_definitions
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'nodes.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['node_id','node_type','name','properties_json'])
        for fault in sorted({d.target_fault_id for d in definitions}):w.writerow([fault,'FaultMode',fault,'{}'])
        for d in definitions:w.writerow([d.evidence_definition_id,'EvidenceDefinition',d.name,json.dumps(d.model_dump(mode='json'))])
    relation={'REQUIRED':'REQUIRES_EVIDENCE','SUPPORTING':'SUPPORTED_BY','COUNTER':'CONTRADICTED_BY','EXCLUSION':'EXCLUDES'}
    with (output/'edges.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['edge_id','source_id','relation','target_id','properties_json'])
        for d in definitions:w.writerow(['EDGE-'+content_hash(d.model_dump(mode='json'))[:20],d.target_fault_id,relation[d.role.value],d.evidence_definition_id,'{}'])
    (output/'provenance.json').write_text(json.dumps({'package_sha256':builder.package_sha256,'source':'CONFIGURED_EVIDENCE_DEFINITIONS','active_kg_modified':False},indent=2)+'\n', encoding='utf-8')
    return output


def project_registered_telemetry(frame: pd.DataFrame, *, schema_sha256: str):
    """Accept one pinned recorder superset; unknown layouts remain rejected."""
    import json
    from pathlib import Path
    from astrograph.evidence.configured_runtime import column_schema_sha256
    current = column_schema_sha256(list(frame.columns))
    if current == schema_sha256:
        return frame
    path = Path(__file__).resolve().parents[3] / 'configs/datasets/adcs_v5_2_2/telemetry_projection.json'
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = config['canonical_columns']
    if schema_sha256 != config['canonical_sha256'] or column_schema_sha256(canonical) != schema_sha256:
        raise ValueError('no registered projection for this evidence schema')
    if current not in config['accepted_input_sha256'] or len(set(frame.columns)) != len(frame.columns):
        raise ValueError('unregistered telemetry layout')
    if not set(canonical) <= set(frame.columns):
        raise ValueError('projection cannot supply missing channels')
    return frame.loc[:, canonical].copy()
