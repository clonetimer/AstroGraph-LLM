from __future__ import annotations
import json
from pathlib import Path
from astrograph.agents.mission.replay import ReplayManifest,run_replay


def test_phm_replay_is_explicitly_offline_and_truth_aware(tmp_path:Path):
    dataset=tmp_path/'dataset';dataset.mkdir();(dataset/'sample_index.csv').write_text('sample_id,split\n',encoding='utf-8')
    project=tmp_path/'project';(project/'scripts/integration').mkdir(parents=True);(project/'scripts/integration/run_phm_v60_regression.py').write_text('# fixture',encoding='utf-8')
    workspace=tmp_path/'workspace';workspace.mkdir()
    def worker(args):
        out=Path(args[args.index('--output-dir')+1]);out.mkdir(parents=True)
        (out/'phm_v60_regression_report.json').write_text(json.dumps({'model_strategy':'promoted_extra_trees','selected_chain_sample':'S','split_audit':{'status':'PASS'},'three_stage_result':{'detection_metrics':{'macro_f1':.9},'localization_metrics':{'macro_f1':1},'classification_metrics_deployed':{'macro_f1':1},'end_to_end_metrics':{'macro_f1':.96}},'chains':{'machine_learning_rule':{'status':'SUCCEEDED'}}}),encoding='utf-8')
    out=run_replay(project_root=project,workspace_root=workspace,dataset_root=dataset,manifest=ReplayManifest(dataset_kind='phm_v6_retrospective'),run_subprocess=worker)
    assert out['status']=='SUCCEEDED'
    assert out['dataset_source_kind']=='DIRECTORY'
    assert out['offline_evaluation_truth_used'] is True
    assert out['online_diagnosis_claim_allowed'] is False


def test_adcs_replay_uses_same_contract_but_zip_worker(tmp_path:Path):
    dataset=tmp_path/'ADCS.zip';dataset.write_bytes(b'fixture')
    project=tmp_path/'project';(project/'scripts').mkdir(parents=True);(project/'scripts/evaluate_three_task_agent_zip.py').write_text('# fixture',encoding='utf-8')
    workspace=tmp_path/'workspace';workspace.mkdir()
    def worker(args):
        out=Path(args[args.index('--output')+1]);out.mkdir(parents=True)
        (out/'report.json').write_text(json.dumps({
            'schema_version':'x','windows':12,'active_fault_windows':8,'normal_windows':4,
            'semantic_inference_sha256':'a'*64,'inference_sha256':'b'*64,'cohort':'development',
            'model_weights_changed':False,'fusion_weights_changed':False,'tool_calls':2,'eager_tool_calls':3,
        }),encoding='utf-8')
    out=run_replay(project_root=project,workspace_root=workspace,dataset_zip=dataset,
        manifest=ReplayManifest(dataset_kind='adcs_v5_2_2_retrospective',adcs_cohort='development'),run_subprocess=worker)
    assert out['status']=='SUCCEEDED'
    assert out['dataset_source_kind']=='ZIP'
    assert out['semantic_inference_sha256']=='a'*64
    assert out['offline_evaluation_truth_used'] is True
    assert out['online_diagnosis_claim_allowed'] is False
