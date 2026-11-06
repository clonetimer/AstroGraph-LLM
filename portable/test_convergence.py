"""Regression tests for concrete V2.5 delivery and recovery defects."""
import io
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from test_workbench import app, file_input, connection, wait_job
from astrograph.integrations.external_diagnosis_analysis import _digest, _source_id
from astrograph.integrations.long_report_analysis import run_hierarchical_analysis


def bundle_for(task_id='long-task'):
    s = {'filename':'long.txt','role':'diagnosis_results','locator':'line:1',
         'sha256':'a'*64, 'text':'独立观测。'*14000}
    s['source_id'] = _source_id(task_id, s)
    return {'schema_version':'astrograph-external-diagnosis-input.v1','task_id':task_id,
            'input_sha256':_digest([s]), 'sources':[s], 'character_count':len(s['text'])}


class RecordingModel:
    def __init__(self, long_summary=False):
        self.payloads = []
        self.long_summary = long_summary
    def generate_structured(self, *, agent_id, output_model, payload):
        self.payloads.append(payload)
        sid = payload['sources'][0]['source_id']
        chunk = '-chunk-' in payload['task_id']
        return {'summary':('摘要'*2400+'END_OF_SUMMARY' if chunk and self.long_summary else '温度异常记录'),
                'findings':[{'text':'上游观测', 'citation_ids':[sid]}], 'hypotheses':[],
                'information_gaps':[], 'recommended_review_actions':[],
                'common_findings':[] if chunk else [{'text':'共同观测', 'citation_ids':[sid]}],
                'report_disagreements':[] if chunk else [{'text':'时间未对齐', 'citation_ids':[sid]}],
                'complementary_evidence':[] if chunk else [{'text':'互补记录', 'citation_ids':[sid]}],
                'event_alignment':[] if chunk else [{'text':'需核实时间', 'citation_ids':[sid]}]}


def backup_bytes(job_id, extra=None, status='RUNNING'):
    entries={f'jobs/{job_id}/job.json':json.dumps({'job_id':job_id,'status':status}).encode()}
    entries.update(extra or {})
    manifest={'schema_version':'astrograph-portable-backup.v1', 'files':[
        {'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        for name,data in entries.items()]}
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('backup.json',json.dumps(manifest))
        for name,data in entries.items(): z.writestr(name,data)
    return output.getvalue()


class ConvergenceTests(unittest.TestCase):
    def test_full_chunk_summaries_reach_synthesis_without_cutoff(self):
        model=RecordingModel(long_summary=True)
        with tempfile.TemporaryDirectory() as d:
            result=run_hierarchical_analysis(task_id='long-task',task='核查',input_bundle=bundle_for(),
                llm_backend=model,output_root=Path(d),execution_fingerprint='fp')
        self.assertEqual(result['generation']['status'],'SUCCEEDED')
        for source in model.payloads[-1]['sources']:
            self.assertIn('END_OF_SUMMARY',source['text'])
            self.assertEqual(json.loads(source['text'])['summary'][-14:], 'END_OF_SUMMARY'[-14:])

    def test_synthesis_comparison_citations_resolve_to_original_registry(self):
        bundle=bundle_for(); model=RecordingModel()
        with tempfile.TemporaryDirectory() as d:
            result=run_hierarchical_analysis(task_id='long-task',task='核查',input_bundle=bundle,
                llm_backend=model,output_root=Path(d))
        exp=result['model_explanation']; known={s['source_id'] for s in exp['sources']}
        for field in ['findings','common_findings','report_disagreements','complementary_evidence','event_alignment']:
            for item in exp[field]:
                self.assertTrue(set(item['citation_ids']) <= known,field)

    def test_corrupted_input_is_rejected_before_hierarchical_model_call(self):
        bundle=bundle_for(); bundle['sources'][0]['text'] += '篡改'
        model=RecordingModel()
        with tempfile.TemporaryDirectory() as d:
            result=run_hierarchical_analysis(task_id='long-task',task='核查',input_bundle=bundle,
                llm_backend=model,output_root=Path(d))
        self.assertEqual(result['generation']['status'],'FAILED')
        self.assertEqual(result['generation'].get('reason_code'),'INVALID_INPUT_BINDING')
        self.assertEqual(model.payloads,[])

    def test_model_failure_offers_retry_without_reimport(self):
        class FailedModel:
            def generate_structured(self,**kwargs): raise RuntimeError('test failure')
        with tempfile.TemporaryDirectory() as d:
            store=app.Workbench(Path(d))
            with mock.patch.object(app,'_backend',return_value=FailedModel()):
                job=store.create_job({'task':'核查','connection':connection(),'files':[file_input()]})
                saved=wait_job(store,job['job_id'])
            self.assertTrue(saved['job'].get('resumable'))
            original=(store._job_root(job['job_id'])/'input_original'/'诊断.txt').read_bytes()
            with mock.patch.object(app,'_backend',return_value=RecordingModel()):
                store.resume_job(job['job_id'],connection())
                recovered=wait_job(store,job['job_id'])
            self.assertEqual(recovered['job']['generation_status'],'SUCCEEDED')
            self.assertFalse(recovered['job'].get('resumable',False))
            self.assertEqual((store._job_root(job['job_id'])/'input_original'/'诊断.txt').read_bytes(),original)

    def test_backup_is_blocked_during_active_model_operation(self):
        with tempfile.TemporaryDirectory() as d:
            store=app.Workbench(Path(d)); store._operation.acquire()
            try:
                with self.assertRaises(app.WorkbenchError) as ctx: store.create_backup()
                self.assertEqual(ctx.exception.code,'MODEL_BUSY')
            finally: store._operation.release()

    def test_restore_is_blocked_during_active_model_operation(self):
        with tempfile.TemporaryDirectory() as d:
            store=app.Workbench(Path(d)); store._operation.acquire()
            try:
                with self.assertRaises(app.WorkbenchError) as ctx: store.restore_backup(backup_bytes('a'*32))
                self.assertEqual(ctx.exception.code,'MODEL_BUSY')
            finally: store._operation.release()
            self.assertEqual(store.list_jobs(),[])

    def test_restored_inflight_job_is_immediately_resumable_not_forever_running(self):
        with tempfile.TemporaryDirectory() as d:
            store=app.Workbench(Path(d)); store.restore_backup(backup_bytes('a'*32))
            self.assertEqual(store.get('a'*32)['status'],'INTERRUPTED')

    def test_restore_rejects_windows_path_aliases_before_any_write(self):
        job_id='a'*32
        for extra in [{f'jobs/{job_id}/input/A.txt':b'A',f'jobs/{job_id}/input/a.txt':b'B'},
                      {f'jobs/{job_id}/input/con.txt':b'x'},
                      {f'jobs/{job_id}/input/file.':b'x'},
                      {f'jobs/{job_id}/input//x.txt':b'x'}]:
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as d:
                store=app.Workbench(Path(d))
                with self.assertRaises(app.WorkbenchError): store.restore_backup(backup_bytes(job_id,extra))
                self.assertEqual(store.list_jobs(),[])

if __name__=='__main__': unittest.main(verbosity=2)
