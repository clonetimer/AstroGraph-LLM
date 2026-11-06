"""Delivery regressions: exercise state, archive, and synthesis behavior."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from test_workbench import app, connection, file_input
from astrograph.integrations.external_diagnosis_analysis import _digest, _source_id
from astrograph.integrations.long_report_analysis import run_hierarchical_analysis


def source_bundle(task_id='a'*32, text='原始诊断：温度异常。' * 1300):
    record = {'role': 'diagnosis_results', 'filename': '诊断.txt', 'locator': 'line:1',
              'sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': text}
    record['source_id'] = _source_id(task_id, record)
    return {'schema_version': 'astrograph-external-diagnosis-input.v1', 'task_id': task_id,
            'sources': [record], 'input_sha256': _digest([record]), 'character_count': len(text)}


def result(sid, summary='模型对原文的分析。'):
    return {'summary': summary, 'findings': [{'text': '报告描述异常。', 'citation_ids': [sid]}],
            'hypotheses': [], 'information_gaps': [], 'recommended_review_actions': [], 'review_actions': []}


def backup(files):
    out = io.BytesIO()
    manifest = {'schema_version': 'astrograph-portable-backup.v1', 'files': [
        {'path': path, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        for path, raw in files.items()]}
    with zipfile.ZipFile(out, 'w') as z:
        z.writestr('backup.json', json.dumps(manifest))
        for path, raw in files.items(): z.writestr(path, raw)
    return out.getvalue()


class ImmediateThread:
    def __init__(self, *, target, args=(), daemon=True, name=None): self.target, self.args = target, args
    def start(self): self.target(*self.args)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='astrograph-closeout-')
        self.root = Path(self.temp.name)
        self.store = app.Workbench(self.root)
    def tearDown(self):
        for thread in list(self.store._threads.values()):
            if hasattr(thread, 'join'): thread.join(5)
        self.temp.cleanup()

    def interrupted(self, bundle=None):
        bundle = bundle or source_bundle(text='温度异常。')
        settings = connection()
        job = {'job_id': bundle['task_id'], 'task': '分析原文', 'status': 'INTERRUPTED',
               'created_at': '2026-11-03T00:00:00Z', 'model': settings['model'], 'mode': settings['mode'],
               'execution_fingerprint': app._execution_fingerprint(settings),
               'input_sha256': bundle['input_sha256'], 'character_count': bundle['character_count']}
        root = self.store._job_root(job['job_id'])
        app._json_write(root/'job.json', job)
        app._json_write(root/'output/external_diagnosis_input.json', bundle)
        return job, settings

    def test_create_returns_acceptance_snapshot_even_when_worker_completes_immediately(self):
        class Backend:
            def generate_structured(self, *, agent_id, output_model, payload):
                return result(payload['sources'][0]['source_id'])
        with mock.patch.object(app, '_backend', return_value=Backend()), mock.patch.object(app.threading, 'Thread', ImmediateThread):
            response = self.store.create_job({'task': '分析', 'connection': connection(), 'files': [file_input()]})
        self.assertEqual(response['status'], 'RUNNING')
        self.assertEqual(self.store.get(response['job_id'])['status'], 'SUCCEEDED')

    def test_resume_returns_acceptance_snapshot_even_when_worker_completes_immediately(self):
        job, settings = self.interrupted()
        class Backend:
            def generate_structured(self, *, agent_id, output_model, payload):
                return result(payload['sources'][0]['source_id'])
        with mock.patch.object(app, '_backend', return_value=Backend()), mock.patch.object(app.threading, 'Thread', ImmediateThread):
            response = self.store.resume_job(job['job_id'], settings)
        self.assertEqual(response['status'], 'RUNNING')
        self.assertEqual(self.store.get(job['job_id'])['status'], 'SUCCEEDED')

    def test_backup_refuses_snapshot_while_analysis_is_active(self):
        self.store._operation.acquire()
        try:
            with self.assertRaises(app.WorkbenchError) as error: self.store.create_backup()
            self.assertEqual(error.exception.code, 'MODEL_BUSY')
        finally: self.store._operation.release()

    def test_restore_refuses_mutation_while_analysis_is_active(self):
        self.store._operation.acquire()
        try:
            with self.assertRaises(app.WorkbenchError) as error: self.store.restore_backup(backup({}))
            self.assertEqual(error.exception.code, 'MODEL_BUSY')
        finally: self.store._operation.release()

    def test_restore_turns_foreign_running_job_into_resumable_interrupted_job(self):
        job_id = 'b'*32
        raw = json.dumps({'job_id': job_id, 'status': 'RUNNING'}).encode()
        self.store.restore_backup(backup({f'jobs/{job_id}/job.json': raw}))
        self.assertEqual(self.store.get(job_id)['status'], 'INTERRUPTED')
        self.assertEqual(self.store.get(job_id)['reason_code'], 'RESTORED_INCOMPLETE_TASK')

    def test_restore_rejects_windows_aliases_before_writing_any_job(self):
        job_id='c'*32
        raw=json.dumps({'job_id':job_id,'status':'SUCCEEDED'}).encode()
        for extra in ['JOB.JSON', 'input/CON.txt', './job.json', 'input/a.txt.', 'input/a.txt ', 'input//x.txt']:
            with self.subTest(extra=extra):
                target = app.Workbench(self.root / ('restore-' + hashlib.sha256(extra.encode()).hexdigest()[:8]))
                payload = backup({f'jobs/{job_id}/job.json':raw, f'jobs/{job_id}/{extra}':b'{}'})
                with self.assertRaises(app.WorkbenchError): target.restore_backup(payload)
                self.assertFalse(target._job_root(job_id).exists())

    def test_resume_checks_actual_source_hash_before_any_model_invocation(self):
        bundle=source_bundle(text='原始温度异常。')
        job,settings=self.interrupted(bundle)
        bundle['sources'][0]['text']='被修改的温度正常。'
        app._json_write(self.store._job_root(job['job_id'])/'output/external_diagnosis_input.json',bundle)
        with mock.patch.object(app, '_backend') as backend:
            with self.assertRaises(app.WorkbenchError) as error: self.store.resume_job(job['job_id'], settings)
            self.assertEqual(error.exception.code, 'RESUME_INPUT_CHANGED')
            backend.assert_not_called()
        self.assertEqual(self.store.get(job['job_id'])['status'], 'INTERRUPTED')

    def test_long_report_comparison_citations_resolve_in_published_sources(self):
        bundle=source_bundle()
        class Backend:
            def generate_structured(self, *, agent_id, output_model, payload):
                sid=payload['sources'][0]['source_id']; output=result(sid)
                if not '-chunk-' in payload['task_id']:
                    for field in ['common_findings','report_disagreements','complementary_evidence','event_alignment']:
                        output[field]=[{'text':'需要对照原文。','citation_ids':[sid]}]
                return output
        run=run_hierarchical_analysis(task_id=bundle['task_id'],task='比较',input_bundle=bundle,llm_backend=Backend(),output_root=self.root/'long')
        self.assertEqual(run['generation']['status'],'SUCCEEDED')
        exp=run['model_explanation']; allowed={s['source_id'] for s in exp['sources']}
        for field in ['common_findings','report_disagreements','complementary_evidence','event_alignment']:
            self.assertLessEqual(set(exp[field][0]['citation_ids']),allowed)

    def test_long_report_never_silently_slices_chunk_summaries(self):
        bundle=source_bundle(); seen=[]
        class Backend:
            def generate_structured(self, *, agent_id, output_model, payload):
                sid=payload['sources'][0]['source_id']
                if '-chunk-' in payload['task_id']:
                    return result(sid,'解释内容。'*900+'TAIL_EVIDENCE_913')
                seen.extend(s['text'] for s in payload['sources']);return result(sid)
        run=run_hierarchical_analysis(task_id=bundle['task_id'],task='分析',input_bundle=bundle,llm_backend=Backend(),output_root=self.root/'long')
        self.assertEqual(run['generation']['status'],'SUCCEEDED')
        self.assertTrue(seen)
        self.assertTrue(all('TAIL_EVIDENCE_913' in text for text in seen))
        for text in seen: self.assertIsInstance(json.loads(text),dict)

    def test_synthesis_over_budget_stops_explicitly_without_sending_truncated_context(self):
        bundle=source_bundle(text='温度异常；'*24000); calls=[]
        class Backend:
            def generate_structured(self, *, agent_id, output_model, payload):
                calls.append(payload['task_id'])
                return result(payload['sources'][0]['source_id'], '分块解释。'*1100)
        # Pin a deliberately smaller budget; production now permits 120k summaries.
        with mock.patch('astrograph.integrations.long_report_analysis.MAX_SYNTHESIS_CHARACTERS', 60_000):
            run=run_hierarchical_analysis(task_id=bundle['task_id'],task='分析',input_bundle=bundle,llm_backend=Backend(),output_root=self.root/'long')
        self.assertEqual(run['generation']['reason_code'], 'SYNTHESIS_CONTEXT_TOO_LARGE')
        self.assertNotIn(bundle['task_id'], calls)
        self.assertTrue((self.root/'long/checkpoint.json').is_file())

if __name__ == '__main__': unittest.main(verbosity=2)
