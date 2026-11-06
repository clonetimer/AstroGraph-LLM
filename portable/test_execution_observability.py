"""Observable execution: actual requests, safe receipts, immutable artifacts."""
import importlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from test_workbench import app, builder, file_input, connection, model_server, wait_job
from test_closeout import source_bundle, result


class ExecutionObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def module(self):
        self.assertTrue(hasattr(app, 'execution_snapshot'), 'shared execution recorder is not wired')
        return importlib.import_module('astrograph.integrations.execution_observability')

    def recorder(self, kind='analysis'):
        m = self.module()
        return m.RunJournal(self.root, task_id='a'*32, kind=kind,
                            bundle=source_bundle(text='PRIVATE_REPORT'), model='test-model', mode='local')

    def test_sources_and_events_are_metadata_only_and_unknown_usage_is_not_zero(self):
        m = self.module(); run = self.recorder()
        run.event('INPUT_VALIDATED', completed=1, total=1)
        run.finish('FAILED', reason_code='MODEL_OUTPUT_INVALID')
        data=m.execution_snapshot(self.root, 'a'*32)
        self.assertEqual(data['runs'][0]['status'], 'FAILED')
        self.assertEqual(data['runs'][0]['materials'][0]['filename'], '诊断.txt')
        self.assertIsNone(data['usage']['input_tokens'])
        self.assertIsNone(data['usage']['estimated_cost'])
        self.assertNotIn('PRIVATE_REPORT', json.dumps(data))
        with self.assertRaises(ValueError): run.event('secret-api-key')

    def test_http_repair_attempts_are_counted_and_no_secrets_persist(self):
        m=self.module(); run=self.recorder()
        backend=app._backend({**connection(), 'api_key':'SECRET_TOKEN', 'response_format_strategy':'plain_json'})
        attempts=[]
        def request(url, **kwargs):
            attempts.append(kwargs['payload'])
            kwargs['response_observer'](b'PRIVATE_SERVER_RESPONSE', http_status=200)
            content='not-json' if len(attempts)==1 else json.dumps(result('source:test'))
            return {'choices':[{'message':{'content':content},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':11,'completion_tokens':7}}
        from astrograph.integrations.external_diagnosis_analysis import ExternalDiagnosisAnalysis
        with m.run_scope(run), mock.patch('astrograph.integrations.bounded_http.request_json', side_effect=request):
            backend.generate_structured(agent_id='external-diagnosis-analysis', output_model=ExternalDiagnosisAnalysis,
                payload={'task':'PRIVATE_QUESTION', 'sources':[], 'system_policy':'policy'})
            run.finish('SUCCEEDED')
        snap=m.execution_snapshot(self.root,'a'*32); records=snap['runs'][0]['requests']
        self.assertEqual(len(attempts),2);self.assertEqual(len(records),2)
        self.assertEqual(snap['usage']['input_tokens'],22)
        self.assertEqual(snap['usage']['output_tokens'],14)
        self.assertTrue(all(x['http_status']==200 for x in records))
        disk='\n'.join(p.read_text(encoding='utf-8') for p in self.root.rglob('*.json'))
        for secret in ['SECRET_TOKEN','PRIVATE_REPORT','PRIVATE_QUESTION','PRIVATE_SERVER_RESPONSE','Authorization']:
            self.assertNotIn(secret,disk)

    def test_partial_and_bad_usage_never_become_complete_totals(self):
        m=self.module(); run=self.recorder()
        with m.run_scope(run):
            run.http_call(lambda observer: {'usage':{'prompt_tokens':10,'completion_tokens':4}}, request_sha256='a'*64)
            run.http_call(lambda observer: {'usage':{'prompt_tokens':True,'completion_tokens':-3}}, request_sha256='b'*64)
            run.finish('FAILED')
        usage=m.execution_snapshot(self.root,'a'*32)['usage']
        self.assertEqual(usage['http_requests'],2)
        self.assertEqual(usage['known_input_tokens'],10)
        self.assertIsNone(usage['input_tokens'])
        self.assertEqual(usage['input_reported_requests'],1)

    def test_transport_error_is_counted_without_raw_exception(self):
        m=self.module(); run=self.recorder()
        def fail(observer):
            observer(b'SECRET response',http_status=401)
            raise RuntimeError('SECRET response with key')
        with m.run_scope(run):
            with self.assertRaises(RuntimeError):run.http_call(fail,request_sha256='a'*64)
            run.finish('FAILED')
        snap=m.execution_snapshot(self.root,'a'*32)
        self.assertEqual(snap['usage']['failed_requests'],1)
        self.assertEqual(snap['runs'][0]['requests'][0]['http_status'],401)
        self.assertNotIn('SECRET',json.dumps(snap))

    def test_artifact_versions_are_immutable_and_hash_checked(self):
        m=self.module(); run=self.recorder()
        run.artifact('report.md',b'original');run.finish('SUCCEEDED')
        self.assertEqual(m.read_artifact(self.root,'a'*32,run.run_id,'report.md'), b'original')
        with self.assertRaises((ValueError,FileExistsError)):run.artifact('report.md',b'changed')
        with self.assertRaises((ValueError,KeyError)):m.read_artifact(self.root,'a'*32,run.run_id,'../report.md')
        p=self.root/'execution'/'runs'/run.run_id/'artifacts'/'report.md';p.write_bytes(b'tampered')
        with self.assertRaises(ValueError):m.read_artifact(self.root,'a'*32,run.run_id,'report.md')

    def test_restart_marks_only_unfinished_run_interrupted_and_no_model_calls(self):
        m=self.module(); done=self.recorder();done.finish('SUCCEEDED');unfinished=self.recorder()
        m.recover_interrupted(self.root,'a'*32)
        snaps=m.execution_snapshot(self.root,'a'*32)['runs']
        by_id={x['run_id']:x for x in snaps}
        self.assertEqual(by_id[done.run_id]['status'],'SUCCEEDED')
        self.assertEqual(by_id[unfinished.run_id]['status'],'INTERRUPTED')
        self.assertEqual(by_id[unfinished.run_id]['requests'],[])

    def test_legacy_job_is_readable_without_invented_events(self):
        m=self.module();snapshot=m.execution_snapshot(self.root,'a'*32)
        self.assertTrue(snapshot['legacy']);self.assertEqual(snapshot['runs'],[])

    def test_context_is_thread_local_and_reset_after_exception(self):
        m=self.module();run=self.recorder();observed=[]
        with self.assertRaises(RuntimeError):
            with m.run_scope(run):
                thread=threading.Thread(target=lambda:observed.append(m.current_run()))
                thread.start();thread.join()
                raise RuntimeError('do not log content')
        self.assertEqual(observed,[None]);self.assertIsNone(m.current_run())
        self.assertNotIn('do not log content',json.dumps(m.execution_snapshot(self.root,'a'*32)))

    def test_workbench_real_http_records_run_materials_artifacts_and_followup(self):
        self.module()
        with model_server() as model:
            url = model["base_url"]
            store=app.Workbench(self.root)
            job=store.create_job({'task':'核查','connection':connection(url),'files':[file_input()]})
            data=wait_job(store,job['job_id']);execution=data['execution']
            self.assertEqual(execution['runs'][0]['status'],'SUCCEEDED',
                f"status={data['job'].get('status')} reason={data['job'].get('reason_code')} "
                f"requests={[(q.get('status'),q.get('http_status'),q.get('latency_ms')) for q in execution['runs'][0]['requests'][-12:]]} "
                f"model_calls={model['calls']}")
            self.assertEqual(execution['usage']['http_requests'],model['calls'])
            self.assertIsNone(execution['usage']['input_tokens'])  # fixture supplies no usage
            self.assertEqual({a['name'] for a in execution['runs'][0]['artifacts']},{'report.json','report.md'})
            run_id=execution['runs'][0]['run_id']
            original=store.export_run(job['job_id'],run_id,'report.json')[0]
            state=store.conversation(job['job_id'])
            from uuid import uuid4
            store.follow_up(job['job_id'],{'question':'依据？','connection':connection(url),
                'input_sha256':state['input_sha256'],'request_id':str(uuid4())})
            snapshot=store.execution(job['job_id'])
            self.assertEqual(len(snapshot['runs']),2)
            self.assertEqual(snapshot['usage']['http_requests'],model['calls'])
            self.assertEqual(store.export_run(job['job_id'],run_id,'report.json')[0],original)

    def test_failed_generation_is_not_successful_execution_and_retry_preserves_prior(self):
        self.module()
        class Bad:
            def generate_structured(self,**kwargs):return {'summary':'invalid output'}
        class Good:
            def generate_structured(self,*,payload,**kwargs):return result(payload['sources'][0]['source_id'])
        store=app.Workbench(self.root)
        with mock.patch.object(app,'_backend',return_value=Bad()):
            job=store.create_job({'task':'核查','connection':connection(),'files':[file_input()]})
            data=wait_job(store,job['job_id'])
        prior=data['execution']['runs'][0];self.assertEqual(prior['status'],'FAILED')
        raw=store.export_run(job['job_id'],prior['run_id'],'report.json')[0]
        with mock.patch.object(app,'_backend',return_value=Good()):
            store.resume_job(job['job_id'],connection());updated=wait_job(store,job['job_id'])
        self.assertEqual(len(updated['execution']['runs']),2)
        self.assertEqual(store.export_run(job['job_id'],prior['run_id'],'report.json')[0],raw)
        self.assertEqual(updated['execution']['runs'][-1]['status'],'SUCCEEDED')

    def test_restored_incomplete_run_is_immediately_interrupted(self):
        from test_convergence import backup_bytes
        m=self.module();run=self.recorder()
        root=self.root/'export'; root.mkdir()
        # A valid archive can originate from a crash and contain RUNNING receipts.
        raw=(run.path/'run.json').read_bytes()
        store=app.Workbench(self.root/'restore')
        store.restore_backup(backup_bytes('a'*32,{
            'jobs/'+('a'*32)+'/execution/runs/'+run.run_id+'/run.json':raw}))
        snapshot=store.execution('a'*32)
        self.assertEqual(snapshot['runs'][0]['status'],'INTERRUPTED')
        self.assertEqual(snapshot['usage']['http_requests'],0)

    def test_restored_unknown_nested_fields_are_not_exposed(self):
        m=self.module();run=self.recorder();run.finish('SUCCEEDED')
        path=run.path/'run.json';value=json.loads(path.read_text(encoding='utf-8'))
        value['api_key']='PRIVATE_KEY'
        value['events'][0]['counts']={'total':1,'api_key':'PRIVATE_KEY'}
        value['requests']=[{'sequence':1,'status':'RESPONSE_RECEIVED','stage':'MODEL_ANALYSIS',
            'usage':{'input_tokens':5,'output_tokens':2,'prompt':'PRIVATE_TEXT'},
            'response':'PRIVATE_TEXT'}]
        path.write_text(json.dumps(value))
        exposed=json.dumps(m.execution_snapshot(self.root,'a'*32))
        self.assertNotIn('PRIVATE_KEY',exposed)
        self.assertNotIn('PRIVATE_TEXT',exposed)

    def test_unreadable_run_does_not_silently_become_zero_cost(self):
        m=self.module();run=self.recorder();run.finish('SUCCEEDED')
        path=run.path/'run.json';value=json.loads(path.read_text(encoding='utf-8'));value['requests']='not-a-list'
        path.write_text(json.dumps(value))
        data=m.execution_snapshot(self.root,'a'*32)
        self.assertEqual(data['unavailable_runs'],1)
        self.assertFalse(data['legacy'])
        self.assertIsNone(data['usage']['input_tokens'])

    def test_same_canonical_assets_are_staged_without_copying_full_platform(self):
        target=self.root/'staged';builder.stage_app(builder.ROOT if hasattr(builder,'ROOT') else Path(__file__).resolve().parents[1],target)
        source=Path(__file__).resolve().parents[1]/'src/astrograph/resources'
        for name in ['execution_view.js','execution_view.css']:
            self.assertEqual((target/'assets'/name).read_bytes(),(source/name).read_bytes())
        self.assertFalse((target/'astrograph/diagnosis').exists())

    def test_receipt_write_failure_after_http_is_terminal_not_a_paid_retry(self):
        m=self.module();run=self.recorder();backend=app._backend({**connection(), 'response_format_strategy':'plain_json'})
        from astrograph.integrations.external_diagnosis_analysis import ExternalDiagnosisAnalysis
        original=m._write;failed=[];calls=[]
        def flaky_write(path,value):
            if value.get('requests') and value['requests'][-1]['status']=='RESPONSE_RECEIVED' and not failed:
                failed.append(True)
                raise OSError('disk issue contains PRIVATE_PATH')
            return original(path,value)
        def request(*args,**kwargs):
            calls.append(1)
            kwargs['response_observer'](b'not persisted',http_status=200)
            return {'choices':[{'message':{'content':json.dumps(result('source:test'))},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':5,'completion_tokens':4}}
        with mock.patch.object(m,'_write',side_effect=flaky_write), mock.patch(
                'astrograph.integrations.bounded_http.request_json',side_effect=request):
            with self.assertRaisesRegex(RuntimeError,'EXECUTION_RECORD_WRITE_FAILED'),m.run_scope(run):
                backend.generate_structured(agent_id='external-diagnosis-analysis',output_model=ExternalDiagnosisAnalysis,
                    payload={'task':'test','sources':[],'system_policy':'policy'})
        self.assertEqual(len(calls),1)
        snapshot=m.execution_snapshot(self.root,'a'*32)
        self.assertEqual(snapshot['runs'][0]['status'],'FAILED')
        self.assertNotIn('PRIVATE_PATH',json.dumps(snapshot))
