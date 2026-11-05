import json,tempfile,time,threading,unittest,sys
from pathlib import Path
from unittest.mock import patch
from test_workbench import app,file_input as item,connection,model_server

def done(store,jid,timeout=15):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        result=store.read_job(jid)
        if result['job']['status'] not in {'RUNNING','QUEUED','CANCEL_REQUESTED'}:return result
        time.sleep(.03)
    raise AssertionError('task never finished')
class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.store=app.Workbench(Path(self.tmp.name)/'data')
    def selection(self):
        w=self.store.contexts.create_workspace(name='ADCS',device='A',event='E')
        e=self.store.contexts.create_entry(w['workspace_id'],kind='user_note',title='单位复核',body='单位可能为rpm，须核验。',source_ref='人工说明',actor='operator')
        self.store.contexts.review_entry(w['workspace_id'],e['entry_id'],expected_revision=1,reviewed=True,actor='reviewer')
        return {'workspace_id':w['workspace_id'],'entries':[{'entry_id':e['entry_id'],'revision':2}]}
    def test_context_enters_analysis_followup_and_backup_without_editing_original(self):
        selection=self.selection()
        with model_server() as model:
            c=connection(model['base_url']);job=self.store.create_job({'task':'说明原因','files':[item()],'connection':c,'context':selection})
            result=done(self.store,job['job_id']);self.assertEqual(result['report']['analysis']['knowledge_enrichment']['generation']['status'],'SUCCEEDED')
            snap=result['report']['reviewed_context'];self.assertEqual(snap['entries'][0]['revision'],2)
            request=json.loads(model['bodies'][0]['messages'][-1]['content'])['payload'];self.assertEqual(request['reviewed_context'],snap)
            raw=(self.store._job_root(job['job_id'])/'input_original/诊断.txt').read_bytes()
            e=selection['entries'][0]
            self.store.contexts.revise_entry(selection['workspace_id'],e['entry_id'],expected_revision=2,title='修改',body='新说明',source_ref='',actor='operator')
            self.assertTrue(self.store._queue.wait_idle(5))
            self.store.follow_up(job['job_id'],{'question':'为什么','input_sha256':result['job']['input_sha256'],'request_id':'repeat-1','connection':c})
            payload=json.loads(model['bodies'][-1]['messages'][-1]['content'])['payload'];self.assertEqual(payload['reviewed_context'],snap)
            self.assertEqual((self.store._job_root(job['job_id'])/'input_original/诊断.txt').read_bytes(),raw)
            b=self.store.create_backup();other=app.Workbench(Path(self.tmp.name)/'other');other.restore_backup(b)
            self.assertEqual(other.contexts.export_bundle(),self.store.contexts.export_bundle())
            self.assertEqual(other.read_job(job['job_id'])['report']['reviewed_context'],snap)
    def test_changed_endpoint_never_inherits_key_and_unknown_phase_rejected(self):
        c=connection();ph=app.parse_phase_connections(c,{'chunk':{'base_url':'http://127.0.0.1:2345/v1','model':'small'}})
        self.assertEqual(ph['chunk']['api_key'],'')
        with self.assertRaises(app.WorkbenchError):app.parse_phase_connections(c,{'unknown':{'model':'x'}})
    def test_explicit_queue_stage_models_followup_and_secrets_not_persisted(self):
        with model_server() as model:
            c=connection(model['base_url']);ph={'chunk':{'model':'small'},'synthesis':{'model':'large'},'follow_up':{'model':'qa'}}
            a=self.store.create_job({'task':'one','files':[item()],'connection':c,'phases':ph,'queue':True})
            b=self.store.create_job({'task':'two','files':[item()],'connection':c,'phases':ph,'queue':True})
            r=done(self.store,a['job_id']);done(self.store,b['job_id'])
            self.assertEqual(model['calls'],2, f"job_a={r['job']}, job_b={self.store.get(b['job_id'])}")
            self.assertTrue(self.store._queue.wait_idle(5))
            self.store.follow_up(a['job_id'],{'question':'explain','input_sha256':r['job']['input_sha256'],'request_id':'phase-qa','connection':c,'phases':ph})
            self.assertEqual(model['bodies'][-1]['model'],'qa')
            execution=self.store.read_job(a['job_id'])['execution'];self.assertTrue(any(x.get('model')=='qa' for run in execution['runs'] for x in run['requests']))
            for p in self.store.root.rglob('*.json'):
                text=p.read_text();self.assertNotIn(c['api_key'],text);self.assertNotIn(model['base_url'],text)
    def test_cancel_running_and_waiting_never_publish_report(self):
        started=threading.Event();release=threading.Event();calls=[]
        class Backend:
            model_id='mock'
            def generate_structured(self,**kw):started.set();release.wait(5);calls.append(1);return {}
        try:
            with patch.object(app,'_backend',return_value=Backend()):
                a=self.store.create_job({'task':'a','files':[item()],'connection':connection()});self.assertTrue(started.wait(2))
                b=self.store.create_job({'task':'b','files':[item()],'connection':connection(),'queue':True})
                self.assertEqual(self.store.get(b['job_id'])['status'],'QUEUED')
                self.assertEqual(self.store.cancel_job(b['job_id'])['status'],'CANCELLED')
                self.store.cancel_job(a['job_id']);release.set();self.store._queue.wait_idle(8)
            self.assertEqual(self.store.get(a['job_id'])['status'],'CANCELLED');self.assertEqual(calls,[1])
            for j in (a,b):
                self.assertFalse((self.store._job_root(j['job_id'])/'output/report.json').exists())
                self.assertTrue((self.store._job_root(j['job_id'])/'input_original/诊断.txt').exists())
        finally:release.set()
    def test_read_context_while_queue_active_but_do_not_mutate_it(self):
        selection=self.selection();started=threading.Event();release=threading.Event()
        class Backend:
            model_id='test'
            def generate_structured(self,**kw):
                started.set();release.wait(5);return {}
        try:
            with patch.object(app,'_backend',return_value=Backend()):
                job=self.store.create_job({'task':'a','files':[item()],'connection':connection()})
                self.assertTrue(started.wait(2))
                data=self.store.context_api('GET','workspaces/'+selection['workspace_id']+'/entries')
                self.assertEqual(data['entries'][0]['revision'],2)
                with self.assertRaises(app.WorkbenchError):
                    self.store.context_api('POST','workspaces',{'name':'b'})
                self.store.cancel_job(job['job_id']);release.set()
                self.assertTrue(self.store._queue.wait_idle(8))
        finally:
            release.set();self.store._queue.wait_idle(8)

    def test_long_report_routes_actual_chunk_and_synthesis_requests(self):
        with model_server() as model:
            c=connection(model['base_url']);ph={'chunk':{'model':'small'},'synthesis':{'model':'large'}}
            text='\n'.join(('不同观测记录 %03d '%i)+('轮速变化需要结合电流解释。'*70) for i in range(70))
            j=self.store.create_job({'task':'compare','files':[item(text=text)],'connection':c,'phases':ph})
            result=done(self.store,j['job_id'],timeout=30)
            self.assertTrue(self.store._queue.wait_idle(5))
            self.assertEqual(result['report']['analysis']['knowledge_enrichment']['generation']['status'],'SUCCEEDED')
            names=[x['model'] for x in model['bodies']]
            self.assertGreater(names.count('small'),1);self.assertEqual(names[-1],'large')
            self.assertEqual(len(names),names.count('small')+1)

    def test_resume_phase_change_rejected_and_queued_restart_manual(self):
        with model_server() as model:
            c=connection(model['base_url']);ph={'chunk':{'model':'small'}}
            j=self.store.create_job({'task':'a','files':[item()],'connection':c,'phases':ph});done(self.store,j['job_id'])
            root=self.store._job_root(j['job_id']);meta=self.store.get(j['job_id']);meta['status']='QUEUED';app._json_write(root/'job.json',meta)
            restored=app.Workbench(self.store.root);self.assertEqual(restored.get(j['job_id'])['status'],'INTERRUPTED')
            with self.assertRaises(app.WorkbenchError):restored.resume_job(j['job_id'],c,phases={'chunk':{'model':'changed'}})
            self.assertEqual(model['calls'],1)
if __name__=='__main__':unittest.main()
