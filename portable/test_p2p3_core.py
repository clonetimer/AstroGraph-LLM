import importlib.util,json,tempfile,threading,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'src/astrograph/integrations'
def load(name):
    p=ROOT/(name+'.py')
    if not p.exists():raise AssertionError('Missing shared module: '+name)
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class ContextTests(unittest.TestCase):
    def setUp(self):
        self.m=load('reviewed_context');self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.s=self.m.ContextStore(Path(self.tmp.name));self.w=self.s.create_workspace(name='卫星甲',device='ADCS',event='事件甲')
    def note(self):
        x=self.s.create_entry(self.w['workspace_id'],kind='user_note',title='单位',body='轮速单位应人工核对。',source_ref='记录A',actor='operator')
        return self.s.review_entry(self.w['workspace_id'],x['entry_id'],expected_revision=1,reviewed=True,actor='reviewer')
    def test_review_freeze_revision_and_scope(self):
        x=self.note();sel=[{'entry_id':x['entry_id'],'revision':2}];snap=self.s.snapshot(self.w['workspace_id'],sel)
        self.assertEqual(self.m.validate_snapshot(snap),snap)
        self.s.revise_entry(self.w['workspace_id'],x['entry_id'],expected_revision=2,title='单位',body='改变',source_ref='B',actor='op')
        self.assertEqual(snap['entries'][0]['body'],'轮速单位应人工核对。')
        with self.assertRaises(self.m.ContextError):self.s.snapshot(self.w['workspace_id'],sel)
        other=self.s.create_workspace(name='乙',device='B',event='B')
        with self.assertRaises(self.m.ContextError):self.s.snapshot(other['workspace_id'],sel)
        self.assertEqual(len(self.s.history(self.w['workspace_id'],x['entry_id'])),3)
    def test_drafts_duplicates_limits_and_empty_selection(self):
        x=self.s.create_entry(self.w['workspace_id'],kind='model_hypothesis',title='猜测',body='未证实',source_ref='',actor='op')
        with self.assertRaises(self.m.ContextError):self.s.snapshot(self.w['workspace_id'],[{'entry_id':x['entry_id'],'revision':1}])
        x=self.note();a={'entry_id':x['entry_id'],'revision':2}
        with self.assertRaises(self.m.ContextError):self.s.snapshot(self.w['workspace_id'],[a,a])
        self.assertEqual(self.s.snapshot(self.w['workspace_id'],[])['entries'],[])
        with self.assertRaises(self.m.ContextError):self.s.list_entries('../escape')
        with self.assertRaises(self.m.ContextError):self.s.create_entry(self.w['workspace_id'],kind='truth',title='x',body='x',source_ref='',actor='op')
    def test_snapshot_tampering_and_backup_conflicts(self):
        x=self.note();snap=self.s.snapshot(self.w['workspace_id'],[{'entry_id':x['entry_id'],'revision':2}]);snap['entries'][0]['body']='tampered'
        with self.assertRaises(self.m.ContextError):self.m.validate_snapshot(snap)
        bundle=self.s.export_bundle();t=self.m.ContextStore(Path(self.tmp.name)/'restore');t.import_bundle(bundle);t.import_bundle(bundle)
        self.assertEqual(t.export_bundle(),bundle)
        self.s.create_entry(self.w['workspace_id'],kind='user_note',title='新增',body='变化',source_ref='',actor='op')
        with self.assertRaises(self.m.ContextError):t.import_bundle(self.s.export_bundle())
        self.assertEqual(t.export_bundle(),bundle)
    def test_stale_review_and_frozen_task_write(self):
        x=self.note();snap=self.s.snapshot(self.w['workspace_id'],[{'entry_id':x['entry_id'],'revision':2}])
        with self.assertRaises(self.m.ContextError):self.s.review_entry(self.w['workspace_id'],x['entry_id'],expected_revision=1,reviewed=True,actor='r')
        root=Path(self.tmp.name)/'job';self.m.write_task_context(root,snap)
        self.assertEqual(self.m.load_task_context(root),snap)
        changed=self.s.snapshot(self.w['workspace_id'],[])
        with self.assertRaises(self.m.ContextError):self.m.write_task_context(root,changed)
class ExecutionTests(unittest.TestCase):
    def setUp(self):self.m=load('stage_execution')
    def test_routes_and_cancel_after_response(self):
        calls=[];token=threading.Event()
        class B:
            def __init__(self,name):self.model_id=name
            def generate_structured(self,**kw):calls.append(self.model_id);return self.model_id
        r=self.m.StageRouter({'analysis':B('main'),'chunk':B('small'),'synthesis':B('large')})
        self.assertEqual(self.m.for_stage(r,'chunk').generate_structured(),'small');self.assertEqual(self.m.for_stage(r,'follow_up').generate_structured(),'main')
        with self.assertRaises(ValueError):self.m.for_stage(r,'unknown')
        with self.m.cancellation_scope(token):
            token.set()
            with self.assertRaises(self.m.AnalysisCancelled):r.generate_structured()
        self.assertEqual(calls,['small','main'])
    def test_queue_bound_default_busy_fifo_cancel_and_no_callback_leak(self):
        q=self.m.BoundedExecutor(max_waiting=1);release=threading.Event();started=threading.Event();calls=[];states=[]
        def run(t):started.set();release.wait(4);calls.append('one')
        persist=lambda k,s:states.append((k,s))
        try:
            q.submit('one',run,persist=persist);self.assertTrue(started.wait(2))
            with self.assertRaises(ValueError):q.submit('other',lambda t:None,persist=persist)
            q.submit('two',lambda t:calls.append('two'),persist=persist,allow_queue=True)
            with self.assertRaises(ValueError):q.submit('three',lambda t:None,persist=persist,allow_queue=True)
            self.assertEqual(q.cancel('two'),'CANCELLED')
        finally:release.set();q.wait_idle(6)
        self.assertEqual(calls,['one']);self.assertIn(('two','CANCELLED'),states)
        def fail(*a):raise OSError('disk')
        with self.assertRaises(OSError):q.submit('bad',lambda t:calls.append('bad'),persist=fail)
        self.assertFalse(q.busy());self.assertEqual(calls,['one'])
    def test_publication_wins_cancellation_atomically(self):
        q=self.m.BoundedExecutor();published=threading.Event();release=threading.Event()
        def run(t):
            with q.publication('one'):pass
            published.set();release.wait(4)
        try:
            q.submit('one',run,persist=lambda *a:None);self.assertTrue(published.wait(2))
            with self.assertRaises(ValueError):q.cancel('one')
        finally:release.set();q.wait_idle(6)
if __name__=='__main__':unittest.main()

class RestorationAtomicityTests(unittest.TestCase):
    def test_context_import_rolls_back_when_job_publication_fails(self):
        m=load('reviewed_context')
        with tempfile.TemporaryDirectory() as root:
            source=m.ContextStore(Path(root)/'one');source.create_workspace(name='A')
            target=m.ContextStore(Path(root)/'two')
            def fail():raise OSError('publication failed')
            with self.assertRaises(OSError):target.import_bundle(source.export_bundle(),before_commit=fail)
            self.assertEqual(target.list_workspaces(),[])
    def test_worker_persist_failure_does_not_execute_queued_callback(self):
        m=load('stage_execution');q=m.BoundedExecutor();calls=[];states=[]
        def persist(key,state):
            states.append(state)
            if states.count('RUNNING')==2:raise OSError('disk unavailable')
        q.submit('task',lambda token:calls.append(1),persist=persist)
        self.assertTrue(q.wait_idle(3));self.assertEqual(calls,[])
        self.assertFalse(q.items);self.assertIn('FAILED',states)
