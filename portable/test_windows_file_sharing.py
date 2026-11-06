"""Simulate Windows atomic replacement/read sharing conflicts without mocking LLMs."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from test_workbench import app
from astrograph.integrations import execution_observability as journal

class WindowsFileSharingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_job_read_recovers_two_transient_sharing_denials(self):
        path = self.root / 'job.json'
        app._json_write(path, {'status': 'RUNNING'})
        original = Path.read_text
        calls = []
        def flaky(item, *a, **kw):
            if item == path:
                calls.append(1)
                if len(calls) <= 2:
                    raise PermissionError(13, 'transient sharing denial', str(item))
            return original(item, *a, **kw)
        with mock.patch.object(Path, 'read_text', flaky):
            self.assertEqual(app._json_read(path), {'status': 'RUNNING'})
        self.assertEqual(len(calls), 3)

    def test_job_write_recovers_atomic_replace_denial(self):
        path = self.root / 'job.json'
        app._json_write(path, {'status': 'QUEUED'})
        original = Path.replace
        calls = []
        def flaky(item, dest):
            if dest == path:
                calls.append(1)
                if len(calls) == 1:
                    raise PermissionError(13, 'transient sharing denial', str(item))
            return original(item, dest)
        with mock.patch.object(Path, 'replace', flaky):
            app._json_write(path, {'status': 'RUNNING'})
        self.assertEqual(app._json_read(path)['status'], 'RUNNING')
        self.assertEqual(len(calls), 2)
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_receipt_reader_and_writer_retry_short_sharing_denials(self):
        job_id = 'a' * 32
        run = journal.RunJournal(self.root, task_id=job_id, kind='analysis',
                                 bundle={'task_id': job_id, 'sources': [], 'input_sha256': 'b' * 64})
        path = self.root / 'execution' / 'runs' / run.run_id / 'run.json'
        original_read = Path.read_text
        original_replace = Path.replace
        calls = {'read': 0, 'replace': 0}
        def read(item, *a, **kw):
            if item == path:
                calls['read'] += 1
                if calls['read'] == 1:
                    raise PermissionError(13, 'transient sharing denial', str(item))
            return original_read(item, *a, **kw)
        def replace(item, dest):
            if dest == path:
                calls['replace'] += 1
                if calls['replace'] == 1:
                    raise PermissionError(13, 'transient sharing denial', str(item))
            return original_replace(item, dest)
        with mock.patch.object(Path, 'read_text', read), mock.patch.object(Path, 'replace', replace):
            run.event('INPUT_VALIDATED')
            self.assertEqual(journal._load(self.root, job_id, run.run_id)['status'], 'RUNNING')
        self.assertEqual(calls, {'read': 2, 'replace': 2})

    def test_permanent_permission_denial_remains_an_error(self):
        path = self.root / 'job.json'
        app._json_write(path, {'ok': True})
        original = Path.read_text
        count = []
        def denied(item, *a, **kw):
            if item == path:
                count.append(1)
                raise PermissionError(13, 'permanently denied', str(item))
            return original(item, *a, **kw)
        with mock.patch.object(Path, 'read_text', denied):
            with self.assertRaises(PermissionError):
                app._json_read(path)
        self.assertTrue(1 <= len(count) <= 10)
