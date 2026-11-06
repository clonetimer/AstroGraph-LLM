"""Import failures must be actionable without leaking arbitrary parser output."""
import tempfile
from pathlib import Path
import subprocess
import unittest
from unittest import mock
from test_workbench import app, binary_input, make_pdf, connection

class DocumentGuardTests(unittest.TestCase):
    def test_scan_has_specific_ocr_message_and_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(app,'_backend') as backend:
            store=app.Workbench(Path(d))
            with self.assertRaises(app.WorkbenchError) as ctx:
                store.create_job({'task':'核查','connection':connection(),
                    'files':[binary_input(make_pdf(''), '扫描.pdf')]})
            self.assertEqual(ctx.exception.code,'DOCUMENT_NEEDS_OCR')
            self.assertIn('OCR',ctx.exception.message)
            backend.assert_not_called()
            self.assertEqual(store.list_jobs(),[])

    def test_document_timeout_is_bounded_and_releases_operation(self):
        with tempfile.TemporaryDirectory() as d:
            store=app.Workbench(Path(d))
            with mock.patch.object(app,'run_managed_subprocess',side_effect=subprocess.TimeoutExpired('parse',30)):
                with self.assertRaises(app.WorkbenchError) as ctx:
                    store.create_job({'task':'核查','connection':connection(),
                        'files':[binary_input(make_pdf(), '异常.pdf')]})
            self.assertEqual(ctx.exception.code,'DOCUMENT_PARSE_TIMEOUT')
            self.assertEqual(store.list_jobs(),[])
            self.assertTrue(store._operation.acquire(blocking=False))
            store._operation.release()

if __name__=='__main__': unittest.main(verbosity=2)
