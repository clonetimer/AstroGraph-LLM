"""Verify real static delivery and the offline V2.7 packaging boundary."""
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from test_workbench import app

class NavigationDeliveryTests(unittest.TestCase):
    def test_navigation_script_is_served_same_origin_without_generic_file_access(self):
        with tempfile.TemporaryDirectory() as folder:
            server=app.make_server(app.Workbench(Path(folder)),port=0)
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            try:
                url=f'http://127.0.0.1:{server.server_port}'
                with urllib.request.urlopen(url+'/navigation.js',timeout=10) as response:
                    self.assertEqual(response.headers.get_content_type(),'text/javascript')
                    self.assertEqual(response.headers['Cache-Control'],'no-store')
                    self.assertEqual(response.read(),(app.BASE/'assets/navigation.js').read_bytes())
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(url+'/../workbench.py',timeout=10)
                self.assertEqual(error.exception.code,404)
            finally:server.shutdown();server.server_close();worker.join(2)

    def test_v27_builder_rejects_wrong_baseline_without_writing_output(self):
        script=Path(__file__).with_name('refresh_v27.py')
        self.assertTrue(script.is_file(),'V2.7 offline builder not implemented')
        spec=importlib.util.spec_from_file_location('v27_builder',script)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fake=root/'wrong.zip';fake.write_bytes(b'wrong baseline')
            with self.assertRaisesRegex(ValueError,'baseline SHA-256'):
                mod.refresh_distribution(Path(__file__).resolve().parent.parent,fake,root/'out')
            self.assertFalse((root/'out').exists())

if __name__=='__main__':unittest.main(verbosity=2)
