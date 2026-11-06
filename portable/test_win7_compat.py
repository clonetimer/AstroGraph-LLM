"""Run behavior contracts with the Python 3.8 compatibility dependency set."""
from pathlib import Path
import hashlib
import tempfile
import unittest
from test_workbench import app, builder
from astrograph.retrieval.structured_document import SectionFirstDocumentIngestor


class Win7CompatibilityTests(unittest.TestCase):
    def test_build_hash_streams_file_without_new_hashlib_api(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / '中文 file.txt'
            f.write_bytes(b'abc')
            self.assertEqual(builder.sha256(f), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_markdown_local_asset_preserves_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'local.png').write_bytes(b'image')
            items, warnings = SectionFirstDocumentIngestor().parse_markdown('![sample](local.png)', asset_root=root)
            self.assertEqual(items[0].asset_sha256, hashlib.sha256(b'image').hexdigest())

    def test_markdown_asset_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); allowed = root / 'allowed'; allowed.mkdir()
            (root / 'secret.txt').write_text('not an allowed asset')
            items, warnings = SectionFirstDocumentIngestor().parse_markdown('![sample](../secret.txt)', asset_root=allowed)
            self.assertIsNone(items[0].asset_sha256)
            self.assertTrue(any('OUTSIDE_ROOT_BLOCKED' in s for s in warnings))

if __name__ == '__main__': unittest.main()
