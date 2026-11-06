"""Hash-pinned Windows 7 vendor downloads must fail closed without network in tests."""
from __future__ import annotations
import hashlib
from pathlib import Path
import tempfile
import unittest

import prepare_win7_vendor as vendor


class Win7VendorTests(unittest.TestCase):
    def test_rejects_nonofficial_host_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "runtime.zip"
            with self.assertRaisesRegex(ValueError, "UNTRUSTED_VENDOR_DOWNLOAD_URL"):
                vendor.fetch_pinned(
                    "https://example.invalid/evil.zip", target, "0" * 64,
                    maximum=1024, hosts={"www.python.org"},
                )
            self.assertFalse(target.exists())

    def test_cached_file_must_match_pinned_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "pinned.whl"
            contents = b"locally audited binary sample"
            target.write_bytes(contents)
            digest = hashlib.sha256(contents).hexdigest()
            receipt = vendor.fetch_pinned(
                "https://files.pythonhosted.org/pinned.whl", target, digest,
                maximum=4096, hosts={"files.pythonhosted.org"},
            )
            self.assertEqual(receipt["sha256"], digest)
            self.assertEqual(target.read_bytes(), contents)

    def test_rejects_invalid_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "INVALID_PINNED_DIGEST"):
                vendor.fetch_pinned(
                    "https://www.python.org/pinned.zip", Path(directory) / "runtime.zip",
                    "g" * 64, maximum=4096, hosts={"www.python.org"},
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
