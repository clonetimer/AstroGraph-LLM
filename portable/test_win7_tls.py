"""Legacy runtime still verifies certificate chains and server hostnames."""
import ssl
import unittest
from test_workbench import app

class Win7TLSTests(unittest.TestCase):
    def test_trust_context_keeps_hostname_and_chain_checks(self):
        from astrograph.integrations.http_worker import verified_tls_context
        context = verified_tls_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertGreaterEqual(context.minimum_version, ssl.TLSVersion.TLSv1_2)
        self.assertGreater(len(context.get_ca_certs()), 50)

if __name__ == '__main__': unittest.main()
