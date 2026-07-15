import unittest
from shared.utils.security import is_safe_url

class TestSecurityUtils(unittest.TestCase):
    def test_safe_public_url(self):
        # Public domains should be safe
        self.assertTrue(is_safe_url("https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"))
        self.assertTrue(is_safe_url("http://example.com"))

    def test_invalid_protocols(self):
        # Only http/https are allowed
        self.assertFalse(is_safe_url("ftp://example.com/file.zip"))
        self.assertFalse(is_safe_url("file:///etc/passwd"))
        self.assertFalse(is_safe_url("gopher://example.com"))
        self.assertFalse(is_safe_url("javascript:alert(1)"))

    def test_private_ips_blocked_by_default(self):
        # Private IPs/loopback are not allowed by default (allow_private_ips=False)
        self.assertFalse(is_safe_url("http://127.0.0.1/callback", allow_private_ips=False))
        self.assertFalse(is_safe_url("http://localhost:5196/results", allow_private_ips=False))
        self.assertFalse(is_safe_url("https://192.168.1.100/data", allow_private_ips=False))
        self.assertFalse(is_safe_url("http://10.0.0.1/", allow_private_ips=False))

    def test_private_ips_allowed_optionally(self):
        # Private IPs/loopback should be allowed if allow_private_ips=True
        self.assertTrue(is_safe_url("http://127.0.0.1/callback", allow_private_ips=True))
        self.assertTrue(is_safe_url("http://localhost:5196/results", allow_private_ips=True))

    def test_allowed_hosts_restriction(self):
        # Hostname restrictions should match precisely
        allowed = ["localhost", "pms.uav-inspection.internal"]
        self.assertTrue(is_safe_url("http://localhost:5196/results", allowed_hosts=allowed, allow_private_ips=True))
        self.assertTrue(is_safe_url("https://pms.uav-inspection.internal/api", allowed_hosts=allowed, allow_private_ips=True))
        
        # Non-matching hosts must be blocked
        self.assertFalse(is_safe_url("http://external-attacker.com/steal", allowed_hosts=allowed, allow_private_ips=True))
        self.assertFalse(is_safe_url("https://google.com", allowed_hosts=allowed, allow_private_ips=True))

if __name__ == "__main__":
    unittest.main()
