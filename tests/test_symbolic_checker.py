"""Unit tests for symbolic checker"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments' / 'mitigations'))

from symbolic_checker import (
    extract_cve_ids,
    check_cve_ids,
    sanitize_response,
    load_fallback_cves
)


class TestCVEExtraction(unittest.TestCase):
    """Test CVE ID extraction"""

    def test_extract_single_cve(self):
        """Test extracting a single CVE ID"""
        text = "The vulnerability CVE-2021-44228 affects Log4j"
        cves = extract_cve_ids(text)

        self.assertEqual(len(cves), 1)
        self.assertEqual(cves[0], 'CVE-2021-44228')

    def test_extract_multiple_cves(self):
        """Test extracting multiple CVE IDs"""
        text = "CVE-2021-44228 and CVE-2021-45046 are related vulnerabilities"
        cves = extract_cve_ids(text)

        self.assertEqual(len(cves), 2)
        self.assertIn('CVE-2021-44228', cves)
        self.assertIn('CVE-2021-45046', cves)

    def test_extract_no_cves(self):
        """Test text with no CVE IDs"""
        text = "This text has no CVE identifiers"
        cves = extract_cve_ids(text)

        self.assertEqual(len(cves), 0)

    def test_extract_case_insensitive(self):
        """Test case-insensitive extraction"""
        text = "cve-2021-44228 is the same as CVE-2021-44228"
        cves = extract_cve_ids(text)

        # Should normalize to uppercase
        self.assertEqual(len(cves), 2)
        self.assertTrue(all(cve.isupper() for cve in cves))

    def test_extract_with_context(self):
        """Test extraction from realistic text"""
        text = """
        The Log4Shell vulnerability (CVE-2021-44228) is a critical
        remote code execution bug. Related issues include CVE-2021-45046
        and CVE-2021-45105.
        """
        cves = extract_cve_ids(text)

        self.assertEqual(len(cves), 3)
        self.assertIn('CVE-2021-44228', cves)


class TestCVEVerification(unittest.TestCase):
    """Test CVE verification"""

    def setUp(self):
        """Set up known CVEs for testing"""
        self.known_cves = {
            'CVE-2021-44228',
            'CVE-2017-0144',
            'CVE-2014-0160'
        }

    def test_all_verified(self):
        """Test when all CVEs are verified"""
        cve_list = ['CVE-2021-44228', 'CVE-2017-0144']
        verified, fabricated = check_cve_ids(cve_list, self.known_cves)

        self.assertEqual(len(verified), 2)
        self.assertEqual(len(fabricated), 0)

    def test_all_fabricated(self):
        """Test when all CVEs are fabricated"""
        cve_list = ['CVE-2024-99999', 'CVE-2023-88888']
        verified, fabricated = check_cve_ids(cve_list, self.known_cves)

        self.assertEqual(len(verified), 0)
        self.assertEqual(len(fabricated), 2)

    def test_mixed(self):
        """Test mix of verified and fabricated"""
        cve_list = ['CVE-2021-44228', 'CVE-2024-99999', 'CVE-2017-0144']
        verified, fabricated = check_cve_ids(cve_list, self.known_cves)

        self.assertEqual(len(verified), 2)
        self.assertEqual(len(fabricated), 1)
        self.assertIn('CVE-2024-99999', fabricated)


class TestResponseSanitization(unittest.TestCase):
    """Test response sanitization"""

    def test_redact_mode(self):
        """Test redacting fabricated CVEs"""
        response = "The vulnerability CVE-2024-99999 affects Apache"
        fabricated = ['CVE-2024-99999']

        sanitized = sanitize_response(response, fabricated, mode='redact')

        self.assertNotIn('CVE-2024-99999', sanitized)
        self.assertIn('[UNKNOWN CVE]', sanitized)

    def test_flag_mode(self):
        """Test flagging fabricated CVEs"""
        response = "CVE-2024-99999 is critical"
        fabricated = ['CVE-2024-99999']

        sanitized = sanitize_response(response, fabricated, mode='flag')

        self.assertIn('CVE-2024-99999', sanitized)
        self.assertIn('[FABRICATED', sanitized)

    def test_remove_mode(self):
        """Test removing fabricated CVEs"""
        response = "The CVE-2024-99999 vulnerability is severe"
        fabricated = ['CVE-2024-99999']

        sanitized = sanitize_response(response, fabricated, mode='remove')

        self.assertNotIn('CVE-2024-99999', sanitized)
        self.assertNotIn('[UNKNOWN', sanitized)

    def test_no_fabricated(self):
        """Test with no fabricated CVEs"""
        response = "CVE-2021-44228 is Log4Shell"
        fabricated = []

        sanitized = sanitize_response(response, fabricated)

        self.assertEqual(sanitized, response)


class TestFallbackCVEs(unittest.TestCase):
    """Test fallback CVE list"""

    def test_fallback_not_empty(self):
        """Test that fallback list is not empty"""
        fallback = load_fallback_cves()

        self.assertGreater(len(fallback), 0)

    def test_fallback_contains_log4shell(self):
        """Test that fallback contains Log4Shell"""
        fallback = load_fallback_cves()

        self.assertIn('CVE-2021-44228', fallback)

    def test_fallback_all_valid_format(self):
        """Test that all fallback CVEs have valid format"""
        import re

        fallback = load_fallback_cves()
        pattern = r'CVE-\d{4}-\d{4,7}'

        for cve in fallback:
            self.assertIsNotNone(
                re.fullmatch(pattern, cve),
                f"Invalid CVE format: {cve}"
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
