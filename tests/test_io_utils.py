"""Unit tests for I/O utilities"""

import unittest
import json
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

from utils.io_utils import (
    load_json_file,
    save_json_file,
    load_pilot_results
)


class TestIOUtils(unittest.TestCase):
    """Test I/O utility functions"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()

    def test_save_and_load_json(self):
        """Test saving and loading JSON files"""
        test_data = {
            'test_key': 'test_value',
            'numbers': [1, 2, 3],
            'nested': {'a': 1, 'b': 2}
        }

        file_path = self.temp_path / 'test.json'

        # Save
        save_json_file(test_data, file_path)
        self.assertTrue(file_path.exists())

        # Load
        loaded_data = load_json_file(file_path)
        self.assertEqual(loaded_data, test_data)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            load_json_file(self.temp_path / 'nonexistent.json')

    def test_save_creates_directory(self):
        """Test that save_json_file creates parent directories"""
        nested_path = self.temp_path / 'nested' / 'dir' / 'test.json'
        test_data = {'key': 'value'}

        save_json_file(test_data, nested_path)

        self.assertTrue(nested_path.exists())
        self.assertTrue(nested_path.parent.exists())

    def test_load_pilot_results_empty_dir(self):
        """Test loading from empty directory"""
        results = load_pilot_results(self.temp_path)
        self.assertEqual(results, [])

    def test_load_pilot_results_with_files(self):
        """Test loading pilot results from directory"""
        # Create test result files
        result1 = [
            {'prompt_id': 'p1', 'model': 'm1', 'response': 'r1'},
            {'prompt_id': 'p2', 'model': 'm1', 'response': 'r2'}
        ]
        result2 = [
            {'prompt_id': 'p3', 'model': 'm2', 'response': 'r3'}
        ]

        save_json_file(result1, self.temp_path / 'pilot_model1.json')
        save_json_file(result2, self.temp_path / 'pilot_model2.json')

        # Load and verify
        results = load_pilot_results(self.temp_path)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['prompt_id'], 'p1')
        self.assertEqual(results[2]['prompt_id'], 'p3')

    def test_load_pilot_results_with_wrapper(self):
        """Test loading results with 'results' key wrapper"""
        wrapped_data = {
            'metadata': {'model': 'test'},
            'results': [
                {'prompt_id': 'p1', 'response': 'r1'}
            ]
        }

        save_json_file(wrapped_data, self.temp_path / 'pilot_test.json')

        results = load_pilot_results(self.temp_path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['prompt_id'], 'p1')


class TestCVEExtraction(unittest.TestCase):
    """Test CVE extraction utilities"""

    def test_cve_pattern(self):
        """Test CVE ID pattern matching"""
        import re

        pattern = r'CVE-\d{4}-\d{4,7}'

        # Valid CVE IDs
        valid_cves = [
            'CVE-2021-44228',
            'CVE-2023-12345',
            'CVE-2024-1234567'
        ]

        for cve in valid_cves:
            match = re.search(pattern, cve, re.IGNORECASE)
            self.assertIsNotNone(match, f"Failed to match valid CVE: {cve}")

        # Invalid patterns
        invalid = [
            'CVE-21-1234',      # Year too short
            'CVE-2021-123',     # ID too short
            'CVE 2021 1234',    # No hyphens
        ]

        for invalid_cve in invalid:
            match = re.search(pattern, invalid_cve)
            self.assertIsNone(match, f"Incorrectly matched invalid CVE: {invalid_cve}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
