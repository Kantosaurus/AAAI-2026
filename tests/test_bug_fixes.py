"""
Test suite to verify bug fixes
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))


class TestTypingFixes(unittest.TestCase):
    """Test that type hints are Python 3.8+ compatible"""

    def test_abstention_detector_imports(self):
        """Test that abstention_detector can be imported"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments' / 'mitigations'))
            from abstention_detector import compute_confidence_from_logprobs
            # If we get here, import succeeded
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import abstention_detector: {e}")


class TestSymbolicCheckerValidation(unittest.TestCase):
    """Test symbolic checker validation logic"""

    def test_disjoint_sets_check(self):
        """Test that verified and fabricated sets are disjoint"""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments' / 'mitigations'))
        from symbolic_checker import check_cve_ids

        known_cves = {'CVE-2021-44228', 'CVE-2017-0144'}

        # Test normal case
        cve_list = ['CVE-2021-44228', 'CVE-2024-99999']
        verified, fabricated = check_cve_ids(cve_list, known_cves)

        # Should be disjoint
        self.assertEqual(len(set(verified) & set(fabricated)), 0)

        # Verified should have known CVE
        self.assertIn('CVE-2021-44228', verified)

        # Fabricated should have unknown CVE
        self.assertIn('CVE-2024-99999', fabricated)


class TestCohenKappaEdgeCases(unittest.TestCase):
    """Test Cohen's kappa edge case handling"""

    def test_perfect_chance_agreement(self):
        """Test kappa when p_e = 1.0"""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'annotations'))
        from compute_agreement import compute_cohens_kappa

        # Create annotations where agreement is purely by chance
        # (This is a constructed edge case)
        annotations = {
            ('p1', 'm1'): [
                {'hallucination_binary': 0},
                {'hallucination_binary': 0}
            ],
        }

        kappa, stats = compute_cohens_kappa(annotations)

        # Should not crash and should return a valid number
        self.assertIsInstance(kappa, float)
        self.assertFalse(float('inf') == kappa)
        self.assertFalse(float('-inf') == kappa)

    def test_no_agreement_beyond_chance(self):
        """Test kappa when p_o = p_e"""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'annotations'))
        from compute_agreement import compute_cohens_kappa

        # This would require specific distribution
        # For now, just test that function handles edge cases
        annotations = {
            ('p1', 'm1'): [
                {'hallucination_binary': 0},
                {'hallucination_binary': 0}
            ],
            ('p2', 'm1'): [
                {'hallucination_binary': 1},
                {'hallucination_binary': 1}
            ],
        }

        kappa, stats = compute_cohens_kappa(annotations)

        # Should compute successfully
        self.assertIsInstance(kappa, float)
        self.assertGreaterEqual(kappa, -1.0)
        self.assertLessEqual(kappa, 1.0)


class TestPrepareAnnotationBatchesValidation(unittest.TestCase):
    """Test annotation batch preparation validation"""

    def test_empty_directory_error(self):
        """Test that empty directory raises helpful error"""
        import tempfile
        sys.path.insert(0, str(Path(__file__).parent.parent / 'annotations'))
        from prepare_annotation_batches import load_pilot_results

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should raise FileNotFoundError with helpful message
            with self.assertRaises(FileNotFoundError) as cm:
                load_pilot_results(Path(tmpdir))

            # Check error message is helpful
            error_msg = str(cm.exception)
            self.assertIn('pilot_*.json', error_msg)

    def test_nonexistent_directory_error(self):
        """Test that nonexistent directory raises error"""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'annotations'))
        from prepare_annotation_batches import load_pilot_results

        nonexistent = Path('/nonexistent/directory/that/does/not/exist')

        with self.assertRaises(FileNotFoundError) as cm:
            load_pilot_results(nonexistent)

        # Should mention directory not found
        error_msg = str(cm.exception)
        self.assertIn('not found', error_msg.lower())


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
