#!/usr/bin/env python3
"""
Validate Gold Truth CVE Dataset

Performs integrity checks on the gold truth dataset:
- CVE ID format validation
- Duplicate detection
- Collision check (synthetic CVEs that now exist in NVD)
- CVSS score range validation
- Required field completeness

Usage:
    python validate_gold_truth.py --input gold_truth_cves.json --report validation_report.txt
    python validate_gold_truth.py --check-collisions --input gold_truth_cves.json
"""

import argparse
import json
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import requests
import time


class GoldTruthValidator:
    """Validate gold truth CVE dataset"""

    CVE_ID_PATTERN = re.compile(r'^CVE-\d{4}-\d{4,7}$')

    def __init__(self, check_nvd: bool = False):
        """
        Initialize validator

        Args:
            check_nvd: Whether to check synthetic CVEs against live NVD API
        """
        self.check_nvd = check_nvd
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_entries': 0,
            'real_cves': 0,
            'synthetic_cves': 0,
            'format_errors': 0,
            'duplicates': 0,
            'collisions': 0,
            'missing_fields': 0,
            'cvss_errors': 0
        }

    def validate_dataset(self, dataset: Dict) -> bool:
        """
        Validate complete dataset

        Args:
            dataset: Gold truth dataset dict

        Returns:
            True if validation passes, False otherwise
        """
        print("Validating gold truth dataset...")
        print()

        # Check metadata
        if 'metadata' not in dataset:
            self.errors.append("Missing 'metadata' section")
            return False

        if 'cves' not in dataset:
            self.errors.append("Missing 'cves' section")
            return False

        cves = dataset['cves']
        self.stats['total_entries'] = len(cves)

        print(f"Total entries: {len(cves)}")
        print()

        # Track CVE IDs for duplicate detection
        seen_cve_ids = set()

        # Validate each CVE entry
        for i, cve_entry in enumerate(cves):
            self._validate_entry(cve_entry, i, seen_cve_ids)

        # Check for synthetic CVE collisions if requested
        if self.check_nvd:
            synthetic_cves = [
                entry for entry in cves
                if not entry.get('exists', True)
            ]
            print(f"\nChecking {len(synthetic_cves)} synthetic CVEs against NVD API...")
            self._check_collisions(synthetic_cves)

        # Print results
        self._print_results()

        return len(self.errors) == 0

    def _validate_entry(self, entry: Dict, index: int, seen_ids: set):
        """Validate single CVE entry"""
        # Check CVE ID
        cve_id = entry.get('cve_id')
        if not cve_id:
            self.errors.append(f"Entry {index}: Missing cve_id")
            self.stats['missing_fields'] += 1
            return

        # Validate CVE ID format
        if not self.CVE_ID_PATTERN.match(cve_id):
            self.errors.append(f"{cve_id}: Invalid CVE ID format")
            self.stats['format_errors'] += 1

        # Check for duplicates
        if cve_id in seen_ids:
            self.errors.append(f"{cve_id}: Duplicate entry")
            self.stats['duplicates'] += 1
        seen_ids.add(cve_id)

        # Check exists field
        exists = entry.get('exists')
        if exists is None:
            self.errors.append(f"{cve_id}: Missing 'exists' field")
            self.stats['missing_fields'] += 1
            return

        # Count by type
        if exists:
            self.stats['real_cves'] += 1
            self._validate_real_cve(entry, cve_id)
        else:
            self.stats['synthetic_cves'] += 1
            self._validate_synthetic_cve(entry, cve_id)

    def _validate_real_cve(self, entry: Dict, cve_id: str):
        """Validate real CVE entry from NVD"""
        # Check for nvd_data
        if 'nvd_data' not in entry:
            self.errors.append(f"{cve_id}: Missing 'nvd_data' for real CVE")
            self.stats['missing_fields'] += 1
            return

        nvd_data = entry['nvd_data']

        # Validate CVSS score if present
        cvss_score = nvd_data.get('cvss_v3_score')
        if cvss_score is not None:
            if not isinstance(cvss_score, (int, float)):
                self.errors.append(f"{cve_id}: CVSS score is not numeric")
                self.stats['cvss_errors'] += 1
            elif not (0 <= cvss_score <= 10):
                self.errors.append(f"{cve_id}: CVSS score {cvss_score} out of range (0-10)")
                self.stats['cvss_errors'] += 1

        # Check description
        if not nvd_data.get('description'):
            self.warnings.append(f"{cve_id}: Missing or empty description")

    def _validate_synthetic_cve(self, entry: Dict, cve_id: str):
        """Validate synthetic CVE entry"""
        # Check for synthetic_data
        if 'synthetic_data' not in entry:
            self.warnings.append(f"{cve_id}: Missing 'synthetic_data' for synthetic CVE")

        # Check test_expectations
        if 'test_expectations' not in entry:
            self.warnings.append(f"{cve_id}: Missing 'test_expectations'")

    def _check_collisions(self, synthetic_cves: List[Dict]):
        """Check if synthetic CVEs now exist in NVD (collision detection)"""
        for i, entry in enumerate(synthetic_cves):
            cve_id = entry['cve_id']

            # Rate limiting
            if i > 0 and i % 5 == 0:
                print(f"  Checked {i}/{len(synthetic_cves)}...")
                time.sleep(6)  # 5 requests per 30 seconds

            exists_in_nvd = self._check_nvd_existence(cve_id)

            if exists_in_nvd:
                self.errors.append(
                    f"{cve_id}: COLLISION - Synthetic CVE now exists in NVD!"
                )
                self.stats['collisions'] += 1
            time.sleep(0.6)

        print(f"  Completed NVD collision check")

    def _check_nvd_existence(self, cve_id: str) -> bool:
        """Check if CVE exists in NVD via API"""
        try:
            url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
            response = requests.get(
                url,
                params={"cveId": cve_id},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('totalResults', 0) > 0
            else:
                self.warnings.append(
                    f"{cve_id}: NVD API check failed (HTTP {response.status_code})"
                )
                return False

        except Exception as e:
            self.warnings.append(f"{cve_id}: NVD API check failed ({str(e)})")
            return False

    def _print_results(self):
        """Print validation results"""
        print()
        print("=" * 60)
        print("Validation Results")
        print("=" * 60)
        print()

        print("Statistics:")
        print(f"  Total entries:    {self.stats['total_entries']}")
        print(f"  Real CVEs:        {self.stats['real_cves']}")
        print(f"  Synthetic CVEs:   {self.stats['synthetic_cves']}")
        print()

        print("Issues Found:")
        print(f"  Format errors:    {self.stats['format_errors']}")
        print(f"  Duplicates:       {self.stats['duplicates']}")
        print(f"  Missing fields:   {self.stats['missing_fields']}")
        print(f"  CVSS errors:      {self.stats['cvss_errors']}")
        if self.check_nvd:
            print(f"  Collisions:       {self.stats['collisions']}")
        print()

        if self.errors:
            print(f"ERRORS ({len(self.errors)}):")
            for error in self.errors[:20]:  # Limit output
                print(f"  ✗ {error}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more errors")
            print()

        if self.warnings:
            print(f"WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:
                print(f"  ⚠ {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
            print()

        if not self.errors and not self.warnings:
            print("✓ All validation checks passed!")
            print()
        elif not self.errors:
            print("✓ No critical errors, but warnings present")
            print()
        else:
            print("✗ Validation failed with errors")
            print()

    def save_report(self, filepath: str):
        """Save validation report to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Gold Truth CVE Dataset Validation Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
            f.write("\n")

            f.write("Statistics:\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            if self.errors:
                f.write(f"ERRORS ({len(self.errors)}):\n")
                for error in self.errors:
                    f.write(f"  ✗ {error}\n")
                f.write("\n")

            if self.warnings:
                f.write(f"WARNINGS ({len(self.warnings)}):\n")
                for warning in self.warnings:
                    f.write(f"  ⚠ {warning}\n")
                f.write("\n")

            if not self.errors and not self.warnings:
                f.write("✓ All validation checks passed!\n")

        print(f"Validation report saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate gold truth CVE dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input gold truth JSON file'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Output validation report file (optional)'
    )
    parser.add_argument(
        '--check-collisions',
        action='store_true',
        help='Check synthetic CVEs against live NVD API for collisions (slow)'
    )

    args = parser.parse_args()

    # Load dataset
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate
    validator = GoldTruthValidator(check_nvd=args.check_collisions)
    success = validator.validate_dataset(dataset)

    # Save report if requested
    if args.report:
        validator.save_report(args.report)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
