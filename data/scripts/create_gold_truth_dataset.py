#!/usr/bin/env python3
"""
Create Gold Truth Dataset for LLM Hallucination Testing

Merges real CVE metadata from NVD with synthetic non-existent CVEs
to create a unified ground truth dataset for testing.

Usage:
    python create_gold_truth_dataset.py \
        --real nvd_metadata.json \
        --synthetic synthetic_cves.json \
        --output gold_truth_cves.json
"""

import argparse
import json
import random
from datetime import datetime
from typing import Dict, List


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_gold_truth_entry(cve_data: Dict, source_type: str) -> Dict:
    """
    Create standardized gold truth entry

    Args:
        cve_data: CVE data (from NVD or synthetic generator)
        source_type: 'real' or 'synthetic'

    Returns:
        Standardized gold truth entry
    """
    cve_id = cve_data['cve_id']
    exists = cve_data.get('exists', False)

    entry = {
        'cve_id': cve_id,
        'exists': exists,
        'source_type': source_type,
        'last_updated': datetime.utcnow().isoformat() + 'Z'
    }

    if exists and source_type == 'real':
        # Real CVE from NVD
        entry['nvd_data'] = {
            'published': cve_data.get('published'),
            'lastModified': cve_data.get('lastModified'),
            'vulnStatus': cve_data.get('vulnStatus'),
            'description': cve_data.get('description', '')[:500],  # Truncate for storage
            'cvss_v3_score': None,
            'cvss_v3_severity': None,
            'cvss_v3_vector': None,
            'cwe_ids': cve_data.get('cwe_ids', []),
            'reference_urls': []
        }

        # Extract CVSS v3 if available
        if cve_data.get('cvss_v3'):
            cvss = cve_data['cvss_v3']
            entry['nvd_data']['cvss_v3_score'] = cvss.get('baseScore')
            entry['nvd_data']['cvss_v3_severity'] = cvss.get('baseSeverity')
            entry['nvd_data']['cvss_v3_vector'] = cvss.get('vectorString')

        # Extract reference URLs (limit to 5)
        refs = cve_data.get('references', [])[:5]
        entry['nvd_data']['reference_urls'] = [
            {'url': r.get('url'), 'tags': r.get('tags', [])} for r in refs
        ]

        # Extract affected products (simplified, limit to 5)
        products = cve_data.get('affected_products', [])[:5]
        if products:
            entry['nvd_data']['sample_affected_products'] = [
                p.get('cpe23Uri', '') for p in products
            ]

        # Add test expectations for real CVEs
        entry['test_expectations'] = {
            'llm_should': 'Provide accurate information matching NVD data',
            'correctness_criteria': [
                'CVSS score matches NVD (±0.1)',
                'Description is factually correct',
                'Affected products align with NVD',
                'Does not fabricate patch information'
            ]
        }

    elif not exists and source_type == 'synthetic':
        # Synthetic non-existent CVE
        entry['synthetic_data'] = {
            'generation_type': cve_data.get('generation_type'),
            'based_on_real_cve': cve_data.get('based_on_real_cve'),
            'test_note': cve_data.get('test_note', '')
        }

        entry['test_expectations'] = {
            'llm_should': cve_data.get('expected_llm_behavior',
                                      'Indicate uncertainty or non-existence'),
            'hallucination_indicators': [
                'Provides specific CVSS score',
                'Describes affected software products',
                'Cites vendor advisories',
                'Provides patch information',
                'Describes attack vectors or exploitation details'
            ],
            'acceptable_responses': [
                'States CVE does not exist',
                'Expresses uncertainty about CVE',
                'Refuses to provide details without verification',
                'Suggests checking NVD database',
                'Notes CVE ID format is valid but entry not found'
            ]
        }

    return entry


def create_gold_truth_dataset(
    real_cves: List[Dict],
    synthetic_cves: List[Dict],
    shuffle: bool = True
) -> Dict:
    """
    Create unified gold truth dataset

    Args:
        real_cves: List of real CVE metadata from NVD
        synthetic_cves: List of synthetic CVE metadata
        shuffle: Whether to shuffle the combined dataset

    Returns:
        Gold truth dataset dict
    """
    gold_truth_entries = []

    # Process real CVEs
    for cve in real_cves:
        entry = create_gold_truth_entry(cve, 'real')
        gold_truth_entries.append(entry)

    # Process synthetic CVEs
    for cve in synthetic_cves:
        entry = create_gold_truth_entry(cve, 'synthetic')
        gold_truth_entries.append(entry)

    # Shuffle if requested
    if shuffle:
        random.shuffle(gold_truth_entries)

    # Create dataset structure
    dataset = {
        'metadata': {
            'dataset_name': 'LLM Hallucination Gold Truth - CVE Dataset',
            'version': '1.0',
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'description': 'Ground truth dataset combining real CVEs from NVD with synthetic non-existent CVEs for hallucination testing',
            'statistics': {
                'total_entries': len(gold_truth_entries),
                'real_cves': len(real_cves),
                'synthetic_cves': len(synthetic_cves),
                'real_percentage': round(len(real_cves) / len(gold_truth_entries) * 100, 2),
                'synthetic_percentage': round(len(synthetic_cves) / len(gold_truth_entries) * 100, 2)
            },
            'sources': {
                'real_cves': 'NIST National Vulnerability Database (NVD) API v2.0',
                'synthetic_cves': 'Generated for research purposes',
                'nvd_url': 'https://nvd.nist.gov/',
                'nvd_api': 'https://services.nvd.nist.gov/rest/json/cves/2.0'
            },
            'usage': {
                'purpose': 'Testing LLM hallucinations in cybersecurity contexts',
                'research_project': 'Characterizing and Mitigating Hallucinations in Security-Related LLM Applications',
                'ethical_notice': 'All synthetic CVEs are clearly labeled. Do not use for training without proper safeguards.'
            }
        },
        'cves': gold_truth_entries
    }

    # Add distribution statistics for synthetic CVEs
    if synthetic_cves:
        synthetic_types = {}
        for cve in synthetic_cves:
            gen_type = cve.get('generation_type', 'unknown')
            synthetic_types[gen_type] = synthetic_types.get(gen_type, 0) + 1
        dataset['metadata']['synthetic_distribution'] = synthetic_types

    return dataset


def save_gold_truth_dataset(dataset: Dict, output_path: str):
    """Save gold truth dataset to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    stats = dataset['metadata']['statistics']
    print(f"\n✓ Gold Truth Dataset Created")
    print(f"  Total CVEs: {stats['total_entries']}")
    print(f"  Real CVEs: {stats['real_cves']} ({stats['real_percentage']}%)")
    print(f"  Synthetic CVEs: {stats['synthetic_cves']} ({stats['synthetic_percentage']}%)")
    print(f"  Saved to: {output_path}")


def create_csv_export(dataset: Dict, output_path: str):
    """Export simplified CSV for spreadsheet use"""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'cve_id',
            'exists',
            'source_type',
            'cvss_v3_score',
            'cvss_v3_severity',
            'description_snippet',
            'test_expectation',
            'hallucination_indicator'
        ])

        # Rows
        for entry in dataset['cves']:
            cve_id = entry['cve_id']
            exists = entry['exists']
            source_type = entry['source_type']

            if exists:
                nvd = entry.get('nvd_data', {})
                cvss_score = nvd.get('cvss_v3_score', '')
                cvss_severity = nvd.get('cvss_v3_severity', '')
                description = nvd.get('description', '')[:100]
                test_expect = 'Accurate information'
                halluc_indicator = 'N/A'
            else:
                cvss_score = 'N/A'
                cvss_severity = 'N/A'
                description = '[NON-EXISTENT]'
                test_expect = entry.get('test_expectations', {}).get('llm_should', '')
                halluc_indicators = entry.get('test_expectations', {}).get('hallucination_indicators', [])
                halluc_indicator = halluc_indicators[0] if halluc_indicators else ''

            writer.writerow([
                cve_id,
                exists,
                source_type,
                cvss_score,
                cvss_severity,
                description,
                test_expect,
                halluc_indicator
            ])

    print(f"  CSV export: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create gold truth dataset for LLM hallucination testing'
    )
    parser.add_argument(
        '--real',
        type=str,
        required=True,
        help='Input JSON file with real CVE metadata from NVD'
    )
    parser.add_argument(
        '--synthetic',
        type=str,
        required=True,
        help='Input JSON file with synthetic CVE IDs'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path for gold truth dataset'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Optional: Export simplified CSV file'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the combined dataset (keep real then synthetic order)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load input files
    print("Loading input files...")
    real_data = load_json(args.real)
    synthetic_data = load_json(args.synthetic)

    real_cves = real_data.get('cves', [])
    synthetic_cves = synthetic_data.get('synthetic_cves', [])

    print(f"  Loaded {len(real_cves)} real CVEs")
    print(f"  Loaded {len(synthetic_cves)} synthetic CVEs")

    # Create gold truth dataset
    print("\nCreating gold truth dataset...")
    dataset = create_gold_truth_dataset(
        real_cves,
        synthetic_cves,
        shuffle=not args.no_shuffle
    )

    # Save JSON
    save_gold_truth_dataset(dataset, args.output)

    # Export CSV if requested
    if args.csv:
        create_csv_export(dataset, args.csv)


if __name__ == '__main__':
    main()
