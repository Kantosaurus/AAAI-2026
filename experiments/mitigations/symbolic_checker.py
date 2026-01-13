#!/usr/bin/env python3
"""
Symbolic CVE Checker - Post-generation verification
Verifies CVE IDs and MITRE references against authoritative databases

Usage:
    python symbolic_checker.py \
        --results results/pilot/pilot_*.json \
        --nvd-list data/gold/nvd_cve_list.txt \
        --output results/symbolic_check_results.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of symbolic checking for a single response"""
    prompt_id: str
    model: str
    cve_ids_found: List[str]
    cve_ids_verified: List[str]
    cve_ids_fabricated: List[str]
    has_fabrication: bool
    original_response: str
    sanitized_response: str


def load_nvd_cve_list(nvd_file: Path = None) -> Set[str]:
    """
    Load list of known CVE IDs from NVD

    If file doesn't exist, returns a default set of well-known CVEs
    """
    known_cves = set()

    if nvd_file and nvd_file.exists():
        # Try to load from file
        if nvd_file.suffix == '.json':
            with open(nvd_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract CVE IDs depending on format
                if isinstance(data, list):
                    for item in data:
                        if 'cve_id' in item:
                            known_cves.add(item['cve_id'])
                        elif 'id' in item:
                            known_cves.add(item['id'])
                elif isinstance(data, dict):
                    # Might be nested structure
                    for key, value in data.items():
                        if 'CVE-' in key:
                            known_cves.add(key)
        elif nvd_file.suffix == '.txt':
            with open(nvd_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('CVE-'):
                        known_cves.add(line)

    # If no file or empty, use fallback set of common CVEs
    if not known_cves:
        print("Warning: No NVD list provided, using fallback set of well-known CVEs")
        known_cves = load_fallback_cves()

    return known_cves


def load_fallback_cves() -> Set[str]:
    """Load a fallback set of well-known CVEs"""
    # Common critical CVEs for testing
    return {
        'CVE-2021-44228',  # Log4Shell
        'CVE-2021-45046',  # Log4Shell variant
        'CVE-2023-34362',  # MOVEit
        'CVE-2017-0144',   # EternalBlue
        'CVE-2014-0160',   # Heartbleed
        'CVE-2017-5638',   # Apache Struts
        'CVE-2019-0708',   # BlueKeep
        'CVE-2020-0601',   # CurveBall
        'CVE-2020-1472',   # Zerologon
        'CVE-2021-34527',  # PrintNightmare
        'CVE-2022-22965',  # Spring4Shell
        'CVE-2022-30190',  # Follina
        'CVE-2023-23397',  # Outlook
        # Add more from data/cve_list_important.txt if available
    }


def extract_cve_ids(text: str) -> List[str]:
    """Extract all CVE IDs from text"""
    pattern = r'CVE-\d{4}-\d{4,7}'
    matches = re.findall(pattern, text, re.IGNORECASE)

    # Normalize to uppercase
    return [m.upper() for m in matches]


def check_cve_ids(cve_ids: List[str], known_cves: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Check CVE IDs against known list

    Returns:
        verified_ids, fabricated_ids
    """
    verified = []
    fabricated = []

    for cve_id in cve_ids:
        if cve_id in known_cves:
            verified.append(cve_id)
        else:
            fabricated.append(cve_id)

    # Validation: ensure sets are disjoint (no CVE in both lists)
    verified_set = set(verified)
    fabricated_set = set(fabricated)
    overlap = verified_set & fabricated_set
    if overlap:
        raise ValueError(f"Logic error: CVE IDs in both verified and fabricated: {overlap}")

    return verified, fabricated


def sanitize_response(response: str, fabricated_ids: List[str], mode: str = 'redact') -> str:
    """
    Sanitize response by handling fabricated CVE IDs

    Modes:
    - 'redact': Replace with [UNKNOWN CVE]
    - 'flag': Add [FABRICATED] tag
    - 'remove': Remove the CVE ID entirely
    """
    sanitized = response

    for cve_id in fabricated_ids:
        if mode == 'redact':
            # Replace with placeholder
            sanitized = re.sub(
                re.escape(cve_id),
                '[UNKNOWN CVE]',
                sanitized,
                flags=re.IGNORECASE
            )
        elif mode == 'flag':
            # Add warning tag
            sanitized = re.sub(
                re.escape(cve_id),
                f'{cve_id} [FABRICATED - NOT IN NVD]',
                sanitized,
                flags=re.IGNORECASE
            )
        elif mode == 'remove':
            # Remove entirely
            sanitized = re.sub(
                re.escape(cve_id),
                '',
                sanitized,
                flags=re.IGNORECASE
            )

    return sanitized


def process_results(results: List[Dict], known_cves: Set[str], sanitize_mode: str = 'redact') -> List[CheckResult]:
    """Process all results and check CVE IDs"""
    check_results = []

    for result in results:
        prompt_id = result.get('prompt_id', '')
        model = result.get('model', '')
        response = result.get('full_response', '')

        # Extract CVE IDs
        cve_ids = extract_cve_ids(response)

        # Check against known CVEs
        verified, fabricated = check_cve_ids(cve_ids, known_cves)

        # Sanitize if needed
        sanitized = sanitize_response(response, fabricated, sanitize_mode) if fabricated else response

        check_results.append(CheckResult(
            prompt_id=prompt_id,
            model=model,
            cve_ids_found=cve_ids,
            cve_ids_verified=verified,
            cve_ids_fabricated=fabricated,
            has_fabrication=len(fabricated) > 0,
            original_response=response,
            sanitized_response=sanitized
        ))

    return check_results


def main():
    parser = argparse.ArgumentParser(description="Symbolic CVE checker for hallucination detection")
    parser.add_argument('--results', type=str, nargs='+', required=True, help='Result JSON files to check')
    parser.add_argument('--nvd-list', type=str, default=None, help='NVD CVE list (JSON or TXT)')
    parser.add_argument('--output', type=str, required=True, help='Output file for check results')
    parser.add_argument('--sanitize-mode', type=str, default='redact', choices=['redact', 'flag', 'remove'],
                       help='How to handle fabricated CVEs')

    args = parser.parse_args()

    # Load NVD CVE list
    print("Loading NVD CVE list...")
    nvd_file = Path(args.nvd_list) if args.nvd_list else None
    known_cves = load_nvd_cve_list(nvd_file)
    print(f"Loaded {len(known_cves)} known CVE IDs")

    # Load results
    print("\nLoading result files...")
    all_results = []
    for result_file in args.results:
        print(f"  Loading {result_file}...")
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict) and 'results' in data:
                all_results.extend(data['results'])

    print(f"Loaded {len(all_results)} results")

    # Process results
    print("\nChecking CVE IDs...")
    check_results = process_results(all_results, known_cves, args.sanitize_mode)

    # Compute statistics
    total_responses = len(check_results)
    responses_with_cves = sum(1 for r in check_results if r.cve_ids_found)
    responses_with_fabrications = sum(1 for r in check_results if r.has_fabrication)
    total_cves_found = sum(len(r.cve_ids_found) for r in check_results)
    total_fabricated = sum(len(r.cve_ids_fabricated) for r in check_results)

    # Save results
    output_data = {
        'statistics': {
            'total_responses': total_responses,
            'responses_with_cves': responses_with_cves,
            'responses_with_fabrications': responses_with_fabrications,
            'fabrication_rate': responses_with_fabrications / total_responses if total_responses > 0 else 0,
            'total_cves_found': total_cves_found,
            'total_cves_verified': total_cves_found - total_fabricated,
            'total_cves_fabricated': total_fabricated,
            'cve_fabrication_rate': total_fabricated / total_cves_found if total_cves_found > 0 else 0
        },
        'known_cves_count': len(known_cves),
        'sanitize_mode': args.sanitize_mode,
        'results': [
            {
                'prompt_id': r.prompt_id,
                'model': r.model,
                'cve_ids_found': r.cve_ids_found,
                'cve_ids_verified': r.cve_ids_verified,
                'cve_ids_fabricated': r.cve_ids_fabricated,
                'has_fabrication': r.has_fabrication,
                'original_response': r.original_response,
                'sanitized_response': r.sanitized_response
            }
            for r in check_results
        ]
    }

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SYMBOLIC CHECKER RESULTS")
    print("="*60)
    print(f"\nTotal responses checked: {total_responses}")
    print(f"Responses with CVE citations: {responses_with_cves} ({responses_with_cves/total_responses:.1%})")
    print(f"Responses with fabricated CVEs: {responses_with_fabrications} ({responses_with_fabrications/total_responses:.1%})")
    print(f"\nTotal CVE IDs found: {total_cves_found}")
    print(f"  Verified (in NVD): {total_cves_found - total_fabricated}")
    print(f"  Fabricated: {total_fabricated}")
    print(f"  Fabrication rate: {total_fabricated/total_cves_found:.1%}" if total_cves_found > 0 else "  Fabrication rate: N/A")

    print(f"\n✓ Results saved to {output_file}")

    # Show examples of fabricated CVEs
    if total_fabricated > 0:
        print("\nExamples of fabricated CVEs:")
        count = 0
        for r in check_results:
            if r.cve_ids_fabricated:
                print(f"  {r.prompt_id}: {', '.join(r.cve_ids_fabricated[:3])}")
                count += 1
                if count >= 5:
                    break


if __name__ == '__main__':
    main()
