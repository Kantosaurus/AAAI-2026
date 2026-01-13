#!/usr/bin/env python3
"""
Symbolic CVE Checker - Post-generation verification (Enhanced Version)
Verifies CVE IDs and MITRE references against authoritative databases

This version uses:
- Shared utilities for I/O operations
- Proper logging instead of print statements
- Enhanced error handling

Usage:
    python symbolic_checker_v2.py \
        --results results/pilot/pilot_*.json \
        --nvd-list data/gold/nvd_cve_list.txt \
        --output results/symbolic_check_results.json \
        --log-file logs/symbolic_checker.log
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io_utils import load_json_file, save_json_file, load_multiple_result_files
from utils.logging_utils import setup_logger

# Initialize logger (will be configured in main)
logger = None


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

    Args:
        nvd_file: Path to NVD data file (JSON or TXT)

    Returns:
        Set of known CVE IDs
    """
    known_cves = set()

    if nvd_file and nvd_file.exists():
        logger.info(f"Loading NVD CVE list from: {nvd_file}")

        try:
            if nvd_file.suffix == '.json':
                data = load_json_file(nvd_file)

                # Extract CVE IDs depending on format
                if isinstance(data, list):
                    for item in data:
                        if 'cve_id' in item:
                            known_cves.add(item['cve_id'])
                        elif 'id' in item:
                            known_cves.add(item['id'])
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if 'CVE-' in key:
                            known_cves.add(key)

            elif nvd_file.suffix == '.txt':
                with open(nvd_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('CVE-'):
                            known_cves.add(line)

            logger.info(f"Loaded {len(known_cves)} CVEs from file")

        except Exception as e:
            logger.error(f"Error loading NVD file: {e}")
            logger.warning("Falling back to default CVE set")

    # If no file or empty, use fallback set
    if not known_cves:
        logger.warning("No NVD list provided, using fallback set of well-known CVEs")
        known_cves = load_fallback_cves()

    return known_cves


def load_fallback_cves() -> Set[str]:
    """
    Load a fallback set of well-known CVEs

    Returns:
        Set of common critical CVEs
    """
    fallback = {
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
    }

    logger.info(f"Using fallback set of {len(fallback)} well-known CVEs")
    return fallback


def extract_cve_ids(text: str) -> List[str]:
    """
    Extract all CVE IDs from text

    Args:
        text: Text to search for CVE IDs

    Returns:
        List of CVE IDs found (normalized to uppercase)
    """
    pattern = r'CVE-\d{4}-\d{4,7}'
    matches = re.findall(pattern, text, re.IGNORECASE)

    # Normalize to uppercase
    normalized = [m.upper() for m in matches]

    if normalized:
        logger.debug(f"Extracted {len(normalized)} CVE IDs from text")

    return normalized


def check_cve_ids(cve_ids: List[str], known_cves: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Check CVE IDs against known list

    Args:
        cve_ids: List of CVE IDs to check
        known_cves: Set of known valid CVE IDs

    Returns:
        Tuple of (verified_ids, fabricated_ids)
    """
    verified = []
    fabricated = []

    for cve_id in cve_ids:
        if cve_id in known_cves:
            verified.append(cve_id)
            logger.debug(f"Verified CVE: {cve_id}")
        else:
            fabricated.append(cve_id)
            logger.warning(f"Fabricated CVE detected: {cve_id}")

    return verified, fabricated


def sanitize_response(response: str, fabricated_ids: List[str], mode: str = 'redact') -> str:
    """
    Sanitize response by handling fabricated CVE IDs

    Args:
        response: Original response text
        fabricated_ids: List of fabricated CVE IDs
        mode: Sanitization mode ('redact', 'flag', or 'remove')

    Returns:
        Sanitized response text
    """
    if not fabricated_ids:
        return response

    sanitized = response
    logger.debug(f"Sanitizing response with mode: {mode}")

    for cve_id in fabricated_ids:
        if mode == 'redact':
            sanitized = re.sub(
                re.escape(cve_id),
                '[UNKNOWN CVE]',
                sanitized,
                flags=re.IGNORECASE
            )
        elif mode == 'flag':
            sanitized = re.sub(
                re.escape(cve_id),
                f'{cve_id} [FABRICATED - NOT IN NVD]',
                sanitized,
                flags=re.IGNORECASE
            )
        elif mode == 'remove':
            sanitized = re.sub(
                re.escape(cve_id),
                '',
                sanitized,
                flags=re.IGNORECASE
            )

    return sanitized


def process_results(results: List[Dict], known_cves: Set[str], sanitize_mode: str = 'redact') -> List[CheckResult]:
    """
    Process all results and check CVE IDs

    Args:
        results: List of model output results
        known_cves: Set of known CVE IDs
        sanitize_mode: Mode for sanitizing fabricated CVEs

    Returns:
        List of CheckResult objects
    """
    logger.info(f"Processing {len(results)} results")
    check_results = []

    for i, result in enumerate(results):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(results)} results")

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

    logger.info(f"Processing complete: {len(check_results)} results checked")
    return check_results


def main():
    parser = argparse.ArgumentParser(description="Symbolic CVE checker for hallucination detection")
    parser.add_argument('--results', type=str, nargs='+', required=True, help='Result JSON files to check')
    parser.add_argument('--nvd-list', type=str, default=None, help='NVD CVE list (JSON or TXT)')
    parser.add_argument('--output', type=str, required=True, help='Output file for check results')
    parser.add_argument('--sanitize-mode', type=str, default='redact', choices=['redact', 'flag', 'remove'],
                       help='How to handle fabricated CVEs')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Set up logging
    global logger
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger(
        __name__,
        level=getattr(__import__('logging'), log_level),
        log_file=Path(args.log_file) if args.log_file else None
    )

    logger.info("="*60)
    logger.info("SYMBOLIC CVE CHECKER - Starting")
    logger.info("="*60)

    # Load NVD CVE list
    nvd_file = Path(args.nvd_list) if args.nvd_list else None
    known_cves = load_nvd_cve_list(nvd_file)
    logger.info(f"Loaded {len(known_cves)} known CVE IDs")

    # Load results
    logger.info("Loading result files...")
    try:
        all_results = load_multiple_result_files(args.results)
        logger.info(f"Loaded {len(all_results)} results from {len(args.results)} file patterns")
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1

    # Process results
    logger.info("Checking CVE IDs...")
    try:
        check_results = process_results(all_results, known_cves, args.sanitize_mode)
    except Exception as e:
        logger.error(f"Error processing results: {e}")
        return 1

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

    logger.info(f"Saving results to: {args.output}")
    try:
        save_json_file(output_data, args.output)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1

    # Print summary
    logger.info("="*60)
    logger.info("SYMBOLIC CHECKER RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total responses checked: {total_responses}")
    logger.info(f"Responses with CVE citations: {responses_with_cves} ({responses_with_cves/total_responses:.1%})")
    logger.info(f"Responses with fabricated CVEs: {responses_with_fabrications} ({responses_with_fabrications/total_responses:.1%})")
    logger.info(f"\nTotal CVE IDs found: {total_cves_found}")
    logger.info(f"  Verified (in NVD): {total_cves_found - total_fabricated}")
    logger.info(f"  Fabricated: {total_fabricated}")
    if total_cves_found > 0:
        logger.info(f"  Fabrication rate: {total_fabricated/total_cves_found:.1%}")

    # Show examples of fabricated CVEs
    if total_fabricated > 0:
        logger.info("\nExamples of fabricated CVEs:")
        count = 0
        for r in check_results:
            if r.cve_ids_fabricated:
                logger.info(f"  {r.prompt_id}: {', '.join(r.cve_ids_fabricated[:3])}")
                count += 1
                if count >= 5:
                    break

    logger.info("="*60)
    logger.info("SYMBOLIC CVE CHECKER - Complete")
    logger.info("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
