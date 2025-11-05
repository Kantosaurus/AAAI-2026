#!/usr/bin/env python3
"""
NVD CVE Metadata Fetcher
Pulls CVE metadata from NIST National Vulnerability Database API v2.0

Usage:
    python fetch_nvd_metadata.py --cve-list cve_list.txt --output nvd_metadata.json
    python fetch_nvd_metadata.py --year 2023 --output nvd_2023.json
    python fetch_nvd_metadata.py --api-key YOUR_KEY --year 2023 --output nvd_2023.json

Requirements:
    pip install requests
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests


class NVDFetcher:
    """Fetch CVE metadata from NVD API v2.0"""

    BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NVD Fetcher

        Args:
            api_key: Optional NVD API key for higher rate limits
                     Without key: 5 requests per 30 seconds
                     With key: 50 requests per 30 seconds
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"apiKey": api_key})

        # Rate limiting
        self.requests_made = 0
        self.window_start = time.time()
        self.rate_limit = 50 if api_key else 5
        self.window_seconds = 30

    def _rate_limit_wait(self):
        """Implement rate limiting"""
        self.requests_made += 1

        if self.requests_made >= self.rate_limit:
            elapsed = time.time() - self.window_start
            if elapsed < self.window_seconds:
                sleep_time = self.window_seconds - elapsed + 1  # +1 for safety
                print(f"Rate limit reached. Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)

            # Reset window
            self.requests_made = 0
            self.window_start = time.time()
        else:
            # Small delay between requests
            time.sleep(0.6)

    def fetch_cve_by_id(self, cve_id: str) -> Optional[Dict]:
        """
        Fetch single CVE by ID

        Args:
            cve_id: CVE identifier (e.g., CVE-2021-44228)

        Returns:
            Dict with CVE metadata or None if not found
        """
        self._rate_limit_wait()

        try:
            response = self.session.get(
                self.BASE_URL,
                params={"cveId": cve_id},
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            if data.get('totalResults', 0) == 0:
                return None

            # Extract first vulnerability
            vuln = data['vulnerabilities'][0]['cve']
            return self._extract_metadata(vuln, cve_id)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {cve_id}: {e}", file=sys.stderr)
            return None

    def fetch_cves_by_year(self, year: int, start_index: int = 0) -> List[Dict]:
        """
        Fetch all CVEs published in a given year

        Args:
            year: Year to fetch (e.g., 2023)
            start_index: Pagination start index

        Returns:
            List of CVE metadata dicts
        """
        cves = []
        results_per_page = 2000  # Max allowed by API

        pub_start = f"{year}-01-01T00:00:00.000"
        pub_end = f"{year}-12-31T23:59:59.999"

        print(f"Fetching CVEs for year {year}...")

        while True:
            self._rate_limit_wait()

            try:
                response = self.session.get(
                    self.BASE_URL,
                    params={
                        "pubStartDate": pub_start,
                        "pubEndDate": pub_end,
                        "resultsPerPage": results_per_page,
                        "startIndex": start_index
                    },
                    timeout=60
                )
                response.raise_for_status()

                data = response.json()
                total_results = data.get('totalResults', 0)
                vulnerabilities = data.get('vulnerabilities', [])

                print(f"  Fetched {len(vulnerabilities)} CVEs (total: {total_results}, offset: {start_index})")

                for vuln_wrapper in vulnerabilities:
                    vuln = vuln_wrapper['cve']
                    cve_id = vuln.get('id', 'UNKNOWN')
                    metadata = self._extract_metadata(vuln, cve_id)
                    if metadata:
                        cves.append(metadata)

                # Check if we have all results
                if start_index + len(vulnerabilities) >= total_results:
                    break

                start_index += results_per_page

            except requests.exceptions.RequestException as e:
                print(f"Error fetching year {year} at index {start_index}: {e}", file=sys.stderr)
                break

        print(f"Total CVEs fetched for {year}: {len(cves)}")
        return cves

    def _extract_metadata(self, vuln: Dict, cve_id: str) -> Dict:
        """
        Extract relevant metadata from NVD CVE object

        Args:
            vuln: Raw CVE object from NVD API
            cve_id: CVE identifier

        Returns:
            Simplified metadata dict
        """
        # Extract description (English)
        descriptions = vuln.get('descriptions', [])
        description = next(
            (d['value'] for d in descriptions if d.get('lang') == 'en'),
            "No description available"
        )

        # Extract CVSS scores
        metrics = vuln.get('metrics', {})
        cvss_v3 = None
        cvss_v2 = None

        # CVSS v3.x
        for version in ['cvssMetricV31', 'cvssMetricV30']:
            if version in metrics and metrics[version]:
                cvss_data = metrics[version][0]['cvssData']
                cvss_v3 = {
                    'version': cvss_data.get('version', '3.1'),
                    'baseScore': cvss_data.get('baseScore'),
                    'baseSeverity': cvss_data.get('baseSeverity'),
                    'vectorString': cvss_data.get('vectorString'),
                    'attackVector': cvss_data.get('attackVector'),
                    'attackComplexity': cvss_data.get('attackComplexity'),
                    'privilegesRequired': cvss_data.get('privilegesRequired'),
                    'userInteraction': cvss_data.get('userInteraction'),
                    'scope': cvss_data.get('scope'),
                    'confidentialityImpact': cvss_data.get('confidentialityImpact'),
                    'integrityImpact': cvss_data.get('integrityImpact'),
                    'availabilityImpact': cvss_data.get('availabilityImpact')
                }
                break

        # CVSS v2
        if 'cvssMetricV2' in metrics and metrics['cvssMetricV2']:
            cvss_data = metrics['cvssMetricV2'][0]['cvssData']
            cvss_v2 = {
                'version': '2.0',
                'baseScore': cvss_data.get('baseScore'),
                'vectorString': cvss_data.get('vectorString')
            }

        # Extract CWE information
        weaknesses = vuln.get('weaknesses', [])
        cwe_ids = []
        for weakness in weaknesses:
            for desc in weakness.get('description', []):
                cwe_id = desc.get('value', '')
                if cwe_id.startswith('CWE-'):
                    cwe_ids.append(cwe_id)

        # Extract references
        references = []
        for ref in vuln.get('references', [])[:10]:  # Limit to 10 references
            references.append({
                'url': ref.get('url'),
                'source': ref.get('source'),
                'tags': ref.get('tags', [])
            })

        # Extract configurations (affected products)
        configurations = vuln.get('configurations', [])
        affected_products = []
        for config in configurations:
            for node in config.get('nodes', []):
                for cpe_match in node.get('cpeMatch', [])[:5]:  # Limit per node
                    if cpe_match.get('vulnerable', True):
                        affected_products.append({
                            'cpe23Uri': cpe_match.get('criteria', ''),
                            'versionStartIncluding': cpe_match.get('versionStartIncluding'),
                            'versionEndExcluding': cpe_match.get('versionEndExcluding'),
                            'versionEndIncluding': cpe_match.get('versionEndIncluding')
                        })

        return {
            'cve_id': cve_id,
            'exists': True,
            'source': 'NVD',
            'published': vuln.get('published'),
            'lastModified': vuln.get('lastModified'),
            'vulnStatus': vuln.get('vulnStatus'),  # ANALYZED, MODIFIED, etc.
            'description': description,
            'cvss_v3': cvss_v3,
            'cvss_v2': cvss_v2,
            'cwe_ids': cwe_ids,
            'references': references,
            'affected_products': affected_products[:20],  # Limit total
            'fetched_at': datetime.utcnow().isoformat() + 'Z'
        }


def load_cve_list(filepath: str) -> List[str]:
    """Load CVE IDs from text file (one per line)"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip().startswith('CVE-')]


def save_metadata(data: List[Dict], output_path: str):
    """Save metadata to JSON file"""
    output = {
        'metadata': {
            'total_cves': len(data),
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'source': 'NIST NVD API v2.0',
            'api_url': 'https://services.nvd.nist.gov/rest/json/cves/2.0'
        },
        'cves': data
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(data)} CVE entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch CVE metadata from NIST NVD API v2.0'
    )
    parser.add_argument(
        '--cve-list',
        type=str,
        help='Path to text file with CVE IDs (one per line)'
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Fetch all CVEs published in this year (e.g., 2023)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='NVD API key for higher rate limits (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path'
    )

    args = parser.parse_args()

    if not args.cve_list and not args.year:
        parser.error("Either --cve-list or --year must be specified")

    # Initialize fetcher
    fetcher = NVDFetcher(api_key=args.api_key)

    cve_metadata = []

    # Fetch by list
    if args.cve_list:
        cve_ids = load_cve_list(args.cve_list)
        print(f"Loaded {len(cve_ids)} CVE IDs from {args.cve_list}")

        for i, cve_id in enumerate(cve_ids, 1):
            print(f"[{i}/{len(cve_ids)}] Fetching {cve_id}...")
            metadata = fetcher.fetch_cve_by_id(cve_id)
            if metadata:
                cve_metadata.append(metadata)
            else:
                print(f"  WARNING: {cve_id} not found in NVD")

    # Fetch by year
    if args.year:
        year_cves = fetcher.fetch_cves_by_year(args.year)
        cve_metadata.extend(year_cves)

    # Save results
    if cve_metadata:
        save_metadata(cve_metadata, args.output)
    else:
        print("No CVE metadata fetched.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
