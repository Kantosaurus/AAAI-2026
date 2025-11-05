# Seed List of Public Sources for LLM Hallucination Research

**Project:** Characterizing and Mitigating Hallucinations in Security-Related LLM Applications
**Purpose:** RAG knowledge base construction and benchmark validation
**Last Updated:** November 5, 2025

---

## Overview

This document provides a comprehensive list of authoritative public cybersecurity data sources for constructing retrieval-augmented generation (RAG) knowledge bases and validating LLM outputs. All sources listed are publicly accessible and authorized for academic research use.

---

## 1. NIST National Vulnerability Database (NVD)

### 1.1 Primary Resources

**Main Portal:**
- URL: https://nvd.nist.gov/
- Description: Comprehensive vulnerability database maintained by NIST
- Update Frequency: Continuous (hourly for new CVEs)
- License: Public domain (U.S. Government work)

**API Access:**
- API Documentation: https://nvd.nist.gov/developers/vulnerabilities
- API Endpoint: https://services.nvd.nist.gov/rest/json/cves/2.0
- Rate Limit: 5 requests per 30 seconds (without API key), 50 requests per 30 seconds (with API key)
- API Key Registration: https://nvd.nist.gov/developers/request-an-api-key
- Response Format: JSON

**Data Feeds (Deprecated but still useful for historical data):**
- JSON Data Feeds: https://nvd.nist.gov/vuln/data-feeds
- Note: Transitioning to API-only after December 2023

### 1.2 Metadata to Collect

For each CVE entry, extract:
```yaml
Required Fields:
  - cve_id: String (e.g., "CVE-2024-1234")
  - description: String (English description)
  - published_date: ISO 8601 timestamp
  - last_modified_date: ISO 8601 timestamp
  - cvss_v3_score: Float (0.0-10.0)
  - cvss_v3_vector: String (e.g., "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
  - cvss_v3_severity: Enum [LOW, MEDIUM, HIGH, CRITICAL]
  - cwe_ids: List of Strings (e.g., ["CWE-79", "CWE-89"])
  - cpe_affected: List of CPE URIs (affected products)
  - references: List of URLs (advisories, patches, exploits)
  - vendor_name: String (if available)
  - product_name: String (if available)

Optional Fields:
  - cvss_v2_score: Float (legacy, if available)
  - exploit_available: Boolean (derived from references)
  - patch_available: Boolean (derived from references)
  - configurations: CPE configuration details
```

### 1.3 Collection Strategy

**Initial Snapshot:**
```bash
# Collect all CVEs from 2015-present (approx. 120,000 CVEs)
# Estimated time: 8-12 hours with rate limiting

for year in {2015..2025}; do
  curl "https://services.nvd.nist.gov/rest/json/cves/2.0?pubStartDate=${year}-01-01T00:00:00.000&pubEndDate=${year}-12-31T23:59:59.999" \
    -H "apiKey: YOUR_API_KEY" \
    -o "nvd_cves_${year}.json"
  sleep 1  # Rate limit compliance
done
```

**Incremental Updates:**
- Run daily to collect CVEs modified in last 24 hours
- Use `lastModStartDate` and `lastModEndDate` parameters
- Store with timestamp for version control

**Data Validation:**
- Check for missing CVSS scores (some CVEs are still being analyzed)
- Validate CVE ID format: `CVE-YYYY-NNNNN` or `CVE-YYYY-NNNNNN`
- Flag rejected/disputed CVEs (check `vulnStatus` field)

---

## 2. MITRE ATT&CK Framework

### 2.1 Primary Resources

**Main Portal:**
- URL: https://attack.mitre.org/
- Description: Knowledge base of adversary tactics and techniques
- Update Frequency: Quarterly major releases, monthly minor updates
- License: Creative Commons Attribution 3.0 (CC BY 3.0)

**TAXII/STIX Data:**
- ATT&CK STIX Data: https://github.com/mitre-attack/attack-stix-data
- Collection URL: https://cti-taxii.mitre.org/stix/collections/
- Format: STIX 2.0/2.1 JSON
- No rate limiting (direct GitHub download)

**Latest Release:**
- Enterprise ATT&CK v14: https://github.com/mitre-attack/attack-stix-data/tree/master/enterprise-attack
- Mobile ATT&CK: https://github.com/mitre-attack/attack-stix-data/tree/master/mobile-attack
- ICS ATT&CK: https://github.com/mitre-attack/attack-stix-data/tree/master/ics-attack

### 2.2 Metadata to Collect

**For Techniques:**
```yaml
Required Fields:
  - technique_id: String (e.g., "T1059.001")
  - technique_name: String (e.g., "PowerShell")
  - parent_technique: String (if sub-technique)
  - tactic: List of Strings (e.g., ["Execution", "Persistence"])
  - description: String (detailed description)
  - detection: String (how to detect)
  - mitigation_ids: List of Strings (e.g., ["M1038", "M1049"])
  - platforms: List of Strings (e.g., ["Windows", "Linux", "macOS"])
  - data_sources: List of Strings (e.g., ["Process: Process Creation"])
  - permissions_required: List of Strings (e.g., ["User", "Administrator"])
  - is_subtechnique: Boolean

Optional Fields:
  - procedure_examples: List of {actor, description} (real-world usage)
  - references: List of URLs
  - attack_version: String (e.g., "14.0")
  - deprecated: Boolean
  - revoked: Boolean
```

**For Mitigations:**
```yaml
Required Fields:
  - mitigation_id: String (e.g., "M1038")
  - mitigation_name: String (e.g., "Execution Prevention")
  - description: String
  - techniques_addressed: List of Strings (technique IDs)
```

**For Groups (APTs):**
```yaml
Required Fields:
  - group_id: String (e.g., "G0016")
  - group_name: String (e.g., "APT29")
  - aliases: List of Strings
  - description: String
  - techniques_used: List of Strings (technique IDs)
  - software_used: List of Strings (software IDs)
```

### 2.3 Collection Strategy

**Initial Download:**
```bash
# Clone the STIX data repository
git clone https://github.com/mitre-attack/attack-stix-data.git
cd attack-stix-data

# Parse enterprise-attack/enterprise-attack.json
python3 << EOF
import json

with open('enterprise-attack/enterprise-attack.json', 'r') as f:
    attack_data = json.load(f)

# Extract techniques, mitigations, groups
techniques = [obj for obj in attack_data['objects'] if obj['type'] == 'attack-pattern']
mitigations = [obj for obj in attack_data['objects'] if obj['type'] == 'course-of-action']
groups = [obj for obj in attack_data['objects'] if obj['type'] == 'intrusion-set']

print(f"Techniques: {len(techniques)}")
print(f"Mitigations: {len(mitigations)}")
print(f"Groups: {len(groups)}")
EOF
```

**Quarterly Updates:**
- Monitor MITRE's release notes: https://attack.mitre.org/resources/updates/
- Re-download entire dataset (file sizes: ~20-30 MB per matrix)
- Compare versions to identify new techniques

**Data Validation:**
- Ensure all technique IDs match pattern: `T[0-9]{4}(\.[0-9]{3})?`
- Verify tactic names match official list (14 tactics in Enterprise ATT&CK)
- Check for revoked/deprecated techniques (filter or flag)

---

## 3. Common Vulnerabilities and Exposures (CVE) - MITRE

### 3.1 Primary Resources

**CVE List:**
- URL: https://cve.mitre.org/
- Description: CVE ID allocation and assignment (authoritative source)
- Note: For detailed vulnerability data, use NVD (above)

**CVE Services:**
- GitHub Repository: https://github.com/CVEProject/cvelistV5
- Format: JSON 5.0
- Contains: CVE records in JSON format before NVD enrichment

**Bulk Download:**
- URL: https://github.com/CVEProject/cvelistV5/tree/main/cves
- Organization: Directory structure by year and CVE number ranges
- Update: Continuous via GitHub commits

### 3.2 Metadata to Collect

```yaml
Required Fields:
  - cve_id: String
  - state: Enum [PUBLISHED, REJECTED, RESERVED]
  - assigning_cna: String (CVE Numbering Authority)
  - date_reserved: ISO 8601 timestamp
  - date_published: ISO 8601 timestamp
  - descriptions: List of {lang, value}
  - affected_products: List of {vendor, product, versions}
  - references: List of URLs
  - credits: List of {name, type} (who discovered it)

Use Case:
  - Cross-reference with NVD for enriched data
  - Identify newly reserved CVEs not yet in NVD
  - Track CVE state transitions (RESERVED → PUBLISHED)
```

### 3.3 Collection Strategy

**Use NVD as primary source; use this for:**
- Monitoring newly reserved CVEs (potential future vulnerabilities)
- Identifying rejected CVEs (hallucination validation dataset)
- Cross-referencing CNA assignments

---

## 4. Common Weakness Enumeration (CWE) - MITRE

### 4.1 Primary Resources

**Main Portal:**
- URL: https://cwe.mitre.org/
- Description: Categorization of software weaknesses
- Current Version: CWE v4.13 (check for latest)
- License: Free to use for any purpose

**XML/CSV Downloads:**
- Download Page: https://cwe.mitre.org/data/downloads.html
- XML Data: https://cwe.mitre.org/data/xml/cwec_latest.xml.zip
- CSV Research View: https://cwe.mitre.org/data/csv/699.csv.zip (Top 25)
- Update Frequency: 2-3 times per year

### 4.2 Metadata to Collect

```yaml
Required Fields:
  - cwe_id: String (e.g., "CWE-79")
  - cwe_name: String (e.g., "Cross-site Scripting (XSS)")
  - description: String
  - extended_description: String (if available)
  - likelihood_of_exploit: Enum [High, Medium, Low]
  - weakness_type: Enum [Class, Base, Variant, Compound]
  - abstraction: Enum [Pillar, Class, Base, Variant]
  - related_attack_patterns: List of CAPEC IDs
  - common_consequences: List of {scope, impact, likelihood}
  - detection_methods: List of {method, effectiveness}
  - mitigation_strategies: List of {phase, description}
  - observed_examples: List of {reference, description}

Optional Fields:
  - parent_cwe: String (hierarchical relationship)
  - child_cwes: List of Strings
  - applicable_platforms: List of {language, class, prevalence}
```

### 4.3 Collection Strategy

```bash
# Download and parse CWE XML
wget https://cwe.mitre.org/data/xml/cwec_latest.xml.zip
unzip cwec_latest.xml.zip

# Parse with Python
import xml.etree.ElementTree as ET
tree = ET.parse('cwec_v4.13.xml')
root = tree.getroot()

weaknesses = root.findall('.//{http://cwe.mitre.org/cwe-7}Weakness')
print(f"Total CWEs: {len(weaknesses)}")
```

**Use Cases:**
- Map CVEs to CWEs for categorization
- Validate LLM-generated CWE assignments
- Build weakness taxonomy for prompt construction

---

## 5. Vendor Security Advisories

### 5.1 Microsoft Security Response Center (MSRC)

**Main Portal:**
- URL: https://msrc.microsoft.com/
- Security Updates: https://msrc.microsoft.com/update-guide/
- RSS Feed: https://api.msrc.microsoft.com/update-guide/rss

**API Access:**
- MSRC CVRF API: https://api.msrc.microsoft.com/cvrf/v2.0/
- Documentation: https://api.portal.msrc.microsoft.com/
- Example: `https://api.msrc.microsoft.com/cvrf/v2.0/cvrf/2024-Jan`
- Rate Limit: Not publicly documented; monitor HTTP 429 responses
- Format: JSON or XML (CVRF - Common Vulnerability Reporting Framework)

**Metadata to Collect:**
```yaml
Required Fields:
  - bulletin_id: String (e.g., "MS24-001" or CVE ID)
  - title: String
  - published_date: ISO 8601 timestamp
  - severity: Enum [Critical, Important, Moderate, Low]
  - impact: String (e.g., "Remote Code Execution")
  - affected_products: List of product build numbers
  - cve_ids: List of CVE IDs
  - kb_articles: List of KB numbers (patches)
  - workarounds: String (if available)
  - mitigations: String
  - faq: String (if available)
  - acknowledgments: List of researchers

Collection Strategy:
  - Monthly download of CVRF documents
  - Parse for CVE cross-references
  - Extract mitigation guidance for RAG knowledge base
```

### 5.2 Apple Security Updates

**Main Portal:**
- URL: https://support.apple.com/en-us/HT201222
- Format: Web page (no official API)
- Update Frequency: Irregular (with iOS/macOS releases)

**RSS Feed:**
- URL: https://developer.apple.com/news/rss/news.rss
- Note: Includes security updates but mixed with other news

**Metadata to Collect:**
```yaml
Required Fields:
  - advisory_id: String (e.g., "HT213841")
  - product: String (e.g., "iOS 17.0", "macOS Sonoma 14.0")
  - published_date: ISO 8601 timestamp
  - cve_ids: List of CVE IDs
  - description: String (per CVE)
  - impact: String (per CVE, e.g., "arbitrary code execution")
  - entry_point: String (if available, e.g., "maliciously crafted web content")
  - credit: List of researchers

Collection Strategy:
  - Web scraping with rate limiting (1 request per 5 seconds)
  - Use Beautiful Soup or Scrapy
  - Store HTML snapshots for reproducibility
  - Manual verification recommended (no API guarantees stability)
```

**Scraping Example:**
```python
import requests
from bs4 import BeautifulSoup
import time

url = "https://support.apple.com/en-us/HT201222"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Identify advisory links (structure may change)
advisories = soup.find_all('a', href=lambda x: x and 'HT' in x)
time.sleep(5)  # Rate limiting
```

### 5.3 Cisco Security Advisories

**Main Portal:**
- URL: https://sec.cloudapps.cisco.com/security/center/publicationListing.x
- Security Advisories: https://tools.cisco.com/security/center/publicationListing.x

**Cisco PSIRT openVuln API:**
- URL: https://developer.cisco.com/docs/psirt/
- API Endpoint: https://api.cisco.com/security/advisories/
- Authentication: OAuth 2.0 (requires Cisco API Console registration)
- Rate Limit: 5 requests per second
- Format: JSON

**API Registration:**
- Register at: https://apiconsole.cisco.com/
- Request application credentials (Client ID + Client Secret)

**Metadata to Collect:**
```yaml
Required Fields:
  - advisory_id: String (e.g., "cisco-sa-iosxe-webui-privesc-j22SaA4z")
  - advisory_title: String
  - publication_date: ISO 8601 timestamp
  - last_updated: ISO 8601 timestamp
  - cves: List of {cve_id, cvss_base_score}
  - sir: String (Security Impact Rating: Critical, High, Medium, Low)
  - first_published: ISO 8601 timestamp
  - summary: String
  - affected_products: List of product names
  - vulnerable_products: List of {product, version}
  - workarounds: String
  - fixed_software: List of {product, version}
  - exploit_public: Boolean
  - exploit_available: Boolean

Collection Strategy:
  - Quarterly bulk download via API
  - Filter by advisory_year for incremental updates
  - Cross-reference CVEs with NVD
```

**API Example:**
```bash
# Get OAuth token
TOKEN=$(curl -X POST "https://id.cisco.com/oauth2/default/v1/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET&grant_type=client_credentials" \
  | jq -r '.access_token')

# Fetch advisories by year
curl "https://api.cisco.com/security/advisories/year/2024" \
  -H "Authorization: Bearer $TOKEN"
```

### 5.4 Red Hat Security Advisories

**Main Portal:**
- URL: https://access.redhat.com/security/security-updates/
- CVE Database: https://access.redhat.com/security/cve/

**Red Hat Security Data API:**
- URL: https://access.redhat.com/documentation/en-us/red_hat_security_data_api/
- Endpoint: https://access.redhat.com/labs/securitydataapi/
- No authentication required for public data
- Rate Limit: Not strictly enforced; be respectful (1-2 req/sec)
- Format: JSON

**Metadata to Collect:**
```yaml
Required Fields:
  - rhsa_id: String (e.g., "RHSA-2024:0001")
  - cve_ids: List of CVE IDs
  - severity: Enum [Critical, Important, Moderate, Low]
  - issued_date: ISO 8601 timestamp
  - affected_packages: List of {name, version}
  - fixed_packages: List of {name, version}
  - rhel_versions: List of affected RHEL versions
  - bugzilla_ids: List of bug tracker IDs
  - description: String

API Examples:
  - All CVEs: https://access.redhat.com/labs/securitydataapi/cve.json
  - Specific CVE: https://access.redhat.com/labs/securitydataapi/cve/CVE-2024-1234.json
  - Advisories by year: https://access.redhat.com/labs/securitydataapi/cvrf/year/2024.json
```

### 5.5 Ubuntu Security Notices (USN)

**Main Portal:**
- URL: https://ubuntu.com/security/notices
- CVE Tracker: https://ubuntu.com/security/cves

**Ubuntu CVE Tracker (Git Repository):**
- URL: https://git.launchpad.net/ubuntu-cve-tracker
- Clone: `git clone https://git.launchpad.net/ubuntu-cve-tracker`
- Format: Custom text format (parseable)
- Update: Daily git pulls

**Metadata to Collect:**
```yaml
Required Fields:
  - usn_id: String (e.g., "USN-6123-1")
  - cve_ids: List of CVE IDs
  - published_date: ISO 8601 timestamp
  - title: String
  - packages: List of {name, version, release}
  - ubuntu_releases: List of {codename, version} (e.g., ["focal/20.04 LTS"])
  - priority: Enum [Critical, High, Medium, Low, Negligible]
  - references: List of URLs

Collection Strategy:
  - Clone Git repo and parse active/ directory
  - Extract CVE files (format: CVE-YYYY-NNNNN)
  - Parse Ubuntu-specific fields (Priority, Patches, Notes)
```

---

## 6. Exploit Databases (Use with Caution)

### 6.1 Exploit Database (Exploit-DB)

**Main Portal:**
- URL: https://www.exploit-db.com/
- Maintained by: Offensive Security

**GitHub Mirror:**
- URL: https://github.com/offensive-security/exploitdb
- CSV Database: https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv
- Update Frequency: Daily

**Metadata to Collect (METADATA ONLY - NOT EXPLOIT CODE):**
```yaml
Required Fields:
  - edb_id: String (Exploit-DB ID)
  - cve_ids: List of CVE IDs (if applicable)
  - exploit_title: String
  - exploit_date: ISO 8601 timestamp
  - author: String
  - platform: String (e.g., "linux", "windows", "web")
  - exploit_type: String (e.g., "remote", "local", "webapps")
  - port: Integer (if network exploit)

DO NOT COLLECT:
  - Full exploit code (only references)
  - Working shellcode
  - Malware binaries

Collection Strategy:
  - Download CSV metadata only
  - Use for cross-referencing CVEs with public exploits
  - Flag CVEs with public exploits for hallucination testing
```

### 6.2 VulnHub and HackerOne Disclosed Reports

**VulnHub (for CTF-style vulnerability testing):**
- URL: https://www.vulnhub.com/
- Use: Controlled environments for testing mitigation strategies
- Note: Not a data source, but useful for integration phase testing

**HackerOne Disclosed Reports (Bug Bounty):**
- URL: https://hackerone.com/hacktivity
- Format: Web interface (no official API for public reports)
- Use: Real-world vulnerability disclosure patterns
- Collection: Manual review only; respect disclosure policies

---

## 7. Threat Intelligence Feeds (Public)

### 7.1 AlienVault Open Threat Exchange (OTX)

**Main Portal:**
- URL: https://otx.alienvault.com/
- API Documentation: https://otx.alienvault.com/api
- Authentication: Requires free account + API key
- Rate Limit: 10 requests per second
- Format: JSON

**Metadata to Collect:**
```yaml
Pulses (threat reports):
  - pulse_id: String
  - pulse_name: String
  - description: String
  - author: String
  - created: ISO 8601 timestamp
  - indicators: List of {type, value} (IPs, domains, hashes)
  - tags: List of Strings
  - attack_ids: List of MITRE ATT&CK technique IDs
  - industries: List of targeted industries
  - malware_families: List of Strings

Use Cases:
  - Real-world IOCs for testing malware triage workflows
  - ATT&CK technique mappings for validation
  - Threat actor TTPs for prompt construction
```

### 7.2 MISP Threat Sharing (CIRCL Public Feeds)

**Main Portal:**
- URL: https://www.circl.lu/doc/misp/
- Public MISP Instance: https://www.misp-project.org/feeds/

**MISP Format:**
- Format: JSON (MISP Event format)
- Events contain: Attributes (IOCs), Objects (structured data), Tags, Galaxies (threat intel)

**Public Feeds:**
- CIRCL OSINT Feed: https://www.circl.lu/doc/misp/feed-osint/
- Note: Requires MISP instance to consume feeds efficiently

**Metadata to Collect:**
```yaml
Events:
  - event_id: String
  - event_info: String (title/description)
  - threat_level: Enum [High, Medium, Low, Undefined]
  - analysis: Enum [Initial, Ongoing, Completed]
  - timestamp: Unix timestamp
  - published_timestamp: Unix timestamp
  - attributes: List of {category, type, value}
  - tags: List of {name, colour}
  - galaxies: List of {type, name, description}

Use Cases:
  - Structured threat intel for RAG
  - Galaxy clusters for threat actor/malware knowledge
```

---

## 8. Security Configuration Benchmarks

### 8.1 CIS Benchmarks (Center for Internet Security)

**Main Portal:**
- URL: https://www.cisecurity.org/cis-benchmarks/
- Access: Free registration required
- Format: PDF documents (manual download)

**Available Benchmarks:**
- Operating Systems: Windows Server, Linux distributions, macOS
- Cloud Platforms: AWS, Azure, GCP, Kubernetes
- Applications: Apache, Nginx, Docker, databases
- Mobile: iOS, Android

**Metadata to Collect (from PDFs):**
```yaml
Recommendations:
  - benchmark_id: String (e.g., "CIS_Win_Server_2022_v1.0")
  - recommendation_number: String (e.g., "1.1.1")
  - title: String
  - profile_applicability: List of Strings (Level 1, Level 2)
  - description: String
  - rationale: String
  - impact: String
  - remediation: String (configuration steps)
  - default_value: String
  - references: List of URLs

Collection Strategy:
  - Manual download (no bulk API)
  - PDF parsing with pdfplumber or PyPDF2
  - Store as structured JSON for RAG
  - Update quarterly (CIS releases schedule)
```

### 8.2 NIST Special Publications (SP 800 Series)

**Main Portal:**
- URL: https://csrc.nist.gov/publications/sp800
- Format: PDF documents (free download)
- License: Public domain

**Key Publications for Security Configuration:**
- SP 800-53: Security and Privacy Controls
- SP 800-123: Guide to General Server Security
- SP 800-171: Protecting Controlled Unclassified Information
- SP 800-207: Zero Trust Architecture

**Metadata to Collect:**
```yaml
Controls (from SP 800-53):
  - control_id: String (e.g., "AC-2")
  - control_name: String (e.g., "Account Management")
  - control_family: String (e.g., "Access Control")
  - control_text: String (full description)
  - supplemental_guidance: String
  - control_enhancements: List of Strings
  - related_controls: List of control IDs
  - references: List of documents

Use Cases:
  - Validate LLM-generated security recommendations
  - Ground secure configuration prompts in authoritative guidance
```

---

## 9. Malware Intelligence Repositories

### 9.1 MalwareBazaar (Abuse.ch)

**Main Portal:**
- URL: https://bazaar.abuse.ch/
- API Documentation: https://bazaar.abuse.ch/api/
- Rate Limit: Not strictly defined; use responsibly
- Format: JSON

**API Endpoints:**
- Recent samples: https://mb-api.abuse.ch/api/v1/ (POST with query)
- Query by hash, tag, signature

**Metadata to Collect (HASHES ONLY - NOT BINARIES):**
```yaml
Samples:
  - sha256_hash: String
  - md5_hash: String
  - first_seen: ISO 8601 timestamp
  - file_type: String (e.g., "exe", "dll", "apk")
  - file_size: Integer (bytes)
  - signature: String (malware family, e.g., "Emotet")
  - tags: List of Strings
  - delivery_method: String
  - intelligence: List of {key, value}

DO NOT DOWNLOAD:
  - Actual malware binaries
  - Unpacked payloads
  - Execution artifacts

Use Cases:
  - Malware family names for prompt construction
  - IOCs (hashes) for triage workflow testing
  - Delivery method patterns for analysis
```

### 9.2 VirusTotal (Public API)

**Main Portal:**
- URL: https://www.virustotal.com/
- API Documentation: https://developers.virustotal.com/reference/overview
- Rate Limit (Free): 4 requests per minute
- Authentication: API key (free registration)

**Metadata to Collect:**
```yaml
File Reports (by hash):
  - sha256: String
  - malicious_votes: Integer
  - harmless_votes: Integer
  - last_analysis_stats: {malicious, suspicious, undetected, harmless}
  - last_analysis_results: Map of {vendor: {category, result}}
  - tags: List of Strings
  - names: List of Strings (file names seen in wild)
  - signature_info: {subject, issuer} (if signed)

Use Cases:
  - Validate LLM claims about malware detection rates
  - Cross-reference malware families
  - DO NOT use for malware distribution
```

---

## 10. Security Mailing Lists and Advisories

### 10.1 Full Disclosure Mailing List

**Archive:**
- URL: https://seclists.org/fulldisclosure/
- Format: Email archives (web-scrapable)
- Update: Real-time (as disclosures are posted)

**Collection Strategy:**
- Web scraping with rate limiting
- Parse email threads for CVE references
- Extract vulnerability descriptions (text only)
- Use for hallucination pattern analysis (some posts contain errors/speculation)

### 10.2 OSS Security Mailing List (OSS-Sec)

**Archive:**
- URL: https://seclists.org/oss-sec/
- Format: Email archives
- Focus: Open-source software vulnerabilities

**Use Cases:**
- Early CVE disclosures before NVD enrichment
- Context around CVE assignments
- Dispute/rejection discussions

---

## 11. Collection Automation Framework

### 11.1 Recommended Tech Stack

```yaml
Orchestration:
  - Apache Airflow (DAG-based scheduling)
  - Prefect (modern alternative)

Data Storage:
  - PostgreSQL (relational data: CVEs, advisories)
  - Elasticsearch (full-text search for RAG)
  - MinIO/S3 (object storage for raw JSON/XML)

ETL Pipeline:
  - Python 3.10+ with libraries:
    - requests (HTTP client)
    - beautifulsoup4 (web scraping)
    - pandas (data manipulation)
    - pydantic (data validation)
    - sqlalchemy (ORM)
  - Rate limiting: requests_ratelimiter

Version Control:
  - DVC (Data Version Control) for dataset versioning
  - Git for code and metadata schemas
```

### 11.2 Sample Collection DAG (Airflow)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def collect_nvd_cves(**context):
    # NVD API collection logic
    pass

def collect_mitre_attack(**context):
    # MITRE ATT&CK STIX parsing
    pass

def collect_msrc_advisories(**context):
    # MSRC CVRF collection
    pass

def sanitize_datasets(**context):
    # Run sanitization pipeline (see README_SAFETY.md)
    pass

def index_to_elasticsearch(**context):
    # Index for RAG retrieval
    pass

default_args = {
    'owner': 'research_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 5),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'cybersecurity_data_collection',
    default_args=default_args,
    description='Collect cybersecurity data for LLM research',
    schedule_interval='@daily',
    catchup=False,
)

t1 = PythonOperator(task_id='collect_nvd', python_callable=collect_nvd_cves, dag=dag)
t2 = PythonOperator(task_id='collect_mitre', python_callable=collect_mitre_attack, dag=dag)
t3 = PythonOperator(task_id='collect_msrc', python_callable=collect_msrc_advisories, dag=dag)
t4 = PythonOperator(task_id='sanitize', python_callable=sanitize_datasets, dag=dag)
t5 = PythonOperator(task_id='index_data', python_callable=index_to_elasticsearch, dag=dag)

[t1, t2, t3] >> t4 >> t5
```

---

## 12. Quality Assurance and Validation

### 12.1 Data Quality Checks

**Run these checks after each collection cycle:**
```python
import pandas as pd

def validate_cve_dataset(df):
    """Validate CVE dataset quality."""
    checks = {
        'cve_id_format': df['cve_id'].str.match(r'CVE-\d{4}-\d{4,7}').all(),
        'no_nulls_required': df[['cve_id', 'description', 'published_date']].notna().all().all(),
        'cvss_range': df['cvss_v3_score'].between(0, 10, inclusive='both').all(),
        'date_valid': pd.to_datetime(df['published_date'], errors='coerce').notna().all(),
        'no_duplicates': ~df['cve_id'].duplicated().any(),
    }

    passed = sum(checks.values())
    total = len(checks)
    print(f"Passed {passed}/{total} quality checks")

    for check, result in checks.items():
        if not result:
            print(f"FAILED: {check}")

    return all(checks.values())
```

### 12.2 Deduplication Strategy

- Use CVE ID as primary key for vulnerabilities
- Use technique ID for MITRE ATT&CK
- Use SHA-256 hash for malware samples
- For advisories: Vendor ID + Advisory ID composite key

### 12.3 Update Frequency Recommendations

```yaml
Daily:
  - NVD CVE incremental updates
  - Vendor advisory RSS feeds
  - Threat intelligence feeds (OTX, MISP)

Weekly:
  - Exploit-DB metadata
  - Security mailing lists

Monthly:
  - CIS Benchmarks (check for new versions)
  - NIST SP 800 series (check for updates)

Quarterly:
  - MITRE ATT&CK (aligned with their release schedule)
  - Full dataset audit and sanitization review
```

---

## 13. Legal and Ethical Compliance

### 13.1 Terms of Service Review

Before collecting data, review and document acceptance of:
- [ ] NVD Terms of Use
- [ ] MITRE ATT&CK Licensing (CC BY 3.0)
- [ ] Vendor API Terms (Microsoft, Cisco, etc.)
- [ ] Rate limiting policies
- [ ] Attribution requirements

### 13.2 Data Usage Restrictions

**This data is authorized ONLY for:**
- Academic research purposes
- Defensive security applications
- Hallucination characterization and mitigation

**This data is NOT authorized for:**
- Developing novel exploits
- Attacking systems without authorization
- Commercial security products without proper licensing
- Reselling or redistributing (check individual licenses)

---

## 14. Maintenance and Contact Information

**Data Collection Pipeline Owner:** [Research Team Lead]
**Last Audit Date:** November 5, 2025
**Next Scheduled Review:** December 5, 2025

**External Resource Contacts (if issues arise):**
- NVD Support: nvd@nist.gov
- MITRE ATT&CK: attack@mitre.org
- MSRC: secure@microsoft.com
- Cisco PSIRT: psirt@cisco.com

**Internal Contacts:**
- Data Engineering: [CONTACT]
- Ethics/Compliance: [CONTACT]
- Research PI: [CONTACT]

---

## Appendix: Quick Reference URLs

```markdown
# Core Sources
- NVD API: https://services.nvd.nist.gov/rest/json/cves/2.0
- MITRE ATT&CK: https://github.com/mitre-attack/attack-stix-data
- CWE Data: https://cwe.mitre.org/data/xml/cwec_latest.xml.zip

# Vendor Advisories
- Microsoft MSRC: https://api.msrc.microsoft.com/cvrf/v2.0/
- Apple Security: https://support.apple.com/en-us/HT201222
- Cisco PSIRT: https://api.cisco.com/security/advisories/
- Red Hat Security: https://access.redhat.com/labs/securitydataapi/

# Threat Intelligence
- AlienVault OTX: https://otx.alienvault.com/api
- MalwareBazaar: https://bazaar.abuse.ch/api/

# Standards
- CIS Benchmarks: https://www.cisecurity.org/cis-benchmarks/
- NIST SP 800: https://csrc.nist.gov/publications/sp800
```

---

**End of Seed List Document**
**Version 1.0 | November 5, 2025**
