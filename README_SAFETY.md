# Safety Guidance for LLM Hallucination Research in Cybersecurity

**Project:** Characterizing and Mitigating Hallucinations in Security-Related LLM Applications
**Timeline:** November 5, 2025 - November 30, 2025
**Status:** Active Research

---

## ⚠️ Critical Safety Notice

This research involves the study of large language model (LLM) hallucinations in cybersecurity contexts. While the research is defensive in nature, it necessarily involves the generation and analysis of security-related content including vulnerability descriptions, attack techniques, and exploit methodologies.

**All researchers, collaborators, and users of this codebase MUST:**
1. Read and acknowledge this safety guidance before accessing any research materials
2. Follow all sanitization procedures outlined below
3. Report any safety violations immediately to the project lead
4. Obtain explicit authorization before testing on any systems not owned by the research team

---

## 1. Data Sanitization Requirements

### 1.1 MANDATORY Sanitization Before Data Ingestion

All data collected from external sources MUST be sanitized according to these rules:

#### Personal Identifiable Information (PII)
**REMOVE or REDACT:**
- Full names (replace with `[RESEARCHER_NAME]`, `[ANALYST_NAME]`, etc.)
- Email addresses (replace with `[EMAIL_REDACTED]`)
- Phone numbers (replace with `[PHONE_REDACTED]`)
- Physical addresses (replace with `[ADDRESS_REDACTED]`)
- Social security numbers, government IDs
- Biometric data
- IP geolocation that identifies individuals

**Example:**
```
Before: "Reported by John Doe (john.doe@company.com) on 2024-05-15"
After: "Reported by [RESEARCHER_NAME] ([EMAIL_REDACTED]) on 2024-05-15"
```

#### Network and Infrastructure Details
**REMOVE or ANONYMIZE:**
- Internal IP addresses (replace with RFC 1918 equivalents: 10.x.x.x, 192.168.x.x)
- Internal hostnames and DNS records (replace with `[INTERNAL_HOST]`)
- Network topology diagrams with real infrastructure
- VPN configurations and network ACLs with real IPs
- Database connection strings with real hosts
- API endpoints pointing to production systems

**Example:**
```
Before: "Vulnerable server at 172.16.45.23 (prod-db-01.company.internal)"
After: "Vulnerable server at 192.168.1.100 ([INTERNAL_HOST])"
```

#### Authentication and Secrets
**REMOVE COMPLETELY (no placeholders in public datasets):**
- Passwords, password hashes (even if expired)
- API keys and tokens
- SSH private keys, TLS certificates and private keys
- OAuth secrets and session tokens
- AWS access keys, GCP service account keys, Azure credentials
- JWT tokens with real signatures
- Kerberos tickets

**For internal research datasets, replace with:**
```
[CREDENTIAL_REDACTED]
[API_KEY_REDACTED]
[TOKEN_REDACTED]
```

#### Exploit Code and Malware
**SANITIZE or RESTRICT:**
- Full exploit code: Remove or obfuscate critical components (e.g., shellcode)
- Malware binaries: Store as SHA-256 hashes only, NEVER full binaries
- Reverse shells: Replace callback IPs/domains with `[ATTACKER_IP]`
- Payload delivery URLs: Replace with `[MALICIOUS_URL]` or use defanged URLs
- Working weaponized exploits: Break functionality (remove critical byte sequences)

**Example:**
```python
# Before: Working exploit
payload = b"\x90\x90\x90\xeb\x1f\x5e\x89\x76\x08\x31\xc0\x88\x46\x07"

# After: Sanitized for research
payload = "[SHELLCODE_REDACTED]  # Original: 14 bytes of x86 shellcode"
```

**Defanging URLs:**
```
Before: http://malicious-site.com/exploit.php
After: hxxp://malicious-site[.]com/exploit[.]php
```

#### Organizational Information
**REMOVE or GENERALIZE:**
- Company names (unless already public in CVEs)
- Department structures and org charts
- Employee counts and roles
- Internal project codenames
- Merger/acquisition details
- Financial data related to breaches
- Customer lists

**Replace with generic identifiers:**
```
[ORGANIZATION_A], [FINANCIAL_INSTITUTION], [HEALTHCARE_PROVIDER], etc.
```

#### Vulnerability Details
**SANITIZE CAREFULLY:**
- **Published CVEs:** Keep CVE IDs and public descriptions intact
- **Unpublished vulnerabilities:** REMOVE entirely or coordinate responsible disclosure
- **0-day exploits:** DO NOT include in datasets; coordinate with vendors
- **Patch-gap vulnerabilities:** Ensure patches are publicly available before inclusion
- **Configuration errors:** Generalize to pattern rather than specific instance

---

### 1.2 Sanitization Pipeline

All data MUST pass through this pipeline before analysis:

```
1. Automated PII Detection
   ↓ (Run regex + NER models)
2. Manual Review
   ↓ (Human verification of automated redactions)
3. Security Content Review
   ↓ (Check for weaponizable exploits)
4. Legal/Ethics Review
   ↓ (Compliance check)
5. Version Control Snapshot
   ↓ (Immutable audit trail)
6. Dataset Release → Research Use
```

**Required Tools:**
- PII detection: `presidio`, `spaCy NER`, custom regex patterns
- Secret scanning: `detect-secrets`, `truffleHog`, `git-secrets`
- Manual review: Spreadsheet tracking with dual review
- Audit logging: Git commits with GPG signatures

---

## 2. Prohibited Content

### 2.1 NEVER Include in Datasets or Publications

**Strictly Prohibited:**
1. **Active 0-day exploits** with no public patch
2. **Weaponized malware binaries** (hashes only)
3. **Credentials** for any live system (even test systems)
4. **Undisclosed vulnerabilities** without vendor coordination
5. **Social engineering playbooks** targeting real organizations
6. **Insider threat scenarios** with real organizational details
7. **Ransomware payment wallets** or negotiation tactics
8. **Supply chain compromise** techniques targeting specific vendors
9. **Critical infrastructure attack** scenarios (power grid, water, healthcare)
10. **Exploit kits** with active C2 infrastructure

### 2.2 Restricted Content (Requires Special Approval)

**Requires PI and Ethics Board Approval:**
1. Adversarial prompt injection techniques (if novel)
2. Model extraction or stealing attacks
3. Jailbreak prompts for safety-aligned models
4. Denial-of-service techniques (even research-context)
5. Automated vulnerability scanning code
6. Red team playbooks with specific TTPs
7. Privilege escalation exploit chains
8. Data exfiltration methodologies

**Approval Process:**
- Written justification for research necessity
- Mitigation plan for potential misuse
- Responsible disclosure timeline
- Legal review for CFAA compliance

### 2.3 Context-Dependent Content

**Requires Ethical Framing:**
- Penetration testing commands (must include authorization context)
- Malware analysis (must emphasize detection/defense)
- Vulnerability assessments (must focus on remediation)
- Attack path analysis (must be for hardening, not offense)

**Example of Proper Framing:**
```markdown
## Prompt: Vulnerability Assessment (Defensive Context)

"As an authorized security auditor with written permission from the system owner,
analyze the following configuration for potential vulnerabilities and provide
remediation steps prioritized by severity."

[Include authorization scope, rules of engagement]
```

---

## 3. Sanitization Procedures by Data Source

### 3.1 CVE/NVD Data
```yaml
Source: NIST National Vulnerability Database
URL: https://nvd.nist.gov/
Sanitization Required:
  - PII: Remove submitter emails if present
  - References: Validate URLs are still active
  - CWE mappings: Keep intact
  - CVSS scores: Keep intact
  - Vendor names: Keep (public information)
Prohibited:
  - Do not use CVE data to develop exploits
  - Do not combine with proprietary vulnerability databases
```

### 3.2 MITRE ATT&CK Data
```yaml
Source: MITRE ATT&CK Framework
URL: https://attack.mitre.org/
Sanitization Required:
  - Techniques: Keep technique IDs and descriptions
  - Procedure examples: Remove organizational context
  - Mitigations: Keep intact (defensive)
  - Detections: Keep intact (defensive)
Prohibited:
  - Do not use to plan real attacks
  - Do not enhance with private threat intelligence without authorization
```

### 3.3 Vendor Security Advisories
```yaml
Source: Microsoft/Apple/Cisco/etc. Security Bulletins
Sanitization Required:
  - Advisory IDs: Keep intact
  - Affected versions: Keep intact
  - Workarounds: Keep intact
  - Customer reports: REMOVE or anonymize
  - Internal ticket IDs: REMOVE
Prohibited:
  - Do not use pre-disclosure advisory drafts
  - Do not scrape vendor-embargoed content
```

### 3.4 Exploit Databases (Exploit-DB, Metasploit)
```yaml
Source: Public exploit repositories
Sanitization Required:
  - Exploit code: Sanitize shellcode and payloads
  - Author info: Keep only if public attribution
  - Target info: Generalize (e.g., "Windows Server 2019" not "10.0.0.5")
  - Proof-of-concept: Break weaponization (remove automation)
Prohibited:
  - Do not include exploits marked as "not for distribution"
  - Do not enhance exploits with novel bypasses
  - Do not create fully automated exploit chains
```

### 3.5 LLM-Generated Content
```yaml
Source: Model outputs during research
Sanitization Required:
  - Hallucinated CVEs: Flag clearly as NON-EXISTENT
  - Generated exploit code: Validate it doesn't accidentally work
  - Recommended configurations: Verify security before sharing
  - Generated credentials: REMOVE even if fake (to avoid pattern learning)
Prohibited:
  - Do not assume hallucinated content is safe
  - Do not publish working exploits generated by accident
  - Do not use generated content to train models without review
```

---

## 4. Safe Handling Procedures

### 4.1 Development Environment Setup

**Required Isolation:**
```bash
# All research conducted in isolated environment
1. Air-gapped or firewall-restricted development machine
2. Separate network from production systems
3. Virtual machines for malware/exploit analysis
4. Encrypted storage for sensitive datasets
5. Multi-factor authentication for all access
```

**Prohibited:**
- Research on personal laptops with production system access
- Shared credentials between research and other projects
- Cloud storage without encryption-at-rest
- Public GitHub repos with non-sanitized data

### 4.2 Code Repository Safety

**Before Committing to Version Control:**
```bash
# Run these checks EVERY commit
1. git secrets --scan                    # Check for secrets
2. detect-secrets scan --baseline .secrets.baseline
3. Manual review of git diff
4. Ensure .gitignore excludes:
   - *.key, *.pem, *.crt (certificates)
   - .env files with credentials
   - datasets/* (large binary files)
   - /raw_data/ (unsanitized data)
```

**Required .gitignore entries:**
```
# Credentials and secrets
.env
*.key
*.pem
*.crt
*.p12
secrets/
credentials/

# Unsanitized data
raw_data/
unsanitized/
private_datasets/

# Malware and exploits
*.exe
*.dll
*.bin
exploits/weaponized/

# Large binary files
*.pcap
*.dump
*.img
```

### 4.3 Publication and Sharing

**Before Sharing Research Materials:**
1. **Dataset review:**
   - Run sanitization pipeline again
   - Third-party review by independent researcher
   - Legal clearance from institution

2. **Code review:**
   - Remove hardcoded secrets/paths
   - Add safety warnings to README
   - Include LICENSE with liability disclaimer

3. **Paper/Presentation review:**
   - Ethics board approval for human subjects
   - Vendor notification for disclosed vulnerabilities (90-day lead time)
   - Conference/journal ethical standards compliance

**Responsible Disclosure Template:**
```
To: security@vendor.com
Subject: Security Research Disclosure - [Brief Description]

Dear Security Team,

I am conducting academic research on [topic] at [institution].
During this research, I have identified [issue type] that may affect [product].

Summary: [High-level description without exploit details]
Impact: [Severity assessment]
Affected Versions: [Version numbers]

I am committed to responsible disclosure and will not publish
details until [90 days from this date] or until a patch is available,
whichever comes first.

Please acknowledge receipt and provide a timeline for remediation.

Contact: [Institutional email, no personal email]
PGP Key: [If available]
```

---

## 5. Incident Response

### 5.1 Safety Violation Reporting

**If you discover ANY of the following, report immediately:**
- Unsanitized PII in datasets
- Working exploits in public repositories
- Unauthorized access to systems during research
- Data breach or exposure of sensitive research materials
- Accidental publication of prohibited content
- Ethical violations or misuse of research

**Reporting Process:**
1. **Immediately cease** the activity that caused the violation
2. **Document** what happened (screenshots, logs, timestamps)
3. **Notify** project PI and institutional security within 1 hour
4. **Contain** the issue (take down repos, revoke access, etc.)
5. **Remediate** according to incident response plan
6. **Learn** and update policies to prevent recurrence

**Emergency Contacts:**
- Project Lead: [CONTACT_INFO]
- Institutional IT Security: [CONTACT_INFO]
- Ethics Review Board: [CONTACT_INFO]
- Legal Counsel: [CONTACT_INFO]

### 5.2 Self-Assessment Questions

**Before proceeding with any research activity, ask:**
1. ✅ Have I obtained all necessary authorizations?
2. ✅ Is this activity defensive and ethically justified?
3. ✅ Have I sanitized all data according to this guide?
4. ✅ Am I testing only on systems I own or have written permission to test?
5. ✅ Can this research be misused, and have I mitigated that risk?
6. ✅ Would I be comfortable presenting this work publicly?
7. ✅ Does this comply with institutional policies and legal requirements?

**If you answer NO to any question, STOP and seek guidance.**

---

## 6. Training and Acknowledgment

### 6.1 Required Training
All research team members MUST complete:
- [ ] Institutional IRB/ethics training
- [ ] Cybersecurity research ethics course
- [ ] CITI Program "Responsible Conduct of Research"
- [ ] Annual security awareness training
- [ ] This safety guidance document (with signed acknowledgment)

### 6.2 Acknowledgment Form

```
I, [NAME], acknowledge that I have read and understood the safety guidance
for the LLM Hallucination Research in Cybersecurity project.

I agree to:
- Follow all sanitization procedures outlined in this document
- Not include any prohibited content in research materials
- Report safety violations immediately
- Obtain authorization before testing on any systems
- Prioritize defensive security and ethical research practices

Signature: ___________________  Date: ___________________

Supervisor Approval: ___________________  Date: ___________________
```

---

## 7. References and Resources

### 7.1 Ethical Guidelines
- [ACM Code of Ethics](https://www.acm.org/code-of-ethics)
- [CERT Guide to Coordinated Vulnerability Disclosure](https://vuls.cert.org/confluence/display/CVD)
- [Menlo Report - Ethical Principles for ICT Research](https://www.dhs.gov/sites/default/files/publications/CSD-MenloPrinciplesCORE-20120803_1.pdf)

### 7.2 Legal Frameworks
- [Computer Fraud and Abuse Act (CFAA)](https://www.justice.gov/jm/jm-9-48000-computer-fraud)
- [GDPR Article 89 - Research Exemptions](https://gdpr-info.eu/art-89-gdpr/)
- [Export Administration Regulations (EAR) - Cryptography](https://www.bis.doc.gov/index.php/policy-guidance/encryption)

### 7.3 Technical Resources
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Microsoft Security Development Lifecycle](https://www.microsoft.com/en-us/securityengineering/sdl)

---

## 8. Version History

| Version | Date | Changes | Approved By |
|---------|------|---------|-------------|
| 1.0 | 2025-11-05 | Initial safety guidance document | [PI_NAME] |

---

**Remember: When in doubt, err on the side of caution. No research finding is worth compromising safety, privacy, or ethics.**

**Questions? Contact: [PROJECT_LEAD_EMAIL]**
