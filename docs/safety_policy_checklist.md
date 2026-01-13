# Safety Policy Checklist for LLM Hallucination Research in Cybersecurity Contexts

**Project:** Characterizing and Mitigating Hallucinations in Security-Related LLM Applications
**Timeline:** Ongoing research
**Last Updated:** January 13, 2026
**Version:** 2.4

---

## 1. Data Collection and Handling

### 1.1 Source Data Validation
- [ ] All vulnerability data sourced from official public databases (NVD, CVE, MITRE ATT&CK)
- [ ] Vendor advisories collected only from official channels (no third-party aggregators without verification)
- [ ] No collection of zero-day vulnerabilities or private security research without explicit authorization
- [ ] All data sources documented with URLs, access dates, and licensing information
- [ ] Version control maintained for all datasets with timestamped snapshots

### 1.2 Sensitive Information Screening
- [ ] Remove all personally identifiable information (PII) from collected data
- [ ] Remove internal IP addresses, hostnames, and network topology details
- [ ] Sanitize authentication credentials, API keys, tokens, certificates
- [ ] Remove customer-specific vulnerability data or incident reports
- [ ] Strip organizational context from security incidents
- [ ] Redact usernames, email addresses, and employee information

### 1.3 Malware and Exploit Code Handling
- [ ] Malware samples stored only as hashes or metadata (no live samples)
- [ ] Exploit code limited to publicly disclosed PoCs from official sources
- [ ] No weaponized exploits or custom attack tools included in datasets
- [ ] All code snippets reviewed for actual exploitability before inclusion
- [ ] Sandboxed execution environment required for any code validation
- [ ] Clear labels on all exploit-related content marking it as research-only

---

## 2. Benchmark Construction Safety

### 2.1 Prompt Design Ethics
- [ ] Prompts designed for defensive security purposes only
- [ ] No prompts that teach novel attack techniques without context
- [ ] All penetration testing prompts include ethical context/authorization framing
- [ ] Malware analysis prompts focus on detection/mitigation, not creation
- [ ] Vulnerability assessment prompts emphasize remediation guidance
- [ ] No social engineering or phishing scenario prompts

### 2.2 Output Validation
- [ ] All generated outputs manually reviewed before analysis
- [ ] Outputs screened for accidentally generated real vulnerabilities
- [ ] No distribution of potentially exploitable code without sanitization
- [ ] False-positive hallucinated CVEs flagged to prevent confusion
- [ ] Fabricated attack paths documented as incorrect to prevent misuse

---

## 3. Model Evaluation and Experimentation

### 3.1 Responsible Testing Boundaries
- [ ] All LLM testing conducted in isolated research environment
- [ ] No testing on production cybersecurity systems without authorization
- [ ] Red team exercises limited to authorized test environments
- [ ] Penetration testing simulations use synthetic/sandboxed targets only
- [ ] No actual vulnerability scanning of external systems
- [ ] Rate limiting and API usage within service provider terms

### 3.2 Interpretability Analysis Safety
- [ ] Attention analysis and probing limited to understanding, not exploitation
- [ ] Causal tracing results not used to develop adversarial techniques
- [ ] Model internals documentation kept confidential if under NDA
- [ ] No reverse engineering of proprietary model architectures
- [ ] Interpretability findings shared only in academic context

---

## 4. Mitigation Strategy Development

### 4.1 RAG Implementation Safety
- [ ] Knowledge bases limited to public, authorized sources only
- [ ] No ingestion of leaked, classified, or confidential security documents
- [ ] CVE database snapshots timestamped and version-controlled
- [ ] NIST/MITRE data usage compliant with terms of use
- [ ] Retrieval mechanisms do not expose internal database structure
- [ ] Query logging reviewed for potential misuse patterns

### 4.2 Hybrid Symbolic-Neural Approaches
- [ ] Rule-based components validated against security best practices
- [ ] Logic systems do not encode insecure-by-default configurations
- [ ] Symbolic reasoning outputs verified against ground truth
- [ ] No hardcoding of attack signatures that could aid adversaries

### 4.3 Uncertainty Calibration
- [ ] Abstention mechanisms tested for false negatives in critical contexts
- [ ] Confidence thresholds calibrated to err on side of caution
- [ ] Low-confidence security recommendations flagged for human review
- [ ] No automation of high-risk decisions without human-in-the-loop

---

## 5. Applied Workflow Integration

### 5.1 Tool Integration Safety
- [ ] Integration with existing tools (IDS, SIEM) in read-only mode initially
- [ ] No automatic remediation actions without explicit authorization
- [ ] Logging and audit trails for all LLM-generated recommendations
- [ ] Rollback mechanisms for any automated security configuration changes
- [ ] Human approval required for critical security decisions
- [ ] Fail-safe defaults if LLM component becomes unavailable

### 5.2 Red Team and Adversarial Testing
- [ ] Adversarial testing conducted only in controlled lab environment
- [ ] Prompt injection experiments limited to research systems
- [ ] No publication of adversarial techniques before mitigation deployment
- [ ] Coordinated disclosure process for any discovered LLM vulnerabilities
- [ ] Adversarial examples not shared publicly if actively exploitable
- [ ] Ethical review board approval for adversarial research protocols

---

## 6. Publication and Disclosure

### 6.1 Responsible Disclosure
- [ ] Vulnerabilities in third-party LLMs disclosed privately to vendors first
- [ ] 90-day disclosure window provided before public release
- [ ] Severity assessment conducted before publication
- [ ] Coordination with CERT/CC if systemic issues discovered
- [ ] No "full disclosure" of unpatched critical vulnerabilities

### 6.2 Research Artifacts Safety
- [ ] Datasets published with clear usage restrictions
- [ ] Code repositories include prominent safety warnings
- [ ] Benchmarks exclude prompts that could enable immediate attacks
- [ ] Model outputs sanitized before inclusion in papers/repos
- [ ] Reproducibility balanced with safety (not all details disclosed)

### 6.3 Dual-Use Considerations
- [ ] Research framed in defensive security context throughout
- [ ] Potential offensive applications explicitly discussed and mitigated
- [ ] Recommendations for defenders prioritized over attacker insights
- [ ] Consultation with security ethics experts before major publications
- [ ] Institutional IRB/ethics review completed before human subject studies

---

## 7. Compliance and Legal

### 7.1 Regulatory Compliance
- [ ] Computer Fraud and Abuse Act (CFAA) compliance verified
- [ ] No unauthorized access to computer systems during research
- [ ] GDPR/privacy law compliance for any EU-related data
- [ ] Export control regulations reviewed for cryptographic research
- [ ] Terms of service compliance for all LLM API usage
- [ ] Academic institutional policies followed

### 7.2 Intellectual Property
- [ ] Proper attribution for all third-party datasets and tools
- [ ] Licensing restrictions on security databases respected
- [ ] No proprietary vulnerability data used without permission
- [ ] Open-source license compliance for all integrated tools
- [ ] Patent considerations for novel mitigation techniques

---

## 8. Ongoing Monitoring and Review

### 8.1 Periodic Safety Audits
- [ ] Weekly review of research activities against this checklist
- [ ] Monthly safety meeting with research team and advisors
- [ ] Quarterly external security expert review
- [ ] Incident response plan maintained for safety violations
- [ ] Continuous monitoring of research dataset access logs

### 8.2 Adaptive Safety Measures
- [ ] Safety policy updated as research evolves
- [ ] New threat models incorporated into safety considerations
- [ ] Lessons learned from incidents documented and shared
- [ ] Feedback loop with cybersecurity community established
- [ ] Version control for this checklist with dated revisions

---

## Emergency Contacts

- **Institutional IRB/Ethics Committee:** Contact your institution
- **Cybersecurity Incident Response:** Contact your institution's security team
- **Research Advisor/Principal Investigator:** Contact your supervisor
- **Legal/Compliance Office:** Contact your institution's legal department
- **CERT Coordination Center:** cert@cert.org

---

## Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Principal Investigator | | | |
| Research Team Lead | | | |
| Ethics Review Board | | | |
| Legal/Compliance Officer | | | |

---

**Note:** This checklist must be reviewed and signed before commencing each major phase of the research project. Any deviations from this policy require documented justification and approval from the Principal Investigator and Ethics Review Board.
