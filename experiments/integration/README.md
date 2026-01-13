# Integration Tests (Phase E)

This directory contains simulated cybersecurity workflows to evaluate hallucination impact in realistic scenarios.

## Overview

**Goal:** Test mitigations in end-to-end workflows to measure real-world impact

**Workflows:**
1. **Vulnerability Triage Pipeline** - Analyst assistant for CVE assessment
2. **Malware Triage** - Initial malware family identification
3. **Pentest Reporting** - Log analysis and remediation suggestions

All workflows are **sanitized and non-operational** - no actual exploitation or live testing.

## Files

- `vuln_triage_workflow.py` - Vulnerability triage simulation
- `malware_triage_workflow.py` - Malware analysis simulation
- `pentest_reporting_workflow.py` - Pentest log analysis simulation
- `evaluate_workflows.py` - End-to-end evaluation
- `workflow_scenarios.json` - Test scenarios for each workflow

## Safety Note

**CRITICAL:** All workflows use sanitized, non-operational scenarios:
- No actual exploit code generation
- No execution of commands
- No access to live systems
- Defensive guidance only

## Setup

```bash
pip install anthropic google-generativeai transformers
```

## Workflows

### 1. Vulnerability Triage Pipeline

**Scenario:** Security analyst receives vendor advisory and needs to assess relevance

**Workflow:**
1. Extract CVE IDs from advisory text
2. Query LLM for CVE details and severity
3. Symbolic checker verifies CVE IDs
4. RAG grounds response with NVD data
5. Analyst receives verified assessment

**Metrics:**
- Incorrect triage decisions (due to hallucinated CVEs)
- False escalations
- Missed critical vulnerabilities

```bash
python vuln_triage_workflow.py \
    --scenarios workflow_scenarios.json \
    --model claude-3-5-sonnet-20241022 \
    --use-symbolic-check \
    --output results/vuln_triage_results.json
```

### 2. Malware Triage Workflow

**Scenario:** SOC analyst needs high-level malware family identification

**Workflow:**
1. Provide sanitized malware indicators (hashes, behaviors)
2. Query LLM for malware family and characteristics
3. Check against MITRE ATT&CK database
4. Return defensive recommendations

**Metrics:**
- Fabricated malware families
- Incorrect defensive recommendations
- Missed detection opportunities

```bash
python malware_triage_workflow.py \
    --scenarios workflow_scenarios.json \
    --model claude-3-5-sonnet-20241022 \
    --output results/malware_triage_results.json
```

### 3. Pentest Reporting Workflow

**Scenario:** Convert sanitized pentest logs into findings report

**Workflow:**
1. Provide sanitized log excerpts (no exploit commands)
2. LLM summarizes finding and suggests remediation
3. Verify suggested CVEs exist
4. Output sanitized report

**Metrics:**
- Fabricated remediation steps
- Incorrect severity assessments
- Misleading recommendations

```bash
python pentest_reporting_workflow.py \
    --scenarios workflow_scenarios.json \
    --model claude-3-5-sonnet-20241022 \
    --output results/pentest_reporting_results.json
```

### End-to-End Evaluation

```bash
python evaluate_workflows.py \
    --vuln-triage results/vuln_triage_results.json \
    --malware-triage results/malware_triage_results.json \
    --pentest-reporting results/pentest_reporting_results.json \
    --output docs/integration_report.md
```

## Expected Results

**Without Mitigations:**
- 20-40% of triage decisions affected by hallucinations
- Fabricated CVE citations cause incorrect prioritization
- Misleading remediation advice

**With Symbolic Checker:**
- Nearly 0% fabricated CVE citations
- Correct escalation of novel threats
- Some residual hallucinations in descriptions

**With RAG Grounding:**
- 50-70% reduction in hallucinations
- Higher factual accuracy
- May miss very recent CVEs not in index

**With Abstention:**
- 10-30% of uncertain cases withheld
- Lower hallucination rate on answered cases
- Trade-off: some valid questions unanswered

## Integration Metrics

- **Critical Errors:** Hallucinations causing severe misclassification
- **Escalation Accuracy:** Correct identification of high-priority items
- **False Negative Rate:** Real threats missed due to abstention
- **Analyst Confidence:** Would results require additional verification?

## Scenarios File Format

```json
{
  "vuln_triage_scenarios": [
    {
      "id": "vt_001",
      "advisory_text": "Vendor X reports critical vulnerability CVE-2023-12345 in Product Y",
      "ground_truth": {
        "cve_id": "CVE-2023-12345",
        "exists": true,
        "severity": "CRITICAL"
      }
    }
  ],
  "malware_scenarios": [...],
  "pentest_scenarios": [...]
}
```
