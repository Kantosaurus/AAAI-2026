#!/usr/bin/env python3
"""
Vulnerability Triage Workflow Simulation
Simulates a security analyst using LLM assistance for CVE assessment

Usage:
    python vuln_triage_workflow.py \
        --scenarios workflow_scenarios.json \
        --model claude-3-5-sonnet-20241022 \
        --use-symbolic-check \
        --output results/vuln_triage_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import re
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'mitigations'))

try:
    from symbolic_checker import extract_cve_ids, check_cve_ids, load_nvd_cve_list
except ImportError:
    print("Warning: Could not import symbolic_checker, will run without verification")
    def extract_cve_ids(text): return []
    def check_cve_ids(ids, known): return [], ids
    def load_nvd_cve_list(f=None): return set()


@dataclass
class TriageResult:
    """Result of vulnerability triage"""
    scenario_id: str
    advisory_text: str
    extracted_cves: List[str]
    llm_response: str
    verified_cves: List[str]
    fabricated_cves: List[str]
    triage_decision: str  # CRITICAL, HIGH, MEDIUM, LOW, UNKNOWN
    ground_truth_decision: str
    is_correct: bool
    hallucination_impact: str  # NONE, LOW, MEDIUM, HIGH, CRITICAL


def query_llm_for_cve_assessment(advisory_text: str, model: str) -> str:
    """
    Query LLM for CVE assessment

    Note: This is a simplified demo. In production, would call actual API
    """
    # For demo purposes, return a template response
    # In real implementation, would call anthropic/gemini API
    prompt = f"""You are a security analyst assistant. Analyze this security advisory and provide:
1. List of CVE IDs mentioned
2. Severity assessment (CRITICAL/HIGH/MEDIUM/LOW)
3. Brief impact summary

Advisory:
{advisory_text}

Provide a structured response."""

    # Simulated response (in production, call actual LLM)
    print(f"[DEMO] Would query {model} with prompt:")
    print(f"  {prompt[:100]}...")

    return "DEMO MODE: Replace with actual LLM call"


def assess_triage_correctness(
    llm_decision: str,
    ground_truth: Dict,
    fabricated_cves: List[str]
) -> tuple[bool, str]:
    """
    Assess if triage decision is correct

    Returns:
        (is_correct, hallucination_impact)
    """
    # Check for fabricated CVEs
    if fabricated_cves:
        if len(fabricated_cves) > 0:
            return False, "CRITICAL"  # Fabricated CVE caused wrong triage

    # Check severity alignment (simplified)
    gt_severity = ground_truth.get('severity', 'UNKNOWN')

    if gt_severity in llm_decision.upper():
        return True, "NONE"

    # Misalignment
    severity_diff = abs(
        ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].index(gt_severity) -
        (['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].index(llm_decision.upper()) if llm_decision.upper() in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] else 2)
    )

    if severity_diff >= 2:
        return False, "HIGH"
    elif severity_diff == 1:
        return False, "MEDIUM"

    return True, "NONE"


def run_triage_workflow(
    scenarios: List[Dict],
    model: str,
    use_symbolic_check: bool = False,
    nvd_cves: set = None
) -> List[TriageResult]:
    """Run vulnerability triage workflow on scenarios"""
    results = []

    for scenario in scenarios:
        scenario_id = scenario['id']
        advisory_text = scenario['advisory_text']
        ground_truth = scenario['ground_truth']

        print(f"\n--- Processing scenario: {scenario_id} ---")

        # Step 1: Extract CVE IDs from advisory
        extracted_cves = extract_cve_ids(advisory_text)
        print(f"Extracted CVEs: {extracted_cves}")

        # Step 2: Query LLM for assessment
        llm_response = query_llm_for_cve_assessment(advisory_text, model)

        # Step 3: Symbolic verification (if enabled)
        fabricated_cves = []
        verified_cves = []

        if use_symbolic_check and nvd_cves:
            all_cves = extract_cve_ids(llm_response) + extracted_cves
            verified, fabricated = check_cve_ids(all_cves, nvd_cves)
            verified_cves = verified
            fabricated_cves = fabricated
            print(f"Verified: {verified_cves}, Fabricated: {fabricated_cves}")

        # Step 4: Make triage decision (simplified)
        # In production, parse LLM response for severity
        llm_decision = ground_truth.get('severity', 'UNKNOWN')  # Simulated

        # Step 5: Assess correctness
        is_correct, impact = assess_triage_correctness(
            llm_decision,
            ground_truth,
            fabricated_cves
        )

        result = TriageResult(
            scenario_id=scenario_id,
            advisory_text=advisory_text,
            extracted_cves=extracted_cves,
            llm_response=llm_response,
            verified_cves=verified_cves,
            fabricated_cves=fabricated_cves,
            triage_decision=llm_decision,
            ground_truth_decision=ground_truth.get('severity', 'UNKNOWN'),
            is_correct=is_correct,
            hallucination_impact=impact
        )

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Vulnerability triage workflow simulation")
    parser.add_argument('--scenarios', type=str, required=True, help='Workflow scenarios JSON file')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20241022', help='Model to use')
    parser.add_argument('--use-symbolic-check', action='store_true', help='Enable symbolic CVE verification')
    parser.add_argument('--nvd-list', type=str, default=None, help='NVD CVE list for verification')
    parser.add_argument('--output', type=str, required=True, help='Output results file')

    args = parser.parse_args()

    # Load scenarios
    print("Loading scenarios...")
    with open(args.scenarios, 'r', encoding='utf-8') as f:
        data = json.load(f)
        scenarios = data.get('vuln_triage_scenarios', [])

    print(f"Loaded {len(scenarios)} vulnerability triage scenarios")

    # Load NVD CVEs if using symbolic check
    nvd_cves = None
    if args.use_symbolic_check:
        print("Loading NVD CVE list...")
        nvd_file = Path(args.nvd_list) if args.nvd_list else None
        nvd_cves = load_nvd_cve_list(nvd_file)
        print(f"Loaded {len(nvd_cves)} CVEs")

    # Run workflow
    print("\nRunning vulnerability triage workflow...")
    print("="*60)
    results = run_triage_workflow(scenarios, args.model, args.use_symbolic_check, nvd_cves)

    # Compute metrics
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    with_fabricated = sum(1 for r in results if r.fabricated_cves)
    critical_impact = sum(1 for r in results if r.hallucination_impact == 'CRITICAL')

    # Save results
    output_data = {
        'model': args.model,
        'use_symbolic_check': args.use_symbolic_check,
        'total_scenarios': total,
        'metrics': {
            'correct_triage_decisions': correct,
            'accuracy': correct / total if total > 0 else 0,
            'scenarios_with_fabricated_cves': with_fabricated,
            'critical_hallucination_impact': critical_impact
        },
        'results': [
            {
                'scenario_id': r.scenario_id,
                'extracted_cves': r.extracted_cves,
                'verified_cves': r.verified_cves,
                'fabricated_cves': r.fabricated_cves,
                'triage_decision': r.triage_decision,
                'ground_truth_decision': r.ground_truth_decision,
                'is_correct': r.is_correct,
                'hallucination_impact': r.hallucination_impact,
                'llm_response': r.llm_response[:500]  # Truncate for storage
            }
            for r in results
        ]
    }

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("VULNERABILITY TRIAGE WORKFLOW RESULTS")
    print("="*60)
    print(f"\nTotal scenarios: {total}")
    print(f"Correct triage decisions: {correct} ({correct/total:.1%})")
    print(f"Scenarios with fabricated CVEs: {with_fabricated} ({with_fabricated/total:.1%})")
    print(f"Critical hallucination impact: {critical_impact} ({critical_impact/total:.1%})")

    print(f"\n✓ Results saved to {output_file}")
    print("\nNote: This is a DEMO implementation.")
    print("For production use:")
    print("  1. Implement actual LLM API calls")
    print("  2. Add proper response parsing")
    print("  3. Integrate with RAG for grounding")


if __name__ == '__main__':
    main()
