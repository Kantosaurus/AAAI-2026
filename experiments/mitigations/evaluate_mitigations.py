#!/usr/bin/env python3
"""
Comparative Evaluation of Mitigation Strategies
Compares baseline vs RAG vs symbolic checking vs abstention

Usage:
    python evaluate_mitigations.py \
        --baseline results/pilot/pilot_*.json \
        --symbolic results/symbolic_check_results.json \
        --abstention results/abstention_results.json \
        --annotations annotations/adjudication/final_annotations.csv \
        --output results/mitigation_comparison.json
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class MitigationMetrics:
    """Metrics for a mitigation strategy"""
    name: str
    total_responses: int
    hallucinations_detected: int
    hallucinations_prevented: int
    false_positives: int  # Correct responses marked as problems
    abstention_count: int
    precision: float  # Of remaining responses, what % are correct
    recall: float  # Of correct responses, what % were preserved
    hallucination_reduction: float  # % reduction in hallucinations
    utility_loss: float  # % of correct responses withheld


def load_annotations(csv_file: Path) -> Dict[str, bool]:
    """
    Load ground truth annotations

    Returns:
        Dict mapping (prompt_id, model) -> is_hallucination
    """
    annotations = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['prompt_id'], row['model'])
            annotations[key] = int(row.get('hallucination_binary', 0)) == 1
    return annotations


def evaluate_baseline(results: List[Dict], annotations: Dict[str, bool]) -> MitigationMetrics:
    """Evaluate baseline (no mitigation)"""
    total = len(results)
    hallucinations = 0
    correct = 0

    for result in results:
        key = (result['prompt_id'], result['model'])
        if key in annotations:
            if annotations[key]:
                hallucinations += 1
            else:
                correct += 1

    return MitigationMetrics(
        name="Baseline (No Mitigation)",
        total_responses=total,
        hallucinations_detected=0,
        hallucinations_prevented=0,
        false_positives=0,
        abstention_count=0,
        precision=correct / total if total > 0 else 0,
        recall=1.0,  # All correct responses preserved
        hallucination_reduction=0.0,
        utility_loss=0.0
    )


def evaluate_symbolic_checker(
    symbolic_results: Dict,
    annotations: Dict[str, bool]
) -> MitigationMetrics:
    """Evaluate symbolic checker mitigation"""
    results = symbolic_results['results']
    stats = symbolic_results['statistics']

    total = len(results)
    hallucinations_prevented = 0
    false_positives = 0
    correct_preserved = 0

    for result in results:
        key = (result['prompt_id'], result['model'])
        is_hallucination = annotations.get(key, False)
        has_fabrication = result['has_fabrication']

        if has_fabrication:
            if is_hallucination:
                hallucinations_prevented += 1
            else:
                false_positives += 1
        elif not is_hallucination:
            correct_preserved += 1

    baseline_hallucinations = sum(1 for k, v in annotations.items() if v)

    return MitigationMetrics(
        name="Symbolic Checker",
        total_responses=total,
        hallucinations_detected=stats.get('responses_with_fabrications', 0),
        hallucinations_prevented=hallucinations_prevented,
        false_positives=false_positives,
        abstention_count=0,  # Symbolic checker doesn't abstain, just flags
        precision=(correct_preserved + hallucinations_prevented) / total if total > 0 else 0,
        recall=correct_preserved / (total - baseline_hallucinations) if total > baseline_hallucinations else 0,
        hallucination_reduction=hallucinations_prevented / baseline_hallucinations if baseline_hallucinations > 0 else 0,
        utility_loss=false_positives / (total - baseline_hallucinations) if total > baseline_hallucinations else 0
    )


def evaluate_abstention(
    abstention_results: Dict,
    annotations: Dict[str, bool]
) -> MitigationMetrics:
    """Evaluate abstention strategy"""
    decisions = abstention_results['decisions']
    stats = abstention_results['statistics']

    total = len(decisions)
    abstained = stats['abstained']
    hallucinations_prevented = 0
    correct_withheld = 0

    for decision in decisions:
        key = (decision['prompt_id'], decision['model'])
        is_hallucination = annotations.get(key, False)
        should_abstain = decision['should_abstain']

        if should_abstain:
            if is_hallucination:
                hallucinations_prevented += 1
            else:
                correct_withheld += 1

    baseline_hallucinations = sum(1 for k, v in annotations.items() if v)
    total_correct = total - baseline_hallucinations

    return MitigationMetrics(
        name="Uncertainty Abstention",
        total_responses=total,
        hallucinations_detected=abstained,
        hallucinations_prevented=hallucinations_prevented,
        false_positives=correct_withheld,
        abstention_count=abstained,
        precision=(total - abstained) / total if total > 0 else 0,
        recall=(total_correct - correct_withheld) / total_correct if total_correct > 0 else 0,
        hallucination_reduction=hallucinations_prevented / baseline_hallucinations if baseline_hallucinations > 0 else 0,
        utility_loss=correct_withheld / total_correct if total_correct > 0 else 0
    )


def print_comparison(metrics_list: List[MitigationMetrics]):
    """Print comparison table"""
    print("\n" + "="*80)
    print("MITIGATION STRATEGY COMPARISON")
    print("="*80)

    # Header
    print(f"\n{'Strategy':<25} {'Total':<8} {'H.Prev':<8} {'FP':<8} {'Abst':<8} {'Prec':<8} {'Recall':<8} {'H.Red':<8} {'U.Loss':<8}")
    print("-" * 80)

    # Rows
    for m in metrics_list:
        print(f"{m.name:<25} {m.total_responses:<8} {m.hallucinations_prevented:<8} {m.false_positives:<8} "
              f"{m.abstention_count:<8} {m.precision:<8.2f} {m.recall:<8.2f} {m.hallucination_reduction:<8.2%} {m.utility_loss:<8.2%}")

    print("\nLegend:")
    print("  Total: Total responses processed")
    print("  H.Prev: Hallucinations prevented")
    print("  FP: False positives (correct responses flagged)")
    print("  Abst: Abstentions")
    print("  Prec: Precision of remaining responses")
    print("  Recall: Fraction of correct responses preserved")
    print("  H.Red: Hallucination reduction rate")
    print("  U.Loss: Utility loss (correct responses withheld)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate mitigation strategies")
    parser.add_argument('--baseline', type=str, nargs='+', required=True, help='Baseline result files')
    parser.add_argument('--symbolic', type=str, default=None, help='Symbolic checker results')
    parser.add_argument('--abstention', type=str, default=None, help='Abstention results')
    parser.add_argument('--annotations', type=str, required=True, help='Ground truth annotations CSV')
    parser.add_argument('--output', type=str, required=True, help='Output comparison file')

    args = parser.parse_args()

    # Load annotations
    print("Loading ground truth annotations...")
    annotations = load_annotations(Path(args.annotations))
    print(f"Loaded {len(annotations)} annotations")

    # Load baseline results
    print("\nLoading baseline results...")
    baseline_results = []
    for f in args.baseline:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            if isinstance(data, list):
                baseline_results.extend(data)
            elif isinstance(data, dict) and 'results' in data:
                baseline_results.extend(data['results'])
    print(f"Loaded {len(baseline_results)} baseline results")

    # Evaluate all strategies
    metrics_list = []

    # Baseline
    print("\nEvaluating baseline...")
    baseline_metrics = evaluate_baseline(baseline_results, annotations)
    metrics_list.append(baseline_metrics)

    # Symbolic checker
    if args.symbolic:
        print("Evaluating symbolic checker...")
        with open(args.symbolic, 'r', encoding='utf-8') as f:
            symbolic_data = json.load(f)
        symbolic_metrics = evaluate_symbolic_checker(symbolic_data, annotations)
        metrics_list.append(symbolic_metrics)

    # Abstention
    if args.abstention:
        print("Evaluating abstention strategy...")
        with open(args.abstention, 'r', encoding='utf-8') as f:
            abstention_data = json.load(f)
        abstention_metrics = evaluate_abstention(abstention_data, annotations)
        metrics_list.append(abstention_metrics)

    # Print comparison
    print_comparison(metrics_list)

    # Save results
    output_data = {
        'metrics': [
            {
                'name': m.name,
                'total_responses': m.total_responses,
                'hallucinations_prevented': m.hallucinations_prevented,
                'false_positives': m.false_positives,
                'abstention_count': m.abstention_count,
                'precision': m.precision,
                'recall': m.recall,
                'hallucination_reduction': m.hallucination_reduction,
                'utility_loss': m.utility_loss
            }
            for m in metrics_list
        ]
    }

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
