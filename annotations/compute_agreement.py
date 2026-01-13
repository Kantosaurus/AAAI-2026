#!/usr/bin/env python3
"""
Compute Inter-Annotator Agreement Metrics
Calculates Cohen's kappa and other agreement statistics from dual annotations

Usage:
    python compute_agreement.py --annotations annotations/batches/*.csv --output annotations/agreement_report.json
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import json


def load_annotations(csv_files: List[Path]) -> Dict[str, Dict]:
    """
    Load annotations from CSV files

    Returns:
        Dict mapping (prompt_id, model) to list of annotator judgments
    """
    annotations = defaultdict(list)

    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows or rows without annotations
                if not row.get('hallucination_binary'):
                    continue

                key = (row['prompt_id'], row['model'])
                annotations[key].append({
                    'annotator': row['annotator'],
                    'hallucination_binary': int(row['hallucination_binary']),
                    'hallucination_types': row.get('hallucination_types', ''),
                    'severity': row.get('severity', ''),
                    'citation_correctness': row.get('citation_correctness', ''),
                    'notes': row.get('notes', '')
                })

    return dict(annotations)


def compute_cohens_kappa(annotations: Dict[str, Dict]) -> Tuple[float, Dict]:
    """
    Compute Cohen's kappa for binary hallucination labels

    Returns:
        (kappa, detailed_stats)
    """
    # Filter to only items with exactly 2 annotations
    dual_annotated = {
        key: anns for key, anns in annotations.items()
        if len(anns) == 2
    }

    if not dual_annotated:
        return 0.0, {'error': 'No dual-annotated samples found'}

    # Build agreement matrix
    agree_both_0 = 0
    agree_both_1 = 0
    disagree_0_1 = 0
    disagree_1_0 = 0

    for key, anns in dual_annotated.items():
        label_1 = anns[0]['hallucination_binary']
        label_2 = anns[1]['hallucination_binary']

        if label_1 == 0 and label_2 == 0:
            agree_both_0 += 1
        elif label_1 == 1 and label_2 == 1:
            agree_both_1 += 1
        elif label_1 == 0 and label_2 == 1:
            disagree_0_1 += 1
        else:  # label_1 == 1 and label_2 == 0
            disagree_1_0 += 1

    n = len(dual_annotated)

    # Observed agreement
    p_o = (agree_both_0 + agree_both_1) / n

    # Expected agreement by chance
    annotator1_0 = agree_both_0 + disagree_0_1
    annotator1_1 = agree_both_1 + disagree_1_0
    annotator2_0 = agree_both_0 + disagree_1_0
    annotator2_1 = agree_both_1 + disagree_0_1

    p_e = ((annotator1_0 * annotator2_0) + (annotator1_1 * annotator2_1)) / (n * n)

    # Cohen's kappa
    # Special case: if p_e == 1.0, agreement is by pure chance, kappa undefined
    # Special case: if p_o == p_e, no agreement beyond chance, kappa = 0
    if abs(p_e - 1.0) < 1e-10:
        kappa = 0.0  # Perfect chance agreement, no meaningful kappa
    elif abs(p_o - p_e) < 1e-10:
        kappa = 0.0  # No agreement beyond chance
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    stats = {
        'kappa': kappa,
        'observed_agreement': p_o,
        'expected_agreement': p_e,
        'n_samples': n,
        'confusion_matrix': {
            'both_no_hallucination': agree_both_0,
            'both_hallucination': agree_both_1,
            'annotator1_no_annotator2_yes': disagree_0_1,
            'annotator1_yes_annotator2_no': disagree_1_0
        }
    }

    return kappa, stats


def compute_type_agreement(annotations: Dict[str, Dict]) -> Dict:
    """Compute agreement on hallucination types (for cases where both marked hallucination)"""
    dual_annotated = {
        key: anns for key, anns in annotations.items()
        if len(anns) == 2 and all(a['hallucination_binary'] == 1 for a in anns)
    }

    if not dual_annotated:
        return {'n_samples': 0, 'agreement': 0.0}

    exact_matches = 0
    partial_matches = 0

    for key, anns in dual_annotated.items():
        types_1 = set(anns[0]['hallucination_types'].split(',')) if anns[0]['hallucination_types'] else set()
        types_2 = set(anns[1]['hallucination_types'].split(',')) if anns[1]['hallucination_types'] else set()

        # Clean whitespace
        types_1 = {t.strip() for t in types_1 if t.strip()}
        types_2 = {t.strip() for t in types_2 if t.strip()}

        if types_1 == types_2:
            exact_matches += 1
        elif types_1 & types_2:  # Non-empty intersection
            partial_matches += 1

    n = len(dual_annotated)
    return {
        'n_samples': n,
        'exact_match_rate': exact_matches / n if n > 0 else 0,
        'partial_match_rate': (exact_matches + partial_matches) / n if n > 0 else 0
    }


def identify_disagreements(annotations: Dict[str, Dict]) -> List[Dict]:
    """Identify all disagreement cases for adjudication"""
    disagreements = []

    for key, anns in annotations.items():
        if len(anns) == 2:
            label_1 = anns[0]['hallucination_binary']
            label_2 = anns[1]['hallucination_binary']

            if label_1 != label_2:
                disagreements.append({
                    'prompt_id': key[0],
                    'model': key[1],
                    'annotator_1': anns[0]['annotator'],
                    'annotator_1_label': label_1,
                    'annotator_1_notes': anns[0]['notes'],
                    'annotator_2': anns[1]['annotator'],
                    'annotator_2_label': label_2,
                    'annotator_2_notes': anns[1]['notes']
                })

    return disagreements


def generate_agreement_report(annotations: Dict[str, Dict], output_file: Path):
    """Generate comprehensive agreement report"""

    # Compute metrics
    kappa, kappa_stats = compute_cohens_kappa(annotations)
    type_agreement = compute_type_agreement(annotations)
    disagreements = identify_disagreements(annotations)

    # Prepare report
    report = {
        'summary': {
            'total_unique_samples': len(annotations),
            'dual_annotated_samples': kappa_stats.get('n_samples', 0),
            'disagreement_count': len(disagreements),
            'disagreement_rate': len(disagreements) / kappa_stats.get('n_samples', 1)
        },
        'cohens_kappa': kappa_stats,
        'type_agreement': type_agreement,
        'disagreements': disagreements
    }

    # Save JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("INTER-ANNOTATOR AGREEMENT REPORT")
    print("="*60)
    print(f"\nCohen's Kappa: {kappa:.3f}")

    # Interpret kappa
    if kappa < 0:
        interpretation = "Poor (worse than chance)"
    elif kappa < 0.2:
        interpretation = "Slight"
    elif kappa < 0.4:
        interpretation = "Fair"
    elif kappa < 0.6:
        interpretation = "Moderate"
    elif kappa < 0.8:
        interpretation = "Substantial"
    else:
        interpretation = "Almost Perfect"

    print(f"Interpretation: {interpretation}")
    print(f"\nObserved Agreement: {kappa_stats.get('observed_agreement', 0):.1%}")
    print(f"Expected Agreement (chance): {kappa_stats.get('expected_agreement', 0):.1%}")
    print(f"\nTotal Samples: {kappa_stats.get('n_samples', 0)}")
    print(f"Disagreements: {len(disagreements)} ({len(disagreements)/kappa_stats.get('n_samples', 1):.1%})")

    print("\nConfusion Matrix:")
    cm = kappa_stats.get('confusion_matrix', {})
    print(f"  Both labeled NO:  {cm.get('both_no_hallucination', 0)}")
    print(f"  Both labeled YES: {cm.get('both_hallucination', 0)}")
    print(f"  A1=NO, A2=YES:    {cm.get('annotator1_no_annotator2_yes', 0)}")
    print(f"  A1=YES, A2=NO:    {cm.get('annotator1_yes_annotator2_no', 0)}")

    if type_agreement['n_samples'] > 0:
        print(f"\nHallucination Type Agreement (among cases both marked as hallucination):")
        print(f"  Exact match: {type_agreement['exact_match_rate']:.1%}")
        print(f"  Partial match: {type_agreement['partial_match_rate']:.1%}")

    print(f"\n✓ Full report saved to: {output_file}")

    if disagreements:
        print(f"\n⚠ {len(disagreements)} disagreements require adjudication")
        print(f"  See 'disagreements' section in report for details")


def main():
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement metrics")
    parser.add_argument('--annotations', nargs='+', required=True, help='Annotation CSV files')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file for report')

    args = parser.parse_args()

    # Load annotations
    csv_files = [Path(f) for f in args.annotations]
    print(f"Loading annotations from {len(csv_files)} files...")
    annotations = load_annotations(csv_files)

    if not annotations:
        print("Error: No annotations loaded")
        return

    print(f"Loaded annotations for {len(annotations)} unique samples")

    # Generate report
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    generate_agreement_report(annotations, output_file)


if __name__ == '__main__':
    main()
