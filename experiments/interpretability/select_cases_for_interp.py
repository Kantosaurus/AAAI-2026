#!/usr/bin/env python3
"""
Select Cases for Interpretability Analysis
Choose diverse, reproducible hallucination cases from annotated pilot results

Usage:
    python select_cases_for_interp.py \
        --annotations annotations/adjudication/final_annotations.csv \
        --results results/pilot/ \
        --output experiments/interpretability/selected_cases.json \
        --n-cases 30
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def load_annotations(csv_file: Path) -> Dict[str, Dict]:
    """Load adjudicated annotations"""
    annotations = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['prompt_id'], row['model'])
            annotations[key] = {
                'hallucination_binary': int(row['hallucination_binary']),
                'hallucination_types': row.get('hallucination_types', ''),
                'severity': row.get('severity', ''),
                'citation_correctness': row.get('citation_correctness', ''),
                'notes': row.get('notes', '')
            }
    return annotations


def load_pilot_results(results_dir: Path) -> Dict[str, Dict]:
    """Load pilot results"""
    results = {}
    for json_file in results_dir.glob("pilot_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get('results', [])
            for item in items:
                key = (item['prompt_id'], item['model'])
                results[key] = item
    return results


def select_diverse_cases(annotations: Dict, results: Dict, n_cases: int = 30,
                         model_filter: str = None) -> List[Dict]:
    """
    Select diverse hallucination cases for interpretability

    Selection criteria:
    1. Only hallucinations (binary = 1)
    2. Diverse hallucination types
    3. Mix of severities
    4. From specified model (for reproducibility)
    5. Reproducible at temp=0.0
    """

    # Filter to hallucinations from target model
    candidates = []
    for key, annotation in annotations.items():
        if annotation['hallucination_binary'] == 1:
            prompt_id, model = key

            # Filter by model if specified
            if model_filter and model_filter not in model:
                continue

            # Get corresponding result
            if key in results:
                result = results[key]

                # Prefer deterministic samples (temp=0.0)
                temp = result.get('sampling_params', {}).get('temperature', 1.0)

                candidates.append({
                    'key': key,
                    'prompt_id': prompt_id,
                    'model': model,
                    'temperature': temp,
                    'annotation': annotation,
                    'result': result,
                    'severity': annotation.get('severity', 'Unknown'),
                    'types': annotation.get('hallucination_types', '').split(',')
                })

    # Sort by priority: deterministic first, then by severity
    severity_order = {'High': 0, 'Medium': 1, 'Low': 2, 'Unknown': 3}
    candidates.sort(key=lambda x: (
        x['temperature'],  # Prefer temp=0
        severity_order.get(x['severity'], 3),  # Then by severity
    ))

    # Select diverse cases
    selected = []
    type_counts = defaultdict(int)

    for candidate in candidates:
        if len(selected) >= n_cases:
            break

        # Ensure diversity in hallucination types
        types = [t.strip() for t in candidate['types'] if t.strip()]

        # Add case
        selected.append({
            'prompt_id': candidate['prompt_id'],
            'model': candidate['model'],
            'prompt': candidate['result'].get('prompt', ''),
            'response': candidate['result']['full_response'],
            'hallucination_types': types,
            'severity': candidate['severity'],
            'temperature': candidate['temperature'],
            'notes': candidate['annotation']['notes'],
            'sampling_params': candidate['result'].get('sampling_params', {}),
            'tokens_used': candidate['result'].get('tokens_used', {}),
            'timestamp': candidate['result'].get('timestamp', '')
        })

        # Track type diversity
        for t in types:
            type_counts[t] += 1

    return selected


def export_selected_cases(cases: List[Dict], output_file: Path):
    """Export selected cases to JSON"""
    output = {
        'metadata': {
            'n_cases': len(cases),
            'selection_criteria': [
                'Only confirmed hallucinations (binary = 1)',
                'Diverse hallucination types',
                'Mix of severity levels',
                'Preference for deterministic samples (temp=0.0)',
                'Reproducible cases only'
            ]
        },
        'cases': cases,
        'type_distribution': {}
    }

    # Compute type distribution
    type_counts = defaultdict(int)
    for case in cases:
        for t in case.get('hallucination_types', []):
            type_counts[t] += 1

    output['type_distribution'] = dict(type_counts)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SELECTED CASES FOR INTERPRETABILITY")
    print("="*60)
    print(f"\nTotal cases selected: {len(cases)}")

    severity_counts = defaultdict(int)
    temp_counts = defaultdict(int)
    for case in cases:
        severity_counts[case.get('severity', 'Unknown')] += 1
        temp_counts[case.get('temperature', 1.0)] += 1

    print("\nSeverity distribution:")
    for severity in ['High', 'Medium', 'Low', 'Unknown']:
        if severity in severity_counts:
            print(f"  {severity}: {severity_counts[severity]}")

    print("\nTemperature distribution:")
    for temp in sorted(temp_counts.keys()):
        print(f"  {temp}: {temp_counts[temp]}")

    print("\nHallucination type distribution:")
    for htype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        if htype:
            print(f"  {htype}: {count}")

    print(f"\n✓ Cases exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Select cases for interpretability analysis")
    parser.add_argument('--annotations', type=str, required=True, help='CSV file with adjudicated annotations')
    parser.add_argument('--results', type=str, required=True, help='Directory with pilot result JSON files')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file for selected cases')
    parser.add_argument('--n-cases', type=int, default=30, help='Number of cases to select')
    parser.add_argument('--model', type=str, default=None, help='Filter to specific model (e.g., "Qwen")')

    args = parser.parse_args()

    # Load data
    print("Loading annotations...")
    annotations = load_annotations(Path(args.annotations))
    print(f"Loaded {len(annotations)} annotations")

    print("Loading pilot results...")
    results = load_pilot_results(Path(args.results))
    print(f"Loaded {len(results)} results")

    # Select cases
    print(f"\nSelecting {args.n_cases} cases...")
    if args.model:
        print(f"Filtering to model: {args.model}")

    selected = select_diverse_cases(annotations, results, args.n_cases, args.model)

    # Export
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    export_selected_cases(selected, output_file)


if __name__ == '__main__':
    main()
