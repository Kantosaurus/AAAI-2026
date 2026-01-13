#!/usr/bin/env python3
"""
Prepare Annotation Batches from Pilot Results
Loads pilot outputs and creates randomized annotation batches for independent annotators

Usage:
    python prepare_annotation_batches.py --results results/pilot/ --output annotations/batches/
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict
import csv


def load_pilot_results(results_dir: Path) -> List[Dict]:
    """Load all pilot result JSON files"""
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    json_files = list(results_dir.glob("pilot_*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No pilot result files (pilot_*.json) found in {results_dir}\n"
            f"Make sure to run the pilot script first: python run_pilot.py"
        )

    results = []
    for json_file in json_files:
        print(f"Loading {json_file.name}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                elif isinstance(data, dict) and 'results' in data:
                    results.extend(data['results'])
                else:
                    print(f"Warning: Unexpected format in {json_file.name}, skipping")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {json_file.name}: {e}")
            continue

    if not results:
        raise ValueError(
            f"No results loaded from {len(json_files)} files. "
            f"Check that JSON files contain valid pilot results."
        )

    print(f"Loaded {len(results)} total responses")
    return results


def create_annotation_batches(results: List[Dict], num_annotators: int = 2,
                              overlap_ratio: float = 1.0, seed: int = 42) -> Dict[str, List[Dict]]:
    """
    Create annotation batches with controlled overlap

    Args:
        results: List of model responses
        num_annotators: Number of independent annotators
        overlap_ratio: Fraction of samples to have dual annotation (1.0 = all samples)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping annotator_id to list of samples
    """
    random.seed(seed)

    # Shuffle results for randomization (blind annotators to model order)
    shuffled = results.copy()
    random.shuffle(shuffled)

    # Calculate overlap
    total_samples = len(shuffled)
    overlap_count = int(total_samples * overlap_ratio)

    batches = {}

    if overlap_ratio == 1.0:
        # All samples annotated by all annotators (highest quality)
        for i in range(num_annotators):
            batches[f"annotator_{i+1}"] = shuffled.copy()
    else:
        # Partial overlap scheme
        overlap_samples = shuffled[:overlap_count]
        unique_samples = shuffled[overlap_count:]

        # Distribute unique samples
        unique_per_annotator = len(unique_samples) // num_annotators

        for i in range(num_annotators):
            annotator_samples = overlap_samples.copy()
            start_idx = i * unique_per_annotator
            end_idx = (i + 1) * unique_per_annotator if i < num_annotators - 1 else len(unique_samples)
            annotator_samples.extend(unique_samples[start_idx:end_idx])

            # Shuffle each annotator's batch independently
            random.shuffle(annotator_samples)
            batches[f"annotator_{i+1}"] = annotator_samples

    return batches


def export_annotation_csv(batch: List[Dict], output_file: Path, annotator_id: str):
    """Export batch as CSV template for annotation"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'prompt_id',
            'model',
            'annotator',
            'hallucination_binary',
            'hallucination_types',
            'severity',
            'citation_correctness',
            'notes',
            '---PROMPT---',
            '---RESPONSE---'
        ])

        # Data rows
        for item in batch:
            # Extract prompt from result (might be nested)
            prompt_text = item.get('prompt', item.get('prompt_text', ''))

            writer.writerow([
                item.get('prompt_id', ''),
                item.get('model', ''),
                annotator_id,
                '',  # hallucination_binary (to be filled)
                '',  # hallucination_types (to be filled)
                '',  # severity (to be filled)
                '',  # citation_correctness (to be filled)
                '',  # notes (to be filled)
                prompt_text,
                item.get('full_response', '')
            ])

    print(f"Exported {len(batch)} samples to {output_file}")


def generate_annotation_summary(batches: Dict[str, List[Dict]], output_file: Path):
    """Generate summary statistics about annotation batches"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Annotation Batch Summary\n\n")
        f.write(f"**Generated:** {Path(__file__).name}\n")
        f.write(f"**Number of Annotators:** {len(batches)}\n\n")

        total_annotations = sum(len(batch) for batch in batches.values())
        unique_samples = len(set(
            item['prompt_id']
            for batch in batches.values()
            for item in batch
        ))

        f.write(f"**Total Annotation Tasks:** {total_annotations}\n")
        f.write(f"**Unique Samples:** {unique_samples}\n")
        f.write(f"**Average Annotations per Sample:** {total_annotations / unique_samples:.2f}\n\n")

        f.write("## Per-Annotator Breakdown\n\n")
        for annotator_id, batch in sorted(batches.items()):
            f.write(f"### {annotator_id}\n")
            f.write(f"- Samples: {len(batch)}\n")

            # Count by model
            model_counts = {}
            for item in batch:
                model = item.get('model', 'unknown')
                model_counts[model] = model_counts.get(model, 0) + 1

            f.write(f"- Models:\n")
            for model, count in sorted(model_counts.items()):
                f.write(f"  - {model}: {count}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare annotation batches from pilot results")
    parser.add_argument('--results', type=str, required=True, help='Directory containing pilot result JSON files')
    parser.add_argument('--output', type=str, required=True, help='Output directory for annotation batches')
    parser.add_argument('--num-annotators', type=int, default=2, help='Number of independent annotators')
    parser.add_argument('--overlap', type=float, default=1.0, help='Fraction of samples with dual annotation (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return

    results = load_pilot_results(results_dir)
    if not results:
        print("Error: No results loaded")
        return

    # Create batches
    print(f"\nCreating batches for {args.num_annotators} annotators with {args.overlap*100}% overlap...")
    batches = create_annotation_batches(results, args.num_annotators, args.overlap, args.seed)

    # Export batches
    print("\nExporting annotation batches...")
    for annotator_id, batch in batches.items():
        output_file = output_dir / f"{annotator_id}_batch.csv"
        export_annotation_csv(batch, output_file, annotator_id)

    # Generate summary
    summary_file = output_dir / "batch_summary.md"
    generate_annotation_summary(batches, summary_file)
    print(f"\nSummary written to {summary_file}")

    print("\n✓ Annotation batches prepared successfully!")
    print(f"  Annotators should complete their CSV files in: {output_dir}")
    print(f"  After completion, run compute_agreement.py for IAA metrics")


if __name__ == '__main__':
    main()
