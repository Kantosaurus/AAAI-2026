#!/usr/bin/env python3
"""
Activation Probes for Hallucination Detection
Train linear probes to detect hallucination-predictive features in model activations

Usage:
    python activation_probes.py \
        --cases selected_cases.json \
        --model Qwen/Qwen2.5-14B-Instruct \
        --output results/probes/
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class ProbeResult:
    """Results from training a probe on a specific layer"""
    layer_idx: int
    train_accuracy: float
    test_accuracy: float
    train_auc: float
    test_auc: float
    n_train: int
    n_test: int


def extract_activations(model, tokenizer, texts: List[str], layer_idx: int,
                       device: str = 'cuda') -> torch.Tensor:
    """
    Extract activations from a specific layer for a list of texts

    Returns:
        Tensor of shape (n_texts, hidden_dim) - mean-pooled activations
    """
    activations = []

    for text in tqdm(texts, desc=f"Extracting layer {layer_idx}", leave=False):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        if device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get hidden states from specific layer
            hidden_states = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)

            # Mean pool across sequence
            pooled = hidden_states.mean(dim=1)  # (1, hidden_dim)
            activations.append(pooled.cpu())

    return torch.cat(activations, dim=0)  # (n_texts, hidden_dim)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> ProbeResult:
    """Train a linear probe and evaluate"""

    # Train logistic regression
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)

    # Predictions
    train_pred = probe.predict(X_train)
    test_pred = probe.predict(X_test)
    train_proba = probe.predict_proba(X_train)[:, 1]
    test_proba = probe.predict_proba(X_test)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_auc = roc_auc_score(y_train, train_proba) if len(np.unique(y_train)) > 1 else 0.0
    test_auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.0

    return ProbeResult(
        layer_idx=-1,  # To be filled by caller
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        train_auc=train_auc,
        test_auc=test_auc,
        n_train=len(y_train),
        n_test=len(y_test)
    )


def prepare_dataset(cases: List[Dict], annotations_path: Path = None) -> Tuple[List[str], List[int]]:
    """
    Prepare dataset for probe training

    Returns:
        texts, labels (1 = hallucination, 0 = no hallucination)
    """
    texts = []
    labels = []

    # Cases are already labeled hallucinations
    for case in cases:
        # The prompt + response
        text = case['prompt'] + " " + case['response']
        texts.append(text)
        labels.append(1)  # All selected cases are hallucinations

    # Need negative examples (non-hallucinations)
    # Load from pilot results where hallucination_binary = 0
    # For now, we'll use a simple heuristic: assume cases not in selected_cases are negatives
    # In practice, should load from adjudicated annotations

    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Train activation probes for hallucination detection")
    parser.add_argument('--cases', type=str, required=True, help='JSON file with selected cases')
    parser.add_argument('--model', type=str, required=True, help='Model name/path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--max-cases', type=int, default=100, help='Max cases to use (for speed)')

    args = parser.parse_args()

    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers not installed")
        return

    # Load cases
    print("Loading cases...")
    with open(args.cases, 'r', encoding='utf-8') as f:
        data = json.load(f)
        cases = data['cases'][:args.max_cases]

    print(f"Loaded {len(cases)} hallucination cases")
    print("Note: Need negative examples (non-hallucinations) for balanced training")
    print("Current implementation is a simplified demo. Extend with full annotation data.")

    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        device_map='auto' if args.device == 'cuda' else None,
        output_hidden_states=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers")

    # Prepare texts
    texts = [case['prompt'] + " " + case['response'] for case in cases]
    labels = [1] * len(cases)  # All are hallucinations

    # For demo: create synthetic negatives by truncating prompts (not ideal, but simple)
    # In production: use actual non-hallucination cases
    negative_texts = [case['prompt'] for case in cases]  # Just prompts without hallucinated responses
    negative_labels = [0] * len(negative_texts)

    all_texts = texts + negative_texts
    all_labels = labels + negative_labels

    print(f"\nDataset: {len(texts)} hallucinations, {len(negative_texts)} non-hallucinations")

    # Split train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )

    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Train probes for each layer
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Sample a subset of layers to save time
    layers_to_probe = list(range(0, n_layers, max(1, n_layers // 10)))  # Sample ~10 layers

    for layer_idx in tqdm(layers_to_probe, desc="Training probes"):
        print(f"\n--- Layer {layer_idx}/{n_layers} ---")

        # Extract activations
        print("Extracting train activations...")
        X_train = extract_activations(model, tokenizer, train_texts, layer_idx, args.device)

        print("Extracting test activations...")
        X_test = extract_activations(model, tokenizer, test_texts, layer_idx, args.device)

        # Convert to numpy
        X_train_np = X_train.numpy()
        X_test_np = X_test.numpy()
        y_train_np = np.array(train_labels)
        y_test_np = np.array(test_labels)

        # Train probe
        print("Training probe...")
        result = train_probe(X_train_np, y_train_np, X_test_np, y_test_np)
        result.layer_idx = layer_idx
        results.append(result)

        print(f"Train ACC: {result.train_accuracy:.3f}, AUC: {result.train_auc:.3f}")
        print(f"Test ACC: {result.test_accuracy:.3f}, AUC: {result.test_auc:.3f}")

    # Save results
    output_file = output_dir / "probe_results.json"
    output_data = {
        'model': args.model,
        'n_layers_probed': len(results),
        'total_layers': n_layers,
        'results': [
            {
                'layer_idx': r.layer_idx,
                'train_accuracy': r.train_accuracy,
                'test_accuracy': r.test_accuracy,
                'train_auc': r.train_auc,
                'test_auc': r.test_auc,
                'n_train': r.n_train,
                'n_test': r.n_test
            }
            for r in results
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("PROBE PERFORMANCE SUMMARY")
    print("="*60)

    best_layer = max(results, key=lambda r: r.test_auc)
    print(f"\nBest layer: {best_layer.layer_idx}")
    print(f"  Test AUC: {best_layer.test_auc:.3f}")
    print(f"  Test ACC: {best_layer.test_accuracy:.3f}")

    print("\nAll layers:")
    for r in sorted(results, key=lambda x: x.layer_idx):
        print(f"  Layer {r.layer_idx:2d}: Test AUC={r.test_auc:.3f}, ACC={r.test_accuracy:.3f}")


if __name__ == '__main__':
    main()
