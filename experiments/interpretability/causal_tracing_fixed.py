#!/usr/bin/env python3
"""
Causal Tracing for Hallucination Localization (Fixed Version)
Identifies which layers and attention heads causally contribute to hallucinated tokens

Based on: "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)

Fixes:
- Proper single-layer restoration (not affecting subsequent layers)
- Validation of target token detection
- Better error handling

Usage:
    python causal_tracing_fixed.py \
        --cases selected_cases.json \
        --model Qwen/Qwen2.5-14B-Instruct \
        --output results/causal_traces/ \
        --n-cases 10
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import copy

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Error: transformers not installed. Install with: pip install transformers torch")


@dataclass
class CausalTraceResult:
    """Results from causal tracing on a single case"""
    prompt_id: str
    hallucinated_token: str
    hallucinated_token_id: int
    token_position: int
    layer_effects: List[float]  # Effect of restoring each layer
    baseline_prob: float
    corrupted_prob: float


def add_noise_to_activations(activations: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to activations to corrupt them"""
    noise = torch.randn_like(activations) * noise_level * activations.std()
    return activations + noise


def get_model_and_tokenizer(model_name: str, device: str = 'cuda'):
    """Load model and tokenizer"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers not available")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def causal_trace_layer_fixed(
    model,
    tokenizer,
    prompt: str,
    target_token: str,
    noise_level: float = 0.1,
    device: str = 'cuda'
) -> CausalTraceResult:
    """
    Perform causal tracing to identify critical layers (FIXED VERSION)

    Algorithm:
    1. Run clean forward pass, cache all layer activations
    2. Run corrupted pass (all layers noised), measure degraded probability
    3. For each layer: Run with ONLY that layer clean, others noised
    4. Measure how much each layer's clean activation recovers the probability

    FIX: Each restoration run starts fresh with noise, not cumulative
    """

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    if device == 'cuda':
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_ids = inputs['input_ids']

    # Find target token
    target_token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if not target_token_ids:
        raise ValueError(f"Could not tokenize target token: {target_token}")

    target_token_id = target_token_ids[0]

    # Step 1: Clean forward pass - cache activations
    print(f"  Running clean forward pass...")
    with torch.no_grad():
        clean_outputs = model(**inputs, output_hidden_states=True)
        clean_logits = clean_outputs.logits
        clean_hidden_states = clean_outputs.hidden_states

    # Find position of target token
    target_position = -1  # Default to last position
    predicted_ids = clean_logits[0].argmax(dim=-1)

    # Try to find exact match
    for pos in range(len(predicted_ids)):
        if predicted_ids[pos].item() == target_token_id:
            target_position = pos
            break

    # If not found, use last position
    if target_position == -1:
        print(f"  Warning: Target token {target_token} not found in predictions, using last position")
        target_position = len(predicted_ids) - 1

    baseline_prob = torch.softmax(clean_logits[0, target_position], dim=-1)[target_token_id].item()
    print(f"  Baseline probability: {baseline_prob:.4f}")

    # Step 2: Fully corrupted forward pass
    print(f"  Running corrupted forward pass...")

    # Store corrupted states for each layer
    corrupted_states = []

    def corruption_hook(module, input, output):
        """Hook to corrupt and cache activations"""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        corrupted = add_noise_to_activations(hidden.clone(), noise_level)
        corrupted_states.append(corrupted.detach())

        if isinstance(output, tuple):
            return (corrupted,) + output[1:]
        return corrupted

    # Attach corruption hooks
    hooks = []
    for layer in model.model.layers:
        hook = layer.register_forward_hook(corruption_hook)
        hooks.append(hook)

    with torch.no_grad():
        corrupted_outputs = model(**inputs, output_hidden_states=False)
        corrupted_logits = corrupted_outputs.logits

    # Remove hooks
    for hook in hooks:
        hook.remove()

    corrupted_prob = torch.softmax(corrupted_logits[0, target_position], dim=-1)[target_token_id].item()
    print(f"  Corrupted probability: {corrupted_prob:.4f}")

    if baseline_prob <= corrupted_prob:
        print(f"  Warning: Corrupted prob >= baseline prob, noise may be too small")

    # Step 3: Test each layer individually
    print(f"  Testing individual layer restorations...")
    n_layers = len(model.model.layers)
    layer_effects = []

    for layer_idx in tqdm(range(n_layers), desc="Tracing layers", leave=False):
        # For THIS run, corrupt all layers EXCEPT layer_idx

        current_layer = [0]  # Track which layer we're at

        def selective_corruption_hook(module, input, output, restore_idx=layer_idx):
            """Corrupt all layers except the one we're testing"""
            idx = current_layer[0]
            current_layer[0] += 1

            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # If this is the layer we're testing, use CLEAN activation
            if idx == restore_idx:
                # Use clean hidden state from step 1
                clean = clean_hidden_states[idx + 1]  # +1 because hidden_states[0] is embeddings
                if isinstance(output, tuple):
                    return (clean,) + output[1:]
                return clean
            else:
                # Corrupt this layer
                corrupted = add_noise_to_activations(hidden.clone(), noise_level)
                if isinstance(output, tuple):
                    return (corrupted,) + output[1:]
                return corrupted

        # Attach selective hooks
        hooks = []
        for layer in model.model.layers:
            hook = layer.register_forward_hook(selective_corruption_hook)
            hooks.append(hook)

        # Run with only target layer clean
        with torch.no_grad():
            restored_outputs = model(**inputs, output_hidden_states=False)
            restored_logits = restored_outputs.logits

        # Remove hooks
        for hook in hooks:
            hook.remove()

        restored_prob = torch.softmax(restored_logits[0, target_position], dim=-1)[target_token_id].item()

        # Calculate effect: how much did restoring this layer help?
        # Normalized: 0 = no effect, 1 = full recovery
        if baseline_prob > corrupted_prob:
            effect = (restored_prob - corrupted_prob) / (baseline_prob - corrupted_prob)
        else:
            effect = 0.0

        layer_effects.append(float(effect))

    return CausalTraceResult(
        prompt_id="",  # To be filled by caller
        hallucinated_token=target_token,
        hallucinated_token_id=target_token_id,
        token_position=target_position,
        layer_effects=layer_effects,
        baseline_prob=baseline_prob,
        corrupted_prob=corrupted_prob
    )


def identify_hallucinated_token(case: Dict) -> Optional[str]:
    """
    Identify the most likely hallucinated token from a case

    Heuristics:
    1. Look for CVE-YYYY-XXXXX patterns in response
    2. Extract first non-existent CVE if it's a synthetic probe
    3. Otherwise, extract first proper noun/entity
    """
    response = case['response']
    notes = case.get('notes', '')

    # Try to extract fabricated CVE from notes
    import re
    cve_pattern = r'CVE-\d{4}-\d{4,7}'

    # Check notes for fabricated CVE
    cves_in_notes = re.findall(cve_pattern, notes)
    if cves_in_notes:
        return cves_in_notes[0]

    # Check response for any CVE
    cves_in_response = re.findall(cve_pattern, response)
    if cves_in_response:
        return cves_in_response[0]

    # Fallback: return first capitalized word (likely entity)
    words = response.split()
    for word in words:
        if word and word[0].isupper() and len(word) > 3:
            return word

    return None


def main():
    parser = argparse.ArgumentParser(description="Causal tracing for hallucination localization (FIXED)")
    parser.add_argument('--cases', type=str, required=True, help='JSON file with selected cases')
    parser.add_argument('--model', type=str, required=True, help='Model name/path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--n-cases', type=int, default=10, help='Number of cases to trace')
    parser.add_argument('--noise-level', type=float, default=0.1, help='Noise level for corruption')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Load cases
    print("Loading cases...")
    with open(args.cases, 'r', encoding='utf-8') as f:
        data = json.load(f)
        cases = data['cases'][:args.n_cases]

    print(f"Loaded {len(cases)} cases for tracing")

    # Load model
    model, tokenizer = get_model_and_tokenizer(args.model, args.device)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run causal tracing on each case
    results = []

    for idx, case in enumerate(tqdm(cases, desc="Tracing cases")):
        print(f"\n--- Case {idx+1}/{len(cases)}: {case['prompt_id']} ---")

        # Identify hallucinated token
        target_token = identify_hallucinated_token(case)
        if not target_token:
            print(f"Warning: Could not identify hallucinated token for {case['prompt_id']}, skipping")
            continue

        print(f"Target token: {target_token}")

        # Run causal tracing
        try:
            result = causal_trace_layer_fixed(
                model, tokenizer,
                case['prompt'],
                target_token,
                noise_level=args.noise_level,
                device=args.device
            )
            result.prompt_id = case['prompt_id']
            results.append(result)

            # Find most important layers
            top_layers = sorted(enumerate(result.layer_effects), key=lambda x: -x[1])[:3]
            print(f"Top 3 layers: {[(l, f'{e:.3f}') for l, e in top_layers]}")

        except Exception as e:
            print(f"Error tracing case {case['prompt_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    output_file = output_dir / "causal_trace_results_fixed.json"
    output_data = {
        'model': args.model,
        'n_cases': len(results),
        'noise_level': args.noise_level,
        'results': [
            {
                'prompt_id': r.prompt_id,
                'hallucinated_token': r.hallucinated_token,
                'token_position': r.token_position,
                'baseline_prob': r.baseline_prob,
                'corrupted_prob': r.corrupted_prob,
                'layer_effects': r.layer_effects,
                'top_3_layers': sorted(enumerate(r.layer_effects), key=lambda x: -x[1])[:3]
            }
            for r in results
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print(f"Processed {len(results)}/{len(cases)} cases successfully")


if __name__ == '__main__':
    main()
