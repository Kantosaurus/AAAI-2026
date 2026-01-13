#!/usr/bin/env python3
"""
Causal Tracing for Hallucination Localization
Identifies which layers and attention heads causally contribute to hallucinated tokens

Based on: "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)

Usage:
    python causal_tracing.py \
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


def run_with_cache(model, tokenizer, prompt: str, device: str = 'cuda') -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Run model and cache all hidden states

    Returns:
        logits, hidden_states_per_layer
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    if device == 'cuda':
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.logits, outputs.hidden_states


def causal_trace_layer(
    model,
    tokenizer,
    prompt: str,
    target_token: str,
    noise_level: float = 0.1,
    device: str = 'cuda'
) -> CausalTraceResult:
    """
    Perform causal tracing to identify critical layers

    Algorithm:
    1. Run clean forward pass, record activations and target token probability
    2. Run corrupted pass (with noise), record degraded probability
    3. For each layer: restore clean activations, measure probability recovery
    4. Layers with high recovery are causally important
    """

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt')
    if device == 'cuda':
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_ids = inputs['input_ids']

    # Find target token
    target_token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if not target_token_ids:
        raise ValueError(f"Could not tokenize target token: {target_token}")

    target_token_id = target_token_ids[0]

    # Step 1: Clean forward pass
    with torch.no_grad():
        clean_outputs = model(**inputs, output_hidden_states=True)
        clean_logits = clean_outputs.logits
        clean_hidden_states = clean_outputs.hidden_states

    # Find position of target token in output
    target_position = -1  # Default to last position
    for pos in range(clean_logits.size(1)):
        if clean_logits[0, pos].argmax().item() == target_token_id:
            target_position = pos
            break

    baseline_prob = torch.softmax(clean_logits[0, target_position], dim=-1)[target_token_id].item()

    # Step 2: Corrupted forward pass (add noise to all hidden states)
    # We'll use hooks to inject noise
    corrupted_hidden_states = []

    def corruption_hook(module, input, output):
        """Hook to corrupt activations"""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        corrupted = add_noise_to_activations(hidden, noise_level)
        corrupted_hidden_states.append(corrupted.detach().clone())

        if isinstance(output, tuple):
            return (corrupted,) + output[1:]
        return corrupted

    # Attach hooks to all layers
    hooks = []
    for layer in model.model.layers:
        hook = layer.register_forward_hook(corruption_hook)
        hooks.append(hook)

    with torch.no_grad():
        corrupted_outputs = model(**inputs)
        corrupted_logits = corrupted_outputs.logits

    # Remove hooks
    for hook in hooks:
        hook.remove()

    corrupted_prob = torch.softmax(corrupted_logits[0, target_position], dim=-1)[target_token_id].item()

    # Step 3: Restore each layer individually and measure recovery
    layer_effects = []
    n_layers = len(clean_hidden_states) - 1  # Exclude embedding layer

    for layer_idx in tqdm(range(n_layers), desc="Tracing layers", leave=False):
        # Create restoration hook for specific layer
        def restoration_hook(module, input, output, layer_idx=layer_idx):
            """Restore clean activations for this layer only"""
            if isinstance(output, tuple):
                return (clean_hidden_states[layer_idx + 1],) + output[1:]
            return clean_hidden_states[layer_idx + 1]

        # Attach hook to specific layer
        hook = model.model.layers[layer_idx].register_forward_hook(restoration_hook)

        # Run with this layer restored
        with torch.no_grad():
            restored_outputs = model(**inputs)
            restored_logits = restored_outputs.logits

        hook.remove()

        # Measure recovery
        restored_prob = torch.softmax(restored_logits[0, target_position], dim=-1)[target_token_id].item()

        # Effect = how much probability recovered
        effect = (restored_prob - corrupted_prob) / (baseline_prob - corrupted_prob + 1e-10)
        layer_effects.append(effect)

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
    parser = argparse.ArgumentParser(description="Causal tracing for hallucination localization")
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
            result = causal_trace_layer(
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
            continue

    # Save results
    output_file = output_dir / "causal_trace_results.json"
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
