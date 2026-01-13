#!/usr/bin/env python3
"""
Uncertainty-based Abstention Detector
Identifies low-confidence responses where the model should abstain from answering

Usage:
    python abstention_detector.py \
        --results results/pilot/pilot_*.json \
        --threshold 0.3 \
        --output results/abstention_results.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class AbstentionDecision:
    """Decision on whether to abstain from a response"""
    prompt_id: str
    model: str
    should_abstain: bool
    confidence_score: float
    abstention_reasons: List[str]
    has_hedging: bool
    has_low_token_prob: bool
    original_response: str
    abstention_message: str


# Hedging phrases that indicate uncertainty
HEDGING_PHRASES = [
    "i'm not sure",
    "i don't know",
    "i cannot verify",
    "i cannot confirm",
    "i'm uncertain",
    "may be",
    "might be",
    "possibly",
    "perhaps",
    "could be",
    "it's unclear",
    "unclear whether",
    "not certain",
    "cannot determine",
    "insufficient information",
    "unable to verify",
    "no information",
    "unknown",
    "i don't have",
    "i lack",
    "cannot access",
    "cannot find",
    "does not exist",
    "may not exist",
]


def detect_hedging(text: str) -> tuple[bool, List[str]]:
    """
    Detect hedging phrases in response

    Returns:
        (has_hedging, list_of_found_phrases)
    """
    text_lower = text.lower()
    found_phrases = []

    for phrase in HEDGING_PHRASES:
        if phrase in text_lower:
            found_phrases.append(phrase)

    return len(found_phrases) > 0, found_phrases


def compute_confidence_from_logprobs(token_logprobs: Optional[List[Dict]], threshold: float = -2.0) -> Tuple[bool, float]:
    """
    Compute confidence from token log probabilities

    Returns:
        (has_low_confidence, avg_confidence_score)
    """
    if not token_logprobs:
        return False, 1.0  # No logprobs available, assume confident

    # Extract top token logprobs
    top_logprobs = []
    for token_data in token_logprobs[:50]:  # First 50 tokens
        if isinstance(token_data, dict) and 'top_logprobs' in token_data:
            # Get top logprob
            probs = token_data['top_logprobs']
            if probs:
                top_logprobs.append(max(probs))

    if not top_logprobs:
        return False, 1.0

    # Convert to probabilities
    avg_logprob = np.mean(top_logprobs)
    avg_prob = np.exp(avg_logprob)

    # Check if average probability is below threshold
    has_low_confidence = avg_logprob < threshold

    return has_low_confidence, avg_prob


def detect_explicit_abstention(text: str) -> bool:
    """Check if response already explicitly abstains"""
    abstention_markers = [
        "i cannot provide",
        "i cannot answer",
        "i won't provide",
        "i will not provide",
        "i should not",
        "i cannot generate",
        "refuse to",
        "decline to",
    ]

    text_lower = text.lower()
    return any(marker in text_lower for marker in abstention_markers)


def compute_confidence_score(result: Dict) -> Tuple[float, List[str]]:
    """
    Compute overall confidence score

    Returns:
        (confidence_score, reasons_for_low_confidence)
    """
    response = result.get('full_response', '')
    token_logprobs = result.get('token_logprobs', None)

    score = 1.0  # Start with full confidence
    reasons = []

    # Check 1: Hedging phrases
    has_hedging, hedging_phrases = detect_hedging(response)
    if has_hedging:
        score *= 0.5  # Reduce confidence by 50%
        reasons.append(f"hedging_phrases: {', '.join(hedging_phrases[:3])}")

    # Check 2: Token probabilities (if available)
    if token_logprobs:
        has_low_prob, avg_prob = compute_confidence_from_logprobs(token_logprobs)
        if has_low_prob:
            score *= 0.6  # Reduce confidence by 40%
            reasons.append(f"low_token_probability: {avg_prob:.3f}")

    # Check 3: Very short response (might indicate uncertainty)
    if len(response.split()) < 20:
        score *= 0.9  # Small penalty
        reasons.append(f"short_response: {len(response.split())} words")

    # Check 4: Explicit abstention already present
    if detect_explicit_abstention(response):
        score = 0.0
        reasons.append("explicit_abstention")

    return score, reasons


def make_abstention_decision(
    result: Dict,
    threshold: float = 0.5,
    require_high_precision: bool = True
) -> AbstentionDecision:
    """
    Decide whether to abstain from this response

    Args:
        result: Model output result
        threshold: Confidence threshold below which to abstain
        require_high_precision: If True, only abstain when very certain (high precision)
    """
    prompt_id = result.get('prompt_id', '')
    model = result.get('model', '')
    response = result.get('full_response', '')

    # Compute confidence
    confidence, reasons = compute_confidence_score(result)

    # Decision
    should_abstain = confidence < threshold

    # Additional filters for high precision
    if require_high_precision and should_abstain:
        # Only abstain if we have strong signals
        has_strong_signal = (
            detect_explicit_abstention(response) or
            len(reasons) >= 2  # Multiple uncertainty signals
        )
        should_abstain = has_strong_signal

    # Generate abstention message
    if should_abstain:
        abstention_msg = (
            "ABSTAINED: This response shows low confidence. "
            "The model should not provide an answer. "
            f"Confidence score: {confidence:.2f}, Reasons: {'; '.join(reasons)}"
        )
    else:
        abstention_msg = ""

    has_hedging = any('hedging' in r for r in reasons)
    has_low_prob = any('low_token_probability' in r for r in reasons)

    return AbstentionDecision(
        prompt_id=prompt_id,
        model=model,
        should_abstain=should_abstain,
        confidence_score=confidence,
        abstention_reasons=reasons,
        has_hedging=has_hedging,
        has_low_token_prob=has_low_prob,
        original_response=response,
        abstention_message=abstention_msg
    )


def main():
    parser = argparse.ArgumentParser(description="Uncertainty-based abstention detector")
    parser.add_argument('--results', type=str, nargs='+', required=True, help='Result JSON files')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for abstention')
    parser.add_argument('--output', type=str, required=True, help='Output file')
    parser.add_argument('--high-precision', action='store_true', help='Require high precision (abstain less often)')

    args = parser.parse_args()

    # Load results
    print("Loading results...")
    all_results = []
    for result_file in args.results:
        print(f"  Loading {result_file}...")
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict) and 'results' in data:
                all_results.extend(data['results'])

    print(f"Loaded {len(all_results)} results")

    # Make abstention decisions
    print("\nAnalyzing confidence and making abstention decisions...")
    decisions = []
    for result in all_results:
        decision = make_abstention_decision(result, args.threshold, args.high_precision)
        decisions.append(decision)

    # Compute statistics
    total = len(decisions)
    abstained = sum(1 for d in decisions if d.should_abstain)
    with_hedging = sum(1 for d in decisions if d.has_hedging)
    with_low_prob = sum(1 for d in decisions if d.has_low_token_prob)

    avg_confidence = np.mean([d.confidence_score for d in decisions])

    # Save results
    output_data = {
        'parameters': {
            'threshold': args.threshold,
            'high_precision_mode': args.high_precision
        },
        'statistics': {
            'total_responses': total,
            'abstained': abstained,
            'abstention_rate': abstained / total if total > 0 else 0,
            'with_hedging': with_hedging,
            'with_low_token_prob': with_low_prob,
            'average_confidence': avg_confidence
        },
        'decisions': [
            {
                'prompt_id': d.prompt_id,
                'model': d.model,
                'should_abstain': d.should_abstain,
                'confidence_score': d.confidence_score,
                'abstention_reasons': d.abstention_reasons,
                'has_hedging': d.has_hedging,
                'has_low_token_prob': d.has_low_token_prob,
                'original_response': d.original_response,
                'abstention_message': d.abstention_message
            }
            for d in decisions
        ]
    }

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("ABSTENTION DETECTOR RESULTS")
    print("="*60)
    print(f"\nTotal responses: {total}")
    print(f"Abstained: {abstained} ({abstained/total:.1%})")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"\nResponses with hedging phrases: {with_hedging} ({with_hedging/total:.1%})")
    print(f"Responses with low token probability: {with_low_prob} ({with_low_prob/total:.1%})")

    print(f"\n✓ Results saved to {output_file}")

    # Show examples
    print("\nExamples of abstained responses:")
    count = 0
    for d in decisions:
        if d.should_abstain:
            print(f"\n{d.prompt_id} (confidence: {d.confidence_score:.3f}):")
            print(f"  Reasons: {', '.join(d.abstention_reasons)}")
            print(f"  Response: {d.original_response[:200]}...")
            count += 1
            if count >= 3:
                break


if __name__ == '__main__':
    main()
