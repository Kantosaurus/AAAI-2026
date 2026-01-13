# Mitigation Experiments (Phase D)

This directory contains implementations and evaluations of hallucination mitigation strategies.

## Overview

**Goal:** Test practical interventions to reduce hallucinations in security-related LLM outputs

**Mitigation Strategies:**
1. **RAG (Retrieval-Augmented Generation):** Ground responses with retrieved facts from NVD/MITRE
2. **Symbolic Checker:** Post-generation verification of CVE IDs against authoritative databases
3. **Uncertainty-based Abstention:** Detect low-confidence responses and abstain from answering

## Files

- `rag_grounding.py` - RAG implementation with local NVD metadata retrieval
- `symbolic_checker.py` - CVE/MITRE ID verification module
- `abstention_detector.py` - Uncertainty-based abstention strategies
- `evaluate_mitigations.py` - Comparative evaluation framework
- `build_retrieval_index.py` - Build local search index from NVD/MITRE data

## Setup

### 1. Build Retrieval Index

```bash
# Download NVD metadata (already done in data collection phase)
python build_retrieval_index.py \
    --nvd-data ../../data/gold/nvd_metadata.json \
    --output retrieval_index.pkl
```

### 2. Install Dependencies

```bash
pip install faiss-cpu sentence-transformers scikit-learn
```

## Usage

### RAG Grounding

```bash
python rag_grounding.py \
    --prompts ../../data/prompts/hallu-sec-benchmark.json \
    --index retrieval_index.pkl \
    --model claude-3-5-sonnet-20241022 \
    --output results/rag_results.json \
    --top-k 3
```

**How it works:**
1. For each prompt, retrieve top-K similar CVEs/documents from local index
2. Augment prompt with retrieved snippets as grounding context
3. Model generates response with explicit citation requirements
4. Compare hallucination rate with/without RAG

### Symbolic Checker

```bash
python symbolic_checker.py \
    --results ../../results/pilot/pilot_*.json \
    --nvd-index retrieval_index.pkl \
    --output results/symbolic_check_results.json
```

**How it works:**
1. Parse model response for CVE-YYYY-XXXXX patterns
2. Check each CVE ID against local NVD index
3. Flag or redact fabricated IDs
4. Optionally replace with "[UNKNOWN CVE]" placeholder

### Uncertainty-based Abstention

```bash
python abstention_detector.py \
    --results ../../results/pilot/pilot_*.json \
    --threshold 0.3 \
    --output results/abstention_results.json
```

**How it works:**
1. Analyze token probabilities (for local models with logprobs)
2. Detect hedging phrases ("I'm not sure", "may be", "possibly")
3. Compute confidence score
4. Abstain if confidence < threshold

### Comparative Evaluation

```bash
python evaluate_mitigations.py \
    --baseline ../../results/pilot/pilot_*.json \
    --rag results/rag_results.json \
    --symbolic results/symbolic_check_results.json \
    --abstention results/abstention_results.json \
    --annotations ../../annotations/adjudication/final_annotations.csv \
    --output results/mitigation_comparison.json
```

## Metrics

For each mitigation strategy:

- **Hallucination Reduction:** (baseline_rate - mitigated_rate) / baseline_rate
- **Precision:** Fraction of remaining claims that are correct
- **Recall:** Fraction of correct claims not filtered out
- **Abstention Rate:** Fraction of queries where system abstains
- **Utility Loss:** Fraction of correct answers incorrectly withheld

## Expected Results

**RAG Grounding:**
- Should reduce fabricated CVE citations significantly
- May introduce retrieval errors if index is incomplete
- Expected: 30-50% hallucination reduction

**Symbolic Checker:**
- Nearly 100% effective at catching fabricated CVE IDs
- No effect on other hallucination types
- Very low false positive rate
- Expected: Eliminates fabricated CVE citations

**Uncertainty Abstention:**
- Effectiveness depends on calibration quality
- May withhold many correct answers (high utility loss)
- Expected: 20-40% hallucination reduction, 10-30% abstention rate

## Safety Note

All mitigations operate on sanitized prompts and do not enable exploitation. Symbolic checker uses local NVD metadata only (no web queries at runtime).
