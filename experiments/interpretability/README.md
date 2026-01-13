# Interpretability Experiments (Phase D)

This directory contains notebooks and utilities for mechanistic interpretability analysis of hallucination cases.

## Overview

**Goal:** Identify internal mechanisms responsible for hallucinations in security-related LLM outputs

**Approach:**
1. **Causal Tracing:** Identify layers/heads that causally contribute to hallucinated outputs
2. **Activation Probing:** Train linear probes to detect hallucination-predictive features
3. **Attention Analysis:** Examine attention patterns during hallucination
4. **Intervention Experiments:** Test if manipulating activations can prevent hallucination

## Requirements

```bash
pip install transformers torch transformer-lens circuitsvis baukit
```

## Files

- `causal_tracing.py` - Causal tracing utilities for identifying critical components
- `activation_probes.py` - Train and evaluate linear probes on activations
- `attention_analysis.py` - Visualize and analyze attention patterns
- `intervention_experiments.py` - Test mitigation via activation patching
- `analysis_notebook.ipynb` - Interactive analysis notebook
- `selected_cases.json` - 20-30 hallucination cases selected for deep analysis

## Workflow

### 1. Select Cases for Analysis

```bash
python select_cases_for_interp.py \
    --annotations ../annotations/adjudication/final_annotations.csv \
    --results ../../results/pilot/ \
    --output selected_cases.json \
    --n-cases 30
```

### 2. Run Causal Tracing

```bash
python causal_tracing.py \
    --cases selected_cases.json \
    --model Qwen/Qwen2.5-14B-Instruct \
    --output results/causal_traces/
```

### 3. Train Activation Probes

```bash
python activation_probes.py \
    --cases selected_cases.json \
    --model Qwen/Qwen2.5-14B-Instruct \
    --output results/probes/
```

### 4. Analyze Results

Open `analysis_notebook.ipynb` in Jupyter to explore:
- Layer-by-layer contribution to hallucinations
- Which attention heads are most implicated
- Where "CVE exists" features emerge temporally
- Effectiveness of intervention strategies

## Expected Outputs

- **Causal trace plots:** Heatmaps showing which layers contribute to hallucinated tokens
- **Probe accuracies:** AUC curves for binary "hallucination" feature detection per layer
- **Attention visualizations:** Head-specific attention patterns
- **Intervention results:** Success rates for preventing hallucination via patching

## Key Questions to Answer

1. **Localization:** Which layers/heads are most responsible for hallucinations?
2. **Temporality:** At what token position does the model "decide" to hallucinate?
3. **Features:** Can we detect a "CVE exists" or "fabricate" feature in activations?
4. **Interventions:** Can we patch activations to reduce hallucinations without breaking generation?

## Safety Note

All experiments use sanitized prompts from the benchmark. No unsafe content is generated or analyzed.
