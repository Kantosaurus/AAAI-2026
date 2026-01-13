#!/usr/bin/env python3
"""
Analysis Template for Hallucination Research
Convert this to a Jupyter notebook for interactive analysis

To convert to notebook:
    jupyter nbconvert --to notebook analysis_template.py
"""

# %% [markdown]
# # LLM Hallucination Analysis - Security Domain
#
# This notebook analyzes hallucination patterns across models and prompts.
#
# **Sections:**
# 1. Data Loading
# 2. Descriptive Statistics
# 3. Hallucination Rate by Model
# 4. Hallucination Rate by Category
# 5. Mitigation Effectiveness
# 6. Interpretability Insights
# 7. Integration Workflow Results

# %% Setup
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %% [markdown]
# ## 1. Data Loading

# %% Load pilot results
def load_pilot_results(results_dir='../results/pilot/'):
    """Load all pilot result files"""
    results = []
    results_path = Path(results_dir)

    for json_file in results_path.glob('pilot_*.json'):
        print(f"Loading {json_file.name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict) and 'results' in data:
                results.extend(data['results'])

    df = pd.DataFrame(results)
    print(f"Loaded {len(df)} results")
    return df

# Load data
pilot_df = load_pilot_results()
pilot_df.head()

# %% Load annotations
def load_annotations(csv_file='../annotations/adjudication/final_annotations.csv'):
    """Load adjudicated annotations"""
    if not Path(csv_file).exists():
        print(f"Warning: Annotations file not found: {csv_file}")
        return None

    annot_df = pd.read_csv(csv_file)
    print(f"Loaded {len(annot_df)} annotations")
    return annot_df

annotations_df = load_annotations()
if annotations_df is not None:
    annotations_df.head()

# %% [markdown]
# ## 2. Descriptive Statistics

# %% Basic statistics
print("="*60)
print("PILOT RESULTS SUMMARY")
print("="*60)

print(f"\nTotal responses: {len(pilot_df)}")
print(f"Unique prompts: {pilot_df['prompt_id'].nunique()}")
print(f"Models tested: {pilot_df['model'].nunique()}")

print("\nModels:")
for model in pilot_df['model'].unique():
    count = len(pilot_df[pilot_df['model'] == model])
    print(f"  {model}: {count} responses")

# %% Response length distribution
pilot_df['response_length'] = pilot_df['full_response'].str.split().str.len()

plt.figure(figsize=(10, 6))
plt.hist(pilot_df['response_length'], bins=50, edgecolor='black')
plt.xlabel('Response Length (words)')
plt.ylabel('Frequency')
plt.title('Distribution of Response Lengths')
plt.axvline(pilot_df['response_length'].median(), color='red', linestyle='--', label=f'Median: {pilot_df["response_length"].median():.0f}')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Hallucination Rate by Model

# %% Merge with annotations
if annotations_df is not None:
    # Merge pilot results with annotations
    merged_df = pilot_df.merge(
        annotations_df,
        on=['prompt_id', 'model'],
        how='left'
    )

    print(f"\nMerged {len(merged_df)} records")
    print(f"Annotated: {merged_df['hallucination_binary'].notna().sum()}")
else:
    print("Skipping annotation-based analysis (no annotations loaded)")
    merged_df = pilot_df

# %% Hallucination rate by model
if 'hallucination_binary' in merged_df.columns:
    model_halluc = merged_df.groupby('model')['hallucination_binary'].agg(['mean', 'count'])
    model_halluc.columns = ['Hallucination Rate', 'Count']

    print("\nHallucination Rate by Model:")
    print(model_halluc.sort_values('Hallucination Rate', ascending=False))

    # Plot
    plt.figure(figsize=(12, 6))
    model_halluc['Hallucination Rate'].sort_values(ascending=False).plot(kind='barh')
    plt.xlabel('Hallucination Rate')
    plt.ylabel('Model')
    plt.title('Hallucination Rate by Model')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Hallucination Rate by Category

# %% By prompt category
if 'prompt_category' in merged_df.columns and 'hallucination_binary' in merged_df.columns:
    category_halluc = merged_df.groupby('prompt_category')['hallucination_binary'].agg(['mean', 'count'])
    category_halluc.columns = ['Hallucination Rate', 'Count']

    print("\nHallucination Rate by Category:")
    print(category_halluc.sort_values('Hallucination Rate', ascending=False))

    # Plot
    plt.figure(figsize=(10, 6))
    category_halluc['Hallucination Rate'].sort_values(ascending=False).plot(kind='bar')
    plt.xlabel('Category')
    plt.ylabel('Hallucination Rate')
    plt.title('Hallucination Rate by Prompt Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# %% By synthetic probe status
if 'is_synthetic_probe' in merged_df.columns and 'hallucination_binary' in merged_df.columns:
    probe_halluc = merged_df.groupby('is_synthetic_probe')['hallucination_binary'].agg(['mean', 'count'])
    probe_halluc.index = ['Real CVE', 'Synthetic Probe']
    probe_halluc.columns = ['Hallucination Rate', 'Count']

    print("\nHallucination Rate: Real vs Synthetic:")
    print(probe_halluc)

    # Plot
    plt.figure(figsize=(8, 6))
    probe_halluc['Hallucination Rate'].plot(kind='bar')
    plt.xlabel('Prompt Type')
    plt.ylabel('Hallucination Rate')
    plt.title('Hallucination Rate: Real CVEs vs Synthetic Probes')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Mitigation Effectiveness

# %% Load mitigation results
def load_mitigation_results(mitigation_file='../experiments/mitigations/results/mitigation_comparison.json'):
    """Load mitigation comparison results"""
    if not Path(mitigation_file).exists():
        print(f"Warning: Mitigation results not found: {mitigation_file}")
        return None

    with open(mitigation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mitigation_df = pd.DataFrame(data['metrics'])
    return mitigation_df

mitigation_df = load_mitigation_results()
if mitigation_df is not None:
    print("\nMitigation Strategy Comparison:")
    print(mitigation_df[['name', 'hallucination_reduction', 'precision', 'recall', 'utility_loss']])

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hallucination Reduction
    axes[0, 0].barh(mitigation_df['name'], mitigation_df['hallucination_reduction'])
    axes[0, 0].set_xlabel('Reduction Rate')
    axes[0, 0].set_title('Hallucination Reduction')

    # Precision
    axes[0, 1].barh(mitigation_df['name'], mitigation_df['precision'])
    axes[0, 1].set_xlabel('Precision')
    axes[0, 1].set_title('Precision of Remaining Responses')

    # Recall
    axes[1, 0].barh(mitigation_df['name'], mitigation_df['recall'])
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_title('Recall (Correct Responses Preserved)')

    # Utility Loss
    axes[1, 1].barh(mitigation_df['name'], mitigation_df['utility_loss'])
    axes[1, 1].set_xlabel('Utility Loss')
    axes[1, 1].set_title('Utility Loss (Correct Responses Withheld)')

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Interpretability Insights

# %% Load causal tracing results
def load_causal_traces(trace_file='../experiments/interpretability/results/causal_trace_results.json'):
    """Load causal tracing results"""
    if not Path(trace_file).exists():
        print(f"Warning: Causal trace results not found: {trace_file}")
        return None

    with open(trace_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

causal_data = load_causal_traces()
if causal_data is not None:
    print(f"\nCausal Tracing: {causal_data['n_cases']} cases analyzed")

    # Extract layer effects across all cases
    all_layer_effects = []
    for result in causal_data['results']:
        all_layer_effects.append(result['layer_effects'])

    # Average across cases
    avg_effects = np.mean(all_layer_effects, axis=0)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(avg_effects, marker='o')
    plt.xlabel('Layer Index')
    plt.ylabel('Average Causal Effect')
    plt.title('Layer-wise Causal Contribution to Hallucinations')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Find most important layers
    top_3_layers = np.argsort(avg_effects)[-3:][::-1]
    print(f"\nMost important layers: {top_3_layers}")
    for layer_idx in top_3_layers:
        print(f"  Layer {layer_idx}: Effect = {avg_effects[layer_idx]:.3f}")

# %% Load probe results
def load_probe_results(probe_file='../experiments/interpretability/results/probe_results.json'):
    """Load activation probe results"""
    if not Path(probe_file).exists():
        print(f"Warning: Probe results not found: {probe_file}")
        return None

    with open(probe_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return pd.DataFrame(data['results'])

probe_df = load_probe_results()
if probe_df is not None:
    print("\nActivation Probe Performance:")
    print(probe_df[['layer_idx', 'test_accuracy', 'test_auc']].sort_values('test_auc', ascending=False))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(probe_df['layer_idx'], probe_df['test_auc'], marker='o', label='Test AUC')
    plt.xlabel('Layer Index')
    plt.ylabel('AUC')
    plt.title('Hallucination Detection Probe Performance by Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Integration Workflow Results

# %% Load workflow results
def load_workflow_results(workflow_file='../experiments/integration/results/vuln_triage_results.json'):
    """Load integration workflow results"""
    if not Path(workflow_file).exists():
        print(f"Warning: Workflow results not found: {workflow_file}")
        return None

    with open(workflow_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

workflow_data = load_workflow_results()
if workflow_data is not None:
    print("\nVulnerability Triage Workflow Results:")
    print(f"  Accuracy: {workflow_data['metrics']['accuracy']:.1%}")
    print(f"  Scenarios with fabricated CVEs: {workflow_data['metrics']['scenarios_with_fabricated_cves']}")
    print(f"  Critical hallucination impact: {workflow_data['metrics']['critical_hallucination_impact']}")

# %% [markdown]
# ## Summary and Recommendations
#
# Based on the analysis:
#
# 1. **Hallucination Rates:**
#    - [Fill in after running on actual data]
#
# 2. **Most Vulnerable Categories:**
#    - [Fill in after running on actual data]
#
# 3. **Best Mitigation Strategy:**
#    - [Fill in after running on actual data]
#
# 4. **Interpretability Findings:**
#    - Key layers: [Fill in]
#    - Detection feasibility: [Fill in]
#
# 5. **Recommendations for Deployment:**
#    - Use symbolic checker for CVE verification
#    - Consider RAG for factual grounding
#    - Implement confidence-based abstention for critical decisions

# %% Export summary report
def export_summary(output_file='../results/analysis_summary.json'):
    """Export analysis summary"""
    summary = {
        'pilot_results': {
            'total_responses': len(pilot_df),
            'unique_prompts': pilot_df['prompt_id'].nunique(),
            'models_tested': pilot_df['model'].nunique()
        }
    }

    if 'hallucination_binary' in merged_df.columns:
        summary['hallucination_analysis'] = {
            'overall_rate': merged_df['hallucination_binary'].mean(),
            'by_model': model_halluc['Hallucination Rate'].to_dict()
        }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary exported to {output_path}")

export_summary()

# %%
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Review visualizations and metrics")
print("2. Identify patterns and outliers")
print("3. Prepare final report and slides")
