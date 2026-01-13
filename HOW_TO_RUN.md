# How to Run: Complete Step-by-Step Guide

This guide walks you through running the complete hallucination research pipeline from start to finish.

---

## Prerequisites

### System Requirements

**For API Models (Claude, Gemini):**
- Windows/Mac/Linux
- Python 3.8+
- Internet connection
- API keys

**For Local Models (Qwen, Mistral, Phi-3):**
- GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A100) OR
- CPU with 32GB+ RAM (much slower)
- 50GB disk space for models

### Software Installation

```bash
# 1. Clone/navigate to repository
cd C:\Users\wooai\Documents\AAAI-2026

# 2. Install Python dependencies
pip install -r experiments/pilot/requirements.txt

# 3. For interpretability (optional)
pip install transformer-lens circuitsvis

# 4. For analysis (optional)
pip install jupyter pandas matplotlib seaborn

# 5. For RAG (optional)
pip install sentence-transformers faiss-cpu
```

### API Keys Setup

If using Claude or Gemini:

```bash
# Option 1: Environment variables (Windows)
set ANTHROPIC_API_KEY=your_anthropic_key_here
set GOOGLE_API_KEY=your_google_key_here

# Option 1: Environment variables (Linux/Mac)
export ANTHROPIC_API_KEY=your_anthropic_key_here
export GOOGLE_API_KEY=your_google_key_here

# Option 2: Create .env file
# Create file: experiments/pilot/.env
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

---

## Phase 0: Verify Setup

```bash
cd experiments/pilot

# Check if everything is configured correctly
python validate_setup.py
```

Expected output:
```
✓ Python version: 3.x
✓ Required packages installed
✓ Benchmark file exists (393 prompts)
✓ API keys configured (if using APIs)
✓ GPU available (if using local models)
```

---

## Phase 1: Run Test Pilot (15-20 minutes)

**Purpose:** Verify everything works with a small subset before running the full pilot.

```bash
cd experiments/pilot

# Run small test with 2 models, 50 prompts
python run_pilot.py --config config_small_test.json
```

**What this does:**
- Tests Claude and Gemini on 50 prompts
- Takes ~15-20 minutes
- Costs ~$2-3
- Outputs to `results/pilot_test/`

**Check results:**
```bash
# List output files
ls ../../results/pilot_test/

# Quick check for errors
python -c "import json; data = json.load(open('../../results/pilot_test/pilot_claude-3-5-sonnet_temp0.0.json')); print(f'Responses: {len(data)}'); print(f'Errors: {sum(1 for r in data if r.get(\"error\"))}')"
```

---

## Phase 2: Run Full Pilot (10-14 hours)

**Purpose:** Collect complete dataset across all models and temperatures.

### Option A: Run Everything at Once

```bash
cd experiments/pilot

# Run full pilot: 5 models × 2 temps × 393 prompts = 3,930 responses
python run_pilot.py --config config_full_pilot.json
```

**Time:** 10-14 hours
**Cost:** $30-40 (API calls only)
**Output:** `results/pilot/pilot_*.json` (10 files)

### Option B: Run Models Separately (Recommended)

Safer approach - run each model separately so you can monitor:

```bash
cd experiments/pilot

# 1. Claude (2-3 hours)
python run_pilot.py \
    --prompts ../../data/prompts/hallu-sec-benchmark.json \
    --model claude-3-5-sonnet-20241022 \
    --temperature 0.0 \
    --output ../../results/pilot/

python run_pilot.py \
    --prompts ../../data/prompts/hallu-sec-benchmark.json \
    --model claude-3-5-sonnet-20241022 \
    --temperature 0.7 \
    --output ../../results/pilot/

# 2. Gemini (2-3 hours)
python run_pilot.py \
    --prompts ../../data/prompts/hallu-sec-benchmark.json \
    --model gemini-1.5-pro \
    --temperature 0.0 \
    --output ../../results/pilot/

python run_pilot.py \
    --prompts ../../data/prompts/hallu-sec-benchmark.json \
    --model gemini-1.5-pro \
    --temperature 0.7 \
    --output ../../results/pilot/

# 3. Local models (if you have GPU)
# Qwen2.5-14B (3-4 hours)
python run_pilot.py \
    --prompts ../../data/prompts/hallu-sec-benchmark.json \
    --model Qwen/Qwen2.5-14B-Instruct \
    --device cuda \
    --temperature 0.0 \
    --output ../../results/pilot/

# Mistral-7B (2-3 hours)
python run_pilot.py \
    --prompts ../../data/prompts/hallu-sec-benchmark.json \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --device cuda \
    --temperature 0.0 \
    --output ../../results/pilot/

# Phi-3-mini (2-3 hours)
python run_pilot.py \
    --prompts ../../data/prompts/hallu-sec-benchmark.json \
    --model microsoft/Phi-3-mini-128k-instruct \
    --device cuda \
    --temperature 0.0 \
    --output ../../results/pilot/
```

**Monitoring:**
- Progress bars show completion status
- Check `results/pilot/` for output files
- If interrupted: use `--resume` flag to continue

### Resume After Interruption

```bash
# If pilot was interrupted, resume from checkpoint
python run_pilot.py --config config_full_pilot.json --resume
```

---

## Phase 3: Prepare Annotations (30 minutes)

**Purpose:** Create annotation batches for human labeling.

```bash
cd annotations

# Create randomized batches for 2 annotators with 100% overlap
python prepare_annotation_batches.py \
    --results ../results/pilot/ \
    --output batches/ \
    --num-annotators 2 \
    --overlap 1.0 \
    --seed 42
```

**Output:**
- `batches/annotator_1_batch.csv` (3,930 rows)
- `batches/annotator_2_batch.csv` (3,930 rows)
- `batches/batch_summary.md`

**Next steps:**
1. Two annotators independently label CSV files
2. Use `rubric.md` as labeling guide
3. Fill in columns: hallucination_binary, types, severity, citation_correctness, notes

---

## Phase 4: Compute Inter-Annotator Agreement (5 minutes)

**Purpose:** Calculate Cohen's kappa and identify disagreements.

```bash
cd annotations

# After both annotators complete their batches
python compute_agreement.py \
    --annotations batches/annotator_1_batch.csv batches/annotator_2_batch.csv \
    --output agreement_report.json
```

**Output:**
- Cohen's kappa score
- Confusion matrix
- List of disagreements for adjudication

**If disagreements exist:**
1. Review disagreement cases
2. Adjudicator resolves conflicts
3. Create `adjudication/final_annotations.csv`

---

## Phase 5: Run Mitigations (1-2 hours)

### 5.1 Build Retrieval Index (for RAG)

```bash
cd experiments/mitigations

# Option 1: Use sample data (demo)
python build_retrieval_index.py \
    --nvd-data ../../data/gold/nvd_metadata.json \
    --output retrieval_index.pkl \
    --index-type semantic

# Option 2: Use simple index (no dependencies)
python build_retrieval_index.py \
    --nvd-data ../../data/gold/nvd_metadata.json \
    --output retrieval_index.pkl \
    --index-type simple
```

### 5.2 Run Symbolic Checker

```bash
cd experiments/mitigations

python symbolic_checker.py \
    --results ../../results/pilot/pilot_*.json \
    --nvd-list ../../data/cve_list_important.txt \
    --output results/symbolic_check_results.json \
    --sanitize-mode redact
```

**Output:** Detects fabricated CVE IDs, ~100% effective

### 5.3 Run Abstention Detector

```bash
cd experiments/mitigations

python abstention_detector.py \
    --results ../../results/pilot/pilot_*.json \
    --threshold 0.5 \
    --output results/abstention_results.json \
    --high-precision
```

**Output:** Identifies low-confidence responses

### 5.4 Run RAG Grounding (Demo)

```bash
cd experiments/mitigations

python rag_grounding.py \
    --prompts ../../data/prompts/hallu-sec-benchmark.json \
    --index retrieval_index.pkl \
    --model claude-3-5-sonnet-20241022 \
    --top-k 3 \
    --output results/rag_results.json \
    --max-prompts 50
```

**Note:** This is demo mode. For production, implement actual API calls.

### 5.5 Compare Mitigations

```bash
cd experiments/mitigations

python evaluate_mitigations.py \
    --baseline ../../results/pilot/pilot_*.json \
    --symbolic results/symbolic_check_results.json \
    --abstention results/abstention_results.json \
    --annotations ../../annotations/adjudication/final_annotations.csv \
    --output results/mitigation_comparison.json
```

**Output:** Comparison table with precision, recall, utility loss

---

## Phase 6: Interpretability Analysis (2-4 hours)

### 6.1 Select Cases for Analysis

```bash
cd experiments/interpretability

python select_cases_for_interp.py \
    --annotations ../../annotations/adjudication/final_annotations.csv \
    --results ../../results/pilot/ \
    --output selected_cases.json \
    --n-cases 30 \
    --model Qwen
```

### 6.2 Run Causal Tracing

```bash
cd experiments/interpretability

python causal_tracing.py \
    --cases selected_cases.json \
    --model Qwen/Qwen2.5-14B-Instruct \
    --output results/causal_traces/ \
    --n-cases 10 \
    --device cuda
```

**Output:** Layer-by-layer causal effects

### 6.3 Train Activation Probes

```bash
cd experiments/interpretability

python activation_probes.py \
    --cases selected_cases.json \
    --model Qwen/Qwen2.5-14B-Instruct \
    --output results/probes/ \
    --max-cases 100 \
    --device cuda
```

**Output:** Probe performance per layer (AUC scores)

---

## Phase 7: Integration Testing (30 minutes)

```bash
cd experiments/integration

python vuln_triage_workflow.py \
    --scenarios workflow_scenarios.json \
    --model claude-3-5-sonnet-20241022 \
    --use-symbolic-check \
    --nvd-list ../../data/cve_list_important.txt \
    --output results/vuln_triage_results.json
```

**Output:** Triage accuracy with/without mitigations

---

## Phase 8: Analysis & Visualization (1 hour)

### Convert to Jupyter Notebook

```bash
cd notebooks

# Convert Python template to notebook
jupyter nbconvert --to notebook analysis_template.py

# Launch Jupyter
jupyter notebook analysis_template.ipynb
```

### Run Analysis

Open the notebook and run all cells:
1. Load pilot results
2. Load annotations
3. Generate statistics
4. Create visualizations
5. Compare mitigations
6. Export summary

**Output:**
- Visualizations (charts, plots)
- `results/analysis_summary.json`

---

## Quick Command Reference

### Most Common Commands

```bash
# Test pilot (quick verification)
cd experiments/pilot && python run_pilot.py --config config_small_test.json

# Full pilot (main data collection)
cd experiments/pilot && python run_pilot.py --config config_full_pilot.json

# Symbolic checker (fastest mitigation)
cd experiments/mitigations && python symbolic_checker.py --results ../../results/pilot/pilot_*.json --output results/symbolic_check_results.json

# Analysis notebook
cd notebooks && jupyter notebook analysis_template.ipynb
```

### Check Progress

```bash
# Count completed responses
ls results/pilot/*.json | wc -l

# Check for errors in results
python -c "import json, glob; files = glob.glob('results/pilot/*.json'); total = sum(len(json.load(open(f))) for f in files); errors = sum(sum(1 for r in json.load(open(f)) if r.get('error')) for f in files); print(f'Total: {total}, Errors: {errors}')"

# View summary of one result file
python -c "import json; data = json.load(open('results/pilot/pilot_claude-3-5-sonnet_temp0.0.json')); print(json.dumps({'file': 'claude-3-5-sonnet_temp0.0', 'responses': len(data), 'models': len(set(r['model'] for r in data))}, indent=2))"
```

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r experiments/pilot/requirements.txt
```

### "CUDA out of memory"
```bash
# Use smaller model or CPU
python run_pilot.py --config config_full_pilot.json --device cpu

# Or reduce batch size in config
```

### "Rate limit exceeded"
Edit `config_full_pilot.json`:
```json
{
  "rate_limit": {
    "requests_per_minute": 30  // Reduce from 60
  }
}
```

### "API key not found"
```bash
# Check if keys are set
echo %ANTHROPIC_API_KEY%  # Windows
echo $ANTHROPIC_API_KEY   # Linux/Mac

# If empty, set them
set ANTHROPIC_API_KEY=your_key_here  # Windows
export ANTHROPIC_API_KEY=your_key_here  # Linux/Mac
```

### "Checkpoint file corrupt"
```bash
# Delete checkpoint and restart
rm experiments/pilot/checkpoint.json
python run_pilot.py --config config_full_pilot.json
```

---

## Expected Timeline

| Phase | Duration | Cost |
|-------|----------|------|
| Setup & Test | 30 min | $2-3 |
| Full Pilot | 10-14 hours | $30-40 |
| Annotation Prep | 30 min | $0 |
| Human Annotation | 2-3 days | Labor |
| IAA Computation | 5 min | $0 |
| Mitigations | 1-2 hours | $0 |
| Interpretability | 2-4 hours | $0 |
| Integration | 30 min | $0 |
| Analysis | 1-2 hours | $0 |
| **Total** | **~20 hours compute + 3 days human** | **~$35** |

---

## What to Do If Things Break

1. **Check logs:** Most scripts output detailed error messages
2. **Verify setup:** Run `validate_setup.py` again
3. **Check disk space:** Need 50GB+ free
4. **Check memory:** GPU needs 24GB, CPU needs 32GB
5. **Resume from checkpoint:** Use `--resume` flag
6. **Reduce scope:** Use `--max-prompts 100` for testing
7. **Contact:** Open issue with error message and system info

---

## Next Steps After Completion

1. **Write Final Report:** Summarize findings from analysis
2. **Create Presentation:** Use `analysis_summary.json` for slides
3. **Publish Results:** Share sanitized dataset and findings
4. **Deploy Mitigations:** Integrate symbolic checker into production

---

**Need help?** See [SETUP_GUIDE.md](experiments/pilot/SETUP_GUIDE.md) for detailed troubleshooting.
