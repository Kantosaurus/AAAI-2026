# Phase B Pilot Run Setup Guide

**Goal:** Run benchmark across 5 model configurations with 2 sampling temperatures each (10 total runs)

---

## Model Selection (Your Configuration)

### Closed API Models (High-Capability)
1. **Claude 3.5 Sonnet** (Anthropic)
   - Model: `claude-3-5-sonnet-20241022`
   - Temperatures: 0.0, 0.7
   - Purpose: High-capability closed model baseline

2. **Gemini 1.5 Pro** (Google)
   - Model: `gemini-1.5-pro`
   - Temperatures: 0.0, 0.7
   - Purpose: Alternative high-capability API comparison

### Open Local Models (For Interpretability & Scaling)

3. **Qwen2.5-14B-Instruct** (Alibaba)
   - Model: `Qwen/Qwen2.5-14B-Instruct`
   - Temperatures: 0.0, 0.7
   - Size: 14B parameters
   - Purpose: Main interpretability model (Phase D)
   - **Why Qwen:** Recent, strong performance, good for analysis

4. **Mistral-7B-Instruct-v0.3** (Mistral AI)
   - Model: `mistralai/Mistral-7B-Instruct-v0.3`
   - Temperatures: 0.0, 0.7
   - Size: 7B parameters
   - Purpose: Medium model scaling comparison
   - **Why Mistral:** Industry standard 7B, excellent performance/size ratio

5. **Phi-3-mini-128k-instruct** (Microsoft)
   - Model: `microsoft/Phi-3-mini-128k-instruct`
   - Temperatures: 0.0, 0.7
   - Size: 3.8B parameters
   - Purpose: Small model baseline
   - **Why Phi-3:** Efficient, strong for size, runs on consumer hardware

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
cd experiments/pilot

# Create virtual environment (if not already done)
python -m venv ../../.venv
source ../../.venv/bin/activate  # On Windows: ..\..\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Set Up API Keys

Create a `.env` file in the repo root:

```bash
# .env file (DO NOT commit to git)
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

Or export as environment variables:

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_key"
$env:GOOGLE_API_KEY="your_key"
```

### Step 3: Download Local Models (Optional - First Run)

Models will auto-download from Hugging Face on first run. To pre-download:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download models (one-time, ~30GB total)
models = [
    "Qwen/Qwen2.5-14B-Instruct",      # ~28GB
    "mistralai/Mistral-7B-Instruct-v0.3",  # ~14GB
    "microsoft/Phi-3-mini-128k-instruct"   # ~7.6GB
]

for model_name in models:
    print(f"Downloading {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
```

**Storage Requirements:**
- Qwen2.5-14B: ~28GB
- Mistral-7B: ~14GB
- Phi-3-mini: ~7.6GB
- **Total:** ~50GB

**GPU Requirements:**
- Minimum: 24GB VRAM (RTX 3090, RTX 4090, A5000)
- Recommended: 40GB+ (A100, H100)
- For CPU-only: Set `"device": "cpu"` in config (much slower)

---

## Running the Pilot

### Test Run (50 prompts, 2 models) - Nov 11 (Day 6)

**Purpose:** Verify setup and output capture

```bash
cd experiments/pilot

# Small test run
python run_pilot.py --config config_small_test.json

# This will run:
# - 50 prompts (subset)
# - Claude Sonnet (temp=0.0) - via API
# - Phi-3-mini (temp=0.0) - local
# - Output: results/pilot_test/
```

**Expected time:** ~15-20 minutes (depending on API rate limits)

**Verify:**
1. Check `results/pilot_test/` for JSON outputs
2. Review responses for safety (no exploit code generated)
3. Confirm logprobs captured for local model

### Full Pilot Run (393 prompts, all models) - Nov 12 (Day 7)

**Purpose:** Complete pilot dataset collection

```bash
# Full pilot run - ALL models, ALL prompts
python run_pilot.py --config config_full_pilot.json

# This will run:
# - 393 prompts (full benchmark)
# - 5 models × 2 temperatures = 10 total runs
# - ~3,930 total model calls
# - Output: results/pilot/
```

**Expected time:**
- API models (Claude/Gemini): ~6-8 hours (with rate limiting)
- Local models: ~4-6 hours (depends on GPU)
- **Total:** 10-14 hours

**Optimization tips:**
- Run API models overnight (rate limits)
- Run local models in parallel if you have multiple GPUs
- Use `--num-prompts 100` for faster subset testing

---

## Output Format

Each run produces a JSON file with this structure:

```json
{
  "metadata": {
    "start_time": "2025-11-12T10:00:00",
    "config": {...},
    "total_prompts": 393
  },
  "runs": [
    {
      "model_config": {
        "name": "claude-3-5-sonnet-20241022",
        "type": "claude",
        "temperature": 0.0
      },
      "results": [
        {
          "prompt_id": "prompt_0001",
          "model": "claude-3-5-sonnet-20241022",
          "full_response": "CVE-2021-44228 exists...",
          "tokens_used": {"input": 45, "output": 128, "total": 173},
          "token_logprobs": null,
          "sampling_params": {"temperature": 0.0, "seed": 42},
          "timestamp": "2025-11-12T10:01:23",
          "elapsed_seconds": 2.3,
          "run_id": "a1b2c3d4",
          "error": null,
          "prompt_category": "cve_existence",
          "is_synthetic_probe": false
        },
        ...
      ]
    }
  ]
}
```

**Key fields:**
- `full_response`: Complete model output
- `tokens_used`: Input/output token counts
- `token_logprobs`: Top-5 token probabilities per position (local models only)
- `is_synthetic_probe`: True if this is a hallucination probe (fake CVE, etc.)

---

## Troubleshooting

### Issue: API Rate Limits

**Error:** `Rate limit exceeded`

**Solution:**
```json
// Increase api_delay in config
"api_delay": 2.0  // 2 seconds between calls
```

### Issue: CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
1. **Reduce batch size** (models load one at a time, so this shouldn't happen)
2. **Use smaller models first:**
   ```bash
   # Run only Phi-3-mini to test
   python run_pilot.py --config config_small_test.json
   ```
3. **Use CPU mode:**
   ```json
   "device": "cpu"  // in model config
   ```

### Issue: Model Download Fails

**Error:** `Connection timeout` or `403 Forbidden`

**Solution:**
```bash
# Login to Hugging Face (for gated models)
huggingface-cli login

# Or set token
export HF_TOKEN="your_hf_token"
```

### Issue: Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'anthropic'`

**Solution:**
```bash
pip install -r requirements.txt
```

---

## Sanity Checks (Nov 13 - Day 8)

After pilot runs complete, run sanity checks:

### 1. CVE Citation Heuristic

```python
import json
import re

# Load results
with open('results/pilot/pilot_results_*.json', 'r') as f:
    results = json.load(f)

# Count CVE citations
for run in results['runs']:
    model_name = run['model_config']['name']
    responses = run['results']

    cve_citations = 0
    fabricated = 0

    for r in responses:
        # Find CVE patterns
        cves = re.findall(r'CVE-\d{4}-\d{4,7}', r.get('full_response', ''))
        cve_citations += len(cves)

        # Check if fabricated (synthetic probe)
        if r['is_synthetic_probe'] and cves:
            fabricated += len(cves)

    print(f"{model_name}:")
    print(f"  Total CVE citations: {cve_citations}")
    print(f"  Fabricated (on synthetic): {fabricated}")
    print(f"  Fabrication rate: {fabricated/cve_citations*100:.1f}%")
```

### 2. Response Length Analysis

```python
# Check response lengths
for run in results['runs']:
    model_name = run['model_config']['name']
    responses = run['results']

    lengths = [len(r.get('full_response', '')) for r in responses if not r.get('error')]
    avg_length = sum(lengths) / len(lengths)

    print(f"{model_name}: Avg response length = {avg_length:.0f} chars")
```

### 3. Error Rate

```python
# Check for errors
for run in results['runs']:
    model_name = run['model_config']['name']
    responses = run['results']

    total = len(responses)
    errors = sum(1 for r in responses if r.get('error'))

    print(f"{model_name}: {errors}/{total} errors ({errors/total*100:.1f}%)")
```

---

## Expected Pilot Results Structure

After completion, you should have:

```
results/pilot/
├── pilot_claude-3-5-sonnet_20251112_100000.json
├── pilot_gemini-1.5-pro_20251112_140000.json
├── pilot_Qwen_Qwen2.5-14B-Instruct_20251112_180000.json
├── pilot_mistralai_Mistral-7B-Instruct_20251112_220000.json
├── pilot_microsoft_Phi-3-mini_20251113_020000.json
└── pilot_results_20251113_060000.json  [combined]
```

Each file contains complete runs for that model at both temperatures.

---

## Next Steps (Phase C - Nov 15-19)

Once pilot is complete:

1. **Freeze pilot data** (Nov 14)
   - Copy results to `results/pilot_frozen/`
   - Create manifest with file hashes

2. **Prepare annotation** (Nov 15)
   - Split responses into annotator batches
   - Create annotation spreadsheet template
   - Train annotators on 20 examples

3. **Begin annotation** (Nov 16-17)
   - 2 annotators label independently
   - Daily check-ins for questions

---

## Cost Estimates

### API Costs (Approximate)

**Claude 3.5 Sonnet:**
- Input: $3 per 1M tokens
- Output: $15 per 1M tokens
- Est. per prompt: ~100 input + 300 output tokens
- Total: 393 prompts × 2 runs = 786 calls
- **Est. cost: $15-20**

**Gemini 1.5 Pro:**
- Input: $3.50 per 1M tokens
- Output: $10.50 per 1M tokens
- Similar usage to Claude
- **Est. cost: $15-20**

**Total API cost: $30-40** for full pilot

**Local models: Free** (your GPU)

---

## Questions?

Common questions:

**Q: Can I run on CPU only?**
A: Yes, but very slow (10x-20x). Set `"device": "cpu"` in config.

**Q: What if I don't have 50GB storage?**
A: Start with just Phi-3-mini (7.6GB) for testing. Add others later.

**Q: Can I use different models?**
A: Yes! Edit the config JSON. Any Hugging Face model works for local runs.

**Q: How do I pause/resume a run?**
A: Currently not supported. Run smaller batches with `--num-prompts`.

**Q: What if a model generates unsafe content?**
A: Immediately stop, document the issue, and report per safety protocol (see safety_policy_checklist.md).

---

## Safety Reminder

⚠️ **Before running:**

1. Review outputs for unsafe content (exploit code, attack instructions)
2. Do NOT execute any generated code
3. Flag any concerning outputs for manual review
4. Keep API keys secure (never commit to git)

---

**Ready to start?** Run the test config first to verify everything works!

```bash
python run_pilot.py --config config_small_test.json
```

Good luck with your pilot runs! 🚀
