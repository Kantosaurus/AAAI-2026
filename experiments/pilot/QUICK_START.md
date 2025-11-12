# Quick Start - Pilot Runs

## TL;DR

```bash
# 1. Setup
cd experiments/pilot
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"

# 2. Test run (50 prompts, fast)
python run_pilot.py --config config_small_test.json

# 3. Full pilot (393 prompts, ~10-14 hours)
python run_pilot.py --config config_full_pilot.json

# 4. Check results
ls ../../results/pilot/
```

---

## Your Model Configuration

| Model | Type | Size | Temps | Purpose |
|-------|------|------|-------|---------|
| **Claude 3.5 Sonnet** | Closed API | - | 0.0, 0.7 | High-capability baseline |
| **Gemini 1.5 Pro** | Closed API | - | 0.0, 0.7 | Alternative API comparison |
| **Qwen2.5-14B** | Local | 14B | 0.0, 0.7 | Interpretability (Phase D) |
| **Mistral-7B** | Local | 7B | 0.0, 0.7 | Medium scaling |
| **Phi-3-mini** | Local | 3.8B | 0.0, 0.7 | Small baseline |

**Total runs:** 5 models × 2 temps = 10 runs × 393 prompts = 3,930 model calls

---

## Why These Models?

### Closed API (You'll Run)
- **Claude 3.5 Sonnet:** Latest Anthropic, excellent reasoning
- **Gemini 1.5 Pro:** Google's best, long context

### Open Local (Suggested)

✅ **Qwen2.5-14B-Instruct** - Your choice
- Strong recent model (Oct 2024 release)
- Good for interpretability (Phase D)
- Excellent performance for size

✅ **Mistral-7B-Instruct-v0.3** - Recommended medium
- Industry standard 7B model
- Widely used in research
- Excellent performance/efficiency

✅ **Phi-3-mini-128k-instruct** - Recommended small
- Microsoft's efficient small model
- 3.8B params, runs on consumer GPUs
- Strong performance for size class

### Alternative Suggestions (If You Want Different Models)

**Alternative to Qwen2.5-14B:**
- Llama-3.1-8B-Instruct (Meta, very popular)
- Qwen2.5-7B-Instruct (smaller Qwen)

**Alternative to Mistral-7B:**
- Llama-3.1-8B-Instruct
- Gemma-7B-it (Google)

**Alternative to Phi-3-mini:**
- Gemma-2B-it (Google's 2B)
- TinyLlama-1.1B (very small baseline)

---

## Timeline (Nov 10-14)

### Day 5 (Nov 10) - Setup
- ✅ Benchmark created (393 prompts)
- ⏳ Install dependencies
- ⏳ Set up API keys
- ⏳ Download local models (if needed)

### Day 6 (Nov 11) - Test Run
```bash
python run_pilot.py --config config_small_test.json
```
- 50 prompts
- 2 models (Claude + Phi-3-mini)
- Verify output capture
- **Expected: 15-20 minutes**

### Day 7 (Nov 12) - Full Pilot
```bash
python run_pilot.py --config config_full_pilot.json
```
- 393 prompts
- All 5 models × 2 temps
- **Expected: 10-14 hours**
- Run overnight if needed

### Day 8 (Nov 13) - Sanity Checks
- Review outputs
- Flag fabricated citations
- Compute basic metrics
- Check for unsafe content

### Day 9 (Nov 14) - Freeze & Prepare
- Freeze pilot data
- Create annotator batches
- Prepare for Phase C

---

## Command Cheat Sheet

### Installation
```bash
pip install torch transformers accelerate
pip install anthropic google-generativeai
```

### Run Variants

```bash
# Test with 10 prompts only (very fast)
python run_pilot.py --config config_small_test.json --num-prompts 10

# Test with 50 prompts (recommended first run)
python run_pilot.py --config config_small_test.json

# Full pilot, all models
python run_pilot.py --config config_full_pilot.json

# Full pilot, but only 100 prompts (for faster iteration)
python run_pilot.py --config config_full_pilot.json --num-prompts 100
```

### Check Progress

```bash
# List output files
ls -lh ../../results/pilot/

# Count prompts processed
grep -c "prompt_id" ../../results/pilot/*.json

# Check for errors
grep "error" ../../results/pilot/*.json
```

---

## Hardware Requirements

### Minimum
- **GPU:** RTX 3090 (24GB VRAM)
- **RAM:** 32GB
- **Storage:** 50GB free
- **Internet:** For API calls and model downloads

### Recommended
- **GPU:** RTX 4090 or A100 (40GB+ VRAM)
- **RAM:** 64GB
- **Storage:** 100GB free
- **Internet:** Stable connection for APIs

### CPU-only (Slow)
- Possible but 10x-20x slower
- Not recommended for full pilot
- OK for small tests

---

## Cost Breakdown

| Item | Cost |
|------|------|
| Claude API (786 calls) | $15-20 |
| Gemini API (786 calls) | $15-20 |
| Local models | Free (your GPU) |
| **Total** | **$30-40** |

---

## Output Files

After completion:

```
results/pilot/
├── pilot_claude-3-5-sonnet_[timestamp].json
├── pilot_gemini-1.5-pro_[timestamp].json
├── pilot_Qwen_Qwen2.5-14B-Instruct_[timestamp].json
├── pilot_mistralai_Mistral-7B-Instruct_[timestamp].json
├── pilot_microsoft_Phi-3-mini_[timestamp].json
└── pilot_results_[timestamp].json  (combined)
```

Each JSON contains:
- All prompts processed
- Full model responses
- Token counts
- Logprobs (local models)
- Timing data
- Error logs

---

## Troubleshooting One-Liners

```bash
# API key not found
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."

# CUDA out of memory - use CPU
# Edit config: "device": "cpu"

# Rate limit - slow down
# Edit config: "api_delay": 2.0

# Model download fails - login
huggingface-cli login

# Check if models downloaded
ls ~/.cache/huggingface/hub/
```

---

## Safety Checklist

Before running:
- [ ] API keys in `.env` (not committed to git)
- [ ] GPU has enough VRAM
- [ ] Enough disk space (50GB+)
- [ ] Stable internet connection
- [ ] Time allocated (10-14 hours for full run)

After running:
- [ ] Check outputs for unsafe content
- [ ] Verify no exploit code generated
- [ ] Review any model errors
- [ ] Save results to backup location

---

## What's Next?

After pilot completes → **Phase C (Nov 15-19): Annotation**
- Annotate responses for hallucinations
- Compute inter-annotator agreement
- Calculate hallucination rates per model
- Select cases for interpretability (Phase D)

See `../annotations/rubric.md` for annotation guidelines.

---

**Questions?** See `SETUP_GUIDE.md` for detailed instructions.

**Ready?** Start with the test run:
```bash
python run_pilot.py --config config_small_test.json
```
