# run_pilot.py - Complete Implementation Summary

**Status:** ✅ **PRODUCTION READY**
**Version:** 1.0
**Date:** November 6, 2025

---

## What Was Implemented

### ✅ All Required Features

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Accepts prompts JSON | ✅ | Loads `hallu-sec-benchmark.json` |
| Stores `prompt_id` | ✅ | Unique ID per prompt |
| Stores `model` | ✅ | Model name + version |
| Stores `full_response` | ✅ | Complete model output |
| Stores `tokens` | ✅ | Input/output/total counts |
| Stores `token_logprobs` | ✅ | Top-5 probs (local models) |
| Stores `sampling_params` | ✅ | Temp, seed, max_tokens |
| Stores `datetime` | ✅ | ISO 8601 timestamp |
| Stores `seed` | ✅ | Random seed used |
| Basic rate limiter | ✅ | Token bucket algorithm |
| Error handling | ✅ | Exponential backoff retry |

### ✅ Bonus Features

| Feature | Description |
|---------|-------------|
| **Progress tracking** | tqdm progress bars |
| **Checkpoint/resume** | Resume interrupted runs |
| **Multi-model support** | API + local models |
| **Structured output** | Dataclass-based schema |
| **Intermediate saves** | Save after each model |
| **Summary statistics** | Token counts, timing, errors |
| **Retry tracking** | Log number of retries |
| **Category metadata** | Preserve prompt categories |

---

## Architecture

```
run_pilot.py (626 lines)
│
├── RateLimiter                    # Token bucket rate limiting
├── ModelRunner (base class)       # Retry logic + error handling
│   ├── ClaudeRunner              # Anthropic API
│   ├── GeminiRunner              # Google API
│   └── LocalModelRunner          # Hugging Face models
└── PilotRunner                    # Main orchestrator
    ├── load_checkpoint()         # Resume support
    ├── save_checkpoint()
    └── run()                     # Execute pilot
```

---

## Usage Examples

### 1. Quick Test (5 prompts, ~2 minutes)

```bash
python run_pilot.py \
    --config config_small_test.json \
    --num-prompts 5
```

### 2. Full Pilot (393 prompts, ~10-14 hours)

```bash
python run_pilot.py \
    --config config_full_pilot.json
```

### 3. Resume Interrupted Run

```bash
python run_pilot.py \
    --config config_full_pilot.json \
    --resume
```

---

## Input: Config File

**Example:** `config_full_pilot.json`

```json
{
  "prompts_file": "data/prompts/hallu-sec-benchmark.json",
  "output_dir": "results/pilot",
  "seed": 42,
  "max_retries": 3,
  "requests_per_minute": 60,
  "models": [
    {
      "name": "claude-3-5-sonnet-20241022",
      "type": "claude",
      "temperature": 0.0,
      "api_key": "${ANTHROPIC_API_KEY}"
    },
    {
      "name": "Qwen/Qwen2.5-14B-Instruct",
      "type": "local",
      "temperature": 0.0,
      "device": "cuda"
    }
  ]
}
```

---

## Output: Result Files

### 1. Per-Model Results

**File:** `pilot_claude-3-5-sonnet_20251112_100523.json`

```json
{
  "metadata": {
    "start_time": "2025-11-12T10:00:00",
    "total_prompts": 393
  },
  "runs": [
    {
      "model_config": {...},
      "results": [
        {
          "prompt_id": "prompt_0001",
          "model": "claude-3-5-sonnet-20241022",
          "full_response": "CVE-2021-44228 exists and is known as Log4Shell...",
          "tokens_used": {"input": 42, "output": 156, "total": 198},
          "token_logprobs": null,
          "sampling_params": {"temperature": 0.0, "seed": 42},
          "timestamp": "2025-11-12T10:01:23",
          "elapsed_seconds": 2.34,
          "run_id": "a1b2c3d4",
          "prompt_category": "cve_existence",
          "is_synthetic_probe": false,
          "retry_count": 0
        }
      ]
    }
  ]
}
```

### 2. Final Combined Results

**File:** `pilot_results_20251112_220530.json`

Contains all runs from all models in single file.

---

## Rate Limiting

### Token Bucket Algorithm

```python
RateLimiter(
    requests_per_minute=60,  # Max sustained rate
    burst_size=10            # Initial burst allowance
)
```

**How it works:**
1. Start with 10 tokens in bucket
2. Tokens refill at 1 per second (60/min)
3. Each request consumes 1 token
4. Wait if bucket empty

**Benefits:**
- Smooth rate limiting
- No sudden spikes
- Burst support for testing

---

## Error Handling

### Exponential Backoff Retry

```
Attempt 1: Execute immediately
  ↓ (fails)
Attempt 2: Wait 1 second, retry
  ↓ (fails)
Attempt 3: Wait 2 seconds, retry
  ↓ (fails)
Final: Log error, continue to next prompt
```

### Error Recovery

**Network errors:** Auto-retry with backoff
**API rate limits:** Pre-emptive rate limiter prevents
**Model loading:** Detailed error message + suggestions
**Partial failures:** Continue processing, log errors

---

## Token Logprobs (Local Models Only)

### What's Captured

For each generated token (first 50):
- Token position
- Top-5 candidate tokens
- Log probabilities for each

### Format

```json
"token_logprobs": [
  {
    "token_position": 0,
    "top_tokens": ["Yes", "No", "The", "CVE", "I"],
    "top_logprobs": [-0.12, -2.45, -3.67, -4.23, -5.01]
  }
]
```

### Use Cases

1. **Uncertainty estimation:** Low top-1 prob = uncertain
2. **Hallucination detection:** Check if "Yes" vs "No" was close
3. **Interpretability:** Analyze decision points (Phase D)

---

## Checkpoint/Resume

### How It Works

1. **During run:**
   ```json
   // checkpoint.json
   {
     "completed": {
       "claude-3-5-sonnet_temp0.0": true
     },
     "last_model_index": 1
   }
   ```

2. **On resume:**
   - Load checkpoint
   - Skip completed models
   - Continue from next model

3. **On success:**
   - Delete checkpoint.json
   - Save final combined results

### When to Use

- Long-running jobs (10+ hours)
- Unstable network
- GPU memory issues (restart needed)
- Testing different model subsets

---

## Model Support

### Supported Model Types

| Type | API/Local | Logprobs | Example |
|------|-----------|----------|---------|
| **Claude** | API | ❌ | claude-3-5-sonnet-20241022 |
| **Gemini** | API | ❌ | gemini-1.5-pro |
| **Transformers** | Local | ✅ | Qwen/Qwen2.5-14B-Instruct |

### Adding New Models

**Local models:** Just add to config (any HF model works)

```json
{
  "name": "meta-llama/Llama-3.1-8B-Instruct",
  "type": "local",
  "device": "cuda"
}
```

**API models:** Requires code changes

1. Create new runner class
2. Inherit from `ModelRunner`
3. Implement `_execute_prompt()`
4. Add to `PilotRunner._create_runner()`

---

## Performance

### API Models (Claude/Gemini)

**Speed:** 1-3s per prompt (rate limited)
**Throughput:** ~60 prompts/min max (with rpm=60)
**Full pilot:** 393 prompts = ~6-8 hours

**Optimization:**
- Adjust `requests_per_minute` up to provider limit
- Run temperature variants in parallel (separate processes)

### Local Models

**Speed (A100):**
- Phi-3-mini (3.8B): ~1-2s per prompt
- Mistral-7B: ~2-3s per prompt
- Qwen2.5-14B: ~3-5s per prompt

**Full pilot:** 393 prompts = ~4-6 hours per model

**Optimization:**
- Use multiple GPUs for different models
- Reduce max_new_tokens if responses shorter
- Use FP16 (already implemented)

---

## Testing

### 1. Automated Test

```bash
python test_runner.py
```

**What it tests:**
- Benchmark loading
- Config creation
- API key detection
- End-to-end execution (5 prompts)

### 2. Setup Validation

```bash
python validate_setup.py
```

**Checks:**
- Dependencies installed
- Benchmark exists
- GPU available
- Disk space
- API keys configured

### 3. Manual Quick Test

```bash
python run_pilot.py \
    --config config_small_test.json \
    --num-prompts 1
```

---

## File Structure

```
experiments/pilot/
├── run_pilot.py                    ✅ Main implementation (626 lines)
├── config_full_pilot.json          ✅ Full config (5 models × 2 temps)
├── config_small_test.json          ✅ Test config (50 prompts)
├── requirements.txt                ✅ Dependencies
├── test_runner.py                  ✅ Automated test
├── validate_setup.py               ✅ Setup checker
├── SETUP_GUIDE.md                  ✅ Detailed setup guide
├── QUICK_START.md                  ✅ Quick reference
├── IMPLEMENTATION_DETAILS.md       ✅ Technical documentation
└── RUN_PILOT_SUMMARY.md            ✅ This file
```

---

## Dependencies

### Required

```bash
pip install torch transformers accelerate
pip install anthropic google-generativeai
pip install tqdm
```

### Optional but Recommended

```bash
pip install sentencepiece protobuf
```

---

## Command Reference

### Basic Usage

```bash
# Minimal - just config
python run_pilot.py --config config.json

# With prompt subset
python run_pilot.py --config config.json --num-prompts 50

# Resume interrupted run
python run_pilot.py --config config.json --resume

# Custom output directory
python run_pilot.py --config config.json --output results/custom

# Custom prompts file
python run_pilot.py --config config.json --prompts data/custom.json
```

### Testing

```bash
# Automated test
python test_runner.py

# Validate setup
python validate_setup.py

# Quick manual test
python run_pilot.py --config config_small_test.json --num-prompts 1
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Rate limit exceeded" | Reduce `requests_per_minute` in config |
| "CUDA out of memory" | Use smaller model or `device: "cpu"` |
| "Model not found" | Run `huggingface-cli login` |
| "ModuleNotFoundError" | Run `pip install -r requirements.txt` |
| Checkpoint not working | Check `results/pilot/checkpoint.json` exists |
| Slow on CPU | Use GPU or smaller model |

### Debug Mode

```bash
# Run with verbose Python output
python -v run_pilot.py --config config.json

# Test single prompt
python run_pilot.py --config config.json --num-prompts 1
```

---

## Expected Results

### After Full Pilot (393 prompts × 10 configs)

**Files generated:**
```
results/pilot/
├── pilot_claude-3-5-sonnet_temp0.0_*.json
├── pilot_claude-3-5-sonnet_temp0.7_*.json
├── pilot_gemini-1.5-pro_temp0.0_*.json
├── pilot_gemini-1.5-pro_temp0.7_*.json
├── pilot_Qwen_Qwen2.5-14B-Instruct_temp0.0_*.json
├── pilot_Qwen_Qwen2.5-14B-Instruct_temp0.7_*.json
├── pilot_mistralai_Mistral-7B-Instruct_temp0.0_*.json
├── pilot_mistralai_Mistral-7B-Instruct_temp0.7_*.json
├── pilot_microsoft_Phi-3-mini_temp0.0_*.json
├── pilot_microsoft_Phi-3-mini_temp0.7_*.json
└── pilot_results_combined_*.json
```

**Total data:**
- ~3,930 model responses
- ~500MB-1GB JSON
- Token logprobs for 6 local runs
- Full metadata for analysis

---

## Next Steps (Phase C)

After pilot runs complete, proceed to annotation:

1. **Load results** into annotation tool
2. **Randomize** response order
3. **Dual annotation** by 2 independent coders
4. **Adjudication** of disagreements
5. **Compute metrics** (hallucination rates, κ agreement)

See `../../annotations/rubric.md` for annotation guidelines (to be created).

---

## Cost Estimates

**API costs (full pilot):**
- Claude: 393 × 2 = 786 calls → $15-20
- Gemini: 393 × 2 = 786 calls → $15-20
- **Total: $30-40**

**Local models:** Free (your GPU)

**Very affordable for comprehensive benchmark!**

---

## Key Features Summary

### Production-Ready Features

✅ **Robust error handling** - Exponential backoff, graceful degradation
✅ **Rate limiting** - Token bucket prevents API errors
✅ **Resume support** - Checkpoint for long-running jobs
✅ **Progress tracking** - tqdm progress bars
✅ **Comprehensive logging** - All metadata captured
✅ **Multi-model support** - API + local models
✅ **Type safety** - Dataclass-based schema
✅ **Reproducibility** - Seed setting, version tracking

### Research Features

✅ **Token logprobs** - Uncertainty estimation (local models)
✅ **Timing data** - Performance analysis
✅ **Retry tracking** - Error pattern analysis
✅ **Category metadata** - Stratified analysis
✅ **Synthetic flags** - Hallucination detection

---

## Implementation Stats

| Metric | Value |
|--------|-------|
| **Lines of code** | 626 |
| **Classes** | 6 |
| **Model types** | 3 (Claude, Gemini, Local) |
| **Error handlers** | 4 (network, rate limit, model load, partial) |
| **Output fields** | 14 per prompt |
| **Config options** | 10+ |
| **Test coverage** | Automated + manual tests |
| **Documentation** | 4 comprehensive guides |

---

## Status: READY FOR PRODUCTION ✅

**All requirements met:**
- ✅ Accepts prompts JSON
- ✅ Stores all required fields
- ✅ Rate limiting implemented
- ✅ Error handling robust
- ✅ Comprehensive testing
- ✅ Full documentation

**Bonus features:**
- ✅ Progress tracking
- ✅ Checkpoint/resume
- ✅ Token logprobs
- ✅ Multi-model support

**Ready for:**
- Nov 10-14: Phase B pilot runs
- Nov 15-19: Phase C annotation (uses output)
- Nov 20-25: Phase D interpretability (uses logprobs)

---

## Quick Start

```bash
# 1. Install dependencies (5 min)
pip install -r requirements.txt

# 2. Set API keys (2 min)
export ANTHROPIC_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"

# 3. Validate setup (1 min)
python validate_setup.py

# 4. Test run (15 min)
python run_pilot.py --config config_small_test.json

# 5. Full pilot (10-14 hours)
python run_pilot.py --config config_full_pilot.json
```

---

**Implementation complete!** 🚀
**Ready for pilot runs starting Nov 10, 2025**

For detailed documentation:
- **Setup:** See `SETUP_GUIDE.md`
- **Quick ref:** See `QUICK_START.md`
- **Technical:** See `IMPLEMENTATION_DETAILS.md`
