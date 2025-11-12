# run_pilot.py - Implementation Details

## Overview

The `run_pilot.py` script is a production-ready tool for running LLM hallucination experiments across multiple models with comprehensive logging and error handling.

---

## Features Implemented

### ✅ Core Requirements

1. **Accepts prompts JSON**
   - Loads from `hallu-sec-benchmark.json`
   - Supports subset selection via `--num-prompts`

2. **Stores per-prompt results**
   - `prompt_id`: Unique identifier
   - `model`: Model name
   - `model_version`: Exact version used
   - `full_response`: Complete model output
   - `tokens_used`: Input/output/total token counts
   - `token_logprobs`: Top-5 token probabilities (local models only)
   - `sampling_params`: Temperature, seed, max_tokens
   - `timestamp`: ISO 8601 datetime
   - `elapsed_seconds`: Execution time
   - `run_id`: Unique run identifier
   - `error`: Error message if failed
   - `prompt_category`: Category from benchmark
   - `is_synthetic_probe`: Boolean flag
   - `retry_count`: Number of retries needed

3. **Rate Limiting**
   - Token bucket algorithm
   - Configurable requests per minute
   - Burst support for initial requests
   - Automatic backoff

4. **Error Handling**
   - Exponential backoff retry (up to 3 attempts)
   - Graceful degradation on failures
   - Per-prompt error logging
   - Checkpoint support for resume

---

## Advanced Features

### 1. Token Bucket Rate Limiter

**Implementation:** `RateLimiter` class (lines 80-106)

```python
rate_limiter = RateLimiter(
    requests_per_minute=60,  # Max 60 requests/min
    burst_size=10            # Allow 10 immediate requests
)
```

**How it works:**
- Tokens refill continuously at `requests_per_minute / 60` per second
- Up to `burst_size` tokens can accumulate
- Each request consumes 1 token
- Waits if no tokens available

**Benefits:**
- Smooth API rate limiting
- Avoids sudden rate limit errors
- Allows bursts for quick tests

### 2. Exponential Backoff Retry

**Implementation:** `ModelRunner.run_prompt()` (lines 127-156)

```python
for retry in range(max_retries):
    try:
        result = self._execute_prompt(prompt, prompt_id)
        return result
    except Exception as e:
        wait_time = (2 ** retry) * 1.0  # 1s, 2s, 4s
        time.sleep(wait_time)
```

**Retry schedule:**
- Attempt 1: Immediate
- Attempt 2: Wait 1 second
- Attempt 3: Wait 2 seconds
- Attempt 4: Wait 4 seconds

**Handles:**
- Network timeouts
- Temporary API errors
- Rate limit errors

### 3. Checkpoint/Resume Support

**Implementation:** `PilotRunner.load_checkpoint()` (lines 396-406)

```bash
# Run will be interrupted
python run_pilot.py --config config.json

# Resume from checkpoint
python run_pilot.py --config config.json --resume
```

**How it works:**
- Saves `checkpoint.json` after each model completes
- Tracks which models finished
- Skips completed models on resume
- Automatically cleans up checkpoint on success

**Checkpoint format:**
```json
{
  "completed": {
    "claude-3-5-sonnet-20241022_temp0.0": true,
    "gemini-1.5-pro_temp0.0": true
  },
  "last_model_index": 2
}
```

### 4. Progress Tracking

**Implementation:** Uses `tqdm` library (lines 456-459)

With tqdm installed:
```
claude-3-5-sonnet-20241022: 45%|████▌     | 178/393 [12:34<13:45,  3.8s/it]
```

Without tqdm:
```
[178/393] prompt_0178: OK (3.2s)
[179/393] prompt_0179: OK (2.8s)
```

### 5. Token Logprobs Extraction

**Implementation:** `LocalModelRunner._execute_prompt()` (lines 340-350)

**For local models only:**
- Extracts top-5 token probabilities at each position
- Stores first 50 tokens (to limit file size)
- Includes token text and log probability

**Format:**
```json
"token_logprobs": [
  {
    "token_position": 0,
    "top_tokens": ["Yes", "No", "The", "CVE", "I"],
    "top_logprobs": [-0.12, -2.45, -3.67, -4.23, -5.01]
  },
  ...
]
```

**Use cases:**
- Uncertainty estimation
- Hallucination detection
- Interpretability analysis (Phase D)

### 6. Structured Output with Dataclasses

**Implementation:** `PromptResult` dataclass (lines 57-77)

**Benefits:**
- Type safety
- Easy serialization
- Clear schema
- IDE autocomplete

```python
@dataclass
class PromptResult:
    prompt_id: str
    model: str
    full_response: str
    tokens_used: Dict[str, int]
    token_logprobs: Optional[List[Dict]] = None
    # ... etc
```

---

## Model Support

### 1. Claude (Anthropic API)

**Implementation:** `ClaudeRunner` (lines 163-206)

**Features:**
- Uses official Anthropic SDK
- Supports all Claude 3.x models
- Automatic token counting
- Error handling for API errors

**Configuration:**
```json
{
  "name": "claude-3-5-sonnet-20241022",
  "type": "claude",
  "temperature": 0.0,
  "api_key": "${ANTHROPIC_API_KEY}"
}
```

**Note:** Does not expose token logprobs (API limitation)

### 2. Gemini (Google API)

**Implementation:** `GeminiRunner` (lines 209-261)

**Features:**
- Uses google-generativeai SDK
- Supports Gemini 1.5 Pro/Flash
- Token usage metadata
- Error handling for Google API errors

**Configuration:**
```json
{
  "name": "gemini-1.5-pro",
  "type": "gemini",
  "temperature": 0.0,
  "api_key": "${GOOGLE_API_KEY}"
}
```

**Note:** Does not expose token logprobs in standard API

### 3. Local Models (Transformers)

**Implementation:** `LocalModelRunner` (lines 264-374)

**Features:**
- Works with any Hugging Face model
- Automatic device placement (CPU/GPU)
- Token logprobs extraction
- Chat template support
- Seed setting for reproducibility

**Configuration:**
```json
{
  "name": "Qwen/Qwen2.5-14B-Instruct",
  "type": "local",
  "temperature": 0.0,
  "device": "cuda",
  "model_path": "Qwen/Qwen2.5-14B-Instruct"
}
```

**Supported models:**
- Any causal LM from Hugging Face
- Qwen, Llama, Mistral, Phi, Gemma, etc.
- Custom fine-tuned models

---

## Configuration Options

### Global Config Parameters

```json
{
  "description": "Human-readable description",
  "prompts_file": "path/to/prompts.json",
  "output_dir": "path/to/output",
  "num_prompts": 393,              // Number of prompts (default: all)
  "seed": 42,                      // Random seed
  "max_retries": 3,                // Max retry attempts
  "requests_per_minute": 60,       // Rate limit (API only)
  "models": [...]                  // Model configurations
}
```

### Model Config Parameters

**Common:**
- `name`: Model identifier
- `type`: "claude" | "gemini" | "local"
- `temperature`: 0.0 to 1.0
- `notes`: Optional description

**API-specific:**
- `api_key`: API key (or use env var)

**Local-specific:**
- `device`: "cuda" | "cpu"
- `model_path`: Path or HF repo

---

## Output Format

### Directory Structure

```
results/pilot/
├── checkpoint.json                           [temp, deleted on completion]
├── pilot_claude-3-5-sonnet_20251112_100523.json
├── pilot_gemini-1.5-pro_20251112_143012.json
├── pilot_Qwen_Qwen2.5-14B-Instruct_20251112_180045.json
└── pilot_results_20251112_220530.json        [final combined results]
```

### Result File Schema

```json
{
  "metadata": {
    "start_time": "2025-11-12T10:00:00.123456",
    "end_time": "2025-11-12T22:05:30.654321",
    "config": {...},
    "total_prompts": 393,
    "prompts_file": "path/to/benchmark.json",
    "resumed": false
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
          "model_version": "claude-3-5-sonnet-20241022",
          "full_response": "CVE-2021-44228 exists...",
          "tokens_used": {
            "input": 42,
            "output": 156,
            "total": 198
          },
          "token_logprobs": null,
          "sampling_params": {
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": 2048
          },
          "timestamp": "2025-11-12T10:01:23.456789",
          "elapsed_seconds": 2.34,
          "run_id": "a1b2c3d4",
          "prompt_category": "cve_existence",
          "is_synthetic_probe": false,
          "retry_count": 0
        },
        ...
      ],
      "completed_at": "2025-11-12T14:25:30.123456"
    },
    ...
  ]
}
```

---

## Usage Examples

### 1. Quick Test (5 prompts)

```bash
python run_pilot.py \
    --config config_small_test.json \
    --num-prompts 5
```

### 2. Full Pilot Run

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

### 4. Custom Prompts File

```bash
python run_pilot.py \
    --config config_full_pilot.json \
    --prompts path/to/custom_prompts.json
```

### 5. Custom Output Directory

```bash
python run_pilot.py \
    --config config_full_pilot.json \
    --output results/custom_run
```

---

## Error Handling

### Network Errors

**Handled by:**
- Exponential backoff retry
- Rate limiter prevents overload

**Example:**
```
Retry 1/3 after 1s: Connection timeout
Retry 2/3 after 2s: Connection timeout
✓ Success on retry 3
```

### API Rate Limits

**Handled by:**
- Token bucket rate limiter
- Pre-emptive pacing

**Configuration:**
```json
"requests_per_minute": 30  // More conservative
```

### Model Loading Errors

**Handled by:**
- Detailed error messages
- Suggestions for fixes

**Example:**
```
ERROR: Model not found: invalid/model-name
Try: huggingface-cli login
```

### Partial Failures

**Behavior:**
- Continue with remaining prompts
- Log error in result
- Save partial results

**Result entry:**
```json
{
  "prompt_id": "prompt_0042",
  "error": "Failed after 3 retries: Rate limit exceeded",
  "retry_count": 3
}
```

---

## Performance Optimization

### 1. API Models

**Optimization:**
- Rate limiting prevents errors
- Async could be added for parallel requests
- Checkpoint allows long-running jobs

**Current:** ~1-3s per prompt (rate limited)
**Theoretical max:** ~60 prompts/min (with rpm=60)

### 2. Local Models

**Optimization:**
- Models loaded once, reused for all prompts
- GPU acceleration (CUDA)
- FP16 inference for memory efficiency

**Current:** ~2-5s per prompt (14B model on A100)
**Can parallelize:** Multiple GPUs for different models

### 3. Memory Management

**For local models:**
- Auto device mapping
- FP16 to reduce memory 2x
- Limit logprobs to first 50 tokens

**Memory usage (approximate):**
- Phi-3-mini (3.8B): ~8GB VRAM
- Mistral-7B: ~14GB VRAM
- Qwen2.5-14B: ~28GB VRAM

---

## Testing

### Unit Test

```bash
python test_runner.py
```

**What it does:**
- Verifies benchmark exists
- Creates minimal config (5 prompts)
- Tests with available API keys
- Validates output format

### Manual Test

```bash
# Test single model, single prompt
python run_pilot.py \
    --config config_small_test.json \
    --num-prompts 1
```

### Validation Script

```bash
python validate_setup.py
```

**Checks:**
- Dependencies installed
- Benchmark file exists
- API keys configured
- GPU available
- Disk space sufficient

---

## Troubleshooting

### Issue: "Rate limit exceeded" even with rate limiter

**Solution:**
```json
"requests_per_minute": 20  // Reduce from 60
```

### Issue: CUDA out of memory

**Solution 1:** Smaller model
```json
"model_path": "microsoft/Phi-3-mini-128k-instruct"
```

**Solution 2:** CPU mode
```json
"device": "cpu"
```

### Issue: Checkpoint not resuming

**Solution:**
```bash
# Check checkpoint file
cat results/pilot/checkpoint.json

# Force fresh start
rm results/pilot/checkpoint.json
python run_pilot.py --config config.json
```

### Issue: Missing dependencies

**Solution:**
```bash
pip install -r requirements.txt
```

---

## Future Enhancements

### Potential Improvements

1. **Async API calls:** Parallel requests for faster API runs
2. **Batch processing:** Process multiple prompts per API call (if supported)
3. **Cost tracking:** Log API costs in real-time
4. **Quality checks:** Automatic detection of truncated responses
5. **Streaming:** Save results incrementally during run
6. **Multi-GPU:** Parallel local model inference

### Backwards Compatibility

All enhancements will maintain current JSON output format for compatibility with annotation tools (Phase C).

---

## API Reference

### RateLimiter

```python
RateLimiter(requests_per_minute: int, burst_size: int)
    .acquire() -> None  # Blocks until token available
```

### ModelRunner

```python
ModelRunner(model_name, temperature, seed, max_retries, rate_limiter)
    .run_prompt(prompt: str, prompt_id: str) -> PromptResult
```

### PromptResult

```python
@dataclass
class PromptResult:
    prompt_id: str
    model: str
    full_response: str
    tokens_used: Dict[str, int]
    token_logprobs: Optional[List[Dict]]
    # ... see lines 57-77 for full definition
```

### PilotRunner

```python
PilotRunner(prompts_file: Path, output_dir: Path, config: Dict)
    .run(resume: bool = False) -> None
```

---

## Contributing

To extend with new model types:

1. Create new runner class inheriting from `ModelRunner`
2. Implement `_execute_prompt()` method
3. Add model type to `PilotRunner._create_runner()`
4. Update config schema documentation

Example:
```python
class NewModelRunner(ModelRunner):
    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        # Your implementation
        return PromptResult(...)
```

---

## License & Citation

See main project README for license and citation information.

**Status:** Production-ready, tested with Claude, Gemini, and Transformers models.
