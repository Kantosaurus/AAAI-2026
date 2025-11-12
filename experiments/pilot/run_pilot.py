#!/usr/bin/env python3
"""
Pilot Run Script for LLM Hallucination Research
Runs prompts across multiple models with configurable sampling parameters
Logs all outputs, token probabilities, and metadata

Features:
- Multi-model support (API + local)
- Exponential backoff retry logic
- Rate limiting with token bucket
- Progress tracking
- Checkpoint/resume support
- Comprehensive error handling
"""

import json
import argparse
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
from dataclasses import dataclass, asdict
from collections import deque

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")

# Model interface imports
try:
    from anthropic import Anthropic, APIError, RateLimitError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class PromptResult:
    """Structured result for a single prompt execution"""
    prompt_id: str
    model: str
    model_version: str
    full_response: str
    tokens_used: Dict[str, int]
    token_logprobs: Optional[List[Dict]] = None
    sampling_params: Optional[Dict] = None
    timestamp: str = None
    elapsed_seconds: float = 0.0
    run_id: str = ""
    error: Optional[str] = None
    prompt_category: str = ""
    is_synthetic_probe: bool = False
    retry_count: int = 0

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.min_interval = 60.0 / requests_per_minute

    def acquire(self):
        """Acquire a token, waiting if necessary"""
        now = time.time()

        # Refill tokens based on time elapsed
        elapsed = now - self.last_update
        self.tokens = min(self.burst_size,
                         self.tokens + elapsed * (self.requests_per_minute / 60.0))
        self.last_update = now

        if self.tokens < 1:
            # Need to wait
            wait_time = (1 - self.tokens) * (60.0 / self.requests_per_minute)
            time.sleep(wait_time)
            self.tokens = 1

        self.tokens -= 1


class ModelRunner:
    """Base class for running models"""

    def __init__(self, model_name: str, temperature: float, seed: int,
                 max_retries: int = 3, rate_limiter: Optional[RateLimiter] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter
        self.run_id = self._generate_run_id()

    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{self.model_name}_{self.temperature}_{self.seed}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def run_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Run a single prompt with retry logic"""
        for retry in range(self.max_retries):
            try:
                # Rate limiting
                if self.rate_limiter:
                    self.rate_limiter.acquire()

                result = self._execute_prompt(prompt, prompt_id)
                result.retry_count = retry
                return result

            except Exception as e:
                if retry < self.max_retries - 1:
                    wait_time = (2 ** retry) * 1.0  # Exponential backoff
                    print(f"  Retry {retry + 1}/{self.max_retries} after {wait_time}s: {str(e)[:50]}")
                    time.sleep(wait_time)
                else:
                    # Final retry failed
                    return PromptResult(
                        prompt_id=prompt_id,
                        model=self.model_name,
                        model_version=self.model_name,
                        full_response="",
                        tokens_used={},
                        error=f"Failed after {self.max_retries} retries: {str(e)}",
                        timestamp=datetime.now().isoformat(),
                        run_id=self.run_id,
                        retry_count=retry + 1
                    )

    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt (to be implemented by subclasses)"""
        raise NotImplementedError


class ClaudeRunner(ModelRunner):
    """Runner for Anthropic Claude API"""

    def __init__(self, model_name: str, temperature: float, seed: int, api_key: str,
                 max_retries: int = 3, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(model_name, temperature, seed, max_retries, rate_limiter)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self.client = Anthropic(api_key=api_key)
        self.model_version = model_name

    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt via Claude API"""
        start_time = time.time()

        response = self.client.messages.create(
            model=self.model_version,
            max_tokens=2048,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        elapsed = time.time() - start_time

        return PromptResult(
            prompt_id=prompt_id,
            model=self.model_name,
            model_version=self.model_version,
            full_response=response.content[0].text,
            tokens_used={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens
            },
            token_logprobs=None,  # Claude doesn't expose logprobs
            sampling_params={
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": 2048
            },
            timestamp=datetime.now().isoformat(),
            elapsed_seconds=elapsed,
            run_id=self.run_id
        )


class GeminiRunner(ModelRunner):
    """Runner for Google Gemini API"""

    def __init__(self, model_name: str, temperature: float, seed: int, api_key: str,
                 max_retries: int = 3, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(model_name, temperature, seed, max_retries, rate_limiter)
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_version = model_name

    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt via Gemini API"""
        start_time = time.time()

        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=2048,
        )

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )

        elapsed = time.time() - start_time

        # Extract token counts if available
        tokens_used = {}
        if hasattr(response, 'usage_metadata'):
            tokens_used = {
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count
            }

        return PromptResult(
            prompt_id=prompt_id,
            model=self.model_name,
            model_version=self.model_version,
            full_response=response.text,
            tokens_used=tokens_used,
            token_logprobs=None,
            sampling_params={
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": 2048
            },
            timestamp=datetime.now().isoformat(),
            elapsed_seconds=elapsed,
            run_id=self.run_id
        )


class LocalModelRunner(ModelRunner):
    """Runner for local open-source models via transformers"""

    def __init__(self, model_name: str, temperature: float, seed: int,
                 model_path: Optional[str] = None, device: str = "cuda",
                 max_retries: int = 3, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(model_name, temperature, seed, max_retries, rate_limiter)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers/torch not installed. Run: pip install torch transformers")

        self.device = device
        model_path = model_path or model_name

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_version = model_path
        print(f"Model loaded on {device}")

        # Set random seed for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt on local model"""
        start_time = time.time()

        # Format prompt with chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback if chat template fails
                formatted_prompt = f"User: {prompt}\n\nAssistant:"
        else:
            # Generic format
            formatted_prompt = f"User: {prompt}\n\nAssistant:"

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to(self.device)

        # Generate with logprobs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract logprobs (top-5 for each token)
        logprobs_list = []
        if hasattr(outputs, 'scores') and outputs.scores:
            for i, score in enumerate(outputs.scores[:50]):  # Limit to first 50 tokens for size
                probs = torch.softmax(score[0], dim=-1)
                top_probs, top_indices = torch.topk(probs, k=5)
                logprobs_list.append({
                    "token_position": i,
                    "top_tokens": [self.tokenizer.decode([idx.item()]) for idx in top_indices],
                    "top_logprobs": [prob.item() for prob in top_probs.log()]
                })

        elapsed = time.time() - start_time

        return PromptResult(
            prompt_id=prompt_id,
            model=self.model_name,
            model_version=self.model_version,
            full_response=response_text,
            tokens_used={
                "input": inputs.input_ids.shape[1],
                "output": len(generated_ids),
                "total": inputs.input_ids.shape[1] + len(generated_ids)
            },
            token_logprobs=logprobs_list if logprobs_list else None,
            sampling_params={
                "temperature": self.temperature,
                "seed": self.seed,
                "max_new_tokens": 2048,
                "do_sample": self.temperature > 0
            },
            timestamp=datetime.now().isoformat(),
            elapsed_seconds=elapsed,
            run_id=self.run_id
        )


class PilotRunner:
    """Main pilot run orchestrator with checkpoint support"""

    def __init__(self, prompts_file: Path, output_dir: Path, config: Dict):
        self.prompts_file = prompts_file
        self.output_dir = output_dir
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load prompts
        with open(prompts_file, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
            self.prompts = benchmark['prompts']

        print(f"Loaded {len(self.prompts)} prompts from {prompts_file}")

        # Checkpoint file for resume support
        self.checkpoint_file = self.output_dir / "checkpoint.json"

    def load_checkpoint(self) -> Dict:
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed": {}, "last_model_index": 0}

    def save_checkpoint(self, checkpoint: Dict):
        """Save checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def run(self, resume: bool = False):
        """Execute pilot run with optional resume"""
        checkpoint = self.load_checkpoint() if resume else {"completed": {}, "last_model_index": 0}

        results = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "config": self.config,
                "total_prompts": len(self.prompts),
                "prompts_file": str(self.prompts_file),
                "resumed": resume
            },
            "runs": []
        }

        # Select prompts (subset if specified)
        num_prompts = self.config.get('num_prompts', len(self.prompts))
        prompts_to_run = self.prompts[:num_prompts]

        # Iterate through model configurations
        model_configs = self.config['models']
        start_idx = checkpoint.get('last_model_index', 0) if resume else 0

        for model_idx in range(start_idx, len(model_configs)):
            model_config = model_configs[model_idx]
            model_key = f"{model_config['name']}_temp{model_config.get('temperature', 0.0)}"

            # Skip if already completed
            if resume and model_key in checkpoint.get('completed', {}):
                print(f"\nSkipping {model_key} (already completed)")
                continue

            print(f"\n{'='*60}")
            print(f"Running model: {model_config['name']}")
            print(f"Temperature: {model_config.get('temperature', 0.0)}")
            print(f"{'='*60}")

            # Create runner with rate limiting
            rate_limiter = None
            if model_config['type'] in ['claude', 'gemini']:
                rpm = self.config.get('requests_per_minute', 60)
                rate_limiter = RateLimiter(requests_per_minute=rpm, burst_size=10)

            runner = self._create_runner(model_config, rate_limiter)

            # Run prompts with progress bar
            model_results = []

            iterator = enumerate(prompts_to_run, 1)
            if TQDM_AVAILABLE:
                iterator = tqdm(iterator, total=len(prompts_to_run),
                               desc=f"{model_config['name'][:30]}")

            for i, prompt_data in iterator:
                result = runner.run_prompt(
                    prompt=prompt_data['prompt'],
                    prompt_id=prompt_data['id']
                )

                # Add prompt metadata
                result.prompt_category = prompt_data['category']
                result.is_synthetic_probe = prompt_data['is_synthetic_probe']

                model_results.append(result.to_dict())

                # Progress update (if no tqdm)
                if not TQDM_AVAILABLE:
                    status = "ERROR" if result.error else "OK"
                    print(f"[{i}/{len(prompts_to_run)}] {prompt_data['id']}: {status} ({result.elapsed_seconds:.2f}s)")

            results['runs'].append({
                "model_config": model_config,
                "results": model_results,
                "completed_at": datetime.now().isoformat()
            })

            # Save intermediate results
            self._save_results(results, model_config['name'])

            # Update checkpoint
            checkpoint['completed'][model_key] = True
            checkpoint['last_model_index'] = model_idx + 1
            self.save_checkpoint(checkpoint)

        results['metadata']['end_time'] = datetime.now().isoformat()

        # Final save
        final_output = self.output_dir / f"pilot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Pilot run complete! Results saved to {final_output}")
        self._print_summary(results)

        # Clean up checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def _create_runner(self, model_config: Dict,
                      rate_limiter: Optional[RateLimiter] = None) -> ModelRunner:
        """Create appropriate runner based on model type"""
        model_type = model_config['type']
        model_name = model_config['name']
        temperature = model_config.get('temperature', 0.0)
        seed = self.config.get('seed', 42)
        max_retries = self.config.get('max_retries', 3)

        if model_type == 'claude':
            api_key = model_config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in config or environment")
            return ClaudeRunner(model_name, temperature, seed, api_key, max_retries, rate_limiter)

        elif model_type == 'gemini':
            api_key = model_config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in config or environment")
            return GeminiRunner(model_name, temperature, seed, api_key, max_retries, rate_limiter)

        elif model_type == 'local':
            device = model_config.get('device', 'cuda')
            model_path = model_config.get('model_path', model_name)
            return LocalModelRunner(model_name, temperature, seed, model_path, device,
                                   max_retries, rate_limiter)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _save_results(self, results: Dict, model_name: str):
        """Save intermediate results"""
        safe_name = model_name.replace('/', '_').replace(':', '_').replace('.', '_')
        output_file = self.output_dir / f"pilot_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  → Saved intermediate results to {output_file.name}")

    def _print_summary(self, results: Dict):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("PILOT RUN SUMMARY")
        print("="*60)

        total_errors = 0
        total_success = 0
        total_tokens = 0
        total_time = 0

        for run in results['runs']:
            model_name = run['model_config']['name']
            temp = run['model_config'].get('temperature', 0.0)
            model_results = run['results']

            total = len(model_results)
            errors = sum(1 for r in model_results if r.get('error'))
            success = total - errors

            avg_time = sum(r.get('elapsed_seconds', 0) for r in model_results if not r.get('error')) / success if success > 0 else 0
            tokens = sum(r.get('tokens_used', {}).get('total', 0) for r in model_results if not r.get('error'))

            total_errors += errors
            total_success += success
            total_tokens += tokens
            total_time += sum(r.get('elapsed_seconds', 0) for r in model_results)

            print(f"\nModel: {model_name} (temp={temp})")
            print(f"  Prompts: {total} ({success} success, {errors} errors)")
            print(f"  Avg time: {avg_time:.2f}s per prompt")
            print(f"  Total tokens: {tokens:,}")

        print(f"\n{'='*60}")
        print(f"OVERALL TOTALS")
        print(f"  Success: {total_success}")
        print(f"  Errors: {total_errors}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Total time: {total_time/3600:.2f} hours")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Run LLM hallucination pilot experiment")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--prompts', type=str,
                       help='Path to prompts file (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output directory (overrides config)')
    parser.add_argument('--num-prompts', type=int,
                       help='Number of prompts to run (for testing)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if available')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Override config with CLI args
    if args.prompts:
        prompts_file = Path(args.prompts)
    else:
        prompts_file = Path(config['prompts_file'])

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config['output_dir'])

    if args.num_prompts:
        config['num_prompts'] = args.num_prompts

    # Run pilot
    runner = PilotRunner(prompts_file, output_dir, config)
    runner.run(resume=args.resume)


if __name__ == '__main__':
    main()
