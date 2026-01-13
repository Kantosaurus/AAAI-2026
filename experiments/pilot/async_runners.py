"""
Async Model Runners for LLM Hallucination Research

Provides async-compatible model runners for parallel API calls.
Supports Claude (native async), Gemini (thread wrapper), and local models.
"""

import asyncio
import time
import os
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from async_rate_limiter import AsyncRateLimiter

# Model interface imports
try:
    from anthropic import AsyncAnthropic, APIError, RateLimitError
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
    """Structured result for a single prompt execution."""
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
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class AsyncModelRunner(ABC):
    """Abstract base class for async model runners."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        seed: int,
        max_retries: int = 3,
        rate_limiter: Optional[AsyncRateLimiter] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter
        self.run_id = self._generate_run_id()

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{self.model_name}_{self.temperature}_{self.seed}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    async def run_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Run a single prompt with async retry logic."""
        for retry in range(self.max_retries):
            try:
                # Async rate limiting
                if self.rate_limiter:
                    await self.rate_limiter.acquire()

                result = await self._execute_prompt(prompt, prompt_id)
                result.retry_count = retry
                return result

            except Exception as e:
                if retry < self.max_retries - 1:
                    wait_time = (2 ** retry) * 1.0  # Exponential backoff
                    print(f"  Retry {retry + 1}/{self.max_retries} after {wait_time}s: {str(e)[:50]}")
                    await asyncio.sleep(wait_time)
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

    @abstractmethod
    async def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt (to be implemented by subclasses)."""
        pass


class AsyncClaudeRunner(AsyncModelRunner):
    """Async runner for Anthropic Claude API using AsyncAnthropic."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        seed: int,
        api_key: str,
        max_retries: int = 3,
        rate_limiter: Optional[AsyncRateLimiter] = None
    ):
        super().__init__(model_name, temperature, seed, max_retries, rate_limiter)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model_version = model_name

    async def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt via Claude API asynchronously."""
        start_time = time.time()

        response = await self.client.messages.create(
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


class AsyncGeminiRunner(AsyncModelRunner):
    """Async runner for Google Gemini API using thread wrapper."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        seed: int,
        api_key: str,
        max_retries: int = 3,
        rate_limiter: Optional[AsyncRateLimiter] = None
    ):
        super().__init__(model_name, temperature, seed, max_retries, rate_limiter)
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_version = model_name

    def _sync_generate(self, prompt: str) -> Any:
        """Synchronous generation for thread wrapper."""
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=2048,
        )
        return self.model.generate_content(prompt, generation_config=generation_config)

    async def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt via Gemini API using thread wrapper."""
        start_time = time.time()

        # Use thread executor for sync API
        response = await asyncio.to_thread(self._sync_generate, prompt)

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


class AsyncLocalModelRunner(AsyncModelRunner):
    """Async runner for local models via transformers using thread wrapper."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        seed: int,
        model_path: Optional[str] = None,
        device: str = "cuda",
        max_retries: int = 3,
        rate_limiter: Optional[AsyncRateLimiter] = None
    ):
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

    def _sync_generate(self, prompt: str, prompt_id: str) -> PromptResult:
        """Synchronous generation for thread wrapper."""
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
                formatted_prompt = f"User: {prompt}\n\nAssistant:"
        else:
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
            for i, score in enumerate(outputs.scores[:50]):  # Limit to first 50 tokens
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

    async def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt on local model using thread wrapper."""
        return await asyncio.to_thread(self._sync_generate, prompt, prompt_id)


def create_async_runner(
    model_config: Dict,
    rate_limiter: Optional[AsyncRateLimiter],
    seed: int,
    max_retries: int
) -> AsyncModelRunner:
    """
    Factory function to create appropriate async runner based on model type.

    Args:
        model_config: Model configuration dictionary
        rate_limiter: Optional rate limiter instance
        seed: Random seed for reproducibility
        max_retries: Maximum retry attempts

    Returns:
        Appropriate AsyncModelRunner subclass instance
    """
    model_type = model_config['type']
    model_name = model_config['name']
    temperature = model_config.get('temperature', 0.0)

    if model_type == 'claude':
        api_key = model_config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in config or environment")
        return AsyncClaudeRunner(
            model_name, temperature, seed, api_key, max_retries, rate_limiter
        )

    elif model_type == 'gemini':
        api_key = model_config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in config or environment")
        return AsyncGeminiRunner(
            model_name, temperature, seed, api_key, max_retries, rate_limiter
        )

    elif model_type == 'local':
        device = model_config.get('device', 'cuda')
        model_path = model_config.get('model_path', model_name)
        return AsyncLocalModelRunner(
            model_name, temperature, seed, model_path, device, max_retries, rate_limiter
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
