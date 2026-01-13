"""
Async Pilot Runner for LLM Hallucination Research

Orchestrates parallel API calls using asyncio with semaphore-based concurrency control.
Provides significant speedup over sequential execution while respecting rate limits.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from async_rate_limiter import AsyncRateLimiter
from async_runners import (
    AsyncModelRunner,
    PromptResult,
    create_async_runner
)


class AsyncPilotRunner:
    """
    Async pilot run orchestrator with semaphore-based concurrency control.

    Features:
    - Parallel API calls with configurable concurrency
    - Per-model rate limiting
    - Checkpoint/resume support
    - Progress tracking
    - Graceful error handling
    """

    def __init__(
        self,
        prompts_file: Path,
        output_dir: Path,
        config: Dict,
        max_concurrency: int = 5
    ):
        """
        Initialize async pilot runner.

        Args:
            prompts_file: Path to benchmark prompts JSON
            output_dir: Output directory for results
            config: Configuration dictionary
            max_concurrency: Maximum concurrent API calls (default: 5)
        """
        self.prompts_file = prompts_file
        self.output_dir = output_dir
        self.config = config
        self.max_concurrency = config.get('async_settings', {}).get('max_concurrency', max_concurrency)
        self.batch_size = config.get('async_settings', {}).get('batch_size', 10)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load prompts
        with open(prompts_file, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
            self.prompts = benchmark['prompts']

        print(f"Loaded {len(self.prompts)} prompts from {prompts_file}")
        print(f"Async mode: max_concurrency={self.max_concurrency}, batch_size={self.batch_size}")

        # Checkpoint file for resume support
        self.checkpoint_file = self.output_dir / "checkpoint.json"

        # Semaphore for concurrency control
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore (must be created in async context)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    def load_checkpoint(self) -> Dict:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed": {}, "last_model_index": 0}

    async def save_checkpoint_async(self, checkpoint: Dict) -> None:
        """Save checkpoint asynchronously."""
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(self.checkpoint_file, 'w') as f:
                await f.write(json.dumps(checkpoint, indent=2))
        else:
            # Fallback to sync write in thread
            await asyncio.to_thread(self._save_checkpoint_sync, checkpoint)

    def _save_checkpoint_sync(self, checkpoint: Dict) -> None:
        """Synchronous checkpoint save for fallback."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    async def run_prompt_with_semaphore(
        self,
        runner: AsyncModelRunner,
        prompt_data: Dict
    ) -> PromptResult:
        """
        Run a single prompt with semaphore control.

        Args:
            runner: Async model runner instance
            prompt_data: Prompt data dictionary

        Returns:
            PromptResult instance
        """
        async with self._get_semaphore():
            result = await runner.run_prompt(
                prompt=prompt_data['prompt'],
                prompt_id=prompt_data['id']
            )
            # Add prompt metadata
            result.prompt_category = prompt_data['category']
            result.is_synthetic_probe = prompt_data['is_synthetic_probe']
            return result

    async def run_model_batch(
        self,
        runner: AsyncModelRunner,
        prompts: List[Dict],
        model_name: str
    ) -> List[Dict]:
        """
        Run all prompts for a single model with parallel execution.

        Args:
            runner: Async model runner instance
            prompts: List of prompt data dictionaries
            model_name: Model name for progress display

        Returns:
            List of result dictionaries
        """
        tasks = [
            self.run_prompt_with_semaphore(runner, prompt_data)
            for prompt_data in prompts
        ]

        results = []
        errors = 0
        success = 0

        if TQDM_AVAILABLE:
            # Use async tqdm for progress
            for coro in async_tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"{model_name[:30]}"
            ):
                result = await coro
                if result.error:
                    errors += 1
                else:
                    success += 1
                results.append(result.to_dict())
        else:
            # Simple progress without tqdm
            completed = 0
            for coro in asyncio.as_completed(tasks):
                result = await coro
                completed += 1
                if result.error:
                    errors += 1
                else:
                    success += 1
                results.append(result.to_dict())
                print(f"[{completed}/{len(tasks)}] {result.prompt_id}: {'ERROR' if result.error else 'OK'}")

        print(f"  Batch complete: {success} success, {errors} errors")
        return results

    async def run(self, resume: bool = False) -> Dict:
        """
        Execute pilot run with parallel execution.

        Args:
            resume: Whether to resume from checkpoint

        Returns:
            Results dictionary
        """
        checkpoint = self.load_checkpoint() if resume else {"completed": {}, "last_model_index": 0}

        results = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "config": self.config,
                "total_prompts": len(self.prompts),
                "prompts_file": str(self.prompts_file),
                "resumed": resume,
                "async_mode": True,
                "max_concurrency": self.max_concurrency
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
            print(f"Running model: {model_config['name']} (async)")
            print(f"Temperature: {model_config.get('temperature', 0.0)}")
            print(f"{'='*60}")

            # Create rate limiter for API models
            rate_limiter = None
            if model_config['type'] in ['claude', 'gemini']:
                rpm = self.config.get('requests_per_minute', 60)
                rate_limiter = AsyncRateLimiter(requests_per_minute=rpm, burst_size=10)

            # Create async runner
            runner = create_async_runner(
                model_config=model_config,
                rate_limiter=rate_limiter,
                seed=self.config.get('seed', 42),
                max_retries=self.config.get('max_retries', 3)
            )

            # Run prompts in parallel
            model_results = await self.run_model_batch(
                runner=runner,
                prompts=prompts_to_run,
                model_name=model_config['name']
            )

            results['runs'].append({
                "model_config": model_config,
                "results": model_results,
                "completed_at": datetime.now().isoformat()
            })

            # Save intermediate results
            await self._save_results_async(results, model_config['name'])

            # Update checkpoint
            checkpoint['completed'][model_key] = True
            checkpoint['last_model_index'] = model_idx + 1
            await self.save_checkpoint_async(checkpoint)

        results['metadata']['end_time'] = datetime.now().isoformat()

        # Final save
        final_output = self.output_dir / f"pilot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(final_output, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            with open(final_output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Pilot run complete! Results saved to {final_output}")
        self._print_summary(results)

        # Clean up checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

        return results

    async def _save_results_async(self, results: Dict, model_name: str) -> None:
        """Save intermediate results asynchronously."""
        safe_name = model_name.replace('/', '_').replace(':', '_').replace('.', '_')
        output_file = self.output_dir / f"pilot_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        if AIOFILES_AVAILABLE:
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            await asyncio.to_thread(
                lambda: self._save_results_sync(results, output_file)
            )

        print(f"\n  -> Saved intermediate results to {output_file.name}")

    def _save_results_sync(self, results: Dict, output_file: Path) -> None:
        """Synchronous save for fallback."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def _print_summary(self, results: Dict) -> None:
        """Print summary statistics."""
        print("\n" + "="*60)
        print("PILOT RUN SUMMARY (ASYNC)")
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

            avg_time = (
                sum(r.get('elapsed_seconds', 0) for r in model_results if not r.get('error')) / success
                if success > 0 else 0
            )
            tokens = sum(
                r.get('tokens_used', {}).get('total', 0)
                for r in model_results if not r.get('error')
            )

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
        print(f"  Concurrency: {self.max_concurrency} parallel requests")
        print("="*60)
