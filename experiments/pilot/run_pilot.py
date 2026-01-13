#!/usr/bin/env python3
"""
Pilot Run Script for LLM Hallucination Research

Runs prompts across multiple models with configurable sampling parameters.
Uses async execution for parallel API calls with significant performance improvements.

Features:
- Async multi-model support (API + local)
- Parallel execution with semaphore-based concurrency control
- Exponential backoff retry logic
- Rate limiting with async token bucket
- Progress tracking
- Checkpoint/resume support
- Comprehensive error handling
"""

import asyncio
import json
import argparse
from pathlib import Path

from async_pilot_runner import AsyncPilotRunner


async def main_async(
    prompts_file: Path,
    output_dir: Path,
    config: dict,
    resume: bool = False
) -> dict:
    """
    Async main function for pilot execution.

    Args:
        prompts_file: Path to prompts benchmark file
        output_dir: Output directory for results
        config: Configuration dictionary
        resume: Whether to resume from checkpoint

    Returns:
        Results dictionary
    """
    runner = AsyncPilotRunner(prompts_file, output_dir, config)
    results = await runner.run(resume=resume)
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM hallucination pilot experiment (async)"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to config JSON file'
    )
    parser.add_argument(
        '--prompts', type=str,
        help='Path to prompts file (overrides config)'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--num-prompts', type=int,
        help='Number of prompts to run (for testing)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from checkpoint if available'
    )
    parser.add_argument(
        '--max-concurrency', type=int, default=5,
        help='Maximum concurrent API calls (default: 5)'
    )

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

    # Set async settings from CLI if not in config
    if 'async_settings' not in config:
        config['async_settings'] = {}
    if args.max_concurrency:
        config['async_settings']['max_concurrency'] = args.max_concurrency

    # Run pilot with asyncio
    asyncio.run(main_async(prompts_file, output_dir, config, args.resume))


if __name__ == '__main__':
    main()
