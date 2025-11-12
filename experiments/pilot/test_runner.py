#!/usr/bin/env python3
"""
Test script for run_pilot.py
Creates a minimal test configuration and validates functionality
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def create_test_config():
    """Create a minimal test configuration"""

    test_config = {
        "description": "Minimal test - validates run_pilot.py functionality",
        "prompts_file": "../../data/prompts/hallu-sec-benchmark.json",
        "output_dir": "../../results/pilot_test",
        "num_prompts": 5,  # Just 5 prompts for quick test
        "seed": 42,
        "max_retries": 2,
        "requests_per_minute": 30,
        "models": []
    }

    # Check for API keys
    import os

    # Add Claude if API key available
    if os.environ.get('ANTHROPIC_API_KEY'):
        test_config['models'].append({
            "name": "claude-3-5-sonnet-20241022",
            "type": "claude",
            "temperature": 0.0,
            "notes": "Test with Claude"
        })
        print("✓ Found ANTHROPIC_API_KEY - will test Claude")
    else:
        print("⚠ No ANTHROPIC_API_KEY - skipping Claude test")

    # Add Gemini if API key available
    if os.environ.get('GOOGLE_API_KEY'):
        test_config['models'].append({
            "name": "gemini-1.5-pro",
            "type": "gemini",
            "temperature": 0.0,
            "notes": "Test with Gemini"
        })
        print("✓ Found GOOGLE_API_KEY - will test Gemini")
    else:
        print("⚠ No GOOGLE_API_KEY - skipping Gemini test")

    if not test_config['models']:
        print("\n❌ ERROR: No API keys found!")
        print("Set at least one:")
        print("  export ANTHROPIC_API_KEY='your_key'")
        print("  export GOOGLE_API_KEY='your_key'")
        return None

    # Save config
    config_path = Path(__file__).parent / "config_test_auto.json"
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)

    print(f"\n✓ Created test config: {config_path}")
    print(f"  Models to test: {len(test_config['models'])}")
    print(f"  Prompts: {test_config['num_prompts']}")

    return config_path

def verify_benchmark():
    """Verify benchmark file exists"""
    benchmark_path = Path(__file__).parent.parent.parent / 'data' / 'prompts' / 'hallu-sec-benchmark.json'

    if not benchmark_path.exists():
        print(f"❌ Benchmark not found at {benchmark_path}")
        return False

    try:
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Benchmark loaded: {len(data['prompts'])} prompts")
        return True
    except Exception as e:
        print(f"❌ Error loading benchmark: {e}")
        return False

def run_test():
    """Run the test"""
    print("="*60)
    print("TESTING run_pilot.py")
    print("="*60)

    # Verify benchmark
    if not verify_benchmark():
        return 1

    # Create test config
    config_path = create_test_config()
    if not config_path:
        return 1

    # Run test
    print("\n" + "="*60)
    print("Running pilot test...")
    print("="*60)

    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "run_pilot.py", "--config", str(config_path)],
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print("\n✅ TEST PASSED!")
            print("\nCheck results in: results/pilot_test/")
            return 0
        else:
            print(f"\n❌ TEST FAILED (exit code {result.returncode})")
            return 1

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(run_test())
