#!/usr/bin/env python3
"""
Setup Validation Script
Checks that all dependencies and configurations are ready for pilot runs
"""

import sys
import json
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    required = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'anthropic': 'Anthropic Claude API',
        'google.generativeai': 'Google Gemini API',
    }

    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(module)

    return len(missing) == 0, missing

def check_benchmark():
    """Check if benchmark file exists and is valid"""
    print("\nChecking benchmark file...")
    benchmark_path = Path(__file__).parent.parent.parent / 'data' / 'prompts' / 'hallu-sec-benchmark.json'

    if not benchmark_path.exists():
        print(f"  ✗ Benchmark not found at {benchmark_path}")
        return False

    try:
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = len(data['prompts'])
        real = data['metadata']['real_prompts']
        synthetic = data['metadata']['synthetic_prompts']

        print(f"  ✓ Benchmark loaded: {total} prompts")
        print(f"    - Real: {real}")
        print(f"    - Synthetic: {synthetic}")
        print(f"    - Categories: {len(data['metadata']['categories'])}")
        return True
    except Exception as e:
        print(f"  ✗ Error loading benchmark: {e}")
        return False

def check_configs():
    """Check if config files exist"""
    print("\nChecking configuration files...")
    config_dir = Path(__file__).parent

    configs = [
        'config_full_pilot.json',
        'config_small_test.json'
    ]

    all_exist = True
    for config_file in configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"  ✓ {config_file}")
        else:
            print(f"  ✗ {config_file} - MISSING")
            all_exist = False

    return all_exist

def check_api_keys():
    """Check if API keys are configured"""
    print("\nChecking API keys...")
    import os

    keys = {
        'ANTHROPIC_API_KEY': 'Claude',
        'GOOGLE_API_KEY': 'Gemini'
    }

    all_set = True
    for env_var, service in keys.items():
        if os.environ.get(env_var):
            masked = os.environ[env_var][:10] + '...'
            print(f"  ✓ {service} API key ({masked})")
        else:
            print(f"  ✗ {service} API key - NOT SET (optional for local models only)")
            all_set = False

    return all_set

def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"    - VRAM: {gpu_mem:.1f}GB")

            if gpu_mem < 20:
                print(f"    ⚠ Warning: Low VRAM. May struggle with larger models.")
            return True
        else:
            print(f"  ✗ No GPU available (CUDA not available)")
            print(f"    - Can still run on CPU (slower)")
            return False
    except ImportError:
        print(f"  ✗ PyTorch not installed")
        return False

def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space...")
    import shutil

    repo_root = Path(__file__).parent.parent.parent
    total, used, free = shutil.disk_usage(repo_root)

    free_gb = free / 1e9
    print(f"  Available: {free_gb:.1f}GB")

    if free_gb < 50:
        print(f"  ⚠ Warning: Low disk space. Need ~50GB for models + results.")
        return False
    else:
        print(f"  ✓ Sufficient disk space")
        return True

def check_output_dirs():
    """Check if output directories exist"""
    print("\nChecking output directories...")
    repo_root = Path(__file__).parent.parent.parent

    dirs = [
        repo_root / 'results' / 'pilot',
        repo_root / 'results' / 'pilot_test'
    ]

    for directory in dirs:
        if directory.exists():
            print(f"  ✓ {directory.relative_to(repo_root)}")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  + Created {directory.relative_to(repo_root)}")

    return True

def main():
    print("="*60)
    print("PILOT RUN SETUP VALIDATION")
    print("="*60)

    checks = {
        'Dependencies': check_dependencies(),
        'Benchmark': check_benchmark(),
        'Configurations': check_configs(),
        'API Keys': (check_api_keys(), True),  # Optional
        'GPU': (check_gpu(), True),  # Optional
        'Disk Space': check_disk_space(),
        'Output Directories': check_output_dirs(),
    }

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    critical_pass = True
    optional_pass = True

    for check_name, result in checks.items():
        if isinstance(result, tuple):
            passed, critical = result
        else:
            passed = result
            critical = True

        status = "✓ PASS" if passed else "✗ FAIL"
        required = "(REQUIRED)" if critical else "(OPTIONAL)"

        print(f"{check_name:20} {status:10} {required}")

        if critical and not passed:
            critical_pass = False
        if not critical and not passed:
            optional_pass = False

    print("="*60)

    if critical_pass:
        print("✅ ALL CRITICAL CHECKS PASSED")
        if not optional_pass:
            print("⚠️  Some optional features not available (API keys or GPU)")
            print("   - Can still run local models on CPU")
            print("   - Skip API models if no keys configured")
        print("\n🚀 Ready to run pilot!")
        print("\nNext steps:")
        print("  1. Test run:  python run_pilot.py --config config_small_test.json")
        print("  2. Full run:  python run_pilot.py --config config_full_pilot.json")
        return 0
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nMissing critical components:")

        # Show what's missing
        if isinstance(checks['Dependencies'], tuple):
            success, missing = checks['Dependencies']
            if not success:
                print(f"  - Install: pip install {' '.join(missing)}")

        if not checks['Benchmark']:
            print(f"  - Generate benchmark first (should already exist)")

        if not checks['Configurations']:
            print(f"  - Config files missing in experiments/pilot/")

        print("\nSee SETUP_GUIDE.md for detailed instructions")
        return 1

if __name__ == '__main__':
    sys.exit(main())
