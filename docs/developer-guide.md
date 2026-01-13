# Developer Guide

**Version:** 2.4
**Last Updated:** January 13, 2026
**Document ID:** DOC-DEV-001

---

## Table of Contents

1. [Development Overview](#1-development-overview)
2. [Development Environment](#2-development-environment)
3. [Code Architecture](#3-code-architecture)
4. [Contributing Code](#4-contributing-code)
5. [Adding New Features](#5-adding-new-features)
6. [Testing Guidelines](#6-testing-guidelines)
7. [Code Style Guide](#7-code-style-guide)
8. [Debugging](#8-debugging)
9. [Performance Optimization](#9-performance-optimization)
10. [Release Process](#10-release-process)

---

## 1. Development Overview

### 1.1 Project Philosophy

This project follows these principles:

| Principle | Implementation |
|-----------|----------------|
| **Modularity** | Each component is self-contained with clear interfaces |
| **Type Safety** | Comprehensive type hints and dataclasses |
| **Testability** | Isolated components with mock-friendly interfaces |
| **Documentation** | Docstrings, comments, and external docs |
| **Safety First** | All security content is sanitized |

### 1.2 Technology Stack

```
Language:        Python 3.8+ (3.10+ recommended)
Package Manager: pip with requirements.txt
Testing:         pytest, pytest-cov, pytest-asyncio
Linting:         ruff, mypy
Formatting:      black, isort
Documentation:   Markdown, docstrings
CI/CD:           GitHub Actions
Dashboard:       React 18, Vite, Recharts
```

### 1.3 Repository Structure

```
AAAI-2026/
├── data/                    # Data and datasets
│   ├── prompts/            # Benchmark prompts
│   ├── scripts/            # Data generation scripts
│   └── outputs/            # Generated data files
│
├── experiments/             # Main experiment code
│   ├── pilot/              # Pilot execution engine
│   ├── interpretability/   # Mechanistic analysis
│   ├── mitigations/        # Hallucination mitigations
│   └── integration/        # End-to-end workflows
│
├── annotations/             # Annotation pipeline
│   ├── rubric.md           # Labeling guidelines
│   └── *.py                # Annotation utilities
│
├── notebooks/               # Analysis notebooks
├── results/                 # Output storage
├── docs/                    # Documentation
├── tests/                   # Test suite
├── dashboard/               # React web dashboard
│   ├── src/components/     # Chart components
│   ├── src/hooks/          # Data loading hooks
│   └── src/utils/          # Data transformations
├── .github/workflows/       # CI/CD pipelines
│   ├── ci.yml              # Main CI workflow
│   └── pr.yml              # PR checks
└── scripts/                 # Utility scripts
```

---

## 2. Development Environment

### 2.1 Initial Setup

```bash
# Clone repository
git clone https://github.com/Kantosaurus/AAAI-2026.git
cd AAAI-2026

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r experiments/pilot/requirements.txt
pip install pytest pytest-cov mypy flake8 black isort

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 2.2 Environment Variables

```bash
# Required for API testing
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."

# Optional
export HF_TOKEN="hf_..."
export LOGLEVEL="DEBUG"
export CUDA_VISIBLE_DEVICES="0"
```

### 2.3 IDE Configuration

#### VS Code Settings (`.vscode/settings.json`)
```json
{
    "python.pythonPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm Configuration
1. Set Project Interpreter to `.venv`
2. Enable pytest as test runner
3. Configure flake8 as external tool
4. Enable type checking (mypy)

---

## 3. Code Architecture

### 3.1 Core Components

#### Pilot Runner (`experiments/pilot/run_pilot.py`)

```
run_pilot.py (626 lines)
├── PromptResult (dataclass)      # Lines 57-77
├── RateLimiter (class)           # Lines 80-106
├── ModelRunner (ABC)             # Lines 108-156
├── ClaudeRunner (class)          # Lines 163-206
├── GeminiRunner (class)          # Lines 209-261
├── LocalModelRunner (class)      # Lines 264-374
└── PilotRunner (class)           # Lines 396-500
```

#### Class Hierarchy

```
ModelRunner (ABC)
    ├── ClaudeRunner
    ├── GeminiRunner
    └── LocalModelRunner

PilotRunner
    ├── uses RateLimiter
    ├── uses CheckpointManager
    └── creates ModelRunner instances
```

### 3.2 Data Flow

```
Input:
  hallu-sec-benchmark.json (393 prompts)
       │
       ▼
Processing:
  PilotRunner
       │
       ├── RateLimiter.acquire()
       ├── ModelRunner.run_prompt()
       ├── PromptResult creation
       └── Checkpoint saving
       │
       ▼
Output:
  pilot_results_*.json
```

### 3.3 Key Design Patterns

#### Factory Pattern (Model Creation)
```python
def _create_runner(self, model_config: Dict) -> ModelRunner:
    model_type = model_config.get("type", "local")

    if model_type == "claude":
        return ClaudeRunner(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model_name=model_config["name"],
            temperature=model_config.get("temperature", 0.0),
            ...
        )
    elif model_type == "gemini":
        return GeminiRunner(...)
    else:
        return LocalModelRunner(...)
```

#### Template Method Pattern (Model Execution)
```python
class ModelRunner(ABC):
    def run_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Template method - handles retry logic."""
        self.rate_limiter.acquire()
        for retry in range(self.max_retries):
            try:
                return self._execute_prompt(prompt, prompt_id)  # Hook
            except Exception:
                time.sleep(2 ** retry)
        return self._create_error_result(prompt_id, "Max retries exceeded")

    @abstractmethod
    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Hook method - implemented by subclasses."""
        pass
```

#### Token Bucket Pattern (Rate Limiting)
```python
class RateLimiter:
    def __init__(self, requests_per_minute: int, burst_size: int = 10):
        self.rate = requests_per_minute / 60.0
        self.max_tokens = burst_size
        self.tokens = burst_size
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            # Refill tokens based on elapsed time
            # Wait if no tokens available
            pass
```

---

## 4. Contributing Code

### 4.1 Contribution Workflow

```
1. Fork repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request
6. Code review
7. Merge
```

### 4.2 Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/<name>` | `feature/add-gpt4-support` |
| Bug Fix | `fix/<name>` | `fix/rate-limiter-edge-case` |
| Documentation | `docs/<name>` | `docs/update-api-reference` |
| Refactor | `refactor/<name>` | `refactor/model-runner` |

### 4.3 Commit Messages

```
<type>(<scope>): <description>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples:**
```
feat(pilot): add OpenAI GPT-4 support

- Add GPT4Runner class
- Update config schema
- Add tests

Closes #123
```

```
fix(rate-limiter): handle edge case when tokens < 0

The rate limiter could get into negative token state under
high concurrency. Added floor check.
```

### 4.4 Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No security issues introduced
```

---

## 5. Adding New Features

### 5.1 Adding a New Model

**Step 1: Create Runner Class**

```python
# experiments/pilot/run_pilot.py

class OpenAIRunner(ModelRunner):
    """Runner for OpenAI API.

    Args:
        api_key: OpenAI API key
        **kwargs: Passed to ModelRunner
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=2048
        )

        elapsed = time.time() - start_time

        return PromptResult(
            prompt_id=prompt_id,
            model=self.model_name,
            model_version=self.model_name,
            full_response=response.choices[0].message.content,
            tokens_used={
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            },
            elapsed_seconds=elapsed,
            timestamp=datetime.now().isoformat()
        )
```

**Step 2: Update Factory**

```python
def _create_runner(self, model_config: Dict) -> ModelRunner:
    model_type = model_config.get("type", "local")

    if model_type == "openai":  # Add new type
        return OpenAIRunner(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name=model_config["name"],
            temperature=model_config.get("temperature", 0.0),
            seed=self.seed,
            max_retries=self.max_retries,
            rate_limiter=self.rate_limiter
        )
    # ... existing types
```

**Step 3: Update Configuration**

```json
{
  "models": [
    {
      "name": "gpt-4o",
      "type": "openai",
      "temperature": 0.0
    }
  ]
}
```

**Step 4: Add Tests**

```python
# tests/test_openai_runner.py

def test_openai_runner_creation():
    runner = OpenAIRunner(
        api_key="test_key",
        model_name="gpt-4o",
        temperature=0.0,
        ...
    )
    assert runner.model_name == "gpt-4o"

@pytest.mark.integration
def test_openai_runner_execution():
    # Requires real API key
    runner = OpenAIRunner(...)
    result = runner.run_prompt("Hello", "test_001")
    assert result.full_response
```

### 5.2 Adding a New Mitigation

**Step 1: Create Mitigation Class**

```python
# experiments/mitigations/new_mitigation.py

class NewMitigation:
    """Description of new mitigation strategy.

    Args:
        param1: First parameter
        param2: Second parameter
    """

    def __init__(self, param1: str, param2: float = 0.5):
        self.param1 = param1
        self.param2 = param2

    def apply(self, response: str) -> Dict[str, Any]:
        """Apply mitigation to response.

        Args:
            response: Model response text

        Returns:
            Dict with mitigation results
        """
        # Implementation
        return {
            "original": response,
            "mitigated": modified_response,
            "changes": []
        }
```

**Step 2: Integrate with Evaluation**

```python
# experiments/mitigations/evaluate_mitigations.py

def evaluate_new_mitigation(results_path, mitigation_params):
    mitigation = NewMitigation(**mitigation_params)

    # Apply to all results
    # Calculate metrics
    # Return evaluation
```

**Step 3: Add CLI**

```python
# experiments/mitigations/new_mitigation.py

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--param1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Run mitigation
```

### 5.3 Adding a New Analysis

```python
# notebooks/new_analysis.py

"""
New Analysis: [Description]

This notebook analyzes [what it analyzes].
"""

# %% [markdown]
# # Setup

# %%
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load data
with open('results/pilot/pilot_results.json') as f:
    data = json.load(f)

# %% [markdown]
# # Analysis

# %%
# Your analysis code here

# %% [markdown]
# # Results

# %%
# Summary and visualization
```

---

## 6. Testing Guidelines

### 6.1 Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_pilot_runner.py     # Unit tests
├── test_model_runners.py    # Runner tests
├── test_mitigations.py      # Mitigation tests
├── test_annotations.py      # Annotation tests
└── integration/             # Integration tests
    ├── test_full_pilot.py
    └── test_end_to_end.py
```

### 6.2 Writing Unit Tests

```python
# tests/test_pilot_runner.py

import pytest
from experiments.pilot.run_pilot import RateLimiter, PromptResult

class TestRateLimiter:
    def test_initialization(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        assert limiter.rate == 1.0
        assert limiter.max_tokens == 10

    def test_acquire_immediate(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        # First 10 should be immediate
        for _ in range(10):
            limiter.acquire()  # Should not block

    def test_acquire_rate_limited(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)
        import time
        start = time.time()
        limiter.acquire()  # Immediate
        limiter.acquire()  # Should wait ~1 second
        elapsed = time.time() - start
        assert elapsed >= 0.9

class TestPromptResult:
    def test_to_dict(self):
        result = PromptResult(
            prompt_id="test",
            model="test-model",
            model_version="1.0",
            full_response="Hello",
            tokens_used={"total": 10}
        )
        d = result.to_dict()
        assert d["prompt_id"] == "test"
        assert d["tokens_used"]["total"] == 10

    def test_from_dict(self):
        d = {
            "prompt_id": "test",
            "model": "test-model",
            "model_version": "1.0",
            "full_response": "Hello",
            "tokens_used": {"total": 10}
        }
        result = PromptResult.from_dict(d)
        assert result.prompt_id == "test"
```

### 6.3 Writing Integration Tests

```python
# tests/integration/test_full_pilot.py

import pytest
import json
from pathlib import Path

@pytest.fixture
def sample_config():
    return {
        "prompts_file": "tests/fixtures/sample_prompts.json",
        "output_dir": "tests/output",
        "num_prompts": 5,
        "models": [
            {"name": "test-model", "type": "mock", "temperature": 0.0}
        ]
    }

@pytest.mark.integration
def test_full_pilot_execution(sample_config, tmp_path):
    from experiments.pilot.run_pilot import PilotRunner

    sample_config["output_dir"] = str(tmp_path)

    runner = PilotRunner(
        prompts_file=Path(sample_config["prompts_file"]),
        output_dir=tmp_path,
        config=sample_config
    )

    runner.run()

    # Verify output
    output_files = list(tmp_path.glob("pilot_*.json"))
    assert len(output_files) > 0

    with open(output_files[0]) as f:
        results = json.load(f)

    assert len(results["runs"]) > 0
    assert len(results["runs"][0]["results"]) == 5
```

### 6.4 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=experiments --cov-report=html

# Run specific test file
pytest tests/test_pilot_runner.py

# Run specific test
pytest tests/test_pilot_runner.py::TestRateLimiter::test_initialization

# Run integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring GPU
pytest -m "not gpu"

# Skip tests requiring API keys
pytest -m "not api"

# Run with verbose output
pytest -v

# Run with output capture disabled
pytest -s
```

### 6.5 CI/CD Pipeline

The project uses GitHub Actions for continuous integration.

#### Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push to main, PR | Full CI: lint, test, type-check |
| `pr.yml` | Pull requests | Lightweight PR validation |

#### CI Jobs

```yaml
# .github/workflows/ci.yml
jobs:
  lint:        # ruff linting
  test:        # pytest with coverage
  type-check:  # mypy type checking
  syntax:      # Python syntax validation
```

#### Running CI Locally

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linting (ruff)
ruff check .

# Run type checking
mypy experiments/ --ignore-missing-imports

# Run all tests
pytest --cov=experiments

# Run syntax validation
python -m py_compile experiments/pilot/run_pilot.py
```

#### Test Markers

```python
# pytest.ini markers
@pytest.mark.slow       # Long-running tests
@pytest.mark.gpu        # Requires GPU
@pytest.mark.api        # Requires API keys
@pytest.mark.integration # Integration tests
```

### 6.5 Test Fixtures

```python
# tests/conftest.py

import pytest
import json
from pathlib import Path

@pytest.fixture
def sample_prompts():
    return [
        {
            "id": "prompt_001",
            "prompt": "Test prompt 1",
            "category": "test",
            "is_synthetic_probe": False
        },
        {
            "id": "prompt_002",
            "prompt": "Test prompt 2",
            "category": "test",
            "is_synthetic_probe": True
        }
    ]

@pytest.fixture
def sample_results():
    return {
        "metadata": {"total_prompts": 2},
        "runs": [{
            "model_config": {"name": "test-model"},
            "results": [
                {"prompt_id": "prompt_001", "full_response": "Response 1"},
                {"prompt_id": "prompt_002", "full_response": "Response 2"}
            ]
        }]
    }

@pytest.fixture
def mock_rate_limiter():
    """Rate limiter that doesn't actually limit."""
    class MockRateLimiter:
        def acquire(self):
            pass
    return MockRateLimiter()
```

---

## 7. Code Style Guide

### 7.1 Python Style

Follow PEP 8 with these additions:

```python
# Line length: 88 characters (black default)
# Use double quotes for strings
# Use trailing commas in multi-line structures

# Good
long_function_call(
    argument_one="value",
    argument_two="value",  # Trailing comma
)

# Bad
long_function_call(argument_one="value",
                   argument_two="value")
```

### 7.2 Type Hints

```python
from typing import Dict, List, Optional, Tuple, Any

# Function signatures
def process_results(
    results: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Tuple[int, float]:
    """Process results and return count and rate."""
    pass

# Class attributes
class Analyzer:
    results: List[Dict]
    threshold: float

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.results = []
```

### 7.3 Docstrings

Use Google-style docstrings:

```python
def analyze_response(
    response: str,
    ground_truth: Dict[str, Any],
    strict: bool = True
) -> Dict[str, Any]:
    """Analyze a model response against ground truth.

    This function compares the model response to the ground truth
    and identifies any hallucinations or errors.

    Args:
        response: The model's response text.
        ground_truth: Dictionary containing ground truth data with
            keys 'exists', 'description', and 'refs'.
        strict: If True, any discrepancy is flagged as error.
            If False, minor variations are allowed.

    Returns:
        Dictionary containing:
            - is_hallucination: Boolean indicating hallucination
            - errors: List of specific errors found
            - confidence: Confidence score (0.0 to 1.0)

    Raises:
        ValueError: If response is empty or ground_truth is malformed.

    Example:
        >>> result = analyze_response(
        ...     "CVE-2021-44228 is Log4Shell",
        ...     {"exists": True, "description": "Log4Shell"}
        ... )
        >>> print(result["is_hallucination"])
        False
    """
    pass
```

### 7.4 Imports

```python
# Standard library
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

# Local
from experiments.pilot.run_pilot import PilotRunner
from experiments.mitigations.symbolic_checker import SymbolicChecker
```

### 7.5 Error Handling

```python
# Use specific exceptions
class PilotError(Exception):
    """Base exception for pilot errors."""
    pass

class ModelLoadError(PilotError):
    """Raised when model fails to load."""
    pass

class APIError(PilotError):
    """Raised on API errors."""
    pass

# Handle exceptions appropriately
try:
    result = runner.run_prompt(prompt, prompt_id)
except APIError as e:
    logger.warning(f"API error for {prompt_id}: {e}")
    result = create_error_result(prompt_id, str(e))
except Exception as e:
    logger.error(f"Unexpected error for {prompt_id}: {e}")
    raise
```

---

## 8. Debugging

### 8.1 Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.debug("Detailed debug info")
logger.info("General info")
logger.warning("Warning message")
logger.error("Error occurred")
```

### 8.2 Debugging Techniques

#### Interactive Debugging
```python
# Insert breakpoint
import pdb; pdb.set_trace()

# Or use built-in (Python 3.7+)
breakpoint()

# In IPython/Jupyter
%debug  # Post-mortem debugging
```

#### Debugging API Calls
```python
# Enable verbose logging for HTTP requests
import http.client
http.client.HTTPConnection.debuglevel = 1

import logging
logging.getLogger("urllib3").setLevel(logging.DEBUG)
```

#### Debugging Memory Issues
```python
# Track memory usage
import tracemalloc

tracemalloc.start()
# ... your code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()

# For GPU memory
import torch
print(torch.cuda.memory_summary())
```

### 8.3 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM Error | Model too large | Reduce batch size, use FP16 |
| Rate Limit | Too many requests | Lower `requests_per_minute` |
| Timeout | Network/API slow | Increase timeout, add retry |
| Import Error | Missing dependency | `pip install <package>` |
| Type Error | Wrong argument type | Check type hints |

---

## 9. Performance Optimization

### 9.1 Profiling

```python
# Time profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)

# Memory profiling
from memory_profiler import profile

@profile
def memory_heavy_function():
    # ... code ...
    pass
```

### 9.2 Optimization Techniques

#### Batch Processing
```python
# Instead of
for prompt in prompts:
    result = process(prompt)

# Use batching
batch_size = 10
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i + batch_size]
    results = process_batch(batch)
```

#### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_lookup(cve_id: str) -> Dict:
    # Database lookup
    return lookup_nvd(cve_id)
```

#### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# For I/O-bound tasks (API calls)
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process, items))

# For CPU-bound tasks
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(compute, items))
```

### 9.3 GPU Optimization

```python
# Use FP16 for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Disable gradient computation for inference
with torch.no_grad():
    outputs = model.generate(...)

# Clear GPU cache
torch.cuda.empty_cache()
```

---

## 10. Release Process

### 10.1 Version Numbering

Follow Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### 10.2 Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number updated
- [ ] Code review completed
- [ ] Security review (if applicable)
- [ ] Performance testing (if applicable)

### 10.3 Release Steps

```bash
# 1. Ensure on main branch
git checkout main
git pull

# 2. Create release branch
git checkout -b release/v2.1.0

# 3. Update version
# Edit version in relevant files

# 4. Update CHANGELOG
# Add release notes

# 5. Commit changes
git add .
git commit -m "chore: prepare release v2.1.0"

# 6. Create tag
git tag -a v2.1.0 -m "Release v2.1.0"

# 7. Push
git push origin release/v2.1.0
git push origin v2.1.0

# 8. Create PR and merge
# 9. Create GitHub release
```

---

## Document Control

| Attribute | Value |
|-----------|-------|
| Document ID | DOC-DEV-001 |
| Version | 2.4 |
| Classification | Internal |
| Author | Research Team |
| Approval Date | January 13, 2026 |

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 2.4 | 2026-01-13 | Added CI/CD pipeline and dashboard sections | Research Team |
| 2.0 | 2026-01-13 | Complete developer guide | Research Team |
| 1.0 | 2025-11-06 | Initial developer guide | Research Team |
