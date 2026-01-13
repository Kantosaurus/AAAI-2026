# Unit Tests

This directory contains unit tests for the hallucination research codebase.

## Running Tests

### Run All Tests

```bash
# From project root
python -m pytest tests/

# Or using unittest
python -m unittest discover tests/
```

### Run Specific Test File

```bash
python -m pytest tests/test_io_utils.py
python -m pytest tests/test_symbolic_checker.py
```

### Run with Coverage

```bash
pip install pytest-cov
python -m pytest tests/ --cov=experiments --cov-report=html
```

## Test Structure

- `test_io_utils.py` - Tests for I/O utilities (loading/saving files)
- `test_symbolic_checker.py` - Tests for CVE verification and sanitization
- More tests to be added for other modules

## Writing New Tests

Follow these conventions:

1. **File naming:** `test_<module_name>.py`
2. **Class naming:** `Test<Functionality>`
3. **Method naming:** `test_<what_it_tests>`
4. **Use descriptive docstrings**

Example:
```python
class TestCVEExtraction(unittest.TestCase):
    """Test CVE ID extraction"""

    def test_extract_single_cve(self):
        """Test extracting a single CVE ID"""
        # Test code here
```

## Dependencies

```bash
pip install pytest pytest-cov
```

Or use unittest (built-in to Python, no installation needed).
