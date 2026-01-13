# Troubleshooting Guide

**Version:** 2.4
**Last Updated:** January 13, 2026
**Document ID:** DOC-TROUBLE-001

---

## Table of Contents

1. [Quick Diagnostics](#1-quick-diagnostics)
2. [Installation Issues](#2-installation-issues)
3. [Configuration Issues](#3-configuration-issues)
4. [Runtime Errors](#4-runtime-errors)
5. [API Errors](#5-api-errors)
6. [GPU and Memory Issues](#6-gpu-and-memory-issues)
7. [Data and Results Issues](#7-data-and-results-issues)
8. [Performance Issues](#8-performance-issues)
9. [Annotation Issues](#9-annotation-issues)
10. [Dashboard Issues](#10-dashboard-issues)
11. [Emergency Procedures](#11-emergency-procedures)

---

## 1. Quick Diagnostics

### 1.1 System Health Check

Run this script to diagnose common issues:

```bash
cd experiments/pilot
python validate_setup.py
```

**Expected Output:**
```
[1/6] Python version... OK (3.10.12)
[2/6] Dependencies... OK
[3/6] Benchmark file... OK (393 prompts)
[4/6] API keys... OK
[5/6] GPU availability... OK
[6/6] Disk space... OK

All checks passed!
```

### 1.2 Common Quick Fixes

| Symptom | Quick Fix |
|---------|-----------|
| Module not found | `pip install -r experiments/pilot/requirements.txt` |
| API key error | `export ANTHROPIC_API_KEY="your_key"` |
| CUDA not found | `export CUDA_VISIBLE_DEVICES=0` |
| Permission denied | `chmod +x script.py` |
| File not found | Verify you're in correct directory |

### 1.3 Log Collection

When reporting issues, collect these logs:

```bash
# System info
python --version
pip list | grep -E "anthropic|google|torch|transformers"
nvidia-smi

# Recent errors
cat pilot.log | tail -100

# Environment
env | grep -E "API_KEY|CUDA|HF" | sed 's/=.*/=***/'
```

---

## 2. Installation Issues

### 2.1 Python Version Mismatch

**Error:**
```
SyntaxError: invalid syntax
# or
ModuleNotFoundError: No module named 'dataclasses'
```

**Solution:**
```bash
# Check version
python --version

# Install correct version (3.8+)
# Using pyenv:
pyenv install 3.10.12
pyenv local 3.10.12

# Or using conda:
conda create -n hallu python=3.10
conda activate hallu
```

### 2.2 Dependency Installation Failures

**Error:**
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

**Solutions:**

```bash
# Upgrade pip
pip install --upgrade pip

# Install with specific index
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install without CUDA (CPU only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Error:**
```
ERROR: Could not build wheels for some packages
```

**Solutions:**
```bash
# Install build dependencies
pip install wheel setuptools

# On Windows, install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# On Linux
sudo apt-get install python3-dev build-essential
```

### 2.3 Virtual Environment Issues

**Error:**
```
-bash: .venv/bin/activate: No such file or directory
```

**Solution:**
```bash
# Create virtual environment
python -m venv .venv

# On Windows
python -m venv .venv
.\.venv\Scripts\activate
```

**Error:**
```
ModuleNotFoundError despite being installed
```

**Solution:**
```bash
# Verify using correct Python
which python
# Should show: /path/to/AAAI-2026/.venv/bin/python

# If not, reactivate
source .venv/bin/activate
```

---

## 3. Configuration Issues

### 3.1 Invalid JSON Configuration

**Error:**
```
json.decoder.JSONDecodeError: Expecting ',' delimiter
```

**Solution:**
```bash
# Validate JSON
python -c "import json; json.load(open('config.json'))"

# Common issues:
# - Trailing commas (remove them)
# - Missing quotes around strings
# - Single quotes instead of double quotes
```

**Correct JSON:**
```json
{
  "models": [
    {
      "name": "claude-3-5-sonnet",
      "type": "claude",
      "temperature": 0.0
    }
  ]
}
```

### 3.2 Missing Configuration Fields

**Error:**
```
KeyError: 'temperature'
```

**Solution:**
Ensure all required fields are present:

```json
{
  "name": "model-name",
  "type": "claude",
  "temperature": 0.0  // Required
}
```

### 3.3 Invalid File Paths

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '../../data/prompts/hallu-sec-benchmark.json'
```

**Solution:**
```bash
# Verify current directory
pwd
# Should be in experiments/pilot

# Verify file exists
ls -la ../../data/prompts/

# Use absolute paths in config
{
  "prompts_file": "/full/path/to/hallu-sec-benchmark.json"
}
```

---

## 4. Runtime Errors

### 4.1 Import Errors

**Error:**
```python
ImportError: cannot import name 'Anthropic' from 'anthropic'
```

**Solution:**
```bash
# Upgrade package
pip install --upgrade anthropic

# If still failing, reinstall
pip uninstall anthropic
pip install anthropic>=0.18.0
```

### 4.2 Type Errors

**Error:**
```python
TypeError: expected str, got tuple
```

**Solution:**
Check function signatures and return types. This often happens when:
- Function returns tuple but code expects single value
- Passing wrong type to function

```python
# Wrong
result = function()
print(result.text)

# Right
result, status = function()
print(result.text)
```

### 4.3 Timeout Errors

**Error:**
```
TimeoutError: Request timed out after 60 seconds
```

**Solution:**
```python
# Increase timeout in code
response = client.messages.create(
    ...,
    timeout=120  # Increase from default
)

# Or in config
{
  "timeout": 120,
  "max_retries": 5
}
```

### 4.4 Checkpoint Resume Failures

**Error:**
```
KeyError: 'completed' in checkpoint
```

**Solution:**
```bash
# Delete corrupted checkpoint and restart
rm results/pilot/checkpoint.json
python run_pilot.py --config config.json

# Or manually fix checkpoint
cat results/pilot/checkpoint.json
# Should be: {"completed": {}, "last_model_index": 0}
```

---

## 5. API Errors

### 5.1 Authentication Errors

**Error (Claude):**
```
anthropic.AuthenticationError: Invalid API key
```

**Solution:**
```bash
# Verify key is set
echo $ANTHROPIC_API_KEY

# Test key
python -c "
import anthropic
client = anthropic.Anthropic()
print('Key valid')
"

# Common issues:
# - Key has leading/trailing whitespace
# - Key is expired
# - Key doesn't have required permissions
```

**Error (Gemini):**
```
google.api_core.exceptions.Unauthenticated: API key not valid
```

**Solution:**
```bash
# Verify key
echo $GOOGLE_API_KEY

# Test key
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-pro')
print('Key valid')
"
```

### 5.2 Rate Limit Errors

**Error:**
```
anthropic.RateLimitError: Rate limit exceeded
```

**Solution:**
```json
// Reduce rate in config
{
  "requests_per_minute": 30  // Reduce from 60
}
```

```python
# Or increase delay programmatically
rate_limiter = RateLimiter(
    requests_per_minute=20,
    burst_size=5
)
```

### 5.3 Model Not Found

**Error:**
```
NotFoundError: Model 'claude-3-5-sonnet-20241022' not found
```

**Solution:**
```bash
# Check model name spelling
# Correct: claude-3-5-sonnet-20241022
# Wrong: claude-3.5-sonnet, claude-35-sonnet

# List available models
python -c "
import anthropic
client = anthropic.Anthropic()
# Check Anthropic docs for current model names
"
```

### 5.4 Request Too Large

**Error:**
```
BadRequestError: Request too large
```

**Solution:**
```json
// Reduce max_tokens
{
  "max_tokens": 1024  // Reduce from 2048
}
```

---

## 6. GPU and Memory Issues

### 6.1 CUDA Not Available

**Error:**
```python
RuntimeError: CUDA is not available
```

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 6.2 Out of Memory (GPU)

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

```python
# Solution 1: Use smaller model
{
  "model_path": "microsoft/Phi-3-mini-128k-instruct"  # 3.8B instead of 14B
}

# Solution 2: Use CPU
{
  "device": "cpu"
}

# Solution 3: Use FP16
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16  # Half precision
)

# Solution 4: Clear cache
import torch
torch.cuda.empty_cache()
```

### 6.3 Out of Memory (RAM)

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
```bash
# Check available memory
free -h

# Process results in batches
python -c "
import json
# Load one file at a time, not all at once
"

# Increase swap space (Linux)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 6.4 GPU Memory Leak

**Symptom:** GPU memory usage increases over time

**Solution:**
```python
# Clear cache after each model
torch.cuda.empty_cache()

# Delete model when done
del model
del tokenizer
torch.cuda.empty_cache()

# Use context manager
with torch.no_grad():
    outputs = model.generate(...)
```

---

## 7. Data and Results Issues

### 7.1 Corrupted JSON

**Error:**
```
json.decoder.JSONDecodeError: Expecting value
```

**Solution:**
```bash
# Check file
head -20 results/pilot/pilot_results.json
tail -20 results/pilot/pilot_results.json

# Validate JSON
python -c "import json; json.load(open('file.json'))"

# If corrupted, restore from backup or re-run
```

### 7.2 Missing Results

**Symptom:** Fewer results than expected

**Solution:**
```bash
# Count results
python -c "
import json
from pathlib import Path

for f in Path('results/pilot').glob('pilot_*.json'):
    data = json.load(open(f))
    for run in data.get('runs', []):
        n = len(run.get('results', []))
        print(f'{f.name}: {n} results')
"

# Check for errors
grep -l '"error"' results/pilot/*.json
```

### 7.3 Inconsistent Data

**Symptom:** Results don't match expected format

**Solution:**
```python
# Validate result structure
required_fields = [
    "prompt_id", "model", "full_response",
    "tokens_used", "timestamp"
]

for result in results:
    missing = [f for f in required_fields if f not in result]
    if missing:
        print(f"Missing: {missing} in {result['prompt_id']}")
```

### 7.4 Benchmark File Issues

**Error:**
```
KeyError: 'prompts' in benchmark file
```

**Solution:**
```bash
# Verify benchmark structure
python -c "
import json
data = json.load(open('data/prompts/hallu-sec-benchmark.json'))
print(f\"Keys: {data.keys()}\")
print(f\"Prompts: {len(data.get('prompts', []))}\")
"
```

---

## 8. Performance Issues

### 8.1 Slow API Responses

**Symptom:** API calls take 30+ seconds

**Solutions:**
```bash
# Check network latency
ping api.anthropic.com

# Use regional endpoint if available
# Check API status page
```

### 8.2 Slow Local Model Inference

**Symptom:** Local model takes 30+ seconds per prompt

**Solutions:**
```python
# Ensure using GPU
model = model.to("cuda")

# Use FP16
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Reduce max_tokens
generation_config = GenerationConfig(max_new_tokens=512)
```

### 8.3 Progress Stalling

**Symptom:** Progress bar stuck

**Solutions:**
```bash
# Check if process is running
ps aux | grep python

# Check for network issues
ping api.anthropic.com

# Check logs
tail -f pilot.log

# Kill and resume
Ctrl+C
python run_pilot.py --config config.json --resume
```

### 8.4 Disk Space Running Out

**Error:**
```
OSError: No space left on device
```

**Solution:**
```bash
# Check disk space
df -h

# Clean up
rm -rf ~/.cache/huggingface/hub/models--*  # Be careful!
rm -f results/pilot/checkpoint.json
rm -rf __pycache__

# Archive old results
tar -czvf results_archive.tar.gz results/
rm -rf results/old_runs/
```

---

## 9. Annotation Issues

### 9.1 Batch Preparation Failures

**Error:**
```
FileNotFoundError: No pilot results found
```

**Solution:**
```bash
# Verify results exist
ls results/pilot/*.json

# Check path in command
python prepare_annotation_batches.py \
    --results ../results/pilot/  # Correct path
```

### 9.2 Agreement Calculation Errors

**Error:**
```
ValueError: Annotation lists have different lengths
```

**Solution:**
```bash
# Check annotation files have same entries
wc -l annotations/batches/annotator_*.csv

# Verify prompt_id alignment
python -c "
import pandas as pd
a1 = pd.read_csv('annotator_1.csv')
a2 = pd.read_csv('annotator_2.csv')
print(f'A1: {len(a1)}, A2: {len(a2)}')
print(f'Common: {len(set(a1.prompt_id) & set(a2.prompt_id))}')
"
```

### 9.3 Low Agreement Scores

**Symptom:** Cohen's kappa < 0.4

**Solutions:**
1. Review rubric with annotators
2. Discuss disagreement examples
3. Clarify edge cases
4. Provide more training examples
5. Adjudicate disagreements with third annotator

---

## 10. Dashboard Issues

### 10.1 npm install Failures

**Error:**
```
npm ERR! code ENOENT
npm ERR! syscall open
```

**Solution:**
```bash
# Verify in correct directory
cd dashboard

# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### 10.2 Development Server Not Starting

**Error:**
```
Error: Cannot find module 'vite'
```

**Solution:**
```bash
cd dashboard
npm install vite --save-dev
npm run dev
```

### 10.3 Data Not Loading

**Symptom:** Dashboard shows empty state even with results file

**Solutions:**
```bash
# Check results file format
python -c "
import json
data = json.load(open('results/pilot/pilot_results.json'))
print('Keys:', data.keys())
print('Has metadata:', 'metadata' in data)
print('Has runs:', 'runs' in data)
"

# Expected structure:
# - metadata: {}
# - runs: []
```

### 10.4 Charts Not Rendering

**Symptom:** Empty chart containers

**Solutions:**
```bash
# Check browser console for errors
# Press F12 in browser

# Verify data transformation
# Check that dataTransforms.js handles your data format

# Try with URL parameter
http://localhost:5173?results=/absolute/path/to/results.json
```

### 10.5 Upload Not Working

**Symptom:** File upload does nothing

**Solutions:**
```javascript
// Check file type - must be .json
// Check browser console for errors

// Verify file size - very large files may hang
// Split large result files if needed
```

### 10.6 Build Failures

**Error:**
```
Build failed with errors
```

**Solution:**
```bash
cd dashboard

# Check for TypeScript/JSX errors
npm run build 2>&1 | head -50

# Clear build cache
rm -rf dist .cache
npm run build
```

---

## 11. Emergency Procedures

### 11.1 System Crash During Long Run

**Scenario:** System crashes during 10-hour pilot run

**Recovery:**
```bash
# Check what was saved
ls -la results/pilot/

# Check checkpoint
cat results/pilot/checkpoint.json

# Resume from checkpoint
python run_pilot.py --config config.json --resume

# If checkpoint corrupted, identify completed models
ls results/pilot/pilot_*.json

# Create new checkpoint manually
echo '{"completed": {"model_1_temp0.0": true}, "last_model_index": 1}' > results/pilot/checkpoint.json
```

### 11.2 API Key Compromised

**Scenario:** API key exposed in logs or code

**Immediate Actions:**
```bash
# 1. Revoke key immediately
# Go to https://console.anthropic.com or https://makersuite.google.com

# 2. Generate new key

# 3. Update environment
export ANTHROPIC_API_KEY="new_key"

# 4. Check git history
git log --all -S "sk-ant" --oneline

# 5. If committed, force push cleaned history or rotate all keys
```

### 11.3 Results Data Loss

**Scenario:** Results directory accidentally deleted

**Recovery:**
```bash
# Check for backups
ls backups/

# Restore from backup
tar -xzvf backups/latest.tar.gz

# If no backup, re-run affected portions
python run_pilot.py --config config.json
```

### 11.4 Unsafe Content Generated

**Scenario:** Model generates concerning content

**Immediate Actions:**
```bash
# 1. Stop execution
Ctrl+C

# 2. Document the issue
# Save prompt ID, model, and response

# 3. Review prompt
# Check if prompt itself was problematic

# 4. Report per safety policy
# See README_SAFETY.md

# 5. Add to exclusion list if needed
```

---

## Appendix: Error Code Reference

| Error Code | Category | Solution |
|------------|----------|----------|
| E001 | Installation | Reinstall dependencies |
| E002 | Configuration | Validate JSON |
| E003 | Authentication | Check API keys |
| E004 | Rate Limit | Reduce request rate |
| E005 | Memory | Use smaller model or CPU |
| E006 | Network | Check connectivity |
| E007 | Data | Validate file format |
| E008 | Checkpoint | Reset or fix checkpoint |

---

## Contact and Support

For unresolved issues:

1. Check existing GitHub issues
2. Create new issue with:
   - Error message (full traceback)
   - System information
   - Steps to reproduce
   - Logs (sanitized of API keys)

---

## Document Control

| Attribute | Value |
|-----------|-------|
| Document ID | DOC-TROUBLE-001 |
| Version | 2.4 |
| Classification | Internal |
| Author | Research Team |
| Approval Date | January 13, 2026 |

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 2.4 | 2026-01-13 | Added dashboard troubleshooting section | Research Team |
| 2.0 | 2026-01-13 | Comprehensive troubleshooting guide | Research Team |
| 1.0 | 2025-11-06 | Initial troubleshooting | Research Team |
