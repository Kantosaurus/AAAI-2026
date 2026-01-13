# Deployment Guide

**Version:** 2.4
**Last Updated:** January 13, 2026
**Document ID:** DOC-DEPLOY-001

---

## Table of Contents

1. [Deployment Overview](#1-deployment-overview)
2. [Prerequisites](#2-prerequisites)
3. [Installation Procedures](#3-installation-procedures)
4. [Configuration](#4-configuration)
5. [Environment Setup](#5-environment-setup)
6. [Verification](#6-verification)
7. [Production Deployment](#7-production-deployment)
8. [Maintenance Procedures](#8-maintenance-procedures)
9. [Backup and Recovery](#9-backup-and-recovery)
10. [Security Hardening](#10-security-hardening)

---

## 1. Deployment Overview

### 1.1 Deployment Architectures

#### Development Environment
```
┌─────────────────────────────────────────────────────────────────┐
│                    Developer Workstation                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Python    │  │   Local     │  │   Test      │             │
│  │   3.10+     │  │   Models    │  │   Data      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Local GPU (Optional)                      ││
│  │                    RTX 3090 / RTX 4090                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

#### Production Environment
```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Server                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Python    │  │   Model     │  │   Results   │             │
│  │   Runtime   │  │   Storage   │  │   Storage   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │               │               │                        │
│         └───────────────┼───────────────┘                        │
│                         ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           GPU Cluster (A100 / H100)                          ││
│  │           Multi-GPU for parallel execution                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                         │                                        │
│                         ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           External APIs (Claude, Gemini)                     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Deployment Checklist

| Phase | Task | Status |
|-------|------|--------|
| Pre-deployment | Hardware verification | ☐ |
| Pre-deployment | Software prerequisites | ☐ |
| Installation | Clone repository | ☐ |
| Installation | Install dependencies | ☐ |
| Configuration | API key setup | ☐ |
| Configuration | Model configuration | ☐ |
| Verification | Setup validation | ☐ |
| Verification | Test run | ☐ |
| Production | Full pilot execution | ☐ |

---

## 2. Prerequisites

### 2.1 Hardware Requirements

#### Minimum Configuration
| Component | Requirement | Notes |
|-----------|-------------|-------|
| CPU | 4+ cores | Intel i5/AMD Ryzen 5+ |
| RAM | 16 GB | 32 GB recommended |
| Storage | 50 GB free | SSD recommended |
| Network | Broadband | For API calls |

#### Recommended Configuration
| Component | Requirement | Notes |
|-----------|-------------|-------|
| CPU | 8+ cores | Intel i7/AMD Ryzen 7+ |
| RAM | 64 GB | For local models |
| GPU | 24GB+ VRAM | RTX 3090/4090, A100 |
| Storage | 100 GB free | NVMe SSD |
| Network | 100+ Mbps | Stable connection |

#### GPU-Specific Requirements

| Model | VRAM Required | Recommended GPU |
|-------|--------------|-----------------|
| Phi-3-mini | 8 GB | RTX 3070+ |
| Mistral-7B | 14 GB | RTX 3090+ |
| Qwen2.5-14B | 28 GB | RTX 4090 / A100 |
| All local models | 40 GB+ | A100 40GB |

### 2.2 Software Requirements

#### Operating System
- **Linux** (Ubuntu 20.04+ recommended)
- **macOS** (12.0+ for M1/M2 support)
- **Windows** (10/11, WSL2 recommended for local models)

#### Python
```bash
# Required version
python --version
# Python 3.8+ required
# Python 3.10+ recommended

# Verify pip
pip --version
```

#### CUDA (for GPU support)
```bash
# Check CUDA version
nvcc --version
# CUDA 11.8+ recommended

# Check GPU availability
nvidia-smi
```

### 2.3 Account Requirements

| Service | Purpose | Registration URL |
|---------|---------|------------------|
| Anthropic | Claude API | https://console.anthropic.com |
| Google AI | Gemini API | https://makersuite.google.com/app/apikey |
| Hugging Face | Model downloads | https://huggingface.co/join |
| NVD | API key (optional) | https://nvd.nist.gov/developers/request-an-api-key |

---

## 3. Installation Procedures

### 3.1 Repository Setup

```bash
# Clone repository
git clone https://github.com/Kantosaurus/AAAI-2026.git
cd AAAI-2026

# Verify repository structure
ls -la
# Should see: data/, experiments/, annotations/, docs/, etc.
```

### 3.2 Python Environment Setup

#### Option A: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.\.venv\Scripts\activate

# Verify activation
which python
# Should point to .venv/bin/python
```

#### Option B: Conda Environment
```bash
# Create conda environment
conda create -n hallu-research python=3.10 -y

# Activate
conda activate hallu-research

# Verify
python --version
```

### 3.3 Dependency Installation

#### Core Dependencies
```bash
# Install core requirements
cd experiments/pilot
pip install -r requirements.txt

# Verify installation
python -c "import anthropic; import google.generativeai; print('Core OK')"
```

#### Full Dependencies (Including Analysis)
```bash
# Install all dependencies
pip install anthropic google-generativeai transformers torch tqdm
pip install jupyter pandas matplotlib seaborn scikit-learn
pip install sentence-transformers faiss-cpu
pip install transformer-lens  # For interpretability

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 3.4 Dashboard Setup (Optional)

```bash
# Navigate to dashboard directory
cd dashboard

# Install Node.js dependencies
npm install

# Start development server
npm run dev
# Dashboard available at http://localhost:5173

# Build for production
npm run build
# Output in dashboard/dist/
```

### 3.5 Model Downloads (Optional, for Local Models)

```bash
# Login to Hugging Face (required for some models)
huggingface-cli login

# Pre-download models (optional - will auto-download on first run)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    'microsoft/Phi-3-mini-128k-instruct',  # 7.6 GB
    'mistralai/Mistral-7B-Instruct-v0.3',   # 14 GB
    'Qwen/Qwen2.5-14B-Instruct',            # 28 GB
]

for model in models:
    print(f'Downloading {model}...')
    AutoTokenizer.from_pretrained(model)
    AutoModelForCausalLM.from_pretrained(model)
    print(f'  Done!')
"
```

**Estimated Download Times:**
| Model | Size | Time (100 Mbps) |
|-------|------|-----------------|
| Phi-3-mini | 7.6 GB | ~10 min |
| Mistral-7B | 14 GB | ~20 min |
| Qwen2.5-14B | 28 GB | ~40 min |

---

## 4. Configuration

### 4.1 API Key Configuration

#### Method 1: Environment Variables (Recommended)
```bash
# Linux/macOS - Add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export GOOGLE_API_KEY="AIza..."
export HF_TOKEN="hf_..."  # Optional

# Reload shell
source ~/.bashrc

# Verify
echo $ANTHROPIC_API_KEY
```

#### Method 2: .env File
```bash
# Create .env file in project root
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=AIza...
HF_TOKEN=hf_...
EOF

# Add to .gitignore (should already be there)
echo ".env" >> .gitignore
```

#### Method 3: Direct in Config (Not Recommended)
```json
// Only for testing - never commit API keys
{
  "models": [
    {
      "name": "claude-3-5-sonnet",
      "api_key": "sk-ant-api03-..."  // DON'T DO THIS
    }
  ]
}
```

### 4.2 Pilot Configuration

#### Test Configuration (config_small_test.json)
```json
{
  "description": "Small test - 50 prompts, 2 models",
  "prompts_file": "../../data/prompts/hallu-sec-benchmark.json",
  "output_dir": "../../results/pilot_test",
  "num_prompts": 50,
  "seed": 42,
  "max_retries": 3,
  "requests_per_minute": 30,
  "models": [
    {
      "name": "claude-3-5-sonnet-20241022",
      "type": "claude",
      "temperature": 0.0
    },
    {
      "name": "microsoft/Phi-3-mini-128k-instruct",
      "type": "local",
      "device": "cuda",
      "temperature": 0.0
    }
  ]
}
```

#### Full Production Configuration (config_full_pilot.json)
```json
{
  "description": "Full pilot - 393 prompts, 10 model configs",
  "prompts_file": "../../data/prompts/hallu-sec-benchmark.json",
  "output_dir": "../../results/pilot",
  "num_prompts": null,
  "seed": 42,
  "max_retries": 3,
  "requests_per_minute": 60,
  "models": [
    // See api-reference.md for full configuration
  ]
}
```

### 4.3 Custom Configuration

Create custom configuration for specific needs:

```json
{
  "description": "Custom run - Claude only, temp sweep",
  "prompts_file": "../../data/prompts/hallu-sec-benchmark.json",
  "output_dir": "../../results/custom",
  "num_prompts": 100,
  "seed": 42,
  "max_retries": 5,
  "requests_per_minute": 30,
  "models": [
    {
      "name": "claude-3-5-sonnet-20241022",
      "type": "claude",
      "temperature": 0.0,
      "notes": "Deterministic"
    },
    {
      "name": "claude-3-5-sonnet-20241022",
      "type": "claude",
      "temperature": 0.3,
      "notes": "Low creativity"
    },
    {
      "name": "claude-3-5-sonnet-20241022",
      "type": "claude",
      "temperature": 0.7,
      "notes": "Medium creativity"
    },
    {
      "name": "claude-3-5-sonnet-20241022",
      "type": "claude",
      "temperature": 1.0,
      "notes": "High creativity"
    }
  ]
}
```

---

## 5. Environment Setup

### 5.1 Directory Structure Verification

```bash
# Verify complete structure
tree -L 2 --dirsfirst

# Expected output:
# AAAI-2026/
# ├── data/
# │   ├── prompts/
# │   ├── scripts/
# │   └── outputs/
# ├── experiments/
# │   ├── pilot/
# │   ├── interpretability/
# │   ├── mitigations/
# │   └── integration/
# ├── annotations/
# ├── results/
# ├── notebooks/
# ├── docs/
# └── ...
```

### 5.2 Results Directory Setup

```bash
# Create results directories
mkdir -p results/pilot
mkdir -p results/pilot_test
mkdir -p results/analysis

# Set permissions
chmod 755 results/
```

### 5.3 GPU Configuration

```bash
# Check GPU availability
nvidia-smi

# Set visible GPUs (if multiple)
export CUDA_VISIBLE_DEVICES=0

# Verify PyTorch can see GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### 5.4 Memory Optimization

For systems with limited GPU memory:

```python
# In config, use CPU for large models
{
  "name": "Qwen/Qwen2.5-14B-Instruct",
  "type": "local",
  "device": "cpu",  # Use CPU instead of CUDA
  "temperature": 0.0
}
```

```bash
# Or limit GPU memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 6. Verification

### 6.1 Setup Validation Script

```bash
cd experiments/pilot
python validate_setup.py
```

**Expected Output:**
```
Checking setup for LLM Hallucination Pilot...

[1/6] Python version... OK (3.10.12)
[2/6] Dependencies... OK
[3/6] Benchmark file... OK (393 prompts)
[4/6] API keys...
  - ANTHROPIC_API_KEY: OK
  - GOOGLE_API_KEY: OK
[5/6] GPU availability... OK (NVIDIA RTX 4090, 24GB)
[6/6] Disk space... OK (245 GB free)

All checks passed! Ready to run pilot.
```

### 6.2 Test Run

```bash
# Run minimal test (5 prompts, ~2 minutes)
python run_pilot.py --config config_small_test.json --num-prompts 5

# Check output
ls -la ../../results/pilot_test/
cat ../../results/pilot_test/pilot_*.json | head -100
```

### 6.3 API Verification

```bash
# Test Claude API
python -c "
import anthropic
import os

client = anthropic.Anthropic()
response = client.messages.create(
    model='claude-3-5-sonnet-20241022',
    max_tokens=100,
    messages=[{'role': 'user', 'content': 'Say hello'}]
)
print('Claude API: OK')
print(f'Response: {response.content[0].text[:50]}...')
"

# Test Gemini API
python -c "
import google.generativeai as genai
import os

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-pro')
response = model.generate_content('Say hello')
print('Gemini API: OK')
print(f'Response: {response.text[:50]}...')
"
```

### 6.4 Model Loading Verification

```bash
# Test local model loading
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'microsoft/Phi-3-mini-128k-instruct'
print(f'Loading {model_name}...')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)

print('Model loaded successfully!')
print(f'Device: {next(model.parameters()).device}')
"
```

---

## 7. Production Deployment

### 7.1 Pre-Production Checklist

| Item | Verified |
|------|----------|
| API keys configured and tested | ☐ |
| GPU memory sufficient for models | ☐ |
| Disk space (100GB+ free) | ☐ |
| Network stability verified | ☐ |
| Backup location prepared | ☐ |
| Monitoring setup (optional) | ☐ |

### 7.2 Full Pilot Execution

```bash
# Navigate to pilot directory
cd experiments/pilot

# Run full pilot (10-14 hours)
python run_pilot.py --config config_full_pilot.json

# For background execution
nohup python run_pilot.py --config config_full_pilot.json > pilot.log 2>&1 &

# Monitor progress
tail -f pilot.log
```

### 7.3 Checkpoint and Resume

```bash
# If interrupted, resume from checkpoint
python run_pilot.py --config config_full_pilot.json --resume

# Check checkpoint status
cat ../../results/pilot/checkpoint.json
```

### 7.4 Progress Monitoring

```bash
# Count completed prompts
for f in ../../results/pilot/pilot_*.json; do
    echo "$f: $(grep -c '"prompt_id"' "$f") prompts"
done

# Check for errors
grep -l '"error"' ../../results/pilot/*.json
```

### 7.5 Post-Execution Verification

```bash
# Verify all files created
ls -lh ../../results/pilot/

# Expected: 10+ JSON files (one per model config)
# Total size: 500MB - 1GB

# Count total results
python -c "
import json
from pathlib import Path

results_dir = Path('../../results/pilot')
total = 0
for f in results_dir.glob('pilot_*.json'):
    data = json.loads(f.read_text())
    for run in data.get('runs', []):
        total += len(run.get('results', []))
print(f'Total responses: {total}')
# Expected: 3,930 (393 prompts × 10 configs)
"
```

---

## 8. Maintenance Procedures

### 8.1 Regular Maintenance Tasks

#### Daily (During Active Research)
- [ ] Check disk space
- [ ] Review error logs
- [ ] Verify API quota usage

#### Weekly
- [ ] Update dependencies (`pip install -U -r requirements.txt`)
- [ ] Backup results
- [ ] Review checkpoint files

#### Monthly
- [ ] Update NVD data
- [ ] Review MITRE ATT&CK updates
- [ ] Check for model updates

### 8.2 Dependency Updates

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install -U anthropic

# Update all (carefully)
pip install -U -r requirements.txt

# Verify after update
python validate_setup.py
```

### 8.3 Data Maintenance

```bash
# Clean temporary files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Clean incomplete checkpoints
rm -f results/*/checkpoint.json

# Archive old results
tar -czvf results_archive_$(date +%Y%m%d).tar.gz results/
```

### 8.4 Model Cache Management

```bash
# Check Hugging Face cache size
du -sh ~/.cache/huggingface/

# Clear specific model cache
rm -rf ~/.cache/huggingface/hub/models--microsoft--Phi-3-mini*

# Clear all (warning: re-download required)
rm -rf ~/.cache/huggingface/hub/
```

---

## 9. Backup and Recovery

### 9.1 Backup Strategy

#### Critical Data (Backup Immediately)
| Data | Location | Frequency |
|------|----------|-----------|
| Results | `results/pilot/*.json` | After each run |
| Annotations | `annotations/*.csv` | After each session |
| Configurations | `experiments/*/config*.json` | On change |

#### Less Critical (Backup Weekly)
| Data | Location | Frequency |
|------|----------|-----------|
| Benchmark | `data/prompts/*.json` | Weekly |
| Scripts | All `.py` files | Weekly (via git) |

### 9.2 Backup Procedures

```bash
# Create timestamped backup
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup critical data
cp -r results/ "$BACKUP_DIR/"
cp -r annotations/ "$BACKUP_DIR/"
cp experiments/pilot/config*.json "$BACKUP_DIR/"

# Compress backup
tar -czvf "${BACKUP_DIR}.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

# Verify backup
tar -tzvf "${BACKUP_DIR}.tar.gz" | head
```

### 9.3 Recovery Procedures

```bash
# List available backups
ls -la backups/

# Restore from backup
tar -xzvf backups/20260113_120000.tar.gz

# Restore specific files
tar -xzvf backups/20260113_120000.tar.gz results/pilot/

# Verify restoration
python validate_setup.py
```

### 9.4 Disaster Recovery

If critical data is lost:

1. **Stop all running processes**
   ```bash
   pkill -f run_pilot.py
   ```

2. **Check for checkpoint files**
   ```bash
   find . -name "checkpoint.json"
   ```

3. **Restore from backup**
   ```bash
   tar -xzvf backups/latest.tar.gz
   ```

4. **Resume interrupted runs**
   ```bash
   python run_pilot.py --config config.json --resume
   ```

---

## 10. Security Hardening

### 10.1 API Key Security

```bash
# Never commit API keys
echo ".env" >> .gitignore
echo "*.key" >> .gitignore

# Check for accidental commits
git log --oneline --all -S "sk-ant" | head
git log --oneline --all -S "AIza" | head

# Use environment variables only
unset ANTHROPIC_API_KEY  # After session
```

### 10.2 File Permissions

```bash
# Secure configuration files
chmod 600 .env
chmod 600 experiments/pilot/config*.json

# Secure results
chmod 700 results/
chmod 600 results/pilot/*.json
```

### 10.3 Network Security

```bash
# Use HTTPS for all API calls (default)
# Verify TLS certificates (Python default behavior)

# For sensitive environments, consider:
# - VPN for API access
# - Firewall rules for outbound connections
# - Proxy configuration if required
```

### 10.4 Audit Logging

```bash
# Enable comprehensive logging
export LOGLEVEL=DEBUG

# Log all API calls
python run_pilot.py --config config.json 2>&1 | tee pilot_$(date +%Y%m%d).log

# Review logs for issues
grep -i error pilot_*.log
grep -i warning pilot_*.log
```

### 10.5 Safety Compliance

Before each run, verify:
- [ ] All prompts are sanitized (use provided benchmark)
- [ ] No sensitive data in configurations
- [ ] Output directories are secured
- [ ] Safety policy checklist reviewed

---

## Document Control

| Attribute | Value |
|-----------|-------|
| Document ID | DOC-DEPLOY-001 |
| Version | 2.4 |
| Classification | Internal |
| Author | Research Team |
| Approval Date | January 13, 2026 |

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 2.4 | 2026-01-13 | Added dashboard deployment instructions | Research Team |
| 2.0 | 2026-01-13 | Complete deployment documentation | Research Team |
| 1.0 | 2025-11-06 | Initial deployment guide | Research Team |
