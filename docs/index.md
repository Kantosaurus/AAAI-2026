# LLM Hallucination Research Framework - Documentation Hub

**Version:** 2.0
**Last Updated:** January 13, 2026
**Status:** Implementation Complete - Ready for Production

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Quick Navigation](#quick-navigation)
3. [Getting Started](#getting-started)
4. [Documentation Structure](#documentation-structure)
5. [System Requirements](#system-requirements)
6. [Support & Resources](#support--resources)

---

## Executive Summary

### Project Overview

This repository contains a comprehensive research framework for **characterizing and mitigating hallucinations in Large Language Models (LLMs)** within cybersecurity contexts. The project implements an end-to-end infrastructure for:

- **Benchmark Generation** - 393 security-focused prompts with verified ground truth
- **Multi-Model Evaluation** - Support for Claude, Gemini, and local Transformers models
- **Annotation Pipeline** - Systematic hallucination labeling with inter-annotator agreement
- **Interpretability Analysis** - Mechanistic understanding via causal tracing and probing
- **Mitigation Strategies** - RAG, symbolic checking, and uncertainty-based abstention
- **Integration Testing** - Real-world vulnerability triage workflow simulations

### Key Metrics

| Metric | Value |
|--------|-------|
| Total Prompts | 393 |
| Real/Grounded Queries | 250 (63.6%) |
| Synthetic Probes | 143 (36.4%) |
| Model Configurations | 10 (5 models × 2 temperatures) |
| Total Expected Responses | 3,930 |
| Lines of Python Code | ~5,000+ |
| Documentation Pages | 52+ files |

### Implementation Status

| Phase | Status | Completion |
|-------|--------|------------|
| Phase A: Dataset Construction | Complete | 100% |
| Phase B: Pilot Infrastructure | Complete | 100% |
| Phase C: Annotation Pipeline | Complete | 100% |
| Phase D: Interpretability & Mitigations | Complete | 100% |
| Phase E: Integration Testing | Complete | 100% |

---

## Quick Navigation

### For New Users

| Goal | Start Here |
|------|------------|
| Run the system immediately | [Quick Start Guide](../experiments/pilot/QUICK_START.md) |
| Understand the project | [User Guide](user-guide.md) |
| Set up your environment | [Deployment Guide](deployment-guide.md) |

### For Developers

| Goal | Start Here |
|------|------------|
| Understand the architecture | [Architecture Documentation](architecture.md) |
| Extend the codebase | [Developer Guide](developer-guide.md) |
| API reference | [API Reference](api-reference.md) |

### For Researchers

| Goal | Start Here |
|------|------------|
| Research methodology | [Implementation Plan](implementation.md) |
| Dataset details | [Benchmark Summary](../data/prompts/BENCHMARK_SUMMARY.md) |
| Annotation guidelines | [Annotation Rubric](../annotations/rubric.md) |

### For Operations

| Goal | Start Here |
|------|------------|
| Deployment procedures | [Deployment Guide](deployment-guide.md) |
| Troubleshooting | [Troubleshooting Guide](troubleshooting-guide.md) |
| Safety protocols | [Safety Policy](safety_policy_checklist.md) |

---

## Getting Started

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Clone repository
git clone https://github.com/yourusername/AAAI-2026.git
cd AAAI-2026

# Install dependencies
pip install -r experiments/pilot/requirements.txt
```

### Minimal Setup (5 Minutes)

```bash
# 1. Set API keys
export ANTHROPIC_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"

# 2. Validate setup
cd experiments/pilot
python validate_setup.py

# 3. Run test (50 prompts, ~15 min)
python run_pilot.py --config config_small_test.json
```

### Full Production Run (10-14 Hours)

```bash
python run_pilot.py --config config_full_pilot.json
```

---

## Documentation Structure

### Core Documentation

```
docs/
├── index.md                     # This file - documentation hub
├── architecture.md              # System architecture and design
├── api-reference.md             # Complete API documentation
├── deployment-guide.md          # Installation and deployment
├── user-guide.md                # End-user documentation
├── developer-guide.md           # Developer onboarding
├── troubleshooting-guide.md     # Problem resolution guide
├── changelog.md                 # Version history
├── implementation.md            # Research methodology
├── safety_policy_checklist.md   # Safety compliance
└── public_sources_seed_list.md  # Data source reference
```

### Component Documentation

```
experiments/
├── pilot/
│   ├── QUICK_START.md           # Fast setup guide
│   ├── SETUP_GUIDE.md           # Detailed setup instructions
│   ├── IMPLEMENTATION_DETAILS.md # Technical deep dive
│   └── RUN_PILOT_SUMMARY.md     # Pilot execution overview
├── interpretability/
│   └── README.md                # Interpretability experiments
├── mitigations/
│   └── README.md                # Mitigation strategies
└── integration/
    └── README.md                # Integration testing

data/
├── prompts/
│   └── BENCHMARK_SUMMARY.md     # Dataset documentation
└── README_DATA_COLLECTION.md    # Data collection procedures

annotations/
└── rubric.md                    # Annotation guidelines
```

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.8+ |
| RAM | 16GB |
| Storage | 50GB free |
| Internet | Required for API calls |

### Recommended Requirements

| Component | Recommendation |
|-----------|----------------|
| Python | 3.10+ |
| RAM | 64GB |
| GPU | 24GB+ VRAM (RTX 3090, A100) |
| Storage | 100GB free |
| Internet | Stable broadband |

### API Requirements

| Service | Purpose | Cost Estimate |
|---------|---------|---------------|
| Anthropic Claude | High-capability LLM | $15-20 |
| Google Gemini | Alternative LLM | $15-20 |
| Hugging Face | Model downloads | Free |

**Total Estimated Cost:** $30-40 for full pilot

---

## Support & Resources

### Documentation Links

- [Main README](../README.md)
- [Safety Documentation](../README_SAFETY.md)
- [How to Run](../HOW_TO_RUN.md)
- [Implementation Status](../IMPLEMENTATION_STATUS.md)

### External Resources

- [NIST NVD](https://nvd.nist.gov/) - Vulnerability database
- [MITRE ATT&CK](https://attack.mitre.org/) - Threat framework
- [Anthropic Claude](https://docs.anthropic.com/) - Claude API docs
- [Google Gemini](https://ai.google.dev/) - Gemini API docs

### Getting Help

1. Review relevant documentation section
2. Check [Troubleshooting Guide](troubleshooting-guide.md)
3. Search existing GitHub issues
4. Open new issue with:
   - Clear description
   - Steps to reproduce
   - System information
   - Error messages

### Safety Concerns

If you discover safety issues:

1. **Stop** the concerning activity immediately
2. **Document** with screenshots and logs
3. **Report** to project maintainers
4. **Review** [Safety Policy](safety_policy_checklist.md)

---

## Document Information

| Attribute | Value |
|-----------|-------|
| Document ID | DOC-INDEX-001 |
| Version | 2.0 |
| Classification | Public |
| Author | Research Team |
| Reviewer | Project Lead |
| Approval Date | January 13, 2026 |

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 2.0 | 2026-01-13 | Enterprise documentation overhaul | Research Team |
| 1.0 | 2025-11-06 | Initial documentation | Research Team |

---

**Next:** Start with the [Quick Start Guide](../experiments/pilot/QUICK_START.md) or explore the [Architecture Documentation](architecture.md).
