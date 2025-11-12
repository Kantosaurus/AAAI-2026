# Experiments Directory

This directory contains all experimental code and configurations for the LLM hallucination research project.

---

## Directory Structure

```
experiments/
├── pilot/                      # Phase B: Pilot runs (Nov 10-14)
│   ├── run_pilot.py           # Main pilot execution script
│   ├── config_full_pilot.json # Full 5-model configuration
│   ├── config_small_test.json # Test configuration (50 prompts)
│   ├── requirements.txt       # Python dependencies
│   ├── SETUP_GUIDE.md        # Detailed setup instructions
│   └── QUICK_START.md        # TL;DR quick reference
│
├── interpretability/          # Phase D: Mechanistic analysis (Nov 20-25)
│   └── (to be added in Phase D)
│
└── mitigations/              # Phase D: Mitigation experiments (Nov 20-25)
    └── (to be added in Phase D)
```

---

## Current Status: Phase B Setup Complete ✅

### Completed (Nov 6 - Day 1)
- [x] Benchmark generated: 393 prompts (250 real + 143 synthetic)
- [x] All prompts sanitized and safety-checked
- [x] Pilot run script implemented
- [x] Model configurations created
- [x] Documentation complete

### Next Steps (Nov 10-14)
- [ ] Install dependencies
- [ ] Configure API keys
- [ ] Run test pilot (50 prompts)
- [ ] Run full pilot (393 prompts × 10 model configs)
- [ ] Sanity check outputs

---

## Quick Start

### 1. Setup (5 minutes)
```bash
cd pilot/
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"
```

### 2. Test Run (15 minutes)
```bash
python run_pilot.py --config config_small_test.json
```

### 3. Full Pilot (10-14 hours)
```bash
python run_pilot.py --config config_full_pilot.json
```

### 4. Results
```bash
ls ../results/pilot/
```

---

## Model Configuration

Your selected models for pilot:

| # | Model | Type | Size | Purpose |
|---|-------|------|------|---------|
| 1 | Claude 3.5 Sonnet | Closed API | - | High-capability baseline |
| 2 | Gemini 1.5 Pro | Closed API | - | Alternative API |
| 3 | Qwen2.5-14B-Instruct | Open Local | 14B | Interpretability |
| 4 | Mistral-7B-Instruct | Open Local | 7B | Medium scaling |
| 5 | Phi-3-mini-128k | Open Local | 3.8B | Small baseline |

Each model tested at 2 temperatures (0.0, 0.7) = **10 total configurations**

---

## Phase Timeline

### Phase B: Pilot Runs (Nov 10-14) ⏳
**Goal:** Collect model outputs across all configurations

**Day 5 (Nov 10):**
- Implement run_pilot.py ✅
- Install dependencies
- Download local models

**Day 6 (Nov 11):**
- Run 50-prompt test
- Verify output capture
- Check for unsafe content

**Day 7 (Nov 12):**
- Full pilot: 393 prompts × 10 configs
- 3,930 total model calls
- Est. 10-14 hours runtime

**Day 8 (Nov 13):**
- Sanity checks (CVE citation rates)
- Flag fabricated citations
- Basic metrics

**Day 9 (Nov 14):**
- Freeze pilot data
- Prepare annotation batches

### Phase C: Annotation (Nov 15-19)
**Goal:** Annotate responses for hallucinations

- Finalize annotation rubric
- Train annotators
- Dual annotation with adjudication
- Compute inter-annotator agreement
- Calculate hallucination metrics

### Phase D: Interpretability & Mitigations (Nov 20-25)
**Goal:** Understand causes and test fixes

- Causal tracing on open models
- Activation probing
- RAG experiments
- Symbolic checker implementation
- Uncertainty-based abstention

### Phase E: Integration & Finalization (Nov 26-30)
**Goal:** Applied workflows and deliverables

- Simulated workflows
- Integration testing
- Final report writing
- Slide deck creation

---

## Expected Outputs

### After Phase B (Nov 14)
```
results/pilot/
├── pilot_claude-3-5-sonnet_*.json       (2 files: temp 0.0, 0.7)
├── pilot_gemini-1.5-pro_*.json          (2 files)
├── pilot_Qwen_Qwen2.5-14B-Instruct_*.json  (2 files)
├── pilot_mistralai_Mistral-7B-Instruct_*.json  (2 files)
├── pilot_microsoft_Phi-3-mini_*.json    (2 files)
└── pilot_results_combined.json          (all runs)
```

**Total:** ~10 JSON files, ~500MB-1GB combined

### Data Format
Each result file contains:
- Full model responses (text)
- Token counts (input/output)
- Token logprobs (local models only)
- Timing data
- Error logs
- Prompt metadata (category, is_synthetic)

---

## Safety Protocols

### Before Running
1. Review safety_policy_checklist.md
2. Ensure API keys are secure (use .env, not committed)
3. Verify benchmark prompts are sanitized
4. Allocate time for supervision

### During Running
1. Monitor outputs for unsafe content
2. Stop immediately if exploit code generated
3. Document any concerning outputs
4. Maintain rate limits (respect API ToS)

### After Running
1. Review all outputs for safety
2. Flag any unsafe generations
3. Store results securely
4. Backup to encrypted location

### If Unsafe Content Detected
1. **Stop execution immediately**
2. Document the issue (prompt ID, model, response)
3. Report to project PI
4. Do NOT publish or share unsafe outputs
5. Consider coordinated disclosure if model vulnerability

---

## Hardware Requirements

### For Local Models (Recommended)
- **GPU:** RTX 4090 or A100 (24-48GB VRAM)
- **RAM:** 64GB
- **Storage:** 100GB free
- **OS:** Linux (Ubuntu 22.04) or Windows with WSL2

### Minimum (Will Work But Slow)
- **GPU:** RTX 3090 (24GB VRAM)
- **RAM:** 32GB
- **Storage:** 50GB free

### CPU-Only (Not Recommended)
- Possible but 10-20x slower
- Use only for small tests
- Not viable for full pilot

---

## Cost Estimates

### API Costs
- Claude: ~$15-20 (393 prompts × 2 temps)
- Gemini: ~$15-20 (393 prompts × 2 temps)
- **Total API: $30-40**

### Local Models
- Free (use your own GPU)
- Electricity: negligible

### Storage
- Free (use existing infrastructure)

**Total Project Cost: $30-40** (very affordable!)

---

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Solution: Use smaller models first, or set device to CPU

**"Rate limit exceeded"**
- Solution: Increase `api_delay` in config to 2.0 or higher

**"Model download failed"**
- Solution: `huggingface-cli login` and retry

**"ModuleNotFoundError"**
- Solution: `pip install -r requirements.txt`

See `pilot/SETUP_GUIDE.md` for detailed troubleshooting.

---

## Documentation

- **Detailed Setup:** `pilot/SETUP_GUIDE.md`
- **Quick Reference:** `pilot/QUICK_START.md`
- **Implementation Plan:** `../docs/implementation.md`
- **Safety Policy:** `../docs/safety_policy_checklist.md`
- **Benchmark Info:** `../data/prompts/BENCHMARK_SUMMARY.md`

---

## Questions or Issues?

1. Check documentation in `pilot/`
2. Review troubleshooting section
3. Check GitHub issues (if applicable)
4. Contact project lead

---

## Next Phase Preview

After pilot runs complete, Phase C begins:

**Annotation Workflow:**
1. Load pilot results
2. Randomize response order
3. Dual annotation by 2 independent annotators
4. Adjudication of disagreements
5. Calculate inter-annotator agreement (Cohen's κ)
6. Compute hallucination metrics per model

See `../annotations/rubric.md` for annotation guidelines (to be created in Phase C).

---

**Status:** Ready for Phase B pilot runs (Nov 10-14)
**Last Updated:** November 6, 2025
**Version:** 1.0
