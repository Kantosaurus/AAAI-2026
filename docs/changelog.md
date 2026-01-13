# Changelog

All notable changes to the LLM Hallucination Research Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- OpenAI GPT-4 model support

---

## [2.4.0] - 2026-01-13

### Changed
- **Dashboard Theme Update**
  - Converted from dark theme to light theme with white background
  - Updated CSS variables for light-friendly colors
  - Adjusted shadows, borders, and contrasts for white backgrounds
  - Improved readability with darker text colors

### Removed
- **Mock Data Removal**
  - Removed `dashboard/src/data/sampleResults.json` sample data file
  - Dashboard now requires real pilot results data (no demo mode)
  - Updated `useResults.js` to require actual data source

### Added
- **Data Loading Interface**
  - File upload interface for loading pilot results JSON
  - URL query parameter support (`?results=/path/to/file.json`)
  - Empty state UI with upload instructions
  - Better error handling for missing or invalid data

---

## [2.3.0] - 2026-01-13

### Added
- **Web Dashboard for Results Visualization**
  - React 18 + Vite SPA with "Research Lab Terminal" aesthetic
  - `dashboard/` directory with full frontend application
  - 7 modular components:
    - Header - Project title and metadata display
    - MetricCard - Animated summary statistics with count-up effect
    - ModelComparison - Horizontal bar chart with temperature toggle
    - CategoryBreakdown - Donut chart with legend
    - TokenUsage - Stacked area chart for input/output tokens
    - ResponseTimes - Multi-line chart per model
    - ErrorRates - Error distribution with severity coloring
  - Asymmetrical grid layout with bold typography
  - Light theme with cyan/coral accent colors
  - Framer Motion animations and micro-interactions
  - Data transformation utilities for pilot results JSON

### Dependencies
- Added `react`, `react-dom`, `recharts`, `framer-motion`, `lucide-react`
- Vite for development and build tooling

---

## [2.2.0] - 2026-01-13

### Added
- **Automated CI/CD Pipeline**
  - `.github/workflows/ci.yml` - Main CI workflow with lint, test, type-check jobs
  - `.github/workflows/pr.yml` - Pull request checks workflow
  - `pytest.ini` - Pytest configuration with markers for slow/GPU/API tests
  - `pyproject.toml` - Tool configuration (ruff, black, mypy, coverage)
  - `requirements-dev.txt` - Development dependencies
  - Multi-version Python testing (3.8, 3.10, 3.11)
  - Coverage reporting with Codecov integration
  - Syntax validation for all Python files
  - Import checking for async modules

### Dependencies
- Added `pytest>=7.0.0`, `pytest-cov>=4.0.0`, `pytest-asyncio>=0.21.0`
- Added `ruff>=0.1.0`, `black>=23.0.0`, `mypy>=1.0.0`
- Added `types-aiofiles>=23.0.0`, `types-tqdm>=4.0.0`

---

## [2.1.0] - 2026-01-13

### Added
- **Async API Calls for Parallel Execution**
  - `async_rate_limiter.py` - Async token bucket rate limiter using asyncio.Lock
  - `async_runners.py` - Async model runners (AsyncClaudeRunner, AsyncGeminiRunner, AsyncLocalModelRunner)
  - `async_pilot_runner.py` - Async execution orchestrator with semaphore-based concurrency
  - Configurable max_concurrency (default: 5 parallel requests)
  - Performance improvement: ~4x faster execution (2-4 hours vs 10-14 hours)

### Changed
- `run_pilot.py` converted to async-only execution using asyncio.run()
- Added `--max-concurrency` CLI flag for runtime configuration
- Config files updated with `async_settings` block

### Dependencies
- Added `aiofiles>=23.0.0` for async file I/O

---

## [2.0.0] - 2026-01-13

### Added
- **Enterprise Documentation Suite**
  - `docs/index.md` - Master documentation hub
  - `docs/architecture.md` - Comprehensive system architecture
  - `docs/api-reference.md` - Complete API documentation
  - `docs/deployment-guide.md` - Installation and deployment procedures
  - `docs/user-guide.md` - End-user documentation
  - `docs/developer-guide.md` - Developer onboarding guide
  - `docs/troubleshooting-guide.md` - Problem resolution guide
  - `docs/changelog.md` - This file

- **Bug Fixes and Robustness**
  - Fixed type hint compatibility for Python 3.8-3.9
  - Fixed causal tracing hook logic for accurate layer isolation
  - Added division by zero protection in agreement calculation
  - Added validation for disjoint CVE sets in symbolic checker
  - Added robust format validation for token logprobs
  - Added error handling for empty results directory

### Changed
- Updated all existing documentation for consistency
- Improved README.md with clearer structure
- Enhanced error messages throughout codebase

### Fixed
- `abstention_detector.py`: Line 82 - Changed `tuple[bool, float]` to `Tuple[bool, float]`
- `causal_tracing.py`: Lines 173-190 - Fixed hook restoration logic
- `compute_agreement.py`: Line 97 - Better handling of edge cases
- `symbolic_checker.py`: Added set disjointness validation
- `prepare_annotation_batches.py`: Added empty directory handling

---

## [1.1.0] - 2025-11-26

### Added
- **Phase E: Integration Testing**
  - `vuln_triage_workflow.py` - Vulnerability triage simulation
  - `workflow_scenarios.json` - Test scenario definitions
  - `experiments/integration/README.md` - Integration documentation

- **Phase D: Interpretability Experiments**
  - `select_cases_for_interp.py` - Case selection for analysis
  - `causal_tracing.py` - Causal tracing implementation
  - `activation_probes.py` - Linear probe training and evaluation
  - `experiments/interpretability/README.md` - Interpretability documentation

- **Phase D: Mitigation Strategies**
  - `symbolic_checker.py` - CVE verification against NVD
  - `abstention_detector.py` - Uncertainty-based abstention
  - `rag_grounding.py` - Retrieval-augmented generation
  - `build_retrieval_index.py` - Index construction utility
  - `evaluate_mitigations.py` - Comparative evaluation framework
  - `experiments/mitigations/README.md` - Mitigation documentation

### Changed
- Improved pilot runner with better error handling
- Enhanced checkpoint/resume functionality

---

## [1.0.0] - 2025-11-06

### Added
- **Phase A: Dataset Construction**
  - 393 security-focused prompts (250 real, 143 synthetic)
  - 5 categories: CVE Existence, Vulnerability Summary, Malware Description, Secure Configuration, Pentest Reporting
  - Ground truth with NVD and MITRE ATT&CK references
  - Sanitization with 106 safety annotations
  - `hallu-sec-benchmark.json` - Complete benchmark dataset
  - `sanitization_report.json` - Safety verification report
  - `BENCHMARK_SUMMARY.md` - Dataset documentation

- **Phase B: Pilot Infrastructure**
  - `run_pilot.py` - Main execution script (626 lines)
  - Multi-model support (Claude, Gemini, Transformers)
  - Token bucket rate limiting
  - Exponential backoff retry logic
  - Checkpoint/resume functionality
  - Progress tracking with tqdm
  - Token logprobs extraction for local models
  - `config_full_pilot.json` - Full pilot configuration
  - `config_small_test.json` - Test configuration
  - `validate_setup.py` - Setup validation utility
  - `test_runner.py` - Automated testing

- **Phase C: Annotation Pipeline**
  - `rubric.md` - 12-section annotation guidelines
  - `annotations_raw.csv` - Annotation template
  - `prepare_annotation_batches.py` - Batch preparation
  - `compute_agreement.py` - Inter-annotator agreement

- **Data Collection Scripts**
  - `fetch_nvd_metadata.py` - NVD API integration
  - `generate_synthetic_cves.py` - Synthetic CVE generation
  - `create_gold_truth_dataset.py` - Dataset construction
  - `validate_gold_truth.py` - Validation utilities

- **Documentation**
  - `README.md` - Project overview
  - `README_SAFETY.md` - Safety protocols
  - `HOW_TO_RUN.md` - Execution guide
  - `IMPLEMENTATION_STATUS.md` - Status tracking
  - `IMPLEMENTATION_COMPLETE.md` - Completion report
  - `PHASE_B_COMPLETE.md` - Phase summary
  - `docs/implementation.md` - Research methodology
  - `docs/safety_policy_checklist.md` - Safety compliance
  - `docs/public_sources_seed_list.md` - Data sources
  - `experiments/pilot/SETUP_GUIDE.md` - Setup instructions
  - `experiments/pilot/QUICK_START.md` - Quick reference
  - `experiments/pilot/IMPLEMENTATION_DETAILS.md` - Technical details
  - `experiments/pilot/RUN_PILOT_SUMMARY.md` - Execution summary

- **Analysis Tools**
  - `notebooks/analysis_template.py` - Analysis notebook template

### Model Support
- Claude 3.5 Sonnet (Anthropic API)
- Gemini 1.5 Pro (Google API)
- Qwen2.5-14B-Instruct (Local)
- Mistral-7B-Instruct-v0.3 (Local)
- Phi-3-mini-128k-instruct (Local)

### Security
- All prompts sanitized for safety
- No exploit code or weaponizable content
- Defensive framing throughout
- No PII, credentials, or sensitive data

---

## [0.1.0] - 2025-11-05

### Added
- Initial repository structure
- Project scaffolding
- Basic README

---

## Version Comparison

| Version | Prompts | Models | Mitigations | Documentation | Features |
|---------|---------|--------|-------------|---------------|----------|
| 2.4.0 | 393 | 5 | 3 | 52+ files | Dashboard (production-ready) |
| 2.3.0 | 393 | 5 | 3 | 52+ files | Dashboard, CI/CD |
| 2.2.0 | 393 | 5 | 3 | 52+ files | CI/CD Pipeline |
| 2.1.0 | 393 | 5 | 3 | 52+ files | Async execution |
| 2.0.0 | 393 | 5 | 3 | 52+ files | Enterprise docs |
| 1.1.0 | 393 | 5 | 3 | 25+ files | - |
| 1.0.0 | 393 | 5 | 0 | 15+ files | - |
| 0.1.0 | 0 | 0 | 0 | 1 file | - |

---

## Migration Guides

### Upgrading from 1.x to 2.0

No breaking changes. Simply pull latest code:

```bash
git pull origin main
pip install -r experiments/pilot/requirements.txt
```

New documentation is additive and doesn't affect existing workflows.

### Upgrading from 0.x to 1.0

Complete rewrite. Start fresh with new installation:

```bash
git clone https://github.com/Kantosaurus/AAAI-2026.git
cd AAAI-2026
pip install -r experiments/pilot/requirements.txt
```

---

## Deprecation Notices

### Deprecated in 2.0
- None

### Removed in 2.0
- None

---

## Known Issues

### Current Issues
1. Token logprobs not available for API models (Claude, Gemini)
2. Checkpoint resume may not preserve exact random state
3. RAG grounding requires pre-built index

### Workarounds
1. Use local models for logprob-dependent analysis
2. Use same seed for reproducibility
3. Run `build_retrieval_index.py` before RAG experiments

---

## Contributors

- Research Team - Initial implementation
- Claude Code - Documentation assistance

---

## Links

- [Documentation Index](index.md)
- [GitHub Repository](https://github.com/Kantosaurus/AAAI-2026)
- [Issue Tracker](https://github.com/Kantosaurus/AAAI-2026/issues)

---

## Document Control

| Attribute | Value |
|-----------|-------|
| Document ID | DOC-CHANGELOG-001 |
| Version | 2.4 |
| Classification | Public |
| Author | Research Team |
| Approval Date | January 13, 2026 |
