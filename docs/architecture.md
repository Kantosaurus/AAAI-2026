# System Architecture Documentation

**Version:** 2.0
**Last Updated:** January 13, 2026
**Document ID:** DOC-ARCH-001

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [System Components](#2-system-components)
3. [Data Flow Architecture](#3-data-flow-architecture)
4. [Component Details](#4-component-details)
5. [Integration Patterns](#5-integration-patterns)
6. [Security Architecture](#6-security-architecture)
7. [Scalability Considerations](#7-scalability-considerations)
8. [Technology Stack](#8-technology-stack)

---

## 1. Architectural Overview

### 1.1 System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM Hallucination Research Framework                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   External   │    │   Research   │    │   Analysis   │                  │
│  │    Data      │───▶│   Pipeline   │───▶│   & Report   │                  │
│  │   Sources    │    │              │    │   Generation │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                        Core Infrastructure                        │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │      │
│  │  │  Benchmark │  │   Model    │  │ Annotation │  │ Mitigation │  │      │
│  │  │   Engine   │  │   Runner   │  │  Pipeline  │  │   Engine   │  │      │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

| Principle | Description |
|-----------|-------------|
| **Modularity** | Each component is self-contained with clear interfaces |
| **Configurability** | JSON-based configuration for all runtime parameters |
| **Resilience** | Error handling, retry logic, and checkpoint/resume support |
| **Reproducibility** | Deterministic seeding, versioned outputs, complete logging |
| **Safety-First** | All operations sanitized; no exploit execution |
| **Extensibility** | Easy to add new models, mitigations, or analysis methods |

### 1.3 High-Level Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                          │
│  CLI Tools | Jupyter Notebooks | Analysis Scripts                │
├─────────────────────────────────────────────────────────────────┤
│                      Application Layer                           │
│  Pilot Runner | Annotation Pipeline | Mitigation Engine          │
├─────────────────────────────────────────────────────────────────┤
│                       Service Layer                              │
│  Model Runners | Rate Limiters | Symbolic Checker | RAG System   │
├─────────────────────────────────────────────────────────────────┤
│                         Data Layer                               │
│  Benchmark Data | Results Storage | Annotation DB | Indexes      │
├─────────────────────────────────────────────────────────────────┤
│                      External Services                           │
│  Claude API | Gemini API | Hugging Face | NVD/MITRE             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. System Components

### 2.1 Component Overview

```
AAAI-2026/
├── data/                    # Data Layer
│   ├── prompts/            # Benchmark prompts and ground truth
│   ├── scripts/            # Data collection scripts
│   └── outputs/            # Generated datasets
│
├── experiments/             # Application Layer
│   ├── pilot/              # Main execution engine
│   ├── interpretability/   # Mechanistic analysis
│   ├── mitigations/        # Hallucination mitigations
│   └── integration/        # End-to-end workflows
│
├── annotations/             # Annotation Pipeline
│   ├── rubric.md           # Labeling guidelines
│   └── *.py                # Annotation utilities
│
├── results/                 # Output Storage
│   ├── pilot/              # Model responses
│   └── analysis/           # Analysis outputs
│
├── notebooks/               # Presentation Layer
│   └── analysis_template.py # Analysis notebooks
│
└── docs/                    # Documentation
    └── *.md                 # Technical documentation
```

### 2.2 Component Dependencies

```
                    ┌─────────────────┐
                    │   Benchmark     │
                    │   Dataset       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Pilot   │  │Annotation│  │Integration│
        │  Runner  │  │ Pipeline │  │  Testing │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             ▼             ▼             ▼
        ┌──────────────────────────────────────┐
        │           Model Runners               │
        │  Claude | Gemini | Transformers       │
        └────────────────┬─────────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐
        │Symbolic │ │   RAG   │ │Abstention│
        │Checker  │ │Grounding│ │Detector  │
        └─────────┘ └─────────┘ └─────────┘
              │          │          │
              └──────────┼──────────┘
                         ▼
                  ┌─────────────┐
                  │Interpretability│
                  │  Analysis     │
                  └─────────────┘
```

---

## 3. Data Flow Architecture

### 3.1 Primary Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   External  │     │  Benchmark  │     │   Pilot     │     │   Results   │
│   Sources   │────▶│  Generation │────▶│   Runner    │────▶│   Storage   │
│             │     │             │     │             │     │             │
│ - NVD       │     │ - 393       │     │ - 3,930     │     │ - JSON      │
│ - MITRE     │     │   prompts   │     │   responses │     │ - Logprobs  │
│ - Advisories│     │ - Ground    │     │ - Metadata  │     │ - Timing    │
│             │     │   truth     │     │ - Tokens    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                        ┌──────────────────────────┘
                                        ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Final     │     │  Mitigation │     │  Annotation │     │   Human     │
│   Report    │◀────│  Evaluation │◀────│  Processing │◀────│   Labeling  │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### 3.2 Data Format Specifications

#### 3.2.1 Benchmark Prompt Format

```json
{
  "id": "prompt_0001",
  "original_id": "cve_existence-REAL-001",
  "category": "cve_existence",
  "prompt": "Does CVE-2021-44228 exist? Provide details.",
  "is_synthetic_probe": false,
  "gold_label": {
    "exists": true,
    "description": "Log4Shell - Critical RCE in Apache Log4j"
  },
  "gold_refs": ["CVE-2021-44228"],
  "safety_note": null
}
```

#### 3.2.2 Pilot Result Format

```json
{
  "prompt_id": "prompt_0001",
  "model": "claude-3-5-sonnet-20241022",
  "model_version": "claude-3-5-sonnet-20241022",
  "full_response": "CVE-2021-44228 exists and is known as Log4Shell...",
  "tokens_used": {
    "input": 42,
    "output": 156,
    "total": 198
  },
  "token_logprobs": null,
  "sampling_params": {
    "temperature": 0.0,
    "seed": 42,
    "max_tokens": 2048
  },
  "timestamp": "2026-01-13T10:01:23.456789",
  "elapsed_seconds": 2.34,
  "run_id": "a1b2c3d4",
  "error": null,
  "prompt_category": "cve_existence",
  "is_synthetic_probe": false,
  "retry_count": 0
}
```

#### 3.2.3 Annotation Format

```csv
prompt_id,model,annotator,hallucination_binary,hallucination_types,severity,citation_correctness,notes
prompt_0001,claude-3-5-sonnet,annotator_1,0,none,N/A,correct,"All facts verified"
prompt_0042,claude-3-5-sonnet,annotator_1,1,fabricated_external_reference,high,fabricated,"CVE-2024-99999 does not exist"
```

---

## 4. Component Details

### 4.1 Pilot Runner (`run_pilot.py`)

**Purpose:** Execute LLM evaluations across multiple models with comprehensive logging

**Architecture:**

```
┌────────────────────────────────────────────────────────────────────┐
│                         PilotRunner                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │    Config    │  │  Checkpoint  │  │   Progress   │             │
│  │    Loader    │  │   Manager    │  │   Tracker    │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│           │               │               │                        │
│           └───────────────┼───────────────┘                        │
│                           ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                     ModelRunner Factory                      │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────────────┐   │  │
│  │  │  Claude   │  │  Gemini   │  │  LocalModelRunner     │   │  │
│  │  │  Runner   │  │  Runner   │  │  (Transformers)       │   │  │
│  │  └───────────┘  └───────────┘  └───────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                           │                                        │
│                           ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                      RateLimiter                             │  │
│  │              (Token Bucket Algorithm)                        │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

**Key Classes:**

| Class | Purpose | Lines |
|-------|---------|-------|
| `PilotRunner` | Orchestrates execution | 396-500 |
| `ModelRunner` | Abstract base for model execution | 108-156 |
| `ClaudeRunner` | Claude API integration | 163-206 |
| `GeminiRunner` | Gemini API integration | 209-261 |
| `LocalModelRunner` | Transformers integration | 264-374 |
| `RateLimiter` | Token bucket rate limiting | 80-106 |
| `PromptResult` | Structured output dataclass | 57-77 |

### 4.2 Symbolic Checker (`symbolic_checker.py`)

**Purpose:** Verify CVE IDs against authoritative databases

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                       SymbolicChecker                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  CVE Parser  │  │  NVD Index   │  │  MITRE Index │          │
│  │  (Regex)     │  │  Lookup      │  │  Lookup      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                │                 │                    │
│         └────────────────┼─────────────────┘                    │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Verification Engine                        ││
│  │  - Parse CVE patterns from response                         ││
│  │  - Check each against verified index                        ││
│  │  - Classify as verified/fabricated                          ││
│  │  - Optional: Replace fabricated with [UNKNOWN]              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `extract_cve_ids()` | Parse CVE-YYYY-NNNNN patterns |
| `verify_cve()` | Check against NVD index |
| `classify_citations()` | Separate verified/fabricated |
| `redact_fabricated()` | Replace with placeholder |

### 4.3 Abstention Detector (`abstention_detector.py`)

**Purpose:** Detect low-confidence responses for potential abstention

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     AbstentionDetector                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Detection Strategies                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │   Hedging    │  │   Logprob    │  │  Confidence  │   │   │
│  │  │   Phrases    │  │   Analysis   │  │    Scoring   │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Abstention Decision                         ││
│  │  if confidence < threshold:                                 ││
│  │      recommend_abstention = True                            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Detection Methods:**

| Method | Description | Weight |
|--------|-------------|--------|
| Hedging Analysis | Detects uncertainty phrases | 0.3 |
| Logprob Analysis | Analyzes token probabilities | 0.5 |
| Pattern Matching | Identifies refusal patterns | 0.2 |

### 4.4 Interpretability Framework

**Purpose:** Mechanistic analysis of hallucination causes

**Components:**

```
┌─────────────────────────────────────────────────────────────────┐
│                   Interpretability Pipeline                      │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │    Case      │────▶│   Causal     │────▶│  Activation  │    │
│  │  Selection   │     │   Tracing    │     │   Probing    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                              │                    │             │
│                              ▼                    ▼             │
│                       ┌─────────────────────────────────┐       │
│                       │       Analysis Results          │       │
│                       │  - Layer contributions          │       │
│                       │  - Critical attention heads     │       │
│                       │  - Probe AUC scores            │       │
│                       └─────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Integration Patterns

### 5.1 Model Integration Pattern

```python
# Abstract model runner pattern
class ModelRunner(ABC):
    """Abstract base for all model integrations"""

    def __init__(self, model_name: str, temperature: float,
                 seed: int, max_retries: int, rate_limiter: RateLimiter):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter

    @abstractmethod
    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Subclasses implement specific model calls"""
        pass

    def run_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Wrapper with retry logic and rate limiting"""
        self.rate_limiter.acquire()
        for retry in range(self.max_retries):
            try:
                return self._execute_prompt(prompt, prompt_id)
            except Exception as e:
                wait_time = (2 ** retry) * 1.0
                time.sleep(wait_time)
        return self._create_error_result(prompt_id, str(e))
```

### 5.2 Checkpoint Pattern

```python
# Checkpoint/resume pattern
class CheckpointManager:
    """Manages checkpoint state for resumable operations"""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.state = self._load_or_create()

    def _load_or_create(self) -> dict:
        if self.checkpoint_path.exists():
            return json.loads(self.checkpoint_path.read_text())
        return {"completed": {}, "last_model_index": 0}

    def mark_complete(self, model_key: str):
        self.state["completed"][model_key] = True
        self._save()

    def is_complete(self, model_key: str) -> bool:
        return self.state["completed"].get(model_key, False)

    def cleanup(self):
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
```

### 5.3 Rate Limiting Pattern

```python
# Token bucket rate limiter
class RateLimiter:
    """Token bucket algorithm for smooth rate limiting"""

    def __init__(self, requests_per_minute: int, burst_size: int = 10):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.max_tokens = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens,
                             self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            # Wait for token to become available
            wait_time = (1 - self.tokens) / self.rate
            time.sleep(wait_time)
            self.tokens = 0
```

---

## 6. Security Architecture

### 6.1 Security Principles

| Principle | Implementation |
|-----------|----------------|
| **Defense in Depth** | Multiple sanitization layers |
| **Least Privilege** | Minimal API permissions |
| **Input Validation** | All prompts sanitized before use |
| **Output Filtering** | No exploit code in outputs |
| **Audit Logging** | Complete execution traces |

### 6.2 Data Security

```
┌─────────────────────────────────────────────────────────────────┐
│                      Security Controls                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Input Sanitization                       │   │
│  │  - Remove PII                                            │   │
│  │  - Redact credentials                                    │   │
│  │  - Defang URLs                                           │   │
│  │  - Add defensive framing                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Runtime Controls                         │   │
│  │  - No code execution                                     │   │
│  │  - API key isolation (env vars)                          │   │
│  │  - Rate limiting                                         │   │
│  │  - Timeout enforcement                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Output Validation                        │   │
│  │  - Review for unsafe content                             │   │
│  │  - Flag fabricated CVEs                                  │   │
│  │  - Human review of samples                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 API Key Management

```bash
# Recommended: Environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."

# Alternative: .env file (excluded from git)
# .env
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# .gitignore must include:
.env
*.key
credentials/
```

---

## 7. Scalability Considerations

### 7.1 Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel Execution Options                    │
│                                                                  │
│  Option 1: Model Parallelism                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   GPU 1:     │  │   GPU 2:     │  │   GPU 3:     │          │
│  │   Qwen-14B   │  │   Mistral-7B │  │   Phi-3      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Option 2: Prompt Parallelism                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Process pool: 4-8 workers processing prompts            │   │
│  │  (For local models with sufficient VRAM)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Option 3: API Parallelism (Not Recommended)                    │
│  Rate limits make parallel API calls inefficient                │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Performance Optimization

| Optimization | Benefit | Trade-off |
|--------------|---------|-----------|
| FP16 Inference | 2x memory reduction | Minor precision loss |
| Token caching | Faster repeated prompts | Memory usage |
| Batch processing | Higher throughput | Latency increase |
| Async I/O | Better resource utilization | Complexity |

### 7.3 Resource Estimates

| Configuration | Prompts | Models | Time | Cost |
|---------------|---------|--------|------|------|
| Test run | 50 | 2 | 15 min | $2 |
| Small pilot | 100 | 3 | 1-2 hr | $5 |
| Full pilot | 393 | 10 | 10-14 hr | $35 |
| Extended | 1000+ | 10+ | 24+ hr | $100+ |

---

## 8. Technology Stack

### 8.1 Core Technologies

```
┌─────────────────────────────────────────────────────────────────┐
│                       Technology Stack                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Programming Language                                            │
│  └── Python 3.8+ (3.10+ recommended)                            │
│                                                                  │
│  LLM Frameworks                                                  │
│  ├── anthropic (Claude API)                                     │
│  ├── google-generativeai (Gemini API)                           │
│  └── transformers (Local models)                                │
│                                                                  │
│  ML/DL Stack                                                     │
│  ├── torch (PyTorch)                                            │
│  ├── transformer-lens (Interpretability)                        │
│  └── scikit-learn (Analysis)                                    │
│                                                                  │
│  Data Processing                                                 │
│  ├── pandas                                                      │
│  ├── numpy                                                       │
│  └── json (built-in)                                            │
│                                                                  │
│  Search & Retrieval                                              │
│  ├── faiss-cpu (Vector search)                                  │
│  └── sentence-transformers (Embeddings)                         │
│                                                                  │
│  Visualization                                                   │
│  ├── matplotlib                                                  │
│  ├── seaborn                                                     │
│  └── jupyter                                                     │
│                                                                  │
│  Utilities                                                       │
│  ├── tqdm (Progress bars)                                       │
│  ├── requests (HTTP)                                            │
│  └── dataclasses (built-in)                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Version Requirements

```
# requirements.txt (core)
anthropic>=0.18.0
google-generativeai>=0.3.0
transformers>=4.35.0
torch>=2.0.0
tqdm>=4.65.0

# requirements.txt (optional)
jupyter>=1.0.0
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.2.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
transformer-lens>=1.0.0
```

### 8.3 External Dependencies

| Service | Purpose | Rate Limit |
|---------|---------|------------|
| Anthropic Claude | LLM inference | 60 RPM |
| Google Gemini | LLM inference | Varies |
| Hugging Face | Model downloads | Unlimited |
| NVD API | CVE data | 5-50 RPM |
| MITRE ATT&CK | Threat data | Unlimited |

---

## Document Control

| Attribute | Value |
|-----------|-------|
| Document ID | DOC-ARCH-001 |
| Version | 2.0 |
| Classification | Internal |
| Author | Research Team |
| Approval Date | January 13, 2026 |

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 2.0 | 2026-01-13 | Complete architecture documentation | Research Team |
| 1.0 | 2025-11-06 | Initial architecture | Research Team |
