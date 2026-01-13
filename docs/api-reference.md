# API Reference Documentation

**Version:** 2.0
**Last Updated:** January 13, 2026
**Document ID:** DOC-API-001

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Classes](#2-core-classes)
3. [Pilot Runner API](#3-pilot-runner-api)
4. [Model Runner APIs](#4-model-runner-apis)
5. [Mitigation APIs](#5-mitigation-apis)
6. [Annotation APIs](#6-annotation-apis)
7. [Interpretability APIs](#7-interpretability-apis)
8. [Data Structures](#8-data-structures)
9. [Configuration Reference](#9-configuration-reference)
10. [CLI Reference](#10-cli-reference)

---

## 1. Overview

### 1.1 API Design Principles

| Principle | Description |
|-----------|-------------|
| **Type Safety** | All APIs use type hints and dataclasses |
| **Configuration-Driven** | JSON configuration for runtime parameters |
| **Error Handling** | Explicit exceptions with descriptive messages |
| **Logging** | Comprehensive logging at INFO and DEBUG levels |
| **Resumability** | Long-running operations support checkpoint/resume |

### 1.2 Module Structure

```python
# Import patterns
from experiments.pilot.run_pilot import PilotRunner, PromptResult
from experiments.mitigations.symbolic_checker import SymbolicChecker
from experiments.mitigations.abstention_detector import AbstentionDetector
from annotations.compute_agreement import compute_cohens_kappa
```

---

## 2. Core Classes

### 2.1 PromptResult

**Location:** `experiments/pilot/run_pilot.py:57-77`

**Purpose:** Structured container for model response data

```python
@dataclass
class PromptResult:
    """Container for a single prompt's execution result.

    Attributes:
        prompt_id: Unique identifier from benchmark
        model: Model name/identifier
        model_version: Exact model version used
        full_response: Complete text response from model
        tokens_used: Dict with input/output/total token counts
        token_logprobs: List of token probability dicts (local models only)
        sampling_params: Dict with temperature, seed, max_tokens
        timestamp: ISO 8601 execution timestamp
        elapsed_seconds: Execution time in seconds
        run_id: Unique run identifier
        error: Error message if failed, None otherwise
        prompt_category: Category from benchmark
        is_synthetic_probe: Whether this is a hallucination probe
        retry_count: Number of retries needed
    """
    prompt_id: str
    model: str
    model_version: str
    full_response: str
    tokens_used: Dict[str, int]
    token_logprobs: Optional[List[Dict]] = None
    sampling_params: Optional[Dict] = None
    timestamp: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    run_id: Optional[str] = None
    error: Optional[str] = None
    prompt_category: Optional[str] = None
    is_synthetic_probe: Optional[bool] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptResult':
        """Create instance from dictionary."""
        return cls(**data)
```

**Usage Example:**

```python
result = PromptResult(
    prompt_id="prompt_0001",
    model="claude-3-5-sonnet-20241022",
    model_version="claude-3-5-sonnet-20241022",
    full_response="CVE-2021-44228 is Log4Shell...",
    tokens_used={"input": 42, "output": 156, "total": 198},
    elapsed_seconds=2.34
)

# Serialize to JSON
result_dict = result.to_dict()
json.dumps(result_dict)
```

### 2.2 RateLimiter

**Location:** `experiments/pilot/run_pilot.py:80-106`

**Purpose:** Token bucket rate limiting for API calls

```python
class RateLimiter:
    """Token bucket rate limiter for smooth API rate limiting.

    Args:
        requests_per_minute: Maximum sustained request rate
        burst_size: Maximum burst capacity (default: 10)

    Example:
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        limiter.acquire()  # Blocks until token available
        make_api_call()
    """

    def __init__(self, requests_per_minute: int, burst_size: int = 10):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.max_tokens = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """Acquire a token, blocking if necessary.

        Thread-safe. Blocks until a token becomes available.
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens,
                             self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.rate
            time.sleep(wait_time)
            self.tokens = 0

    def get_available_tokens(self) -> float:
        """Return current available tokens (for monitoring)."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            return min(self.max_tokens,
                      self.tokens + elapsed * self.rate)
```

**Usage Example:**

```python
# Create limiter: 30 requests/minute with burst of 5
limiter = RateLimiter(requests_per_minute=30, burst_size=5)

# Use in loop
for prompt in prompts:
    limiter.acquire()  # Will block if rate exceeded
    response = api.call(prompt)
```

---

## 3. Pilot Runner API

### 3.1 PilotRunner

**Location:** `experiments/pilot/run_pilot.py:396-500`

**Purpose:** Orchestrate pilot execution across multiple models

```python
class PilotRunner:
    """Main orchestration class for pilot runs.

    Args:
        prompts_file: Path to benchmark JSON file
        output_dir: Directory for output files
        config: Configuration dictionary

    Example:
        runner = PilotRunner(
            prompts_file=Path("data/prompts/hallu-sec-benchmark.json"),
            output_dir=Path("results/pilot"),
            config=config_dict
        )
        runner.run(resume=False)
    """

    def __init__(self, prompts_file: Path, output_dir: Path, config: Dict):
        self.prompts_file = prompts_file
        self.output_dir = output_dir
        self.config = config
        self.checkpoint_path = output_dir / "checkpoint.json"

    def run(self, resume: bool = False) -> None:
        """Execute pilot across all configured models.

        Args:
            resume: If True, resume from checkpoint

        Raises:
            FileNotFoundError: If prompts file doesn't exist
            ValueError: If configuration is invalid
        """
        pass

    def load_prompts(self, num_prompts: Optional[int] = None) -> List[Dict]:
        """Load and optionally subset prompts.

        Args:
            num_prompts: Number of prompts to load (None = all)

        Returns:
            List of prompt dictionaries
        """
        pass

    def load_checkpoint(self) -> Dict:
        """Load checkpoint state if exists.

        Returns:
            Checkpoint dictionary with 'completed' and 'last_model_index'
        """
        pass

    def save_checkpoint(self, completed: Dict[str, bool], last_index: int) -> None:
        """Save checkpoint state.

        Args:
            completed: Dict mapping model_key to completion status
            last_index: Index of last processed model
        """
        pass

    def cleanup_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion."""
        pass

    def _create_runner(self, model_config: Dict) -> 'ModelRunner':
        """Factory method to create appropriate ModelRunner.

        Args:
            model_config: Model configuration dict

        Returns:
            Appropriate ModelRunner subclass instance
        """
        pass

    def _save_results(self, run_results: List[Dict], timestamp: str) -> Path:
        """Save results to JSON file.

        Args:
            run_results: List of run result dictionaries
            timestamp: ISO timestamp for filename

        Returns:
            Path to saved file
        """
        pass
```

### 3.2 PilotRunner Configuration

```python
# Configuration schema
config_schema = {
    "description": str,           # Human-readable description
    "prompts_file": str,          # Path to prompts JSON
    "output_dir": str,            # Output directory
    "num_prompts": int,           # Number of prompts (optional)
    "seed": int,                  # Random seed
    "max_retries": int,           # Retry attempts
    "requests_per_minute": int,   # Rate limit
    "models": [                   # Model configurations
        {
            "name": str,          # Model name
            "type": str,          # "claude" | "gemini" | "local"
            "temperature": float, # 0.0 to 1.0
            "api_key": str,       # API key or env var
            "device": str,        # "cuda" | "cpu" (local only)
            "model_path": str,    # HF path (local only)
            "notes": str          # Optional notes
        }
    ]
}
```

---

## 4. Model Runner APIs

### 4.1 ModelRunner (Abstract Base)

**Location:** `experiments/pilot/run_pilot.py:108-156`

```python
class ModelRunner(ABC):
    """Abstract base class for model execution.

    Subclasses must implement _execute_prompt().

    Args:
        model_name: Model identifier
        temperature: Sampling temperature (0.0-1.0)
        seed: Random seed for reproducibility
        max_retries: Maximum retry attempts
        rate_limiter: RateLimiter instance
    """

    def __init__(self, model_name: str, temperature: float,
                 seed: int, max_retries: int, rate_limiter: RateLimiter):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter

    @abstractmethod
    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute a single prompt. Must be implemented by subclasses.

        Args:
            prompt: Prompt text
            prompt_id: Prompt identifier

        Returns:
            PromptResult with response data
        """
        pass

    def run_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt with retry logic.

        Args:
            prompt: Prompt text
            prompt_id: Prompt identifier

        Returns:
            PromptResult (may contain error if all retries failed)
        """
        self.rate_limiter.acquire()
        last_error = None

        for retry in range(self.max_retries):
            try:
                result = self._execute_prompt(prompt, prompt_id)
                result.retry_count = retry
                return result
            except Exception as e:
                last_error = e
                wait_time = (2 ** retry) * 1.0
                time.sleep(wait_time)

        return self._create_error_result(prompt_id, str(last_error))

    def _create_error_result(self, prompt_id: str, error: str) -> PromptResult:
        """Create error result after all retries exhausted."""
        return PromptResult(
            prompt_id=prompt_id,
            model=self.model_name,
            model_version=self.model_name,
            full_response="",
            tokens_used={"input": 0, "output": 0, "total": 0},
            error=error,
            retry_count=self.max_retries
        )
```

### 4.2 ClaudeRunner

**Location:** `experiments/pilot/run_pilot.py:163-206`

```python
class ClaudeRunner(ModelRunner):
    """Runner for Claude API (Anthropic).

    Args:
        api_key: Anthropic API key
        **kwargs: Passed to ModelRunner

    Example:
        runner = ClaudeRunner(
            model_name="claude-3-5-sonnet-20241022",
            temperature=0.0,
            seed=42,
            max_retries=3,
            rate_limiter=limiter,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        result = runner.run_prompt("What is CVE-2021-44228?", "p001")
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)

    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt via Claude API.

        Returns:
            PromptResult with response and token counts
            Note: token_logprobs is always None (API limitation)
        """
        start_time = time.time()

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        elapsed = time.time() - start_time

        return PromptResult(
            prompt_id=prompt_id,
            model=self.model_name,
            model_version=self.model_name,
            full_response=message.content[0].text,
            tokens_used={
                "input": message.usage.input_tokens,
                "output": message.usage.output_tokens,
                "total": message.usage.input_tokens + message.usage.output_tokens
            },
            token_logprobs=None,  # Not available via API
            sampling_params={
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": 2048
            },
            timestamp=datetime.now().isoformat(),
            elapsed_seconds=elapsed,
            run_id=str(uuid.uuid4())[:8]
        )
```

### 4.3 GeminiRunner

**Location:** `experiments/pilot/run_pilot.py:209-261`

```python
class GeminiRunner(ModelRunner):
    """Runner for Gemini API (Google).

    Args:
        api_key: Google API key
        **kwargs: Passed to ModelRunner

    Example:
        runner = GeminiRunner(
            model_name="gemini-1.5-pro",
            temperature=0.0,
            seed=42,
            max_retries=3,
            rate_limiter=limiter,
            api_key=os.environ["GOOGLE_API_KEY"]
        )
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt via Gemini API."""
        pass  # Implementation similar to ClaudeRunner
```

### 4.4 LocalModelRunner

**Location:** `experiments/pilot/run_pilot.py:264-374`

```python
class LocalModelRunner(ModelRunner):
    """Runner for local Transformers models.

    Supports token probability logging for interpretability.

    Args:
        model_path: Hugging Face model path
        device: "cuda" or "cpu"
        **kwargs: Passed to ModelRunner

    Example:
        runner = LocalModelRunner(
            model_name="qwen-14b",
            model_path="Qwen/Qwen2.5-14B-Instruct",
            device="cuda",
            temperature=0.0,
            seed=42,
            max_retries=3,
            rate_limiter=limiter
        )
    """

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def _execute_prompt(self, prompt: str, prompt_id: str) -> PromptResult:
        """Execute prompt locally with logprob extraction.

        Returns:
            PromptResult with token_logprobs populated (first 50 tokens)
        """
        pass

    def _extract_logprobs(self, outputs, num_tokens: int = 50) -> List[Dict]:
        """Extract top-5 token probabilities.

        Args:
            outputs: Model generation outputs
            num_tokens: Number of tokens to extract (default: 50)

        Returns:
            List of dicts with token_position, top_tokens, top_logprobs
        """
        logprobs = []
        for i in range(min(num_tokens, len(outputs.scores))):
            probs = torch.softmax(outputs.scores[i][0], dim=-1)
            top5 = torch.topk(probs, 5)
            logprobs.append({
                "token_position": i,
                "top_tokens": [self.tokenizer.decode(t) for t in top5.indices],
                "top_logprobs": [float(torch.log(p)) for p in top5.values]
            })
        return logprobs
```

---

## 5. Mitigation APIs

### 5.1 SymbolicChecker

**Location:** `experiments/mitigations/symbolic_checker.py`

```python
class SymbolicChecker:
    """Verify CVE citations against authoritative databases.

    Args:
        nvd_index_path: Path to NVD index file
        mitre_index_path: Optional path to MITRE index

    Example:
        checker = SymbolicChecker(nvd_index_path="nvd_index.json")
        result = checker.check_response(response_text)
    """

    def __init__(self, nvd_index_path: str,
                 mitre_index_path: Optional[str] = None):
        self.nvd_ids = self._load_index(nvd_index_path)
        self.mitre_ids = self._load_index(mitre_index_path) if mitre_index_path else set()

    def extract_cve_ids(self, text: str) -> List[str]:
        """Extract CVE IDs from text.

        Args:
            text: Response text to parse

        Returns:
            List of CVE-YYYY-NNNNN patterns found
        """
        pattern = r'CVE-\d{4}-\d{4,7}'
        return list(set(re.findall(pattern, text)))

    def verify_cve(self, cve_id: str) -> bool:
        """Check if CVE exists in index.

        Args:
            cve_id: CVE identifier

        Returns:
            True if CVE exists in NVD index
        """
        return cve_id in self.nvd_ids

    def check_response(self, response: str) -> Dict[str, Any]:
        """Check all CVE citations in a response.

        Args:
            response: Model response text

        Returns:
            Dict with:
                - total_cves: Number of CVEs found
                - verified: List of verified CVE IDs
                - fabricated: List of fabricated CVE IDs
                - fabrication_rate: Fraction fabricated
        """
        cve_ids = self.extract_cve_ids(response)
        verified = [cve for cve in cve_ids if self.verify_cve(cve)]
        fabricated = [cve for cve in cve_ids if not self.verify_cve(cve)]

        return {
            "total_cves": len(cve_ids),
            "verified": verified,
            "fabricated": fabricated,
            "fabrication_rate": len(fabricated) / len(cve_ids) if cve_ids else 0.0
        }

    def redact_fabricated(self, response: str,
                          placeholder: str = "[UNKNOWN CVE]") -> str:
        """Replace fabricated CVE IDs with placeholder.

        Args:
            response: Original response text
            placeholder: Replacement text

        Returns:
            Response with fabricated CVEs replaced
        """
        result = self.check_response(response)
        text = response
        for cve in result["fabricated"]:
            text = text.replace(cve, placeholder)
        return text
```

### 5.2 AbstentionDetector

**Location:** `experiments/mitigations/abstention_detector.py`

```python
class AbstentionDetector:
    """Detect low-confidence responses for potential abstention.

    Args:
        threshold: Confidence threshold (0.0-1.0)
        hedging_phrases: Optional custom hedging phrases

    Example:
        detector = AbstentionDetector(threshold=0.3)
        should_abstain, confidence = detector.analyze(response, logprobs)
    """

    # Default hedging phrases
    DEFAULT_HEDGING_PHRASES = [
        "I'm not sure",
        "I cannot verify",
        "may be",
        "possibly",
        "might be",
        "I don't have information",
        "uncertain",
        "I cannot confirm"
    ]

    def __init__(self, threshold: float = 0.3,
                 hedging_phrases: Optional[List[str]] = None):
        self.threshold = threshold
        self.hedging_phrases = hedging_phrases or self.DEFAULT_HEDGING_PHRASES

    def analyze_hedging(self, response: str) -> Tuple[bool, float]:
        """Analyze response for hedging phrases.

        Args:
            response: Model response text

        Returns:
            Tuple of (contains_hedging, hedging_score)
        """
        response_lower = response.lower()
        matches = sum(1 for phrase in self.hedging_phrases
                     if phrase.lower() in response_lower)
        score = min(1.0, matches / 3)  # Normalize
        return matches > 0, score

    def analyze_logprobs(self, logprobs: List[Dict]) -> Tuple[bool, float]:
        """Analyze token probabilities for uncertainty.

        Args:
            logprobs: Token probability data from LocalModelRunner

        Returns:
            Tuple of (low_confidence, average_entropy)
        """
        if not logprobs:
            return False, 0.5  # Default neutral

        # Calculate average entropy of first token predictions
        entropies = []
        for token_data in logprobs[:10]:  # First 10 tokens
            probs = [np.exp(lp) for lp in token_data.get("top_logprobs", [])]
            if probs:
                entropy = -sum(p * np.log(p + 1e-10) for p in probs)
                entropies.append(entropy)

        avg_entropy = np.mean(entropies) if entropies else 0.5
        normalized = min(1.0, avg_entropy / 2.0)  # Normalize to 0-1
        return normalized > 0.7, normalized

    def analyze(self, response: str,
                logprobs: Optional[List[Dict]] = None) -> Tuple[bool, float]:
        """Full analysis for abstention recommendation.

        Args:
            response: Model response text
            logprobs: Optional token probability data

        Returns:
            Tuple of (should_abstain, confidence_score)
        """
        # Hedging analysis (weight: 0.4)
        has_hedging, hedge_score = self.analyze_hedging(response)

        # Logprob analysis (weight: 0.6)
        if logprobs:
            low_conf, entropy_score = self.analyze_logprobs(logprobs)
            combined = 0.4 * hedge_score + 0.6 * entropy_score
        else:
            combined = hedge_score
            low_conf = has_hedging

        confidence = 1.0 - combined
        should_abstain = confidence < self.threshold

        return should_abstain, confidence
```

### 5.3 RAGGrounding

**Location:** `experiments/mitigations/rag_grounding.py`

```python
class RAGGrounding:
    """Retrieval-augmented generation for grounding responses.

    Args:
        index_path: Path to FAISS index
        metadata_path: Path to document metadata
        top_k: Number of documents to retrieve

    Example:
        rag = RAGGrounding(
            index_path="retrieval_index.faiss",
            metadata_path="doc_metadata.json",
            top_k=3
        )
        augmented_prompt = rag.augment_prompt(prompt)
    """

    def __init__(self, index_path: str, metadata_path: str, top_k: int = 3):
        self.index = faiss.read_index(index_path)
        self.metadata = json.loads(Path(metadata_path).read_text())
        self.top_k = top_k
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant documents for query.

        Args:
            query: Search query

        Returns:
            List of document dicts with content and metadata
        """
        embedding = self.encoder.encode([query])
        distances, indices = self.index.search(embedding, self.top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                doc = self.metadata[idx].copy()
                doc["score"] = float(distances[0][i])
                results.append(doc)

        return results

    def augment_prompt(self, prompt: str) -> str:
        """Augment prompt with retrieved context.

        Args:
            prompt: Original prompt

        Returns:
            Prompt with grounding context prepended
        """
        docs = self.retrieve(prompt)

        if not docs:
            return prompt

        context = "Reference information:\n"
        for i, doc in enumerate(docs, 1):
            context += f"{i}. {doc['content'][:500]}...\n"

        return f"{context}\n\nQuestion: {prompt}\n\nProvide your answer, citing the reference information above when applicable."
```

---

## 6. Annotation APIs

### 6.1 PrepareAnnotationBatches

**Location:** `annotations/prepare_annotation_batches.py`

```python
def prepare_batches(results_dir: Path,
                    output_dir: Path,
                    num_annotators: int = 2,
                    overlap: float = 1.0,
                    seed: int = 42) -> Dict[str, Path]:
    """Prepare randomized annotation batches.

    Args:
        results_dir: Directory containing pilot results
        output_dir: Directory for batch files
        num_annotators: Number of annotators
        overlap: Fraction of overlap between annotators (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping annotator_id to batch file path

    Example:
        batches = prepare_batches(
            results_dir=Path("results/pilot"),
            output_dir=Path("annotations/batches"),
            num_annotators=2,
            overlap=1.0
        )
    """
    pass
```

### 6.2 ComputeAgreement

**Location:** `annotations/compute_agreement.py`

```python
def compute_cohens_kappa(annotations_a: List[int],
                         annotations_b: List[int]) -> float:
    """Compute Cohen's kappa for inter-annotator agreement.

    Args:
        annotations_a: Binary labels from annotator A (0 or 1)
        annotations_b: Binary labels from annotator B (0 or 1)

    Returns:
        Kappa coefficient (-1 to 1, >0.6 is good agreement)

    Raises:
        ValueError: If lists have different lengths
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError("Annotation lists must have same length")

    # Calculate observed agreement
    agree = sum(a == b for a, b in zip(annotations_a, annotations_b))
    p_o = agree / len(annotations_a)

    # Calculate expected agreement by chance
    a_pos = sum(annotations_a) / len(annotations_a)
    b_pos = sum(annotations_b) / len(annotations_b)
    p_e = (a_pos * b_pos) + ((1 - a_pos) * (1 - b_pos))

    # Calculate kappa
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def compute_agreement_report(annotation_files: List[Path]) -> Dict[str, Any]:
    """Compute comprehensive agreement metrics.

    Args:
        annotation_files: List of paths to annotation CSVs

    Returns:
        Dict with:
            - cohens_kappa: Kappa coefficient
            - percent_agreement: Raw agreement rate
            - disagreement_analysis: Breakdown of disagreement types
            - per_category_kappa: Kappa by prompt category
    """
    pass
```

---

## 7. Interpretability APIs

### 7.1 CausalTracing

**Location:** `experiments/interpretability/causal_tracing.py`

```python
class CausalTracer:
    """Causal tracing for identifying hallucination-critical components.

    Based on Meng et al. (2022) methodology.

    Args:
        model: Transformers model
        tokenizer: Model tokenizer
        device: Computation device

    Example:
        tracer = CausalTracer(model, tokenizer, "cuda")
        results = tracer.trace(prompt, hallucinated_token_idx=15)
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def corrupt_activations(self, activations: torch.Tensor,
                           noise_level: float = 3.0) -> torch.Tensor:
        """Add noise to activations for corruption.

        Args:
            activations: Original activation tensor
            noise_level: Standard deviations of noise

        Returns:
            Corrupted activations
        """
        noise = torch.randn_like(activations) * noise_level
        return activations + noise

    def restore_layer(self, layer_idx: int,
                     clean_activations: torch.Tensor) -> Callable:
        """Create hook to restore clean activations at layer.

        Args:
            layer_idx: Layer index
            clean_activations: Activations to restore

        Returns:
            Hook function
        """
        def hook(module, input, output):
            return clean_activations
        return hook

    def trace(self, prompt: str,
             hallucinated_token_idx: int) -> Dict[str, Any]:
        """Run causal tracing experiment.

        Args:
            prompt: Input prompt
            hallucinated_token_idx: Index of hallucinated token

        Returns:
            Dict with per-layer effect sizes and critical layers
        """
        pass


def select_cases_for_interp(annotations_path: Path,
                            results_dir: Path,
                            n_cases: int = 30) -> List[Dict]:
    """Select diverse hallucination cases for analysis.

    Args:
        annotations_path: Path to final annotations
        results_dir: Path to pilot results
        n_cases: Number of cases to select

    Returns:
        List of case dicts with prompt, response, and metadata
    """
    pass
```

### 7.2 ActivationProbes

**Location:** `experiments/interpretability/activation_probes.py`

```python
class ActivationProbe:
    """Linear probe for detecting hallucination features.

    Args:
        layer_idx: Layer to probe
        hidden_dim: Hidden dimension size

    Example:
        probe = ActivationProbe(layer_idx=24, hidden_dim=4096)
        probe.train(train_activations, train_labels)
        auc = probe.evaluate(test_activations, test_labels)
    """

    def __init__(self, layer_idx: int, hidden_dim: int):
        self.layer_idx = layer_idx
        self.probe = LogisticRegression(max_iter=1000)

    def extract_activations(self, model, tokenizer,
                           prompts: List[str],
                           token_idx: int = -1) -> np.ndarray:
        """Extract activations from model.

        Args:
            model: Transformers model
            tokenizer: Model tokenizer
            prompts: List of prompts
            token_idx: Token position to extract (-1 = last)

        Returns:
            Array of shape (n_prompts, hidden_dim)
        """
        pass

    def train(self, activations: np.ndarray, labels: np.ndarray) -> None:
        """Train probe on labeled activations.

        Args:
            activations: Shape (n_samples, hidden_dim)
            labels: Binary labels (0 = no hallucination, 1 = hallucination)
        """
        self.probe.fit(activations, labels)

    def evaluate(self, activations: np.ndarray,
                labels: np.ndarray) -> Dict[str, float]:
        """Evaluate probe performance.

        Args:
            activations: Test activations
            labels: Test labels

        Returns:
            Dict with auc, accuracy, precision, recall
        """
        predictions = self.probe.predict_proba(activations)[:, 1]
        auc = roc_auc_score(labels, predictions)
        accuracy = self.probe.score(activations, labels)

        return {
            "auc": auc,
            "accuracy": accuracy,
            "layer": self.layer_idx
        }
```

---

## 8. Data Structures

### 8.1 Benchmark Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "HalluSecBenchmark",
  "type": "object",
  "properties": {
    "metadata": {
      "type": "object",
      "properties": {
        "version": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "total_prompts": {"type": "integer"},
        "categories": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["version", "total_prompts"]
    },
    "prompts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "original_id": {"type": "string"},
          "category": {"type": "string"},
          "prompt": {"type": "string"},
          "is_synthetic_probe": {"type": "boolean"},
          "gold_label": {
            "type": "object",
            "properties": {
              "exists": {"type": ["boolean", "null"]},
              "description": {"type": "string"}
            }
          },
          "gold_refs": {"type": "array", "items": {"type": "string"}},
          "safety_note": {"type": ["string", "null"]}
        },
        "required": ["id", "category", "prompt", "is_synthetic_probe"]
      }
    }
  },
  "required": ["metadata", "prompts"]
}
```

### 8.2 Results Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PilotResults",
  "type": "object",
  "properties": {
    "metadata": {
      "type": "object",
      "properties": {
        "start_time": {"type": "string", "format": "date-time"},
        "end_time": {"type": "string", "format": "date-time"},
        "config": {"type": "object"},
        "total_prompts": {"type": "integer"}
      }
    },
    "runs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "model_config": {"type": "object"},
          "results": {
            "type": "array",
            "items": {"$ref": "#/definitions/PromptResult"}
          },
          "completed_at": {"type": "string", "format": "date-time"}
        }
      }
    }
  },
  "definitions": {
    "PromptResult": {
      "type": "object",
      "properties": {
        "prompt_id": {"type": "string"},
        "model": {"type": "string"},
        "full_response": {"type": "string"},
        "tokens_used": {"type": "object"},
        "token_logprobs": {"type": ["array", "null"]},
        "sampling_params": {"type": "object"},
        "timestamp": {"type": "string"},
        "elapsed_seconds": {"type": "number"},
        "error": {"type": ["string", "null"]}
      }
    }
  }
}
```

---

## 9. Configuration Reference

### 9.1 Full Pilot Configuration

```json
{
  "description": "Full pilot configuration - all models, all prompts",
  "prompts_file": "../../data/prompts/hallu-sec-benchmark.json",
  "output_dir": "../../results/pilot",
  "num_prompts": null,
  "seed": 42,
  "max_retries": 3,
  "requests_per_minute": 60,
  "models": [
    {
      "name": "claude-3-5-sonnet-20241022",
      "type": "claude",
      "temperature": 0.0,
      "notes": "High-capability baseline (deterministic)"
    },
    {
      "name": "claude-3-5-sonnet-20241022",
      "type": "claude",
      "temperature": 0.7,
      "notes": "High-capability baseline (exploratory)"
    },
    {
      "name": "gemini-1.5-pro",
      "type": "gemini",
      "temperature": 0.0,
      "notes": "Alternative API (deterministic)"
    },
    {
      "name": "gemini-1.5-pro",
      "type": "gemini",
      "temperature": 0.7,
      "notes": "Alternative API (exploratory)"
    },
    {
      "name": "Qwen/Qwen2.5-14B-Instruct",
      "type": "local",
      "model_path": "Qwen/Qwen2.5-14B-Instruct",
      "device": "cuda",
      "temperature": 0.0,
      "notes": "Main interpretability model"
    },
    {
      "name": "Qwen/Qwen2.5-14B-Instruct",
      "type": "local",
      "model_path": "Qwen/Qwen2.5-14B-Instruct",
      "device": "cuda",
      "temperature": 0.7,
      "notes": "Main interpretability model (exploratory)"
    },
    {
      "name": "mistralai/Mistral-7B-Instruct-v0.3",
      "type": "local",
      "model_path": "mistralai/Mistral-7B-Instruct-v0.3",
      "device": "cuda",
      "temperature": 0.0,
      "notes": "Medium scaling"
    },
    {
      "name": "mistralai/Mistral-7B-Instruct-v0.3",
      "type": "local",
      "model_path": "mistralai/Mistral-7B-Instruct-v0.3",
      "device": "cuda",
      "temperature": 0.7,
      "notes": "Medium scaling (exploratory)"
    },
    {
      "name": "microsoft/Phi-3-mini-128k-instruct",
      "type": "local",
      "model_path": "microsoft/Phi-3-mini-128k-instruct",
      "device": "cuda",
      "temperature": 0.0,
      "notes": "Small baseline"
    },
    {
      "name": "microsoft/Phi-3-mini-128k-instruct",
      "type": "local",
      "model_path": "microsoft/Phi-3-mini-128k-instruct",
      "device": "cuda",
      "temperature": 0.7,
      "notes": "Small baseline (exploratory)"
    }
  ]
}
```

---

## 10. CLI Reference

### 10.1 run_pilot.py

```
usage: run_pilot.py [-h] --config CONFIG [--prompts PROMPTS]
                    [--output OUTPUT] [--num-prompts NUM]
                    [--resume]

Run LLM hallucination pilot experiments

options:
  -h, --help           show this help message and exit
  --config CONFIG      Path to configuration JSON file
  --prompts PROMPTS    Override prompts file from config
  --output OUTPUT      Override output directory from config
  --num-prompts NUM    Number of prompts to run (default: all)
  --resume             Resume from checkpoint

examples:
  # Run test configuration
  python run_pilot.py --config config_small_test.json

  # Run full pilot
  python run_pilot.py --config config_full_pilot.json

  # Run subset of prompts
  python run_pilot.py --config config_full_pilot.json --num-prompts 100

  # Resume interrupted run
  python run_pilot.py --config config_full_pilot.json --resume
```

### 10.2 symbolic_checker.py

```
usage: symbolic_checker.py [-h] --results RESULTS --nvd-index INDEX
                           [--output OUTPUT] [--redact]

Check CVE citations against NVD database

options:
  -h, --help              show this help message and exit
  --results RESULTS       Path to pilot results (supports glob)
  --nvd-index INDEX       Path to NVD index file
  --output OUTPUT         Output file for results
  --redact                Replace fabricated CVEs with placeholder

examples:
  # Check results
  python symbolic_checker.py \
    --results ../../results/pilot/pilot_*.json \
    --nvd-index nvd_index.json \
    --output results/symbolic_check.json

  # Check and redact
  python symbolic_checker.py \
    --results ../../results/pilot/pilot_claude.json \
    --nvd-index nvd_index.json \
    --output results/redacted.json \
    --redact
```

### 10.3 prepare_annotation_batches.py

```
usage: prepare_annotation_batches.py [-h] --results RESULTS
                                     --output OUTPUT
                                     [--num-annotators N]
                                     [--overlap FLOAT]
                                     [--seed SEED]

Prepare annotation batches from pilot results

options:
  -h, --help           show this help message and exit
  --results RESULTS    Directory with pilot results
  --output OUTPUT      Output directory for batches
  --num-annotators N   Number of annotators (default: 2)
  --overlap FLOAT      Overlap fraction (default: 1.0)
  --seed SEED          Random seed (default: 42)

examples:
  python prepare_annotation_batches.py \
    --results ../results/pilot \
    --output batches/ \
    --num-annotators 2 \
    --overlap 1.0
```

---

## Document Control

| Attribute | Value |
|-----------|-------|
| Document ID | DOC-API-001 |
| Version | 2.0 |
| Classification | Internal |
| Author | Research Team |
| Approval Date | January 13, 2026 |

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 2.0 | 2026-01-13 | Complete API documentation | Research Team |
| 1.0 | 2025-11-06 | Initial API reference | Research Team |
