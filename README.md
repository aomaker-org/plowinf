# Path: plowinf/README.md
# plowinf: Deep Dive into Local AI Inference & Code Processing
# Max Column: 80 Columns

Autonomous offline Small Language Model (SLM) inference laboratory, code
processor, and file-backed task queue processor for edge, developer
workstation, and zero-cloud environments.

Repository: <https://github.com/aomaker-org/plowinf>

---

## Key Features

- **Multi-Backend Inference Router**:
  - **Ollama**: Native model tags (`qwen2:1.5b`), dynamic
    `.gguf` auto-registration, and real-time token streaming.
  - **OpenAI-Compatible**: Integrates with vLLM, LM Studio, and LiteRT-LM.
  - **llama2.c**: Zero-dependency C runner execution on CPU (~23+ tok/s).
  - **llama-cli**: Direct quantized `.gguf` evaluation via llama.cpp.
  - **Mock Backend**: Deterministic offline verification for CI pipelines.
- **Inference Benchmarking & Observability**:
  - Built-in latency, token throughput (tok/s), and memory RSS profiling via
    `plowinf benchmark`.
  - Cryptographic audit trail logging every prompt, model response, and SHA256
    hash (`logs/forensics/audit_journal.jsonl`).
- **Autonomous Code Processing**:
  - Interactive or queued code review (`plowinf review`).
  - Unit test generation with pytest scaffolding (`plowinf test`).
  - PEP 257 compliant docstring generation (`plowinf docstring`).
  - Unified diff preview and safe patch application (`plowinf apply`).
- **Atomic File-Backed Queue**:
  - Strict POSIX file locking (`fcntl.flock`) preventing task collisions.
  - State partitioning (`incoming/`, `processing/`, `done/`, `failed/`).
- **Zero-Cloud Survivability & Air-Gap Mandate**:
  - Operates 100% offline without remote network access (Directive 18).
  - External AI API traffic blocked; strictly localhost loopback interfaces.

---

## Quickstart

### 1. Environment Setup (Mandatory `uv`)

```bash
cd /home/fekerr/src/plowinf
uv venv
uv pip install -e ".[dev]"
```

### 2. Verify Backend Availability

```bash
uv run plowinf backends
```

### 3. Run Inference Benchmark

```bash
uv run plowinf benchmark --runs 3
```

### 4. Run Interactive Code Review

```bash
uv run plowinf review src/plowinf/queue.py
```

### 5. Run Unit Test Generation

```bash
uv run plowinf test src/plowinf/backends.py
```

### 6. Start Background Daemon

```bash
uv run plowinf daemon --interval 2.0
```

---

## Testing

Execute the test suite using pytest via `uv`:

```bash
uv run pytest
```

or via Makefile:

```bash
make test
```

---

## Repository Structure

```
plowinf/
├── src/
│   ├── plowinf/             # Core inference & processing package
│   │   ├── backends.py      # Multi-backend inference adapters (Ollama, vLLM, llama.cpp)
│   │   ├── cli.py           # plowinf command-line interface
│   │   ├── forensics.py     # SHA-256 cryptographic audit ledger & telemetry
│   │   ├── patcher.py       # Safe diff and patch application engine
│   │   ├── queue.py         # Atomic POSIX file-locked queue
│   │   ├── safety.py        # Air-gap enforcer & additive Git safety
│   │   └── worker.py        # Code worker daemon & prompt builder
│   └── local_code_plower/   # Compatibility shim
├── models/
│   ├── configs/             # Model surgery and passthrough configs
│   └── Modelfiles/          # Ollama Modelfile recipes
├── tools/
│   ├── plowinf_viewer.sh    # Real-time activity and work product viewer
│   └── run_with_logging.py  # Directive 02 universal execution logger
├── tests/                   # Complete unit test suite
├── docs/                    # Architectural and inference guides
├── pyproject.toml           # Package configuration & entrypoints
├── Makefile                 # Standardized development targets
├── AI.md                    # Operational directives and guardrails
└── README.md                # Project documentation
```

<!-- end of file: plowinf/README.md -->
