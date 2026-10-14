# Path: plowinf/docs/gemweb_context_pack.md
# All-in-One Context Pack for gemini.google.com
# Max Column: 80 Columns

Notice: This bundled context file is prepared for paste/upload into
gemini.google.com sessions to enable deep architectural and design discussions
about the `plowinf` repository and its local AI inference workflows.

================================================================================
SECTION 1: SYSTEM & WORKSPACE SUMMARY
================================================================================
- Repository: plowinf (https://github.com/aomaker-org/plowinf.git)
- Active Branch: feature/plowinf-core
- Target Host: Ubuntu Linux 24.04 LTS (x86_64, 12GB RAM, host: fg26)
- Runtime: Python 3.12 managed via uv (0.12.15)
- Active Local Engines:
  * Ollama on http://127.0.0.1:11434 (model: qwen2.5-coder:1.5b)
  * OpenAI-Compatible on http://127.0.0.1:11434/v1 or port 8000
  * llama-cli in ~/.local/bin/llama-cli (built from ~/src/llama.cpp)
  * llama2.c runner in ~/src/clouds-labs/labs/llama2.c/run
- Local Model Weights:
  * ~/ai_models/qwen2.5-coder-1.5b.gguf (940 MB Q4_K_M GGUF)
  * ~/ai_models/qwen2-1.5b.gguf (892 MB Q4 quantized GGUF)
  * ~/ai_models/stories15M.bin (58 MB compact Llama2.c weights)
- Verification State: 54/54 unit tests passing via `uv run pytest`

<!-- FILE_START: docs/gemini_web_agy_guide.md -->
# Path: plowinf/docs/gemini_web_agy_guide.md
# Concise Pairing & Operational Guide: Gemini-Web (gemini.google.com) & AGY
# Max Column: 80 Columns

## 1. Overview & Dual-Agent Operating Model

This guide provides a high-density, concise briefing for `gemini.google.com`
(`gemini-web`) when collaborating with `agy` (Google Antigravity CLI) in the
`plowinf` workspace on Linux baremetal host `fg26`.

- **gemini-web (`gemini.google.com`)**: Strategic reasoning, long-context
  architectural analysis, test planning, refactoring blueprints.
- **agy (Antigravity CLI)**: Local execution agent on Linux baremetal `fg26`.
  Executes commands, runs tests via `uv`, manages Git branches, audits code.

## 2. Host Environment & Topology

- **Host Machine**: `fg26` (Ubuntu 24.04 LTS, x86_64, 12 GB physical RAM).
- **Primary Repository**:
  - `/home/fekerr/src/plowinf`: Dedicated local SLM inference & code processing.
  - Remote: `https://github.com/aomaker-org/plowinf.git`
  - Active Branch: `feature/plowinf-core`
- **Related Local Repositories**:
  - `/home/fekerr/src/llama.cpp`: Upstream C++ engine & built binaries.
  - `/home/fekerr/src/clouds-labs`: Distributed systems, labs/llama2.c.
- **Local Model Weights & Runtimes**:
  - `/home/fekerr/ai_models/qwen2.5-coder-1.5b.gguf`: 940 MB Q4_K_M GGUF.
  - `/home/fekerr/ai_models/qwen2-1.5b.gguf`: 892 MB Q4 quantized GGUF.
  - `/home/fekerr/ai_models/stories15M.bin`: 58 MB compact Llama2.c weights.
  - Ollama Daemon: `http://127.0.0.1:11434` (default `qwen2.5-coder:1.5b`).
  - OpenAI-Compat: `http://127.0.0.1:11434/v1` or port 8000.
  - `llama-cli`: `/home/fekerr/.local/bin/llama-cli` (built from source).
  - `run` (llama2.c): `/home/fekerr/src/clouds-labs/labs/llama2.c/run`.

## 3. Supported Model Backends & Performance Profiles

All 5 inference backends are tested, active, and verified:

1. **`ollama`** (`OllamaBackend`):
   - Protocol: HTTP `/api/generate` & `/api/tags` on `127.0.0.1:11434`.
   - Latency: ~1.95s on CPU (warm), ~6.6 words/s (~9.5 tok/s).
   - Features: Dynamic `.gguf` auto-registration via `/api/create`, streaming.
2. **`openai_compat`** (`OpenAICompatBackend`):
   - Protocol: HTTP `/v1/chat/completions` on `127.0.0.1:11434/v1`
     or `:8000/v1`.
   - Latency: ~2.12s on CPU, ~6.1 words/s (~8.8 tok/s).
   - Features: Standard OpenAI wire protocol compatibility with vLLM/LM Studio.
3. **`llama_cpp_cli`** (`LlamaCppCliBackend`):
   - Engine: Direct subprocess execution of `llama-cli`.
   - Latency: ~4.25s on CPU (includes process spawn & 892MB GGUF mmap).
   - Features: Zero-daemon offline execution with `--single-turn --simple-io`.
4. **`llama2c`** (`Llama2cBackend`):
   - Engine: Pure C native runner execution on `stories15M.bin`.
   - Latency: ~412 ms on CPU, **~116 words/s (~155 tok/s)**.
   - Features: Instantaneous cold-start (<5ms), zero framework overhead.
5. **`mock`** (`MockBackend`):
   - Deterministic offline heuristic for CI test suites (<0.1 ms).

## 4. Non-Negotiable Directives for AGY (Rules to Respect)

- **[Directive 01] Mandatory `uv`**:
  - Always use `uv run python ...`, `uv pip ...`, `uv venv`.
  - NEVER output bare `python` or `pip` invocations.
- **[Directive 02] Zero Silent Suppression**:
  - Never pipe to `/dev/null` or mask failures with `|| true`.
  - Always log additively with ISO-8601 UTC microsecond timestamps.
- **[Directive 04] Strict Column Bounds & File Envelopes**:
  - Markdown/Text: max **80 columns**.
  - Code (`.py`, `.sh`): target 80, max **120 characters**.
  - File Header: `# Path: <repo-relative-path>`
  - File Footer:
    - Markdown: `<!-- end of file: <path> -->`
    - Python/Shell: `# end of file: <path>`
- **[Directive 07] Git Branch & Scoped Additive Safety**:
  - Dedicated feature branches (`feature/*`); never commit to `main`.
  - Untracked worker outputs/drafts use `_rev001` additive revisions.
  - Git-tracked source (`src/`, `tests/`, `docs/`, `tools/`, configs) is
    updated in place on canonical filenames and versioned via Git commits.
  - `git push --force` is strictly prohibited.
- **[Directive 08] Turn-Sync Audit**:
  - Always verify with `make sync` before finishing turns.
- **[Directive 17] Telemetry & Heartbeats**:
  - Daemons emit heartbeats to `logs/heartbeat/plowinf.json`.
- **[Directive 18] Zero-Cloud & Air-Gap Mandate**:
  - 100% offline local inference (`127.0.0.1`, `localhost`).
  - Outbound cloud AI APIs (OpenAI/Anthropic/Gemini API) are prohibited.

## 5. Optimal Prompting Pattern for Gemini-Web -> AGY

1. **Specify Target Files**: Give absolute or repo-relative file paths.
2. **Include Headers/Footers**: Format code chunks with Directive 04 envelopes.
3. **Use `uv` in Commands**: e.g., `uv run pytest tests/`,
   `uv run plowinf backends`.
4. **Prefer Scoped Updates**: Modify tracked source files in place via clean
   diffs on `feature/*` branches; reserve `_rev001` for generated artifacts.
5. **Request Turn-Sync**: End recommendations with `make sync` validation.

<!-- end of file: plowinf/docs/gemini_web_agy_guide.md -->
<!-- FILE_END: docs/gemini_web_agy_guide.md -->

<!-- FILE_START: AI.md -->
# Path: plowinf/AI.md
# Workspace Specification & Consolidated AI Directives (`AI.md`)
# Max Column: 80 Columns

Notice to AI Agents: This document defines operational directives, environment
boundaries, and repository hygiene for all assistants operating within the
`plowinf` workspace.

---

## Level-1 Executive Directives Kernel

### [Directive 01] OPERATIONAL BOUNDARIES & COMMAND INTEGRITY
- **Constructive Pushback**: AI assistants are strictly forbidden from acting
  as blind "yes-men". You MUST challenge human input with industry best
  practices, architectural integrity, and safety guidelines.
- **Mandatory `uv` for Python**: Always use `uv` for Python execution, virtual
  environments, and package installations (`uv run`, `uv venv`, `uv pip`).
  Never invoke raw `python` or `pip` directly.
- **Safe Shell Usage**: Always use standard Linux bash. Prohibit
  `powershell.exe`.

### [Directive 02] LOGGING & DIAGNOSTIC OBSERVABILITY
- **Zero Silent Suppression**: NEVER redirect diagnostic streams to `/dev/null`
  or suppress command failures with `|| true` unless strictly required.
- **Structured Timestamps**: All logs must carry ISO-8601 UTC microsecond
  timestamps.

### [Directive 04] FILE FORMATTING, LINE WRAPPING, & HEADER/FOOTER RULES
- **Line Wrapping**:
  - Text and Markdown (`.md`): Maximum **80 columns**.
  - Code files (`.py`, `.sh`): Target **80 columns**; max **120 characters**.
- **File Headers & Footers**:
  - Every file MUST contain a clear header identifying the file path.
  - Every file MUST contain a footer comment specifying the relative file path.
    - Markdown: `<!-- end of file: plowinf/AI.md -->`
    - Python/Shell: `# end of file: plowinf/AI.md`

### [Directive 07] GIT BRANCHING & WORKFLOW STANDARDS
- **Branch-Per-Feature Mandate**: Work in dedicated feature branches
  (`feature/*`, `fix/*`). Never commit directly to `main`.
- **Force-Push Safety**: `git push --force` requires explicit human approval.
- **Git History Immutability**: Git history must NEVER be overwritten,
  squashed, or lost. Destructive commands (`reset --hard`, `clean -fd`,
  `rebase`) are strictly prohibited in automated workflows.
- **Scoped Additive File Operations (`_rev001` vs. Git-Tracked Source)**:
  - **Untracked Artifacts & Worker Outputs**: All generated work products,
    drafts, reviews, logs, and reports must be strictly additive. Overwriting
    existing artifacts in place is prohibited; use `AdditiveFileHandler`
    versioned revisions (`_rev001`, `_rev002`) to preserve historical drafts.
  - **Git-Tracked Repository Source**: Files in `src/`, `tests/`, `docs/`,
    `tools/`, and root configs MUST keep canonical filenames in place so
    Python imports and `uv run pytest` collection remain deterministic without
    duplicate `_revNNN` modules. Version history for tracked source is
    preserved via atomic Git commits on `feature/*` or `fix/*` branches.

### [Directive 08] AUTOMATED TURN-SYNC HANDOVER
- Run `make sync` at turn completion to verify git status, run tests, and stage
  commits cleanly.

### [Directive 14] MODEL QUOTA PRESERVATION & LOCAL INFERENCE
- Maximize use of local offline models (Ollama, GGUF, llama2.c, vLLM) for
  routine code generation, docstrings, reviews, and test authoring to preserve
  cloud model quotas.

### [Directive 15] DUAL-AGENT COORDINATION
- Co-exist cleanly with other assistants using distinct branch namespaces
  (`feature/agy-*` vs `feature/copilot-*`) and pre-flight git status audits.

### [Directive 17] PROCESS HEARTBEATS & TELEMETRY
- Background workers must emit status heartbeats to
  `logs/heartbeat/plowinf.json` with timestamps, RSS memory usage, and task
  counters.

### [Directive 18] ZERO-CLOUD SURVIVABILITY & AIR-GAP MANDATE
- **Zero External I/O**: `plowinf` operations MUST run local-only. All model
  inferences are strictly restricted to local loopback addresses (`127.0.0.1`,
  `localhost`) using local model weights (`.gguf`, `.bin`) and local daemons.
- **Zero Cloud AI Quota Burn**: Outbound external AI API calls (OpenAI,
  Anthropic, Gemini cloud APIs) are programmatically rejected by air-gap
  guardrails.

<!-- end of file: plowinf/AI.md -->
<!-- FILE_END: AI.md -->

<!-- FILE_START: pyproject.toml -->
[project]
name = "plowinf"
version = "0.1.0"
description = """Offline AI inference engine, SLM code worker,
and queue processor"""
readme = "README.md"
requires-python = ">=3.10"
authors = [
    { name = "Fred Kerr", email = "fekerr@gmail.com" }
]
license = { text = "Apache-2.0" }
dependencies = [
    "requests>=2.31.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]

[project.scripts]
plowinf = "plowinf.cli:main"
plower = "plowinf.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
<!-- FILE_END: pyproject.toml -->

<!-- FILE_START: src/plowinf/backends.py -->
# Path: plowinf/src/plowinf/backends.py
# (Core inference backend interfaces and adapters)
# Highlights:
# - SLMBackend: Abstract interface (is_available, generate, list_models)
# - OllamaBackend: Dynamic GGUF registration, tags, parameter tuning, stream
# - OpenAICompatBackend: Auto-discovers v1 endpoint, standard completions
# - LlamaCppCliBackend: Spawns llama-cli on local GGUF (--single-turn)
# - Llama2cBackend: Executes native C runner binary on compact .bin weights
# - MockBackend: Deterministic offline CI verification
# - BackendRouter: Resolves best available engine
<!-- FILE_END: src/plowinf/backends.py -->

<!-- FILE_START: src/plowinf/safety.py -->
# Path: plowinf/src/plowinf/safety.py
# (Safety and air-gap guardrails)
# Highlights:
# - validate_local_only_url(url): Blocks non-loopback hostnames
# - validate_additive_artifact_path(path): Protects canonical Git source trees
# - AdditiveFileHandler: Safe versioned draft revisioning (_rev001)
# - GitSafetyGuard: Intercepts destructive git operations (--force, reset)
<!-- FILE_END: src/plowinf/safety.py -->

<!-- FILE_START: src/plowinf/queue.py -->
# Path: plowinf/src/plowinf/queue.py
# (File-backed atomic task queue)
# Highlights:
# - POSIX advisory file locking (fcntl.flock)
# - State transitions: incoming/ -> processing/ -> done/ (or failed/)
# - Atomic moves guarantee collision-free multi-worker concurrency
<!-- FILE_END: src/plowinf/queue.py -->

<!-- FILE_START: src/plowinf/worker.py -->
# Path: plowinf/src/plowinf/worker.py
# (Autonomous code worker engine)
# Highlights:
# - build_prompt(action, code, prompt): Templates for review, test, docstring
# - process_task(task): Dispatches to BackendRouter, writes work products
# - emit_heartbeat(): Emits status, RSS memory, task metrics to logs
# - run_daemon(): Background polling loop with signal handling (SIGINT, SIGTERM)
<!-- FILE_END: src/plowinf/worker.py -->

<!-- FILE_START: tools/test_backends_perf.py -->
# Path: plowinf/tools/test_backends_perf.py
# (Multi-backend comparative benchmark suite)
# Measures average latency, word throughput, pass distributions, and RSS
# Run via: uv run tools/test_backends_perf.py
<!-- FILE_END: tools/test_backends_perf.py -->

<!-- FILE_START: tools/inspect_gguf_arch.py -->
# Path: plowinf/tools/inspect_gguf_arch.py
# (GGUF binary header parser and Ollama architecture comparison inspector)
# Highlights:
# - parse_gguf_summary(path): GGUF v2/v3 header, metadata, tensor budgets
# - resolve_ollama_gguf_blob(tag): Resolves Ollama manifests to blob paths
# - compare_summaries(): Side-by-side GQA, SwiGLU, parameter & size deltas
<!-- FILE_END: tools/inspect_gguf_arch.py -->

<!-- FILE_START: tools/qwen_distill_moe_lab.py -->
# Path: plowinf/tools/qwen_distill_moe_lab.py
# (Qwen MoE distillation planner and mergekit-moe blueprint generator)
# Highlights:
# - calculate_moe_budgets(): Dense vs Upcycled vs Sub-FFN RAM and tok/s
# - emit_mergekit_moe_yaml(): 4-expert mergekit-moe config for plowinf
# - run_hf_profiler(): PyTorch BI_i layer influence and SwiGLU sparsity
<!-- FILE_END: tools/qwen_distill_moe_lab.py -->

<!-- FILE_START: tools/audit_20l_surgery.py -->
# Path: plowinf/tools/audit_20l_surgery.py
# (Weight norm audit and A/B code AST validation across 20L vs 28L models)
# Highlights:
# - audit_layer_mapping(): GGUF tensor Frobenius/RMS norms layer mapping
# - run_ab_benchmark(): AST and JSON validation prompts on local Ollama
<!-- FILE_END: tools/audit_20l_surgery.py -->

<!-- end of file: plowinf/docs/gemweb_context_pack.md -->
