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
