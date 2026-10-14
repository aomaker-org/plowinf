Adopting the scoped additive file policy resolves the tension between **[Directive 07] (`AI.md`)** and Python's module/test discovery rules (`pyproject.toml`) without weakening historical auditability.

---

## 1. Architectural Reasoning & Attribution

### Problem Statement & Source Attribution

1. **Directive Conflict (`plowinf/AI.md`, Section `### [Directive 07]`):** The original wording stated: *"All file generation operations must be additive. Overwriting existing files in place is prohibited; use versioned revisions (`_rev001`) to preserve all historical drafts."*
2. **Agent Guide Propagation (`plowinf/docs/gemini_web_agy_guide.md`, Sections 4 & 5):** Summarized Directive 07 as *"Additive file operations (no destructive file overwriting)"*, which instructs `agy` to avoid in-place edits even when refactoring source code or tests.
3. **Pytest & Package Discovery (`plowinf/pyproject.toml`):** Configured with `testpaths = ["tests"]` and `python_files = ["test_*.py"]`. If an agent or `AdditiveFileHandler` writes `tests/test_backends_rev001.py` instead of updating `tests/test_backends.py` in place, `uv run pytest` collects both files (`test_backends.py` and `test_backends_rev001.py`), running obsolete assertions alongside current ones. Similarly, creating `src/plowinf/worker_rev001.py` breaks canonical imports (`from plowinf import worker`) required by `plowinf.cli:main`.

### The Two-Tier Resolution Policy

To preserve both **deterministic package/test layout** and **non-destructive worker artifact history**, file operations in `plowinf` are partitioned into two explicit domains:

| Domain | Repository Paths | Versioning & Mutation Mechanism | Governing Enforcement |
| --- | --- | --- | --- |
| **1. Canonical Git-Tracked Source** | `src/`, `tests/`, `docs/`, `tools/`, `pyproject.toml`, `Makefile`, `AI.md`, `README.md` | **In-place updates on canonical filenames.** Historical revisions are preserved via atomic Git commits on `feature/*` or `fix/*` branches with immutable Git history. | `GitSafetyGuard` + `AdditiveFileHandler` path rejection on canonical source trees. |
| **2. Untracked Worker Artifacts & Logs** | `logs/`, `queue/`, worker output directories, generated reviews, draft docstrings, benchmark dumps | **Strictly additive (`_rev001`, `_rev002`)** or append-only with ISO-8601 UTC microsecond timestamps. In-place overwrites are prohibited. | `AdditiveFileHandler` in `src/plowinf/safety.py`. |

---

## 2. Updated Workspace Specification (`plowinf/AI.md`)

Below is the complete, 80-column-compliant replacement for `plowinf/AI.md` incorporating the scoped **Directive 07** definition and Directive 04 header/footer envelopes.

```markdown
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

```

---

## 3. Updated Pairing Guide (`plowinf/docs/gemini_web_agy_guide.md`)

Below is the complete, 80-column-compliant replacement for `plowinf/docs/gemini_web_agy_guide.md` updating Section 4 (`[Directive 07]`) and Section 5 (Item 4) to align `agy` with the two-tier policy.

```markdown
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
  - `/home/fekerr/ai_models/qwen2-1.5b.gguf`: 892 MB Q4 quantized GGUF.
  - `/home/fekerr/ai_models/stories15M.bin`: 58 MB compact Llama2.c weights.
  - Ollama Daemon: `http://127.0.0.1:11434` (Ollama tag `qwen2:1.5b`).
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
   - Protocol: HTTP `/v1/chat/completions` on `127.0.0.1:11434/v1` or `:8000/v1`.
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
3. **Use `uv` in Commands**: e.g., `uv run pytest tests/`, `uv run plowinf backends`.
4. **Prefer Scoped Updates**: Modify tracked source files in place via clean
   diffs on `feature/*` branches; reserve `_rev001` for generated artifacts.
5. **Request Turn-Sync**: End recommendations with `make sync` validation.

<!-- end of file: plowinf/docs/gemini_web_agy_guide.md -->

```

---

## 4. Programmatic Guardrail for `src/plowinf/safety.py`

Because `gemweb_context_pack.md` provides only the structural summary of `src/plowinf/safety.py` (lines 265–272), we should also enforce this policy programmatically inside `AdditiveFileHandler` so autonomous workers (`src/plowinf/worker.py`) cannot accidentally emit `_rev001.py` files inside `src/` or `tests/`.

Here is the canonical path guard to integrate into `AdditiveFileHandler` in `plowinf/src/plowinf/safety.py`:

```python
# Path: plowinf/src/plowinf/safety.py
# (Scoped canonical source guard addition for AdditiveFileHandler)

from pathlib import Path

CANONICAL_SOURCE_DIRS: tuple[str, ...] = (
    "src",
    "tests",
    "docs",
    "tools",
)
CANONICAL_ROOT_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "Makefile",
    "AI.md",
    "README.md",
)


def validate_additive_artifact_path(
    target_path: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    """Ensure AdditiveFileHandler (_rev001) is not used on Git-tracked source.

    Per Directive 07, canonical repository source files under src/, tests/,
    docs/, tools/, and root config files must retain canonical filenames and
    rely on atomic Git commits for history rather than _revNNN duplicates.
    """
    resolved_target = Path(target_path).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )

    try:
        rel_path = resolved_target.relative_to(root)
    except ValueError:
        # Target is outside the repository root (e.g., /tmp or pytest tmp_path)
        return resolved_target

    parts = rel_path.parts
    if not parts:
        return resolved_target

    if parts[0] in CANONICAL_SOURCE_DIRS or (
        len(parts) == 1 and parts[0] in CANONICAL_ROOT_FILES
    ):
        raise ValueError(
            f"Directive 07 violation: '{rel_path}' is inside a canonical "
            f"Git-tracked path ({parts[0]}). Update canonical source files "
            f"in place on a feature/* branch; reserve AdditiveFileHandler "
            f"(_rev001) for untracked worker artifacts and drafts."
        )

    return resolved_target

# end of file: plowinf/src/plowinf/safety.py

```

---

## 5. Ready-to-Execute Handover Prompt for `agy`

Copy and paste the block below directly into `agy` on `fg26` to apply the documentation updates, wire the programmatic guard into `src/plowinf/safety.py`, add unit test coverage, regenerate `docs/gemweb_context_pack.md`, and run `make sync`:

```markdown
Execute the Directive 07 Scoped Additive File Policy update across `plowinf` on
branch `feature/plowinf-core`:

1. Update `AI.md` in place with the scoped `[Directive 07]` policy separating
   **Untracked Artifacts & Worker Outputs** (`AdditiveFileHandler` `_rev001`
   revisions) from **Git-Tracked Repository Source** (`src/`, `tests/`, `docs/`,
   `tools/`, and root configs updated in place on canonical filenames via Git
   commits). Enforce max 80 columns and Directive 04 header/footer envelopes.
2. Update `docs/gemini_web_agy_guide.md` in place (Section 4 `[Directive 07]`
   and Section 5 Item 4) to reflect the same scoped additive policy. Enforce
   max 80 columns and Directive 04 header/footer envelopes.
3. Inspect `src/plowinf/safety.py` and add `validate_additive_artifact_path()`
   (and invoke it in `AdditiveFileHandler`) so any attempt to generate `_rev001`
   files inside repository `src/`, `tests/`, `docs/`, `tools/`, or canonical
   root config files raises a `ValueError` citing Directive 07, while paths
   outside those trees (such as `logs/`, `queue/`, `artifacts/`, or `pytest`
   `tmp_path` fixtures) continue functioning normally.
4. Add unit tests in `tests/` verifying that `validate_additive_artifact_path`
   rejects paths inside `src/` and `tests/` under `repo_root` while permitting
   artifact directories and `tmp_path`.
5. Refresh `docs/gemweb_context_pack.md` so the bundled context pack reflects
   the updated `AI.md` and `docs/gemini_web_agy_guide.md`.
6. Verify formatting, column bounds, and test suite execution using:
   - `uv run pytest -v`
   - `make sync`

```
