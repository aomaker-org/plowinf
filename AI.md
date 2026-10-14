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
