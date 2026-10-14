# Path: plowinf/AI.md
# Workspace Specification & Consolidated AI Directives (`AI.md`)
# Max Column: 80 Columns

Notice to AI Agents: This document defines operational directives, environment
boundaries, and repository hygiene for all assistants operating within the
`plowinf` workspace.

---

## [Architecture] Two-Tier AI Directive Hierarchy

To minimize recursive token burn and preserve context quotas in `agy`,
directives are partitioned into a two-tier hierarchy:

1. **Level-1 Executive Directives Kernel (Embedded Below)**:
   The foundational operational guardrails required on every prompt turn:
   - `[Directive 01]`: Challenge human input, mandatory `uv`, shell safety.
   - `[Directive 02]`: Full I/O logging; zero `/dev/null` or silent suppression.
   - `[Directive 04]`: Max 80 columns (text/md), max 120 chars (code); headers.
   - `[Directive 07]`: Branch-per-feature; clean `main` guardrails; force-push.
   - `[Directive 08]`: Automated turn-sync handover via `make sync`.
   - `[Directive 14]`: Model quota preservation & local inference utilization.
   - `[Directive 15]`: Dual-agent (`agy` & `copilot`) coordination.
   - `[Directive 17]`: Process heartbeats and telemetry emission.
   - `[Directive 18]`: Zero-cloud survivability and offline execution.

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
- **Additive-Only File Operations**: All file generation operations must be
  additive. Overwriting existing files in place is prohibited; use versioned
  revisions (`_rev001`) to preserve all historical drafts.

### [Directive 08] AUTOMATED TURN-SYNC HANDOVER
- Run `make sync` at turn completion to verify git status, run tests, and stage
  commits cleanly.

### [Directive 14] MODEL QUOTA PRESERVATION & LOCAL INFERENCE
- Maximize use of local offline models (Ollama, GGUF, llama2.c, vLLM) for routine
  code generation, docstrings, reviews, and test authoring to preserve cloud
  model quotas.

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
