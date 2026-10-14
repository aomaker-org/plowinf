# Path: plowinf/AGENTS.md
# AGENTS.md - Unified AI Assistant Governance Pointer
# Max Column: 80 Columns

All AI coding assistants (`agy`, `copilot`, `gemini-web`, and background agents)
operating within the `plowinf` workspace MUST read and strictly adhere to the
repository specification and directives in `AI.md`.

## Core Directives Summary
- **[Directive 01]**: Challenge human input with best practices and safety.
  Mandatory `uv` for all Python environment and execution workflows.
- **[Directive 02]**: Full I/O logging; zero `/dev/null` or silent suppression.
- **[Directive 04]**: Max 80 columns (text/md), max 120 chars (code); mandatory
  file path headers and footers.
- **[Directive 07]**: Branch-per-feature mandate; clean `main` guardrails;
  `git push --force` prohibited; strictly additive file & Git operations.
- **[Directive 08]**: Automated turn-sync handover via `make sync`.
- **[Directive 14]**: Model quota preservation & local inference utilization.
- **[Directive 15]**: Multi-agent coordination, pre-flight git audits, distinct
  branch namespaces, and zero toe-stepping.
- **[Directive 17]**: Heartbeat, status reporting, and state persistence.
- **[Directive 18]**: Zero-cloud survivability & Air-Gap mandate — local-only
  execution with zero external I/O, zero AI cloud quota burn, and offline SLMs.

For the authoritative specification and guides, see:
- [`AI.md`](AI.md) - Full AI kernel directives.
- [`docs/architecture.md`](docs/architecture.md) - Core architecture spec.
- [`docs/inference_guide.md`](docs/inference_guide.md) - Inference & SLM guide.

<!-- end of file: plowinf/AGENTS.md -->
