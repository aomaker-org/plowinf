# Path: plowinf/docs/architecture.md
# plowinf Architectural Specification
# Max Column: 80 Columns

## 1. Overview & Purpose

`plowinf` is a dedicated repository and runtime environment designed for deep
dives into local AI inference, Small Language Models (SLMs), code processing,
and offline task orchestration.

Key design goals:
1. **Local AI Inference Deep Dive**: Experiment with, benchmark, and evaluate
   local inference backends (Ollama, vLLM/OpenAI-compatible, llama.cpp,
   llama2.c, and deterministic mock adapters).
2. **Offline Code Processing**: Autonomous code review, unit test generation,
   docstring creation, and AST-aware prompt analysis without sending code to
   external cloud APIs.
3. **Atomic File-Backed Queue**: Concurrency coordination via POSIX file locks
   (`fcntl.flock`) across partitions (`incoming/`, `processing/`, `done/`,
   `failed/`).
4. **Air-Gap Safety & Cryptographic Forensics**: Full compliance with
   Directive 02 (complete audit logging) and Directive 18 (100% offline
   air-gapped execution).

---

## 2. Component Topology

```
+-----------------------------------------------------------------+
|                       CLI Engine (plowinf)                      |
|  (review, test, docstring, prompt, benchmark, enqueue, daemon)  |
+--------------------------------+--------------------------------+
                                 |
                                 v
+--------------------------------+--------------------------------+
|                     Code Worker Supervisor                      |
|           - Stream Logger (Directive 02)                        |
|           - Heartbeat Emitter (Directive 17)                    |
|           - Prompt Builder & Task Formatter                     |
+---------------+--------------------------------+----------------+
                |                                |
                v                                v
+---------------+----------------+ +-------------+----------------+
|       File-Backed Queue        | |        Backend Router        |
|  - records/queue/incoming/     | |  1. Ollama (HTTP/GGUF)       |
|  - records/queue/processing/   | |  2. OpenAI-Compat (/v1)      |
|  - records/queue/done/         | |  3. llama2.c (Native C)      |
|  - records/queue/failed/       | |  4. llama-cli (GGUF CLI)     |
|  - fcntl.flock concurrency     | |  5. Mock (Deterministic CI)  |
+--------------------------------+ +------------------------------+
```

---

## 3. Supported Model Backends

1. **Ollama HTTP API (`OllamaBackend`)**:
   - High-throughput endpoint on `http://127.0.0.1:11434`.
   - Supports registered tags (e.g. `qwen2.5-coder:1.5b`) and
     dynamic GGUF auto-registration via `/api/create`.
   - Supports token streaming via `generate_stream()`.
2. **OpenAI-Compatible Endpoints (`OpenAICompatBackend`)**:
   - Universal adapter for local inference servers exposing
     `/v1/chat/completions` (vLLM, LM Studio, LiteRT-LM).
3. **Native C Runner (`Llama2cBackend`)**:
   - Microsecond startup, zero-dependency C binary (`run`) executing compact
     weight files (`stories15M.bin`). Delivers ~23 tok/s on standard CPU cores.
4. **llama.cpp CLI (`LlamaCppCliBackend`)**:
   - Direct subprocess execution of `llama-cli` on quantized GGUF weights.
5. **Deterministic Mock (`MockBackend`)**:
   - Zero-overhead heuristic fallback for offline unit testing and dry runs.

<!-- end of file: plowinf/docs/architecture.md -->
