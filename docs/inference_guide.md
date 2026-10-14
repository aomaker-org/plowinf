# Path: plowinf/docs/inference_guide.md
# Local AI Inference & SLM Deep Dive Guide
# Max Column: 80 Columns

## 1. Introduction

`plowinf` provides a hands-on laboratory for exploring local AI inference
runtimes and optimizing Small Language Model (SLM) performance on workstation
and edge hardware.

---

## 2. Supported Inference Engines

### Ollama (`http://127.0.0.1:11434`)
- **Strengths**: Automated GPU layer offloading (CUDA, ROCm, Metal), unified
  Modelfile format, tag management, streaming HTTP API.
- **Dynamic Registration**: Point `plowinf` directly at any `.gguf` file, and
  `OllamaBackend` will automatically register it via the `/api/create` endpoint.
- **Commands**:
  ```bash
  # Check available models
  plowinf backends

  # Run prompt
  plowinf prompt "Explain binary search in Python" --backend ollama

  # Benchmark latency and throughput
  plowinf benchmark --backend ollama --runs 5
  ```

### OpenAI-Compatible Server (vLLM / LM Studio / LiteRT-LM)
- **Strengths**: Continuous batching, PagedAttention, multi-LoRA support,
  drop-in client compatibility.
- **Configuration**: Expose `/v1/chat/completions` on `http://127.0.0.1:8000/v1`.
- **Usage**:
  ```bash
  plowinf review src/plowinf/queue.py --backend openai_compat
  ```

### llama-cli (`llama.cpp`)
- **Strengths**: Zero daemon overhead, fine-grained CPU thread configuration,
  direct quantized GGUF execution.
- **Usage**:
  ```bash
  plowinf prompt "Hello" --backend llama_cpp --model path/to/model.gguf
  ```

### llama2.c Native Runner
- **Strengths**: Pure C implementation with zero Python/framework overhead.
  Instantaneous cold-start (<5ms), ~23+ tok/s CPU inference on standard x86_64.
- **Usage**:
  ```bash
  plowinf prompt "Once upon a time" --backend llama2c
  ```

---

## 3. Model Surgery & Custom Distillations

`plowinf` includes configuration and recipe scaffolding in `models/` for:
- Layer passthrough slicing (via `mergekit`).
- Quantization configurations (Q4_K_M, Q8_0, F16).
- Custom Modelfile templates for system prompts, stop tokens, and repetition
  penalties.

See [`models/Modelfiles/Modelfile.template`](../models/Modelfiles/Modelfile.template).

<!-- end of file: plowinf/docs/inference_guide.md -->
