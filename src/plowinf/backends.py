# Path: plowinf/src/plowinf/backends.py
"""Backend adapters for local Small Language Models (SLMs) and inference engines.

Provides unified generation interfaces for Ollama, OpenAI-compatible servers
(vLLM, LM Studio, LiteRT-LM), llama-cli, llama2.c, and deterministic mock
fallbacks for local AI inference research and code processing.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, Generator, List, Optional
import requests

from .safety import validate_local_only_url

logger = logging.getLogger("plowinf.backends")


def get_optimal_physical_threads() -> int:
    """Detect physical CPU cores to prevent Haswell SMT cache contention.

    On x86_64 host (fg26), binding threads to physical cores rather than
    logical hyperthreads maximizes L1/L2 cache locality during GEMV decoding.
    """
    try:
        cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        socket_core: set[tuple[str, str]] = set()
        current_socket = "0"
        current_core = "0"
        for line in cpu_info.splitlines():
            if line.startswith("physical id"):
                current_socket = line.split(":")[-1].strip()
            elif line.startswith("core id"):
                current_core = line.split(":")[-1].strip()
                socket_core.add((current_socket, current_core))
        if socket_core:
            return max(1, len(socket_core))
    except Exception:
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2)


class SLMBackend(ABC):
    """Abstract interface for local code worker and inference backends."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend runtime and model dependencies are reachable."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text completion from prompt."""
        pass

    def list_models(self) -> List[str]:
        """List models available on this backend (if supported)."""
        return []


class OllamaBackend(SLMBackend):
    """Adapter for local Ollama HTTP API (default: http://127.0.0.1:11434).

    Supports standard Ollama model tags (e.g. 'qwen2.5-coder:1.5b') or raw local
    '.gguf' file paths. When passed a '.gguf' file, it automatically registers
    the model with the Ollama daemon via the '/api/create' endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5-coder:1.5b",
        timeout: Optional[float] = None,
    ):
        validate_local_only_url(base_url)
        super().__init__(name="ollama")
        self.base_url = base_url.rstrip("/")
        self.raw_model_spec = model
        self.timeout = timeout if timeout is not None else float(os.getenv("PLOWER_TIMEOUT", "600.0"))
        self._effective_model: Optional[str] = None

    def list_models(self) -> List[str]:
        """Query Ollama daemon for all registered local model tags."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception as exc:
            logger.debug("Failed to query Ollama tags: %s", exc)
        return []

    def _resolve_model(self) -> str:
        """Resolve model name, registering .gguf file dynamically if needed."""
        if self._effective_model:
            return self._effective_model

        if self.raw_model_spec.endswith(".gguf") or os.path.isfile(self.raw_model_spec):
            abs_path = os.path.abspath(self.raw_model_spec)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Local GGUF model file not found: {abs_path}")
            base_name = os.path.splitext(os.path.basename(abs_path))[0]
            registered_name = f"plowinf-{base_name.lower().replace('_', '-')}"
            try:
                logger.info(
                    "Registering GGUF '%s' as Ollama model '%s'...",
                    abs_path,
                    registered_name,
                )
                modelfile_content = f"FROM {abs_path}\nPARAMETER temperature 0.2\n"
                resp = requests.post(
                    f"{self.base_url}/api/create",
                    json={"name": registered_name, "modelfile": modelfile_content, "stream": False},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    logger.info("Successfully registered model '%s' in Ollama.", registered_name)
                    self._effective_model = registered_name
                    return self._effective_model
                logger.warning(
                    "Ollama model registration returned %s: %s",
                    resp.status_code,
                    resp.text,
                )
            except Exception as exc:
                logger.warning("Failed to auto-register GGUF in Ollama: %s", exc)

            self._effective_model = registered_name
            return self._effective_model

        if self.raw_model_spec == "qwen2.5-coder:1.5b":
            try:
                available = self.list_models()
                if available:
                    has_coder = any(
                        m == "qwen2.5-coder:1.5b" or m.startswith("qwen2.5-coder:1.5b:")
                        for m in available
                    )
                    if not has_coder:
                        has_qwen2 = any(
                            m == "qwen2:1.5b" or m.startswith("qwen2:1.5b:")
                            for m in available
                        )
                        if has_qwen2:
                            logger.info(
                                "Model 'qwen2.5-coder:1.5b' not found in Ollama; "
                                "falling back to 'qwen2:1.5b'"
                            )
                            self._effective_model = "qwen2:1.5b"
                            return self._effective_model
            except Exception:
                pass

        self._effective_model = self.raw_model_spec
        return self._effective_model

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        model_name = self._resolve_model()
        url = f"{self.base_url}/api/generate"
        options: Dict[str, Any] = {
            "temperature": kwargs.get("temperature", 0.2),
            "num_predict": kwargs.get("max_tokens", kwargs.get("num_predict", 1024)),
            "num_thread": kwargs.get("num_thread", get_optimal_physical_threads()),
        }
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            options["top_k"] = kwargs["top_k"]
        if "repeat_penalty" in kwargs:
            options["repeat_penalty"] = kwargs["repeat_penalty"]
        if "stop" in kwargs:
            options["stop"] = kwargs["stop"]

        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama API error {resp.status_code}: {resp.text}")

        data = resp.json()
        return data.get("response", "").strip()

    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        """Stream generated tokens in real time from Ollama."""
        model_name = self._resolve_model()
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.2),
                "num_predict": kwargs.get("max_tokens", kwargs.get("num_predict", 1024)),
                "num_thread": kwargs.get("num_thread", get_optimal_physical_threads()),
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = requests.post(url, json=payload, stream=True, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama streaming API error {resp.status_code}: {resp.text}")

        for line in resp.iter_lines(decode_unicode=True):
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done", False):
                    break


class OpenAICompatBackend(SLMBackend):
    """Adapter for OpenAI-compatible local servers (vLLM, LM Studio, LiteRT, Ollama /v1)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "qwen2.5-coder:1.5b",
        api_key: str = "EMPTY",
        timeout: Optional[float] = None,
    ):
        effective_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_COMPAT_HOST")
            or "http://127.0.0.1:8000/v1"
        )
        validate_local_only_url(effective_url)
        super().__init__(name="openai_compat")
        self.base_url = effective_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout if timeout is not None else float(os.getenv("PLOWER_TIMEOUT", "600.0"))
        self._url_explicit = base_url is not None or os.getenv("OPENAI_BASE_URL") is not None

    def list_models(self) -> List[str]:
        for candidate_url in ([self.base_url] if self._url_explicit else [self.base_url, "http://127.0.0.1:11434/v1"]):
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                resp = requests.get(f"{candidate_url}/models", headers=headers, timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
                    if models:
                        self.base_url = candidate_url
                        return models
            except Exception:
                pass
        return []

    def is_available(self) -> bool:
        for candidate_url in ([self.base_url] if self._url_explicit else [self.base_url, "http://127.0.0.1:11434/v1"]):
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                resp = requests.get(f"{candidate_url}/models", headers=headers, timeout=3.0)
                if resp.status_code == 200:
                    self.base_url = candidate_url
                    return True
            except Exception:
                pass
        return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        # Resolve model name if set to default or standard coder model
        active_model = self.model
        if active_model in ("default", "qwen2.5-coder:1.5b"):
            models = self.list_models()
            if models:
                if any(m == "qwen2.5-coder:1.5b" or m.startswith("qwen2.5-coder:1.5b:") for m in models):
                    coder_matches = [
                        m for m in models if m == "qwen2.5-coder:1.5b" or m.startswith("qwen2.5-coder:1.5b:")
                    ]
                    active_model = coder_matches[0]
                elif any(m == "qwen2:1.5b" or m.startswith("qwen2:1.5b:") for m in models):
                    qwen2_matches = [
                        m for m in models if m == "qwen2:1.5b" or m.startswith("qwen2:1.5b:")
                    ]
                    active_model = qwen2_matches[0]
                elif self.model in models:
                    active_model = self.model
                else:
                    active_model = models[0]

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": active_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 1024),
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI-compat API error {resp.status_code}: {resp.text}")

        data = resp.json()
        choices = data.get("choices", [])
        if choices and "message" in choices[0]:
            return choices[0]["message"].get("content", "").strip()
        return ""


class LlamaCppCliBackend(SLMBackend):
    """CLI backend driving llama.cpp 'llama-cli' binary for local GGUF models."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        model_path: Optional[str] = None,
        threads: Optional[int] = None,
        timeout: float = 120.0,
    ):
        super().__init__(name="llama_cpp_cli")
        self.binary_path = binary_path or self._find_binary()
        self.model_path = model_path or self._find_model()
        self.threads = threads if threads is not None else get_optimal_physical_threads()
        self.timeout = timeout

    def _find_binary(self) -> str:
        candidates = [
            os.getenv("LLAMA_CLI_PATH", ""),
            "llama-cli",
            os.path.expanduser("~/.local/bin/llama-cli"),
            "/usr/bin/llama-cli",
            "/usr/local/bin/llama-cli",
        ]
        for c in candidates:
            if c and (shutil.which(c) or (os.path.isfile(c) and os.access(c, os.X_OK))):
                return c
        return "llama-cli"

    def _find_model(self) -> str:
        candidates = [
            os.getenv("LLAMA_MODEL_PATH", ""),
            os.path.expanduser("~/ai_models/qwen2.5-coder-1.5b.gguf"),
            os.path.expanduser("~/ai_models/qwen2-1.5b.gguf"),
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        return ""

    def is_available(self) -> bool:
        has_bin = (
            shutil.which(self.binary_path)
            or (os.path.isfile(self.binary_path) and os.access(self.binary_path, os.X_OK))
        )
        if not has_bin:
            return False
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        return True

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError(f"llama-cli binary or model '{self.model_path}' not available.")

        full_prompt = f"{system_prompt + chr(10) if system_prompt else ''}{prompt}"
        cmd = [
            self.binary_path,
            "-m",
            self.model_path,
            "-p",
            full_prompt,
            "-t",
            str(self.threads),
            "-c",
            str(kwargs.get("ctx_size", 4096)),
            "-n",
            str(kwargs.get("max_tokens", kwargs.get("num_predict", 512))),
            "--temp",
            str(kwargs.get("temperature", 0.2)),
            "-fa",
            "on",
            "-b",
            "2048",
            "-ub",
            "512",
            "--single-turn",
            "--simple-io",
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(f"llama-cli exited with {res.returncode}: {res.stderr}")

        raw = res.stdout
        marker = f"> {full_prompt}"
        if marker in raw:
            content_after = raw.split(marker, 1)[1]
            lines = []
            for l in content_after.splitlines():
                if l.strip().startswith("[ Prompt:") or l.strip() == "Exiting...":
                    continue
                lines.append(l)
            return "\n".join(lines).strip()
        elif "> " in raw:
            parts = raw.split("> ")
            if len(parts) >= 2:
                content_after = parts[-1]
                lines = [
                    l for l in content_after.splitlines()
                    if not l.strip().startswith("[ Prompt:") and l.strip() != "Exiting..."
                ]
                return "\n".join(lines).strip()
        return raw.strip()


class Llama2cBackend(SLMBackend):
    """Native C runner adapter for llama2.c binary using compact .bin models."""

    def __init__(
        self,
        runner_bin: Optional[str] = None,
        model_bin: Optional[str] = None,
        tokenizer_bin: Optional[str] = None,
        temperature: float = 0.2,
        steps: int = 256,
        timeout: float = 30.0,
    ):
        super().__init__(name="llama2c")
        self.runner_bin = runner_bin or self._find_runner()
        self.model_bin = model_bin or self._find_model()
        self.tokenizer_bin = tokenizer_bin or self._find_tokenizer()
        self.temperature = temperature
        self.steps = steps
        self.timeout = timeout

    def _find_runner(self) -> str:
        candidates = [
            "run",
            os.path.expanduser("~/src/clouds-labs/labs/llama2.c/run"),
            os.path.expanduser("~/src/llama2.c/run"),
        ]
        for c in candidates:
            if shutil.which(c) or (os.path.isfile(c) and os.access(c, os.X_OK)):
                return c
        return "run"

    def _find_model(self) -> str:
        candidates = [
            "stories15M.bin",
            os.path.expanduser("~/src/clouds-labs/labs/llama2.c/stories15M.bin"),
            os.path.expanduser("~/ai_models/stories15M.bin"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return "stories15M.bin"

    def _find_tokenizer(self) -> str:
        candidates = [
            "tokenizer.bin",
            os.path.expanduser("~/src/clouds-labs/labs/llama2.c/tokenizer.bin"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return "tokenizer.bin"

    def is_available(self) -> bool:
        if not (self.runner_bin and (shutil.which(self.runner_bin) or os.path.isfile(self.runner_bin))):
            return False
        if not (self.model_bin and os.path.isfile(self.model_bin)):
            return False
        return True

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("llama2.c runner binary or weights file not available.")

        steps = kwargs.get("max_tokens", self.steps)
        temp = kwargs.get("temperature", self.temperature)
        cmd = [
            self.runner_bin,
            self.model_bin,
            "-t",
            str(temp),
            "-n",
            str(steps),
            "-i",
            prompt,
        ]
        if self.tokenizer_bin and os.path.isfile(self.tokenizer_bin):
            cmd.extend(["-z", self.tokenizer_bin])

        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(f"llama2.c runner failed ({res.returncode}): {res.stderr}")

        output = res.stdout.strip()
        lines = [line for line in output.splitlines() if not line.startswith("achieved tok/s:")]
        return "\n".join(lines).strip()


class MockBackend(SLMBackend):
    """Deterministic offline fallback backend for testing and CI pipelines."""

    def __init__(self, prefix: str = "[MOCK-SLM-RESPONSE]"):
        super().__init__(name="mock")
        self.prefix = prefix

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        preview = prompt[:180].replace("\n", " ").strip()
        return (
            f"{self.prefix}\n"
            f"Code Analysis Complete.\n"
            f"- Prompt length: {len(prompt)} chars\n"
            f"- Preview: {preview}\n"
            f"- Status: Deterministic verification PASSED."
        )


class BackendRouter:
    """Selects and dispatches requests to the best available SLM backend."""

    def __init__(self, backends: Optional[List[SLMBackend]] = None):
        self.backends = backends or [
            OllamaBackend(),
            OpenAICompatBackend(),
            Llama2cBackend(),
            LlamaCppCliBackend(),
            MockBackend(),
        ]

    def get_active_backend(self) -> SLMBackend:
        for b in self.backends:
            if b.is_available():
                return b
        return MockBackend()

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        active = self.get_active_backend()
        logger.info("Routing generation request to backend: %s", active.name)
        return active.generate(prompt=prompt, system_prompt=system_prompt, **kwargs)


# end of file: plowinf/src/plowinf/backends.py
