# Path: plowinf/src/plowinf/__init__.py
"""Plowinf - Offline AI inference engine, SLM code worker, and queue processor.

Provides tools to explore local AI inference, process code via Ollama and other
local runtimes (OpenAI-compatible, llama-cli, llama2.c), enforce local-only
air-gap safety, and audit operations with cryptographic forensics.
"""

from .backends import (
    BackendRouter,
    Llama2cBackend,
    LlamaCppCliBackend,
    MockBackend,
    OllamaBackend,
    OpenAICompatBackend,
    SLMBackend,
)
from .forensics import (
    ForensicAuditLogger,
    compute_file_sha256,
    compute_str_sha256,
    get_telemetry_snapshot,
)
from .patcher import (
    CodePatcher,
    extract_code_block,
    parse_work_product_metadata,
)
from .queue import (
    FileTaskQueue,
    Task,
    TaskStatus,
)
from .safety import (
    AdditiveFileHandler,
    AirGapViolationError,
    CANONICAL_ROOT_FILES,
    CANONICAL_SOURCE_DIRS,
    GitDestructiveOperationError,
    GitSafetyGuard,
    validate_additive_artifact_path,
    validate_local_only_url,
)
from .worker import (
    CodeWorker,
    wrap_markdown_text,
)

__version__ = "0.1.0"

__all__ = [
    "AdditiveFileHandler",
    "AirGapViolationError",
    "BackendRouter",
    "CANONICAL_ROOT_FILES",
    "CANONICAL_SOURCE_DIRS",
    "CodePatcher",
    "CodeWorker",
    "FileTaskQueue",
    "ForensicAuditLogger",
    "GitDestructiveOperationError",
    "GitSafetyGuard",
    "Llama2cBackend",
    "LlamaCppCliBackend",
    "MockBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "SLMBackend",
    "Task",
    "TaskStatus",
    "compute_file_sha256",
    "compute_str_sha256",
    "extract_code_block",
    "get_telemetry_snapshot",
    "parse_work_product_metadata",
    "validate_additive_artifact_path",
    "validate_local_only_url",
    "wrap_markdown_text",
]

# end of file: plowinf/src/plowinf/__init__.py
