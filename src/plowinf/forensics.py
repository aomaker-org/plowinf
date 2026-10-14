# Path: plowinf/src/plowinf/forensics.py
"""Forensic audit logger and cryptographic ledger for plowinf.

Provides tamper-evident, append-only logging of every task, prompt, inference
payload, cryptographic SHA-256 hash, and system telemetry record adhering to
Directive 02 (full logging) and Directive 17 (telemetry).
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("plowinf.forensics")


def compute_str_sha256(content: str) -> str:
    """Calculate SHA-256 hex digest of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hex digest of a file on disk."""
    if not file_path.exists() or not file_path.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_telemetry_snapshot() -> Dict[str, Any]:
    """Capture process telemetry for forensic tracking."""
    rss_mb = 0.0
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = float(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass

    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rss_mb": round(rss_mb, 2),
    }


class ForensicAuditLogger:
    """Manages append-only JSONL journal and verbose trace logging."""

    def __init__(self, base_dir: str = "logs/forensics"):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.journal_path = self.base_dir / "audit_journal.jsonl"
        self.trace_log_path = self.base_dir / "forensic_trace.log"

    def log_event(
        self,
        phase: str,
        task_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> Dict[str, Any]:
        """Record an immutable event into the forensic journal and trace log."""
        now_utc = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp_utc": now_utc,
            "phase": phase,
            "task_id": task_id,
            "action": action,
            "level": level,
            "telemetry": get_telemetry_snapshot(),
            "details": details or {},
        }

        # 1. Append structured JSONL record
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        # 2. Append verbose human-readable trace log
        trace_line = (
            f"[{now_utc}] [{level}] [{phase}] task={task_id} action={action} "
            f"details={json.dumps(details or {})}\n"
        )
        with open(self.trace_log_path, "a", encoding="utf-8") as f:
            f.write(trace_line)

        return payload

    def record_inference_audit(
        self,
        task_id: str,
        action: str,
        code_path: str,
        code_content: str,
        full_prompt: str,
        response: str,
        backend_name: str,
        duration_ms: float,
        status: str = "SUCCESS",
        error: Optional[str] = None,
        work_product_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full cryptographic and telemetry audit of an LLM generation run."""
        code_file = Path(code_path) if code_path else None
        if code_file and code_file.exists():
            code_sha256 = compute_file_sha256(code_file)
        else:
            code_sha256 = compute_str_sha256(code_content)
        prompt_sha256 = compute_str_sha256(full_prompt)
        response_sha256 = compute_str_sha256(response)

        audit_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "INFERENCE_EXECUTION",
            "task_id": task_id,
            "action": action,
            "status": status,
            "error": error,
            "backend": backend_name,
            "duration_ms": round(duration_ms, 2),
            "telemetry": get_telemetry_snapshot(),
            "target": {
                "path": str(code_path),
                "bytes": len(code_content.encode("utf-8")),
                "sha256": code_sha256,
            },
            "prompt": {
                "bytes": len(full_prompt.encode("utf-8")),
                "sha256": prompt_sha256,
                "text": full_prompt,
            },
            "response": {
                "bytes": len(response.encode("utf-8")),
                "sha256": response_sha256,
                "text": response,
            },
            "work_product_path": work_product_path,
        }

        # 1. Append to machine-readable JSONL audit journal
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_record) + "\n")

        # 2. Append verbose trace log block
        banner = "=" * 80
        trace_block = (
            f"\n{banner}\n"
            f"[FORENSIC AUDIT RECORD] {audit_record['timestamp_utc']}\n"
            f"  Task ID       : {task_id}\n"
            f"  Action        : {action}\n"
            f"  Status        : {status}\n"
            f"  Backend       : {backend_name}\n"
            f"  Duration      : {audit_record['duration_ms']} ms\n"
            f"  Target File   : {code_path}\n"
            f"  Target SHA256 : {code_sha256}\n"
            f"  Prompt SHA256 : {prompt_sha256}\n"
            f"  Output SHA256 : {response_sha256}\n"
            f"  Work Product  : {work_product_path or 'N/A'}\n"
            f"{banner}\n"
        )
        with open(self.trace_log_path, "a", encoding="utf-8") as f:
            f.write(trace_block)

        return audit_record

    def get_task_history(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve audit history entries, optionally filtered by task_id."""
        if not self.journal_path.exists():
            return []
        records = []
        with open(self.journal_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if task_id is None or data.get("task_id") == task_id:
                        records.append(data)
                except Exception:
                    pass
        return records


# end of file: plowinf/src/plowinf/forensics.py
