# Path: plowinf/src/plowinf/worker.py
"""Core execution engine for plowinf.

Orchestrates local model generation, task dispatching, heartbeat emission,
stream logging, and forensic audit logging adhering to Directives 02, 17, and 18.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, Optional

from .backends import BackendRouter, SLMBackend
from .forensics import ForensicAuditLogger
from .queue import FileTaskQueue, Task, TaskStatus
from .safety import AdditiveFileHandler, GitSafetyGuard

logger = logging.getLogger("plowinf.worker")


def wrap_markdown_text(text: str, width: int = 80) -> str:
    """Wrap non-code block lines to maximum width adhering to Directive 04."""
    import textwrap
    lines = text.splitlines()
    out = []
    in_code = False
    for l in lines:
        if l.strip().startswith("```"):
            in_code = not in_code
            out.append(l)
            continue
        if in_code or len(l) <= width:
            out.append(l)
        else:
            if l.lstrip().startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                indent = "   " if l.lstrip()[:2].isdigit() else "  "
                out.append(textwrap.fill(l, width=width, subsequent_indent=indent))
            else:
                out.append(textwrap.fill(l, width=width))
    return "\n".join(out)


class CodeWorker:
    """Worker daemon executing code review, test generation, and docstring tasks."""

    def __init__(
        self,
        queue: Optional[FileTaskQueue] = None,
        router: Optional[BackendRouter] = None,
        forensics: Optional[ForensicAuditLogger] = None,
        log_dir: str = "logs/plowinf",
        heartbeat_path: str = "logs/heartbeat/plowinf.json",
        work_product_dir: str = "workspace/work_products",
    ):
        self.queue = queue or FileTaskQueue()
        self.router = router or BackendRouter()
        self.forensics = forensics or ForensicAuditLogger()
        self.log_dir = Path(log_dir).resolve()
        self.heartbeat_path = Path(heartbeat_path).resolve()
        self.work_product_dir = Path(work_product_dir).resolve()
        self.running = False
        self.processed_count = 0
        self.failed_count = 0

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.work_product_dir.mkdir(parents=True, exist_ok=True)

        self._stream_log_path = self.log_dir / f"stream_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        print(f"[*] [Directive 02/04] Active worker log: {self._stream_log_path}", file=sys.stderr)
        print(f"[*] [Directive 02] Forensic journal: {self.forensics.journal_path}", file=sys.stderr)

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write structured event to stream log."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        with open(self._stream_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def emit_heartbeat(self, status: str = "healthy") -> None:
        """Emit process heartbeat adhering to Directive 17."""
        rss_mb = 0.0
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_mb = float(line.split()[1]) / 1024.0
                        break
        except Exception:
            pass

        heartbeat = {
            "component": "plowinf",
            "pid": os.getpid(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "rss_mb": round(rss_mb, 2),
            "processed_tasks": self.processed_count,
            "failed_tasks": self.failed_count,
            "active_backend": self.router.get_active_backend().name,
            "queue_summary": self.queue.get_status_summary(),
        }
        tmp_file = self.heartbeat_path.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(heartbeat, f, indent=2)
        os.replace(tmp_file, self.heartbeat_path)

    def build_prompt(self, action: str, code: str, prompt: str = "") -> str:
        """Format action-specific instruction prompts for local models."""
        if action == "review":
            return (
                f"You are a meticulous senior code reviewer. Review the following code "
                f"for potential bugs, security flaws, performance bottlenecks, and edge cases.\n"
                f"Provide concise, actionable recommendations.\n\n"
                f"```\n{code}\n```\n"
                f"{('Additional context: ' + prompt) if prompt else ''}"
            )
        elif action == "test":
            return (
                f"You are an expert test engineer. Generate comprehensive unit tests "
                f"(using pytest) for the following code. Cover normal cases, edge cases, "
                f"and potential error conditions.\n\n"
                f"```\n{code}\n```\n"
                f"{('Requirements: ' + prompt) if prompt else ''}"
            )
        elif action == "docstring":
            return (
                f"You are a technical documentation specialist. Generate clean, PEP 257 "
                f"compliant docstrings and typing annotations for all classes and functions "
                f"in the following code:\n\n"
                f"```\n{code}\n```\n"
            )
        elif action == "prompt":
            if code:
                return f"{prompt}\n\nCode context:\n```\n{code}\n```"
            return prompt
        else:
            return f"Action: {action}\n\n{prompt}\n\n```\n{code}\n```"

    def _save_work_product(self, task: Task, code_path: str, content: str) -> Optional[str]:
        """Save generated output to workspace/work_products/."""
        try:
            base_dir = self.work_product_dir
            action_dir = base_dir / f"{task.action}s"
            action_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(code_path).stem if code_path else "general"
            out_file = action_dir / f"{task.task_id}_{stem}.md"

            rel_name = f"workspace/work_products/{task.action}s/{out_file.name}"
            banner = (
                f"# Path: plowinf/{rel_name}\n"
                f"# Work Product: {task.action.upper()} - {stem}\n"
                f"# Task ID: {task.task_id}\n"
                f"# Backend: {task.backend_used or 'local'}\n"
                f"# Generated: {task.completed_at}\n\n"
                f"{wrap_markdown_text(content, width=80)}\n\n"
                f"<!-- end of file: plowinf/{rel_name} -->\n"
            )
            saved_path = AdditiveFileHandler.safe_write_additive(out_file, banner)

            # Non-destructive additive git tracking
            try:
                if (Path.cwd() / ".git").exists():
                    GitSafetyGuard.commit_additive(
                        repo_root=Path.cwd(),
                        file_paths=[saved_path],
                        commit_message=f"work product: {task.action} for {stem} ({task.task_id})",
                    )
            except Exception as git_exc:
                logger.debug("Additive Git commit deferred: %s", git_exc)

            return str(saved_path)
        except Exception as exc:
            logger.warning("Failed to save work product for %s: %s", task.task_id, exc)
            return None

    def process_task(self, task: Task, verbose: bool = False) -> bool:
        """Execute task using the active backend router with forensic logging."""
        code_content = ""
        if task.code_path:
            p = Path(task.code_path).resolve()
            if p.exists():
                try:
                    code_content = p.read_text(encoding="utf-8")
                except Exception as exc:
                    err_msg = f"Failed to read code file {p}: {exc}"
                    self.queue.fail_task(task, err_msg)
                    self.forensics.log_event("TASK_FILE_ERROR", task.task_id, task.action, {"error": err_msg}, "ERROR")
                    self.failed_count += 1
                    return False
            else:
                err_msg = f"Code path not found: {p}"
                self.queue.fail_task(task, err_msg)
                self.forensics.log_event("TASK_FILE_ERROR", task.task_id, task.action, {"error": err_msg}, "ERROR")
                self.failed_count += 1
                return False

        full_prompt = self.build_prompt(task.action, code_content, task.prompt)
        backend = self.router.get_active_backend()
        self._log_event("task_start", {"task_id": task.task_id, "action": task.action, "backend": backend.name})
        self.forensics.log_event("INFERENCE_START", task.task_id, task.action, {"backend": backend.name})

        if verbose:
            print(f"[*] [FORENSIC] Starting inference for {task.task_id} on {backend.name}...")

        start_time = time.time()
        try:
            output = backend.generate(prompt=full_prompt)
            duration_s = round(time.time() - start_time, 3)
            duration_ms = duration_s * 1000.0

            self.queue.complete_task(task, result=output, backend_used=backend.name)
            self.processed_count += 1

            wp_path = self._save_work_product(task, task.code_path, output)

            self.forensics.record_inference_audit(
                task_id=task.task_id,
                action=task.action,
                code_path=task.code_path,
                code_content=code_content,
                full_prompt=full_prompt,
                response=output,
                backend_name=backend.name,
                duration_ms=duration_ms,
                status="SUCCESS",
                work_product_path=wp_path,
            )

            self._log_event(
                "task_complete",
                {"task_id": task.task_id, "backend": backend.name, "duration_s": duration_s, "work_product": wp_path},
            )
            if verbose:
                print(f"[+] [FORENSIC] Task {task.task_id} completed in {duration_s}s. Work product: {wp_path}")
            return True
        except Exception as exc:
            duration_s = round(time.time() - start_time, 3)
            duration_ms = duration_s * 1000.0
            self.queue.fail_task(task, error=str(exc))
            self.failed_count += 1

            self.forensics.record_inference_audit(
                task_id=task.task_id,
                action=task.action,
                code_path=task.code_path,
                code_content=code_content,
                full_prompt=full_prompt,
                response="",
                backend_name=backend.name,
                duration_ms=duration_ms,
                status="FAILED",
                error=str(exc),
            )

            self._log_event(
                "task_fail",
                {"task_id": task.task_id, "backend": backend.name, "error": str(exc), "duration_s": duration_s},
            )
            if verbose:
                print(f"[!] [FORENSIC] Task {task.task_id} failed: {exc}", file=sys.stderr)
            return False

    def run_daemon(self, interval: float = 2.0, run_once: bool = False, verbose: bool = True) -> None:
        """Run continuous background processing loop."""
        self.running = True

        def _sig_handler(signum, frame):
            print(f"\n[*] Received signal {signum}. Shutting down worker...", file=sys.stderr)
            self.running = False

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        print(f"[*] plowinf daemon started. Active backend: {self.router.get_active_backend().name}")
        self.emit_heartbeat(status="running")

        while self.running:
            task = self.queue.dequeue(timeout=0.5)
            if task:
                print(f"[*] Processing task {task.task_id} ({task.action})...")
                success = self.process_task(task, verbose=verbose)
                status_str = "SUCCESS" if success else "FAILED"
                print(f"[+] Task {task.task_id} {status_str}.")
                self.emit_heartbeat(status="running")
            else:
                self.emit_heartbeat(status="idle")

            if run_once:
                break
            time.sleep(interval)

        self.emit_heartbeat(status="stopped")
        print("[*] plowinf daemon stopped.")


# end of file: plowinf/src/plowinf/worker.py
