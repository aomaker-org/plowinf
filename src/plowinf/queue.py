# Path: plowinf/src/plowinf/queue.py
"""File-backed asynchronous task queue for plowinf.

Implements an atomic directory-based state machine with POSIX advisory file
locking (fcntl.flock) across:
  - incoming/   : New unassigned tasks waiting for execution
  - processing/ : Tasks currently being handled by a plower worker
  - done/       : Successfully completed tasks with generated results
  - failed/     : Tasks that encountered unrecoverable errors
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import fcntl
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger("plowinf.queue")


class TaskStatus(str, Enum):
    INCOMING = "incoming"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    action: str
    code_path: str = ""
    prompt: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = TaskStatus.INCOMING.value
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    backend_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(**data)


class FileTaskQueue:
    """Manages atomic file-based task scheduling with fcntl locking."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            self.base_dir = Path("records/queue").resolve()

        self.incoming_dir = self.base_dir / "incoming"
        self.processing_dir = self.base_dir / "processing"
        self.done_dir = self.base_dir / "done"
        self.failed_dir = self.base_dir / "failed"

        for d in [self.incoming_dir, self.processing_dir, self.done_dir, self.failed_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        action: str,
        code_path: str = "",
        prompt: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Enqueue a new task into incoming/."""
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        task = Task(
            task_id=task_id,
            action=action,
            code_path=code_path,
            prompt=prompt,
            extra=extra or {},
        )
        task_file = self.incoming_dir / f"{task_id}.json"
        tmp_file = self.incoming_dir / f".{task_id}.json.tmp"

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2)
        os.replace(tmp_file, task_file)

        logger.info("Enqueued task %s (%s) -> %s", task_id, action, task_file)
        return task

    def dequeue(self, timeout: float = 0.0) -> Optional[Task]:
        """Atomically claim the oldest task from incoming/ into processing/."""
        start_time = time.time()
        while True:
            candidates = sorted(self.incoming_dir.glob("task-*.json"), key=os.path.getmtime)
            for path in candidates:
                lock_file_path = path.with_suffix(".lock")
                try:
                    lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_RDWR)
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (BlockingIOError, OSError):
                    continue

                try:
                    if not path.exists():
                        continue
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    task = Task.from_dict(data)
                    task.status = TaskStatus.PROCESSING.value
                    task.started_at = datetime.now(timezone.utc).isoformat()

                    dest_path = self.processing_dir / path.name
                    with open(dest_path, "w", encoding="utf-8") as f:
                        json.dump(task.to_dict(), f, indent=2)
                    path.unlink(missing_ok=True)
                    logger.info("Claimed task %s -> %s", task.task_id, dest_path)
                    return task
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                    lock_file_path.unlink(missing_ok=True)

            if timeout <= 0.0 or (time.time() - start_time) >= timeout:
                return None
            time.sleep(0.5)

    def complete_task(
        self,
        task: Task,
        result: str,
        backend_used: Optional[str] = None,
    ) -> None:
        """Mark task as done and move JSON into done/."""
        task.status = TaskStatus.DONE.value
        task.result = result
        task.backend_used = backend_used
        task.completed_at = datetime.now(timezone.utc).isoformat()

        proc_file = self.processing_dir / f"{task.task_id}.json"
        done_file = self.done_dir / f"{task.task_id}.json"

        with open(done_file, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2)
        proc_file.unlink(missing_ok=True)
        logger.info("Task %s completed successfully.", task.task_id)

    def fail_task(self, task: Task, error: str) -> None:
        """Mark task as failed and move JSON into failed/."""
        task.status = TaskStatus.FAILED.value
        task.error = error
        task.completed_at = datetime.now(timezone.utc).isoformat()

        proc_file = self.processing_dir / f"{task.task_id}.json"
        failed_file = self.failed_dir / f"{task.task_id}.json"

        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2)
        proc_file.unlink(missing_ok=True)
        logger.warning("Task %s marked as failed: %s", task.task_id, error)

    def get_status_summary(self) -> Dict[str, int]:
        """Return counts of tasks in each queue partition."""
        return {
            "incoming": len(list(self.incoming_dir.glob("task-*.json"))),
            "processing": len(list(self.processing_dir.glob("task-*.json"))),
            "done": len(list(self.done_dir.glob("task-*.json"))),
            "failed": len(list(self.failed_dir.glob("task-*.json"))),
        }

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        """List tasks matching status or across all partitions."""
        dirs = []
        if status == TaskStatus.INCOMING.value or status is None:
            dirs.append(self.incoming_dir)
        if status == TaskStatus.PROCESSING.value or status is None:
            dirs.append(self.processing_dir)
        if status == TaskStatus.DONE.value or status is None:
            dirs.append(self.done_dir)
        if status == TaskStatus.FAILED.value or status is None:
            dirs.append(self.failed_dir)

        tasks = []
        for d in dirs:
            for p in d.glob("task-*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        tasks.append(Task.from_dict(json.load(f)))
                except Exception as exc:
                    logger.warning("Failed to parse task file %s: %s", p, exc)
        return tasks


# end of file: plowinf/src/plowinf/queue.py
