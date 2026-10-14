# Path: plowinf/tests/test_queue.py
"""Unit tests for the file-backed task queue in plowinf."""

import os
from pathlib import Path
import pytest

from plowinf.queue import FileTaskQueue, Task, TaskStatus


@pytest.fixture
def temp_queue(tmp_path):
    q_dir = tmp_path / "queue"
    return FileTaskQueue(base_dir=str(q_dir))


def test_queue_initialization(temp_queue):
    assert temp_queue.incoming_dir.exists()
    assert temp_queue.processing_dir.exists()
    assert temp_queue.done_dir.exists()
    assert temp_queue.failed_dir.exists()

    summary = temp_queue.get_status_summary()
    assert summary == {"incoming": 0, "processing": 0, "done": 0, "failed": 0}


def test_enqueue_and_dequeue_lifecycle(temp_queue):
    task = temp_queue.enqueue(
        action="review",
        code_path="foo.py",
        prompt="Check complexity",
    )
    assert task.status == TaskStatus.INCOMING.value
    assert (temp_queue.incoming_dir / f"{task.task_id}.json").exists()

    summary = temp_queue.get_status_summary()
    assert summary["incoming"] == 1

    claimed = temp_queue.dequeue(timeout=0.1)
    assert claimed is not None
    assert claimed.task_id == task.task_id
    assert claimed.status == TaskStatus.PROCESSING.value
    assert (temp_queue.processing_dir / f"{task.task_id}.json").exists()
    assert not (temp_queue.incoming_dir / f"{task.task_id}.json").exists()


def test_complete_task(temp_queue):
    task = temp_queue.enqueue(action="test", code_path="bar.py")
    claimed = temp_queue.dequeue()
    assert claimed is not None

    temp_queue.complete_task(claimed, result="Tests passed", backend_used="mock")
    assert not (temp_queue.processing_dir / f"{task.task_id}.json").exists()
    assert (temp_queue.done_dir / f"{task.task_id}.json").exists()

    summary = temp_queue.get_status_summary()
    assert summary["done"] == 1
    assert summary["processing"] == 0


def test_fail_task(temp_queue):
    task = temp_queue.enqueue(action="docstring", code_path="baz.py")
    claimed = temp_queue.dequeue()
    assert claimed is not None

    temp_queue.fail_task(claimed, error="Syntax error in source code")
    assert not (temp_queue.processing_dir / f"{task.task_id}.json").exists()
    assert (temp_queue.failed_dir / f"{task.task_id}.json").exists()

    summary = temp_queue.get_status_summary()
    assert summary["failed"] == 1
    assert summary["processing"] == 0


# end of file: plowinf/tests/test_queue.py
