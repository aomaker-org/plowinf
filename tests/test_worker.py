# Path: plowinf/tests/test_worker.py
"""Unit tests for the CodeWorker engine in plowinf."""

import json
from pathlib import Path
import pytest

from plowinf.backends import BackendRouter, MockBackend
from plowinf.forensics import ForensicAuditLogger
from plowinf.queue import FileTaskQueue
from plowinf.worker import CodeWorker


@pytest.fixture
def worker_setup(tmp_path):
    q_dir = tmp_path / "queue"
    log_dir = tmp_path / "logs"
    hb_path = log_dir / "heartbeat.json"
    f_dir = tmp_path / "forensics"
    wp_dir = tmp_path / "work_products"

    queue = FileTaskQueue(base_dir=str(q_dir))
    router = BackendRouter(backends=[MockBackend()])
    forensics = ForensicAuditLogger(base_dir=str(f_dir))
    worker = CodeWorker(
        queue=queue,
        router=router,
        forensics=forensics,
        log_dir=str(log_dir),
        heartbeat_path=str(hb_path),
        work_product_dir=str(wp_dir),
    )
    return worker, queue, hb_path


def test_build_prompt(worker_setup):
    worker, _, _ = worker_setup
    p_review = worker.build_prompt("review", "def foo(): pass", "check style")
    assert "senior code reviewer" in p_review
    assert "check style" in p_review

    p_test = worker.build_prompt("test", "def bar(): pass")
    assert "test engineer" in p_test

    p_doc = worker.build_prompt("docstring", "class Baz: pass")
    assert "PEP 257" in p_doc


def test_emit_heartbeat(worker_setup):
    worker, _, hb_path = worker_setup
    worker.emit_heartbeat(status="testing")

    assert hb_path.exists()
    data = json.loads(hb_path.read_text(encoding="utf-8"))
    assert data["component"] == "plowinf"
    assert data["status"] == "testing"
    assert "timestamp" in data
    assert "rss_mb" in data
    assert data["active_backend"] == "mock"


def test_process_task_successful(worker_setup, tmp_path):
    worker, queue, _ = worker_setup
    sample_code = tmp_path / "sample.py"
    sample_code.write_text("def multiply(a, b): return a * b\n", encoding="utf-8")

    task = queue.enqueue(action="review", code_path=str(sample_code))
    claimed = queue.dequeue()
    assert claimed is not None

    success = worker.process_task(claimed)
    assert success is True
    assert worker.processed_count == 1
    assert worker.failed_count == 0

    done_tasks = queue.list_tasks(status="done")
    assert len(done_tasks) == 1
    assert "Code Analysis Complete" in (done_tasks[0].result or "")


def test_daemon_single_cycle(worker_setup, tmp_path):
    worker, queue, _ = worker_setup
    sample_code = tmp_path / "cycle.py"
    sample_code.write_text("x = 42\n", encoding="utf-8")

    queue.enqueue(action="test", code_path=str(sample_code))
    worker.run_daemon(interval=0.1, run_once=True)

    summary = queue.get_status_summary()
    assert summary["done"] == 1
    assert summary["incoming"] == 0


# end of file: plowinf/tests/test_worker.py
