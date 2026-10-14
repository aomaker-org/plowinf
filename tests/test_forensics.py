# Path: plowinf/tests/test_forensics.py
"""Unit tests for the ForensicAuditLogger and cryptographic ledger."""

import json
from pathlib import Path
import pytest

from plowinf.forensics import (
    ForensicAuditLogger,
    compute_file_sha256,
    compute_str_sha256,
)


@pytest.fixture
def temp_forensics(tmp_path):
    f_dir = tmp_path / "forensics"
    return ForensicAuditLogger(base_dir=str(f_dir))


def test_sha256_computations(tmp_path):
    text = "const a = 123;"
    h1 = compute_str_sha256(text)
    assert len(h1) == 64

    f = tmp_path / "sample.js"
    f.write_text(text, encoding="utf-8")
    h2 = compute_file_sha256(f)
    assert h1 == h2


def test_log_event(temp_forensics):
    rec = temp_forensics.log_event(
        phase="TASK_CLAIM",
        task_id="task-1001",
        action="review",
        details={"source": "test"},
    )
    assert rec["task_id"] == "task-1001"
    assert rec["phase"] == "TASK_CLAIM"
    assert temp_forensics.journal_path.exists()
    assert temp_forensics.trace_log_path.exists()


def test_record_inference_audit(temp_forensics, tmp_path):
    code_file = tmp_path / "dummy.py"
    code_file.write_text("def ping(): return 'pong'", encoding="utf-8")

    rec = temp_forensics.record_inference_audit(
        task_id="task-2002",
        action="test",
        code_path=str(code_file),
        code_content="def ping(): return 'pong'",
        full_prompt="Generate test for ping",
        response="def test_ping(): assert ping() == 'pong'",
        backend_name="mock",
        duration_ms=125.5,
        status="SUCCESS",
        work_product_path="workspace/work_products/tests/dummy_test.py",
    )
    assert rec["task_id"] == "task-2002"
    assert rec["target"]["sha256"] != ""
    assert rec["prompt"]["sha256"] != ""
    assert rec["response"]["sha256"] != ""

    history = temp_forensics.get_task_history(task_id="task-2002")
    assert len(history) == 1
    assert history[0]["backend"] == "mock"


# end of file: plowinf/tests/test_forensics.py
