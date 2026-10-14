# Path: plowinf/tests/test_patcher.py
"""Unit tests for CodePatcher and safe additive patch application engine."""

from pathlib import Path
import subprocess
import pytest

from plowinf.forensics import ForensicAuditLogger
from plowinf.patcher import (
    CodePatcher,
    extract_code_block,
    parse_work_product_metadata,
)


def test_extract_code_block_basic():
    text = """Here is some refactored code:
```javascript
function hello() {
  return "world";
}
```
Done with refactoring.
"""
    code = extract_code_block(text)
    assert code == 'function hello() {\n  return "world";\n}'


def test_extract_code_block_prefers_language():
    text = """Here is a bash snippet:
```bash
echo "ignore me"
```
And the actual js code:
```javascript
const answer = 42;
```
"""
    code = extract_code_block(text, preferred_lang="javascript")
    assert code == "const answer = 42;"


def test_extract_code_block_returns_none_when_empty():
    text = "No code blocks here, just plain text review comments."
    assert extract_code_block(text) is None


def test_parse_work_product_metadata():
    text = """# Path: workspace/work_products/prompts/test.md
# Work Product: PROMPT - heap
# Task ID: task-test1234
# Target File: workspace/targets/mineflayer-pathfinder/lib/heap.js
# Backend: ollama

```javascript
const x = 1;
```
"""
    meta = parse_work_product_metadata(text)
    assert meta.get("task_id") == "task-test1234"
    assert meta.get("target_file") == "workspace/targets/mineflayer-pathfinder/lib/heap.js"
    assert meta.get("action") == "PROMPT - heap"


def test_patcher_dry_run_does_not_modify_target(tmp_path):
    forensics = ForensicAuditLogger(base_dir=str(tmp_path / "forensics"))
    patcher = CodePatcher(repo_root=tmp_path, forensics=forensics)

    target = tmp_path / "target.js"
    target.write_text("const oldVal = 1;\n", encoding="utf-8")

    wp = tmp_path / "wp.md"
    wp.write_text("""# Task ID: task-001
# Target File: target.js

```javascript
const newVal = 2;
```
""", encoding="utf-8")

    result = patcher.apply_patch(work_product_file=wp, dry_run=True)
    assert result["dry_run"] is True
    assert result["applied"] is False
    assert "-const oldVal = 1;" in result["diff"]
    assert "+const newVal = 2;" in result["diff"]
    assert target.read_text(encoding="utf-8") == "const oldVal = 1;\n"


def test_patcher_live_apply_with_backup(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.local"], check=True)

    target = tmp_path / "target.js"
    target.write_text("function original() { return 1; }\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "target.js"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True)

    wp = tmp_path / "wp.md"
    wp.write_text("""# Task ID: task-live-01
# Target File: target.js

```javascript
function refactored() { return 2; }
```
""", encoding="utf-8")

    forensics = ForensicAuditLogger(base_dir=str(tmp_path / "forensics"))
    patcher = CodePatcher(repo_root=tmp_path, forensics=forensics)

    result = patcher.apply_patch(work_product_file=wp, dry_run=False)
    assert result["applied"] is True
    assert "refactored" in target.read_text(encoding="utf-8")

    backup_path = Path(result["backup_file"])
    assert backup_path.exists()
    assert "original" in backup_path.read_text(encoding="utf-8")

    history = forensics.get_task_history(task_id="task-live-01")
    assert any(e.get("action") == "apply" for e in history)


def test_patcher_raises_on_missing_target(tmp_path):
    wp = tmp_path / "wp.md"
    wp.write_text("""# Task ID: task-err
# Target File: nonexistent.js

```javascript
const a = 1;
```
""", encoding="utf-8")

    patcher = CodePatcher(repo_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        patcher.apply_patch(work_product_file=wp, dry_run=True)


def test_patcher_creates_new_target_file_when_specified(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.local"], check=True)

    dummy = tmp_path / "README.md"
    dummy.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True)

    wp = tmp_path / "wp_py.md"
    wp.write_text("""# Task ID: task-port-01
```python
def solve():
    return 42
```
""", encoding="utf-8")

    forensics = ForensicAuditLogger(base_dir=str(tmp_path / "forensics"))
    patcher = CodePatcher(repo_root=tmp_path, forensics=forensics)
    new_target = tmp_path / "experimental" / "solution.py"

    result = patcher.apply_patch(work_product_file=wp, target_file=new_target, dry_run=False)
    assert result["applied"] is True
    assert new_target.exists()
    assert "solve" in new_target.read_text(encoding="utf-8")
    assert result["backup_file"] is None


# end of file: plowinf/tests/test_patcher.py
