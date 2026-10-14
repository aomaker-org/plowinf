# Path: plowinf/tests/test_safety.py
"""Unit tests for air-gap validation and additive file/Git safety guardrails."""

from pathlib import Path
import pytest

from plowinf.safety import (
    CANONICAL_ROOT_FILES,
    CANONICAL_SOURCE_DIRS,
    AdditiveFileHandler,
    AirGapViolationError,
    GitDestructiveOperationError,
    GitSafetyGuard,
    validate_additive_artifact_path,
    validate_local_only_url,
)


def test_air_gap_validation_allows_loopback():
    allowed_urls = [
        "http://127.0.0.1:11434",
        "http://127.0.0.1:8000/v1",
        "http://localhost:11434",
        "http://localhost:8080",
        "http://[::1]:11434",
        "http://0.0.0.0:11434",
    ]
    for url in allowed_urls:
        validate_local_only_url(url)  # Must not raise


def test_air_gap_validation_rejects_external_urls():
    blocked_urls = [
        "https://api.openai.com/v1",
        "https://api.anthropic.com/v1",
        "http://163.192.22.222:11434",
        "http://remote-server.com:8000",
        "http://192.168.1.50:11434",
    ]
    for url in blocked_urls:
        with pytest.raises(AirGapViolationError):
            validate_local_only_url(url)


def test_additive_file_handler_creates_initial(tmp_path):
    target = tmp_path / "notes.md"
    written = AdditiveFileHandler.safe_write_additive(target, "Version 1")
    assert written == target
    assert written.read_text(encoding="utf-8") == "Version 1"


def test_additive_file_handler_idempotent_on_identical(tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("Version 1", encoding="utf-8")

    written = AdditiveFileHandler.safe_write_additive(target, "Version 1")
    assert written == target


def test_additive_file_handler_creates_rev_on_difference(tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("Version 1 original content", encoding="utf-8")

    # Writing different content must NOT overwrite original target
    written = AdditiveFileHandler.safe_write_additive(target, "Version 2 new content")
    assert written != target
    assert written.name == "notes_rev001.md"
    assert target.read_text(encoding="utf-8") == "Version 1 original content"
    assert written.read_text(encoding="utf-8") == "Version 2 new content"

    # Writing a 3rd different content must create rev002
    written3 = AdditiveFileHandler.safe_write_additive(target, "Version 3 third content")
    assert written3.name == "notes_rev002.md"
    assert target.read_text(encoding="utf-8") == "Version 1 original content"


def test_git_safety_guard_blocks_destructive_commands():
    dangerous_commands = [
        ["git", "push", "--force", "origin", "main"],
        ["git", "push", "-f", "origin", "main"],
        ["git", "push", "--force-with-lease", "origin", "main"],
        ["git", "reset", "--hard", "HEAD~1"],
        ["git", "clean", "-fd"],
        ["git", "rebase", "main"],
    ]
    for cmd in dangerous_commands:
        with pytest.raises(GitDestructiveOperationError):
            GitSafetyGuard.audit_command_safety(cmd)


def test_git_safety_guard_allows_additive_commands():
    safe_commands = [
        ["git", "add", "file.py"],
        ["git", "commit", "-m", "[agy] Additive artifact"],
        ["git", "checkout", "-b", "feature/safe-branch"],
        ["git", "status"],
    ]
    for cmd in safe_commands:
        GitSafetyGuard.audit_command_safety(cmd)  # Must not raise


def test_validate_additive_artifact_path_allows_external_tmp(tmp_path):
    target = tmp_path / "notes.md"
    validated = validate_additive_artifact_path(target)
    assert validated == target.resolve()


def test_validate_additive_artifact_path_allows_workspace_artifacts(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    allowed_rel_paths = [
        "workspace/work_products/docstrings/task_001.md",
        "logs/heartbeat/plowinf.json",
        "queue/task_001.json",
        "artifacts/benchmarks/run.json",
    ]
    for rel in allowed_rel_paths:
        target = repo_root / rel
        validated = validate_additive_artifact_path(target, repo_root=repo_root)
        assert validated == target.resolve()


def test_validate_additive_artifact_path_rejects_canonical_sources(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    blocked_rel_paths = [
        "src/plowinf/worker.py",
        "tests/test_safety.py",
        "docs/gemini_web_agy_guide.md",
        "tools/test_backends_perf.py",
    ]
    for rel in blocked_rel_paths:
        target = repo_root / rel
        with pytest.raises(ValueError, match="Directive 07 violation"):
            validate_additive_artifact_path(target, repo_root=repo_root)


def test_validate_additive_artifact_path_rejects_canonical_root_files(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    for root_file in CANONICAL_ROOT_FILES:
        target = repo_root / root_file
        with pytest.raises(ValueError, match="Directive 07 violation"):
            validate_additive_artifact_path(target, repo_root=repo_root)


def test_additive_file_handler_rejects_canonical_source(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target = repo_root / "src" / "plowinf" / "module.py"

    with pytest.raises(ValueError, match="Directive 07 violation"):
        AdditiveFileHandler.safe_write_additive(
            target,
            "new_content = True",
            repo_root=repo_root,
        )

    with pytest.raises(ValueError, match="Directive 07 violation"):
        AdditiveFileHandler.get_additive_path(target, repo_root=repo_root)


# end of file: plowinf/tests/test_safety.py
