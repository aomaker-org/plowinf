# Path: plowinf/tests/test_safety.py
"""Unit tests for air-gap validation and additive file/Git safety guardrails."""

from pathlib import Path
import pytest

from plowinf.safety import (
    AdditiveFileHandler,
    AirGapViolationError,
    GitDestructiveOperationError,
    GitSafetyGuard,
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


# end of file: plowinf/tests/test_safety.py
