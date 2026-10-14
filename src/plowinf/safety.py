# Path: plowinf/src/plowinf/safety.py
"""Safety, air-gap enforcer, and additive file/Git history preservation guardrails.

Enforces:
1. Strict local-only network air-gap (zero external I/O or cloud API leakage).
2. Scoped additive file operations (untracked worker artifacts use _rev001;
   canonical Git-tracked source is updated in place via feature branches).
3. Non-destructive Git history management (blocks force-push, reset --hard, etc).
"""

from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from typing import List, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger("plowinf.safety")


class AirGapViolationError(RuntimeError):
    """Raised when non-localhost or external network destinations are targeted."""
    pass


class GitDestructiveOperationError(RuntimeError):
    """Raised when a destructive Git command or option is attempted."""
    pass


def validate_local_only_url(url: str) -> None:
    """Verify that an endpoint URL strictly targets localhost/loopback."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    allowed_hosts: Set[str] = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    if hostname not in allowed_hosts:
        raise AirGapViolationError(
            f"Air-gap mandate violation: plowinf is restricted to local-only "
            f"resources. Target host '{hostname}' is not a local loopback interface."
        )


CANONICAL_SOURCE_DIRS: tuple[str, ...] = (
    "src",
    "tests",
    "docs",
    "tools",
)
CANONICAL_ROOT_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "Makefile",
    "AI.md",
    "README.md",
)


def validate_additive_artifact_path(
    target_path: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    """Ensure AdditiveFileHandler (_rev001) is not used on Git-tracked source.

    Per Directive 07, canonical repository source files under src/, tests/,
    docs/, tools/, and root config files must retain canonical filenames and
    rely on atomic Git commits for history rather than _revNNN duplicates.
    """
    resolved_target = Path(target_path).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )

    try:
        rel_path = resolved_target.relative_to(root)
    except ValueError:
        # Target is outside the repository root (e.g., /tmp or pytest tmp_path)
        return resolved_target

    parts = rel_path.parts
    if not parts:
        return resolved_target

    if parts[0] in CANONICAL_SOURCE_DIRS or (
        len(parts) == 1 and parts[0] in CANONICAL_ROOT_FILES
    ):
        raise ValueError(
            f"Directive 07 violation: '{rel_path}' is inside a canonical "
            f"Git-tracked path ({parts[0]}). Update canonical source files "
            f"in place on a feature/* branch; reserve AdditiveFileHandler "
            f"(_rev001) for untracked worker artifacts and drafts."
        )

    return resolved_target


class AdditiveFileHandler:
    """Guarantees all file operations create new versioned files without overwriting."""

    @staticmethod
    def compute_sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def get_additive_path(
        cls,
        target_path: str | Path,
        repo_root: str | Path | None = None,
    ) -> Path:
        """Return an uncollided additive file path with numeric revision suffix."""
        resolved = validate_additive_artifact_path(target_path, repo_root=repo_root)
        if not resolved.exists():
            return resolved

        parent = resolved.parent
        stem = resolved.stem
        suffix = resolved.suffix

        revision = 1
        while True:
            candidate = parent / f"{stem}_rev{revision:03d}{suffix}"
            if not candidate.exists():
                return candidate
            revision += 1

    @classmethod
    def safe_write_additive(
        cls,
        target_path: str | Path,
        content: str,
        repo_root: str | Path | None = None,
    ) -> Path:
        """Write content additively, guaranteeing existing files are never overwritten."""
        resolved = validate_additive_artifact_path(target_path, repo_root=repo_root)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        if resolved.exists():
            existing_content = resolved.read_text(encoding="utf-8")
            if cls.compute_sha256(existing_content) == cls.compute_sha256(content):
                logger.info(
                    "File %s already exists with identical content. Preserving.",
                    resolved,
                )
                return resolved
            additive_path = cls.get_additive_path(resolved, repo_root=repo_root)
            logger.info(
                "Target %s exists with different content. Writing additively to %s",
                resolved,
                additive_path,
            )
            resolved = additive_path

        resolved.write_text(content, encoding="utf-8")
        return resolved


class GitSafetyGuard:
    """Protects Git history integrity by enforcing strictly additive commits."""

    PROHIBITED_FLAGS = {
        "--force",
        "-f",
        "--force-with-lease",
        "--hard",
        "rebase",
    }

    @classmethod
    def audit_command_safety(cls, cmd: List[str]) -> None:
        """Audit command arguments to prohibit history-destructive options."""
        tokens = [arg.lower() for arg in cmd]
        for token in tokens:
            if token in cls.PROHIBITED_FLAGS:
                raise GitDestructiveOperationError(
                    f"Prohibited destructive Git operation detected ('{token}'). "
                    f"plowinf enforces strict append-only history immutability."
                )
        for i, token in enumerate(tokens):
            if token == "clean" and i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                if next_tok in {"-f", "-fd", "-df", "--force"}:
                    raise GitDestructiveOperationError(
                        f"Prohibited destructive Git operation detected ('clean {next_tok}'). "
                        f"plowinf enforces strict append-only history immutability."
                    )

    @classmethod
    def commit_additive(
        cls,
        repo_root: Path,
        file_paths: List[Path],
        commit_message: str,
    ) -> Optional[str]:
        """Safely stage and commit files additively without altering history."""
        cls.audit_command_safety(["git", "commit", commit_message])

        for p in file_paths:
            add_cmd = ["git", "-C", str(repo_root), "add", str(p)]
            cls.audit_command_safety(add_cmd)
            subprocess.run(add_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        msg = f"[agy] {commit_message}"
        commit_cmd = ["git", "-C", str(repo_root), "commit", "-m", msg]
        cls.audit_command_safety(commit_cmd)
        res = subprocess.run(
            commit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0:
            logger.info("Additive git commit created: %s", msg)
            return res.stdout.strip()
        return None


# end of file: plowinf/src/plowinf/safety.py
