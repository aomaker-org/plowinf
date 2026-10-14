# Path: plowinf/src/plowinf/patcher.py
"""Safe code extraction and additive patch application engine for plowinf.

Extracts validated code blocks from work products and applies them to target
source files with automated backup creation, unified diff preview, and
additive Git commit tracking.
"""

from datetime import datetime, timezone
import difflib
import logging
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Optional

from .forensics import ForensicAuditLogger, compute_str_sha256
from .safety import GitSafetyGuard

logger = logging.getLogger("plowinf.patcher")


def extract_code_block(markdown_text: str, preferred_lang: Optional[str] = None) -> Optional[str]:
    """Extract code contents from the first matching markdown fenced code block."""
    pattern = r"```(?:([a-zA-Z0-9_\-]+)\n)?(.*?)```"
    matches = list(re.finditer(pattern, markdown_text, re.DOTALL))
    if not matches:
        return None

    if preferred_lang:
        for m in matches:
            lang = (m.group(1) or "").lower()
            if lang == preferred_lang.lower():
                return m.group(2).strip()

    # Default to the largest code block in the artifact
    largest = max(matches, key=lambda m: len(m.group(2)))
    return largest.group(2).strip()


def parse_work_product_metadata(markdown_text: str) -> Dict[str, str]:
    """Extract task ID, target file, and action headers from work product."""
    metadata = {}
    for line in markdown_text.splitlines()[:20]:
        line_clean = line.strip()
        if line_clean.startswith("# Task ID:"):
            metadata["task_id"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("# Target File:"):
            metadata["target_file"] = line_clean.split(":", 1)[1].strip()
        elif line_clean.startswith("# Work Product:"):
            metadata["action"] = line_clean.split(":", 1)[1].strip()
    return metadata


class CodePatcher:
    """Safely applies refactorings from work products to codebase targets."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        forensics: Optional[ForensicAuditLogger] = None,
    ):
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.forensics = forensics or ForensicAuditLogger()

    def generate_diff(self, original_text: str, new_text: str, file_name: str) -> str:
        """Produce unified diff string between original and refactored code."""
        orig_lines = original_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{file_name}",
            tofile=f"b/{file_name}",
        )
        return "".join(diff)

    def apply_patch(
        self,
        work_product_file: Path | str,
        target_file: Optional[Path | str] = None,
        dry_run: bool = False,
        create_branch: bool = False,
    ) -> Dict[str, Any]:
        """Apply refactored code from work product to target source file."""
        wp_input = str(work_product_file)
        wp_path = Path(wp_input)
        if not wp_path.is_absolute():
            candidate = (self.repo_root / wp_input).resolve()
            if candidate.exists():
                wp_path = candidate
            else:
                matches = list(self.repo_root.glob(f"workspace/work_products/**/{wp_input}*.md"))
                if matches:
                    wp_path = matches[0].resolve()
                else:
                    raise FileNotFoundError(f"Work product file not found: {wp_input}")
        else:
            wp_path = wp_path.resolve()
            if not wp_path.exists():
                raise FileNotFoundError(f"Work product file not found: {wp_path}")

        wp_text = wp_path.read_text(encoding="utf-8")
        metadata = parse_work_product_metadata(wp_text)
        task_id = metadata.get("task_id", wp_path.stem.split("_")[0])

        # Resolve target source file
        if target_file:
            resolved_target = Path(target_file).resolve()
        elif "target_file" in metadata:
            resolved_target = (self.repo_root / metadata["target_file"]).resolve()
        else:
            # Fallback heuristic: match stem in workspace/targets
            stem = wp_path.stem.split("_")[-1]
            candidates = (
                list(self.repo_root.glob(f"workspace/targets/**/{stem}.py"))
                + list(self.repo_root.glob(f"workspace/targets/**/{stem}.js"))
                + list(self.repo_root.glob(f"workspace/targets/**/{stem}.ts"))
            )
            if not candidates:
                raise ValueError(f"Could not automatically locate target source file for stem '{stem}'.")
            resolved_target = candidates[0].resolve()

        if not resolved_target.exists():
            if target_file:
                resolved_target.parent.mkdir(parents=True, exist_ok=True)
                original_code = ""
            else:
                raise FileNotFoundError(f"Target file does not exist: {resolved_target}")
        else:
            original_code = resolved_target.read_text(encoding="utf-8")

        extracted_code = extract_code_block(wp_text)
        if not extracted_code:
            raise ValueError(f"No fenced code block found in work product: {wp_path}")

        orig_sha = compute_str_sha256(original_code)
        new_sha = compute_str_sha256(extracted_code)

        diff = self.generate_diff(original_code, extracted_code, resolved_target.name)

        result: Dict[str, Any] = {
            "task_id": task_id,
            "work_product": str(wp_path),
            "target": str(resolved_target),
            "orig_sha256": orig_sha,
            "new_sha256": new_sha,
            "dry_run": dry_run,
            "diff": diff,
            "applied": False,
            "backup_file": None,
            "commit": None,
        }

        if dry_run:
            logger.info("Dry-run patch preview generated for %s (no changes written).", resolved_target)
            return result

        # Create isolated branch if requested
        if create_branch:
            branch_name = f"refactor/{resolved_target.stem}-{task_id}"
            branch_cmd = ["git", "-C", str(self.repo_root), "checkout", "-b", branch_name]
            GitSafetyGuard.audit_command_safety(branch_cmd)
            subprocess.run(branch_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result["branch"] = branch_name

        # Additive backup of original source file before applying patch (if existing)
        commit_files = [resolved_target]
        backup_file = None
        if original_code:
            now_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_file = resolved_target.parent / f"{resolved_target.name}.bak_{now_ts}"
            backup_file.write_text(original_code, encoding="utf-8")
            result["backup_file"] = str(backup_file)
            commit_files.append(backup_file)

        # Write refactored code to target
        resolved_target.write_text(extracted_code + "\n", encoding="utf-8")
        result["applied"] = True

        # Commit additively
        action_verb = "apply" if original_code else "create"
        commit_msg = f"{action_verb}: {resolved_target.name} from {task_id}"
        commit_out = GitSafetyGuard.commit_additive(
            repo_root=self.repo_root,
            file_paths=commit_files,
            commit_message=commit_msg,
        )
        result["commit"] = commit_out

        # Record in forensic audit journal
        self.forensics.log_event(
            phase="PATCH_APPLIED",
            task_id=task_id,
            action="apply",
            details={
                "target": str(resolved_target),
                "orig_sha256": orig_sha,
                "new_sha256": new_sha,
                "backup_file": str(backup_file) if backup_file else None,
                "work_product": str(wp_path),
            },
        )

        return result


# end of file: plowinf/src/plowinf/patcher.py
