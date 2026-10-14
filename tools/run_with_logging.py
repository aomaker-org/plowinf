# Path: plowinf/tools/run_with_logging.py
"""Universal additive I/O logger and execution wrapper complying with Directive 02."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Optional, Tuple


def utc_now_iso() -> str:
    """Return ISO 8601 UTC timestamp with microsecond resolution."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def resolve_node_name() -> str:
    """Resolve node identity from .node_identity or hostname."""
    home_node = Path.home() / ".node_identity"
    if home_node.is_file():
        try:
            name = home_node.read_text(encoding="utf-8").strip()
            if name:
                return name
        except OSError:
            pass
    return os.environ.get("NODE_NAME", os.uname().nodename)


def run_command_with_logging(
    command: str,
    cwd: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    label: str = "task",
    node_id: Optional[str] = None,
    timeout: Optional[int] = None,
    env: Optional[dict] = None,
) -> Tuple[int, str, Path]:
    """Execute command, stream output live, and log additively with ISO timestamps."""
    if node_id is None:
        node_id = resolve_node_name()

    if log_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        log_dir = repo_root / "logs" / "tasks"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
    log_file = log_dir / f"{ts_slug}_{node_id}_{clean_label}.log"
    ledger_file = log_dir / "audit_ledger.jsonl"

    start_iso = utc_now_iso()
    collected_output = []
    output_lock = threading.Lock()

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"[{start_iso}] [META] node={node_id} label={label}\n")
        lf.write(f"[{start_iso}] [META] command={command}\n")
        lf.write(f"[{start_iso}] [META] cwd={cwd or Path.cwd()}\n")
        lf.flush()

        def stream_reader(pipe, stream_tag: str, console_target):
            try:
                for line in iter(pipe.readline, ""):
                    if not line:
                        break
                    ts = utc_now_iso()
                    formatted = f"[{ts}] [{stream_tag}] {line}"
                    lf.write(formatted)
                    lf.flush()
                    with output_lock:
                        collected_output.append(line)
                    console_target.write(line)
                    console_target.flush()
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=merged_env,
        )

        t_out = threading.Thread(
            target=stream_reader,
            args=(proc.stdout, "STDOUT", sys.stdout),
            daemon=True,
        )
        t_err = threading.Thread(
            target=stream_reader,
            args=(proc.stderr, "STDERR", sys.stderr),
            daemon=True,
        )

        t_out.start()
        t_err.start()

        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_ts = utc_now_iso()
            lf.write(f"[{kill_ts}] [META] Timeout expired ({timeout}s). Terminating PID {proc.pid}\n")
            lf.flush()
            proc.kill()
            exit_code = proc.wait()

        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)

        end_iso = utc_now_iso()
        lf.write(f"[{end_iso}] [META] finished with exit_code={exit_code}\n")
        lf.flush()

    audit_entry = {
        "timestamp": end_iso,
        "node_id": node_id,
        "label": label,
        "command": command,
        "cwd": str(cwd or Path.cwd()),
        "exit_code": exit_code,
        "log_path": str(log_file),
    }
    with open(ledger_file, "a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(audit_entry) + "\n")

    full_output_str = "".join(collected_output)
    return exit_code, full_output_str, log_file


def main():
    parser = argparse.ArgumentParser(
        description="Universal Additive I/O Logger (Directive 02 compliant)"
    )
    parser.add_argument("--cmd", required=True, help="Shell command to execute")
    parser.add_argument("--label", default="task", help="Task label for filename and audit ledger")
    parser.add_argument("--log-dir", help="Destination directory for logs (default: logs/tasks)")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds")
    parser.add_argument("--cwd", help="Working directory for command")
    args = parser.parse_args()

    log_path_arg = Path(args.log_dir) if args.log_dir else None
    cwd_path_arg = Path(args.cwd) if args.cwd else None

    exit_code, _, log_file = run_command_with_logging(
        command=args.cmd,
        cwd=cwd_path_arg,
        log_dir=log_path_arg,
        label=args.label,
        timeout=args.timeout,
    )
    print(f"\n[+] Task completed with exit code {exit_code}. Log saved to: {log_file}", file=sys.stderr)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

# end of file: plowinf/tools/run_with_logging.py
