# Path: plowinf/src/plowinf/cli.py
"""Command-line interface for plowinf.

Provides interactive, queued, forensic auditing, and inference benchmarking subcommands:
  - review    : Execute or queue code review
  - test      : Generate unit tests for code
  - docstring : Generate docstrings and type annotations
  - prompt    : Query local model with arbitrary text
  - benchmark : Deep dive performance benchmark (latency, tokens, RSS)
  - enqueue   : Add asynchronous task to file queue
  - daemon    : Run background worker polling loop with verbose logging
  - status    : Display queue and backend health audit
  - backends  : Probe and list available local SLM engines
  - forensics : Inspect cryptographic ledger and forensic execution traces
  - view      : View generated work products and inference logs
  - apply     : Apply refactored code to target source file
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional

from .backends import (
    BackendRouter,
    Llama2cBackend,
    LlamaCppCliBackend,
    MockBackend,
    OllamaBackend,
    OpenAICompatBackend,
    SLMBackend,
)
from .forensics import ForensicAuditLogger, get_telemetry_snapshot
from .patcher import CodePatcher
from .queue import FileTaskQueue
from .worker import CodeWorker


def build_backend(name: Optional[str], model: Optional[str] = None) -> SLMBackend:
    """Instantiate requested or routed backend adapter."""
    if name == "ollama":
        kwargs = {}
        if model:
            kwargs["model"] = model
        return OllamaBackend(**kwargs)
    elif name == "openai_compat":
        kwargs = {}
        if model:
            kwargs["model"] = model
        return OpenAICompatBackend(**kwargs)
    elif name == "llama_cpp":
        kwargs = {}
        if model:
            kwargs["model_path"] = model
        return LlamaCppCliBackend(**kwargs)
    elif name == "llama2c":
        kwargs = {}
        if model:
            kwargs["model_bin"] = model
        return Llama2cBackend(**kwargs)
    elif name == "mock":
        return MockBackend()
    else:
        router = BackendRouter()
        return router.get_active_backend()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plowinf",
        description="plowinf - Offline AI inference engine, SLM code worker, and queue processor",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose forensic diagnostic logging")
    subparsers = parser.add_subparsers(dest="command", help="plowinf subcommands")

    backend_choices = ["ollama", "openai_compat", "llama_cpp", "llama2c", "mock", "auto"]

    # Command: review
    p_review = subparsers.add_parser("review", help="Review code file for bugs and quality")
    p_review.add_argument("file", help="Path to code file")
    p_review.add_argument("--backend", choices=backend_choices, default="auto")
    p_review.add_argument("--model", help="Specific model tag, GGUF file path, or bin path")
    p_review.add_argument("--prompt", default="", help="Additional prompt instructions")

    # Command: test
    p_test = subparsers.add_parser("test", help="Generate unit tests for code file")
    p_test.add_argument("file", help="Path to code file")
    p_test.add_argument("--backend", choices=backend_choices, default="auto")
    p_test.add_argument("--model", help="Specific model tag, GGUF file path, or bin path")
    p_test.add_argument("--prompt", default="", help="Additional test requirements")

    # Command: docstring
    p_doc = subparsers.add_parser("docstring", help="Generate docstrings and typing")
    p_doc.add_argument("file", help="Path to code file")
    p_doc.add_argument("--backend", choices=backend_choices, default="auto")
    p_doc.add_argument("--model", help="Specific model tag, GGUF file path, or bin path")

    # Command: prompt
    p_prompt = subparsers.add_parser("prompt", help="Execute freeform prompt on local model")
    p_prompt.add_argument("text", help="Prompt text")
    p_prompt.add_argument("--code", help="Optional code file for context")
    p_prompt.add_argument("--backend", choices=backend_choices, default="auto")
    p_prompt.add_argument("--model", help="Specific model tag, GGUF file path, or bin path")

    # Command: benchmark (deep dive into inference speed, latency, RSS)
    p_bench = subparsers.add_parser("benchmark", help="Benchmark local inference speed, latency, and RSS")
    p_bench.add_argument("--backend", choices=backend_choices, default="auto")
    p_bench.add_argument("--model", help="Specific model tag or path")
    p_bench.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations")
    p_bench.add_argument(
        "--prompt",
        default="Write a quicksort function in Python with type hints.",
        help="Benchmark prompt",
    )

    # Command: enqueue
    p_enq = subparsers.add_parser("enqueue", help="Add asynchronous task to file queue")
    p_enq.add_argument("action", choices=["review", "test", "docstring", "prompt"])
    p_enq.add_argument("--file", default="", help="Optional code file path")
    p_enq.add_argument("--prompt", default="", help="Prompt or task instructions")

    # Command: daemon
    p_daemon = subparsers.add_parser("daemon", help="Run background queue processing daemon")
    p_daemon.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    p_daemon.add_argument("--once", action="store_true", help="Process at most one cycle and exit")

    # Command: status
    subparsers.add_parser("status", help="Show file queue summary and backend status")

    # Command: backends
    subparsers.add_parser("backends", help="Audit local SLM backend availability")

    # Command: forensics
    p_forensics = subparsers.add_parser("forensics", help="Audit forensic ledger and trace records")
    p_forensics.add_argument("--task", help="Filter forensic records by task ID")
    p_forensics.add_argument("--limit", type=int, default=10, help="Maximum records to display")
    p_forensics.add_argument("--summary", action="store_true", help="Print summary audit of all executions")

    # Command: view
    p_view = subparsers.add_parser("view", help="View generated files and work products")
    p_view.add_argument("--json", action="store_true", help="Output raw JSON stream (pipeable into jq)")
    p_view.add_argument("--latest", action="store_true", help="Display latest completed work product")
    p_view.add_argument("--task", help="Display details for specific task ID")

    # Command: apply
    p_apply = subparsers.add_parser("apply", help="Apply refactored code product to target source file")
    p_apply.add_argument("product", help="Work product file path or task ID")
    p_apply.add_argument("--target", help="Explicit target file path (auto-detected if omitted)")
    p_apply.add_argument("--dry-run", action="store_true", help="Preview unified diff without modifying files")
    p_apply.add_argument("--branch", action="store_true", help="Checkout dedicated refactor branch before applying")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    queue = FileTaskQueue()
    forensics = ForensicAuditLogger()

    if args.command in ["review", "test", "docstring", "prompt"]:
        code_file = getattr(args, "file", "") or getattr(args, "code", "")
        code_content = ""
        if code_file:
            p = Path(code_file).resolve()
            if not p.exists():
                print(f"[!] Error: File '{code_file}' not found.", file=sys.stderr)
                sys.exit(1)
            code_content = p.read_text(encoding="utf-8")

        prompt_str = getattr(args, "prompt", "") or getattr(args, "text", "")
        backend = build_backend(getattr(args, "backend", "auto"), getattr(args, "model", None))
        worker = CodeWorker(queue=queue, router=BackendRouter(backends=[backend]), forensics=forensics)

        prompt = worker.build_prompt(args.command, code_content, prompt_str)
        print(f"[*] Dispatching to backend: {backend.name}")
        start_time = time.time()
        try:
            output = backend.generate(prompt=prompt)
            duration_ms = (time.time() - start_time) * 1000.0
            print("\n" + "=" * 60)
            print(output)
            print("=" * 60)

            # Forensic audit registration
            task_fake_id = f"cli-{int(time.time())}"
            wp_path = worker._save_work_product(
                queue.enqueue(action=args.command, code_path=code_file, prompt=prompt_str),
                code_file,
                output,
            ) if code_file else None

            forensics.record_inference_audit(
                task_id=task_fake_id,
                action=args.command,
                code_path=code_file,
                code_content=code_content,
                full_prompt=prompt,
                response=output,
                backend_name=backend.name,
                duration_ms=duration_ms,
                status="SUCCESS",
                work_product_path=wp_path,
            )
            print(f"[*] [Forensic] Audit recorded: {forensics.journal_path}")
        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000.0
            forensics.record_inference_audit(
                task_id=f"cli-fail-{int(time.time())}",
                action=args.command,
                code_path=code_file,
                code_content=code_content,
                full_prompt=prompt,
                response="",
                backend_name=backend.name,
                duration_ms=duration_ms,
                status="FAILED",
                error=str(exc),
            )
            print(f"[!] Generation failed: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "benchmark":
        backend = build_backend(getattr(args, "backend", "auto"), getattr(args, "model", None))
        print("=" * 65)
        print(f"  plowinf - AI Inference Benchmark: {backend.name}")
        print("=" * 65)
        print(f"  Prompt : {args.prompt}")
        print(f"  Runs   : {args.runs}")
        durations = []
        token_estimates = []
        for i in range(1, args.runs + 1):
            t0 = time.time()
            res = backend.generate(prompt=args.prompt)
            dt = time.time() - t0
            durations.append(dt)
            # Estimate tokens: ~4 chars per token rule of thumb
            approx_tokens = max(1, len(res) // 4)
            token_estimates.append(approx_tokens)
            speed = approx_tokens / dt if dt > 0 else 0
            rss = get_telemetry_snapshot().get("rss_mb", 0)
            print(f"  Run #{i}: {dt*1000:.1f} ms | ~{approx_tokens} tokens | ~{speed:.1f} tok/s | RSS: {rss} MB")

        avg_dt = sum(durations) / len(durations)
        total_tokens = sum(token_estimates)
        avg_speed = total_tokens / sum(durations) if sum(durations) > 0 else 0
        print("-" * 65)
        print(f"  Average Latency : {avg_dt*1000:.1f} ms")
        print(f"  Estimated Speed : ~{avg_speed:.1f} tok/s")
        print("=" * 65)

    elif args.command == "enqueue":
        task = queue.enqueue(
            action=args.action,
            code_path=args.file,
            prompt=args.prompt,
        )
        forensics.log_event("TASK_ENQUEUED", task.task_id, args.action, {"file": args.file, "prompt": args.prompt})
        print(f"[+] Task enqueued successfully: {task.task_id} (status: {task.status})")

    elif args.command == "daemon":
        worker = CodeWorker(queue=queue, forensics=forensics)
        worker.run_daemon(interval=args.interval, run_once=args.once, verbose=args.verbose)

    elif args.command == "status":
        summary = queue.get_status_summary()
        print("=" * 50)
        print("  plowinf - File Queue Status")
        print("=" * 50)
        for k, v in summary.items():
            print(f"  {k:<12}: {v}")
        print("=" * 50)

        router = BackendRouter()
        active = router.get_active_backend()
        print(f"  Active Backend : {active.name}")
        print(f"  Forensic Logs  : {forensics.journal_path}")
        print("=" * 50)

    elif args.command == "backends":
        backends = [
            OllamaBackend(),
            OpenAICompatBackend(),
            LlamaCppCliBackend(),
            Llama2cBackend(),
            MockBackend(),
        ]
        print("=" * 60)
        print("  plowinf - SLM Backend Availability Audit")
        print("=" * 60)
        for b in backends:
            status = "AVAILABLE" if b.is_available() else "UNAVAILABLE"
            models = b.list_models()
            model_info = f" (models: {', '.join(models)})" if models else ""
            print(f"  {b.name:<18}: {status}{model_info}")
        print("=" * 60)

    elif args.command == "forensics":
        history = forensics.get_task_history(task_id=args.task)
        if not history:
            print("No forensic audit records found.")
            return

        print("=" * 70)
        print(f"  plowinf - Forensic Audit Ledger ({len(history)} total records)")
        print("=" * 70)
        for entry in history[-args.limit:]:
            ts = entry.get("timestamp_utc", "UNKNOWN")
            tid = entry.get("task_id", "N/A")
            act = entry.get("action", "N/A")
            st = entry.get("status", entry.get("phase", "N/A"))
            bk = entry.get("backend", "N/A")
            dur = entry.get("duration_ms", "N/A")
            target = entry.get("target", {}).get("path", "")
            target_sha = entry.get("target", {}).get("sha256", "")[:12]
            resp_sha = entry.get("response", {}).get("sha256", "")[:12]

            print(f"  [{ts}] {tid:<16} {act:<10} status={st:<8} backend={bk:<8} dur={dur}ms")
            if target:
                print(f"    Target: {target} (SHA256: {target_sha}...)")
            if resp_sha:
                print(f"    Output SHA256: {resp_sha}...")
        print("=" * 70)

    elif args.command == "view":
        history = [e for e in forensics.get_task_history(task_id=args.task) if e.get("phase") == "INFERENCE_EXECUTION"]
        if args.json:
            for entry in history:
                print(json.dumps(entry))
            return

        if args.latest:
            if not history:
                print("No completed work products found.")
                return
            latest = history[-1]
            wp_path = latest.get("work_product_path")
            print(f"=== Latest Work Product: {latest.get('task_id')} ({latest.get('action')}) ===")
            print(f"  Target File : {latest.get('target', {}).get('path')}")
            print(f"  Artifact    : {wp_path}")
            print(f"  Duration    : {latest.get('duration_ms', 0)/1000.0:.2f}s")
            print("=" * 60)
            if wp_path and Path(wp_path).exists():
                print(Path(wp_path).read_text(encoding="utf-8"))
            return

        print("=" * 75)
        print("  plowinf - Generated Work Products On Disk")
        print("=" * 75)
        wp_dir = Path("workspace/work_products")
        if wp_dir.exists():
            for f in sorted(wp_dir.glob("**/*.md")):
                sz = f"{f.stat().st_size / 1024.0:.1f} KB"
                lines = len(f.read_text(encoding="utf-8").splitlines())
                print(f"  {sz:<9} ({lines:>4} lines)  {f}")
        print("=" * 75)

    elif args.command == "apply":
        patcher = CodePatcher(forensics=forensics)
        try:
            res = patcher.apply_patch(
                work_product_file=args.product,
                target_file=args.target,
                dry_run=args.dry_run,
                create_branch=args.branch,
            )
            mode_label = "Dry-Run Diff Preview" if args.dry_run else "Patch Applied"
            print("=" * 70)
            print(f"  plowinf - {mode_label}")
            print("=" * 70)
            print(f"  Task ID      : {res['task_id']}")
            print(f"  Target File  : {res['target']}")
            print(f"  Original SHA : {res['orig_sha256']}")
            print(f"  Refactor SHA : {res['new_sha256']}")
            if res.get("branch"):
                print(f"  Branch       : {res['branch']}")
            if res.get("backup_file"):
                print(f"  Backup File  : {res['backup_file']}")
            if res.get("commit"):
                print(f"  Git Commit   : {res['commit'].strip()}")
            print("-" * 70)
            print("Unified Diff:")
            print(res["diff"] or "  (No syntactic changes detected)")
            print("=" * 70)
        except Exception as exc:
            print(f"[!] Apply failed: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

# end of file: plowinf/src/plowinf/cli.py
