# Path: plowinf/tools/audit_20l_surgery.py
"""Forensic surgery auditor & A/B code validity benchmark for 20L vs 28L Qwen.

1. Inspects the GGUF norm weights inside Ollama's qwen2.5-1.5b-20L:latest blob
   to diagnose how the 20-layer chop was performed.
2. Runs a deterministic (temperature=0.0) 3-task coding benchmark against
   127.0.0.1:11434 comparing qwen2:1.5b (28L) and qwen2.5-1.5b-20L:latest (20L)
   on AST syntax validity, structural completeness, and Haswell tok/s.

Run via uv (Directive 01):
    uv run python tools/audit_20l_surgery.py
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import time
from typing import Any, BinaryIO

import requests

GGUF_MAGIC = 0x46554747
OLLAMA_GEN_URL = "http://127.0.0.1:11434/api/generate"

BENCHMARK_TASKS: list[dict[str, str]] = [
    {
        "id": "python_codegen",
        "kind": "ast",
        "prompt": (
            "Write ONLY a valid Python function named `fibonacci_iter(n: int) "
            "-> list[int]` that returns the first n Fibonacci numbers. Do not "
            "include markdown prose outside the code block."
        ),
    },
    {
        "id": "code_review_json",
        "kind": "json",
        "prompt": (
            "Review this Python code: `def div(a, b): return a / b`. "
            "Respond ONLY with a valid JSON object containing keys "
            "`\"bug\"` (string) and `\"severity\"` (string)."
        ),
    },
    {
        "id": "pytest_generation",
        "kind": "ast",
        "prompt": (
            "Write ONLY a valid Python pytest function named `test_add()` "
            "that asserts `1 + 1 == 2` and `-1 + 1 == 0`. Output only Python "
            "code."
        ),
    },
]


def utc_now_iso() -> str:
    """Return ISO-8601 UTC microsecond timestamp per Directive 02."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_u32(fp: BinaryIO) -> int:
    raw = fp.read(4)
    if len(raw) != 4:
        raise EOFError("Unexpected EOF reading uint32")
    return int.from_bytes(raw, "little", signed=False)


def _read_u64(fp: BinaryIO) -> int:
    raw = fp.read(8)
    if len(raw) != 8:
        raise EOFError("Unexpected EOF reading uint64")
    return int.from_bytes(raw, "little", signed=False)


def _read_str(fp: BinaryIO) -> str:
    length = _read_u64(fp)
    return fp.read(length).decode("utf-8", errors="replace")


def _skip_kv_value(fp: BinaryIO, val_type: int) -> None:
    fixed_sizes = {
        0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
        6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
    }
    if val_type in fixed_sizes:
        fp.seek(fixed_sizes[val_type], 1)
    elif val_type == 8:
        length = _read_u64(fp)
        fp.seek(length, 1)
    elif val_type == 9:
        elem_type = _read_u32(fp)
        count = _read_u64(fp)
        if elem_type in fixed_sizes:
            fp.seek(fixed_sizes[elem_type] * count, 1)
        else:
            for _ in range(count):
                _skip_kv_value(fp, elem_type)
    else:
        raise ValueError(f"Unknown GGUF metadata type: {val_type}")


def resolve_ollama_blob(model_ref: str) -> Path | None:
    """Locate the local GGUF blob for an Ollama model tag if readable."""
    name_part, tag = (
        model_ref.split(":", 1) if ":" in model_ref else (model_ref, "latest")
    )
    rel = (
        Path("registry.ollama.ai") / "library" / name_part / tag
        if "/" not in name_part
        else Path("registry.ollama.ai") / name_part / tag
    )
    roots = [
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
        Path("/var/lib/ollama/.ollama/models"),
    ]
    for root in roots:
        m_path = root / "manifests" / rel
        if m_path.is_file():
            data = json.loads(m_path.read_text(encoding="utf-8"))
            for layer in data.get("layers", []):
                if "ollama.image.model" in layer.get("mediaType", ""):
                    digest = layer["digest"].replace(":", "-")
                    blob = root / "blobs" / digest
                    if blob.is_file():
                        return blob
    return None


def fingerprint_norm_tensors(gguf_path: Path) -> dict[str, tuple[str, float]]:
    """Return {tensor_name: (sha256_prefix, rms_norm_value)} for FP32 norms."""
    results: dict[str, tuple[str, float]] = {}
    with gguf_path.open("rb") as fp:
        magic = _read_u32(fp)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {gguf_path}")
        _version = _read_u32(fp)
        n_tensors = _read_u64(fp)
        n_kv = _read_u64(fp)

        for _ in range(n_kv):
            _ = _read_str(fp)
            vtype = _read_u32(fp)
            _skip_kv_value(fp, vtype)

        tensor_infos: list[tuple[str, int, int, int]] = []
        for _ in range(n_tensors):
            t_name = _read_str(fp)
            n_dims = _read_u32(fp)
            elems = 1
            for _ in range(n_dims):
                elems *= _read_u64(fp)
            ggml_type = _read_u32(fp)
            offset = _read_u64(fp)
            tensor_infos.append((t_name, elems, ggml_type, offset))

        # GGUF data section is aligned to 32 bytes by default
        pos = fp.tell()
        alignment = 32
        data_start = (pos + alignment - 1) // alignment * alignment

        for t_name, elems, ggml_type, offset in tensor_infos:
            if ggml_type == 0 and t_name.endswith("_norm.weight"):
                fp.seek(data_start + offset)
                raw = fp.read(elems * 4)
                digest = hashlib.sha256(raw).hexdigest()[:12]
                floats = struct.unpack(f"={elems}f", raw)
                rms = math.sqrt(sum(x * x for x in floats) / max(elems, 1))
                results[t_name] = (digest, rms)
    return results


def audit_layer_mapping(model_28: str, model_20: str) -> None:
    """Compare FP32 norm tensor hashes between 28L and 20L models."""
    blob_28 = resolve_ollama_blob(model_28)
    blob_20 = resolve_ollama_blob(model_20)
    if not blob_28 or not blob_20:
        print(
            f"[{utc_now_iso()}] Skipping binary tensor hash diff "
            "(Ollama blob directory not directly readable by current user)."
        )
        return

    norms_28 = fingerprint_norm_tensors(blob_28)
    norms_20 = fingerprint_norm_tensors(blob_20)

    hash_to_28_layer: dict[str, str] = {}
    for t_name, (h_val, _rms) in norms_28.items():
        if t_name.startswith("blk.") and t_name.endswith(".attn_norm.weight"):
            hash_to_28_layer[h_val] = t_name.split(".")[1]

    print("=" * 78)
    print(f"FORENSIC NORM PROFILE: {model_20} ({blob_20.name[:19]}...)")
    print("=" * 78)
    print(
        f"{'20L Layer':<10} | {'attn_norm Hash':<14} | {'attn_norm RMS':>13} | "
        f"{'ffn_norm RMS':>12} | {'Exact Match in qwen2:1.5b'}"
    )
    print("-" * 78)
    for idx in range(20):
        a_key = f"blk.{idx}.attn_norm.weight"
        f_key = f"blk.{idx}.ffn_norm.weight"
        if a_key not in norms_20:
            continue
        a_hash, a_rms = norms_20[a_key]
        _f_hash, f_rms = norms_20.get(f_key, ("n/a", 0.0))
        matched_src = hash_to_28_layer.get(a_hash, "diff base (Qwen2.5 vs 2.0)")
        print(
            f"blk.{idx:<6} | {a_hash:<14} | {a_rms:>13.5f} | "
            f"{f_rms:>12.5f} | {matched_src}"
        )
    out_hash, out_rms = norms_20.get("output_norm.weight", ("n/a", 0.0))
    print("-" * 78)
    print(f"output_norm.weight (20L) : hash={out_hash} | RMS={out_rms:.5f}")
    print("=" * 78)


def extract_code_or_json(text: str) -> str:
    """Strip markdown code fences if present."""
    fenced = re.findall(r"```(?:python|json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        return fenced[0].strip()
    return text.strip()


def validate_output(kind: str, raw_text: str) -> tuple[bool, str]:
    """Validate whether model output is syntactically valid AST or JSON."""
    cleaned = extract_code_or_json(raw_text)
    if not cleaned:
        return False, "Empty response"
    if kind == "ast":
        try:
            tree = ast.parse(cleaned)
            has_func = any(
                isinstance(node, ast.FunctionDef) for node in ast.walk(tree)
            )
            return (
                (True, "Valid Python AST + FunctionDef")
                if has_func
                else (False, "Parsed AST but missing FunctionDef")
            )
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    if kind == "json":
        try:
            obj = json.loads(cleaned)
            ok = isinstance(obj, dict) and "bug" in obj and "severity" in obj
            return (
                (True, "Valid JSON schema")
                if ok
                else (False, "JSON missing required keys")
            )
        except json.JSONDecodeError as exc:
            return False, f"JSONDecodeError: {exc.msg}"
    return False, "Unknown check kind"


def run_ab_benchmark(models: list[str]) -> None:
    """Run deterministic coding prompts against local Ollama models."""
    print(f"\n[{utc_now_iso()}] Starting A/B Inference Triage on 127.0.0.1:11434")
    print("=" * 78)
    summary: dict[str, dict[str, Any]] = {}

    for model in models:
        passed = 0
        tps_list: list[float] = []
        model_missing = False
        print(f"\n>>> Testing Model: {model}")
        print("-" * 78)
        for task in BENCHMARK_TASKS:
            payload = {
                "model": model,
                "prompt": task["prompt"],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 160,
                    "seed": 42,
                },
            }
            t0 = time.perf_counter()
            resp = requests.post(
                OLLAMA_GEN_URL,
                json=payload,
                timeout=60,
                allow_redirects=False,
            )
            elapsed = time.perf_counter() - t0
            try:
                resp.raise_for_status()
            except requests.HTTPError as err:
                if err.response is not None and err.response.status_code == 404:
                    print(
                        f"  [PURGED] Model '{model}' not found in Ollama daemon "
                        "(cleanly purged)."
                    )
                    model_missing = True
                    break
                raise
            data = resp.json()

            out_text = data.get("response", "")
            eval_count = data.get("eval_count", 0)
            eval_ns = data.get("eval_duration", 1)
            tok_s = (
                (eval_count / (eval_ns / 1e9))
                if eval_ns > 0
                else (eval_count / max(elapsed, 0.001))
            )
            tps_list.append(tok_s)

            ok, detail = validate_output(task["kind"], out_text)
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            print(
                f"  [{status}] {task['id']:<18} | {tok_s:>6.2f} tok/s | "
                f"{elapsed:>5.2f}s | {detail}"
            )
        if model_missing:
            summary[model] = {
                "passed": 0,
                "total": len(BENCHMARK_TASKS),
                "avg_tok_s": 0.0,
                "purged": True,
            }
            continue
        avg_tps = sum(tps_list) / max(len(tps_list), 1)
        summary[model] = {
            "passed": passed,
            "total": len(BENCHMARK_TASKS),
            "avg_tok_s": avg_tps,
            "purged": False,
        }

    print("\n" + "=" * 78)
    print("FINAL A/B TRIAGE VERDICT")
    print("=" * 78)
    for model, stats in summary.items():
        p, tot = stats["passed"], stats["total"]
        tps = stats["avg_tok_s"]
        if stats.get("purged"):
            verdict = "PURGED (Cleanly eliminated)"
        else:
            verdict = (
                "HEALTHY (Keep)"
                if p == tot
                else (
                    "DEGRADED (Unhealed chop artifacts)"
                    if p > 0
                    else "BROKEN (Purge via `ollama rm`)"
                )
            )
        print(
            f"  {model:<28} : {p}/{tot} tasks passed | "
            f"{tps:>6.2f} tok/s | {verdict}"
        )
    print("=" * 78)


def main() -> int:
    models = ["qwen2.5-coder:1.5b", "qwen2:1.5b", "qwen2.5-1.5b-20L:latest"]
    try:
        audit_layer_mapping("qwen2:1.5b", "qwen2.5-1.5b-20L:latest")
        run_ab_benchmark(models)
        return 0
    except requests.RequestException as exc:
        print(
            f"[{utc_now_iso()}] Ollama HTTP error: {exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"[{utc_now_iso()}] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# end of file: plowinf/tools/audit_20l_surgery.py
