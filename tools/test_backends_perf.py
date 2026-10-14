# Path: plowinf/tools/test_backends_perf.py
"""Performance measurement and comparative benchmark suite for plowinf backends."""

import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Any

# Ensure src is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from plowinf.backends import (
    Llama2cBackend,
    LlamaCppCliBackend,
    MockBackend,
    OllamaBackend,
    OpenAICompatBackend,
    SLMBackend,
)
from plowinf.forensics import get_telemetry_snapshot


def benchmark_backend(backend: SLMBackend, prompt: str, runs: int = 3, max_tokens: int = 64) -> Dict[str, Any]:
    print(f"\n[*] Benchmarking backend: {backend.name}...")
    if not backend.is_available():
        print(f"    [!] Backend {backend.name} is UNAVAILABLE.")
        return {"name": backend.name, "available": False}

    durations: List[float] = []
    generated_outputs: List[str] = []

    for r in range(1, runs + 1):
        t0 = time.time()
        try:
            out = backend.generate(prompt=prompt, max_tokens=max_tokens)
            dt = time.time() - t0
            durations.append(dt)
            generated_outputs.append(out)
            rss = get_telemetry_snapshot().get("rss_mb", 0)
            approx_tokens = max(1, len(out.split()))
            speed = approx_tokens / dt if dt > 0 else 0
            ms = dt * 1000.0
            print(
                f"    Pass #{r}: {ms:7.1f} ms | ~{approx_tokens:3d} words | "
                f"~{speed:5.1f} words/s | RSS: {rss} MB"
            )
        except Exception as exc:
            print(f"    [!] Pass #{r} failed: {exc}")
            return {"name": backend.name, "available": True, "error": str(exc)}

    avg_dt = sum(durations) / len(durations)
    total_words = sum(len(o.split()) for o in generated_outputs)
    avg_words = total_words / len(generated_outputs)
    avg_speed = avg_words / avg_dt if avg_dt > 0 else 0
    sample_preview = generated_outputs[0].replace("\n", " ")[:120]

    return {
        "name": backend.name,
        "available": True,
        "avg_duration_s": avg_dt,
        "avg_duration_ms": avg_dt * 1000.0,
        "avg_words": avg_words,
        "avg_speed": avg_speed,
        "preview": sample_preview,
        "durations_ms": [round(d * 1000.0, 1) for d in durations],
    }


def main():
    test_prompt = "Write a one-line Python function to compute the factorial of n."
    print("=" * 80)
    print("  PLOWINF - MULTI-BACKEND INFERENCE & PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"  Prompt : \"{test_prompt}\"")
    print(f"  Runs   : 3 per backend")

    backends: List[SLMBackend] = [
        MockBackend(),
        Llama2cBackend(),
        OllamaBackend(model="qwen2.5-coder:1.5b"),
        OpenAICompatBackend(model="qwen2.5-coder:1.5b"),
        LlamaCppCliBackend(threads=4),
    ]

    results = []
    for b in backends:
        res = benchmark_backend(b, prompt=test_prompt, runs=3, max_tokens=64)
        results.append(res)

    print("\n" + "=" * 80)
    print("  COMPARATIVE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"  {'Backend':<16} | {'Status':<11} | {'Avg Latency':<12} | {'Speed (w/s)':<12} | {'Passes (ms)'}")
    print("  " + "-" * 76)

    for r in results:
        b_name = r["name"]
        if not r.get("available"):
            print(f"  {b_name:<16} | {'UNAVAILABLE':<11} | {'N/A':<12} | {'N/A':<12} | N/A")
        elif "error" in r:
            print(f"  {b_name:<16} | {'ERROR':<11} | {'N/A':<12} | {'N/A':<12} | {r['error'][:30]}")
        else:
            lat_str = f"{r['avg_duration_ms']:6.1f} ms"
            spd_str = f"{r['avg_speed']:6.1f} w/s"
            passes_str = ", ".join(str(x) for x in r["durations_ms"])
            print(f"  {b_name:<16} | {'ACTIVE':<11} | {lat_str:<12} | {spd_str:<12} | [{passes_str}]")

    print("\n" + "=" * 80)
    print("  SAMPLE GENERATED OUTPUTS")
    print("=" * 80)
    for r in results:
        if r.get("available") and "preview" in r:
            print(f"  [{r['name']}]")
            print(f"    {r['preview']}...\n")
    print("=" * 80)


if __name__ == "__main__":
    main()

# end of file: plowinf/tools/test_backends_perf.py
