# Path: plowinf/tools/qwen_distill_moe_lab.py
"""Qwen2.5 Layer Profiler, Distillation Planner, and MoE Surgical Toolkit.

Run via uv (Directive 01):
    uv run python tools/qwen_distill_moe_lab.py --mode plan-moe --layers 28
    uv run python tools/qwen_distill_moe_lab.py --mode plan-moe --layers 20
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


def utc_now_iso() -> str:
    """Return ISO-8601 UTC microsecond timestamp per Directive 02."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")


def calculate_qwen_moe_variants(
    num_layers: int = 28,
    d_model: int = 1536,
    d_ff_dense: int = 8960,
    vocab_size: int = 151936,
) -> list[dict[str, Any]]:
    """Compute parameter, Q4 RAM, and Haswell tok/s estimates for 3 variants."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    emb_m = (vocab_size * d_model) / 1e6
    attn_per_l_m = (
        (d_model * 1536 + 1536)
        + 2 * (d_model * 256 + 256)
        + (1536 * d_model)
    ) / 1e6
    ffn_dense_per_l_m = (3 * d_model * d_ff_dense) / 1e6

    dense_total_m = emb_m + num_layers * (attn_per_l_m + ffn_dense_per_l_m)

    up_e4_total_m = emb_m + num_layers * (
        attn_per_l_m + 4 * ffn_dense_per_l_m + (d_model * 4) / 1e6
    )
    up_e4_act_k1_m = emb_m + num_layers * (
        attn_per_l_m + 1 * ffn_dense_per_l_m + (d_model * 4) / 1e6
    )

    d_exp = d_ff_dense // 5  # 1792
    ffn_sub_per_exp_m = (3 * d_model * d_exp) / 1e6
    sub_total_m = emb_m + num_layers * (
        attn_per_l_m + 5 * ffn_sub_per_exp_m + (d_model * 4) / 1e6
    )
    sub_act_m = emb_m + num_layers * (
        attn_per_l_m + 2 * ffn_sub_per_exp_m + (d_model * 4) / 1e6
    )

    q4_mb_per_m = 0.576
    bw_gbs = 14.5

    raw_specs = [
        (f"1. Dense Qwen2.5 ({num_layers}L)", dense_total_m, dense_total_m),
        (
            f"2. Upcycled 4xMoE ({num_layers}L, Top-1)",
            up_e4_total_m,
            up_e4_act_k1_m,
        ),
        (
            f"3. Sub-FFN 1S+4R MoE ({num_layers}L, Top-1)",
            sub_total_m,
            sub_act_m,
        ),
    ]

    results: list[dict[str, Any]] = []
    for name, tot_m, act_m in raw_specs:
        ram_mb = tot_m * q4_mb_per_m
        act_mb = act_m * q4_mb_per_m
        est_tps = (bw_gbs * 1024.0) / (act_mb + 55.0)
        results.append(
            {
                "name": name,
                "num_layers": num_layers,
                "total_params_m": tot_m,
                "active_params_m": act_m,
                "q4_ram_mb": ram_mb,
                "est_tok_per_s": est_tps,
            }
        )
    return results


def compute_qwen_moe_budget(
    num_layers: int = 28,
    d_model: int = 1536,
    d_ff_dense: int = 8960,
    vocab_size: int = 151936,
) -> list[dict[str, Any]]:
    """Print comparative memory and active-parameter table for fg26."""
    variants = calculate_qwen_moe_variants(
        num_layers=num_layers,
        d_model=d_model,
        d_ff_dense=d_ff_dense,
        vocab_size=vocab_size,
    )
    print(f"[{utc_now_iso()}] Qwen2.5 ({num_layers}L) MoE Design Matrix")
    print("=" * 78)
    print(
        f"{'Architecture Variant':<32} | {'Total(M)':>8} | {'Active(M)':>9} | "
        f"{'Q4 RAM':>8} | {'Est tok/s':>9}"
    )
    print("-" * 78)
    for row in variants:
        print(
            f"{row['name']:<32} | {row['total_params_m']:>8.1f} | "
            f"{row['active_params_m']:>9.1f} | "
            f"{row['q4_ram_mb']:>6.0f}MB | {row['est_tok_per_s']:>7.1f}/s"
        )
    print("=" * 78)
    return variants


def emit_mergekit_moe_yaml(num_layers: int = 28) -> str:
    """Return a ready-to-use mergekit-moe config for plowinf worker actions."""
    model_tag = (
        "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        if num_layers == 28
        else "./models/qwen2.5-1.5b-20L-hf"
    )
    return (
        "# Path: plowinf/configs/qwen25_plowinf_4x_moe.yml\n"
        f"base_model: {model_tag}\n"
        "gate_mode: hidden\n"
        "dtype: bfloat16\n"
        "experts_per_token: 2\n"
        "experts:\n"
        f"  - source_model: {model_tag}\n"
        "    positive_prompts:\n"
        '      - "Review the following Python code for bugs and edge cases."\n'
        f"  - source_model: {model_tag}\n"
        "    positive_prompts:\n"
        '      - "Write comprehensive pytest unit tests."\n'
        f"  - source_model: {model_tag}\n"
        "    positive_prompts:\n"
        '      - "Write concise PEP-257 Google-style docstrings."\n'
        f"  - source_model: {model_tag}\n"
        "    positive_prompts:\n"
        '      - "Refactor and implement this algorithm cleanly."\n'
        "# end of file: plowinf/configs/qwen25_plowinf_4x_moe.yml"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qwen2.5 Distillation & MoE Planning Lab for plowinf."
    )
    parser.add_argument(
        "--mode",
        choices=("plan-moe",),
        default="plan-moe",
        help="Execution mode (default: plan-moe)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=28,
        help="Layer count to model (e.g., 28 or 20)",
    )
    args = parser.parse_args(argv)
    compute_qwen_moe_budget(num_layers=args.layers)
    print("\nGenerated mergekit-moe Blueprint:\n")
    print(emit_mergekit_moe_yaml(num_layers=args.layers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# end of file: plowinf/tools/qwen_distill_moe_lab.py
