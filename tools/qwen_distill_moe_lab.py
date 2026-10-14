# Path: plowinf/tools/qwen_distill_moe_lab.py
"""Qwen2.5 Layer Profiler, Distillation Planner, and MoE Surgical Toolkit.

Provides two operational modes for fg26 (12 GB RAM Haswell):
  1. 'plan-moe': Zero-dependency generator that calculates dense-to-MoE memory
     budgets, emits a 4-expert mergekit-moe config for plowinf tasks (review,
     test, docstring, prompt), and outputs the Sub-FFN MoEfication blueprint.
  2. 'profile-hf': PyTorch forward-hook profiler (optional torch dependency)
     that measures Block Influence (BI_i) across all layers to rank layers for
     depth pruning (e.g., 28L -> 20L) and tracks SwiGLU 8960-channel sparsity.

Run via uv (Directive 01):
    uv run python tools/qwen_distill_moe_lab.py --mode plan-moe
    uv run python tools/qwen_distill_moe_lab.py --mode plan-moe --layers 20
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


def utc_now_iso() -> str:
    """Return ISO-8601 UTC microsecond timestamp per Directive 02."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def calculate_moe_budgets(
    num_layers: int = 28,
    d_model: int = 1536,
    d_ff_dense: int = 8960,
    vocab_size: int = 151936,
    q4_mb_per_m: float = 0.576,
    bw_gbs: float = 14.5,
) -> dict[str, Any]:
    """Calculate comparative parameter counts, RAM, and throughput for fg26."""
    emb_m = (vocab_size * d_model) / 1e6
    attn_per_l_m = (
        (d_model * 1536 + 1536)
        + 2 * (d_model * 256 + 256)
        + (1536 * d_model)
    ) / 1e6
    ffn_dense_per_l_m = (3 * d_model * d_ff_dense) / 1e6

    # Variant A: Dense baseline (28L or 20L)
    dense_total_m = emb_m + num_layers * (attn_per_l_m + ffn_dense_per_l_m)
    dense_act_m = dense_total_m

    # Variant B: Upcycled 4x Expert MoE (mergekit-moe, E=4, Top-1)
    up_e4_total_m = emb_m + num_layers * (
        attn_per_l_m + 4 * ffn_dense_per_l_m + (d_model * 4) / 1e6
    )
    up_e4_act_k1_m = emb_m + num_layers * (
        attn_per_l_m + 1 * ffn_dense_per_l_m + (d_model * 4) / 1e6
    )

    # Variant C: Fine-Grained Sub-FFN MoE (8960 split into 1 Shared + 4 Routed
    # experts of d_exp = 1792 each; 1 Shared + Top-1 Routed active = 3584 dim)
    d_exp = d_ff_dense // 5  # 1792
    ffn_sub_per_exp_m = (3 * d_model * d_exp) / 1e6
    sub_total_m = emb_m + num_layers * (
        attn_per_l_m + 5 * ffn_sub_per_exp_m + (d_model * 4) / 1e6
    )
    sub_act_m = emb_m + num_layers * (
        attn_per_l_m + 2 * ffn_sub_per_exp_m + (d_model * 4) / 1e6
    )

    def _stats(tot_m: float, act_m: float) -> tuple[float, float, float]:
        ram_mb = tot_m * q4_mb_per_m
        act_mb = act_m * q4_mb_per_m
        est_tps = (bw_gbs * 1024.0) / (act_mb + 55.0)
        return ram_mb, act_mb, est_tps

    dense_ram, _, dense_tps = _stats(dense_total_m, dense_act_m)
    up_ram, _, up_tps = _stats(up_e4_total_m, up_e4_act_k1_m)
    sub_ram, _, sub_tps = _stats(sub_total_m, sub_act_m)

    return {
        "num_layers": num_layers,
        "d_model": d_model,
        "d_ff_dense": d_ff_dense,
        "vocab_size": vocab_size,
        "emb_m": emb_m,
        "attn_per_l_m": attn_per_l_m,
        "ffn_dense_per_l_m": ffn_dense_per_l_m,
        "dense": {
            "total_m": dense_total_m,
            "active_m": dense_act_m,
            "ram_mb": dense_ram,
            "tps": dense_tps,
        },
        "upcycled_4x": {
            "total_m": up_e4_total_m,
            "active_m": up_e4_act_k1_m,
            "ram_mb": up_ram,
            "tps": up_tps,
        },
        "sub_ffn_moe": {
            "total_m": sub_total_m,
            "active_m": sub_act_m,
            "ram_mb": sub_ram,
            "tps": sub_tps,
        },
    }


def compute_qwen_moe_budget(
    num_layers: int = 28,
    d_model: int = 1536,
    d_ff_dense: int = 8960,
    vocab_size: int = 151936,
) -> dict[str, Any]:
    """Print comparative memory and active-parameter table for fg26."""
    budgets = calculate_moe_budgets(
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
    rows = [
        (
            f"1. Dense Qwen2.5 ({num_layers}L)",
            budgets["dense"]["total_m"],
            budgets["dense"]["active_m"],
            budgets["dense"]["ram_mb"],
            budgets["dense"]["tps"],
        ),
        (
            f"2. Upcycled 4xMoE ({num_layers}L, Top-1)",
            budgets["upcycled_4x"]["total_m"],
            budgets["upcycled_4x"]["active_m"],
            budgets["upcycled_4x"]["ram_mb"],
            budgets["upcycled_4x"]["tps"],
        ),
        (
            f"3. Sub-FFN 1S+4R MoE ({num_layers}L, Top-1)",
            budgets["sub_ffn_moe"]["total_m"],
            budgets["sub_ffn_moe"]["active_m"],
            budgets["sub_ffn_moe"]["ram_mb"],
            budgets["sub_ffn_moe"]["tps"],
        ),
    ]
    for name, tot_m, act_m, ram_mb, est_tps in rows:
        print(
            f"{name:<32} | {tot_m:>8.1f} | {act_m:>9.1f} | "
            f"{ram_mb:>6.0f}MB | {est_tps:>7.1f}/s"
        )
    print("=" * 78)
    return budgets


def emit_mergekit_moe_yaml(num_layers: int = 28) -> str:
    """Return a ready-to-use mergekit-moe config for plowinf worker actions."""
    model_tag = (
        "Qwen/Qwen2.5-1.5B-Instruct"
        if num_layers == 28
        else "./models/qwen2.5-1.5b-20L-hf"
    )
    return f"""# Path: plowinf/configs/qwen25_plowinf_4x_moe.yml
# mergekit-moe configuration for 4-expert plowinf code worker
base_model: {model_tag}
gate_mode: hidden
dtype: bfloat16
experts_per_token: 2
experts:
  - source_model: {model_tag}
    positive_prompts:
      - "Review the following Python code for bugs, security, and edge cases."
      - "Audit this C/Python function for memory safety and error handling."
  - source_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
    positive_prompts:
      - "Write comprehensive pytest unit tests covering edge cases."
      - "Generate deterministic test assertions and fixtures for this module."
  - source_model: {model_tag}
    positive_prompts:
      - "Write concise PEP-257 Google-style docstrings with type annotations."
      - "Document the module architecture, parameters, and return types."
  - source_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
    positive_prompts:
      - "Refactor and implement this algorithm with clean, idiomatic code."
      - "Optimize the computational complexity and memory footprint."
# end of file: plowinf/configs/qwen25_plowinf_4x_moe.yml"""


def run_hf_profiler(model_id: str, sample_text: str) -> int:
    """Profile Layer Block Influence (BI_i) & SwiGLU channel sparsity."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print(
            "Error: '--mode profile-hf' requires torch and transformers.",
            file=sys.stderr,
        )
        return 1

    print(f"[{utc_now_iso()}] Loading HF model for profiling: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
    )
    model.eval()

    layer_io: dict[int, tuple[Any, Any]] = {}
    swiglu_sparsity: dict[int, float] = {}

    def make_layer_hook(idx: int):
        def _hook(_module, inputs, outputs):
            inp = inputs[0].detach()
            out = (
                outputs[0].detach()
                if isinstance(outputs, tuple)
                else outputs.detach()
            )
            layer_io[idx] = (inp, out)
        return _hook

    def make_mlp_down_hook(idx: int):
        def _hook(_module, inputs, _outputs):
            # inputs[0] to down_proj is the 8960-d SwiGLU intermediate vector
            act = inputs[0].detach()
            near_zero = (act.abs() < 0.05).float().mean().item() * 100.0
            swiglu_sparsity[idx] = near_zero
        return _hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(i)))
        hooks.append(
            layer.mlp.down_proj.register_forward_hook(make_mlp_down_hook(i))
        )

    inputs = tokenizer(sample_text, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    print("=" * 78)
    print(
        f"{'Layer':<7} | {'CosSim(in,out)':>14} | {'BlockInfluence':>14} | "
        f"{'SwiGLU <0.05':>13} | {'Prune Candidate'}"
    )
    print("-" * 78)
    scores: list[tuple[float, int]] = []
    for i in sorted(layer_io):
        x_in, x_out = layer_io[i]
        cos = (
            torch.nn.functional.cosine_similarity(x_in, x_out, dim=-1)
            .mean()
            .item()
        )
        bi = 1.0 - cos
        scores.append((bi, i))
        sp = swiglu_sparsity.get(i, 0.0)
        flag = "PROTECT (Boundary)" if i in (0, 1, len(layer_io) - 1) else ""
        print(f"blk.{i:<3} | {cos:>14.5f} | {bi:>14.5f} | {sp:>12.1f}% | {flag}")

    # Rank middle layers (excluding first 2 and final layer) by lowest BI
    prunable = [
        idx for _bi, idx in sorted(scores)
        if idx not in (0, 1, len(layer_io) - 1)
    ]
    drop_8 = sorted(prunable[:8])
    print("-" * 78)
    print(f"Recommended 8 Layers to Drop for 28L -> 20L: {drop_8}")
    print("=" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen2.5 Distillation & MoE Planning Lab for plowinf."
    )
    parser.add_argument(
        "--mode",
        choices=("plan-moe", "profile-hf"),
        default="plan-moe",
        help="Execution mode (default: plan-moe)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=28,
        help="Layer count to model (e.g., 28 or 20)",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="Local HF model directory for --mode profile-hf",
    )
    args = parser.parse_args()

    if args.mode == "plan-moe":
        compute_qwen_moe_budget(num_layers=args.layers)
        print("\nGenerated mergekit-moe Blueprint:\n")
        print(emit_mergekit_moe_yaml(num_layers=args.layers))
        return 0

    if not args.hf_model or not Path(args.hf_model).exists():
        print(
            "Error: Provide a valid local directory via --hf-model.",
            file=sys.stderr,
        )
        return 1
    sample_code = (
        "def review_queue_lock(fd: int) -> bool:\n"
        "    '''Verify advisory POSIX lock state.'''\n"
        "    return fd >= 0\n"
    )
    return run_hf_profiler(args.hf_model, sample_code)


if __name__ == "__main__":
    raise SystemExit(main())

# end of file: plowinf/tools/qwen_distill_moe_lab.py
