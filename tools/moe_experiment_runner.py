# Path: plowinf/tools/moe_experiment_runner.py
"""MoE Experiment Runner & Blueprint Generator for plowinf.

Scaffolds multi-specialist LoRA training configs, mergekit-moe specs, and
Sub-FFN MoEfication partitioning plans for qwen2.5-coder:1.5b on fg26.

Run via uv (Directive 01):
    uv run python tools/moe_experiment_runner.py --experiment upcycle
    uv run python tools/moe_experiment_runner.py --experiment moefify
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    """Return ISO-8601 UTC microsecond timestamp per Directive 02."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def generate_upcycle_blueprint() -> dict[str, Any]:
    """Generate 3-expert mergekit-moe config for plowinf code actions."""
    config_dir = REPO_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""# Path: plowinf/configs/qwen25_coder_3x_moe.yml
# Multi-specialist upcycling spec for plowinf code tasks on fg26
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
gate_mode: hidden
dtype: bfloat16
experts_per_token: 1
experts:
  - source_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
    positive_prompts:
      - "Review the following Python/C code for bugs, security vulnerabilities, and memory safety."
      - "Perform a rigorous code review focusing on race conditions and edge cases."
  - source_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
    positive_prompts:
      - "Write comprehensive pytest unit tests covering boundary conditions and mock fixtures."
      - "Generate robust test assertions for this module."
  - source_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
    positive_prompts:
      - "Refactor and optimize this algorithm for minimal memory footprint and high performance."
      - "Clean up code structure and implement idiomatic type annotations."
# end of file: plowinf/configs/qwen25_coder_3x_moe.yml
"""
    out_path = config_dir / "qwen25_coder_3x_moe.yml"
    out_path.write_text(yaml_content, encoding="utf-8")
    print(f"[{utc_now_iso()}] Generated MergeKit spec: {out_path}")
    return {"config_path": str(out_path), "experts": 3, "gate_mode": "hidden"}


def generate_moefication_plan() -> dict[str, Any]:
    """Output Sub-FFN MoEfication partitioning parameters."""
    plan = {
        "parent_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "d_model": 1536,
        "d_ff_dense": 8960,
        "num_layers": 28,
        "partition": {
            "shared_expert": {
                "count": 1,
                "intermediate_size": 1792,
                "role": "Universal syntax & token grammar (always active)",
            },
            "routed_experts": {
                "count": 4,
                "intermediate_size": 1792,
                "active_top_k": 1,
                "role": "Domain-specific reasoning (review, test, refactor, docs)",
            },
        },
        "estimated_active_params_m": 850.2,
        "estimated_total_params_m": 1543.7,
        "estimated_haswell_tps": 27.3,
    }
    plan_path = REPO_ROOT / "configs" / "moefication_plan_1.5b.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"[{utc_now_iso()}] Generated MoEfication plan: {plan_path}")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MoE Experiment Runner for plowinf."
    )
    parser.add_argument(
        "--experiment",
        choices=("upcycle", "moefication"),
        default="upcycle",
        help="Select MoE construction experiment",
    )
    args = parser.parse_args(argv)

    if args.experiment == "upcycle":
        res = generate_upcycle_blueprint()
        print(json.dumps(res, indent=2))
    elif args.experiment == "moefication":
        res = generate_moefication_plan()
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# end of file: plowinf/tools/moe_experiment_runner.py
