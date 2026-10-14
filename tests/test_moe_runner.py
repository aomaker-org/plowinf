# Path: plowinf/tests/test_moe_runner.py
"""Unit tests for the MoE Experiment Runner & Blueprint Generator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.moe_experiment_runner import (  # noqa: E402
    generate_moefication_plan,
    generate_upcycle_blueprint,
    main,
)


def test_generate_upcycle_blueprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that generate_upcycle_blueprint outputs valid mergekit YAML."""
    monkeypatch.setattr("tools.moe_experiment_runner.REPO_ROOT", tmp_path)
    res = generate_upcycle_blueprint()

    expected_path = tmp_path / "configs" / "qwen25_coder_3x_moe.yml"
    assert res["config_path"] == str(expected_path)
    assert res["experts"] == 3
    assert res["gate_mode"] == "hidden"
    assert expected_path.is_file()

    content = expected_path.read_text(encoding="utf-8")
    assert "base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct" in content
    assert "gate_mode: hidden" in content
    assert "dtype: bfloat16" in content
    assert "experts_per_token: 1" in content
    assert "Review the following Python/C code" in content
    assert "Write comprehensive pytest unit tests" in content
    assert "Refactor and optimize this algorithm" in content
    assert content.startswith("# Path: plowinf/configs/qwen25_coder_3x_moe.yml")
    assert content.strip().endswith("# end of file: plowinf/configs/qwen25_coder_3x_moe.yml")


def test_generate_moefication_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that generate_moefication_plan outputs valid schema and parameters."""
    monkeypatch.setattr("tools.moe_experiment_runner.REPO_ROOT", tmp_path)
    plan = generate_moefication_plan()

    expected_path = tmp_path / "configs" / "moefication_plan_1.5b.json"
    assert expected_path.is_file()
    assert plan["parent_model"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert plan["d_model"] == 1536
    assert plan["d_ff_dense"] == 8960
    assert plan["num_layers"] == 28

    partition = plan["partition"]
    assert partition["shared_expert"]["count"] == 1
    assert partition["shared_expert"]["intermediate_size"] == 1792
    assert partition["routed_experts"]["count"] == 4
    assert partition["routed_experts"]["active_top_k"] == 1
    assert partition["routed_experts"]["intermediate_size"] == 1792

    assert plan["estimated_active_params_m"] == 850.2
    assert plan["estimated_total_params_m"] == 1543.7
    assert plan["estimated_haswell_tps"] == 27.3

    disk_data = json.loads(expected_path.read_text(encoding="utf-8"))
    assert disk_data == plan


def test_moe_runner_cli_upcycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Verify execution of CLI with upcycle experiment option."""
    monkeypatch.setattr("tools.moe_experiment_runner.REPO_ROOT", tmp_path)
    ret = main(["--experiment", "upcycle"])
    assert ret == 0

    out = capsys.readouterr().out
    assert "qwen25_coder_3x_moe.yml" in out
    assert '"experts": 3' in out


def test_moe_runner_cli_moefication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verify execution of CLI with moefication experiment option."""
    monkeypatch.setattr("tools.moe_experiment_runner.REPO_ROOT", tmp_path)
    ret = main(["--experiment", "moefication"])
    assert ret == 0

    out = capsys.readouterr().out
    assert "moefication_plan_1.5b.json" in out
    assert '"estimated_haswell_tps": 27.3' in out


# end of file: plowinf/tests/test_moe_runner.py
