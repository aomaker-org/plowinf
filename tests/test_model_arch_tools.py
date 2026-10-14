# Path: plowinf/tests/test_model_arch_tools.py
"""Unit tests for GGUF architecture inspector and MoE distillation lab."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
from typing import Any

import pytest

# Ensure repo root is on sys.path so tools can be imported cleanly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.inspect_gguf_arch import (
    categorize_tensor,
    compare_summaries,
    inspect_gguf,
    parse_gguf_summary,
    resolve_ollama_gguf_blob,
)
from tools.qwen_distill_moe_lab import (
    calculate_moe_budgets,
    compute_qwen_moe_budget,
    emit_mergekit_moe_yaml,
)


def _pack_str(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_val(val_type: int, val: Any) -> bytes:
    prefix = struct.pack("<I", val_type)
    if val_type == 8:  # STRING
        return prefix + _pack_str(val)
    if val_type == 4:  # UINT32
        return prefix + struct.pack("<I", val)
    if val_type == 6:  # FLOAT32
        return prefix + struct.pack("<f", val)
    if val_type == 10:  # UINT64
        return prefix + struct.pack("<Q", val)
    raise NotImplementedError(f"Unsupported val_type {val_type}")


def create_synthetic_gguf_v3(
    dest: Path,
    num_layers: int = 2,
    arch: str = "qwen2",
    quant_type: int = 2,
) -> Path:
    """Create a minimal, valid binary GGUF v3 file for testing."""
    kv_pairs = [
        ("general.architecture", 8, arch),
        ("general.name", 8, f"Synthetic-{arch}"),
        ("general.file_type", 4, quant_type),
        (f"{arch}.block_count", 4, num_layers),
        (f"{arch}.attention.head_count", 4, 12),
        (f"{arch}.attention.head_count_kv", 4, 2),
        (f"{arch}.embedding_length", 4, 1536),
        (f"{arch}.feed_forward_length", 4, 8960),
        (f"{arch}.context_length", 4, 32768),
    ]

    tensors: list[tuple[str, list[int], int]] = [
        ("token_embd.weight", [1536, 1000], 0),  # F32
    ]
    for i in range(num_layers):
        tensors.extend([
            (f"blk.{i}.attn_q.weight", [1536, 1536], quant_type),
            (f"blk.{i}.attn_k.weight", [1536, 256], quant_type),
            (f"blk.{i}.attn_v.weight", [1536, 256], quant_type),
            (f"blk.{i}.attn_output.weight", [1536, 1536], quant_type),
            (f"blk.{i}.ffn_gate.weight", [1536, 8960], quant_type),
            (f"blk.{i}.ffn_up.weight", [1536, 8960], quant_type),
            (f"blk.{i}.ffn_down.weight", [8960, 1536], quant_type),
            (f"blk.{i}.attn_norm.weight", [1536], 0),
            (f"blk.{i}.ffn_norm.weight", [1536], 0),
        ])
    tensors.append(("output_norm.weight", [1536], 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fp:
        # Header: magic (0x46554747), version (3), n_tensors, n_kv
        fp.write(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kv_pairs)))

        # Metadata KV records
        for key, vtype, val in kv_pairs:
            fp.write(_pack_str(key))
            fp.write(_pack_val(vtype, val))

        # Tensor records
        offset = 0
        for name, dims, qtype in tensors:
            fp.write(_pack_str(name))
            fp.write(struct.pack("<I", len(dims)))
            for d in dims:
                fp.write(struct.pack("<Q", d))
            fp.write(struct.pack("<IQ", qtype, offset))
            offset += 1024

    return dest


def create_mock_ollama_manifest_tree(
    root: Path,
    model_ref: str,
    gguf_path: Path,
) -> Path:
    """Build a mock Ollama manifest and blob storage tree."""
    if ":" in model_ref:
        name, tag = model_ref.split(":", 1)
    else:
        name, tag = model_ref, "latest"

    digest_hash = "mock1234567890abcdef"
    blob_dir = root / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_target = blob_dir / f"sha256-{digest_hash}"
    blob_target.write_bytes(gguf_path.read_bytes())

    manifest_dir = root / "manifests" / "registry.ollama.ai" / "library" / name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / tag

    manifest_data = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{digest_hash}",
                "size": gguf_path.stat().st_size,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    return manifest_path


def test_categorize_tensor():
    assert categorize_tensor("token_embd.weight") == "1. Embeddings / Tied LM Head"
    assert categorize_tensor("output.weight") == "1. Embeddings / Tied LM Head"
    assert categorize_tensor("blk.0.attn_q.weight") == "2. GQA Attention (Q/K/V/O + Biases)"
    assert categorize_tensor("blk.1.ffn_gate_inp.weight") == "3. MoE Router Gates"
    assert categorize_tensor("blk.2.ffn_gate.weight") == "4. SwiGLU FFN / Experts"
    assert categorize_tensor("blk.3.ffn_up_exps.weight") == "4. SwiGLU FFN / Experts"
    assert categorize_tensor("blk.0.attn_norm.weight") == "2. GQA Attention (Q/K/V/O + Biases)"
    assert categorize_tensor("output_norm.weight") == "5. RMSNorm & Output Norm (FP32)"


def test_parse_synthetic_gguf_v3(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    create_synthetic_gguf_v3(gguf_file, num_layers=4, arch="qwen2")

    summary = parse_gguf_summary(gguf_file)
    assert summary["version"] == 3
    assert summary["architecture"] == "qwen2"
    assert summary["detected_layers"] == 4
    assert summary["block_count"] == 4
    assert summary["head_count"] == 12
    assert summary["head_count_kv"] == 2
    assert summary["embedding_length"] == 1536
    assert summary["feed_forward_length"] == 8960
    assert summary["n_tensors"] == 1 + 4 * 9 + 1
    assert "token_embd.weight" in summary["sample_shapes"]
    assert summary["total_params"] > 0


def test_parse_gguf_summary_invalid_magic(tmp_path):
    bad_file = tmp_path / "corrupt.gguf"
    bad_file.write_bytes(b"NOT_A_GGUF_MAGIC_HEADER_1234")
    with pytest.raises(ValueError, match="Invalid GGUF magic header"):
        parse_gguf_summary(bad_file)


def test_parse_gguf_summary_too_small(tmp_path):
    tiny_file = tmp_path / "tiny.gguf"
    tiny_file.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="File too small"):
        parse_gguf_summary(tiny_file)


def test_resolve_ollama_gguf_blob_with_mock_root(tmp_path):
    gguf_file = tmp_path / "synthetic.gguf"
    create_synthetic_gguf_v3(gguf_file, num_layers=2)

    ollama_root = tmp_path / "ollama_models"
    create_mock_ollama_manifest_tree(
        ollama_root, "qwen2.5-1.5b-20L:latest", gguf_file
    )

    resolved = resolve_ollama_gguf_blob(
        "qwen2.5-1.5b-20L:latest",
        search_roots=[ollama_root],
    )
    assert resolved.is_file()
    assert "sha256-mock1234567890abcdef" in resolved.name


def test_resolve_ollama_gguf_blob_not_found(tmp_path):
    empty_root = tmp_path / "empty_ollama"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="Could not resolve Ollama manifest"):
        resolve_ollama_gguf_blob("nonexistent:latest", search_roots=[empty_root])


def test_inspect_gguf_runs(tmp_path, capsys):
    gguf_file = tmp_path / "inspect_test.gguf"
    create_synthetic_gguf_v3(gguf_file, num_layers=3)

    summary = inspect_gguf(gguf_file, label="test-model")
    assert summary["detected_layers"] == 3
    captured = capsys.readouterr()
    assert "GGUF v3" in captured.out
    assert "SwiGLU FFN / Experts" in captured.out


def test_compare_summaries_output(tmp_path, capsys):
    model_a = tmp_path / "model_a.gguf"
    model_b = tmp_path / "model_b.gguf"
    create_synthetic_gguf_v3(model_a, num_layers=28, quant_type=2)  # Q4_0
    create_synthetic_gguf_v3(model_b, num_layers=20, quant_type=12) # Q4_K

    sum_a = parse_gguf_summary(model_a)
    sum_b = parse_gguf_summary(model_b)
    compare_summaries(sum_a, sum_b, "synthetic:28L", "synthetic:20L")

    captured = capsys.readouterr()
    assert "Comparing GGUF Architectures" in captured.out
    assert "20 (-8, -28.6%)" in captured.out
    assert "6:1" in captured.out
    assert "SwiGLU FFN Dim (d_ff)" in captured.out


def test_calculate_moe_budgets_28l_and_20l():
    b28 = calculate_moe_budgets(num_layers=28)
    assert b28["num_layers"] == 28
    assert pytest.approx(b28["dense"]["total_m"], rel=1e-2) == 1543.6
    assert pytest.approx(b28["upcycled_4x"]["total_m"], rel=1e-2) == 5012.0
    assert pytest.approx(b28["sub_ffn_moe"]["total_m"], rel=1e-2) == 1543.8
    assert pytest.approx(b28["sub_ffn_moe"]["active_m"], rel=1e-2) == 850.2
    assert b28["sub_ffn_moe"]["tps"] > b28["dense"]["tps"]

    b20 = calculate_moe_budgets(num_layers=20)
    assert b20["num_layers"] == 20
    assert pytest.approx(b20["dense"]["total_m"], rel=1e-2) == 1169.3
    assert pytest.approx(b20["upcycled_4x"]["total_m"], rel=1e-2) == 3646.7
    assert pytest.approx(b20["sub_ffn_moe"]["total_m"], rel=1e-2) == 1169.4
    assert pytest.approx(b20["sub_ffn_moe"]["active_m"], rel=1e-2) == 673.9
    assert b20["dense"]["ram_mb"] < b28["dense"]["ram_mb"]


def test_compute_qwen_moe_budget_returns_dict(capsys):
    budgets = compute_qwen_moe_budget(num_layers=28)
    assert "dense" in budgets
    assert "upcycled_4x" in budgets
    assert "sub_ffn_moe" in budgets
    captured = capsys.readouterr()
    assert "Qwen2.5 (28L) MoE Design Matrix" in captured.out


def test_emit_mergekit_moe_yaml():
    yaml_28 = emit_mergekit_moe_yaml(num_layers=28)
    assert "base_model: Qwen/Qwen2.5-1.5B-Instruct" in yaml_28
    assert "gate_mode: hidden" in yaml_28
    assert "experts_per_token: 2" in yaml_28
    assert "Review the following Python code" in yaml_28
    assert "pytest unit tests" in yaml_28

    yaml_20 = emit_mergekit_moe_yaml(num_layers=20)
    assert "base_model: ./models/qwen2.5-1.5b-20L-hf" in yaml_20


# end of file: plowinf/tests/test_model_arch_tools.py
