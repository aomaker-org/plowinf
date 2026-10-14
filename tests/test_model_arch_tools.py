# Path: plowinf/tests/test_model_arch_tools.py
"""Unit tests for GGUF architecture inspection and Qwen MoE planning tools."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.inspect_gguf_arch import (  # noqa: E402
    GGUF_MAGIC,
    categorize_tensor,
    compare_gguf_models,
    parse_gguf_summary,
    resolve_ollama_gguf_blob,
)
from tools.qwen_distill_moe_lab import (  # noqa: E402
    calculate_qwen_moe_variants,
    emit_mergekit_moe_yaml,
)


def _u32(val: int) -> bytes:
    return val.to_bytes(4, "little", signed=False)


def _u64(val: int) -> bytes:
    return val.to_bytes(8, "little", signed=False)


def _pack_gguf_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return _u64(len(raw)) + raw


def _write_synthetic_gguf(path: Path, num_layers: int) -> Path:
    """Create a minimal valid GGUF v3 binary header with synthetic layers."""
    tensors: list[tuple[str, list[int], int]] = [
        ("token_embd.weight", [1536, 1000], 12),  # Q4_K
        ("output_norm.weight", [1536], 0),        # F32
    ]
    for i in range(num_layers):
        tensors.extend(
            [
                (f"blk.{i}.attn_norm.weight", [1536], 0),
                (f"blk.{i}.attn_q.weight", [1536, 1536], 12),
                (f"blk.{i}.attn_k.weight", [1536, 256], 12),
                (f"blk.{i}.attn_v.weight", [1536, 256], 12),
                (f"blk.{i}.attn_output.weight", [1536, 1536], 12),
                (f"blk.{i}.ffn_gate.weight", [1536, 8960], 12),
                (f"blk.{i}.ffn_up.weight", [1536, 8960], 12),
                (f"blk.{i}.ffn_down.weight", [8960, 1536], 12),
            ]
        )

    kv_bytes = (
        _pack_gguf_str("general.architecture")
        + _u32(8)
        + _pack_gguf_str("qwen2")
        + _pack_gguf_str("qwen2.block_count")
        + _u32(4)
        + _u32(num_layers)
    )

    tensor_bytes = bytearray()
    for t_name, dims, ggml_type in tensors:
        tensor_bytes.extend(_pack_gguf_str(t_name))
        tensor_bytes.extend(_u32(len(dims)))
        for d in dims:
            tensor_bytes.extend(_u64(d))
        tensor_bytes.extend(_u32(ggml_type))
        tensor_bytes.extend(_u64(0))

    header = (
        _u32(GGUF_MAGIC)
        + _u32(3)
        + _u64(len(tensors))
        + _u64(2)
    )
    path.write_bytes(header + kv_bytes + bytes(tensor_bytes))
    return path


def test_parse_gguf_summary_detects_layers_and_buckets(tmp_path: Path) -> None:
    gguf_20 = _write_synthetic_gguf(tmp_path / "qwen_20l.gguf", num_layers=20)
    info = parse_gguf_summary(gguf_20)

    assert info["version"] == 3
    assert info["architecture"] == "qwen2"
    assert info["detected_layers"] == 20
    assert info["metadata"]["qwen2.block_count"] == 20
    assert "4. SwiGLU FFN / Experts" in info["bucket_elements"]
    assert "2. GQA Attention (Q/K/V/O + Biases)" in info["bucket_elements"]
    assert info["bucket_elements"]["4. SwiGLU FFN / Experts"] > (
        info["bucket_elements"]["2. GQA Attention (Q/K/V/O + Biases)"]
    )


def test_resolve_ollama_gguf_blob_and_compare(tmp_path: Path) -> None:
    models_root = tmp_path / ".ollama" / "models"
    blobs_dir = models_root / "blobs"
    blobs_dir.mkdir(parents=True)

    blob_28 = _write_synthetic_gguf(blobs_dir / "sha256-aaa28", num_layers=28)
    blob_20 = _write_synthetic_gguf(blobs_dir / "sha256-bbb20", num_layers=20)

    for tag_dir, digest in [
        ("qwen2/1.5b", "sha256:aaa28"),
        ("qwen2.5-1.5b-20L/latest", "sha256:bbb20"),
    ]:
        m_path = (
            models_root
            / "manifests"
            / "registry.ollama.ai"
            / "library"
            / tag_dir
        )
        m_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "mediaType": "application/vnd.ollama.image.model",
                            "digest": digest,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    resolved_28 = resolve_ollama_gguf_blob(
        "qwen2:1.5b", search_roots=[models_root]
    )
    resolved_20 = resolve_ollama_gguf_blob(
        "qwen2.5-1.5b-20L:latest", search_roots=[models_root]
    )
    assert resolved_28 == blob_28
    assert resolved_20 == blob_20

    info_a, info_b = compare_gguf_models(
        resolved_28, "qwen2:1.5b", resolved_20, "qwen2.5-1.5b-20L:latest"
    )
    assert info_a["detected_layers"] == 28
    assert info_b["detected_layers"] == 20
    assert info_b["total_elements"] < info_a["total_elements"]


def test_qwen_moe_budget_and_yaml_generation() -> None:
    v28 = calculate_qwen_moe_variants(num_layers=28)
    v20 = calculate_qwen_moe_variants(num_layers=20)

    assert len(v28) == 3
    assert len(v20) == 3
    assert v20[0]["total_params_m"] < v28[0]["total_params_m"]
    assert v20[2]["active_params_m"] < v20[0]["active_params_m"]
    assert v20[2]["est_tok_per_s"] > v20[0]["est_tok_per_s"]

    yaml_20 = emit_mergekit_moe_yaml(num_layers=20)
    assert "experts_per_token: 2" in yaml_20
    assert "qwen2.5-1.5b-20L-hf" in yaml_20

    with pytest.raises(ValueError, match="num_layers must be positive"):
        calculate_qwen_moe_variants(num_layers=0)


def test_categorize_tensor():
    assert categorize_tensor("token_embd.weight") == "1. Embeddings / Tied LM Head"
    assert categorize_tensor("output.weight") == "1. Embeddings / Tied LM Head"
    assert categorize_tensor("blk.0.attn_q.weight") == "2. GQA Attention (Q/K/V/O + Biases)"
    assert categorize_tensor("blk.1.ffn_gate_inp.weight") == "3. MoE Router Gates"
    assert categorize_tensor("blk.2.ffn_gate.weight") == "4. SwiGLU FFN / Experts"
    assert categorize_tensor("blk.3.ffn_up_exps.weight") == "4. SwiGLU FFN / Experts"
    assert categorize_tensor("output_norm.weight") == "5. RMSNorm & Output Norm (FP32)"


def test_parse_gguf_summary_invalid_magic(tmp_path: Path):
    bad_file = tmp_path / "corrupt.gguf"
    bad_file.write_bytes(b"NOT_A_GGUF_MAGIC_HEADER_1234")
    with pytest.raises(ValueError, match="Invalid GGUF magic header"):
        parse_gguf_summary(bad_file)


def test_parse_gguf_summary_too_small(tmp_path: Path):
    tiny_file = tmp_path / "tiny.gguf"
    tiny_file.write_bytes(b"GGUF")
    with pytest.raises((ValueError, EOFError)):
        parse_gguf_summary(tiny_file)


def test_resolve_ollama_gguf_blob_not_found(tmp_path: Path):
    empty_root = tmp_path / "empty_ollama"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="Could not resolve Ollama manifest"):
        resolve_ollama_gguf_blob("nonexistent:latest", search_roots=[empty_root])


# end of file: plowinf/tests/test_model_arch_tools.py
