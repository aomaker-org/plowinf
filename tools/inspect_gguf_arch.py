# Path: plowinf/tools/inspect_gguf_arch.py
"""Offline GGUF binary header & Ollama model architecture inspector.

Parses GGUF v2/v3 model files and local Ollama manifests using only the Python
standard library. Audits layer counts (e.g., 28L vs 20L pruned), GQA head
ratios, SwiGLU FFN dimensions, MoE expert tensors, and RAM budgets on fg26.

Usage via uv (Directive 01):
    uv run python tools/inspect_gguf_arch.py ~/ai_models/qwen2-1.5b.gguf
    uv run python tools/inspect_gguf_arch.py --ollama qwen2.5-1.5b-20L:latest
    uv run python tools/inspect_gguf_arch.py --compare-ollama \\
        qwen2:1.5b qwen2.5-1.5b-20L:latest
    uv run python tools/inspect_gguf_arch.py --compare \\
        ~/ai_models/qwen2-1.5b.gguf /path/to/other.gguf
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import struct
import sys
from typing import Any, BinaryIO

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian uint32

GGUF_VAL_TYPES: dict[int, str] = {
    0: "<B",   # UINT8
    1: "<b",   # INT8
    2: "<H",   # UINT16
    3: "<h",   # INT16
    4: "<I",   # UINT32
    5: "<i",   # INT32
    6: "<f",   # FLOAT32
    7: "<?",   # BOOL
    8: "str",  # STRING
    9: "arr",  # ARRAY
    10: "<Q",  # UINT64
    11: "<q",  # INT64
    12: "<d",  # FLOAT64
}

# (type_name, block_size_elements, bytes_per_block)
GGML_QUANT_SPECS: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}

LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def utc_now_iso() -> str:
    """Return ISO-8601 UTC microsecond timestamp per Directive 02."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def resolve_ollama_gguf_blob(
    model_ref: str,
    search_roots: Sequence[Path] | None = None,
) -> Path:
    """Resolve an Ollama model tag (e.g., qwen2.5-1.5b-20L:latest) to its blob."""
    if ":" in model_ref:
        name_part, tag = model_ref.split(":", 1)
    else:
        name_part, tag = model_ref, "latest"

    if "/" not in name_part:
        rel_manifest = Path("registry.ollama.ai") / "library" / name_part / tag
    else:
        rel_manifest = Path("registry.ollama.ai") / name_part / tag

    candidate_roots = (
        list(search_roots)
        if search_roots is not None
        else [
            Path.home() / ".ollama" / "models",
            Path("/usr/share/ollama/.ollama/models"),
            Path("/var/lib/ollama/.ollama/models"),
        ]
    )

    for root in candidate_roots:
        manifest_path = root / "manifests" / rel_manifest
        if manifest_path.is_file():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for layer in data.get("layers", []):
                media = layer.get("mediaType", "")
                if "ollama.image.model" in media:
                    digest = layer["digest"].replace(":", "-")
                    blob_path = root / "blobs" / digest
                    if blob_path.is_file():
                        return blob_path

    raise FileNotFoundError(
        f"Could not resolve Ollama manifest for '{model_ref}' in "
        f"{[str(r) for r in candidate_roots]}"
    )


def _read_string(fp: BinaryIO) -> str:
    raw = fp.read(8)
    if len(raw) < 8:
        raise EOFError("Unexpected EOF reading GGUF string length")
    length = struct.unpack("<Q", raw)[0]
    return fp.read(length).decode("utf-8", errors="replace")


def _read_value(fp: BinaryIO, val_type: int) -> Any:
    fmt = GGUF_VAL_TYPES.get(val_type)
    if fmt is None:
        raise ValueError(f"Unsupported GGUF metadata value type: {val_type}")
    if fmt == "str":
        return _read_string(fp)
    if fmt == "arr":
        elem_type, count = struct.unpack("<IQ", fp.read(12))
        if count > 32:
            for _ in range(count):
                _read_value(fp, elem_type)
            return f"[Array len={count} type_id={elem_type}]"
        return [_read_value(fp, elem_type) for _ in range(count)]
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fp.read(size))[0]


def _estimate_bytes(n_elems: int, ggml_type: int) -> float:
    spec = GGML_QUANT_SPECS.get(ggml_type)
    if spec is None:
        return float(n_elems)
    _, block_elems, block_bytes = spec
    return (n_elems / block_elems) * block_bytes


def categorize_tensor(t_name: str) -> str:
    """Map a GGUF tensor name into its functional architectural bucket."""
    if "token_embd" in t_name or t_name == "output.weight":
        return "1. Embeddings / Tied LM Head"
    if ".attn_" in t_name:
        return "2. GQA Attention (Q/K/V/O + Biases)"
    if ".ffn_gate_inp" in t_name:
        return "3. MoE Router Gates"
    if "_exps." in t_name or ".ffn_" in t_name:
        return "4. SwiGLU FFN / Experts"
    return "5. RMSNorm & Output Norm (FP32)"


def parse_gguf_summary(path: Path) -> dict[str, Any]:
    """Parse GGUF header and tensor table into a structured dictionary."""
    file_mb = path.stat().st_size / (1024 * 1024)

    with path.open("rb") as fp:
        raw_header = fp.read(24)
        if len(raw_header) < 24:
            raise ValueError(f"File too small to be GGUF: {path}")
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", raw_header)
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"Invalid GGUF magic header in {path}: {hex(magic)}"
            )

        metadata: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_string(fp)
            val_type = struct.unpack("<I", fp.read(4))[0]
            metadata[key] = _read_value(fp, val_type)

        bucket_elements: dict[str, int] = defaultdict(int)
        bucket_bytes: dict[str, float] = defaultdict(float)
        bucket_counts: dict[str, int] = defaultdict(int)
        qtype_counts: dict[str, int] = defaultdict(int)
        layer_indices: set[int] = set()
        sample_shapes: dict[str, list[int]] = {}

        for _ in range(n_tensors):
            t_name = _read_string(fp)
            n_dims = struct.unpack("<I", fp.read(4))[0]
            dims = [struct.unpack("<Q", fp.read(8))[0] for _ in range(n_dims)]
            ggml_type, _offset = struct.unpack("<IQ", fp.read(12))

            m = LAYER_RE.match(t_name)
            if m:
                layer_indices.add(int(m.group(1)))

            elems = 1
            for d in dims:
                elems *= d

            t_bytes = _estimate_bytes(elems, ggml_type)
            q_name = GGML_QUANT_SPECS.get(
                ggml_type, (f"T{ggml_type}", 1, 1)
            )[0]
            qtype_counts[q_name] += 1

            if t_name.startswith("blk.0.") or t_name in (
                "token_embd.weight",
                "output.weight",
            ):
                sample_shapes[t_name] = dims

            cat = categorize_tensor(t_name)
            bucket_elements[cat] += elems
            bucket_bytes[cat] += t_bytes
            bucket_counts[cat] += 1

    arch = str(metadata.get("general.architecture", "unknown"))
    head_count = int(metadata.get(f"{arch}.attention.head_count", 0))
    head_count_kv = int(metadata.get(f"{arch}.attention.head_count_kv", 0))
    embd_len = int(metadata.get(f"{arch}.embedding_length", 0))
    ffn_len = int(metadata.get(f"{arch}.feed_forward_length", 0))
    block_cnt = int(metadata.get(f"{arch}.block_count", len(layer_indices)))

    total_elems = sum(bucket_elements.values())
    total_mb = sum(bucket_bytes.values()) / (1024 * 1024)

    return {
        "path": str(path),
        "file_mb": file_mb,
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "metadata": metadata,
        "architecture": arch,
        "detected_layers": len(layer_indices),
        "total_params": total_elems,
        "total_mb": total_mb,
        "bucket_elements": dict(bucket_elements),
        "bucket_bytes": dict(bucket_bytes),
        "bucket_counts": dict(bucket_counts),
        "qtype_counts": dict(qtype_counts),
        "sample_shapes": sample_shapes,
        "head_count": head_count,
        "head_count_kv": head_count_kv,
        "embedding_length": embd_len,
        "feed_forward_length": ffn_len,
        "block_count": block_cnt,
    }


def inspect_gguf(path: Path, label: str | None = None) -> dict[str, Any]:
    """Parse GGUF header metadata and summarize tensor component budgets."""
    title = f"{label} ({path})" if label else str(path)
    summary = parse_gguf_summary(path)
    print(
        f"[{utc_now_iso()}] Inspecting GGUF: {title} "
        f"({summary['file_mb']:.2f} MB)"
    )

    arch = summary["architecture"]
    print("=" * 78)
    print(
        f"GGUF v{summary['version']} | Arch: {arch} | "
        f"Tensors: {summary['n_tensors']} | "
        f"Detected Layers: {summary['detected_layers']}"
    )
    print("=" * 78)
    meta = summary["metadata"]
    for k in sorted(meta):
        if k.startswith((f"{arch}.", "general.name", "general.file_type")):
            print(f"  {k:<40} : {meta[k]}")

    b_elems = summary["bucket_elements"]
    b_bytes = summary["bucket_bytes"]
    b_counts = summary["bucket_counts"]
    total_elems = summary["total_params"]
    total_mb = summary["total_mb"]

    print("-" * 78)
    print(
        f"{'Component Category':<35} | {'Cnt':>4} | {'Params(M)':>10} | "
        f"{'Size(MB)':>9} | {'Share':>6}"
    )
    print("-" * 78)
    for cat in sorted(b_elems):
        elems = b_elems[cat]
        mb = b_bytes[cat] / (1024 * 1024)
        pct = (elems / total_elems * 100.0) if total_elems else 0.0
        print(
            f"{cat:<35} | {b_counts[cat]:>4} | {elems / 1e6:>10.2f} | "
            f"{mb:>9.2f} | {pct:>5.1f}%"
        )
    print("-" * 78)
    print(
        f"{'TOTAL':<35} | {summary['n_tensors']:>4} | "
        f"{total_elems / 1e6:>10.2f} | {total_mb:>9.2f} | 100.0%"
    )
    print("-" * 78)
    print("Layer 0 & Embedding Tensor Shapes (GGML dim order):")
    for t_name, dims in sorted(summary["sample_shapes"].items()):
        print(f"  {t_name:<30} : {dims}")
    print(f"Quantization Block Distribution : {summary['qtype_counts']}")
    print("=" * 78)
    return summary


def compare_summaries(
    sum_a: dict[str, Any],
    sum_b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> None:
    """Print side-by-side architectural and parameter budget comparison."""
    print(f"[{utc_now_iso()}] Comparing GGUF Architectures")
    print(f"  Model A: {label_a}")
    print(f"  Model B: {label_b}")
    print("=" * 78)
    print(
        f"{'Metric / Architectural Dimension':<34} | "
        f"{'Model A':<19} | {'Model B':<19}"
    )
    print("-" * 78)

    name_a = str(sum_a["metadata"].get("general.name", label_a))
    name_b = str(sum_b["metadata"].get("general.name", label_b))
    print(f"{'Model Name':<34} | {name_a[:19]:<19} | {name_b[:19]:<19}")
    print(
        f"{'Architecture':<34} | {sum_a['architecture']:<19} | "
        f"{sum_b['architecture']:<19}"
    )

    l_a, l_b = sum_a["detected_layers"], sum_b["detected_layers"]
    l_delta = l_b - l_a
    l_pct = (l_delta / l_a * 100.0) if l_a else 0.0
    l_b_str = (
        f"{l_b} ({l_delta:+d}, {l_pct:+.1f}%)" if l_delta != 0 else str(l_b)
    )
    print(
        f"{'Detected Transformer Layers':<34} | {str(l_a):<19} | {l_b_str:<19}"
    )

    bk_a = sum_a.get("block_count", l_a)
    bk_b = sum_b.get("block_count", l_b)
    print(f"{'Config Block Count':<34} | {str(bk_a):<19} | {str(bk_b):<19}")

    arch_a = sum_a["architecture"]
    arch_b = sum_b["architecture"]
    ctx_a = str(sum_a["metadata"].get(f"{arch_a}.context_length", "N/A"))
    ctx_b = str(sum_b["metadata"].get(f"{arch_b}.context_length", "N/A"))
    print(f"{'Context Length':<34} | {ctx_a:<19} | {ctx_b:<19}")

    emb_a = sum_a["embedding_length"]
    emb_b = sum_b["embedding_length"]
    print(
        f"{'Embedding Dim (d_model)':<34} | {str(emb_a):<19} | {str(emb_b):<19}"
    )

    h_a, h_b = sum_a["head_count"], sum_b["head_count"]
    kv_a, kv_b = sum_a["head_count_kv"], sum_b["head_count_kv"]
    ratio_a = (
        f"{h_a // kv_a}:1"
        if (h_a and kv_a and h_a % kv_a == 0)
        else f"{h_a}:{kv_a}"
    )
    ratio_b = (
        f"{h_b // kv_b}:1"
        if (h_b and kv_b and h_b % kv_b == 0)
        else f"{h_b}:{kv_b}"
    )
    print(f"{'GQA Query Heads':<34} | {str(h_a):<19} | {str(h_b):<19}")
    print(f"{'GQA Key/Value Heads':<34} | {str(kv_a):<19} | {str(kv_b):<19}")
    print(f"{'GQA Head Ratio':<34} | {ratio_a:<19} | {ratio_b:<19}")

    head_dim_a = (emb_a // h_a) if (emb_a and h_a) else "N/A"
    head_dim_b = (emb_b // h_b) if (emb_b and h_b) else "N/A"
    print(
        f"{'Head Dimension (d_k)':<34} | {str(head_dim_a):<19} | "
        f"{str(head_dim_b):<19}"
    )

    ffn_a = sum_a["feed_forward_length"]
    ffn_b = sum_b["feed_forward_length"]
    print(
        f"{'SwiGLU FFN Dim (d_ff)':<34} | {str(ffn_a):<19} | {str(ffn_b):<19}"
    )

    p_a = sum_a["total_params"] / 1e6
    p_b = sum_b["total_params"] / 1e6
    p_delta = p_b - p_a
    p_pct = (p_delta / p_a * 100.0) if p_a else 0.0
    p_b_str = (
        f"{p_b:.2f}M ({p_delta:+.2f}M, {p_pct:+.1f}%)"
        if p_delta != 0
        else f"{p_b:.2f}M"
    )
    print(
        f"{'Total Parameter Count':<34} | {f'{p_a:.2f}M':<19} | {p_b_str:<19}"
    )

    sz_a = sum_a["file_mb"]
    sz_b = sum_b["file_mb"]
    sz_delta = sz_b - sz_a
    sz_pct = (sz_delta / sz_a * 100.0) if sz_a else 0.0
    sz_b_str = (
        f"{sz_b:.2f}MB ({sz_delta:+.2f}MB, {sz_pct:+.1f}%)"
        if sz_delta != 0
        else f"{sz_b:.2f}MB"
    )
    print(
        f"{'File / Blob Size':<34} | {f'{sz_a:.2f}MB':<19} | {sz_b_str:<19}"
    )

    print("-" * 78)
    print("Component Parameter Breakdown (M params and % share):")
    all_cats = sorted(
        set(sum_a["bucket_elements"]) | set(sum_b["bucket_elements"])
    )
    for cat in all_cats:
        el_a = sum_a["bucket_elements"].get(cat, 0)
        el_b = sum_b["bucket_elements"].get(cat, 0)
        pct_a = (
            (el_a / sum_a["total_params"] * 100.0)
            if sum_a["total_params"]
            else 0.0
        )
        pct_b = (
            (el_b / sum_b["total_params"] * 100.0)
            if sum_b["total_params"]
            else 0.0
        )
        str_a = f"{el_a / 1e6:>8.2f}M ({pct_a:>4.1f}%)"
        str_b = f"{el_b / 1e6:>8.2f}M ({pct_b:>4.1f}%)"
        print(f"  {cat:<32} | {str_a:<19} | {str_b:<19}")

    print("-" * 78)
    print("Quantization Block Distributions:")
    print(f"  Model A : {sum_a['qtype_counts']}")
    print(f"  Model B : {sum_b['qtype_counts']}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect local GGUF file or Ollama model tag."
    )
    parser.add_argument(
        "gguf_path",
        nargs="?",
        type=Path,
        help="Path to local .gguf file",
    )
    parser.add_argument(
        "--ollama",
        type=str,
        help="Ollama model name:tag (e.g., qwen2.5-1.5b-20L:latest)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("PATH1", "PATH2"),
        help="Compare two local .gguf files side-by-side",
    )
    parser.add_argument(
        "--compare-ollama",
        nargs=2,
        type=str,
        metavar=("MODEL1", "MODEL2"),
        help="Compare two Ollama models side-by-side",
    )
    args = parser.parse_args()

    try:
        if args.compare_ollama:
            tag_a, tag_b = args.compare_ollama
            blob_a = resolve_ollama_gguf_blob(tag_a)
            blob_b = resolve_ollama_gguf_blob(tag_b)
            sum_a = parse_gguf_summary(blob_a)
            sum_b = parse_gguf_summary(blob_b)
            compare_summaries(sum_a, sum_b, f"ollama:{tag_a}", f"ollama:{tag_b}")
            return 0

        if args.compare:
            path_a, path_b = args.compare
            if not path_a.is_file() or not path_b.is_file():
                print(
                    "Error: Both files in --compare must exist.",
                    file=sys.stderr,
                )
                return 1
            sum_a = parse_gguf_summary(path_a)
            sum_b = parse_gguf_summary(path_b)
            compare_summaries(sum_a, sum_b, str(path_a), str(path_b))
            return 0

        if args.ollama:
            blob_path = resolve_ollama_gguf_blob(args.ollama)
            inspect_gguf(blob_path, label=f"ollama:{args.ollama}")
            return 0

        if args.gguf_path:
            if not args.gguf_path.is_file():
                print(
                    f"Error: File not found: {args.gguf_path}",
                    file=sys.stderr,
                )
                return 1
            inspect_gguf(args.gguf_path)
            return 0

        parser.print_help()
        return 1
    except Exception as exc:
        print(f"[{utc_now_iso()}] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# end of file: plowinf/tools/inspect_gguf_arch.py
