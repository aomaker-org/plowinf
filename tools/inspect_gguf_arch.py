# Path: plowinf/tools/inspect_gguf_arch.py
"""Offline GGUF binary header & Ollama model architecture inspector.

Parses GGUF v2/v3 model files and local Ollama manifests using only the Python
standard library. Audits layer counts (e.g., 28L vs 20L pruned), GQA head
ratios, SwiGLU FFN dimensions, MoE expert tensors, and RAM budgets on fg26.

Usage via uv (Directive 01):
    uv run python tools/inspect_gguf_arch.py ~/ai_models/qwen2-1.5b.gguf
    uv run python tools/inspect_gguf_arch.py --ollama qwen2.5-1.5b-20L:latest
    uv run python tools/inspect_gguf_arch.py --compare-ollama \
        qwen2:1.5b qwen2.5-1.5b-20L:latest
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
    0: "=B",   # UINT8
    1: "=b",   # INT8
    2: "=H",   # UINT16
    3: "=h",   # INT16
    4: "=I",   # UINT32
    5: "=i",   # INT32
    6: "=f",   # FLOAT32
    7: "=?",   # BOOL
    8: "str",  # STRING
    9: "arr",  # ARRAY
    10: "=Q",  # UINT64
    11: "=q",  # INT64
    12: "=d",  # FLOAT64
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


def _read_string(fp: BinaryIO) -> str:
    length = _read_u64(fp)
    return fp.read(length).decode("utf-8", errors="replace")


def _read_value(fp: BinaryIO, val_type: int) -> Any:
    fmt = GGUF_VAL_TYPES.get(val_type)
    if fmt is None:
        raise ValueError(f"Unsupported GGUF metadata value type: {val_type}")
    if fmt == "str":
        return _read_string(fp)
    if fmt == "arr":
        elem_type = _read_u32(fp)
        count = _read_u64(fp)
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
        magic = _read_u32(fp)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic header in {path}")
        version = _read_u32(fp)
        n_tensors = _read_u64(fp)
        n_kv = _read_u64(fp)

        metadata: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_string(fp)
            val_type = _read_u32(fp)
            metadata[key] = _read_value(fp, val_type)

        bucket_elements: dict[str, int] = defaultdict(int)
        bucket_bytes: dict[str, float] = defaultdict(float)
        bucket_counts: dict[str, int] = defaultdict(int)
        qtype_counts: dict[str, int] = defaultdict(int)
        layer_indices: set[int] = set()
        sample_shapes: dict[str, list[int]] = {}

        for _ in range(n_tensors):
            t_name = _read_string(fp)
            n_dims = _read_u32(fp)
            dims = [_read_u64(fp) for _ in range(n_dims)]
            ggml_type = _read_u32(fp)
            _offset = _read_u64(fp)

            m = LAYER_RE.match(t_name)
            if m:
                layer_indices.add(int(m.group(1)))

            elems = 1
            for d in dims:
                elems *= d

            t_bytes = _estimate_bytes(elems, ggml_type)
            q_name = GGML_QUANT_SPECS.get(ggml_type, (f"T{ggml_type}", 1, 1))[0]
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
    total_elems = sum(bucket_elements.values())
    total_est_mb = sum(bucket_bytes.values()) / (1024 * 1024)

    return {
        "path": str(path),
        "file_mb": file_mb,
        "version": version,
        "architecture": arch,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "detected_layers": len(layer_indices),
        "metadata": metadata,
        "bucket_elements": dict(bucket_elements),
        "bucket_bytes": dict(bucket_bytes),
        "bucket_counts": dict(bucket_counts),
        "qtype_counts": dict(qtype_counts),
        "sample_shapes": sample_shapes,
        "total_elements": total_elems,
        "total_est_mb": total_est_mb,
    }


def inspect_gguf(path: Path, label: str | None = None) -> dict[str, Any]:
    """Parse and print a formatted architectural summary of a GGUF file."""
    info = parse_gguf_summary(path)
    title = f"{label} ({path})" if label else str(path)
    print(
        f"[{utc_now_iso()}] Inspecting GGUF: {title} "
        f"({info['file_mb']:.2f} MB)"
    )
    arch = info["architecture"]
    print("=" * 78)
    print(
        f"GGUF v{info['version']} | Arch: {arch} | "
        f"Tensors: {info['n_tensors']} | "
        f"Detected Layers: {info['detected_layers']}"
    )
    print("=" * 78)
    meta = info["metadata"]
    for k in sorted(meta):
        if k.startswith((f"{arch}.", "general.name", "general.file_type")):
            print(f"  {k:<40} : {meta[k]}")

    total_elems = info["total_elements"]
    print("-" * 78)
    print(
        f"{'Component Category':<35} | {'Cnt':>4} | {'Params(M)':>10} | "
        f"{'Size(MB)':>9} | {'Share':>6}"
    )
    print("-" * 78)
    for cat in sorted(info["bucket_elements"]):
        elems = info["bucket_elements"][cat]
        cnt = info["bucket_counts"][cat]
        mb = info["bucket_bytes"][cat] / (1024 * 1024)
        pct = (elems / total_elems * 100.0) if total_elems else 0.0
        print(
            f"{cat:<35} | {cnt:>4} | {elems / 1e6:>10.2f} | "
            f"{mb:>9.2f} | {pct:>5.1f}%"
        )
    print("-" * 78)
    print(
        f"{'TOTAL':<35} | {info['n_tensors']:>4} | "
        f"{total_elems / 1e6:>10.2f} | "
        f"{info['total_est_mb']:>9.2f} | 100.0%"
    )
    print("-" * 78)
    print("Layer 0 & Embedding Tensor Shapes (GGML dim order):")
    for t_name, dims in sorted(info["sample_shapes"].items()):
        print(f"  {t_name:<30} : {dims}")
    print(f"Quantization Block Distribution : {info['qtype_counts']}")
    print("=" * 78)
    return info


def compare_gguf_models(
    path_a: Path,
    label_a: str,
    path_b: Path,
    label_b: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Print individual reports and a side-by-side delta table."""
    info_a = inspect_gguf(path_a, label=label_a)
    print()
    info_b = inspect_gguf(path_b, label=label_b)
    print()
    print("=" * 78)
    print(f"SIDE-BY-SIDE COMPARISON: {label_a} vs. {label_b}")
    print("=" * 78)
    print(
        f"{'Metric':<32} | {label_a[:14]:>14} | "
        f"{label_b[:14]:>14} | {'Delta':>11}"
    )
    print("-" * 78)
    la, lb = info_a["detected_layers"], info_b["detected_layers"]
    pa = info_a["total_elements"] / 1e6
    pb = info_b["total_elements"] / 1e6
    ma, mb = info_a["file_mb"], info_b["file_mb"]
    ta, tb = info_a["n_tensors"], info_b["n_tensors"]

    print(
        f"{'Detected Transformer Layers':<32} | {la:>14} | "
        f"{lb:>14} | {lb - la:>+11}"
    )
    print(f"{'Total Tensors':<32} | {ta:>14} | {tb:>14} | {tb - ta:>+11}")
    print(
        f"{'Total Parameters (M)':<32} | {pa:>14.2f} | "
        f"{pb:>14.2f} | {pb - pa:>+11.2f}"
    )
    print(
        f"{'On-Disk File Size (MB)':<32} | {ma:>14.2f} | "
        f"{mb:>14.2f} | {mb - ma:>+11.2f}"
    )
    print("=" * 78)
    return info_a, info_b


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect local GGUF file or Ollama model tags."
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
        "--compare-ollama",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help="Compare two Ollama models side by side",
    )
    args = parser.parse_args(argv)

    try:
        if args.compare_ollama:
            mod_a, mod_b = args.compare_ollama
            blob_a = resolve_ollama_gguf_blob(mod_a)
            blob_b = resolve_ollama_gguf_blob(mod_b)
            compare_gguf_models(blob_a, mod_a, blob_b, mod_b)
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
