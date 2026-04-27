from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved per-step gradients against a trusted reference gradient direction."
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Trusted/reference run directory containing per-step gradient .pt files.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target run directory containing per-step gradient .pt files.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="*",
        default=None,
        help="Optional layer names to include. Defaults to all common tensor gradients.",
    )
    return parser.parse_args()


def list_pt_files(pt_dir: Path) -> Dict[str, Path]:
    if not pt_dir.exists():
        raise FileNotFoundError(f"Gradient directory not found: {pt_dir}")
    if not pt_dir.is_dir():
        raise NotADirectoryError(f"Expected directory: {pt_dir}")

    files = {path.name: path for path in sorted(pt_dir.glob("*.pt"))}
    if not files:
        raise ValueError(f"No .pt files found in {pt_dir}")
    return files


def load_grads(pt_file: Path) -> Dict[str, torch.Tensor]:
    grads = torch.load(pt_file, map_location="cpu")
    if not isinstance(grads, dict):
        raise TypeError(f"Expected dict in {pt_file}, got {type(grads)}")
    return {str(name): value for name, value in grads.items() if isinstance(value, torch.Tensor)}


def alignment_stats(ref: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    if ref.shape != target.shape:
        raise ValueError(f"Gradient shape mismatch: ref={tuple(ref.shape)} target={tuple(target.shape)}")

    ref_values = ref.detach().cpu().float().reshape(-1)
    target_values = target.detach().cpu().float().reshape(-1)

    dot_product = float(torch.dot(ref_values, target_values).item())
    ref_norm = float(torch.linalg.vector_norm(ref_values).item())
    target_norm = float(torch.linalg.vector_norm(target_values).item())
    cosine_similarity = dot_product / (ref_norm * target_norm) if ref_norm and target_norm else math.nan

    ref_sign = torch.sign(ref_values)
    target_sign = torch.sign(target_values)
    sign_compared = (ref_sign != 0) | (target_sign != 0)
    sign_total = int(sign_compared.sum().item())
    sign_same = int((ref_sign[sign_compared] == target_sign[sign_compared]).sum().item())
    sign_agreement = sign_same / sign_total if sign_total else math.nan

    return {
        "num_values": float(ref_values.numel()),
        "dot_product": dot_product,
        "ref_norm": ref_norm,
        "target_norm": target_norm,
        "cosine_similarity": cosine_similarity,
        "sign_agreement": sign_agreement,
    }


def add_alignment(total: Dict[str, float], stats: Dict[str, float]) -> None:
    total["dot_product"] = total.get("dot_product", 0.0) + stats["dot_product"]
    total["ref_norm_sq"] = total.get("ref_norm_sq", 0.0) + stats["ref_norm"] ** 2
    total["target_norm_sq"] = total.get("target_norm_sq", 0.0) + stats["target_norm"] ** 2
    total["num_values"] = total.get("num_values", 0.0) + stats["num_values"]


def add_sign_counts(total: Dict[str, int], ref: torch.Tensor, target: torch.Tensor) -> None:
    ref_values = ref.detach().cpu().float().reshape(-1)
    target_values = target.detach().cpu().float().reshape(-1)
    ref_sign = torch.sign(ref_values)
    target_sign = torch.sign(target_values)
    sign_compared = (ref_sign != 0) | (target_sign != 0)
    total["sign_total"] = total.get("sign_total", 0) + int(sign_compared.sum().item())
    total["sign_same"] = total.get("sign_same", 0) + int(
        (ref_sign[sign_compared] == target_sign[sign_compared]).sum().item()
    )


def finalize_alignment(total: Dict[str, float], sign_counts: Dict[str, int]) -> Dict[str, float]:
    ref_norm = math.sqrt(total.get("ref_norm_sq", 0.0))
    target_norm = math.sqrt(total.get("target_norm_sq", 0.0))
    dot_product = total.get("dot_product", 0.0)
    sign_total = sign_counts.get("sign_total", 0)

    return {
        "num_values": total.get("num_values", 0.0),
        "dot_product": dot_product,
        "ref_norm": ref_norm,
        "target_norm": target_norm,
        "cosine_similarity": dot_product / (ref_norm * target_norm) if ref_norm and target_norm else math.nan,
        "sign_agreement": sign_counts.get("sign_same", 0) / sign_total if sign_total else math.nan,
    }


def calculate_layer_step_alignments(
    ref_dir: Path,
    target_dir: Path,
    requested_layers: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    ref_files = list_pt_files(ref_dir)
    target_files = list_pt_files(target_dir)
    common_names = sorted(set(ref_files) & set(target_files))
    if not common_names:
        raise ValueError(f"No matching .pt filenames between {ref_dir} and {target_dir}")

    results: List[Dict[str, object]] = []
    for file_name in common_names:
        ref_grads = load_grads(ref_files[file_name])
        target_grads = load_grads(target_files[file_name])
        common_layers = sorted(set(ref_grads) & set(target_grads))
        layers = requested_layers if requested_layers else common_layers

        missing = [name for name in layers if name not in ref_grads or name not in target_grads]
        if missing:
            raise KeyError(f"Requested layers missing from one side for {file_name}: {missing}")

        for layer_name in layers:
            row: Dict[str, object] = {
                "step_file": file_name,
                "layer": layer_name,
                **alignment_stats(ref_grads[layer_name], target_grads[layer_name]),
            }
            results.append(row)

    return results


def calculate_step_alignments(
    ref_dir: Path,
    target_dir: Path,
    requested_layers: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    ref_files = list_pt_files(ref_dir)
    target_files = list_pt_files(target_dir)
    common_names = sorted(set(ref_files) & set(target_files))
    if not common_names:
        raise ValueError(f"No matching .pt filenames between {ref_dir} and {target_dir}")

    results: List[Dict[str, object]] = []
    for file_name in common_names:
        ref_grads = load_grads(ref_files[file_name])
        target_grads = load_grads(target_files[file_name])
        common_layers = sorted(set(ref_grads) & set(target_grads))
        layers = requested_layers if requested_layers else common_layers

        missing = [name for name in layers if name not in ref_grads or name not in target_grads]
        if missing:
            raise KeyError(f"Requested layers missing from one side for {file_name}: {missing}")

        total: Dict[str, float] = {}
        sign_counts: Dict[str, int] = {}
        for layer_name in layers:
            ref_grad = ref_grads[layer_name]
            target_grad = target_grads[layer_name]
            add_alignment(total, alignment_stats(ref_grad, target_grad))
            add_sign_counts(sign_counts, ref_grad, target_grad)

        row: Dict[str, object] = {
            "step_file": file_name,
            "num_layers": len(layers),
            **finalize_alignment(total, sign_counts),
        }
        results.append(row)

    return results


def compare_runs(
    ref_dir: Path,
    target_dir: Path,
    requested_layers: Optional[List[str]] = None,
) -> None:
    step_results = calculate_step_alignments(ref_dir, target_dir, requested_layers)
    layer_results = calculate_layer_step_alignments(ref_dir, target_dir, requested_layers)
    layers = sorted({str(row["layer"]) for row in layer_results})

    print(f"ref={ref_dir}")
    print(f"target={target_dir}")
    print(f"matched_steps={len(step_results)}")
    print("step_file,cosine_similarity,dot_product,sign_agreement,ref_norm,target_norm")
    for row in step_results:
        print(
            f"{row['step_file']},"
            f"{float(row['cosine_similarity']):.4f},"
            f"{float(row['dot_product']):.4f},"
            f"{float(row['sign_agreement']):.4f},"
            f"{float(row['ref_norm']):.4f},"
            f"{float(row['target_norm']):.4f}"
        )

    print()
    print("per_layer_cosine_similarity")
    print(",".join(["step"] + layers))
    by_step_layer = {
        (str(row["step_file"]), str(row["layer"])): float(row["cosine_similarity"])
        for row in layer_results
    }
    for row in step_results:
        step_file = str(row["step_file"])
        step_name = step_file.replace("epoch_0000_", "").replace(".pt", "")
        values = [f"{by_step_layer[(step_file, layer)]:.4f}" for layer in layers]
        print(",".join([step_name] + values))


def main() -> None:
    args = parse_args()
    compare_runs(Path(args.ref), Path(args.target), args.layers)


if __name__ == "__main__":
    main()
