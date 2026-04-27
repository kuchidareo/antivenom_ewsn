from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gradient update-location masks from saved per-step gradient .pt files."
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Reference run directory containing per-step gradient .pt files.",
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
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Treat abs(grad) > eps as updated. Default is exact nonzero comparison.",
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


def mask_counts(ref: torch.Tensor, target: torch.Tensor, eps: float = 0.0) -> Dict[str, int]:
    if ref.shape != target.shape:
        raise ValueError(f"Gradient shape mismatch: ref={tuple(ref.shape)} target={tuple(target.shape)}")

    ref_values = ref.detach().cpu().float().reshape(-1)
    target_values = target.detach().cpu().float().reshape(-1)
    ref_mask = ref_values.abs() > eps
    target_mask = target_values.abs() > eps

    both_updated = ref_mask & target_mask
    either_updated = ref_mask | target_mask
    same_mask = ref_mask == target_mask

    return {
        "num_values": int(ref_mask.numel()),
        "ref_updated": int(ref_mask.sum().item()),
        "target_updated": int(target_mask.sum().item()),
        "both_updated": int(both_updated.sum().item()),
        "either_updated": int(either_updated.sum().item()),
        "same_mask": int(same_mask.sum().item()),
    }


def ratios_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    num_values = counts["num_values"]
    ref_updated = counts["ref_updated"]
    target_updated = counts["target_updated"]
    both_updated = counts["both_updated"]
    either_updated = counts["either_updated"]

    return {
        # Main coverage: how much of clean's update-location mask is also updated in target.
        "same_coverage_ratio": both_updated / ref_updated if ref_updated else math.nan,
        "target_coverage_ratio": both_updated / target_updated if target_updated else math.nan,
        "jaccard_ratio": both_updated / either_updated if either_updated else math.nan,
        "same_mask_ratio": counts["same_mask"] / num_values if num_values else math.nan,
        "ref_update_density": ref_updated / num_values if num_values else math.nan,
        "target_update_density": target_updated / num_values if num_values else math.nan,
    }


def add_counts(total: Dict[str, int], counts: Dict[str, int]) -> None:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value


def calculate_layer_step_coverages(
    ref_dir: Path,
    target_dir: Path,
    requested_layers: Optional[List[str]] = None,
    eps: float = 0.0,
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
            counts = mask_counts(ref_grads[layer_name], target_grads[layer_name], eps=eps)
            row: Dict[str, object] = {
                "step_file": file_name,
                "layer": layer_name,
                **counts,
                **ratios_from_counts(counts),
            }
            results.append(row)

    return results


def calculate_step_coverages(
    ref_dir: Path,
    target_dir: Path,
    requested_layers: Optional[List[str]] = None,
    eps: float = 0.0,
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

        total_counts: Dict[str, int] = {}
        for layer_name in layers:
            add_counts(total_counts, mask_counts(ref_grads[layer_name], target_grads[layer_name], eps=eps))

        row: Dict[str, object] = {
            "step_file": file_name,
            "num_layers": len(layers),
            **total_counts,
            **ratios_from_counts(total_counts),
        }
        results.append(row)

    return results


def compare_runs(
    ref_dir: Path,
    target_dir: Path,
    requested_layers: Optional[List[str]] = None,
    eps: float = 0.0,
) -> None:
    step_results = calculate_step_coverages(ref_dir, target_dir, requested_layers, eps=eps)
    layer_results = calculate_layer_step_coverages(ref_dir, target_dir, requested_layers, eps=eps)
    layers = sorted({str(row["layer"]) for row in layer_results})

    print(f"ref={ref_dir}")
    print(f"target={target_dir}")
    print(f"eps={eps}")
    print(f"matched_steps={len(step_results)}")
    print("step_file,same_coverage_ratio,jaccard_ratio,same_mask_ratio,ref_update_density,target_update_density")
    for row in step_results:
        print(
            f"{row['step_file']},"
            f"{float(row['same_coverage_ratio']):.4f},"
            f"{float(row['jaccard_ratio']):.4f},"
            f"{float(row['same_mask_ratio']):.4f},"
            f"{float(row['ref_update_density']):.4f},"
            f"{float(row['target_update_density']):.4f}"
        )

    print()
    print("per_layer_same_coverage_ratio")
    print(",".join(["step"] + layers))
    by_step_layer = {
        (str(row["step_file"]), str(row["layer"])): float(row["same_coverage_ratio"])
        for row in layer_results
    }
    for row in step_results:
        step_file = str(row["step_file"])
        step_name = step_file.replace("epoch_0000_", "").replace(".pt", "")
        values = [f"{by_step_layer[(step_file, layer)]:.4f}" for layer in layers]
        print(",".join([step_name] + values))


def main() -> None:
    args = parse_args()
    compare_runs(Path(args.ref), Path(args.target), args.layers, eps=float(args.eps))


if __name__ == "__main__":
    main()
