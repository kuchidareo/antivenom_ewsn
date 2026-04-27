from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved per-step gradient .pt files from ml_running.py."
    )
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Reference run directory containing per-step gradient .pt files.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
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


def validate_args(args: argparse.Namespace) -> None:
    if args.ref is None or args.target is None:
        raise ValueError("Provide both --ref and --target run directories.")


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


def select_layers(grads: Dict[str, torch.Tensor], requested_layers: Optional[List[str]]) -> List[str]:
    if requested_layers:
        missing = [name for name in requested_layers if name not in grads]
        if missing:
            raise KeyError(f"Requested layers missing from gradient file: {missing}")
        return requested_layers
    return sorted(grads)


def squared_error_sum(ref: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    ref_values = ref.detach().cpu().float().reshape(-1)
    target_values = target.detach().cpu().float().reshape(-1)
    if ref_values.shape != target_values.shape:
        raise ValueError(f"Gradient shape mismatch: ref={tuple(ref.shape)} target={tuple(target.shape)}")

    diff = target_values - ref_values
    return float(torch.sum(diff * diff).item()), int(diff.numel())


def compare_runs(
    ref_dir: Path,
    target_dir: Path,
    requested_layers: Optional[List[str]],
) -> None:
    step_results = calculate_step_rmses(ref_dir, target_dir, requested_layers)

    print(f"ref={ref_dir}")
    print(f"target={target_dir}")
    print(f"matched_steps={len(step_results)}")
    print("step_file,rmse,num_values,num_layers")

    for result in step_results:
        print(
            f"{result['step_file']},"
            f"{result['rmse']:.10g},"
            f"{result['num_values']},"
            f"{result['num_layers']}"
        )

    mean_rmse = sum(float(result["rmse"]) for result in step_results) / len(step_results)
    print(f"mean_step_rmse={mean_rmse:.10g}")


def calculate_layer_step_rmses(
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
            squared_error, num_values = squared_error_sum(ref_grads[layer_name], target_grads[layer_name])
            rmse = math.sqrt(squared_error / num_values) if num_values else math.nan
            results.append(
                {
                    "step_file": file_name,
                    "layer": layer_name,
                    "rmse": rmse,
                    "num_values": num_values,
                }
            )

    return results


def calculate_step_rmses(
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

        total_squared_error = 0.0
        total_values = 0
        for layer_name in layers:
            squared_error, num_values = squared_error_sum(ref_grads[layer_name], target_grads[layer_name])
            total_squared_error += squared_error
            total_values += num_values

        rmse = math.sqrt(total_squared_error / total_values) if total_values else math.nan
        results.append(
            {
                "step_file": file_name,
                "rmse": rmse,
                "num_values": total_values,
                "num_layers": len(layers),
            }
        )

    return results


def main() -> None:
    args = parse_args()
    validate_args(args)

    compare_runs(Path(args.ref), Path(args.target), args.layers)


if __name__ == "__main__":
    main()
