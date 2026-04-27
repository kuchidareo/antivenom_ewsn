from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_LAYER_ORDER = [
    "conv1.weight",
    "conv2.weight",
    "conv3.weight",
    "fc1.weight",
    "fc2.weight",
    "fc3.weight",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize saved gradient weights as 2D heatmaps.")
    p.add_argument("--pt-file", type=str, default=None, help="Path to a saved gradient .pt file")
    p.add_argument("--ref-pt-file", type=str, default=None, help="Reference gradient .pt file for comparison mode")
    p.add_argument("--target-pt-file", type=str, default=None, help="Target gradient .pt file for comparison mode")
    p.add_argument("--ref-pt-dir", type=str, default=None, help="Reference directory containing gradient .pt files")
    p.add_argument("--target-pt-dir", type=str, default=None, help="Target directory containing gradient .pt files")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output plots. Defaults to a path derived from the input .pt file(s)",
    )
    p.add_argument(
        "--layers",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit layer names to plot. Defaults to conv1/2/3.weight and fc1/2/3.weight when present.",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum number of plotted points per layer. Larger tensors are uniformly subsampled.",
    )
    p.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="Unused legacy argument kept for compatibility",
    )
    p.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="Unused legacy argument kept for compatibility",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    has_single = args.pt_file is not None
    has_compare_files = args.ref_pt_file is not None or args.target_pt_file is not None
    has_compare_dirs = args.ref_pt_dir is not None or args.target_pt_dir is not None
    has_compare = has_compare_files or has_compare_dirs

    if has_single and has_compare:
        raise ValueError(
            "Use either --pt-file or comparison inputs (--ref-pt-file/--target-pt-file or --ref-pt-dir/--target-pt-dir), not both"
        )
    if has_single:
        return
    if has_compare_files and has_compare_dirs:
        raise ValueError("Use either the file pair or the directory pair for comparison mode, not both")
    if has_compare_files:
        if args.ref_pt_file is None or args.target_pt_file is None:
            raise ValueError("Comparison file mode requires both --ref-pt-file and --target-pt-file")
        return
    if has_compare_dirs:
        if args.ref_pt_dir is None or args.target_pt_dir is None:
            raise ValueError("Comparison directory mode requires both --ref-pt-dir and --target-pt-dir")
        return
    raise ValueError(
        "Provide either --pt-file, --ref-pt-file/--target-pt-file, or --ref-pt-dir/--target-pt-dir"
    )


def load_grads(pt_file: Path) -> Dict[str, torch.Tensor]:
    grads = torch.load(pt_file, map_location="cpu")
    if not isinstance(grads, dict):
        raise TypeError(f"Expected dict in {pt_file}, got {type(grads)}")
    tensor_grads = {str(k): v for k, v in grads.items() if isinstance(v, torch.Tensor)}
    if not tensor_grads:
        raise ValueError(f"No tensor gradients found in {pt_file}")
    return tensor_grads


def choose_layers(grads: Dict[str, torch.Tensor], requested_layers: List[str] | None) -> List[str]:
    if requested_layers:
        missing = [name for name in requested_layers if name not in grads]
        if missing:
            raise KeyError(f"Requested layers not found in pt file: {missing}")
        return requested_layers

    present_default = [name for name in DEFAULT_LAYER_ORDER if name in grads]
    if present_default:
        return present_default

    return [name for name in sorted(grads) if name.endswith(".weight")]


def choose_common_layers(
    ref_grads: Dict[str, torch.Tensor],
    target_grads: Dict[str, torch.Tensor],
    requested_layers: List[str] | None,
) -> List[str]:
    if requested_layers:
        missing = [name for name in requested_layers if name not in ref_grads or name not in target_grads]
        if missing:
            raise KeyError(f"Requested layers not found in both pt files: {missing}")
        return requested_layers

    present_default = [name for name in DEFAULT_LAYER_ORDER if name in ref_grads and name in target_grads]
    if present_default:
        return present_default

    return [name for name in sorted(set(ref_grads) & set(target_grads)) if name.endswith(".weight")]


def grad_to_matrix(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().float().numpy()
    if arr.ndim == 2:
        return arr
    if arr.ndim == 4:
        # Conv weights: keep output-channel axis and flatten the remaining axes into one position axis.
        return arr.reshape(arr.shape[0], -1)
    raise ValueError(f"Unsupported weight shape {tuple(arr.shape)}; expected 2D fc or 4D conv weight")


def plot_layer(
    layer_name: str,
    mat: np.ndarray,
    out_path: Path,
    elev: float,
    azim: float,
    max_points: int,
    title_suffix: str = "gradient weights",
    z_label: str = "Gradient value",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_title(f"{layer_name} {title_suffix}")
    ax.set_xlabel("Flattened position")
    ax.set_ylabel("Output channel / row")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=z_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def sanitize_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def extract_step_label(pt_file: Path) -> str:
    stem = pt_file.stem
    if stem:
        return stem
    return "step"


def resolve_output_dir(args: argparse.Namespace, target_path: Path, ref_path: Path) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = target_path.with_name(f"{target_path.stem}_minus_{ref_path.stem}_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def render_comparison_pair(
    ref_pt_file: Path,
    target_pt_file: Path,
    output_dir: Path,
    requested_layers: List[str] | None,
    elev: float,
    azim: float,
    max_points: int,
) -> None:
    if not ref_pt_file.exists():
        raise FileNotFoundError(f"Reference gradient pt file not found: {ref_pt_file}")
    if not target_pt_file.exists():
        raise FileNotFoundError(f"Target gradient pt file not found: {target_pt_file}")

    ref_grads = load_grads(ref_pt_file)
    target_grads = load_grads(target_pt_file)
    layers = choose_common_layers(ref_grads, target_grads, requested_layers)

    print("mode=compare")
    print(f"ref_pt_file={ref_pt_file}")
    print(f"target_pt_file={target_pt_file}")
    print(f"output_dir={output_dir}")
    print(f"layers={layers}")

    step_label = extract_step_label(target_pt_file)

    for layer_name in layers:
        ref_mat = grad_to_matrix(ref_grads[layer_name])
        target_mat = grad_to_matrix(target_grads[layer_name])
        if ref_mat.shape != target_mat.shape:
            raise ValueError(
                f"Shape mismatch for {layer_name}: ref={tuple(ref_mat.shape)} target={tuple(target_mat.shape)}"
            )
        diff_mat = target_mat - ref_mat

        layer_dir_name = sanitize_name(layer_name)

        ref_out_path = output_dir / "ref" / layer_dir_name / f"{step_label}_2d.png"
        ref_out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_layer(
            layer_name=layer_name,
            mat=ref_mat,
            out_path=ref_out_path,
            elev=elev,
            azim=azim,
            max_points=max_points,
            title_suffix="reference gradient weights",
            z_label="Reference gradient value",
        )
        print(f"saved={ref_out_path} shape={tuple(ref_mat.shape)}")

        target_out_path = output_dir / "target" / layer_dir_name / f"{step_label}_2d.png"
        target_out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_layer(
            layer_name=layer_name,
            mat=target_mat,
            out_path=target_out_path,
            elev=elev,
            azim=azim,
            max_points=max_points,
            title_suffix="target gradient weights",
            z_label="Target gradient value",
        )
        print(f"saved={target_out_path} shape={tuple(target_mat.shape)}")

        diff_out_path = output_dir / "diff" / layer_dir_name / f"{step_label}_2d.png"
        diff_out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_layer(
            layer_name=layer_name,
            mat=diff_mat,
            out_path=diff_out_path,
            elev=elev,
            azim=azim,
            max_points=max_points,
            title_suffix="gradient difference (target - ref)",
            z_label="Gradient difference",
        )
        print(f"saved={diff_out_path} shape={tuple(diff_mat.shape)}")


def list_pt_files(pt_dir: Path) -> Dict[str, Path]:
    if not pt_dir.exists():
        raise FileNotFoundError(f"Gradient pt directory not found: {pt_dir}")
    if not pt_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {pt_dir}")
    files = {p.name: p for p in sorted(pt_dir.glob("*.pt"))}
    if not files:
        raise ValueError(f"No .pt files found in {pt_dir}")
    return files


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.pt_file is not None:
        pt_file = Path(args.pt_file)
        if not pt_file.exists():
            raise FileNotFoundError(f"Gradient pt file not found: {pt_file}")

        output_dir = Path(args.output_dir) if args.output_dir else pt_file.with_name(f"{pt_file.stem}_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        grads = load_grads(pt_file)
        layers = choose_layers(grads, args.layers)

        print(f"mode=single")
        print(f"pt_file={pt_file}")
        print(f"output_dir={output_dir}")
        print(f"layers={layers}")

        for layer_name in layers:
            mat = grad_to_matrix(grads[layer_name])
            out_path = output_dir / f"{sanitize_name(layer_name)}_2d.png"
            plot_layer(
                layer_name=layer_name,
                mat=mat,
                out_path=out_path,
                elev=float(args.elev),
                azim=float(args.azim),
                max_points=int(args.max_points),
            )
            print(f"saved={out_path} shape={tuple(mat.shape)}")
        return

    if args.ref_pt_file is not None or args.target_pt_file is not None:
        ref_pt_file = Path(args.ref_pt_file)
        target_pt_file = Path(args.target_pt_file)
        output_dir = resolve_output_dir(args, target_pt_file, ref_pt_file)
        render_comparison_pair(
            ref_pt_file=ref_pt_file,
            target_pt_file=target_pt_file,
            output_dir=output_dir,
            requested_layers=args.layers,
            elev=float(args.elev),
            azim=float(args.azim),
            max_points=int(args.max_points),
        )
        return

    ref_pt_dir = Path(args.ref_pt_dir)
    target_pt_dir = Path(args.target_pt_dir)
    ref_files = list_pt_files(ref_pt_dir)
    target_files = list_pt_files(target_pt_dir)
    common_names = sorted(set(ref_files) & set(target_files))
    if not common_names:
        raise ValueError(f"No matching .pt filenames between {ref_pt_dir} and {target_pt_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else target_pt_dir / "comparison_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("mode=compare_dir")
    print(f"ref_pt_dir={ref_pt_dir}")
    print(f"target_pt_dir={target_pt_dir}")
    print(f"matched_files={len(common_names)}")
    print(f"output_dir={output_dir}")

    for name in common_names:
        render_comparison_pair(
            ref_pt_file=ref_files[name],
            target_pt_file=target_files[name],
            output_dir=output_dir,
            requested_layers=args.layers,
            elev=float(args.elev),
            azim=float(args.azim),
            max_points=int(args.max_points),
        )


if __name__ == "__main__":
    main()
