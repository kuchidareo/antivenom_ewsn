from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DEFAULT_CONDITIONS = [
    "clean",
    "augmentation",
    "ood",
    "blurring",
    "label-flip",
    "steganography",
    "occlusion",
]
CONDITION_ALIASES = {
    "augmentation": ["augmentation", "data_augmentation"],
    "data_augmentation": ["data_augmentation", "augmentation"],
}
STEP_RE = re.compile(r"epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.pt$")


def canonical_condition(condition: str) -> str:
    if condition == "data_augmentation":
        return "augmentation"
    return condition


def candidate_condition_dirs(root: Path, condition: str) -> List[Path]:
    names = CONDITION_ALIASES.get(condition, [condition])
    return [root / name for name in names]


def latest_run_dir(condition_dir: Path) -> Optional[Path]:
    if not condition_dir.exists() or not condition_dir.is_dir():
        return None
    run_dirs = [path for path in condition_dir.iterdir() if path.is_dir() and any(path.glob("epoch_*_step_*.pt"))]
    if not run_dirs:
        return None
    return sorted(run_dirs)[-1]


def resolve_run_dir(root: Path, condition: str) -> Optional[Path]:
    for condition_dir in candidate_condition_dirs(root, condition):
        run_dir = latest_run_dir(condition_dir)
        if run_dir is not None:
            return run_dir
    return None


def parse_step_name(name: str) -> Tuple[int, int]:
    match = STEP_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected step file name: {name}")
    return int(match.group("epoch")), int(match.group("step"))


def list_step_files(run_dir: Path, epochs: Sequence[int]) -> Dict[str, Path]:
    wanted = set(int(epoch) for epoch in epochs)
    files: Dict[str, Path] = {}
    for path in run_dir.glob("epoch_*_step_*.pt"):
        epoch, _ = parse_step_name(path.name)
        if epoch in wanted:
            files[path.name] = path
    return files


def load_parameter_grads(pt_file: Path) -> Dict[str, torch.Tensor]:
    loaded = torch.load(pt_file, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict in {pt_file}, got {type(loaded)}")
    grads: Dict[str, torch.Tensor] = {}
    for name, value in loaded.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(name)
        if "." not in key or key.startswith("input"):
            continue
        grads[key] = value.detach().cpu().float()
    return grads


def layer_key(parameter_name: str) -> str:
    return parameter_name.rsplit(".", 1)[0]


def tensor_to_matrix(values: torch.Tensor) -> np.ndarray:
    array = values.numpy()
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    return array.reshape(array.shape[0], -1)


def stack_layer_parameter_matrices(parameter_mats: List[np.ndarray]) -> np.ndarray:
    if not parameter_mats:
        raise ValueError("No parameter matrices to stack")
    max_width = max(mat.shape[1] for mat in parameter_mats)
    padded: List[np.ndarray] = []
    for mat in parameter_mats:
        if mat.shape[1] == max_width:
            padded.append(mat)
            continue
        out = np.full((mat.shape[0], max_width), np.nan, dtype=np.float32)
        out[:, : mat.shape[1]] = mat
        padded.append(out)
    return np.vstack(padded)


def layer_matrices_from_grads(grads: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    per_layer: Dict[str, List[np.ndarray]] = {}
    for name in sorted(grads):
        layer = layer_key(name)
        per_layer.setdefault(layer, []).append(tensor_to_matrix(grads[name]))
    return {layer: stack_layer_parameter_matrices(mats) for layer, mats in per_layer.items()}


def finite_absmax(array: np.ndarray) -> float:
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return 0.0
    return float(np.max(np.abs(valid)))


def nrows_ncols(count: int, max_cols: int = 4) -> Tuple[int, int]:
    ncols = min(max_cols, max(1, count))
    nrows = int(math.ceil(count / ncols))
    return nrows, ncols


def plot_grid(
    matrices_by_label: Sequence[Tuple[str, np.ndarray]],
    title: str,
    output_path: Path,
    vabs_by_label: Dict[str, float],
    cmap: str,
    dpi: int,
) -> None:
    nrows, ncols = nrows_ncols(len(matrices_by_label))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
    flat_axes = list(axes.reshape(-1))

    for ax, (label, matrix) in zip(flat_axes, matrices_by_label):
        safe_vabs = max(float(vabs_by_label[label]), 1e-12)
        masked = np.ma.masked_invalid(matrix)
        ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-safe_vabs, vmax=safe_vabs, interpolation="nearest")
        ax.set_title(f"{label}\nscale=±{safe_vabs:.3g}")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in flat_axes[len(matrices_by_label) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04, wspace=0.10, hspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def shared_scales(items: Sequence[Tuple[str, np.ndarray]]) -> Dict[str, float]:
    vabs = max((finite_absmax(matrix) for _, matrix in items), default=0.0)
    return {label: vabs for label, _ in items}


def per_condition_scales(items: Sequence[Tuple[str, np.ndarray]]) -> Dict[str, float]:
    return {label: finite_absmax(matrix) for label, matrix in items}


def scale_by_mode(items: Sequence[Tuple[str, np.ndarray]], scale_mode: str) -> Dict[str, float]:
    if scale_mode == "per-condition":
        return per_condition_scales(items)
    return shared_scales(items)


def parse_run_overrides(values: Iterable[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONDITION=PATH, got {value}")
        condition, path = value.split("=", 1)
        runs[canonical_condition(condition.strip())] = Path(path).expanduser()
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize per-layer 2D gradient heatmaps across conditions.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--epochs", nargs="*", type=int, default=[0, 1, 8, 9])
    parser.add_argument("--runs", nargs="*", default=[], help="Optional CONDITION=RUN_DIR overrides.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "plots")
    parser.add_argument("--cmap", type=str, default="coolwarm")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--scale-mode",
        choices=["shared", "per-condition"],
        default="shared",
        help="shared uses one scale for every subplot in a figure; per-condition gives each scenario subplot its own scale.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    requested_conditions = [canonical_condition(name) for name in args.conditions]
    requested_conditions = list(dict.fromkeys(requested_conditions))
    if "clean" not in requested_conditions:
        requested_conditions = ["clean"] + requested_conditions

    run_overrides = parse_run_overrides(args.runs)
    run_dirs: Dict[str, Path] = {}
    for condition in requested_conditions:
        run_dir = run_overrides.get(condition) or resolve_run_dir(root, condition)
        if run_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print(f"skipped {condition}: no .pt run found under {aliases}")
            continue
        run_dirs[condition] = run_dir.resolve()

    if "clean" not in run_dirs:
        raise FileNotFoundError("Clean run is required for diff visualization but was not found.")
    conditions = [condition for condition in requested_conditions if condition in run_dirs]
    non_clean = [condition for condition in conditions if condition != "clean"]
    if len(conditions) < 2:
        raise ValueError("Need at least clean plus one target condition.")

    step_files_by_condition: Dict[str, Dict[str, Path]] = {
        condition: list_step_files(run_dirs[condition], args.epochs) for condition in conditions
    }
    common_step_names = sorted(
        set.intersection(*(set(files.keys()) for files in step_files_by_condition.values())),
        key=parse_step_name,
    )
    if not common_step_names:
        raise ValueError("No common step files found across selected conditions/epochs.")

    first_clean_grads = load_parameter_grads(step_files_by_condition["clean"][common_step_names[0]])
    layer_names = sorted(layer_matrices_from_grads(first_clean_grads).keys())

    print(f"conditions={conditions}")
    print(f"epochs={args.epochs}")
    print(f"common_steps={len(common_step_names)}")
    print(f"scale_mode={args.scale_mode}")
    print("rendering raw and diff heatmaps")

    scale_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    for step_name in common_step_names:
        epoch, step = parse_step_name(step_name)
        clean_grads = load_parameter_grads(step_files_by_condition["clean"][step_name])
        clean_layers = layer_matrices_from_grads(clean_grads)
        by_condition_layers: Dict[str, Dict[str, np.ndarray]] = {"clean": clean_layers}
        for condition in non_clean:
            target_grads = load_parameter_grads(step_files_by_condition[condition][step_name])
            by_condition_layers[condition] = layer_matrices_from_grads(target_grads)

        for layer in layer_names:
            raw_items = [(condition, by_condition_layers[condition][layer]) for condition in conditions]
            raw_scales = scale_by_mode(raw_items, args.scale_mode)
            raw_path = output_dir / "raw" / layer / f"{step_name.replace('.pt', '')}_raw.png"
            plot_grid(
                matrices_by_label=raw_items,
                title=f"{layer} raw gradients | epoch={epoch} step={step}",
                output_path=raw_path,
                vabs_by_label=raw_scales,
                cmap=args.cmap,
                dpi=int(args.dpi),
            )
            manifest_rows.append(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "step_file": step_name,
                    "layer": layer,
                    "plot_type": "raw",
                    "path": str(raw_path),
                }
            )

            diff_items = []
            clean_mat = by_condition_layers["clean"][layer]
            for condition in non_clean:
                diff_items.append((f"{condition} - clean", by_condition_layers[condition][layer] - clean_mat))
            diff_scales = scale_by_mode(diff_items, args.scale_mode)
            diff_path = output_dir / "diff_from_clean" / layer / f"{step_name.replace('.pt', '')}_diff.png"
            plot_grid(
                matrices_by_label=diff_items,
                title=f"{layer} gradient diff from clean | epoch={epoch} step={step}",
                output_path=diff_path,
                vabs_by_label=diff_scales,
                cmap=args.cmap,
                dpi=int(args.dpi),
            )
            manifest_rows.append(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "step_file": step_name,
                    "layer": layer,
                    "plot_type": "diff_from_clean",
                    "path": str(diff_path),
                }
            )
            scale_rows.append(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "step_file": step_name,
                    "layer": layer,
                    "scale_mode": args.scale_mode,
                    "raw_absmax_scale_by_subplot": repr(raw_scales),
                    "diff_absmax_scale_by_subplot": repr(diff_scales),
                }
            )

    write_csv(output_dir / "figure_adaptive_scales.csv", scale_rows)
    write_csv(output_dir / "plot_manifest.csv", manifest_rows)
    print(f"saved={output_dir / 'figure_adaptive_scales.csv'}")
    print(f"saved={output_dir / 'plot_manifest.csv'}")
    print(f"saved_plots={len(manifest_rows)}")


if __name__ == "__main__":
    main()
