from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

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
CONDITION_COLORS = {
    "clean": "#1f2937",
    "augmentation": "#1f77b4",
    "ood": "#ff7f0e",
    "blurring": "#2ca02c",
    "label-flip": "#d62728",
    "steganography": "#9467bd",
    "occlusion": "#8c564b",
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
    run_dirs = [
        path
        for path in condition_dir.iterdir()
        if path.is_dir() and any(path.glob("epoch_*_step_*.pt"))
    ]
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
        raise ValueError(f"Unexpected gradient filename: {name}")
    return int(match.group("epoch")), int(match.group("step"))


def list_step_files(run_dir: Path, epochs: Optional[Sequence[int]]) -> Dict[str, Path]:
    wanted = set(int(epoch) for epoch in epochs) if epochs else None
    files: Dict[str, Path] = {}
    for path in run_dir.glob("epoch_*_step_*.pt"):
        epoch, _ = parse_step_name(path.name)
        if wanted is None or epoch in wanted:
            files[path.name] = path
    return files


def is_parameter_grad_name(name: str) -> bool:
    return "." in name and not name.startswith("input")


def load_parameter_grads(pt_file: Path) -> Dict[str, torch.Tensor]:
    try:
        loaded = torch.load(pt_file, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(pt_file, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict in {pt_file}, got {type(loaded)}")
    grads: Dict[str, torch.Tensor] = {}
    for name, value in loaded.items():
        key = str(name)
        if isinstance(value, torch.Tensor) and is_parameter_grad_name(key):
            tensor = value.detach().cpu()
            grads[key] = tensor if tensor.dtype == torch.float32 else tensor.float()
    return grads


def layer_key(parameter_name: str) -> str:
    return parameter_name.rsplit(".", 1)[0]


def layer_vectors_from_grads(grads: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    by_layer: Dict[str, List[torch.Tensor]] = {}
    for name in sorted(grads):
        by_layer.setdefault(layer_key(name), []).append(grads[name].reshape(-1))
    return {layer: torch.cat(values).numpy() for layer, values in by_layer.items()}


def parse_run_overrides(values: Iterable[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONDITION=PATH, got {value}")
        condition, path = value.split("=", 1)
        runs[canonical_condition(condition.strip())] = Path(path).expanduser()
    return runs


def safe_filename(name: str) -> str:
    return name.replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


def finite_or_empty(value: float) -> object:
    if math.isnan(value) or math.isinf(value):
        return ""
    return value


def padded_axis_limits(coords: np.ndarray, padding_fraction: float = 0.08) -> Tuple[float, float, float, float]:
    if coords.size == 0:
        return (-1.0, 1.0, -1.0, 1.0)

    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))
    y_min = float(np.min(coords[:, 1]))
    y_max = float(np.max(coords[:, 1]))
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span == 0.0:
        x_span = max(abs(x_min), 1.0)
    if y_span == 0.0:
        y_span = max(abs(y_min), 1.0)
    return (
        x_min - x_span * padding_fraction,
        x_max + x_span * padding_fraction,
        y_min - y_span * padding_fraction,
        y_max + y_span * padding_fraction,
    )


def pca_2d_matrix(matrix: np.ndarray, block_size: int = 64) -> Tuple[np.ndarray, Tuple[float, float]]:
    n_samples = int(matrix.shape[0])
    if n_samples == 0:
        return np.zeros((0, 2), dtype=np.float32), (0.0, 0.0)

    gram = np.empty((n_samples, n_samples), dtype=np.float64)

    for row_start in range(0, n_samples, block_size):
        row_end = min(row_start + block_size, n_samples)
        row_block = np.asarray(matrix[row_start:row_end], dtype=np.float32)
        for col_start in range(0, n_samples, block_size):
            col_end = min(col_start + block_size, n_samples)
            col_block = np.asarray(matrix[col_start:col_end], dtype=np.float32)
            gram[row_start:row_end, col_start:col_end] = row_block @ col_block.T

    gram = (gram + gram.T) * 0.5
    mean_dot = gram.sum(axis=1) / float(n_samples)
    mean_norm = float(gram.sum()) / float(n_samples * n_samples)
    gram = gram - mean_dot[:, None] - mean_dot[None, :] + mean_norm
    gram = (gram + gram.T) * 0.5
    total_variance = float(np.trace(gram))
    if total_variance <= 0.0:
        return np.zeros((n_samples, 2), dtype=np.float32), (0.0, 0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    component_count = min(2, n_samples)
    scores = eigenvectors[:, :component_count] * np.sqrt(eigenvalues[:component_count])
    if scores.shape[1] < 2:
        scores = np.pad(scores, ((0, 0), (0, 2 - scores.shape[1])), mode="constant")
    explained = eigenvalues / total_variance
    pc1 = float(explained[0]) if explained.size >= 1 else 0.0
    pc2 = float(explained[1]) if explained.size >= 2 else 0.0
    return scores[:, :2].astype(np.float32, copy=False), (pc1, pc2)


def pca_2d(vectors: Sequence[np.ndarray]) -> Tuple[np.ndarray, Tuple[float, float]]:
    matrix = np.stack([np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors])
    return pca_2d_matrix(matrix)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_layer_step_pca(
    points: Sequence[Dict[str, object]],
    layer: str,
    epoch: int,
    global_step: int,
    explained_variance: Tuple[float, float],
    output_path: Path,
    dpi: int,
    title_suffix: str = "",
    axis_limits: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for point in points:
        condition = str(point["condition"])
        x = float(point["pc1"])
        y = float(point["pc2"])
        color = CONDITION_COLORS.get(condition, "#7f7f7f")
        ax.scatter(x, y, s=72, color=color, edgecolor="white", linewidth=0.8, label=condition, zorder=3)
        ax.annotate(condition, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)

    ax.axhline(0.0, color="#d0d0d0", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.8, zorder=1)
    ax.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    suffix = f" {title_suffix}" if title_suffix else ""
    ax.set_title(f"{layer} PCA by condition{suffix} | epoch={epoch} step={global_step}")
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100.0:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100.0:.1f}% var)")
    if axis_limits is not None:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_layer_all_steps_pca(
    points: Sequence[Dict[str, object]],
    layer: str,
    explained_variance: Tuple[float, float],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    conditions = list(dict.fromkeys(str(point["condition"]) for point in points))

    for condition in conditions:
        condition_points = [point for point in points if point["condition"] == condition]
        condition_points.sort(key=lambda point: (int(point["global_step"]), int(point["epoch"])))
        xs = [float(point["pc1"]) for point in condition_points]
        ys = [float(point["pc2"]) for point in condition_points]
        color = CONDITION_COLORS.get(condition, "#7f7f7f")
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.35, zorder=2)
        ax.scatter(
            xs,
            ys,
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.90,
            label=condition,
            zorder=3,
        )

    ax.axhline(0.0, color="#d0d0d0", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.8, zorder=1)
    ax.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_title(f"{layer} PCA across all selected steps")
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100.0:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100.0:.1f}% var)")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def resolve_runs(
    root: Path,
    requested_conditions: Sequence[str],
    run_overrides: Dict[str, Path],
) -> Dict[str, Path]:
    conditions = list(dict.fromkeys(canonical_condition(condition) for condition in requested_conditions))
    run_dirs: Dict[str, Path] = {}
    for condition in conditions:
        run_dir = run_overrides.get(condition) or resolve_run_dir(root, condition)
        if run_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print(f"skipped {condition}: no .pt run found under {aliases}")
            continue
        run_dirs[condition] = run_dir.resolve()
    if len(run_dirs) < 2:
        raise ValueError("Need at least two condition runs for PCA.")
    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="For each layer and batch step, run 2D PCA across condition gradient vectors."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--epochs", nargs="*", type=int, default=[0, 1, 8, 9])
    parser.add_argument("--runs", nargs="*", default=[], help="Optional CONDITION=RUN_DIR overrides.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "plots")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--skip-step-plots",
        action="store_true",
        help="Do not render the per-layer per-step PCA figures.",
    )
    parser.add_argument(
        "--skip-all-steps-plots",
        action="store_true",
        help="Do not render the per-layer PCA figures that include all selected steps.",
    )
    parser.add_argument(
        "--skip-shared-step-plots",
        action="store_true",
        help="Do not render per-step figures projected onto each layer's all-steps PCA axes.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    run_overrides = parse_run_overrides(args.runs)
    run_dirs = resolve_runs(root, args.conditions, run_overrides)
    conditions = [canonical_condition(condition) for condition in args.conditions if canonical_condition(condition) in run_dirs]

    step_files_by_condition = {
        condition: list_step_files(run_dir, args.epochs) for condition, run_dir in run_dirs.items()
    }
    common_step_names = sorted(
        set.intersection(*(set(files.keys()) for files in step_files_by_condition.values())),
        key=parse_step_name,
    )
    if not common_step_names:
        raise ValueError("No common step files found across selected conditions/epochs.")

    first_grads = load_parameter_grads(step_files_by_condition[conditions[0]][common_step_names[0]])
    first_vectors = layer_vectors_from_grads(first_grads)
    layer_names = sorted(first_vectors.keys())
    layer_dims = {layer: int(first_vectors[layer].size) for layer in layer_names}
    del first_vectors
    del first_grads

    print(f"conditions={conditions}")
    print(f"epochs={args.epochs}")
    print(f"common_steps={len(common_step_names)}")
    print(f"layers={layer_names}")
    print("collecting layer-step gradient vectors")

    point_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    all_steps_rows: List[Dict[str, object]] = []
    all_steps_manifest_rows: List[Dict[str, object]] = []
    shared_step_manifest_rows: List[Dict[str, object]] = []
    num_all_step_samples = len(common_step_names) * len(conditions)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="layer_step_pca_", dir=output_dir) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        all_step_vectors_by_layer: Dict[str, np.memmap] = {}
        all_step_samples_by_layer: Dict[str, List[Dict[str, object]]] = {}
        if not args.skip_all_steps_plots:
            all_step_samples_by_layer = {layer: [] for layer in layer_names}
            all_step_vectors_by_layer = {
                layer: np.memmap(
                    tmp_dir / f"{safe_filename(layer)}.dat",
                    dtype=np.float32,
                    mode="w+",
                    shape=(num_all_step_samples, layer_dims[layer]),
                )
                for layer in layer_names
            }

        for step_index, step_name in enumerate(common_step_names):
            epoch, global_step = parse_step_name(step_name)
            vectors_by_condition: Dict[str, Dict[str, np.ndarray]] = {}
            for condition in conditions:
                grads = load_parameter_grads(step_files_by_condition[condition][step_name])
                vectors_by_condition[condition] = layer_vectors_from_grads(grads)
                del grads

            for layer in layer_names:
                vectors = [vectors_by_condition[condition][layer] for condition in conditions]
                if not args.skip_all_steps_plots:
                    for condition_index, condition in enumerate(conditions):
                        sample_index = step_index * len(conditions) + condition_index
                        all_step_vectors_by_layer[layer][sample_index] = vectors_by_condition[condition][layer]
                        all_step_samples_by_layer[layer].append(
                            {
                                "condition": condition,
                                "run_dir": str(run_dirs[condition]),
                                "epoch": epoch,
                                "global_step": global_step,
                                "step_file": step_name,
                                "layer": layer,
                            }
                        )

                if args.skip_step_plots:
                    continue

                coords, explained_variance = pca_2d(vectors)
                layer_points: List[Dict[str, object]] = []
                for condition, coord in zip(conditions, coords):
                    row = {
                        "condition": condition,
                        "run_dir": str(run_dirs[condition]),
                        "epoch": epoch,
                        "global_step": global_step,
                        "step_file": step_name,
                        "layer": layer,
                        "pc1": finite_or_empty(float(coord[0])),
                        "pc2": finite_or_empty(float(coord[1])),
                        "pc1_explained_variance_ratio": explained_variance[0],
                        "pc2_explained_variance_ratio": explained_variance[1],
                    }
                    point_rows.append(row)
                    layer_points.append(row)

                plot_path = output_dir / "layers" / safe_filename(layer) / f"{step_name.replace('.pt', '')}_pca2d.png"
                plot_layer_step_pca(
                    points=layer_points,
                    layer=layer,
                    epoch=epoch,
                    global_step=global_step,
                    explained_variance=explained_variance,
                    output_path=plot_path,
                    dpi=int(args.dpi),
                )
                manifest_rows.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "step_file": step_name,
                        "layer": layer,
                        "path": str(plot_path),
                    }
                )

            del vectors_by_condition

        if not args.skip_all_steps_plots:
            for memmap in all_step_vectors_by_layer.values():
                memmap.flush()

            for layer in layer_names:
                samples = all_step_samples_by_layer[layer]
                coords, explained_variance = pca_2d_matrix(all_step_vectors_by_layer[layer])
                layer_points: List[Dict[str, object]] = []
                layer_points_by_step: Dict[Tuple[int, int, str], List[Dict[str, object]]] = {}
                for sample, coord in zip(samples, coords):
                    row = {
                        "condition": sample["condition"],
                        "run_dir": sample["run_dir"],
                        "epoch": sample["epoch"],
                        "global_step": sample["global_step"],
                        "step_file": sample["step_file"],
                        "layer": sample["layer"],
                        "pc1": finite_or_empty(float(coord[0])),
                        "pc2": finite_or_empty(float(coord[1])),
                        "pc1_explained_variance_ratio": explained_variance[0],
                        "pc2_explained_variance_ratio": explained_variance[1],
                    }
                    all_steps_rows.append(row)
                    layer_points.append(row)
                    step_key = (int(sample["epoch"]), int(sample["global_step"]), str(sample["step_file"]))
                    layer_points_by_step.setdefault(step_key, []).append(row)

                plot_path = output_dir / "layer_all_steps" / f"{safe_filename(layer)}_all_steps_pca2d.png"
                plot_layer_all_steps_pca(
                    points=layer_points,
                    layer=layer,
                    explained_variance=explained_variance,
                    output_path=plot_path,
                    dpi=int(args.dpi),
                )
                all_steps_manifest_rows.append(
                    {
                        "layer": layer,
                        "num_points": len(layer_points),
                        "num_steps": len(common_step_names),
                        "path": str(plot_path),
                    }
                )

                if not args.skip_shared_step_plots:
                    axis_limits = padded_axis_limits(coords)
                    for (epoch, global_step, step_file), step_points in layer_points_by_step.items():
                        shared_plot_path = (
                            output_dir
                            / "layer_shared_axis_steps"
                            / safe_filename(layer)
                            / f"{step_file.replace('.pt', '')}_shared_pca2d.png"
                        )
                        plot_layer_step_pca(
                            points=step_points,
                            layer=layer,
                            epoch=epoch,
                            global_step=global_step,
                            explained_variance=explained_variance,
                            output_path=shared_plot_path,
                            dpi=int(args.dpi),
                            title_suffix="shared all-steps axes",
                            axis_limits=axis_limits,
                        )
                        shared_step_manifest_rows.append(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "step_file": step_file,
                                "layer": layer,
                                "path": str(shared_plot_path),
                            }
                        )

            all_step_vectors_by_layer.clear()

    write_csv(SCRIPT_DIR / "layer_step_pca_2d_points.csv", point_rows)
    write_csv(SCRIPT_DIR / "plot_manifest.csv", manifest_rows)
    write_csv(SCRIPT_DIR / "layer_all_steps_pca_2d_points.csv", all_steps_rows)
    write_csv(SCRIPT_DIR / "layer_all_steps_plot_manifest.csv", all_steps_manifest_rows)
    write_csv(SCRIPT_DIR / "layer_shared_axis_step_plot_manifest.csv", shared_step_manifest_rows)
    print(f"saved={SCRIPT_DIR / 'layer_step_pca_2d_points.csv'}")
    print(f"saved={SCRIPT_DIR / 'plot_manifest.csv'}")
    print(f"saved={SCRIPT_DIR / 'layer_all_steps_pca_2d_points.csv'}")
    print(f"saved={SCRIPT_DIR / 'layer_all_steps_plot_manifest.csv'}")
    print(f"saved={SCRIPT_DIR / 'layer_shared_axis_step_plot_manifest.csv'}")
    print(f"saved_step_plots={len(manifest_rows)}")
    print(f"saved_all_steps_plots={len(all_steps_manifest_rows)}")
    print(f"saved_shared_step_plots={len(shared_step_manifest_rows)}")


if __name__ == "__main__":
    main()
