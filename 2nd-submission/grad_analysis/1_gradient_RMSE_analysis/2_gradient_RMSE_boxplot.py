from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from gradient_step_analysis import calculate_layer_step_rmses, calculate_step_rmses


REF_CONDITION = "clean"
TARGET_CONDITIONS = [
    "clean",
    "data_augmentation",
    "ood",
    "blurring",
    "label-flip",
    "steganography",
    "occlusion",
]
CONDITION_ALIASES = {
    "data_augmentation": ["data_augmentation", "augmentation"],
}
DISPLAY_NAMES = {
    "clean": "clean",
    "data_augmentation": "augmentation",
    "ood": "ood",
    "blurring": "blurring",
    "label-flip": "label-flip",
    "steganography": "steganography",
    "occlusion": "occlusion",
}

FLOPS = {
    "conv1.weight": 693_633_024,
    "conv2.weight": 3_699_376_128,
    "conv3.weight": 3_699_376_128,
    "fc1.weight": 411_041_792,
    "fc2.weight": 524_288,
    "fc3.weight": 12_288,
}


def candidate_condition_dirs(root: Path, condition: str) -> List[Path]:
    names = CONDITION_ALIASES.get(condition, [condition])
    return [root / name for name in names]


def latest_run_dir(condition_dir: Path) -> Optional[Path]:
    if not condition_dir.exists() or not condition_dir.is_dir():
        return None

    run_dirs = [
        path
        for path in condition_dir.iterdir()
        if path.is_dir() and any(path.glob("*.pt"))
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


def safe_filename(name: str) -> str:
    return name.replace(".", "_").replace("-", "_").replace("/", "_")


def flop_weighted_step_rmses(layer_rows: List[dict]) -> List[float]:
    by_step: Dict[str, Dict[str, float]] = {}
    for row in layer_rows:
        layer = str(row["layer"])
        if layer not in FLOPS:
            continue
        by_step.setdefault(str(row["step_file"]), {})[layer] = float(row["rmse"])

    weighted_values: List[float] = []
    total_flops = float(sum(FLOPS.values()))
    for step_file in sorted(by_step):
        step_values = by_step[step_file]
        if not step_values:
            continue
        weighted_mse = 0.0
        used_flops = 0.0
        for layer, flops in FLOPS.items():
            if layer not in step_values:
                continue
            weighted_mse += float(flops) * (step_values[layer] ** 2)
            used_flops += float(flops)
        if used_flops:
            # RMSE is already a square-root metric, so combine layers by
            # FLOP-weighted MSE and take the square root once.
            weighted_values.append(math.sqrt(weighted_mse / total_flops))
    return weighted_values


def collect_rmse_values(root: Path) -> tuple[Dict[str, List[float]], Dict[str, Dict[str, List[float]]], Dict[str, List[float]]]:
    ref_dir = resolve_run_dir(root, REF_CONDITION)
    if ref_dir is None:
        raise FileNotFoundError(f"No run directory with .pt files found for {REF_CONDITION}")

    print(f"reference_condition={REF_CONDITION}")
    print(f"reference_run={ref_dir}")

    all_layer_values: Dict[str, List[float]] = {}
    per_layer_values: Dict[str, Dict[str, List[float]]] = {}
    flop_weighted_values: Dict[str, List[float]] = {}

    for condition in TARGET_CONDITIONS:
        target_dir = ref_dir if condition == REF_CONDITION else resolve_run_dir(root, condition)
        if target_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print(f"skipped {condition}: no run directory with .pt files found under {aliases}")
            continue

        step_rows = calculate_step_rmses(ref_dir=ref_dir, target_dir=target_dir)
        layer_rows = calculate_layer_step_rmses(ref_dir=ref_dir, target_dir=target_dir)

        all_layer_values[condition] = [float(row["rmse"]) for row in step_rows]
        flop_weighted_values[condition] = flop_weighted_step_rmses(layer_rows)

        for row in layer_rows:
            layer = str(row["layer"])
            per_layer_values.setdefault(layer, {}).setdefault(condition, []).append(float(row["rmse"]))

        print(f"{condition}: target={target_dir}, steps={len(step_rows)}")

    return all_layer_values, per_layer_values, flop_weighted_values


def style_boxplot(boxplot: Dict[str, object], color: str) -> None:
    for box in boxplot["boxes"]:
        box.set(facecolor=color, edgecolor="#222222", linewidth=1.0)
    for median in boxplot["medians"]:
        median.set(color="#111111", linewidth=1.6)
    for whisker in boxplot["whiskers"]:
        whisker.set(color="#444444", linewidth=1.0)
    for cap in boxplot["caps"]:
        cap.set(color="#444444", linewidth=1.0)
    for flier in boxplot["fliers"]:
        flier.set(marker="o", markersize=3, markerfacecolor=color, markeredgecolor="#555555", alpha=0.65)


def plot_boxplot(
    values_by_condition: Dict[str, List[float]],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str = "#9fb7d7",
    ax: Optional[plt.Axes] = None,
) -> None:
    labels = [DISPLAY_NAMES[condition] for condition in values_by_condition]
    values = [values_by_condition[condition] for condition in values_by_condition]
    owns_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 5.8))

    boxplot = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    style_boxplot(boxplot, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="x", labelrotation=25)

    if owns_figure:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"saved={output_path}")


def plot_layer_grid(
    per_layer_values: Dict[str, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    layers = sorted(per_layer_values)
    ncols = 3
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.7 * nrows))
    flat_axes = list(axes.reshape(-1)) if hasattr(axes, "reshape") else [axes]

    for ax, layer in zip(flat_axes, layers):
        plot_boxplot(
            per_layer_values[layer],
            title=layer,
            ylabel="RMSE",
            output_path=output_path,
            color="#9fc7b0",
            ax=ax,
        )

    for ax in flat_axes[len(layers) :]:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved={output_path}")


def plot_all(
    all_layer_values: Dict[str, List[float]],
    per_layer_values: Dict[str, Dict[str, List[float]]],
    flop_weighted_values: Dict[str, List[float]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / "layers"
    layer_dir.mkdir(parents=True, exist_ok=True)

    plot_boxplot(
        all_layer_values,
        title="Step-Level All-Layer Gradient RMSE vs Clean",
        ylabel="RMSE",
        output_path=output_dir / "rmse_all_layers_boxplot.png",
        color="#9fb7d7",
    )
    plot_boxplot(
        flop_weighted_values,
        title="Step-Level FLOP-Weighted Gradient RMSE vs Clean",
        ylabel="FLOP-weighted RMSE",
        output_path=output_dir / "rmse_flop_weighted_boxplot.png",
        color="#d7b48f",
    )

    for layer, values_by_condition in sorted(per_layer_values.items()):
        plot_boxplot(
            values_by_condition,
            title=f"Step-Level Gradient RMSE vs Clean: {layer}",
            ylabel="RMSE",
            output_path=layer_dir / f"{safe_filename(layer)}_rmse_boxplot.png",
            color="#9fc7b0",
        )

    plot_layer_grid(per_layer_values, output_dir / "rmse_layer_boxplots.png")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    output_dir = script_dir / "plots"
    all_layer_values, per_layer_values, flop_weighted_values = collect_rmse_values(root)
    if not all_layer_values:
        raise ValueError("No RMSE values were collected")
    plot_all(all_layer_values, per_layer_values, flop_weighted_values, output_dir)


if __name__ == "__main__":
    main()
