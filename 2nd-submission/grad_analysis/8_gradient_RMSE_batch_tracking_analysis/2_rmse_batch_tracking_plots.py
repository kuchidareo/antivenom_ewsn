from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DISPLAY_NAMES = {
    "clean": "clean",
    "augmentation": "augmentation",
    "data_augmentation": "augmentation",
    "ood": "ood",
    "blurring": "blurring",
    "label-flip": "label-flip",
    "steganography": "steganography",
    "occlusion": "occlusion",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_filename(name: str) -> str:
    return name.replace(".", "_").replace("-", "_").replace("/", "_")


def plot_global_timeseries(rows: List[Dict[str, str]], output_path: Path) -> None:
    by_condition: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for condition, condition_rows in by_condition.items():
        condition_rows.sort(key=lambda row: int(row["global_step"]))
        x = [int(row["global_step"]) for row in condition_rows]
        y = [float(row["rmse"]) for row in condition_rows]
        ax.plot(x, y, marker="o", linewidth=1.7, markersize=3.2, label=DISPLAY_NAMES.get(condition, condition))

    ax.set_title("Gradient RMSE vs Clean Across Batches")
    ax.set_xlabel("Global step / batch")
    ax.set_ylabel("All-parameter RMSE")
    ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved={output_path}")


def plot_layer_grid(rows: List[Dict[str, str]], output_path: Path) -> None:
    by_layer_condition: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_layer_condition[row["layer"]][row["condition"]].append(row)

    layers = sorted(by_layer_condition)
    if not layers:
        return

    ncols = 3
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows), squeeze=False)
    flat_axes = list(axes.reshape(-1))

    for ax, layer in zip(flat_axes, layers):
        for condition, condition_rows in sorted(by_layer_condition[layer].items()):
            condition_rows.sort(key=lambda row: int(row["global_step"]))
            x = [int(row["global_step"]) for row in condition_rows]
            y = [float(row["rmse"]) for row in condition_rows]
            ax.plot(x, y, marker="o", linewidth=1.3, markersize=2.5, label=DISPLAY_NAMES.get(condition, condition))
        ax.set_title(layer)
        ax.set_xlabel("Global step / batch")
        ax.set_ylabel("RMSE")
        ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)

    for ax in flat_axes[len(layers) :]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.subplots_adjust(top=0.93)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved={output_path}")


def plot_individual_layers(rows: List[Dict[str, str]], output_dir: Path) -> None:
    by_layer: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    for layer, layer_rows in sorted(by_layer.items()):
        plot_global_timeseries(layer_rows, output_dir / f"{safe_filename(layer)}_rmse_timeseries.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-batch gradient RMSE time series.")
    parser.add_argument("--timeseries-csv", type=Path, default=SCRIPT_DIR / "rmse_batch_timeseries.csv")
    parser.add_argument("--layer-csv", type=Path, default=SCRIPT_DIR / "rmse_layer_batch_timeseries.csv")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "plots")
    args = parser.parse_args()

    step_rows = read_csv(args.timeseries_csv)
    layer_rows = read_csv(args.layer_csv)
    if not step_rows:
        raise ValueError(f"No rows found in {args.timeseries_csv}")

    plot_global_timeseries(step_rows, args.output_dir / "rmse_batch_timeseries.png")
    plot_layer_grid(layer_rows, args.output_dir / "rmse_layer_batch_timeseries_grid.png")
    plot_individual_layers(layer_rows, args.output_dir / "layers")


if __name__ == "__main__":
    main()
