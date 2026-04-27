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


def plot_metric_timeseries(rows: List[Dict[str, str]], metric: str, title: str, ylabel: str, output_path: Path) -> None:
    by_condition: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for condition, condition_rows in by_condition.items():
        condition_rows.sort(key=lambda row: int(row["global_step"]))
        x = [int(row["global_step"]) for row in condition_rows]
        y = [float(row[metric]) for row in condition_rows]
        ax.plot(x, y, marker="o", linewidth=1.7, markersize=3.2, label=DISPLAY_NAMES.get(condition, condition))

    ax.set_title(title)
    ax.set_xlabel("Global step / batch")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved={output_path}")


def plot_layer_grid(rows: List[Dict[str, str]], metric: str, title_prefix: str, output_path: Path) -> None:
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
            y = [float(row[metric]) for row in condition_rows]
            ax.plot(x, y, marker="o", linewidth=1.3, markersize=2.5, label=DISPLAY_NAMES.get(condition, condition))
        ax.set_title(layer)
        ax.set_xlabel("Global step / batch")
        ax.set_ylabel(metric)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)

    for ax in flat_axes[len(layers) :]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        fig.subplots_adjust(top=0.93)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title_prefix)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved={output_path}")


def plot_individual_layers(rows: List[Dict[str, str]], metric: str, output_dir: Path) -> None:
    by_layer: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    for layer, layer_rows in sorted(by_layer.items()):
        plot_metric_timeseries(
            layer_rows,
            metric=metric,
            title=f"{metric} vs Clean Across Batches: {layer}",
            ylabel=metric,
            output_path=output_dir / f"{safe_filename(layer)}_{metric}_timeseries.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-batch gradient update coverage time series.")
    parser.add_argument("--timeseries-csv", type=Path, default=SCRIPT_DIR / "coverage_batch_timeseries.csv")
    parser.add_argument("--layer-csv", type=Path, default=SCRIPT_DIR / "coverage_layer_batch_timeseries.csv")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "plots")
    args = parser.parse_args()

    step_rows = read_csv(args.timeseries_csv)
    layer_rows = read_csv(args.layer_csv)
    if not step_rows:
        raise ValueError(f"No rows found in {args.timeseries_csv}")

    plot_metric_timeseries(
        step_rows,
        metric="same_coverage_ratio",
        title="Gradient Update Coverage vs Clean Across Batches",
        ylabel="Coverage: both nonzero / clean nonzero",
        output_path=args.output_dir / "coverage_batch_timeseries.png",
    )
    plot_metric_timeseries(
        step_rows,
        metric="jaccard_ratio",
        title="Gradient Update Jaccard vs Clean Across Batches",
        ylabel="Jaccard: both nonzero / either nonzero",
        output_path=args.output_dir / "jaccard_batch_timeseries.png",
    )
    plot_layer_grid(
        layer_rows,
        metric="same_coverage_ratio",
        title_prefix="Layer Gradient Update Coverage vs Clean Across Batches",
        output_path=args.output_dir / "coverage_layer_batch_timeseries_grid.png",
    )
    plot_layer_grid(
        layer_rows,
        metric="jaccard_ratio",
        title_prefix="Layer Gradient Update Jaccard vs Clean Across Batches",
        output_path=args.output_dir / "jaccard_layer_batch_timeseries_grid.png",
    )
    plot_individual_layers(layer_rows, metric="same_coverage_ratio", output_dir=args.output_dir / "layers" / "coverage")
    plot_individual_layers(layer_rows, metric="jaccard_ratio", output_dir=args.output_dir / "layers" / "jaccard")


if __name__ == "__main__":
    main()
