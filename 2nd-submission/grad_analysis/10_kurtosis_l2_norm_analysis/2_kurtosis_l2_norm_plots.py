from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONDITIONS = [
    "clean",
    "augmentation",
    "ood",
    "blurring",
    "label-flip",
    "steganography",
    "occlusion",
]
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
METRICS = {
    "grad_excess_kurtosis": {
        "title": "Layer Signed Gradient Excess Kurtosis Across Batches",
        "ylabel": "Excess kurtosis",
        "csv": "kurtosis_layer_batch_timeseries.csv",
        "grid": "grad_excess_kurtosis_layer_batch_timeseries_grid.png",
        "subdir": "grad_excess_kurtosis",
    },
    "abs_grad_excess_kurtosis": {
        "title": "Layer |Gradient| Excess Kurtosis Across Batches",
        "ylabel": "|g| excess kurtosis",
        "csv": "kurtosis_layer_batch_timeseries.csv",
        "grid": "abs_grad_excess_kurtosis_layer_batch_timeseries_grid.png",
        "subdir": "abs_grad_excess_kurtosis",
    },
    "l2_norm": {
        "title": "Layer Gradient L2 Norm Across Batches",
        "ylabel": "L2 norm",
        "csv": "l2_layer_batch_timeseries.csv",
        "grid": "l2_norm_layer_batch_timeseries_grid.png",
        "subdir": "l2_norm",
    },
}


def canonical_condition(condition: str) -> str:
    if condition == "data_augmentation":
        return "augmentation"
    return condition


def safe_filename(name: str) -> str:
    return name.replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def ordered_conditions(present: Sequence[str], requested: Sequence[str]) -> List[str]:
    present_ordered = list(dict.fromkeys(canonical_condition(condition) for condition in present))
    requested_ordered = [canonical_condition(condition) for condition in requested]
    ordered = [condition for condition in requested_ordered if condition in present_ordered]
    ordered.extend(condition for condition in present_ordered if condition not in ordered)
    return ordered


def group_by_layer_condition(
    rows: Sequence[Dict[str, str]],
    metric: str,
    requested_conditions: Sequence[str],
) -> Dict[str, Dict[str, List[tuple[int, float]]]]:
    grouped: Dict[str, Dict[str, List[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        layer = row.get("layer", "")
        condition = canonical_condition(row.get("condition", ""))
        step = parse_float(row.get("global_step"))
        value = parse_float(row.get(metric))
        if not layer or step is None or value is None:
            continue
        grouped[layer][condition].append((int(step), value))

    requested = [canonical_condition(condition) for condition in requested_conditions]
    for layer in list(grouped):
        ordered: Dict[str, List[tuple[int, float]]] = {}
        for condition in ordered_conditions(list(grouped[layer]), requested):
            ordered[condition] = sorted(grouped[layer][condition])
        grouped[layer] = ordered
    return grouped


def plot_layer_grid(
    rows: Sequence[Dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    requested_conditions: Sequence[str],
) -> None:
    by_layer_condition = group_by_layer_condition(rows, metric, requested_conditions)
    layers = sorted(by_layer_condition)
    if not layers:
        raise ValueError(f"No rows available for metric={metric}")

    ncols = 3
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.6 * nrows), squeeze=False)
    flat_axes = list(axes.reshape(-1))

    for ax, layer in zip(flat_axes, layers):
        for condition, points in by_layer_condition[layer].items():
            if not points:
                continue
            steps = [point[0] for point in points]
            values = [point[1] for point in points]
            ax.plot(
                steps,
                values,
                marker="o",
                linewidth=1.4,
                markersize=2.6,
                label=DISPLAY_NAMES.get(condition, condition),
            )
        ax.set_title(layer)
        ax.set_xlabel("Global step / batch")
        ax.set_ylabel(ylabel)
        ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)

    for ax in flat_axes[len(layers) :]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved={output_path}")


def plot_individual_layers(
    rows: Sequence[Dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
    output_dir: Path,
    requested_conditions: Sequence[str],
) -> None:
    by_layer_condition = group_by_layer_condition(rows, metric, requested_conditions)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer, by_condition in sorted(by_layer_condition.items()):
        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        for condition, points in by_condition.items():
            if not points:
                continue
            steps = [point[0] for point in points]
            values = [point[1] for point in points]
            ax.plot(
                steps,
                values,
                marker="o",
                linewidth=1.6,
                markersize=3.0,
                label=DISPLAY_NAMES.get(condition, condition),
            )
        ax.set_title(f"{title}: {layer}")
        ax.set_xlabel("Global step / batch")
        ax.set_ylabel(ylabel)
        ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.legend(loc="best")
        fig.tight_layout()
        output_path = output_dir / f"{safe_filename(layer)}_{metric}_timeseries.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-batch kurtosis and L2 norm time series.")
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "plots")
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    cached_rows: Dict[str, List[Dict[str, str]]] = {}

    for metric, settings in METRICS.items():
        csv_name = str(settings["csv"])
        if csv_name not in cached_rows:
            cached_rows[csv_name] = read_csv_rows(input_dir / csv_name)
        rows = cached_rows[csv_name]
        plot_layer_grid(
            rows=rows,
            metric=metric,
            title=str(settings["title"]),
            ylabel=str(settings["ylabel"]),
            output_path=output_dir / str(settings["grid"]),
            requested_conditions=args.conditions,
        )
        plot_individual_layers(
            rows=rows,
            metric=metric,
            title=str(settings["title"]),
            ylabel=str(settings["ylabel"]),
            output_dir=output_dir / "layers" / str(settings["subdir"]),
            requested_conditions=args.conditions,
        )


if __name__ == "__main__":
    main()
