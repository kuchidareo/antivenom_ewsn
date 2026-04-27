from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from global_grad_norm_temporal_variance_analysis import (
    DEFAULT_CONDITIONS,
    analyze_conditions,
    canonical_condition,
    write_rows,
)


DISPLAY_NAMES = {
    "clean": "clean",
    "clean_ref": "clean_ref",
    "augmentation": "augmentation",
    "data_augmentation": "augmentation",
    "ood": "ood",
    "blurring": "blurring",
    "label-flip": "label-flip",
    "steganography": "steganography",
    "occlusion": "occlusion",
}
METRICS = {
    "batch_loss": {
        "title": "Batch Loss",
        "ylabel": "Loss",
        "filename": "batch_loss_boxplot.png",
        "color": "#8fb4d8",
    },
    "global_grad_norm": {
        "title": "Per-Batch Global Gradient Norm",
        "ylabel": "G_t = ||grad L_t||_2",
        "filename": "global_grad_norm_boxplot.png",
        "color": "#8fb9a8",
    },
    "global_grad_norm_variance": {
        "title": "Temporal Variance of Global Gradient Norm",
        "ylabel": "Rolling Var(G_t)",
        "filename": "global_grad_norm_temporal_variance_boxplot.png",
        "color": "#d7b48f",
    },
    "global_grad_norm_std": {
        "title": "Temporal Std of Global Gradient Norm",
        "ylabel": "Rolling Std(G_t)",
        "filename": "global_grad_norm_temporal_std_boxplot.png",
        "color": "#b6a3d9",
    },
}


def safe_filename(name: str) -> str:
    return name.replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


def parse_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def read_rows(csv_path: Path) -> List[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def ordered_conditions(present: Iterable[str], requested: Sequence[str]) -> List[str]:
    requested_order = [canonical_condition(condition) for condition in requested]
    present_list = list(dict.fromkeys(canonical_condition(condition) for condition in present))
    ordered = [condition for condition in requested_order if condition in present_list]
    ordered.extend(condition for condition in present_list if condition not in ordered)
    return ordered


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


def collect_metric_values(
    rows: Sequence[dict],
    metric: str,
    group_by: str,
    requested_conditions: Sequence[str],
    scope: Optional[str] = None,
    layer: Optional[str] = None,
) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {}
    for row in rows:
        if scope is not None and row.get("scope") != scope:
            continue
        if layer is not None and row.get("layer") != layer:
            continue
        value = parse_float(row.get(metric))
        if value is None:
            continue
        condition = canonical_condition(str(row.get("condition", "")))
        if group_by == "condition":
            key = condition
        elif group_by == "epoch":
            key = f"epoch {row.get('epoch')}"
        else:
            key = f"{DISPLAY_NAMES.get(condition, condition)} epoch {row.get('epoch')}"
        values.setdefault(key, []).append(value)

    if group_by == "condition":
        ordered = ordered_conditions(values, requested_conditions)
        return {condition: values[condition] for condition in ordered if condition in values}
    if group_by == "epoch":
        return dict(sorted(values.items(), key=lambda item: int(item[0].split()[-1])))
    return dict(sorted(values.items()))


def plot_boxplot(
    values_by_group: Dict[str, List[float]],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> None:
    if not values_by_group:
        return

    labels = list(values_by_group)
    values = [values_by_group[label] for label in labels]
    _, ax = plt.subplots(figsize=(max(10.5, 0.7 * len(labels)), 5.8))

    boxplot = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    style_boxplot(boxplot, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="x", labelrotation=25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"saved={output_path}")


def rows_by_condition(
    rows: Sequence[dict],
    requested_conditions: Sequence[str],
    scope: Optional[str] = None,
    layer: Optional[str] = None,
) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        if scope is not None and row.get("scope") != scope:
            continue
        if layer is not None and row.get("layer") != layer:
            continue
        condition = canonical_condition(str(row.get("condition", "")))
        grouped.setdefault(condition, []).append(row)
    ordered = ordered_conditions(grouped, requested_conditions)
    return {
        condition: sorted(grouped[condition], key=lambda row: int(row.get("global_step", 0)))
        for condition in ordered
        if condition in grouped
    }


def plot_timeseries(
    rows: Sequence[dict],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    requested_conditions: Sequence[str],
    scope: Optional[str] = None,
    layer: Optional[str] = None,
) -> None:
    grouped = rows_by_condition(rows, requested_conditions, scope=scope, layer=layer)
    if not grouped:
        return

    _, ax = plt.subplots(figsize=(12, 6))
    for condition, condition_rows in grouped.items():
        points = [
            (int(row["global_step"]), parse_float(row.get(metric)))
            for row in condition_rows
            if parse_float(row.get(metric)) is not None
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, linewidth=1.5, label=DISPLAY_NAMES.get(condition, condition), alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Global step")
    ax.set_ylabel(ylabel)
    ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="best", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"saved={output_path}")


def collect_layers(rows: Sequence[dict]) -> List[str]:
    return sorted({str(row.get("layer")) for row in rows if row.get("scope") == "layer" and row.get("layer")})


def plot_layer_boxplots(
    rows: Sequence[dict],
    metric: str,
    settings: dict,
    layers: Sequence[str],
    output_dir: Path,
    group_by: str,
    requested_conditions: Sequence[str],
) -> None:
    metric_dir = output_dir / "layers" / metric
    for layer in layers:
        values = collect_metric_values(
            rows=rows,
            metric=metric,
            group_by=group_by,
            requested_conditions=requested_conditions,
            scope="layer",
            layer=layer,
        )
        plot_boxplot(
            values_by_group=values,
            title=f"{settings['title']}: {layer} by {group_by}",
            ylabel=str(settings["ylabel"]),
            output_path=metric_dir / f"{safe_filename(layer)}_{settings['filename']}",
            color=str(settings["color"]),
        )


def plot_layer_grid(
    rows: Sequence[dict],
    metric: str,
    settings: dict,
    layers: Sequence[str],
    output_path: Path,
    group_by: str,
    requested_conditions: Sequence[str],
) -> None:
    if not layers:
        return

    ncols = 3
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.7 * nrows))
    flat_axes = list(axes.reshape(-1)) if hasattr(axes, "reshape") else [axes]

    for ax, layer in zip(flat_axes, layers):
        values = collect_metric_values(
            rows=rows,
            metric=metric,
            group_by=group_by,
            requested_conditions=requested_conditions,
            scope="layer",
            layer=layer,
        )
        if not values:
            ax.axis("off")
            continue

        labels = list(values)
        boxplot = ax.boxplot([values[label] for label in labels], tick_labels=labels, patch_artist=True, showmeans=True)
        style_boxplot(boxplot, color=str(settings["color"]))
        ax.set_title(f"{settings['title']}: {layer}")
        ax.set_ylabel(str(settings["ylabel"]))
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.tick_params(axis="x", labelrotation=25)

    for ax in flat_axes[len(layers) :]:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved={output_path}")


def plot_layer_timeseries(
    rows: Sequence[dict],
    metric: str,
    settings: dict,
    layers: Sequence[str],
    output_dir: Path,
    requested_conditions: Sequence[str],
) -> None:
    metric_dir = output_dir / "layers" / f"{metric}_timeseries"
    for layer in layers:
        plot_timeseries(
            rows=rows,
            metric=metric,
            title=f"{settings['title']} Over Time: {layer}",
            ylabel=str(settings["ylabel"]),
            output_path=metric_dir / f"{safe_filename(layer)}_{metric}_timeseries.png",
            requested_conditions=requested_conditions,
            scope="layer",
            layer=layer,
        )


def plot_all(
    rows: Sequence[dict],
    output_dir: Path,
    group_by: str,
    requested_conditions: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, settings in METRICS.items():
        values = collect_metric_values(
            rows=rows,
            metric=metric,
            group_by=group_by,
            requested_conditions=requested_conditions,
            scope="global",
        )
        plot_boxplot(
            values_by_group=values,
            title=f"{settings['title']} by {group_by}",
            ylabel=str(settings["ylabel"]),
            output_path=output_dir / str(settings["filename"]),
            color=str(settings["color"]),
        )

    plot_timeseries(
        rows=rows,
        metric="global_grad_norm",
        title="Per-Batch Global Gradient Norm Over Time",
        ylabel="G_t = ||grad L_t||_2",
        output_path=output_dir / "global_grad_norm_timeseries.png",
        requested_conditions=requested_conditions,
        scope="global",
    )
    plot_timeseries(
        rows=rows,
        metric="global_grad_norm_variance",
        title="Rolling Temporal Variance of Global Gradient Norm Over Time",
        ylabel="Rolling Var(G_t)",
        output_path=output_dir / "global_grad_norm_temporal_variance_timeseries.png",
        requested_conditions=requested_conditions,
        scope="global",
    )

    layers = collect_layers(rows)
    for metric in ("global_grad_norm", "global_grad_norm_variance", "global_grad_norm_std"):
        settings = METRICS[metric]
        plot_layer_boxplots(
            rows=rows,
            metric=metric,
            settings=settings,
            layers=layers,
            output_dir=output_dir,
            group_by=group_by,
            requested_conditions=requested_conditions,
        )
        plot_layer_grid(
            rows=rows,
            metric=metric,
            settings=settings,
            layers=layers,
            output_path=output_dir / f"{metric}_layer_boxplots.png",
            group_by=group_by,
            requested_conditions=requested_conditions,
        )

    plot_layer_timeseries(
        rows=rows,
        metric="global_grad_norm_variance",
        settings=METRICS["global_grad_norm_variance"],
        layers=layers,
        output_dir=output_dir,
        requested_conditions=requested_conditions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create condition-comparison plots for temporal variance of global gradient norm."
    )
    parser.add_argument("--analysis-csv", type=Path, default=Path("global_grad_norm_temporal_variance.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--log-dir", type=Path, default=Path("../logs"))
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--window-size", type=int, default=10, help="Rolling window size over steps.")
    parser.add_argument("--min-window", type=int, default=2, help="Minimum samples before variance is emitted.")
    parser.add_argument(
        "--group-by",
        choices=["epoch", "condition", "condition_epoch"],
        default="condition",
        help="Boxplot grouping. condition compares runs using multiple step data.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute the analysis CSV from latest gradient runs before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    analysis_csv = (
        (script_dir / args.analysis_csv).resolve()
        if not args.analysis_csv.is_absolute()
        else args.analysis_csv
    )
    output_dir = (script_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    if args.recompute or not analysis_csv.exists():
        root = (script_dir / args.root).resolve() if not args.root.is_absolute() else args.root
        log_dir = (script_dir / args.log_dir).resolve() if not args.log_dir.is_absolute() else args.log_dir
        rows = analyze_conditions(
            root=root,
            log_dirs=[log_dir],
            conditions=args.conditions,
            window_size=args.window_size,
            min_window=args.min_window,
        )
        if not rows:
            raise ValueError("No rows were collected")
        write_rows(rows, analysis_csv)
        print(f"rows={len(rows)} output_csv={analysis_csv}")
    else:
        rows = read_rows(analysis_csv)

    plot_all(rows, output_dir, group_by=args.group_by, requested_conditions=args.conditions)


if __name__ == "__main__":
    main()
