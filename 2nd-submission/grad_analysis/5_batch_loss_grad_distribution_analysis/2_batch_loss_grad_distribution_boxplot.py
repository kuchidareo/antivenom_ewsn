from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from batch_loss_grad_distribution_analysis import (
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
        "scope": "batch",
    },
    "grad_variance": {
        "title": "Layer Gradient Variance",
        "ylabel": "Variance",
        "filename": "grad_variance_boxplot.png",
        "color": "#b6a3d9",
        "scope": "layer",
    },
    "grad_excess_kurtosis": {
        "title": "Layer Gradient Excess Kurtosis",
        "ylabel": "Excess kurtosis",
        "filename": "grad_excess_kurtosis_boxplot.png",
        "color": "#8fb9a8",
        "scope": "layer",
    },
    "abs_grad_excess_kurtosis": {
        "title": "Layer |Gradient| Excess Kurtosis",
        "ylabel": "|g| excess kurtosis",
        "filename": "abs_grad_excess_kurtosis_boxplot.png",
        "color": "#d7b48f",
        "scope": "layer",
    },
    "l2_norm": {
        "title": "Layer Gradient L2 Norm",
        "ylabel": "L2 norm",
        "filename": "l2_norm_boxplot.png",
        "color": "#c8ad7f",
        "scope": "layer_energy",
    },
    "max_energy_share": {
        "title": "Max Energy Share",
        "ylabel": "Max layer energy / total energy",
        "filename": "max_energy_share_boxplot.png",
        "color": "#98b7a3",
        "scope": "batch",
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
    scope: str,
    group_by: str,
    requested_conditions: Sequence[str],
    layer: Optional[str] = None,
) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("scope") != scope:
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
    ax: Optional[plt.Axes] = None,
) -> None:
    if not values_by_group:
        return

    labels = list(values_by_group)
    values = [values_by_group[label] for label in labels]
    owns_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(max(10.5, 0.7 * len(labels)), 5.8))

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


def collect_layers(rows: Sequence[dict], scope: str = "layer") -> List[str]:
    return sorted({str(row.get("layer")) for row in rows if row.get("scope") == scope and row.get("layer")})


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
            scope=str(settings["scope"]),
            group_by=group_by,
            requested_conditions=requested_conditions,
            layer=layer,
        )
        plot_boxplot(
            values_by_group=values,
            title=f"{settings['title']}: {layer}",
            ylabel=str(settings["ylabel"]),
            output_path=output_path,
            color=str(settings["color"]),
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
    rows: Sequence[dict],
    output_dir: Path,
    group_by: str,
    requested_conditions: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_values = collect_metric_values(
        rows=rows,
        metric="batch_loss",
        scope="batch",
        group_by=group_by,
        requested_conditions=requested_conditions,
    )
    plot_boxplot(
        values_by_group=loss_values,
        title=f"{METRICS['batch_loss']['title']} by {group_by}",
        ylabel=str(METRICS["batch_loss"]["ylabel"]),
        output_path=output_dir / METRICS["batch_loss"]["filename"],
        color=str(METRICS["batch_loss"]["color"]),
    )

    max_energy_share_values = collect_metric_values(
        rows=rows,
        metric="max_energy_share",
        scope="batch",
        group_by=group_by,
        requested_conditions=requested_conditions,
    )
    plot_boxplot(
        values_by_group=max_energy_share_values,
        title=f"{METRICS['max_energy_share']['title']} by {group_by}",
        ylabel=str(METRICS["max_energy_share"]["ylabel"]),
        output_path=output_dir / METRICS["max_energy_share"]["filename"],
        color=str(METRICS["max_energy_share"]["color"]),
    )

    layers = collect_layers(rows, scope="layer")
    layer_dir = output_dir / "layers"
    for metric in ("grad_variance", "grad_excess_kurtosis", "abs_grad_excess_kurtosis"):
        settings = METRICS[metric]
        metric_dir = layer_dir / metric
        for layer in layers:
            values = collect_metric_values(
                rows=rows,
                metric=metric,
                scope="layer",
                group_by=group_by,
                requested_conditions=requested_conditions,
                layer=layer,
            )
            plot_boxplot(
                values_by_group=values,
                title=f"{settings['title']}: {layer}",
                ylabel=str(settings["ylabel"]),
                output_path=metric_dir / f"{safe_filename(layer)}_{settings['filename']}",
                color=str(settings["color"]),
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

    energy_layers = collect_layers(rows, scope="layer_energy")
    metric = "l2_norm"
    settings = METRICS[metric]
    metric_dir = layer_dir / metric
    for layer in energy_layers:
        values = collect_metric_values(
            rows=rows,
            metric=metric,
            scope=str(settings["scope"]),
            group_by=group_by,
            requested_conditions=requested_conditions,
            layer=layer,
        )
        plot_boxplot(
            values_by_group=values,
            title=f"{settings['title']}: {layer}",
            ylabel=str(settings["ylabel"]),
            output_path=metric_dir / f"{safe_filename(layer)}_{settings['filename']}",
            color=str(settings["color"]),
        )
    plot_layer_grid(
        rows=rows,
        metric=metric,
        settings=settings,
        layers=energy_layers,
        output_path=output_dir / f"{metric}_layer_boxplots.png",
        group_by=group_by,
        requested_conditions=requested_conditions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create boxplots for batch loss, per-layer gradient variance, and excess kurtosis."
    )
    parser.add_argument("--analysis-csv", type=Path, default=Path("batch_loss_grad_distribution.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--log-dir", type=Path, default=Path("../logs"))
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--layers", nargs="*", default=None)
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
            requested_layers=args.layers,
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
