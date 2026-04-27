from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


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
CONDITION_ALIASES = {
    "augmentation": ["augmentation", "data_augmentation"],
    "data_augmentation": ["data_augmentation", "augmentation"],
}

RMSE_METRICS = {
    "rmse": {
        "title": "Step-Level Gradient RMSE vs Clean",
        "ylabel": "RMSE",
        "filename": "rmse_boxplot.png",
        "color": "#8fb4d8",
    },
}
COVERAGE_METRICS = {
    "same_coverage_ratio": {
        "title": "Step-Level Same Coverage vs Clean",
        "ylabel": "Same coverage",
        "filename": "same_coverage_boxplot.png",
        "color": "#b6a3d9",
    },
    "target_coverage_ratio": {
        "title": "Step-Level Target Coverage vs Clean",
        "ylabel": "Target coverage",
        "filename": "target_coverage_boxplot.png",
        "color": "#b6a3d9",
    },
    "jaccard_ratio": {
        "title": "Step-Level Jaccard Coverage vs Clean",
        "ylabel": "Jaccard ratio",
        "filename": "jaccard_boxplot.png",
        "color": "#b6a3d9",
    },
    "jaccard_distance": {
        "title": "Step-Level Jaccard Distance vs Clean",
        "ylabel": "Jaccard distance",
        "filename": "jaccard_distance_boxplot.png",
        "color": "#b6a3d9",
    },
    "same_mask_ratio": {
        "title": "Step-Level Same Mask Ratio vs Clean",
        "ylabel": "Same mask ratio",
        "filename": "same_mask_boxplot.png",
        "color": "#b6a3d9",
    },
}
ALIGNMENT_METRICS = {
    "cosine_similarity": {
        "title": "Step-Level Cosine Similarity vs Clean",
        "ylabel": "Cosine similarity",
        "filename": "cosine_similarity_boxplot.png",
        "color": "#8fb9a8",
    },
    "sign_agreement": {
        "title": "Step-Level Sign Agreement vs Clean",
        "ylabel": "Sign agreement",
        "filename": "sign_agreement_boxplot.png",
        "color": "#8fb9a8",
    },
    "dot_product": {
        "title": "Step-Level Dot Product vs Clean",
        "ylabel": "Dot product",
        "filename": "dot_product_boxplot.png",
        "color": "#8fb9a8",
    },
}
METRIC_GROUPS = {
    "rmse": RMSE_METRICS,
    "coverage": COVERAGE_METRICS,
    "alignment": ALIGNMENT_METRICS,
}


def canonical_condition(condition: str) -> str:
    condition = condition.strip()
    if condition == "data_augmentation":
        return "augmentation"
    return condition


def condition_matches(condition: str, wanted: str) -> bool:
    condition = canonical_condition(condition)
    wanted = canonical_condition(wanted)
    aliases = [canonical_condition(name) for name in CONDITION_ALIASES.get(wanted, [wanted])]
    return condition in aliases


def safe_filename(name: str) -> str:
    return (
        name.replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def parse_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def latest_csv_per_condition(
    log_dirs: Sequence[Path],
    conditions: Sequence[str],
    explicit_csvs: Sequence[Path],
) -> Dict[str, Path]:
    selected: Dict[str, Path] = {}

    for path in explicit_csvs:
        rows = read_csv_rows(path)
        if not rows:
            continue
        condition = canonical_condition(str(rows[0].get("condition", "")))
        if condition:
            selected[condition] = path

    for wanted in conditions:
        canonical = canonical_condition(wanted)
        if canonical in selected:
            continue
        for log_dir in log_dirs:
            if not log_dir.exists():
                continue
            matches: List[Path] = []
            for path in sorted(log_dir.glob("*_grad_analysis.csv")):
                rows = read_csv_rows(path)
                if rows and condition_matches(str(rows[0].get("condition", "")), canonical):
                    matches.append(path)
            if matches:
                selected[canonical] = matches[-1]
                break

    return selected


def collect_values(
    csv_paths: Dict[str, Path],
    metrics: Iterable[str],
    scope: str,
    layer: Optional[str] = None,
) -> Dict[str, Dict[str, List[float]]]:
    values: Dict[str, Dict[str, List[float]]] = {}
    metric_list = list(metrics)

    for condition, path in csv_paths.items():
        condition_values = {metric: [] for metric in metric_list}
        for row in read_csv_rows(path):
            if row.get("scope") != scope:
                continue
            if layer is not None and row.get("layer") != layer:
                continue
            for metric in metric_list:
                value = parse_float(row.get(metric, ""))
                if value is not None:
                    condition_values[metric].append(value)

        if any(condition_values[metric] for metric in metric_list):
            values[condition] = condition_values

    return values


def collect_layers(csv_paths: Dict[str, Path]) -> List[str]:
    layers = set()
    for path in csv_paths.values():
        for row in read_csv_rows(path):
            if row.get("scope") == "layer" and row.get("layer"):
                layers.add(str(row["layer"]))
    return sorted(layers)


def ordered_conditions(values: Dict[str, Dict[str, List[float]]], requested: Sequence[str]) -> List[str]:
    order = [canonical_condition(condition) for condition in requested]
    present = list(values)
    ordered = [condition for condition in order if condition in values]
    ordered.extend(condition for condition in present if condition not in ordered)
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


def metric_title(settings: dict, scope_label: str, layer: Optional[str] = None) -> str:
    title = str(settings["title"])
    if scope_label == "FLOP-weighted":
        title = title.replace("Step-Level", "Step-Level FLOP-Weighted")
    elif layer:
        title = f"{title}: {layer}"
    return title


def metric_ylabel(settings: dict, scope_label: str) -> str:
    ylabel = str(settings["ylabel"])
    if scope_label == "FLOP-weighted":
        return f"FLOP-weighted {ylabel}"
    return ylabel


def plot_metric(
    values: Dict[str, Dict[str, List[float]]],
    metric: str,
    settings: dict,
    output_path: Path,
    requested_conditions: Sequence[str],
    scope_label: str,
    layer: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    conditions = [
        condition
        for condition in ordered_conditions(values, requested_conditions)
        if values.get(condition, {}).get(metric)
    ]
    if not conditions:
        return

    labels = [DISPLAY_NAMES.get(condition, condition) for condition in conditions]
    box_values = [values[condition][metric] for condition in conditions]
    owns_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 5.8))

    boxplot = ax.boxplot(box_values, tick_labels=labels, patch_artist=True, showmeans=True)
    style_boxplot(boxplot, color=str(settings["color"]))
    ax.set_title(metric_title(settings, scope_label=scope_label, layer=layer))
    ax.set_ylabel(metric_ylabel(settings, scope_label=scope_label))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="x", labelrotation=25)

    if owns_figure:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"saved={output_path}")


def plot_metric_group(
    values: Dict[str, Dict[str, List[float]]],
    metric_settings: Dict[str, dict],
    output_dir: Path,
    requested_conditions: Sequence[str],
    scope_label: str,
    filename_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, settings in metric_settings.items():
        plot_metric(
            values=values,
            metric=metric,
            settings=settings,
            output_path=output_dir / f"{filename_prefix}{settings['filename']}",
            requested_conditions=requested_conditions,
            scope_label=scope_label,
        )

    metrics_with_values = [
        metric
        for metric in metric_settings
        if any(values[condition].get(metric) for condition in values)
    ]
    if len(metrics_with_values) <= 1:
        return

    fig, axes = plt.subplots(1, len(metrics_with_values), figsize=(6.1 * len(metrics_with_values), 5.6))
    if len(metrics_with_values) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics_with_values):
        plot_metric(
            values=values,
            metric=metric,
            settings=metric_settings[metric],
            output_path=output_dir / f"{filename_prefix}{metric_settings[metric]['filename']}",
            requested_conditions=requested_conditions,
            scope_label=scope_label,
            ax=ax,
        )
    fig.tight_layout()
    combined_path = output_dir / f"{filename_prefix}combined_boxplots.png"
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)
    print(f"saved={combined_path}")


def plot_layer_grid(
    csv_paths: Dict[str, Path],
    metric: str,
    settings: dict,
    layers: Sequence[str],
    output_path: Path,
    requested_conditions: Sequence[str],
) -> None:
    if not layers:
        return
    ncols = 3
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.7 * nrows))
    flat_axes = list(axes.reshape(-1)) if hasattr(axes, "reshape") else [axes]

    for ax, layer in zip(flat_axes, layers):
        values = collect_values(csv_paths, [metric], scope="layer", layer=layer)
        plot_metric(
            values=values,
            metric=metric,
            settings=settings,
            output_path=output_path,
            requested_conditions=requested_conditions,
            scope_label="Layer",
            layer=layer,
            ax=ax,
        )

    for ax in flat_axes[len(layers) :]:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved={output_path}")


def plot_layer_metrics(
    csv_paths: Dict[str, Path],
    output_dir: Path,
    requested_conditions: Sequence[str],
    layers: Sequence[str],
) -> None:
    layer_root = output_dir / "layers"
    for group_name, metric_settings in METRIC_GROUPS.items():
        group_dir = layer_root / group_name
        for metric, settings in metric_settings.items():
            metric_dir = group_dir / metric
            for layer in layers:
                values = collect_values(csv_paths, [metric], scope="layer", layer=layer)
                plot_metric(
                    values=values,
                    metric=metric,
                    settings=settings,
                    output_path=metric_dir / f"{safe_filename(layer)}_{settings['filename']}",
                    requested_conditions=requested_conditions,
                    scope_label="Layer",
                    layer=layer,
                )
            plot_layer_grid(
                csv_paths=csv_paths,
                metric=metric,
                settings=settings,
                layers=layers,
                output_path=group_dir / f"{metric}_layer_boxplots.png",
                requested_conditions=requested_conditions,
            )


def print_summary(csv_paths: Dict[str, Path], conditions: Sequence[str]) -> None:
    print("selected_csvs:")
    for condition in ordered_conditions({key: {} for key in csv_paths}, conditions):
        path = csv_paths[condition]
        rows = read_csv_rows(path)
        epochs = sorted({int(row["epoch"]) for row in rows if row.get("epoch", "").isdigit()})
        scopes = defaultdict(int)
        for row in rows:
            scopes[row.get("scope", "")] += 1
        epoch_text = f"{epochs[0]}-{epochs[-1]}" if epochs else "none"
        print(
            f"  {DISPLAY_NAMES.get(condition, condition)}: {path} "
            f"rows={len(rows)} epochs={epoch_text} scopes={dict(scopes)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create box plots from ml_running.py *_grad_analysis.csv files."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("../grad_analysis_logs"),
        help="Directory containing *_grad_analysis.csv files.",
    )
    parser.add_argument(
        "--fallback-log-dir",
        type=Path,
        default=Path("../logs"),
        help="Fallback directory for older *_grad_analysis.csv files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Condition order to plot. Missing conditions are skipped.",
    )
    parser.add_argument(
        "--csv",
        nargs="*",
        type=Path,
        default=[],
        help="Optional explicit *_grad_analysis.csv files. Otherwise latest file per condition is used.",
    )
    parser.add_argument(
        "--skip-layer-plots",
        action="store_true",
        help="Only plot all-layer and FLOP-weighted summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    log_dir = (script_dir / args.log_dir).resolve() if not args.log_dir.is_absolute() else args.log_dir
    fallback_log_dir = (
        (script_dir / args.fallback_log_dir).resolve()
        if not args.fallback_log_dir.is_absolute()
        else args.fallback_log_dir
    )
    output_dir = (script_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    explicit_csvs = [
        (script_dir / path).resolve() if not path.is_absolute() else path
        for path in args.csv
    ]

    log_dirs = [log_dir]
    if fallback_log_dir != log_dir:
        log_dirs.append(fallback_log_dir)
    csv_paths = latest_csv_per_condition(log_dirs, args.conditions, explicit_csvs)
    if not csv_paths:
        searched = ", ".join(str(path) for path in log_dirs)
        raise FileNotFoundError(f"No *_grad_analysis.csv files found in: {searched}")

    print_summary(csv_paths, args.conditions)

    all_layer_values = {
        group_name: collect_values(csv_paths, settings.keys(), scope="all_layers")
        for group_name, settings in METRIC_GROUPS.items()
    }
    flop_weighted_values = {
        group_name: collect_values(csv_paths, settings.keys(), scope="flop_weighted")
        for group_name, settings in METRIC_GROUPS.items()
    }

    for group_name, settings in METRIC_GROUPS.items():
        plot_metric_group(
            values=all_layer_values[group_name],
            metric_settings=settings,
            output_dir=output_dir / "all_layers" / group_name,
            requested_conditions=args.conditions,
            scope_label="All layers",
            filename_prefix="",
        )
        plot_metric_group(
            values=flop_weighted_values[group_name],
            metric_settings=settings,
            output_dir=output_dir / "flop_weighted" / group_name,
            requested_conditions=args.conditions,
            scope_label="FLOP-weighted",
            filename_prefix="flop_weighted_",
        )

    if not args.skip_layer_plots:
        layers = collect_layers(csv_paths)
        plot_layer_metrics(csv_paths, output_dir, args.conditions, layers)


if __name__ == "__main__":
    main()
