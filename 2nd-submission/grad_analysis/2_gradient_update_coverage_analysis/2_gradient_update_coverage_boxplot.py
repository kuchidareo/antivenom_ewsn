from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from gradient_update_coverage_analysis import calculate_layer_step_coverages, calculate_step_coverages


REF_CONDITION = "clean"
TARGET_CONDITIONS = [
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
    "data_augmentation": "augmentation",
    "ood": "ood",
    "blurring": "blurring",
    "label-flip": "label-flip",
    "steganography": "steganography",
    "occlusion": "occlusion",
}
EPS = 0.0
FLOPS = {
    "conv1.weight": 693_633_024,
    "conv2.weight": 3_699_376_128,
    "conv3.weight": 3_699_376_128,
    "fc1.weight": 411_041_792,
    "fc2.weight": 524_288,
    "fc3.weight": 12_288,
}
METRICS = {
    "same_coverage_ratio": {
        "title": "Step-Level Same Coverage vs Clean",
        "ylabel": "Same coverage",
        "filename": "same_coverage_boxplot.png",
    },
    "jaccard_ratio": {
        "title": "Step-Level Jaccard Coverage vs Clean",
        "ylabel": "Jaccard ratio",
        "filename": "jaccard_boxplot.png",
    },
    "same_mask_ratio": {
        "title": "Step-Level Same Mask Ratio vs Clean",
        "ylabel": "Same mask ratio",
        "filename": "same_mask_boxplot.png",
    },
}
FLOP_WEIGHTED_METRICS = {
    metric: {
        **settings,
        "title": settings["title"].replace("Step-Level", "Step-Level FLOP-Weighted"),
        "ylabel": f"FLOP-weighted {settings['ylabel']}",
        "filename": f"flop_weighted_{settings['filename']}",
    }
    for metric, settings in METRICS.items()
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


def flop_weighted_step_metrics(layer_rows: List[dict]) -> Dict[str, List[float]]:
    by_step: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in layer_rows:
        layer = str(row["layer"])
        if layer not in FLOPS:
            continue
        step_file = str(row["step_file"])
        by_step.setdefault(step_file, {})[layer] = {
            metric: float(row[metric])
            for metric in METRICS
        }

    weighted_values: Dict[str, List[float]] = {metric: [] for metric in METRICS}
    for step_file in sorted(by_step):
        step_values = by_step[step_file]
        for metric in METRICS:
            weighted_sum = 0.0
            used_flops = 0.0
            for layer, flops in FLOPS.items():
                if layer not in step_values:
                    continue
                weighted_sum += float(flops) * step_values[layer][metric]
                used_flops += float(flops)
            if used_flops:
                weighted_values[metric].append(weighted_sum / used_flops)

    return weighted_values


def collect_step_metrics(root: Path) -> tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    ref_dir = resolve_run_dir(root, REF_CONDITION)
    if ref_dir is None:
        raise FileNotFoundError(f"No run directory with .pt files found for {REF_CONDITION}")

    print(f"reference_condition={REF_CONDITION}")
    print(f"reference_run={ref_dir}")
    print(f"eps={EPS}")

    results: Dict[str, Dict[str, List[float]]] = {}
    flop_weighted_results: Dict[str, Dict[str, List[float]]] = {}
    for condition in TARGET_CONDITIONS:
        target_dir = resolve_run_dir(root, condition)
        if target_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print(f"skipped {condition}: no run directory with .pt files found under {aliases}")
            continue

        step_rows = calculate_step_coverages(ref_dir=ref_dir, target_dir=target_dir, eps=EPS)
        layer_rows = calculate_layer_step_coverages(ref_dir=ref_dir, target_dir=target_dir, eps=EPS)
        results[condition] = {
            metric: [float(row[metric]) for row in step_rows]
            for metric in METRICS
        }
        flop_weighted_results[condition] = flop_weighted_step_metrics(layer_rows)
        print(f"{condition}: target={target_dir}, steps={len(step_rows)}")

    return results, flop_weighted_results


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


def plot_metric(
    metric: str,
    metric_values: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
    metric_settings: Dict[str, Dict[str, str]] = METRICS,
    ax: Optional[plt.Axes] = None,
) -> None:
    labels = [DISPLAY_NAMES[condition] for condition in metric_values]
    values = [metric_values[condition][metric] for condition in metric_values]
    owns_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))

    boxplot = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    style_boxplot(boxplot, color="#b6a3d9")
    ax.set_title(metric_settings[metric]["title"])
    ax.set_ylabel(metric_settings[metric]["ylabel"])
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="x", labelrotation=25)

    if owns_figure:
        plt.tight_layout()
        output_path = output_dir / metric_settings[metric]["filename"]
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"saved={output_path}")


def plot_metric_group(
    metric_values: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
    metric_settings: Dict[str, Dict[str, str]],
    combined_filename: str,
) -> None:
    for metric in metric_settings:
        plot_metric(metric, metric_values, output_dir, metric_settings=metric_settings)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, metric in zip(axes, metric_settings):
        plot_metric(metric, metric_values, output_dir, metric_settings=metric_settings, ax=ax)
    fig.tight_layout()
    combined_path = output_dir / combined_filename
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)
    print(f"saved={combined_path}")


def plot_all(
    metric_values: Dict[str, Dict[str, List[float]]],
    flop_weighted_values: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_group(
        metric_values,
        output_dir,
        metric_settings=METRICS,
        combined_filename="coverage_metrics_boxplots.png",
    )
    plot_metric_group(
        flop_weighted_values,
        output_dir,
        metric_settings=FLOP_WEIGHTED_METRICS,
        combined_filename="flop_weighted_coverage_metrics_boxplots.png",
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    output_dir = script_dir / "plots"
    metric_values, flop_weighted_values = collect_step_metrics(root)
    if not metric_values:
        raise ValueError("No condition metrics were collected")
    plot_all(metric_values, flop_weighted_values, output_dir)


if __name__ == "__main__":
    main()
