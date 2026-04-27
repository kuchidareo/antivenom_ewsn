from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from gradient_trusted_alignment_analysis import calculate_step_alignments


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
METRICS = {
    "cosine_similarity": {
        "title": "Step-Level Cosine Similarity vs Clean",
        "ylabel": "Cosine similarity",
        "filename": "cosine_similarity_boxplot.png",
    },
    "sign_agreement": {
        "title": "Step-Level Sign Agreement vs Clean",
        "ylabel": "Sign agreement",
        "filename": "sign_agreement_boxplot.png",
    },
    "dot_product": {
        "title": "Step-Level Dot Product vs Clean",
        "ylabel": "Dot product",
        "filename": "dot_product_boxplot.png",
    },
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


def collect_step_metrics(root: Path) -> Dict[str, Dict[str, List[float]]]:
    ref_dir = resolve_run_dir(root, REF_CONDITION)
    if ref_dir is None:
        raise FileNotFoundError(f"No run directory with .pt files found for {REF_CONDITION}")

    print(f"trusted_reference_condition={REF_CONDITION}")
    print(f"trusted_reference_run={ref_dir}")

    results: Dict[str, Dict[str, List[float]]] = {}
    for condition in TARGET_CONDITIONS:
        target_dir = resolve_run_dir(root, condition)
        if target_dir is None:
            aliases = ", ".join(str(path.relative_to(root)) for path in candidate_condition_dirs(root, condition))
            print(f"skipped {condition}: no run directory with .pt files found under {aliases}")
            continue

        step_rows = calculate_step_alignments(ref_dir=ref_dir, target_dir=target_dir)
        results[condition] = {
            metric: [float(row[metric]) for row in step_rows]
            for metric in METRICS
        }
        print(f"{condition}: target={target_dir}, steps={len(step_rows)}")

    return results


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
    ax: Optional[plt.Axes] = None,
) -> None:
    labels = [DISPLAY_NAMES[condition] for condition in metric_values]
    values = [metric_values[condition][metric] for condition in metric_values]
    owns_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))

    boxplot = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    style_boxplot(boxplot, color="#8fb9a8")
    ax.set_title(METRICS[metric]["title"])
    ax.set_ylabel(METRICS[metric]["ylabel"])
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="x", labelrotation=25)

    if owns_figure:
        plt.tight_layout()
        output_path = output_dir / METRICS[metric]["filename"]
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"saved={output_path}")


def plot_all(metric_values: Dict[str, Dict[str, List[float]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        plot_metric(metric, metric_values, output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, metric in zip(axes, METRICS):
        plot_metric(metric, metric_values, output_dir, ax=ax)
    fig.tight_layout()
    combined_path = output_dir / "alignment_metrics_boxplots.png"
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)
    print(f"saved={combined_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    output_dir = script_dir / "plots"
    metric_values = collect_step_metrics(root)
    if not metric_values:
        raise ValueError("No condition metrics were collected")
    plot_all(metric_values, output_dir)


if __name__ == "__main__":
    main()
