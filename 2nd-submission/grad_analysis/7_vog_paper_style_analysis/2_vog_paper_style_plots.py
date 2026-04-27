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

from vog_paper_style_analysis import (
    DEFAULT_CONDITIONS,
    analyze_runs,
    canonical_condition,
    resolve_runs,
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
COLORS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
]


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


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ordered_conditions(present: Iterable[str], requested: Sequence[str]) -> List[str]:
    requested_order = [canonical_condition(condition) for condition in requested]
    present_list = list(dict.fromkeys(canonical_condition(condition) for condition in present))
    ordered = [condition for condition in requested_order if condition in present_list]
    ordered.extend(condition for condition in present_list if condition not in ordered)
    return ordered


def decile_label(decile: int) -> str:
    lo = decile * 10
    hi = lo + 10
    if decile == 9:
        return "top 10%"
    if decile == 0:
        return "bottom 10%"
    return f"{lo}-{hi}%"


def latest_epoch_rows(rows: Sequence[dict]) -> List[dict]:
    epochs = [int(row["epoch"]) for row in rows if str(row.get("epoch", "")) != ""]
    if not epochs:
        return list(rows)
    latest = max(epochs)
    return [row for row in rows if int(row["epoch"]) == latest]


def plot_decile_lines(
    bucket_rows: Sequence[dict],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    requested_conditions: Sequence[str],
) -> None:
    rows = latest_epoch_rows(bucket_rows)
    grouped: Dict[str, Dict[int, float]] = defaultdict(dict)
    for row in rows:
        value = parse_float(row.get(metric))
        if value is None:
            continue
        condition = canonical_condition(str(row["condition"]))
        grouped[condition][int(row["class_decile"])] = value

    conditions = ordered_conditions(grouped, requested_conditions)
    if not conditions:
        return

    _, ax = plt.subplots(figsize=(10.5, 5.8))
    for index, condition in enumerate(conditions):
        deciles = sorted(grouped[condition])
        values = [grouped[condition][decile] for decile in deciles]
        ax.plot(
            deciles,
            values,
            marker="o",
            linewidth=1.8,
            label=DISPLAY_NAMES.get(condition, condition),
            color=COLORS[index % len(COLORS)],
        )

    ax.set_title(title)
    ax.set_xlabel("Class-wise VoG percentile bucket")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(10))
    ax.set_xticklabels([decile_label(i) for i in range(10)], rotation=25, ha="right")
    ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="best", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"saved={output_path}")


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


def plot_late_boxplot(
    late_rows: Sequence[dict],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    requested_conditions: Sequence[str],
    color: str,
) -> None:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in late_rows:
        value = parse_float(row.get(metric))
        if value is None:
            continue
        grouped[canonical_condition(str(row["condition"]))].append(value)

    conditions = ordered_conditions(grouped, requested_conditions)
    if not conditions:
        return

    labels = [DISPLAY_NAMES.get(condition, condition) for condition in conditions]
    values = [grouped[condition] for condition in conditions]
    _, ax = plt.subplots(figsize=(max(8.5, 0.8 * len(labels)), 5.8))
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


def write_top_samples(late_rows: Sequence[dict], output_path: Path, top_n: int) -> None:
    fields = [
        "condition",
        "sample_id",
        "dataset_index",
        "target",
        "audit_score",
        "class_percentile_mean_late",
        "top10_persistence_late",
        "misclassified_rate_late",
        "loss_mean_late",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in late_rows:
        grouped[canonical_condition(str(row["condition"]))].append(row)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for condition in sorted(grouped):
            rows = sorted(grouped[condition], key=lambda row: -float(row["audit_score"]))[:top_n]
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
    print(f"saved={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-style VoG ranking analysis.")
    parser.add_argument("--vog-log-root", type=Path, default=Path("../vog_logs"))
    parser.add_argument("--runs", nargs="*", default=[], help="Explicit CONDITION=RUN_DIR mappings.")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--score-source", choices=["auto", "raw", "ranking"], default="auto")
    parser.add_argument("--scope", choices=["global", "layer"], default="global")
    parser.add_argument("--layer", default="")
    parser.add_argument("--late-epochs", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    vog_log_root = (script_dir / args.vog_log_root).resolve() if not args.vog_log_root.is_absolute() else args.vog_log_root
    runs = resolve_runs(vog_log_root, args.conditions, args.runs)

    per_sample_csv, bucket_csv, late_csv = analyze_runs(
        runs=runs,
        output_dir=script_dir,
        score_source=args.score_source,
        scope=args.scope,
        layer=args.layer,
        late_epochs=args.late_epochs,
    )
    bucket_rows = read_rows(bucket_csv)
    late_rows = read_rows(late_csv)
    plot_dir = script_dir / "plots"

    plot_decile_lines(
        bucket_rows,
        metric="error_rate",
        ylabel="Misclassification rate",
        title="Error Rate by Class-Wise VoG Decile",
        output_path=plot_dir / "error_rate_by_class_vog_decile.png",
        requested_conditions=args.conditions,
    )
    plot_decile_lines(
        bucket_rows,
        metric="mean_loss",
        ylabel="Mean loss",
        title="Loss by Class-Wise VoG Decile",
        output_path=plot_dir / "loss_by_class_vog_decile.png",
        requested_conditions=args.conditions,
    )
    plot_late_boxplot(
        late_rows,
        metric="class_percentile_mean_late",
        ylabel="Mean late class-wise percentile",
        title="Late-Stage Class-Wise VoG Percentile",
        output_path=plot_dir / "late_class_percentile_boxplot.png",
        requested_conditions=args.conditions,
        color="#8fb4d8",
    )
    plot_late_boxplot(
        late_rows,
        metric="top10_persistence_late",
        ylabel="Late top-10% persistence",
        title="Late-Stage Top-10% Persistence",
        output_path=plot_dir / "late_top10_persistence_boxplot.png",
        requested_conditions=args.conditions,
        color="#8fb9a8",
    )
    plot_late_boxplot(
        late_rows,
        metric="audit_score",
        ylabel="Audit score",
        title="Late-Stage VoG Audit Score",
        output_path=plot_dir / "late_audit_score_boxplot.png",
        requested_conditions=args.conditions,
        color="#d7b48f",
    )
    write_top_samples(late_rows, script_dir / "vog_paper_style_top_samples.csv", top_n=args.top_n)


if __name__ == "__main__":
    main()
