from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_classifier():
    path = SCRIPT_DIR / "classify_clean_poison_f1_window_sweep.py"
    spec = importlib.util.spec_from_file_location("microbenchmark_classifier", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load classifier from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_windows(value: str) -> list[int]:
    windows = [int(s.strip()) for s in value.split(",") if s.strip()]
    if not windows:
        raise ValueError("--windows must list at least one integer")
    return windows


def _aggregate_window(classifier, summary: pd.DataFrame, window: int) -> pd.DataFrame:
    df = classifier._normalize_feature_columns(summary)
    df = classifier._ensure_derived_columns(df)
    df = classifier._ensure_label_column(df)
    df = df.sort_values("round").copy()
    df["window_id"] = df["round"].astype(int) // int(window)

    group_cols = ["device_id", "poisoning_type", "poisoning_rate", "window_id"]
    if "target_run_csv" in df.columns:
        group_cols.insert(1, "target_run_csv")
    if "target_group" in df.columns:
        group_cols.insert(2 if "target_run_csv" in df.columns else 1, "target_group")
    if "classification_label" in df.columns:
        group_cols.append("classification_label")
    if "target_label" in df.columns:
        group_cols.append("target_label")

    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            label=("label", "max"),
            ot_distance=("ot_distance", "mean"),
            cosine_similarity=("cosine_similarity", "mean"),
        )
        .reset_index()
    )


def _mark_split(classifier, agg: pd.DataFrame, calibration_fraction: float) -> pd.DataFrame:
    out = agg.copy()
    mask = classifier._stratified_calibration_mask(out, calibration_fraction)
    out["split"] = np.where(mask, "calibration", "test")
    return out


def _display_group(row: pd.Series) -> str:
    target_group = str(row.get("target_group", ""))
    poisoning_type = str(row.get("poisoning_type", ""))
    label = "poisoned" if int(row["label"]) == 1 else "clean"
    if poisoning_type in {"augmentation", "ood"}:
        return f"{poisoning_type} clean"
    if "baseclean" in target_group and label == "clean":
        return "baseclean clean"
    if "baseblurring" in target_group or poisoning_type == "blurring":
        return "blurring poisoned"
    return f"{target_group or poisoning_type} {label}".strip()


def _plot_threshold(ax, perf_row: pd.Series, x_min: float, x_max: float) -> None:
    w_ot = float(perf_row["best_w_ot"])
    w_cos = float(perf_row["best_w_cos"])
    threshold = float(perf_row["best_threshold"])
    if not np.isfinite([w_ot, w_cos, threshold]).all():
        return

    if abs(w_cos) < 1e-12:
        if abs(w_ot) < 1e-12:
            return
        x = threshold / w_ot
        ax.axvline(x, color="#222222", linestyle="--", linewidth=1.6, label="threshold")
        return

    xs = np.linspace(x_min, x_max, 200)
    ys = 1.0 - ((threshold - (w_ot * xs)) / w_cos)
    ax.plot(xs, ys, color="#222222", linestyle="--", linewidth=1.6, label="threshold")


def plot_scenario_window(
    classifier,
    summary_csv: Path,
    classification_csv: Path,
    out_path: Path,
    window: int,
    calibration_fraction: float,
) -> None:
    scenario = summary_csv.stem.removesuffix("_summary")
    summary = pd.read_csv(summary_csv)
    perf = pd.read_csv(classification_csv)
    perf_rows = perf[perf["window"] == int(window)]
    if perf_rows.empty:
        return
    perf_row = perf_rows.iloc[0]

    agg = _aggregate_window(classifier, summary, window)
    agg = _mark_split(classifier, agg, calibration_fraction)
    agg["display_group"] = agg.apply(_display_group, axis=1)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    colors = {
        "baseclean clean": "#1f77b4",
        "augmentation clean": "#2ca02c",
        "ood clean": "#17becf",
        "blurring poisoned": "#d62728",
    }
    fallback_colors = {"clean": "#1f77b4", "poisoned": "#d62728"}
    markers = {"calibration": "o", "test": "^"}

    display_order = [
        "baseclean clean",
        "augmentation clean",
        "ood clean",
        "blurring poisoned",
    ]
    display_order.extend(
        group for group in sorted(agg["display_group"].unique()) if group not in display_order
    )
    for display_group in display_order:
        for split_name in ["calibration", "test"]:
            subset = agg[
                (agg["display_group"] == display_group) & (agg["split"] == split_name)
            ]
            if subset.empty:
                continue
            label_name = "poisoned" if int(subset["label"].iloc[0]) == 1 else "clean"
            ax.scatter(
                subset["ot_distance"],
                subset["cosine_similarity"],
                s=62,
                marker=markers[split_name],
                color=colors.get(display_group, fallback_colors[label_name]),
                edgecolor="white",
                linewidth=0.7,
                alpha=0.86 if split_name == "test" else 0.58,
                label=f"{display_group} {split_name}",
            )

    x_values = agg["ot_distance"].astype(float)
    x_pad = max((x_values.max() - x_values.min()) * 0.08, 1e-6)
    x_min = float(x_values.min() - x_pad)
    x_max = float(x_values.max() + x_pad)
    _plot_threshold(ax, perf_row, x_min, x_max)

    ax.set_xlim(x_min, x_max)
    y_values = agg["cosine_similarity"].astype(float)
    y_pad = max((y_values.max() - y_values.min()) * 0.08, 1e-6)
    ax.set_ylim(float(y_values.min() - y_pad), float(y_values.max() + y_pad))
    ax.set_xlabel("OT distance")
    ax.set_ylabel("cosine similarity")
    ax.set_title(f"{scenario}: window={window}")

    score_text = (
        f"score = {perf_row['best_w_ot']:.1f}*OT + "
        f"{perf_row['best_w_cos']:.1f}*(1-cos)\n"
        f"threshold = {perf_row['best_threshold']:.6g}\n"
        f"cal F1 = {perf_row['calibration_f1']:.3f}, "
        f"test F1 = {perf_row['best_f1']:.3f}, "
        f"test FPR = {perf_row['best_fpr']:.3f}, "
        f"test FNR = {perf_row['best_fnr']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        score_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.88},
    )
    ax.grid(True, alpha=0.22)

    present_groups = [group for group in display_order if group in set(agg["display_group"])]
    group_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=colors.get(
                group,
                fallback_colors["poisoned" if "poisoned" in group else "clean"],
            ),
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=8,
            label=group,
        )
        for group in present_groups
    ]
    split_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[split],
            linestyle="",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=8,
            label=split,
        )
        for split in ["calibration", "test"]
    ]
    threshold_handle = Line2D(
        [0],
        [0],
        color="#222222",
        linestyle="--",
        linewidth=1.6,
        label="threshold",
    )
    ax.legend(
        handles=[*group_handles, *split_handles, threshold_handle],
        loc="best",
        fontsize=8,
        title="Group / split",
        title_fontsize=8,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    p.add_argument("--out-dir", default="")
    p.add_argument("--windows", default="1,5,10")
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_dir / "plots"
    windows = _parse_windows(args.windows)
    classifier = _load_classifier()

    for summary_csv in sorted(results_dir.glob("*_summary.csv")):
        scenario = summary_csv.stem.removesuffix("_summary")
        classification_csv = results_dir / f"{scenario}_classification.csv"
        if not classification_csv.exists():
            continue
        for window in windows:
            out_path = out_dir / f"{scenario}_window_{window}.png"
            plot_scenario_window(
                classifier=classifier,
                summary_csv=summary_csv,
                classification_csv=classification_csv,
                out_path=out_path,
                window=window,
                calibration_fraction=float(args.calibration_fraction),
            )
            print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
