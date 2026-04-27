#!/usr/bin/env python3
"""Plot sequential OT distance and cosine similarity trajectories.

This uses the per-round summary CSVs produced by the 2_microbenchmark analysis.
Each scenario gets one figure with two panels:
  - OT distance vs. round
  - cosine similarity vs. round
Lines are individual target runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_RESULTS_DIR = SCENARIO_DIR / "2_microbenchmark" / "results"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_PLOTS_DIR = SCRIPT_DIR / "plots"

METRICS = (
    ("ot_distance", "OT distance"),
    ("cosine_similarity", "Cosine similarity"),
)

GROUP_COLORS = {
    "clean": "#1f77b4",
    "poisoned": "#d62728",
    "augmentation clean": "#2ca02c",
    "ood clean": "#9467bd",
    "baseclean clean": "#1f77b4",
    "blurring poisoned": "#d62728",
}


def display_group(row: pd.Series) -> str:
    poisoning_type = str(row.get("poisoning_type", "")).strip().lower()
    target_group = str(row.get("target_group", "")).strip().lower()
    label = str(row.get("classification_label", "")).strip().lower()

    if poisoning_type == "augmentation" or "data_augmentation" in target_group:
        return "augmentation clean"
    if poisoning_type == "ood" or "ood" in target_group:
        return "ood clean"
    if "baseclean" in target_group:
        return "baseclean clean"
    if poisoning_type == "blurring" or "baseblurring" in target_group or label == "poisoned":
        return "blurring poisoned"
    if label:
        return label
    return target_group or "unknown"


def scenario_name(path: Path, df: pd.DataFrame) -> str:
    if "scenario" in df.columns and df["scenario"].notna().any():
        return str(df["scenario"].dropna().iloc[0])
    return path.name.removesuffix("_summary.csv")


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"round", "target_run_csv", "ot_distance", "cosine_similarity"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round", "target_run_csv", "ot_distance", "cosine_similarity"])
    df["round"] = df["round"].astype(int)
    df["display_group"] = df.apply(display_group, axis=1)
    df["scenario"] = scenario_name(path, df)
    return df


def group_order(df: pd.DataFrame) -> list[str]:
    preferred = [
        "baseclean clean",
        "clean",
        "augmentation clean",
        "ood clean",
        "blurring poisoned",
        "poisoned",
    ]
    present = set(df["display_group"].dropna())
    ordered = [name for name in preferred if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def color_for_group(name: str, index: int) -> str:
    if name in GROUP_COLORS:
        return GROUP_COLORS[name]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle[index % len(cycle)]


def plot_scenario(df: pd.DataFrame, out_path: Path) -> None:
    scenario = str(df["scenario"].iloc[0])
    groups = group_order(df)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True, constrained_layout=True)
    fig.suptitle(f"{scenario}: sequential OT distance and cosine similarity", fontsize=15, fontweight="bold")

    legend_handles: list[Line2D] = []
    for group_index, group in enumerate(groups):
        group_df = df[df["display_group"] == group]
        color = color_for_group(group, group_index)
        legend_handles.append(Line2D([0], [0], color=color, lw=2.8, label=group))

        for metric, ylabel in METRICS:
            ax = axes[0] if metric == "ot_distance" else axes[1]
            for _, run_df in group_df.groupby("target_run_csv", sort=False):
                run_df = run_df.sort_values("round")
                ax.plot(
                    run_df["round"],
                    run_df[metric],
                    color=color,
                    alpha=0.55,
                    linewidth=1.2,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.7)

    axes[-1].set_xlabel("Round index")
    axes[0].legend(handles=legend_handles, loc="best", frameon=True, title="Target group")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        default=DEFAULT_SOURCE_RESULTS_DIR,
        help="Preferred source directory containing *_summary.csv files from 2_microbenchmark.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Fallback source directory containing copied *_summary.csv files.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=SCRIPT_DIR / "sequential_metrics.csv",
        help="Path for the combined copy of all plotted summary data.",
    )
    args = parser.parse_args()

    source_dir = args.source_results_dir
    summary_paths = sorted(source_dir.glob("*_summary.csv"))
    if not summary_paths:
        source_dir = args.data_dir
        summary_paths = sorted(source_dir.glob("*_summary.csv"))
    if not summary_paths:
        raise SystemExit(
            f"No *_summary.csv files found in {args.source_results_dir} or {args.data_dir}"
        )

    frames = [load_summary(path) for path in summary_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.combined_csv, index=False)

    for scenario, scenario_df in combined.groupby("scenario", sort=True):
        safe_name = scenario.replace("/", "_").replace(" ", "_")
        plot_scenario(scenario_df, args.out_dir / f"{safe_name}_sequential_metrics.png")

    print(f"Read summary data from: {source_dir}")
    print(f"Wrote combined data: {args.combined_csv}")
    print(f"Wrote plots: {args.out_dir}")


if __name__ == "__main__":
    main()
