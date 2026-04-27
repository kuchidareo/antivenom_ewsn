from __future__ import annotations

from pathlib import Path
import argparse
from math import ceil
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


METRIC_SPECS: list[tuple[str, str]] = [
    ("ot_distance", "Magnitude score W"),
    ("cosine_similarity", "Similarity score S"),
    ("mem_time_cost_mean_to_baseclean", "Memory time contribution"),
    ("mem_value_cost_mean_to_baseclean", "Memory value contribution"),
    ("mem_shape_cost_mean_to_baseclean", "Memory shape contribution"),
    ("mem_delta_cost_mean_to_baseclean", "Memory delta contribution"),
    ("core_time_cost_mean_to_baseclean", "CPU mean time contribution"),
    ("core_value_cost_mean_to_baseclean", "CPU mean value contribution"),
    ("core_shape_cost_mean_to_baseclean", "CPU mean shape contribution"),
    ("core_delta_cost_mean_to_baseclean", "CPU mean delta contribution"),
]


def _scenario_name(case_name: str) -> str:
    return case_name.split("/", 1)[0]


def _run_display_label(row: pd.Series) -> str:
    run_csv = Path(str(row["target_run_csv"]))
    return f"{row['target_group']}:{run_csv.stem}"


def _case_colors(case_names: list[str]) -> dict[str, Any]:
    scenario_names: list[str] = []
    for case_name in case_names:
        scenario = _scenario_name(case_name)
        if scenario not in scenario_names:
            scenario_names.append(scenario)
    cmap = plt.cm.get_cmap("tab10", max(len(scenario_names), 1))
    return {scenario: cmap(i) for i, scenario in enumerate(scenario_names)}


def _metric_available(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return False
    if col.endswith("_shape_cost_mean_to_baseclean") or col.endswith("_delta_cost_mean_to_baseclean"):
        return bool((series.abs() > 1e-12).any())
    return True


def _plot_run_boxes(
    ax,
    df: pd.DataFrame,
    value_col: str,
    y_label: str,
    color_map: dict[str, Any],
) -> None:
    ordered_runs = (
        df[["target_group", "target_run_csv", "display_label"]]
        .drop_duplicates()
        .sort_values(["target_group", "target_run_csv"])
    )

    data = []
    labels = []
    colors = []
    for _, run_row in ordered_runs.iterrows():
        mask = (
            (df["target_group"] == run_row["target_group"])
            & (df["target_run_csv"] == run_row["target_run_csv"])
        )
        vals = pd.to_numeric(df.loc[mask, value_col], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(str(run_row["display_label"]))
        colors.append(color_map[_scenario_name(str(run_row["target_group"]))])

    if not data:
        ax.set_axis_off()
        return

    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, alpha=0.7, edgecolor="black", linewidth=0.8)
    for median in bp["medians"]:
        median.set(color="black", linewidth=1.0)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)


def _plot_summary(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    plot_df = df.copy()
    plot_df["display_label"] = plot_df.apply(_run_display_label, axis=1)
    case_names = plot_df["target_group"].drop_duplicates().tolist()
    color_map = _case_colors(case_names)

    metric_specs = [(col, label) for col, label in METRIC_SPECS if _metric_available(plot_df, col)]
    if not metric_specs:
        raise ValueError("No plottable metric columns were found in the input CSV.")

    n_boxes = len(plot_df[["target_group", "target_run_csv"]].drop_duplicates())
    ncols = 2 if len(metric_specs) > 1 else 1
    nrows = ceil(len(metric_specs) / ncols)
    fig_w = max(18.0, n_boxes * 0.55 * ncols)
    fig_h = max(4.8 * nrows, 5.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    flat_axes = list(axes.ravel())

    for ax, (value_col, y_label) in zip(flat_axes, metric_specs):
        _plot_run_boxes(ax, plot_df, value_col, y_label, color_map)
        ax.set_title(value_col)

    for ax in flat_axes[len(metric_specs):]:
        ax.set_axis_off()

    legend_handles = [
        Patch(facecolor=color_map[scenario], edgecolor="black", label=scenario)
        for scenario in color_map
    ]
    if legend_handles:
        flat_axes[0].legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.002, 1.0),
            borderaxespad=0.0,
            title="Scenario",
        )

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="summary CSV generated by 31_iterative_scenario_analysis.py")
    p.add_argument("--out", default="", help="output PNG path")
    p.add_argument("--title", default="OT Contribution Box Plot")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    df = pd.read_csv(csv_path)
    required = ["target_group", "target_run_csv", "ot_distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_path = Path(args.out).resolve() if args.out else csv_path.with_name(f"{csv_path.stem}_contribution_box.png")
    _plot_summary(df=df, out_path=out_path, title=str(args.title))
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
