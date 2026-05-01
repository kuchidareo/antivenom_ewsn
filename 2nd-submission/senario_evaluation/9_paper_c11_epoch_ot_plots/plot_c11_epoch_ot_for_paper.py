from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "8_rolling_mean_abs_diff_microbenchmark"
    / "results"
    / "all_scenarios_c11_summary.csv"
)
DEFAULT_ROLLING_POINTS = (
    PROJECT_ROOT
    / "8_rolling_mean_abs_diff_microbenchmark"
    / "results"
    / "all_scenarios_c11_rolling_mad_points.csv"
)
DEFAULT_CLASSIFICATION = (
    PROJECT_ROOT
    / "8_rolling_mean_abs_diff_microbenchmark"
    / "results"
    / "all_scenarios_c11_classification.csv"
)
DEFAULT_OUT_DIR = ROOT_DIR / "results"

SIGNAL_MIX_LABELS = {
    "cpu80_mem20": "0.8 CPU + 0.2 Memory",
    "cpu100_mem0": "1.0 CPU",
    "cpu0_mem100": "1.0 Memory",
}

SCENARIO_LABELS = {
    "base": "Base",
    "adamw_weight_decay": "AdamW WD",
    "batch_norm": "Batch Norm",
    "label_smooth": "Label Smoothing",
    "model_pruning": "Model Pruning",
    "weight_normalization": "Weight Norm",
    "backward_stabilization": "Backward Stabilization",
    "ood": "OOD",
    "data_augmentation": "Data Aug.",
    "data_augmentation_aug_ref": "Data Aug. Ref.",
}

CLASS_ORDER = ("clean", "poisoned")
CLASS_LABELS = {"clean": "Clean", "poisoned": "Poisoned"}
CLASS_COLORS = {"clean": "#2563eb", "poisoned": "#dc2626"}
CLASS_MARKERS = {"clean": "o", "poisoned": "s"}
FOCUSED_MEMORY_SCENARIOS = (
    "adamw_weight_decay",
    "model_pruning",
    "data_augmentation",
    "ood",
)


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing c11 summary CSV: {path}")

    df = pd.read_csv(path)
    required = {
        "scenario",
        "cost_function",
        "signal_mix",
        "classification_label",
        "round",
        "ot_distance",
        "target_run_csv",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df.loc[df["cost_function"] == "c11"].copy()
    df["round"] = df["round"].astype(int)
    df["classification_label"] = df["classification_label"].astype(str)
    return df


def summarize_epoch_trends(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["scenario", "signal_mix", "classification_label", "round"]
    for key, group in df.groupby(group_cols, sort=False):
        scenario, signal_mix, label, round_idx = key
        rows.append(
            {
                "scenario": scenario,
                "signal_mix": signal_mix,
                "classification_label": label,
                "round": int(round_idx),
                "mean_ot_distance": float(group["ot_distance"].mean()),
                "std_ot_distance": float(group["ot_distance"].std(ddof=0)),
                "n_points": int(len(group)),
                "n_runs": int(group["target_run_csv"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scenario", "signal_mix", "classification_label", "round"]
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(axis="x", alpha=0.10, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def plot_mean_range(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    range_alpha: float = 0.16,
) -> None:
    for label in CLASS_ORDER:
        class_df = df.loc[df["classification_label"] == label]
        if class_df.empty:
            continue
        trend = (
            class_df.groupby(x_col, as_index=False)[y_col]
            .agg(["mean", "min", "max"])
            .reset_index()
            .sort_values(x_col)
        )
        x = trend[x_col].to_numpy(dtype=float)
        y = trend["mean"].to_numpy(dtype=float)
        y_min = trend["min"].to_numpy(dtype=float)
        y_max = trend["max"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=CLASS_COLORS[label],
            marker=CLASS_MARKERS[label],
            markersize=4.0,
            linewidth=2.0,
            label=CLASS_LABELS[label],
        )
        ax.fill_between(
            x,
            y_min,
            y_max,
            color=CLASS_COLORS[label],
            alpha=range_alpha,
            linewidth=0,
        )


def plot_one_axis(ax: plt.Axes, trend: pd.DataFrame, scenario: str, signal_mix: str) -> None:
    subset = trend.loc[
        (trend["scenario"] == scenario)
        & (trend["signal_mix"] == signal_mix)
    ].copy()

    for label in CLASS_ORDER:
        series = subset.loc[subset["classification_label"] == label].sort_values("round")
        if series.empty:
            continue
        x = series["round"].to_numpy(dtype=float)
        y = series["mean_ot_distance"].to_numpy(dtype=float)
        std = series["std_ot_distance"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=CLASS_COLORS[label],
            marker=CLASS_MARKERS[label],
            markersize=3.0,
            linewidth=1.7,
            label=CLASS_LABELS[label],
        )
        ax.fill_between(
            x,
            np.maximum(y - std, 0.0),
            y + std,
            color=CLASS_COLORS[label],
            alpha=0.13,
            linewidth=0,
        )

    style_axis(ax)
    ax.set_title(
        f"{SCENARIO_LABELS.get(scenario, scenario.replace('_', ' ').title())}",
        fontsize=9,
        pad=5,
    )
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("OT distance", fontsize=8)


def plot_scenario_figures(trend: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    scenario_dir = out_dir / "per_scenario"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    for scenario in trend["scenario"].drop_duplicates():
        signal_mixes = [
            mix for mix in SIGNAL_MIX_LABELS if mix in set(trend.loc[trend["scenario"] == scenario, "signal_mix"])
        ]
        if not signal_mixes:
            continue

        fig, axes = plt.subplots(
            1,
            len(signal_mixes),
            figsize=(4.2 * len(signal_mixes), 3.2),
            sharey=False,
            constrained_layout=True,
            squeeze=False,
        )
        for ax, signal_mix in zip(axes.ravel(), signal_mixes):
            plot_one_axis(ax, trend, scenario, signal_mix)
            ax.set_title(SIGNAL_MIX_LABELS[signal_mix], fontsize=9, pad=5)

        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(
            f"{SCENARIO_LABELS.get(scenario, scenario.replace('_', ' ').title())}: c11 OT distance by epoch",
            fontsize=12,
        )
        png_path = scenario_dir / f"{scenario}__c11_epoch_ot.png"
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        paths.append(png_path)

    return paths


def plot_backward_memory_all_runs(df: pd.DataFrame, out_dir: Path) -> Path | None:
    subset = df.loc[
        (df["scenario"] == "backward_stabilization")
        & (df["signal_mix"] == "cpu0_mem100")
    ].copy()
    if subset.empty:
        return None
    subset = subset.drop_duplicates(
        subset=["scenario", "signal_mix", "classification_label", "target_run_csv", "round"]
    )

    focused_dir = out_dir / "focused"
    focused_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    plot_mean_range(ax, subset, "round", "ot_distance")
    style_axis(ax)
    ax.set_title("Backward Stabilization: c11 Memory OT Distance", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("OT distance", fontsize=10)
    ax.legend(frameon=False, loc="best")

    out_path = focused_dir / "backward_stabilization__c11_memory100__all_runs_epoch_ot.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_backward_memory_rolling_feature(
    rolling_points_path: Path,
    classification_path: Path,
    out_dir: Path,
) -> Path | None:
    if not rolling_points_path.exists() or not classification_path.exists():
        return None

    points = pd.read_csv(rolling_points_path)
    classification = pd.read_csv(classification_path)
    points = points.loc[
        (points["scenario"] == "backward_stabilization")
        & (points["signal_mix"] == "cpu0_mem100")
        & (points["rolling_window"] == 5)
    ].copy()
    classification = classification.loc[
        (classification["scenario"] == "backward_stabilization")
        & (classification["signal_mix"] == "cpu0_mem100")
        & (classification["rolling_window"] == 5)
    ].copy()
    if points.empty or classification.empty:
        return None

    threshold = float(classification["threshold"].iloc[0])
    focused_dir = out_dir / "focused"
    focused_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    plot_mean_range(ax, points, "window_end_round", "score")

    ax.axhline(
        threshold,
        color="#111827",
        linestyle="--",
        linewidth=1.4,
        label=f"threshold = {threshold:.6f}",
    )

    style_axis(ax)
    ax.set_title("Backward Stabilization: Rolling Mean Abs OT Difference", fontsize=12)
    ax.set_xlabel("Window end epoch", fontsize=10)
    ax.set_ylabel("Rolling mean abs OT diff, w=5", fontsize=10)
    ax.legend(frameon=False, loc="best")

    out_path = focused_dir / "backward_stabilization__c11_memory100__w5_rolling_mad_all_runs.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def memory100_epoch_rows(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    subset = df.loc[
        (df["scenario"] == scenario)
        & (df["signal_mix"] == "cpu0_mem100")
    ].copy()
    return subset.drop_duplicates(
        subset=["scenario", "signal_mix", "classification_label", "target_run_csv", "round"]
    ).sort_values(["classification_label", "target_run_csv", "round"])


def plot_memory100_all_runs(df: pd.DataFrame, scenario: str, out_dir: Path) -> Path | None:
    subset = memory100_epoch_rows(df, scenario)
    if subset.empty:
        return None

    focused_dir = out_dir / "focused"
    focused_dir.mkdir(parents=True, exist_ok=True)
    rows_path = focused_dir / f"{scenario}__c11_memory100_epoch_ot_rows.csv"
    subset.to_csv(rows_path, index=False)

    title = SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title())
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    plot_mean_range(ax, subset, "round", "ot_distance")
    style_axis(ax)
    ax.set_title(f"{title}: c11 Memory OT Distance", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("OT distance", fontsize=10)
    ax.legend(frameon=False, loc="best")

    out_path = focused_dir / f"{scenario}__c11_memory100__all_runs_epoch_ot.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_memory100_best_rolling_feature(
    rolling_points_path: Path,
    classification_path: Path,
    scenario: str,
    out_dir: Path,
) -> Path | None:
    if not rolling_points_path.exists() or not classification_path.exists():
        return None

    points = pd.read_csv(rolling_points_path)
    classification = pd.read_csv(classification_path)
    candidates = classification.loc[
        (classification["scenario"] == scenario)
        & (classification["signal_mix"] == "cpu0_mem100")
    ].copy()
    if candidates.empty:
        return None

    best = candidates.sort_values(
        ["j", "f1", "recall", "precision"],
        ascending=False,
    ).iloc[0]
    rolling_window = int(best["rolling_window"])
    threshold = float(best["threshold"])
    rule = str(best["rule"])
    points = points.loc[
        (points["scenario"] == scenario)
        & (points["signal_mix"] == "cpu0_mem100")
        & (points["rolling_window"] == rolling_window)
    ].copy()
    if points.empty:
        return None

    focused_dir = out_dir / "focused"
    focused_dir.mkdir(parents=True, exist_ok=True)
    points_path = focused_dir / f"{scenario}__c11_memory100__w{rolling_window}_rolling_mad_points.csv"
    points.to_csv(points_path, index=False)

    title = SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title())
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    plot_mean_range(ax, points, "window_end_round", "score")

    comparator = ">=" if rule == "poisoned_if_score_gte_threshold" else "<="
    ax.axhline(
        threshold,
        color="#111827",
        linestyle="--",
        linewidth=1.4,
        label=f"poisoned if score {comparator} {threshold:.6f}",
    )

    style_axis(ax)
    ax.set_title(f"{title}: Rolling Mean Abs OT Difference", fontsize=12)
    ax.set_xlabel("Window end epoch", fontsize=10)
    ax.set_ylabel(f"Rolling mean abs OT diff, w={rolling_window}", fontsize=10)
    ax.legend(frameon=False, loc="best")

    out_path = focused_dir / f"{scenario}__c11_memory100__w{rolling_window}_rolling_mad_all_runs.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper plots for c11 OT distance trends by epoch."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--rolling-points", type=Path, default=DEFAULT_ROLLING_POINTS)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.input)
    trend = summarize_epoch_trends(df)
    trend_path = args.out_dir / "c11_epoch_ot_trends.csv"
    trend.to_csv(trend_path, index=False)

    paths = []
    paths.extend(plot_scenario_figures(trend, args.out_dir))
    focused_path = plot_backward_memory_all_runs(df, args.out_dir)
    if focused_path is not None:
        paths.append(focused_path)
    rolling_path = plot_backward_memory_rolling_feature(
        args.rolling_points,
        args.classification,
        args.out_dir,
    )
    if rolling_path is not None:
        paths.append(rolling_path)
    for scenario in FOCUSED_MEMORY_SCENARIOS:
        raw_path = plot_memory100_all_runs(df, scenario, args.out_dir)
        if raw_path is not None:
            paths.append(raw_path)
        rolling_path = plot_memory100_best_rolling_feature(
            args.rolling_points,
            args.classification,
            scenario,
            args.out_dir,
        )
        if rolling_path is not None:
            paths.append(rolling_path)

    print(f"Loaded: {args.input}")
    print(f"Wrote trend CSV: {trend_path}")
    print(f"Saved {len(paths)} figures under: {args.out_dir}")


if __name__ == "__main__":
    main()
