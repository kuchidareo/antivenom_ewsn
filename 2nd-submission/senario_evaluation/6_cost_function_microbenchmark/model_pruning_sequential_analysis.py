from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results_custom_detection_check"
DEFAULT_SCENARIOS = (
    "model_pruning",
    "weight_normalization",
    "batch_norm",
    "backward_stabilization",
    "label_smooth",
    "adamw_weight_decay",
)
SCENARIO_ALIASES = {
    "label_smoothing": "label_smooth",
    "weight_decay": "adamw_weight_decay",
}
VARIANCE_FEATURE_COLUMNS = [
    "mean_causal_std",
    "min_causal_std",
    "max_causal_std",
    "range_causal_std",
    "std_of_causal_std",
]


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    tp = int(((y_true == "poisoned") & (y_pred == "poisoned")).sum())
    fp = int(((y_true == "clean") & (y_pred == "poisoned")).sum())
    tn = int(((y_true == "clean") & (y_pred == "clean")).sum())
    fn = int(((y_true == "poisoned") & (y_pred == "clean")).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    j = recall - fpr
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "j": j,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
    }


def _threshold_sweep(
    data: pd.DataFrame,
    score_column: str = "causal_post_epoch3_std_ot",
    score_label: str = "epoch_causal_post_epoch3_std_ot",
) -> pd.DataFrame:
    thresholds = sorted(data[score_column].dropna().unique())
    rows = []
    y_true = data["classification_label"]
    for threshold in thresholds:
        y_pred = pd.Series(
            ["poisoned" if value <= threshold else "clean" for value in data[score_column]],
            index=data.index,
        )
        row = {
            "threshold": threshold,
            "score_column": score_column,
            "score_label": score_label,
            "rule": f"poisoned_if_{score_column}_lte_threshold",
        }
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["j", "f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, False, True],
    )


def build_epoch_points(causal: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "cost_function",
        "target_group",
        "target_label",
        "target_run_csv",
        "classification_label",
        "poisoning_type",
        "round",
        "causal_post_epoch3_count",
        "causal_post_epoch3_mean_ot",
        "causal_post_epoch3_std_ot",
    ]
    return (
        causal.dropna(subset=["causal_post_epoch3_std_ot"])[columns]
        .copy()
        .sort_values(["classification_label", "target_group", "target_label", "round"])
    )


def build_epoch_predictions(epoch_points: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    predictions = epoch_points.copy()
    if sweep.empty:
        predictions["epoch_predicted_label"] = pd.NA
        predictions["epoch_threshold"] = pd.NA
        return predictions

    threshold = float(sweep["threshold"].iloc[0])
    predictions["epoch_threshold"] = threshold
    predictions["epoch_predicted_label"] = [
        "poisoned" if value <= threshold else "clean"
        for value in predictions["causal_post_epoch3_std_ot"]
    ]
    return predictions


def build_variance_features(epoch_points: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "scenario",
        "cost_function",
        "target_group",
        "target_label",
        "target_run_csv",
        "classification_label",
        "poisoning_type",
    ]
    features = (
        epoch_points.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_epoch_points=("causal_post_epoch3_std_ot", "size"),
            first_epoch=("round", "min"),
            last_epoch=("round", "max"),
            mean_causal_std=("causal_post_epoch3_std_ot", "mean"),
            min_causal_std=("causal_post_epoch3_std_ot", "min"),
            max_causal_std=("causal_post_epoch3_std_ot", "max"),
            std_of_causal_std=("causal_post_epoch3_std_ot", "std"),
        )
        .sort_values(["classification_label", "target_group", "target_label"])
    )
    features["range_causal_std"] = features["max_causal_std"] - features["min_causal_std"]
    features["std_of_causal_std"] = features["std_of_causal_std"].fillna(0.0)
    return features


def build_variance_threshold_sweep(variance_features: pd.DataFrame) -> pd.DataFrame:
    sweeps = []
    for feature_column in VARIANCE_FEATURE_COLUMNS:
        sweep = _threshold_sweep(
            variance_features,
            score_column=feature_column,
            score_label=f"run_{feature_column}",
        )
        sweep.insert(0, "feature_column", feature_column)
        sweeps.append(sweep)
    if not sweeps:
        return pd.DataFrame()
    return pd.concat(sweeps, ignore_index=True)


def build_variance_predictions(
    variance_features: pd.DataFrame,
    variance_sweep: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for feature_column in VARIANCE_FEATURE_COLUMNS:
        feature_sweep = variance_sweep[variance_sweep["feature_column"] == feature_column]
        if feature_sweep.empty:
            continue
        threshold = float(feature_sweep["threshold"].iloc[0])
        predictions = variance_features.copy()
        predictions["feature_column"] = feature_column
        predictions["feature_threshold"] = threshold
        predictions["feature_score"] = predictions[feature_column]
        predictions["feature_predicted_label"] = [
            "poisoned" if value <= threshold else "clean"
            for value in predictions[feature_column]
        ]
        rows.append(predictions)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_causal_std(df: pd.DataFrame, signal_column: str, min_epoch: int) -> pd.DataFrame:
    if signal_column not in df.columns:
        raise ValueError(f"Missing signal column: {signal_column}")

    group_cols = [
        "scenario",
        "cost_function",
        "target_group",
        "target_label",
        "target_run_csv",
        "classification_label",
        "poisoning_type",
    ]
    post = df[df["round"] >= min_epoch].copy().sort_values([*group_cols, "round"])
    if post.empty:
        raise ValueError(f"No rows found with round >= {min_epoch}")

    grouped = post.groupby(group_cols, dropna=False, sort=False)
    post["causal_post_epoch3_count"] = grouped.cumcount() + 1
    post["causal_post_epoch3_mean_ot"] = grouped[signal_column].expanding().mean().reset_index(level=group_cols, drop=True)
    post["causal_post_epoch3_std_ot"] = grouped[signal_column].expanding().std().reset_index(level=group_cols, drop=True)
    return post.reset_index(drop=True)


def _parse_scenarios(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        return list(DEFAULT_SCENARIOS)
    return [SCENARIO_ALIASES.get(name, name) for name in names]


def plot_epoch_traces(
    df: pd.DataFrame,
    scenario: str,
    signal_column: str,
    min_epoch: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"clean": "#2f6fbb", "poisoned": "#c94b4b"}

    for (_, run_label), run_df in df.groupby(["classification_label", "target_label"], sort=False):
        label = run_df["classification_label"].iloc[0]
        ax.plot(
            run_df["round"],
            run_df[signal_column],
            color=colors.get(label, "#444444"),
            alpha=0.35,
            linewidth=1.3,
        )

    mean_df = (
        df.groupby(["classification_label", "round"], as_index=False)[signal_column]
        .mean()
        .sort_values(["classification_label", "round"])
    )
    for label, label_df in mean_df.groupby("classification_label", sort=False):
        ax.plot(
            label_df["round"],
            label_df[signal_column],
            color=colors.get(label, "#444444"),
            linewidth=3.0,
            label=f"{label} mean",
        )

    ax.axvline(min_epoch, color="#333333", linestyle="--", linewidth=1.2)
    ax.axvspan(min_epoch, df["round"].max(), color="#eeeeee", alpha=0.45)
    ax.set_title(f"{scenario} c11 OT distance by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(signal_column)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_epoch_level_distribution(causal: pd.DataFrame, sweep: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    order = ["clean", "poisoned"]
    colors = {"clean": "#2f6fbb", "poisoned": "#c94b4b"}
    plot_df = causal.dropna(subset=["causal_post_epoch3_std_ot"]).copy()

    values = [
        plot_df.loc[plot_df["classification_label"] == label, "causal_post_epoch3_std_ot"].dropna()
        for label in order
    ]
    ax.boxplot(values, tick_labels=order, widths=0.42, showfliers=False)
    for idx, label in enumerate(order, start=1):
        label_df = plot_df[plot_df["classification_label"] == label].copy()
        if label_df.empty:
            continue
        run_labels = sorted(label_df["target_label"].dropna().unique())
        offsets_by_run = {
            run_label: ((run_idx % 5) - 2) * 0.055
            for run_idx, run_label in enumerate(run_labels)
        }
        epoch_offsets = {
            epoch: ((epoch_idx % 5) - 2) * 0.008
            for epoch_idx, epoch in enumerate(sorted(label_df["round"].dropna().unique()))
        }
        xs = [
            idx
            + offsets_by_run.get(row.target_label, 0.0)
            + epoch_offsets.get(row.round, 0.0)
            for row in label_df.itertuples()
        ]
        ax.scatter(
            xs,
            label_df["causal_post_epoch3_std_ot"],
            s=30,
            color=colors[label],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_title("Epoch-level causal post-epoch-3 OT std classification")
    ax.set_xlabel("Class")
    ax.set_ylabel("std(OT distance seen so far), epochs >= 3")
    if not sweep.empty:
        threshold = float(sweep["threshold"].iloc[0])
        j = float(sweep["j"].iloc[0])
        f1 = float(sweep["f1"].iloc[0])
        ax.axhline(threshold, color="#333333", linestyle="--", linewidth=1.2)
        ax.text(
            0.02,
            0.96,
            f"epoch thr={threshold:.4g}\nJ={j:.3f}, F1={f1:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.86},
        )
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_variance_features(
    variance_features: pd.DataFrame,
    variance_sweep: pd.DataFrame,
    scenario: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(VARIANCE_FEATURE_COLUMNS), figsize=(17, 4.6), sharex=False)
    order = ["clean", "poisoned"]
    colors = {"clean": "#2f6fbb", "poisoned": "#c94b4b"}

    for ax, feature_column in zip(axes, VARIANCE_FEATURE_COLUMNS):
        values = [
            variance_features.loc[
                variance_features["classification_label"] == label,
                feature_column,
            ].dropna()
            for label in order
        ]
        ax.boxplot(values, tick_labels=order, widths=0.42, showfliers=False)
        for idx, label in enumerate(order, start=1):
            label_df = variance_features[variance_features["classification_label"] == label]
            if label_df.empty:
                continue
            offsets = [((run_idx % 5) - 2) * 0.045 for run_idx in range(len(label_df))]
            xs = [idx + offset for offset in offsets]
            ax.scatter(
                xs,
                label_df[feature_column],
                s=34,
                color=colors[label],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

        feature_sweep = variance_sweep[variance_sweep["feature_column"] == feature_column]
        if not feature_sweep.empty:
            threshold = float(feature_sweep["threshold"].iloc[0])
            j = float(feature_sweep["j"].iloc[0])
            f1 = float(feature_sweep["f1"].iloc[0])
            ax.axhline(threshold, color="#333333", linestyle="--", linewidth=1.1)
            ax.text(
                0.04,
                0.96,
                f"thr={threshold:.4g}\nJ={j:.3f}\nF1={f1:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.86},
            )
        ax.set_title(feature_column)
        ax.set_xlabel("Class")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)

    axes[0].set_ylabel("Run-level feature from causal std sequence")
    fig.suptitle(f"{scenario} post-epoch-3 causal std variance features", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_scenario(
    scenario: str,
    summary_csv: Path,
    out_dir: Path,
    signal_column: str,
    min_epoch: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not summary_csv.exists():
        print(f"Skipping {scenario}: missing {summary_csv}")
        return None

    df = pd.read_csv(summary_csv)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    causal = build_causal_std(df, signal_column, min_epoch)
    epoch_points = build_epoch_points(causal)
    epoch_sweep = _threshold_sweep(
        epoch_points,
        score_column="causal_post_epoch3_std_ot",
        score_label="epoch_causal_post_epoch3_std_ot",
    )
    epoch_predictions = build_epoch_predictions(epoch_points, epoch_sweep)
    variance_features = build_variance_features(epoch_points)
    variance_sweep = build_variance_threshold_sweep(variance_features)
    variance_predictions = build_variance_predictions(variance_features, variance_sweep)

    prefix = f"{scenario}__c11_post_epoch{min_epoch}_std"
    epoch_points_path = out_dir / f"{prefix}_epoch_points.csv"
    epoch_sweep_path = out_dir / f"{prefix}_epoch_threshold_sweep.csv"
    epoch_predictions_path = out_dir / f"{prefix}_epoch_predictions.csv"
    variance_features_path = out_dir / f"{prefix}_variance_features.csv"
    variance_sweep_path = out_dir / f"{prefix}_variance_threshold_sweep.csv"
    variance_predictions_path = out_dir / f"{prefix}_variance_predictions.csv"
    trace_plot_path = plots_dir / f"{scenario}__c11_post_epoch{min_epoch}_ot_traces.png"
    std_plot_path = plots_dir / f"{scenario}__c11_post_epoch{min_epoch}_std_by_class.png"
    epoch_plot_path = plots_dir / f"{scenario}__c11_post_epoch{min_epoch}_std_epoch_classification_by_class.png"
    variance_plot_path = plots_dir / f"{scenario}__c11_post_epoch{min_epoch}_std_variance_features_by_class.png"

    epoch_points.to_csv(epoch_points_path, index=False)
    epoch_sweep.to_csv(epoch_sweep_path, index=False)
    epoch_predictions.to_csv(epoch_predictions_path, index=False)
    variance_features.to_csv(variance_features_path, index=False)
    variance_sweep.to_csv(variance_sweep_path, index=False)
    variance_predictions.to_csv(variance_predictions_path, index=False)
    plot_epoch_traces(df, scenario, signal_column, min_epoch, trace_plot_path)
    plot_epoch_level_distribution(causal, epoch_sweep, std_plot_path)
    plot_epoch_level_distribution(causal, epoch_sweep, epoch_plot_path)
    plot_variance_features(variance_features, variance_sweep, scenario, variance_plot_path)

    print(f"Saved {scenario} epoch points: {epoch_points_path}")
    print(f"Saved {scenario} epoch threshold sweep: {epoch_sweep_path}")
    print(f"Saved {scenario} epoch predictions: {epoch_predictions_path}")
    print(f"Saved {scenario} variance features: {variance_features_path}")
    print(f"Saved {scenario} variance threshold sweep: {variance_sweep_path}")
    print(f"Saved {scenario} variance predictions: {variance_predictions_path}")
    print(f"Saved {scenario} trace plot: {trace_plot_path}")
    print(f"Saved {scenario} std plot: {std_plot_path}")
    print(f"Saved {scenario} epoch classification plot: {epoch_plot_path}")
    print(f"Saved {scenario} variance feature plot: {variance_plot_path}")
    if not epoch_sweep.empty:
        print(f"Best {scenario} epoch-level threshold result by Youden J:")
        print(
            epoch_sweep[
                ["threshold", "j", "f1", "precision", "recall", "fpr", "fnr", "tp", "fp", "tn", "fn"]
            ]
            .head(1)
            .to_string(index=False)
        )
    if not variance_sweep.empty:
        print(f"Best {scenario} variance-feature threshold results by Youden J:")
        print(
            variance_sweep[
                [
                    "feature_column",
                    "threshold",
                    "j",
                    "f1",
                    "precision",
                    "recall",
                    "fpr",
                    "fnr",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                ]
            ]
            .groupby("feature_column", sort=False)
            .head(1)
            .to_string(index=False)
        )

    epoch_points.insert(0, "sequential_scenario", scenario)
    epoch_sweep.insert(0, "sequential_scenario", scenario)
    epoch_predictions.insert(0, "sequential_scenario", scenario)
    variance_features.insert(0, "sequential_scenario", scenario)
    variance_sweep.insert(0, "sequential_scenario", scenario)
    variance_predictions.insert(0, "sequential_scenario", scenario)
    return (
        epoch_points,
        epoch_sweep,
        epoch_predictions,
        variance_features,
        variance_sweep,
        variance_predictions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential c11 analysis using post-epoch-3 OT stability."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional single summary CSV. If omitted, --scenarios are read from --out-dir.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--signal-column", default="ot_distance")
    parser.add_argument("--min-epoch", type=int, default=3)
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help=(
            "Comma-separated scenario list. Accepts label_smoothing as label_smooth "
            "and weight_decay as adamw_weight_decay."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.summary_csv is not None:
        scenario = args.summary_csv.name.split("__", 1)[0]
        outputs.append(
            run_scenario(
                scenario=scenario,
                summary_csv=args.summary_csv,
                out_dir=args.out_dir,
                signal_column=args.signal_column,
                min_epoch=args.min_epoch,
            )
        )
    else:
        for scenario in _parse_scenarios(args.scenarios):
            outputs.append(
                run_scenario(
                    scenario=scenario,
                    summary_csv=args.out_dir / f"{scenario}__c11_summary.csv",
                    out_dir=args.out_dir,
                    signal_column=args.signal_column,
                    min_epoch=args.min_epoch,
                )
            )

    outputs = [output for output in outputs if output is not None]
    if outputs:
        combined_epoch_points = pd.concat([output[0] for output in outputs], ignore_index=True)
        combined_epoch_sweep = pd.concat([output[1] for output in outputs], ignore_index=True)
        combined_epoch_predictions = pd.concat(
            [output[2] for output in outputs],
            ignore_index=True,
        )
        combined_variance_features = pd.concat([output[3] for output in outputs], ignore_index=True)
        combined_variance_sweep = pd.concat([output[4] for output in outputs], ignore_index=True)
        combined_variance_predictions = pd.concat(
            [output[5] for output in outputs],
            ignore_index=True,
        )
        combined_epoch_points_path = args.out_dir / f"all_selected__c11_post_epoch{args.min_epoch}_std_epoch_points.csv"
        combined_epoch_sweep_path = args.out_dir / f"all_selected__c11_post_epoch{args.min_epoch}_std_epoch_threshold_sweep.csv"
        combined_epoch_predictions_path = args.out_dir / f"all_selected__c11_post_epoch{args.min_epoch}_std_epoch_predictions.csv"
        combined_variance_features_path = args.out_dir / f"all_selected__c11_post_epoch{args.min_epoch}_std_variance_features.csv"
        combined_variance_sweep_path = args.out_dir / f"all_selected__c11_post_epoch{args.min_epoch}_std_variance_threshold_sweep.csv"
        combined_variance_predictions_path = args.out_dir / f"all_selected__c11_post_epoch{args.min_epoch}_std_variance_predictions.csv"
        combined_epoch_points.to_csv(combined_epoch_points_path, index=False)
        combined_epoch_sweep.to_csv(combined_epoch_sweep_path, index=False)
        combined_epoch_predictions.to_csv(combined_epoch_predictions_path, index=False)
        combined_variance_features.to_csv(combined_variance_features_path, index=False)
        combined_variance_sweep.to_csv(combined_variance_sweep_path, index=False)
        combined_variance_predictions.to_csv(combined_variance_predictions_path, index=False)
        print(f"Saved combined epoch points: {combined_epoch_points_path}")
        print(f"Saved combined epoch threshold sweep: {combined_epoch_sweep_path}")
        print(f"Saved combined epoch predictions: {combined_epoch_predictions_path}")
        print(f"Saved combined variance features: {combined_variance_features_path}")
        print(f"Saved combined variance threshold sweep: {combined_variance_sweep_path}")
        print(f"Saved combined variance predictions: {combined_variance_predictions_path}")


if __name__ == "__main__":
    main()
