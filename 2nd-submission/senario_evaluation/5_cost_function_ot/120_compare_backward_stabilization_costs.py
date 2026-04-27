from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "logs_120"
DEFAULT_OUT_DIR = ROOT_DIR / "results"
SCENARIOS: dict[str, list[tuple[str, str]]] = {
    "backward_stabilization": [
        ("clean", "backward_stabilization/clean"),
        ("blurring", "backward_stabilization/blurring"),
    ],
    "weight_normalization": [
        ("clean", "weight_normalization/clean"),
        ("blurring", "weight_normalization/blurring"),
    ],
    "label_smoothing": [
        ("clean", "label_smooth/clean"),
        ("blurring", "label_smooth/blurring"),
    ],
    "model_pruning": [
        ("clean", "model_pruning/clean"),
        ("blurring", "model_pruning/blurring"),
    ],
    "batch_norm": [
        ("clean", "batch_norm/clean"),
        ("blurring", "batch_norm/blurring"),
    ],
    "adamw_weight_decay": [
        ("clean", "adamw_weight_decay/clean"),
        ("blurring", "adamw_weight_decay/blurring"),
    ],
    "base_clean_vs_blurring": [
        ("baseclean", "baseclean"),
        ("baseblurring", "baseblurring"),
    ],
    "base_clean_vs_augmentation_vs_blurring": [
        ("baseclean", "baseclean"),
        ("augmentation", "data_augmentation/clean"),
        ("baseblurring", "baseblurring"),
    ],
    "base_clean_vs_ood_vs_blurring": [
        ("baseclean", "baseclean"),
        ("ood", "OOD_data_training/clean"),
        ("baseblurring", "baseblurring"),
    ],
}
DEFAULT_COST_FUNCTIONS = (
    "c1_value",
    "c2_delta",
    "c3_time",
    "c4_time_value",
    "c5_frequency",
    "c6_curvature",
    "c7_window",
    "c8_shape_zscore",
    "c9_local_variance",
)
COST_LABELS = {
    "c1_value": "c1 value",
    "c2_delta": "c2 delta",
    "c3_time": "c3 time",
    "c4_time_value": "c4 time+value",
    "c5_frequency": "c5 frequency",
    "c6_curvature": "c6 curvature",
    "c7_window": "c7 local window",
    "c8_shape_zscore": "c8 z-score shape",
    "c9_local_variance": "c9 local variance",
}
CPU_DISTANCE_COL = "core_ot_distance_mean_to_baseclean"
MEM_DISTANCE_COL = "mem_ot_distance_to_baseclean"
CPU_SIMILARITY_COL = "core_similarity_mean_to_template"
MEM_SIMILARITY_COL = "mem_similarity_to_template"
HEATMAP_SIZE = 32


def _load_ot_module() -> Any:
    module_path = ROOT_DIR / "011_unbinned_ot_analysis.py"
    spec = importlib.util.spec_from_file_location("cost_function_ot_011", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OT module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csvs_for_case(case: str) -> list[Path]:
    paths = sorted((DATA_ROOT / case).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV logs found for case: {DATA_ROOT / case}")
    return paths


def _run_label(path: Path, prefix: str) -> str:
    return f"{prefix}_{path.stem}"


def _run_spec(group: str, label: str, path: Path) -> str:
    return f"{group}:{label}={path}"


def _group_name(scenario_name: str, variant_name: str) -> str:
    return f"{scenario_name}_{variant_name}"


def _build_input_specs(
    scenario_name: str,
    variants: list[tuple[str, str]],
) -> tuple[str, str, list[str], set[tuple[str, str]], Path, list[str]]:
    if len(variants) < 2:
        raise ValueError(f"Scenario needs at least reference and target variants: {scenario_name}")

    variant_csvs = [(variant_name, _csvs_for_case(case)) for variant_name, case in variants]
    reference_variant, reference_csvs = variant_csvs[0]
    reference_path = reference_csvs[0]
    reference_group = _group_name(scenario_name, reference_variant)
    reference = _run_spec(reference_group, "reference", reference_path)

    template_variant, template_csvs = variant_csvs[1]
    template_group = _group_name(scenario_name, template_variant)
    template_path = template_csvs[0]
    template_label = _run_label(template_path, template_variant)
    template = _run_spec(template_group, template_label, template_path)

    run_specs: list[str] = []
    target_keys: set[tuple[str, str]] = {(template_group, template_label)}
    target_groups = [_group_name(scenario_name, variant_name) for variant_name, _ in variants]

    for variant_name, csvs in variant_csvs:
        group = _group_name(scenario_name, variant_name)
        start_idx = 1 if group == reference_group else 0
        for path in csvs[start_idx:]:
            if group == template_group and path == template_path:
                continue
            label = _run_label(path, variant_name)
            run_specs.append(_run_spec(group, label, path))
            target_keys.add((group, label))

    return reference, template, run_specs, target_keys, reference_path, target_groups


def _summarize_groups(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cost_function, target_group), group in scores.groupby(["cost_function", "target_group"], sort=False):
        rows.append(
            {
                "cost_function": cost_function,
                "target_group": target_group,
                "num_epoch_rows": len(group),
                "ot_distance_mean": float(group["ot_distance"].mean()),
                "ot_distance_std": float(group["ot_distance"].std(ddof=0)),
                "ot_distance_median": float(group["ot_distance"].median()),
                "mem_ot_distance_mean": float(group["mem_ot_distance_to_baseclean"].mean()),
                "core_ot_distance_mean": float(group["core_ot_distance_mean_to_baseclean"].mean()),
                "similarity_score_mean": float(group["similarity_score"].mean()),
                "similarity_score_std": float(group["similarity_score"].std(ddof=0)),
                "mem_similarity_mean": float(group["mem_similarity_to_template"].mean()),
                "core_similarity_mean": float(group["core_similarity_mean_to_template"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _variant_label(group_name: str, scenario_name: str) -> str:
    prefix = f"{scenario_name}_"
    label = group_name.removeprefix(prefix)
    return label.replace("_", " ")


def _plot_epoch_trends(
    scores: pd.DataFrame,
    scenario_name: str,
    target_groups: list[str],
    out_path: Path,
    distance_col: str = "ot_distance",
    y_label: str = "Mean OT distance",
    title_suffix: str = "OT distance",
) -> None:
    costs = list(dict.fromkeys(scores["cost_function"].tolist()))
    palette = ["#4c78a8", "#e45756", "#54a24b", "#f58518", "#b279a2", "#72b7b2", "#ff9da6", "#9d755d"]
    colors = {group: palette[idx % len(palette)] for idx, group in enumerate(target_groups)}
    labels = {group: _variant_label(group, scenario_name) for group in target_groups}

    n_cols = 2
    n_rows = int(np.ceil(len(costs) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 3.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for ax, cost in zip(axes.ravel(), costs):
        cost_scores = scores.loc[scores["cost_function"] == cost].copy()
        for group_name in target_groups:
            group_scores = cost_scores.loc[cost_scores["target_group"] == group_name]
            if group_scores.empty:
                continue
            trend = (
                group_scores.groupby("epoch", as_index=False)[distance_col]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("epoch")
            )
            x = trend["epoch"].to_numpy(dtype=float)
            y = trend["mean"].to_numpy(dtype=float)
            std = trend["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                color=colors[group_name],
                label=labels[group_name],
            )
            ax.fill_between(x, y - std, y + std, color=colors[group_name], alpha=0.14, linewidth=0)
        ax.set_title(COST_LABELS.get(cost, cost))
        ax.set_xlabel("Epoch index")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    for ax in axes.ravel()[len(costs) :]:
        ax.axis("off")

    title = scenario_name.replace("_", " ")
    fig.suptitle(f"{title} {title_suffix} trend by epoch", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _trial_title(row: pd.Series, scenario_name: str) -> str:
    variant = _variant_label(str(row["target_group"]), scenario_name)
    stem = Path(str(row["target_run_csv"])).stem
    return f"{variant}: {stem}"


def _paired_trials(
    cost_scores: pd.DataFrame,
    scenario_name: str,
    target_groups: list[str],
) -> list[tuple[pd.Series, pd.Series]]:
    if len(target_groups) < 2:
        return []

    clean_group = target_groups[0]
    poisoned_group = target_groups[-1]

    trial_cols = ["target_group", "target_label", "target_run_csv"]
    clean_trials = (
        cost_scores.loc[cost_scores["target_group"] == clean_group, trial_cols]
        .drop_duplicates()
        .sort_values("target_label")
    )
    poisoned_trials = (
        cost_scores.loc[cost_scores["target_group"] == poisoned_group, trial_cols]
        .drop_duplicates()
        .sort_values("target_label")
    )
    pair_count = min(len(clean_trials), len(poisoned_trials))
    return [
        (clean_trials.iloc[idx], poisoned_trials.iloc[idx])
        for idx in range(pair_count)
    ]


def _paired_title(
    clean_trial: pd.Series,
    poisoned_trial: pd.Series,
    scenario_name: str,
    pair_idx: int,
) -> str:
    clean_variant = _variant_label(str(clean_trial["target_group"]), scenario_name)
    poisoned_variant = _variant_label(str(poisoned_trial["target_group"]), scenario_name)
    clean_stem = Path(str(clean_trial["target_run_csv"])).stem
    poisoned_stem = Path(str(poisoned_trial["target_run_csv"])).stem
    return f"pair {pair_idx}: {clean_variant} {clean_stem} vs {poisoned_variant} {poisoned_stem}"


def _plot_cost_trial_epoch_rows(
    scores: pd.DataFrame,
    scenario_name: str,
    cost_function: str,
    target_groups: list[str],
    out_path: Path,
) -> None:
    cost_scores = scores.loc[scores["cost_function"] == cost_function].copy()
    pairs = _paired_trials(cost_scores, scenario_name, target_groups)
    if not pairs:
        return

    n_pairs = len(pairs)
    n_rows = n_pairs * 2

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12, max(1.85 * n_rows, 5.0)),
        sharex=True,
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).ravel()

    clean_label = _variant_label(target_groups[0], scenario_name)
    poisoned_label = _variant_label(target_groups[-1], scenario_name)
    line_specs = (
        (clean_label, "#4c78a8", "o"),
        (poisoned_label, "#e45756", "s"),
    )

    for pair_idx, (clean_trial, poisoned_trial) in enumerate(pairs, start=1):
        for block_idx, (distance_col, signal_name) in enumerate(
            ((CPU_DISTANCE_COL, "CPU raw OT"), (MEM_DISTANCE_COL, "memory raw OT"))
        ):
            ax = axes_arr[(pair_idx - 1) + (block_idx * n_pairs)]
            for trial, (line_label, color, marker) in zip((clean_trial, poisoned_trial), line_specs):
                run_scores = cost_scores.loc[
                    (cost_scores["target_group"] == trial["target_group"])
                    & (cost_scores["target_label"] == trial["target_label"])
                ].sort_values("epoch")
                ax.plot(
                    run_scores["epoch"].to_numpy(dtype=float),
                    run_scores[distance_col].to_numpy(dtype=float),
                    marker=marker,
                    linewidth=1.8,
                    markersize=3.0,
                    color=color,
                    label=line_label,
                )

            ax.set_ylabel(signal_name)
            ax.set_title(
                _paired_title(clean_trial, poisoned_trial, scenario_name, pair_idx),
                loc="left",
                fontsize=9,
            )
            ax.grid(alpha=0.25)
            ax.margins(x=0.02)

    axes_arr[-1].set_xlabel("Epoch index")
    handles, legend_labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.98, 0.99))
    fig.suptitle(
        f"{scenario_name.replace('_', ' ')} {COST_LABELS.get(cost_function, cost_function)} paired clean vs blurring OT",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_measure_index(ot: Any, df: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, dict[int, dict[str, np.ndarray]]]]:
    prepared = ot._prepare_df(df)
    measures: dict[tuple[str, str, str], dict[str, dict[int, dict[str, np.ndarray]]]] = {}
    signal_names = ["mem", *ot.CORE_COLS]

    for key, group in prepared.groupby(["source_group", "source_label", "run_csv", "epoch"], dropna=False):
        source_group, source_label, run_csv, epoch = key
        run_key = (str(source_group), str(source_label), str(run_csv))
        measures.setdefault(run_key, {signal_name: {} for signal_name in signal_names})
        epoch_idx = int(epoch)
        measures[run_key]["mem"][epoch_idx] = ot._epoch_measure(group, "mem_percent")
        for col in ot.CORE_COLS:
            measures[run_key][col][epoch_idx] = ot._epoch_measure(group, col)

    return measures


def _reference_measures(
    ot: Any,
    measure_index: dict[tuple[str, str, str], dict[str, dict[int, dict[str, np.ndarray]]]],
    ref_key: tuple[str, str, str],
) -> dict[str, dict[str, np.ndarray]]:
    ref_run = measure_index[ref_key]
    return {
        signal_name: ot._aggregate_run_measure([ref_run[signal_name][epoch] for epoch in sorted(ref_run[signal_name])])
        for signal_name in ["mem", *ot.CORE_COLS]
    }


def _average_transport_plan(
    ot: Any,
    ref_measures: dict[str, dict[str, np.ndarray]],
    target_run: dict[str, dict[int, dict[str, np.ndarray]]],
    signal_kind: str,
    cost_function: str,
    reg_scale: float,
    alpha: float,
    beta: float,
    window_size: int,
    z_normalize_window: bool,
) -> np.ndarray:
    signal_names = list(ot.CORE_COLS) if signal_kind == "cpu" else ["mem"]
    plans: list[np.ndarray] = []

    for signal_name in signal_names:
        ref_measure = ref_measures[signal_name]
        for epoch_idx in sorted(target_run[signal_name]):
            target_measure = target_run[signal_name][epoch_idx]
            plan, _, _ = ot._transport_plan(
                ref_measure,
                target_measure,
                reg_scale=float(reg_scale),
                alpha=float(alpha),
                beta=float(beta),
                window_size=int(window_size),
                cost_function=cost_function,
                z_normalize_window=bool(z_normalize_window),
            )
            plans.append(ot._resample_matrix(plan, out_rows=HEATMAP_SIZE, out_cols=HEATMAP_SIZE))

    if not plans:
        return np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=float)
    return np.mean(np.stack(plans, axis=0), axis=0)


def _plot_transport_plan_heatmap_rows(
    ot: Any,
    df: pd.DataFrame,
    scores: pd.DataFrame,
    scenario_name: str,
    cost_function: str,
    target_groups: list[str],
    out_path: Path,
    reg_scale: float,
    alpha: float,
    beta: float,
    window_size: int,
    z_normalize_window: bool,
    measure_index: dict[tuple[str, str, str], dict[str, dict[int, dict[str, np.ndarray]]]] | None = None,
    ref_measures: dict[str, dict[str, np.ndarray]] | None = None,
) -> None:
    cost_scores = scores.loc[scores["cost_function"] == cost_function].copy()
    pairs = _paired_trials(cost_scores, scenario_name, target_groups)
    if not pairs or cost_scores.empty:
        return

    first_row = cost_scores.iloc[0]
    ref_key = (
        str(first_row["reference_group"]),
        str(first_row["reference_label"]),
        str(first_row["reference_run_csv"]),
    )
    if measure_index is None:
        measure_index = _build_measure_index(ot, df)
    if ref_key not in measure_index:
        return
    if ref_measures is None:
        ref_measures = _reference_measures(ot, measure_index, ref_key)

    n_pairs = len(pairs)
    n_rows = n_pairs * 2
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(12, max(2.2 * n_rows, 5.5)),
        constrained_layout=True,
        squeeze=False,
    )

    clean_label = _variant_label(target_groups[0], scenario_name)
    poisoned_label = _variant_label(target_groups[-1], scenario_name)
    column_titles = (f"{clean_label} coupling", f"{poisoned_label} coupling", "poisoned - clean")

    for pair_idx, (clean_trial, poisoned_trial) in enumerate(pairs, start=1):
        for block_idx, signal_kind in enumerate(("cpu", "memory")):
            row_idx = (pair_idx - 1) + (block_idx * n_pairs)
            clean_key = (
                str(clean_trial["target_group"]),
                str(clean_trial["target_label"]),
                str(clean_trial["target_run_csv"]),
            )
            poisoned_key = (
                str(poisoned_trial["target_group"]),
                str(poisoned_trial["target_label"]),
                str(poisoned_trial["target_run_csv"]),
            )
            if clean_key not in measure_index or poisoned_key not in measure_index:
                continue

            clean_plan = _average_transport_plan(
                ot,
                ref_measures,
                measure_index[clean_key],
                signal_kind="cpu" if signal_kind == "cpu" else "memory",
                cost_function=cost_function,
                reg_scale=reg_scale,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
                z_normalize_window=z_normalize_window,
            )
            poisoned_plan = _average_transport_plan(
                ot,
                ref_measures,
                measure_index[poisoned_key],
                signal_kind="cpu" if signal_kind == "cpu" else "memory",
                cost_function=cost_function,
                reg_scale=reg_scale,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
                z_normalize_window=z_normalize_window,
            )
            diff = poisoned_plan - clean_plan
            plan_vmax = max(float(np.nanmax(clean_plan)), float(np.nanmax(poisoned_plan)), 1e-12)
            diff_abs = max(float(np.nanmax(np.abs(diff))), 1e-12)

            mats = (clean_plan, poisoned_plan, diff)
            cmaps = ("viridis", "viridis", "coolwarm")
            ranges = ((0.0, plan_vmax), (0.0, plan_vmax), (-diff_abs, diff_abs))
            for col_idx, (mat, cmap, value_range) in enumerate(zip(mats, cmaps, ranges)):
                ax = axes[row_idx, col_idx]
                im = ax.imshow(
                    mat,
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    vmin=value_range[0],
                    vmax=value_range[1],
                )
                if row_idx == 0:
                    ax.set_title(column_titles[col_idx], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 2:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

            axes[row_idx, 0].set_ylabel(
                f"pair {pair_idx}\n{signal_kind}",
                rotation=0,
                ha="right",
                va="center",
                fontsize=9,
            )

    fig.suptitle(
        f"{scenario_name.replace('_', ' ')} {COST_LABELS.get(cost_function, cost_function)} transport plan coupling",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_group_means(
    summary: pd.DataFrame,
    scenario_name: str,
    target_groups: list[str],
    out_path: Path,
) -> None:
    pivot = summary.pivot(index="cost_function", columns="target_group", values="ot_distance_mean")
    pivot = pivot.reindex(DEFAULT_COST_FUNCTIONS)
    labels = [COST_LABELS.get(cost, cost) for cost in pivot.index]
    x = np.arange(len(pivot.index))
    width = min(0.8 / max(len(target_groups), 1), 0.32)
    offsets = (np.arange(len(target_groups)) - (len(target_groups) - 1) / 2.0) * width
    palette = ["#4c78a8", "#e45756", "#54a24b", "#f58518", "#b279a2", "#72b7b2"]

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for idx, group in enumerate(target_groups):
        vals = pivot.get(group, pd.Series(index=pivot.index, dtype=float))
        ax.bar(
            x + offsets[idx],
            vals.to_numpy(dtype=float),
            width=width,
            label=_variant_label(group, scenario_name),
            color=palette[idx % len(palette)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean OT distance")
    ax.set_title(f"{scenario_name.replace('_', ' ')} mean OT distance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_signal_breakdown(
    summary: pd.DataFrame,
    scenario_name: str,
    target_groups: list[str],
    out_path: Path,
) -> None:
    long_rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        case = _variant_label(str(row["target_group"]), scenario_name)
        long_rows.append(
            {
                "cost_function": row["cost_function"],
                "case": case,
                "signal": "memory",
                "distance": row["mem_ot_distance_mean"],
            }
        )
        long_rows.append(
            {
                "cost_function": row["cost_function"],
                "case": case,
                "signal": "cpu cores",
                "distance": row["core_ot_distance_mean"],
            }
        )
    data = pd.DataFrame(long_rows)

    costs = list(DEFAULT_COST_FUNCTIONS)
    x = np.arange(len(costs))
    labels = [_variant_label(group, scenario_name) for group in target_groups]
    palette = ["#4c78a8", "#e45756", "#54a24b", "#f58518", "#b279a2", "#72b7b2"]
    series = [(label, "memory") for label in labels] + [(label, "cpu cores") for label in labels]
    width = min(0.8 / max(len(series), 1), 0.16)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for idx, (case, signal) in enumerate(series):
        vals = []
        for cost in costs:
            match = data[
                (data["case"] == case)
                & (data["signal"] == signal)
                & (data["cost_function"] == cost)
            ]["distance"]
            vals.append(float(match.iloc[0]) if not match.empty else np.nan)
        alpha = 0.62 if signal == "memory" else 1.0
        hatch = "//" if signal == "memory" else None
        ax.bar(
            x + offsets[idx],
            vals,
            width=width,
            color=palette[labels.index(case) % len(palette)],
            alpha=alpha,
            hatch=hatch,
            label=f"{case} {signal}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([COST_LABELS.get(cost, cost) for cost in costs], rotation=20, ha="right")
    ax.set_ylabel("Mean OT distance")
    ax.set_title(f"{scenario_name.replace('_', ' ')} signal contribution by cost")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", ncols=2)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--reg-scale", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--window-size", type=int, default=5)
    p.add_argument("--z-normalize-window", action="store_true")
    p.add_argument(
        "--make-heatmaps",
        action="store_true",
        help="also generate paired transport-plan heatmaps in each scenario backup directory",
    )
    p.add_argument(
        "--scenario",
        action="append",
        choices=list(SCENARIOS),
        help="repeat to run selected scenarios only; default runs all configured comparisons",
    )
    p.add_argument(
        "--cost-function",
        action="append",
        choices=list(DEFAULT_COST_FUNCTIONS),
        help="repeat to run selected costs only; default runs c1 through c9",
    )
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ot = _load_ot_module()
    cost_functions = tuple(args.cost_function) if args.cost_function else DEFAULT_COST_FUNCTIONS
    scenario_names = tuple(args.scenario) if args.scenario else tuple(SCENARIOS)
    all_scenario_scores: list[pd.DataFrame] = []
    all_scenario_summaries: list[pd.DataFrame] = []
    outputs: list[dict[str, str]] = []

    for scenario_name in scenario_names:
        variants = SCENARIOS[scenario_name]
        scenario_out_dir = out_dir / scenario_name
        scenario_out_dir.mkdir(parents=True, exist_ok=True)

        reference, template, run_specs, target_keys, reference_path, target_groups = _build_input_specs(
            scenario_name=scenario_name,
            variants=variants,
        )
        reference_group, reference_label, _ = ot.parse_run_spec(reference)
        template_group, template_label, _ = ot.parse_run_spec(template)
        df = ot.load_logs(reference, [template, *run_specs], device_id=f"{scenario_name}_costs")

        frames: list[pd.DataFrame] = []
        for cost_function in cost_functions:
            scores = ot.build_summary(
                df=df,
                reference_group=reference_group,
                reference_label=reference_label,
                template_group=template_group,
                template_label=template_label,
                target_keys=target_keys,
                reg_scale=float(args.reg_scale),
                alpha=float(args.alpha),
                beta=float(args.beta),
                window_size=int(args.window_size),
                cost_function=cost_function,
                z_normalize_window=bool(args.z_normalize_window),
            )
            scores.insert(0, "scenario", scenario_name)
            scores.insert(1, "reference_csv", str(reference_path))
            frames.append(scores)

        all_scores = pd.concat(frames, ignore_index=True)
        summary = _summarize_groups(all_scores)
        summary.insert(0, "scenario", scenario_name)

        backup_out_dir = scenario_out_dir / "backup"
        backup_out_dir.mkdir(exist_ok=True)
        scores_path = scenario_out_dir / f"{scenario_name}_cost_function_scores.csv"
        summary_path = scenario_out_dir / f"{scenario_name}_cost_function_summary.csv"
        all_scores.to_csv(scores_path, index=False)
        summary.to_csv(summary_path, index=False)

        _plot_epoch_trends(
            all_scores,
            scenario_name,
            target_groups,
            scenario_out_dir / f"{scenario_name}_cost_function_cpu_epoch_trends.png",
            distance_col=CPU_DISTANCE_COL,
            y_label="Mean CPU OT distance",
            title_suffix="CPU OT distance",
        )
        _plot_epoch_trends(
            all_scores,
            scenario_name,
            target_groups,
            scenario_out_dir / f"{scenario_name}_cost_function_memory_epoch_trends.png",
            distance_col=MEM_DISTANCE_COL,
            y_label="Mean memory OT distance",
            title_suffix="memory OT distance",
        )
        _plot_epoch_trends(
            all_scores,
            scenario_name,
            target_groups,
            scenario_out_dir / f"{scenario_name}_cost_function_cpu_similarity_epoch_trends.png",
            distance_col=CPU_SIMILARITY_COL,
            y_label="Mean CPU similarity score",
            title_suffix="CPU similarity score",
        )
        _plot_epoch_trends(
            all_scores,
            scenario_name,
            target_groups,
            scenario_out_dir / f"{scenario_name}_cost_function_memory_similarity_epoch_trends.png",
            distance_col=MEM_SIMILARITY_COL,
            y_label="Mean memory similarity score",
            title_suffix="memory similarity score",
        )
        _plot_group_means(
            summary,
            scenario_name,
            target_groups,
            backup_out_dir / f"{scenario_name}_cost_function_means.png",
        )
        _plot_signal_breakdown(
            summary,
            scenario_name,
            target_groups,
            backup_out_dir / f"{scenario_name}_signal_breakdown.png",
        )
        heatmap_measure_index = None
        heatmap_ref_measures = None
        for cost_function in cost_functions:
            _plot_cost_trial_epoch_rows(
                all_scores,
                scenario_name,
                cost_function,
                target_groups,
                backup_out_dir / f"{scenario_name}_{cost_function}_signal_epoch_rows.png",
            )
            if args.make_heatmaps:
                if heatmap_measure_index is None:
                    heatmap_measure_index = _build_measure_index(ot, df)
                    heatmap_ref_key = (
                        str(reference_group),
                        str(reference_label),
                        str(reference_path),
                    )
                    heatmap_ref_measures = _reference_measures(ot, heatmap_measure_index, heatmap_ref_key)
                _plot_transport_plan_heatmap_rows(
                    ot,
                    df,
                    all_scores,
                    scenario_name,
                    cost_function,
                    target_groups,
                    backup_out_dir / f"{scenario_name}_{cost_function}_transport_plan_heatmaps.png",
                    reg_scale=float(args.reg_scale),
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    window_size=int(args.window_size),
                    z_normalize_window=bool(args.z_normalize_window),
                    measure_index=heatmap_measure_index,
                    ref_measures=heatmap_ref_measures,
                )

        all_scenario_scores.append(all_scores)
        all_scenario_summaries.append(summary)
        outputs.append(
            {
                "scenario": scenario_name,
                "scores_csv": str(scores_path),
                "summary_csv": str(summary_path),
            }
        )

    combined_scores = pd.concat(all_scenario_scores, ignore_index=True)
    combined_summary = pd.concat(all_scenario_summaries, ignore_index=True)
    combined_scores_path = out_dir / "all_scenarios_cost_function_scores.csv"
    combined_summary_path = out_dir / "all_scenarios_cost_function_summary.csv"
    combined_scores.to_csv(combined_scores_path, index=False)
    combined_summary.to_csv(combined_summary_path, index=False)

    print(
        {
            "combined_scores_csv": str(combined_scores_path),
            "combined_summary_csv": str(combined_summary_path),
            "num_rows": int(len(combined_scores)),
            "scenarios": list(scenario_names),
            "cost_functions": list(cost_functions),
            "outputs": outputs,
        }
    )


if __name__ == "__main__":
    main()
