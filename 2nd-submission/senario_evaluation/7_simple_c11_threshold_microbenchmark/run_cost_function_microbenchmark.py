from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = SCRIPT_DIR.parent
COST_OT_DIR = SCENARIO_DIR / "5_cost_function_ot"
COST_FUNCTION = "c11"
WINDOW = 1


@dataclass(frozen=True)
class CaseSelector:
    case: str
    label: str
    poison_types: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    base_clean_case: CaseSelector
    clean_cases: tuple[CaseSelector, ...]
    poisoned_cases: tuple[CaseSelector, ...]


@dataclass(frozen=True)
class SignalMix:
    name: str
    core_weight: float
    mem_weight: float


SCENARIOS = {
    "base": ScenarioConfig(
        name="base",
        base_clean_case=CaseSelector("baseclean", "clean", ("clean",)),
        clean_cases=(CaseSelector("baseclean", "clean", ("clean",)),),
        poisoned_cases=(CaseSelector("baseblurring", "poisoned", ("blurring",)),),
    ),
    "adamw_weight_decay": ScenarioConfig(
        name="adamw_weight_decay",
        base_clean_case=CaseSelector("adamw_weight_decay/clean", "clean", ("clean",)),
        clean_cases=(CaseSelector("adamw_weight_decay/clean", "clean", ("clean",)),),
        poisoned_cases=(CaseSelector("adamw_weight_decay/blurring", "poisoned", ("blurring",)),),
    ),
    "batch_norm": ScenarioConfig(
        name="batch_norm",
        base_clean_case=CaseSelector("batch_norm/clean", "clean", ("clean",)),
        clean_cases=(CaseSelector("batch_norm/clean", "clean", ("clean",)),),
        poisoned_cases=(CaseSelector("batch_norm/blurring", "poisoned", ("blurring",)),),
    ),
    "label_smooth": ScenarioConfig(
        name="label_smooth",
        base_clean_case=CaseSelector("label_smooth/clean", "clean", ("clean",)),
        clean_cases=(CaseSelector("label_smooth/clean", "clean", ("clean",)),),
        poisoned_cases=(CaseSelector("label_smooth/blurring", "poisoned", ("blurring",)),),
    ),
    "model_pruning": ScenarioConfig(
        name="model_pruning",
        base_clean_case=CaseSelector("model_pruning/clean", "clean", ("clean",)),
        clean_cases=(CaseSelector("model_pruning/clean", "clean", ("clean",)),),
        poisoned_cases=(CaseSelector("model_pruning/blurring", "poisoned", ("blurring",)),),
    ),
    "weight_normalization": ScenarioConfig(
        name="weight_normalization",
        base_clean_case=CaseSelector("weight_normalization/clean", "clean", ("clean",)),
        clean_cases=(CaseSelector("weight_normalization/clean", "clean", ("clean",)),),
        poisoned_cases=(CaseSelector("weight_normalization/blurring", "poisoned", ("blurring",)),),
    ),
    "backward_stabilization": ScenarioConfig(
        name="backward_stabilization",
        base_clean_case=CaseSelector("backward_stabilization/clean", "clean", ("clean",)),
        clean_cases=(CaseSelector("backward_stabilization/clean", "clean", ("clean",)),),
        poisoned_cases=(CaseSelector("backward_stabilization/blurring", "poisoned", ("blurring",)),),
    ),
    "ood": ScenarioConfig(
        name="ood",
        base_clean_case=CaseSelector("baseclean", "clean", ("clean",)),
        clean_cases=(CaseSelector("OOD_data_training/clean", "clean", ("ood",)),),
        poisoned_cases=(CaseSelector("baseblurring", "poisoned", ("blurring",)),),
    ),
    "data_augmentation": ScenarioConfig(
        name="data_augmentation",
        base_clean_case=CaseSelector("baseclean", "clean", ("clean",)),
        clean_cases=(CaseSelector("data_augmentation/clean", "clean", ("augmentation",)),),
        poisoned_cases=(CaseSelector("baseblurring", "poisoned", ("blurring",)),),
    ),
    "data_augmentation_aug_ref": ScenarioConfig(
        name="data_augmentation_aug_ref",
        base_clean_case=CaseSelector("data_augmentation/clean", "clean", ("augmentation",)),
        clean_cases=(CaseSelector("data_augmentation/clean", "clean", ("augmentation",)),),
        poisoned_cases=(CaseSelector("baseblurring", "poisoned", ("blurring",)),),
    ),
}


SIGNAL_MIXES = (
    SignalMix("cpu80_mem20", 0.8, 0.2),
    SignalMix("cpu100_mem0", 1.0, 0.0),
    SignalMix("cpu0_mem100", 0.0, 1.0),
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _analysis_module():
    if str(SCENARIO_DIR) not in sys.path:
        sys.path.insert(0, str(SCENARIO_DIR))
    return _load_module("scenario_cost_function_ot", COST_OT_DIR / "011_unbinned_ot_analysis.py")


def _run_info(csv_path: Path) -> dict[str, Any]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            event = (row.get("event") or "").strip()
            if not event.startswith("{"):
                continue
            try:
                payload = json.loads(event)
            except json.JSONDecodeError:
                continue
            run_info = payload.get("run_info")
            if isinstance(run_info, dict):
                return run_info
    raise ValueError(f"Could not find run_info in {csv_path}")


def _case_csvs(root: Path, selector: CaseSelector) -> list[Path]:
    case_dir = root / selector.case
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case directory: {case_dir}")

    selected = []
    accepted = set(selector.poison_types)
    for csv_path in sorted(case_dir.glob("*.csv")):
        poison_type = str(_run_info(csv_path).get("poison_type", "")).strip()
        if poison_type == "none":
            poison_type = "clean"
        if poison_type in accepted:
            selected.append(csv_path)
    return selected


def _make_spec(case_name: str, role: str, index: int, csv_path: Path) -> tuple[str, str]:
    label = f"{role}_{index:02d}_{csv_path.stem}"
    return f"{case_name}:{label}={csv_path.resolve()}", label


def _convert_summary(
    summary: pd.DataFrame,
    scenario: ScenarioConfig,
    reference_csv: Path,
    ref_csv: Path,
    label_by_target: dict[tuple[str, str], str],
) -> pd.DataFrame:
    out = summary.rename(
        columns={
            "poison_frac": "poisoning_rate",
            "epoch": "round",
            "core_ot_distance_mean_to_baseclean": "core_ot_distance_mean_to_base",
            "mem_ot_distance_to_baseclean": "mem_ot_distance_to_base",
        }
    ).copy()
    out["scenario"] = scenario.name
    out["cost_function"] = COST_FUNCTION
    out["base_run_csv"] = str(reference_csv.resolve())
    out["ref_run_csv"] = str(ref_csv.resolve())
    out["classification_label"] = [
        label_by_target[(str(row["target_group"]), str(row["target_label"]))]
        for _, row in out.iterrows()
    ]

    columns = [
        "scenario",
        "cost_function",
        "device_id",
        "target_group",
        "target_label",
        "target_run_csv",
        "base_run_csv",
        "ref_run_csv",
        "poisoning_type",
        "classification_label",
        "poisoning_rate",
        "round",
        "core_ot_distance_mean_to_base",
        "mem_ot_distance_to_base",
    ]
    return out[[c for c in columns if c in out.columns]]


def build_scenario_summary(
    root: Path,
    scenario: ScenarioConfig,
    bins: int,
    reg_scale: float,
    alpha: float,
    beta: float,
    window_size: int,
    z_normalize_window: bool,
) -> pd.DataFrame:
    analysis = _analysis_module()
    base_clean_csvs = _case_csvs(root, scenario.base_clean_case)
    if len(base_clean_csvs) < 2:
        raise ValueError(
            f"{scenario.name}: need at least 2 reference runs in "
            f"{scenario.base_clean_case.case}, found {len(base_clean_csvs)}"
        )

    reference_csv = base_clean_csvs[0]
    ref_csv = base_clean_csvs[1]
    reference_spec, _ = _make_spec(scenario.base_clean_case.case, "base", 0, reference_csv)
    ref_spec, _ = _make_spec(scenario.base_clean_case.case, "ref", 1, ref_csv)
    excluded_base_refs = {reference_csv.resolve(), ref_csv.resolve()}

    run_specs = []
    label_by_target: dict[tuple[str, str], str] = {}
    for selector in [*scenario.clean_cases, *scenario.poisoned_cases]:
        for csv_path in _case_csvs(root, selector):
            if csv_path.resolve() in excluded_base_refs:
                continue
            spec, label = _make_spec(selector.case, selector.label, len(run_specs), csv_path)
            run_specs.append(spec)
            label_by_target[(selector.case, label)] = selector.label

    if not run_specs:
        raise ValueError(f"{scenario.name}: no target runs found")

    reference_group, reference_label, _ = analysis.parse_run_spec(reference_spec)
    ref_group, ref_label, _ = analysis.parse_run_spec(ref_spec)
    parsed_runs = [analysis.parse_run_spec(spec) for spec in run_specs]
    target_keys = {(group, label) for group, label, _ in parsed_runs}

    df = analysis.load_logs(
        reference_spec,
        [ref_spec, *run_specs],
        device_id=f"{scenario.name}_{COST_FUNCTION}_device",
    )
    summary = analysis.build_summary(
        df=df,
        reference_group=reference_group,
        reference_label=reference_label,
        template_group=ref_group,
        template_label=ref_label,
        target_keys=target_keys,
        bins=bins,
        reg_scale=reg_scale,
        alpha=alpha,
        beta=beta,
        window_size=window_size,
        cost_function=COST_FUNCTION,
        z_normalize_window=z_normalize_window,
    )
    return _convert_summary(summary, scenario, reference_csv, ref_csv, label_by_target)


def apply_signal_mix(summary: pd.DataFrame, mix: SignalMix) -> pd.DataFrame:
    out = summary.copy()
    out["signal_mix"] = mix.name
    out["core_signal_weight"] = mix.core_weight
    out["mem_signal_weight"] = mix.mem_weight
    out["ot_distance"] = (
        mix.core_weight * out["core_ot_distance_mean_to_base"]
        + mix.mem_weight * out["mem_ot_distance_to_base"]
    )
    return out


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    tp = int(((y_true == "poisoned") & (y_pred == "poisoned")).sum())
    fp = int(((y_true == "clean") & (y_pred == "poisoned")).sum())
    tn = int(((y_true == "clean") & (y_pred == "clean")).sum())
    fn = int(((y_true == "poisoned") & (y_pred == "clean")).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "j": recall - fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
    }


def build_window1_points(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["window"] = WINDOW
    df["window_id"] = df["round"].astype(int)
    group_cols = [
        "scenario",
        "cost_function",
        "signal_mix",
        "core_signal_weight",
        "mem_signal_weight",
        "device_id",
        "target_run_csv",
        "target_group",
        "target_label",
        "poisoning_type",
        "classification_label",
        "poisoning_rate",
        "window",
        "window_id",
    ]
    return (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(score=("ot_distance", "mean"))
        .sort_values(["classification_label", "target_group", "target_label", "window_id"])
    )


def threshold_sweep(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = points["classification_label"]
    for threshold in sorted(points["score"].dropna().unique()):
        y_pred = pd.Series(
            ["poisoned" if score >= threshold else "clean" for score in points["score"]],
            index=points.index,
        )
        row = {
            "threshold": float(threshold),
            "rule": "poisoned_if_score_gte_threshold",
            "n_clean": int((y_true == "clean").sum()),
            "n_poisoned": int((y_true == "poisoned").sum()),
        }
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["j", "f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, False, True],
    )


def build_predictions(points: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    out = points.copy()
    if classification.empty:
        out["threshold"] = pd.NA
        out["predicted_label"] = pd.NA
        return out

    threshold = float(classification["threshold"].iloc[0])
    out["threshold"] = threshold
    out["predicted_label"] = [
        "poisoned" if score >= threshold else "clean"
        for score in out["score"]
    ]
    return out


def plot_window1_threshold(
    points: pd.DataFrame,
    classification: pd.DataFrame,
    out_path: Path,
) -> None:
    if classification.empty:
        return

    scenario = str(classification["scenario"].iloc[0])
    signal_mix = str(classification["signal_mix"].iloc[0])
    threshold = float(classification["threshold"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    colors = {"clean": "#1f77b4", "poisoned": "#d62728"}
    y_positions = {"clean": 0.0, "poisoned": 1.0}

    for label in ["clean", "poisoned"]:
        subset = points[points["classification_label"] == label].copy()
        if subset.empty:
            continue
        offsets = [((idx % 9) - 4) * 0.018 for idx in range(len(subset))]
        ys = [y_positions[label] + offset for offset in offsets]
        ax.scatter(
            subset["score"],
            ys,
            s=42,
            color=colors[label],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.78,
            label=label,
        )

    ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.6, label="threshold")
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["clean", "poisoned"])
    ax.set_xlabel("window-1 c11 OT score")
    ax.set_ylabel("true class")
    ax.set_title(f"{scenario}: {signal_mix}, window=1")
    ax.grid(axis="x", alpha=0.22)

    row = classification.iloc[0]
    score_text = (
        f"threshold = {threshold:.6g}\n"
        f"J = {row['j']:.3f}, F1 = {row['f1']:.3f}\n"
        f"precision = {row['precision']:.3f}, recall = {row['recall']:.3f}\n"
        f"FPR = {row['fpr']:.3f}, FNR = {row['fnr']:.3f}"
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
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_csv_list(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def _selected_scenarios(value: str) -> list[ScenarioConfig]:
    names = _parse_csv_list(value)
    if not names:
        return [SCENARIOS[name] for name in SCENARIOS]
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenario(s): {unknown}. Available: {sorted(SCENARIOS)}")
    return [SCENARIOS[name] for name in names]


def _clear_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple c11-only cost-function microbenchmark with window-1 threshold classification."
    )
    parser.add_argument("--root", default=str(SCENARIO_DIR / "logs_120"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--bins", type=int, default=0, help="kept for compatibility; unbinned OT ignores it")
    parser.add_argument("--reg-scale", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--z-normalize-window", action="store_true")
    parser.add_argument("--scenarios", default="", help=f"comma-separated subset; available: {','.join(SCENARIOS)}")
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if args.clean_output:
        _clear_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_points = []
    all_sweeps = []
    all_classifications = []
    all_predictions = []
    manifest_rows = []

    for scenario in _selected_scenarios(args.scenarios):
        base_summary = build_scenario_summary(
            root=root,
            scenario=scenario,
            bins=int(args.bins),
            reg_scale=float(args.reg_scale),
            alpha=float(args.alpha),
            beta=float(args.beta),
            window_size=int(args.window_size),
            z_normalize_window=bool(args.z_normalize_window),
        )

        for mix in SIGNAL_MIXES:
            summary = apply_signal_mix(base_summary, mix)
            points = build_window1_points(summary)
            sweep = threshold_sweep(points)
            classification = sweep.head(1).copy()
            predictions = build_predictions(points, classification)

            classification.insert(0, "window", WINDOW)
            classification.insert(0, "mem_signal_weight", mix.mem_weight)
            classification.insert(0, "core_signal_weight", mix.core_weight)
            classification.insert(0, "signal_mix", mix.name)
            classification.insert(0, "cost_function", COST_FUNCTION)
            classification.insert(0, "scenario", scenario.name)

            sweep.insert(0, "window", WINDOW)
            sweep.insert(0, "mem_signal_weight", mix.mem_weight)
            sweep.insert(0, "core_signal_weight", mix.core_weight)
            sweep.insert(0, "signal_mix", mix.name)
            sweep.insert(0, "cost_function", COST_FUNCTION)
            sweep.insert(0, "scenario", scenario.name)

            name = _safe_name(f"{scenario.name}__{COST_FUNCTION}__{mix.name}")
            summary_path = out_dir / f"{name}_summary.csv"
            points_path = out_dir / f"{name}_window1_points.csv"
            sweep_path = out_dir / f"{name}_threshold_sweep.csv"
            classification_path = out_dir / f"{name}_classification.csv"
            predictions_path = out_dir / f"{name}_predictions.csv"
            plot_path = out_dir / "plots" / f"{name}_window1.png"

            summary.to_csv(summary_path, index=False)
            points.to_csv(points_path, index=False)
            sweep.to_csv(sweep_path, index=False)
            classification.to_csv(classification_path, index=False)
            predictions.to_csv(predictions_path, index=False)
            plot_window1_threshold(points, classification, plot_path)

            all_summaries.append(summary)
            all_points.append(points)
            all_sweeps.append(sweep)
            all_classifications.append(classification)
            all_predictions.append(predictions)
            manifest_rows.append(
                {
                    "scenario": scenario.name,
                    "cost_function": COST_FUNCTION,
                    "signal_mix": mix.name,
                    "core_signal_weight": mix.core_weight,
                    "mem_signal_weight": mix.mem_weight,
                    "window": WINDOW,
                    "summary_csv": str(summary_path),
                    "window1_points_csv": str(points_path),
                    "threshold_sweep_csv": str(sweep_path),
                    "classification_csv": str(classification_path),
                    "predictions_csv": str(predictions_path),
                    "plot_png": str(plot_path),
                    "num_summary_rows": len(summary),
                    "num_window1_points": len(points),
                    "num_clean_points": int((points["classification_label"] == "clean").sum()),
                    "num_poisoned_points": int((points["classification_label"] == "poisoned").sum()),
                }
            )
            print(f"Saved {scenario.name} {mix.name}: {classification_path}")

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            out_dir / "all_scenarios_c11_summary.csv",
            index=False,
        )
    if all_points:
        pd.concat(all_points, ignore_index=True).to_csv(
            out_dir / "all_scenarios_c11_window1_points.csv",
            index=False,
        )
    if all_sweeps:
        pd.concat(all_sweeps, ignore_index=True).to_csv(
            out_dir / "all_scenarios_c11_threshold_sweep.csv",
            index=False,
        )
    if all_classifications:
        pd.concat(all_classifications, ignore_index=True).to_csv(
            out_dir / "all_scenarios_c11_classification.csv",
            index=False,
        )
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            out_dir / "all_scenarios_c11_predictions.csv",
            index=False,
        )

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
