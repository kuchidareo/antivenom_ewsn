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

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = SCRIPT_DIR.parent


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
        poisoned_cases=(
            CaseSelector("adamw_weight_decay/blurring", "poisoned", ("blurring",)),
        ),
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
        poisoned_cases=(
            CaseSelector("weight_normalization/blurring", "poisoned", ("blurring",)),
        ),
    ),
    "backward_stabilization": ScenarioConfig(
        name="backward_stabilization",
        base_clean_case=CaseSelector("backward_stabilization/clean", "clean", ("clean",)),
        clean_cases=(CaseSelector("backward_stabilization/clean", "clean", ("clean",)),),
        poisoned_cases=(
            CaseSelector("backward_stabilization/blurring", "poisoned", ("blurring",)),
        ),
    ),
    "ood": ScenarioConfig(
        name="ood",
        base_clean_case=CaseSelector("baseclean", "clean", ("clean",)),
        clean_cases=(
            CaseSelector("baseclean", "clean", ("clean",)),
            CaseSelector("OOD_data_training/clean", "clean", ("ood",)),
        ),
        poisoned_cases=(CaseSelector("baseblurring", "poisoned", ("blurring",)),),
    ),
    "data_augmentation": ScenarioConfig(
        name="data_augmentation",
        base_clean_case=CaseSelector("baseclean", "clean", ("clean",)),
        clean_cases=(
            CaseSelector("baseclean", "clean", ("clean",)),
            CaseSelector("data_augmentation/clean", "clean", ("augmentation",)),
        ),
        poisoned_cases=(CaseSelector("baseblurring", "poisoned", ("blurring",)),),
    ),
    "data_augmentation_aug_ref": ScenarioConfig(
        name="data_augmentation_aug_ref",
        base_clean_case=CaseSelector(
            "data_augmentation/clean", "clean", ("augmentation",)
        ),
        clean_cases=(CaseSelector("data_augmentation/clean", "clean", ("augmentation",)),),
        poisoned_cases=(CaseSelector("baseblurring", "poisoned", ("blurring",)),),
    ),
}


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
    return _load_module(
        "scenario_baseclean_analysis",
        SCENARIO_DIR / "01_cpu_core_sorted_mem_baseclean_analysis.py",
    )


def _classifier_module():
    return _load_module(
        "microbenchmark_classifier",
        SCRIPT_DIR / "classify_clean_poison_f1_window_sweep.py",
    )


def _plotter_module():
    return _load_module(
        "microbenchmark_plotter",
        SCRIPT_DIR / "plot_scenario_classification.py",
    )


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

    accepted = set(selector.poison_types)
    selected = []
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
            "core_type_cosine_mean_to_clean": "core_type_cosine_mean_to_ref",
            "mem_type_cosine_to_clean": "mem_type_cosine_to_ref",
        }
    ).copy()
    out["scenario"] = scenario.name
    out["base_run_csv"] = str(reference_csv.resolve())
    out["ref_run_csv"] = str(ref_csv.resolve())
    out["classification_label"] = [
        label_by_target[(str(row["target_group"]), str(row["target_label"]))]
        for _, row in out.iterrows()
    ]

    columns = [
        "scenario",
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
        "core_type_cosine_mean_to_ref",
        "mem_type_cosine_to_ref",
        "ot_distance",
        "cosine_similarity",
    ]
    return out[[c for c in columns if c in out.columns]]


def build_scenario_summary(
    root: Path,
    scenario: ScenarioConfig,
    bins: int,
) -> pd.DataFrame:
    analysis = _analysis_module()
    base_clean_csvs = _case_csvs(root, scenario.base_clean_case)
    if len(base_clean_csvs) < 2:
        raise ValueError(
            f"{scenario.name}: need at least 2 clean runs in "
            f"{scenario.base_clean_case.case}, found {len(base_clean_csvs)}"
        )

    reference_csv = base_clean_csvs[0]
    ref_csv = base_clean_csvs[1]
    reference_spec, _ = _make_spec(
        scenario.base_clean_case.case, "base", 0, reference_csv
    )
    ref_spec, _ = _make_spec(scenario.base_clean_case.case, "ref", 1, ref_csv)
    excluded_base_refs = {reference_csv.resolve(), ref_csv.resolve()}

    run_specs = []
    label_by_target: dict[tuple[str, str], str] = {}
    for selector in [*scenario.clean_cases, *scenario.poisoned_cases]:
        csvs = _case_csvs(root, selector)
        for csv_path in csvs:
            if csv_path.resolve() in excluded_base_refs:
                continue
            spec, label = _make_spec(
                selector.case,
                selector.label,
                len(run_specs),
                csv_path,
            )
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
        device_id=f"{scenario.name}_device",
    )
    summary = analysis.build_summary(
        df=df,
        reference_group=reference_group,
        reference_label=reference_label,
        template_group=ref_group,
        template_label=ref_label,
        target_keys=target_keys,
        bins=bins,
    )
    return _convert_summary(
        summary=summary,
        scenario=scenario,
        reference_csv=reference_csv,
        ref_csv=ref_csv,
        label_by_target=label_by_target,
    )


def _parse_windows(value: str) -> list[int]:
    windows = [int(s.strip()) for s in value.split(",") if s.strip()]
    if not windows:
        raise ValueError("--windows must list at least one integer")
    return windows


def _clear_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _selected_scenarios(value: str) -> list[ScenarioConfig]:
    if not value.strip():
        return [SCENARIOS[name] for name in SCENARIOS]
    names = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise ValueError(
            f"Unknown scenario(s): {unknown}. Available: {sorted(SCENARIOS)}"
        )
    return [SCENARIOS[name] for name in names]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(SCENARIO_DIR / "logs_120"))
    p.add_argument("--out-dir", default=str(SCRIPT_DIR / "results"))
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--grid-step", type=float, default=0.1)
    p.add_argument("--windows", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--plot-windows", default="1,5,10")
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete --out-dir before writing new summaries, classifications, and plots.",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write CSV outputs; do not generate threshold plots.",
    )
    p.add_argument(
        "--scenarios",
        default="",
        help=f"comma-separated subset; available: {','.join(SCENARIOS)}",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if args.clean_output:
        _clear_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    windows = _parse_windows(args.windows)
    plot_windows = _parse_windows(args.plot_windows)
    classifier = _classifier_module()

    all_perf = []
    manifest_rows = []
    for scenario in _selected_scenarios(args.scenarios):
        summary = build_scenario_summary(root=root, scenario=scenario, bins=int(args.bins))
        summary_path = out_dir / f"{scenario.name}_summary.csv"
        summary.to_csv(summary_path, index=False)

        perf = classifier.run_window_sweep(
            df=summary,
            windows=windows,
            grid_step=float(args.grid_step),
            calibration_fraction=float(args.calibration_fraction),
        )
        perf.insert(0, "scenario", scenario.name)
        perf_path = out_dir / f"{scenario.name}_classification.csv"
        perf.to_csv(perf_path, index=False)

        all_perf.append(perf)
        manifest_rows.append(
            {
                "scenario": scenario.name,
                "summary_csv": str(summary_path),
                "classification_csv": str(perf_path),
                "num_summary_rows": len(summary),
                "num_clean_rows": int((summary["classification_label"] == "clean").sum()),
                "num_poisoned_rows": int(
                    (summary["classification_label"] == "poisoned").sum()
                ),
            }
        )
        print(f"Saved {scenario.name}: {summary_path}")
        print(f"Saved {scenario.name}: {perf_path}")

    if all_perf:
        combined = pd.concat(all_perf, ignore_index=True)
        combined_path = out_dir / "all_scenarios_classification.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined classification CSV: {combined_path}")

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved manifest CSV: {manifest_path}")

    if not args.skip_plots:
        plotter = _plotter_module()
        plots_dir = out_dir / "plots"
        for scenario in _selected_scenarios(args.scenarios):
            summary_csv = out_dir / f"{scenario.name}_summary.csv"
            classification_csv = out_dir / f"{scenario.name}_classification.csv"
            for window in plot_windows:
                out_path = plots_dir / f"{scenario.name}_window_{window}.png"
                plotter.plot_scenario_window(
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
