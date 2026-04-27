from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT_DIR / "logs_120"
EXCLUDED_RUNS = {
    ("adamw_weight_decay/clean", "20260411_171943"),
}


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _heatmap_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    return _load_module("scenario_transport_heatmap_mod", ROOT_DIR / "6_transport_plan_heatmap.py")


def _discover_cases(root: Path) -> dict[str, list[Path]]:
    cases: dict[str, list[Path]] = {}

    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if scenario_dir.name == "__pycache__":
            continue

        direct_csvs = sorted(scenario_dir.glob("*.csv"))
        if direct_csvs:
            cases[scenario_dir.name] = direct_csvs

        for case_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            if case_dir.name == "__pycache__":
                continue
            csvs = sorted(case_dir.glob("*.csv"))
            if csvs:
                cases[f"{scenario_dir.name}/{case_dir.name}"] = csvs

    return cases


def _is_excluded_run(case_name: str, csv_path: Path) -> bool:
    return (case_name, csv_path.stem) in EXCLUDED_RUNS


def _eligible_cases(cases: dict[str, list[Path]], min_runs: int) -> list[str]:
    eligible: list[str] = []
    for case_name in sorted(cases):
        kept = [csv_path for csv_path in cases[case_name] if not _is_excluded_run(case_name, csv_path)]
        if len(kept) >= min_runs:
            eligible.append(case_name)
    return eligible


def _slugify(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def _make_spec(case_name: str, label: str, csv_path: Path) -> str:
    return f"{case_name}:{label}={csv_path.resolve()}"


def _run_key(analysis, spec: str) -> tuple[str, str, str]:
    group, label, path = analysis.parse_run_spec(spec)
    return group, label, str(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--out-dir", default=str(ROOT_DIR / "7_iterative_transport_heatmaps"))
    p.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="minimum CSV count required for a case to be used as a base",
    )
    p.add_argument(
        "--max-targets-per-case",
        type=int,
        default=0,
        help="optional cap per target case; 0 means include all runs",
    )
    p.add_argument(
        "--mode",
        choices=["template", "target", "both"],
        default="both",
        help="which comparison heatmaps to render",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=96,
        help="maximum image side after resampling the transport plan for display",
    )
    p.add_argument(
        "--same-scenario-scale",
        action="store_true",
        help="reuse the same per-signal color scale across all plots within each base scenario",
    )
    p.add_argument(
        "--z-normalize-window",
        action="store_true",
        help="z-normalize each local window before shape comparison in the OT cost",
    )
    p.add_argument(
        "--cost-function",
        choices=["time_distance", "time_distance_shape", "time_distance_delta"],
        default="time_distance_shape",
        help="select the OT cost used to compute the transport plan",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap = _heatmap_module()
    analysis = heatmap._analysis_module()

    cases = _discover_cases(root)
    base_cases = _eligible_cases(cases, min_runs=max(2, int(args.min_runs)))
    if not base_cases:
        raise ValueError(f"No eligible cases found in {root}")

    manifest_rows: list[dict[str, Any]] = []

    for base_case in base_cases:
        base_csvs = [csv_path for csv_path in cases[base_case] if not _is_excluded_run(base_case, csv_path)]
        reference_csv = base_csvs[0]
        template_csv = base_csvs[1]

        target_specs: list[str] = []
        for case_name in base_cases:
            case_csvs = [csv_path for csv_path in cases[case_name] if not _is_excluded_run(case_name, csv_path)]
            start_idx = 2 if case_name == base_case else 0
            selected_csvs = case_csvs[start_idx:]
            if int(args.max_targets_per_case) > 0:
                selected_csvs = selected_csvs[: int(args.max_targets_per_case)]
            for csv_path in selected_csvs:
                target_specs.append(_make_spec(case_name, f"target_{csv_path.stem}", csv_path))

        specs = [ _make_spec(base_case, "template", template_csv) ]
        if args.mode in {"target", "both"}:
            specs.extend(target_specs)

        reference_spec = _make_spec(base_case, "reference", reference_csv)
        template_spec = _make_spec(base_case, "template", template_csv)
        df = analysis.load_logs(reference_spec, specs, device_id="scenario_device")
        run_measures = heatmap._build_run_measures(df=df, analysis=analysis)

        ref_measures = run_measures[_run_key(analysis, reference_spec)]
        template_measures = run_measures[_run_key(analysis, template_spec)]
        shared_scales = None
        if args.same_scenario_scale:
            scale_targets = [template_measures]
            if args.mode in {"target", "both"}:
                for target_spec in target_specs:
                    target_group, _, target_path = analysis.parse_run_spec(target_spec)
                    target_csv = Path(target_path)
                    scale_targets.append(run_measures[(target_group, f"target_{target_csv.stem}", str(target_csv))])
            shared_scales = heatmap._compute_shared_scales(
                ref_measures=ref_measures,
                target_measures_list=scale_targets,
                analysis=analysis,
                max_side=int(args.max_side),
                cost_function=str(args.cost_function),
                z_normalize_window=bool(args.z_normalize_window),
            )

        base_out_dir = out_dir / _slugify(base_case)
        base_out_dir.mkdir(parents=True, exist_ok=True)

        if args.mode in {"template", "both"}:
            template_png = base_out_dir / "00_reference_vs_template_transport_heatmap.png"
            stats = heatmap._plot_transport_heatmaps(
                out_path=template_png,
                pair_title=f"Transport Plan Heatmap: {base_case} Reference vs Template",
                ref_title=reference_csv.stem,
                target_title=template_csv.stem,
                ref_measures=ref_measures,
                target_measures=template_measures,
                analysis=analysis,
                max_side=int(args.max_side),
                shared_scales=shared_scales,
                cost_function=str(args.cost_function),
                z_normalize_window=bool(args.z_normalize_window),
            )
            manifest_rows.append(
                {
                    "base_case": base_case,
                    "reference_csv": str(reference_csv.resolve()),
                    "template_csv": str(template_csv.resolve()),
                    "target_case": base_case,
                    "target_csv": str(template_csv.resolve()),
                    "plot_png": str(template_png.resolve()),
                    "kind": "template",
                    "stats": stats,
                }
            )

        if args.mode in {"target", "both"}:
            for idx, target_spec in enumerate(target_specs, start=1):
                target_group, _, target_path = analysis.parse_run_spec(target_spec)
                target_csv = Path(target_path)
                target_measures = run_measures[(target_group, f"target_{target_csv.stem}", str(target_csv))]
                png_path = base_out_dir / f"{idx:02d}_{_slugify(target_group)}_{target_csv.stem}_transport_heatmap.png"
                stats = heatmap._plot_transport_heatmaps(
                    out_path=png_path,
                    pair_title=f"Transport Plan Heatmap: {base_case} Reference vs {target_group}",
                    ref_title=reference_csv.stem,
                    target_title=target_csv.stem,
                    ref_measures=ref_measures,
                    target_measures=target_measures,
                    analysis=analysis,
                    max_side=int(args.max_side),
                    shared_scales=shared_scales,
                    cost_function=str(args.cost_function),
                    z_normalize_window=bool(args.z_normalize_window),
                )
                manifest_rows.append(
                    {
                        "base_case": base_case,
                        "reference_csv": str(reference_csv.resolve()),
                        "template_csv": str(template_csv.resolve()),
                        "target_case": target_group,
                        "target_csv": str(target_csv.resolve()),
                        "plot_png": str(png_path.resolve()),
                        "kind": "target",
                        "stats": stats,
                    }
                )

    manifest_path = out_dir / "iterative_transport_heatmap_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "num_plots": len(manifest_rows)}, indent=2))


if __name__ == "__main__":
    main()
