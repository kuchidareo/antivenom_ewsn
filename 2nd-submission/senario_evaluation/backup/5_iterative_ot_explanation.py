from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
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


def _viz_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    return _load_module("scenario_ot_viz_mod", ROOT_DIR / "4_visualize_ot_explanation.py")


def _discover_cases(root: Path) -> dict[str, list[Path]]:
    cases: dict[str, list[Path]] = {}

    for name in ("baseclean", "baseblurring"):
        case_dir = root / name
        if case_dir.is_dir():
            csvs = sorted(case_dir.glob("*.csv"))
            if csvs:
                cases[name] = csvs

    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if scenario_dir.name in {"baseclean", "baseblurring", "__pycache__", "4_visualization", "iterative_analysis", "iterative_analysis_test"}:
            continue
        for poison_type in ("clean", "blurring"):
            case_dir = scenario_dir / poison_type
            if case_dir.is_dir():
                csvs = sorted(case_dir.glob("*.csv"))
                if csvs:
                    cases[f"{scenario_dir.name}/{poison_type}"] = csvs

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


def _run_key(viz, spec: str) -> tuple[str, str, str]:
    group, label, path = viz._analysis_module().parse_run_spec(spec)
    return group, label, str(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_DIR))
    p.add_argument("--out-dir", default=str(ROOT_DIR / "5_iterative_ot_explanation"))
    p.add_argument("--bins", type=int, default=50)
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
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    viz = _viz_module()
    analysis = viz._analysis_module()

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
                target_specs.append(_make_spec(case_name, f"run_{csv_path.stem}", csv_path))

        if not target_specs:
            continue

        reference_spec = _make_spec(base_case, "reference", reference_csv)
        template_spec = _make_spec(base_case, "template", template_csv)
        df = analysis.load_logs(reference_spec, [template_spec, *target_specs], device_id="scenario_device")
        run_shapes = viz._build_run_shapes(df=df, analysis=analysis, bins=int(args.bins))

        ref_shapes = run_shapes[_run_key(viz, reference_spec)]
        template_shapes = run_shapes[_run_key(viz, template_spec)]

        base_out_dir = out_dir / _slugify(base_case)
        base_out_dir.mkdir(parents=True, exist_ok=True)

        template_png = base_out_dir / "00_reference_vs_template.png"
        viz._plot_pair_figure(
            out_path=template_png,
            pair_title=f"OT Explanation: {base_case} Reference vs Template",
            ref_title=reference_csv.stem,
            target_title=template_csv.stem,
            ref_shapes=ref_shapes,
            target_shapes=template_shapes,
            analysis=analysis,
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
            }
        )

        for idx, target_spec in enumerate(target_specs, start=1):
            target_group, _, target_path = analysis.parse_run_spec(target_spec)
            target_shapes = run_shapes[(target_group, f"run_{Path(target_path).stem}", str(target_path))]
            png_path = base_out_dir / f"{idx:02d}_{_slugify(target_group)}_{Path(target_path).stem}.png"
            viz._plot_pair_figure(
                out_path=png_path,
                pair_title=f"OT Explanation: {base_case} Reference vs {target_group}",
                ref_title=reference_csv.stem,
                target_title=Path(target_path).stem,
                ref_shapes=ref_shapes,
                target_shapes=target_shapes,
                analysis=analysis,
            )
            manifest_rows.append(
                {
                    "base_case": base_case,
                    "reference_csv": str(reference_csv.resolve()),
                    "template_csv": str(template_csv.resolve()),
                    "target_case": target_group,
                    "target_csv": str(Path(target_path).resolve()),
                    "plot_png": str(png_path.resolve()),
                    "kind": "target",
                }
            )

    manifest_path = out_dir / "iterative_ot_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "num_plots": len(manifest_rows)}, indent=2))


if __name__ == "__main__":
    main()
