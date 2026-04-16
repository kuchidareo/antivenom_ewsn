from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT_DIR / "logs_120"
SIGNAL_ORDER = [("mem", "Memory"), *[(f"core_{idx}", f"CPU Rank {idx}") for idx in range(4)]]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _analysis_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    return _load_module("scenario_unbinned_ot_mod", ROOT_DIR / "11_unbinned_ot_analysis.py")


def _default_csvs(root: Path) -> tuple[Path, Path, Path]:
    baseclean_csvs = sorted((root / "baseclean").glob("*.csv"))
    baseblurring_csvs = sorted((root / "baseblurring").glob("*.csv"))
    if len(baseclean_csvs) < 2:
        raise ValueError(f"Need at least 2 CSVs in {root / 'baseclean'}")
    if not baseblurring_csvs:
        raise ValueError(f"Need at least 1 CSV in {root / 'baseblurring'}")
    return baseclean_csvs[0], baseclean_csvs[1], baseblurring_csvs[0]


def _spec(group: str, label: str, csv_path: Path) -> str:
    return f"{group}:{label}={csv_path.resolve()}"


def _run_key(analysis, spec: str) -> tuple[str, str, str]:
    group, label, path = analysis.parse_run_spec(spec)
    return group, label, str(path)


def _build_run_measures(df: pd.DataFrame, analysis) -> dict[tuple[str, str, str], dict[str, dict[str, np.ndarray]]]:
    processed = analysis._prepare_df(df)
    per_run_measures: dict[tuple[str, str, str], dict[str, list[dict[str, np.ndarray]]]] = {}

    for key, group in processed.groupby(["source_group", "source_label", "run_csv", "epoch"], dropna=False):
        run_key = key[:3]
        per_run_measures.setdefault(run_key, {"mem": [], **{col: [] for col in analysis.CORE_COLS}})
        per_run_measures[run_key]["mem"].append(analysis._epoch_measure(group, "mem_percent"))
        for col in analysis.CORE_COLS:
            per_run_measures[run_key][col].append(analysis._epoch_measure(group, col))

    run_summary: dict[tuple[str, str, str], dict[str, dict[str, np.ndarray]]] = {}
    for run_key, measures in per_run_measures.items():
        run_summary[run_key] = {}
        for signal_name, epoch_measures in measures.items():
            run_summary[run_key][signal_name] = analysis._aggregate_run_measure(epoch_measures)
    return run_summary


def _plan_image(plan: np.ndarray, analysis, max_side: int) -> np.ndarray:
    plan = np.asarray(plan, dtype=float)
    if plan.size == 0:
        return np.zeros((max_side, max_side), dtype=float)
    out_rows = min(max_side, max(16, plan.shape[0]))
    out_cols = min(max_side, max(16, plan.shape[1]))
    return analysis._resample_matrix(plan, out_rows=out_rows, out_cols=out_cols)


def _compute_shared_scales(
    ref_measures: dict[str, dict[str, np.ndarray]],
    target_measures_list: list[dict[str, dict[str, np.ndarray]]],
    analysis,
    max_side: int,
) -> dict[str, dict[str, float]]:
    shared_scales: dict[str, dict[str, float]] = {}

    for signal_name, _signal_label in SIGNAL_ORDER:
        plan_vmax = 0.0
        cost_vmax = 0.0
        ref_measure = ref_measures[signal_name]

        for target_measures in target_measures_list:
            target_measure = target_measures[signal_name]
            plan, cost, _ot_distance = analysis._transport_plan(ref_measure, target_measure)
            plan_img = _plan_image(plan, analysis=analysis, max_side=max_side)
            cost_img = _plan_image(cost * plan if cost.size else cost, analysis=analysis, max_side=max_side)
            if plan_img.size:
                plan_vmax = max(plan_vmax, float(np.nanmax(plan_img)))
            if cost_img.size:
                cost_vmax = max(cost_vmax, float(np.nanmax(cost_img)))

        shared_scales[signal_name] = {
            "plan_vmax": max(plan_vmax, 1e-12),
            "cost_vmax": max(cost_vmax, 1e-12),
        }

    return shared_scales


def _plot_transport_heatmaps(
    out_path: Path,
    pair_title: str,
    ref_title: str,
    target_title: str,
    ref_measures: dict[str, dict[str, np.ndarray]],
    target_measures: dict[str, dict[str, np.ndarray]],
    analysis,
    max_side: int,
    shared_scales: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    nrows = len(SIGNAL_ORDER)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, 3.2 * nrows), constrained_layout=True)

    if nrows == 1:
        axes = np.asarray([axes])

    stats: dict[str, float] = {}
    plot_items: list[dict[str, object]] = []

    for signal_name, signal_label in SIGNAL_ORDER:
        ref_measure = ref_measures[signal_name]
        target_measure = target_measures[signal_name]
        plan, cost, ot_distance = analysis._transport_plan(ref_measure, target_measure)
        plan_img = _plan_image(plan, analysis=analysis, max_side=max_side)
        cost_img = _plan_image(cost * plan if cost.size else cost, analysis=analysis, max_side=max_side)
        plot_items.append(
            {
                "signal_name": signal_name,
                "signal_label": signal_label,
                "ot_distance": ot_distance,
                "plan_img": plan_img,
                "cost_img": cost_img,
                "ref_points": len(ref_measure["t"]),
                "target_points": len(target_measure["t"]),
            }
        )
        stats[f"{signal_name}_ot_distance"] = float(ot_distance)

    for row_idx, item in enumerate(plot_items):
        ax_plan = axes[row_idx, 0]
        ax_cost = axes[row_idx, 1]
        plan_img = np.asarray(item["plan_img"], dtype=float)
        cost_img = np.asarray(item["cost_img"], dtype=float)
        ot_distance = float(item["ot_distance"])
        cost_total = float(np.nansum(cost_img))
        signal_label = str(item["signal_label"])
        ref_points = int(item["ref_points"])
        target_points = int(item["target_points"])
        signal_name = str(item["signal_name"])

        plan_imshow_kwargs = {"origin": "lower", "aspect": "auto", "cmap": "magma"}
        cost_imshow_kwargs = {"origin": "lower", "aspect": "auto", "cmap": "magma"}
        if shared_scales is not None:
            signal_scales = shared_scales.get(signal_name, {})
            plan_imshow_kwargs.update({"vmin": 0.0, "vmax": float(signal_scales.get("plan_vmax", 1e-12))})
            cost_imshow_kwargs.update({"vmin": 0.0, "vmax": float(signal_scales.get("cost_vmax", 1e-12))})

        im_plan = ax_plan.imshow(plan_img, **plan_imshow_kwargs)
        ax_plan.set_title(f"{signal_label} Transport Plan\nOT={ot_distance:.4f}")
        ax_plan.set_ylabel(f"{signal_label}\nRef points")
        ax_plan.set_xlabel("Target points")
        cbar_plan = fig.colorbar(im_plan, ax=ax_plan, fraction=0.046, pad=0.02)
        cbar_plan.ax.set_ylabel("Transport mass", rotation=90)
        ax_plan.text(
            0.01,
            0.98,
            f"ref={ref_points}, target={target_points}",
            transform=ax_plan.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

        im_cost = ax_cost.imshow(cost_img, **cost_imshow_kwargs)
        ax_cost.set_title(f"{signal_label} Cost-Weighted Transport\nsum(P*C)={cost_total:.4f}")
        ax_cost.set_xlabel("Target points")
        ax_cost.set_ylabel("Ref points")
        cbar_cost = fig.colorbar(im_cost, ax=ax_cost, fraction=0.046, pad=0.02)
        cbar_cost.ax.set_ylabel("Contribution", rotation=90)

    fig.suptitle(f"{pair_title}\nReference: {ref_title}    Target: {target_title}", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--out-dir", default=str(ROOT_DIR / "6_transport_heatmaps"))
    p.add_argument("--reference-csv", default="")
    p.add_argument("--template-csv", default="")
    p.add_argument("--target-csv", default="")
    p.add_argument("--base-group", default="baseclean")
    p.add_argument("--target-group", default="baseblurring")
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
        help="reuse the same per-signal color scale across all comparisons in this run",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reference_csv or args.template_csv or args.target_csv:
        if not (args.reference_csv and args.template_csv and args.target_csv):
            raise ValueError("When overriding CSVs, provide --reference-csv, --template-csv, and --target-csv together.")
        reference_csv = Path(args.reference_csv).resolve()
        template_csv = Path(args.template_csv).resolve()
        target_csv = Path(args.target_csv).resolve()
    else:
        reference_csv, template_csv, target_csv = _default_csvs(root)

    analysis = _analysis_module()
    reference_spec = _spec(args.base_group, "reference", reference_csv)
    template_spec = _spec(args.base_group, "template", template_csv)
    target_spec = _spec(args.target_group, "target", target_csv)

    specs = [template_spec]
    if args.mode in {"target", "both"}:
        specs.append(target_spec)
    df = analysis.load_logs(reference_spec, specs, device_id="scenario_device")
    run_measures = _build_run_measures(df=df, analysis=analysis)

    ref_measures = run_measures[_run_key(analysis, reference_spec)]
    template_measures = run_measures[_run_key(analysis, template_spec)]
    shared_scales = None
    if args.same_scenario_scale:
        scale_targets = [template_measures]
        if args.mode in {"target", "both"}:
            scale_targets.append(run_measures[_run_key(analysis, target_spec)])
        shared_scales = _compute_shared_scales(
            ref_measures=ref_measures,
            target_measures_list=scale_targets,
            analysis=analysis,
            max_side=int(args.max_side),
        )

    outputs: list[dict[str, object]] = []
    if args.mode in {"template", "both"}:
        out_path = out_dir / "baseclean_reference_vs_template_transport_heatmap.png"
        stats = _plot_transport_heatmaps(
            out_path=out_path,
            pair_title="Transport Plan Heatmap: Baseclean Reference vs Template",
            ref_title=reference_csv.stem,
            target_title=template_csv.stem,
            ref_measures=ref_measures,
            target_measures=template_measures,
            analysis=analysis,
            max_side=int(args.max_side),
            shared_scales=shared_scales,
        )
        outputs.append(
            {
                "kind": "template",
                "target_csv": str(template_csv),
                "plot_png": str(out_path.resolve()),
                "stats": stats,
            }
        )

    if args.mode in {"target", "both"}:
        target_measures = run_measures[_run_key(analysis, target_spec)]
        out_path = out_dir / "baseclean_reference_vs_target_transport_heatmap.png"
        stats = _plot_transport_heatmaps(
            out_path=out_path,
            pair_title="Transport Plan Heatmap: Baseclean Reference vs Target",
            ref_title=reference_csv.stem,
            target_title=target_csv.stem,
            ref_measures=ref_measures,
            target_measures=target_measures,
            analysis=analysis,
            max_side=int(args.max_side),
            shared_scales=shared_scales,
        )
        outputs.append(
            {
                "kind": "target",
                "target_csv": str(target_csv),
                "plot_png": str(out_path.resolve()),
                "stats": stats,
            }
        )

    manifest = {
        "root": str(root),
        "reference_csv": str(reference_csv),
        "template_csv": str(template_csv),
        "target_csv": str(target_csv),
        "mode": args.mode,
        "max_side": int(args.max_side),
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
