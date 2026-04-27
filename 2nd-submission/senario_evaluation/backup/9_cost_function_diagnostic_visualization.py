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
    return _load_module("scenario_unbinned_ot_cost_diag_mod", ROOT_DIR / "011_unbinned_ot_analysis.py")


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


def _value_diff_matrix(
    ref_measure: dict[str, np.ndarray],
    target_measure: dict[str, np.ndarray],
    squared: bool,
) -> np.ndarray:
    dx = ref_measure["x"][:, None] - target_measure["x"][None, :]
    if squared:
        return dx * dx
    return np.abs(dx)


def _time_diff_matrix(ref_measure: dict[str, np.ndarray], target_measure: dict[str, np.ndarray]) -> np.ndarray:
    return np.abs(ref_measure["t"][:, None] - target_measure["t"][None, :])


def _adaptive_reg_from_cost(cost: np.ndarray, reg_scale: float) -> float:
    positive_cost = np.asarray(cost, dtype=float)
    positive_cost = positive_cost[np.isfinite(positive_cost) & (positive_cost > 0.0)]
    if positive_cost.size == 0:
        return 1e-3
    return max(float(np.median(positive_cost)) * float(reg_scale), 1e-3)


def _binned_mean_curve(
    time_diff: np.ndarray,
    value_diff: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = np.asarray(time_diff, dtype=float).ravel()
    val = np.asarray(value_diff, dtype=float).ravel()
    mask = np.isfinite(tau) & np.isfinite(val)
    tau = tau[mask]
    val = val[mask]
    edges = np.linspace(0.0, 1.0, num=bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(bins, np.nan, dtype=float)
    counts = np.zeros(bins, dtype=int)
    if tau.size == 0:
        return centers, means, counts

    bin_ids = np.digitize(np.clip(tau, 0.0, 1.0), edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)
    for idx in range(bins):
        sel = bin_ids == idx
        counts[idx] = int(np.sum(sel))
        if counts[idx] > 0:
            means[idx] = float(np.mean(val[sel]))
    return centers, means, counts


def _scatter_sample(
    time_diff: np.ndarray,
    value_diff: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(time_diff, dtype=float).ravel()
    val = np.asarray(value_diff, dtype=float).ravel()
    mask = np.isfinite(tau) & np.isfinite(val)
    tau = tau[mask]
    val = val[mask]
    if tau.size <= max_points:
        return tau, val
    idx = np.linspace(0, tau.size - 1, num=max_points, dtype=int)
    return tau[idx], val[idx]


def _best_candidate_sharpness(
    value_diff: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(value_diff, dtype=float)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    order = np.sort(mat, axis=1)
    if order.shape[1] == 1:
        d1 = order[:, 0]
        delta = np.full(order.shape[0], np.nan, dtype=float)
        ratio = np.full(order.shape[0], np.nan, dtype=float)
        ratio[np.isfinite(d1)] = 1.0
        return delta, ratio

    d1 = order[:, 0]
    d2 = order[:, 1]
    delta = d2 - d1
    ratio = d2 / (d1 + float(eps))
    return delta, ratio


def _ratio_xmax_for_pair(
    ref_measures: dict[str, dict[str, np.ndarray]],
    target_measures: dict[str, dict[str, np.ndarray]],
    value_mode: str,
    eps: float,
) -> float:
    squared = value_mode == "squared"
    max_ratio = 0.0
    for signal_name, _signal_label in SIGNAL_ORDER:
        value_diff = _value_diff_matrix(ref_measures[signal_name], target_measures[signal_name], squared=squared)
        _delta, ratio = _best_candidate_sharpness(value_diff, eps=eps)
        valid_ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
        if valid_ratio.size:
            max_ratio = max(max_ratio, float(np.nanmax(valid_ratio)))
    return max_ratio


def _plot_diagnostics(
    out_path: Path,
    pair_title: str,
    ref_title: str,
    target_title: str,
    ref_measures: dict[str, dict[str, np.ndarray]],
    target_measures: dict[str, dict[str, np.ndarray]],
    analysis,
    value_mode: str,
    relation_bins: int,
    scatter_max_points: int,
    reg_scale: float,
    eps: float,
    ratio_xmax: float | None = None,
    ratio_bin_edges: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    squared = value_mode == "squared"
    fig, axes = plt.subplots(nrows=len(SIGNAL_ORDER), ncols=5, figsize=(25, 4.0 * len(SIGNAL_ORDER)), constrained_layout=True)

    if len(SIGNAL_ORDER) == 1:
        axes = np.asarray([axes])

    summary: dict[str, dict[str, float]] = {}

    for row_idx, (signal_name, signal_label) in enumerate(SIGNAL_ORDER):
        ref_measure = ref_measures[signal_name]
        target_measure = target_measures[signal_name]

        time_diff = _time_diff_matrix(ref_measure, target_measure)
        value_diff = _value_diff_matrix(ref_measure, target_measure, squared=squared)
        current_cost = analysis._cost_matrix(ref_measure, target_measure)
        adaptive_reg = _adaptive_reg_from_cost(current_cost, reg_scale=reg_scale)
        delta, ratio = _best_candidate_sharpness(value_diff, eps=eps)
        tau_scatter, val_scatter = _scatter_sample(time_diff, value_diff, max_points=scatter_max_points)
        tau_centers, val_means, _bin_counts = _binned_mean_curve(time_diff, value_diff, bins=relation_bins)

        valid_delta = delta[np.isfinite(delta)]
        valid_ratio = ratio[np.isfinite(ratio)]
        valid_ratio_nonzero = valid_ratio[valid_ratio > 0.0]
        valid_curve = val_means[np.isfinite(val_means)]
        slope = float("nan")
        if np.count_nonzero(np.isfinite(val_means)) >= 2:
            fit = np.polyfit(tau_centers[np.isfinite(val_means)], val_means[np.isfinite(val_means)], deg=1)
            slope = float(fit[0])

        summary[signal_name] = {
            "ref_points": float(len(ref_measure["t"])),
            "target_points": float(len(target_measure["t"])),
            "adaptive_reg": float(adaptive_reg),
            "value_diff_min": float(np.nanmin(value_diff)) if value_diff.size else float("nan"),
            "value_diff_max": float(np.nanmax(value_diff)) if value_diff.size else float("nan"),
            "value_diff_mean": float(np.nanmean(value_diff)) if value_diff.size else float("nan"),
            "time_value_slope": slope,
            "curve_mean": float(np.nanmean(valid_curve)) if valid_curve.size else float("nan"),
            "delta_median": float(np.nanmedian(valid_delta)) if valid_delta.size else float("nan"),
            "delta_mean": float(np.nanmean(valid_delta)) if valid_delta.size else float("nan"),
            "ratio_median": float(np.nanmedian(valid_ratio)) if valid_ratio.size else float("nan"),
            "ratio_mean": float(np.nanmean(valid_ratio)) if valid_ratio.size else float("nan"),
        }

        ax_heat = axes[row_idx, 0]
        ax_scatter = axes[row_idx, 1]
        ax_curve = axes[row_idx, 2]
        ax_delta = axes[row_idx, 3]
        ax_ratio = axes[row_idx, 4]

        im = ax_heat.imshow(value_diff, origin="lower", aspect="auto", cmap="magma")
        heat_title = "Value Gap"
        if squared:
            heat_title += " Squared"
        ax_heat.set_title(f"{signal_label}\n{heat_title} Matrix")
        ax_heat.set_xlabel("Target points")
        ax_heat.set_ylabel("Ref points")
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
        cbar.ax.set_ylabel("Value difference", rotation=90)

        ax_scatter.scatter(tau_scatter, val_scatter, s=6, alpha=0.15, color="#355070", edgecolors="none")
        ax_scatter.set_title(f"{signal_label}\nTime Gap vs Value Gap")
        ax_scatter.set_xlabel("Relative time gap |t-s|")
        ax_scatter.set_ylabel("Value difference")
        ax_scatter.set_xlim(0.0, 1.0)
        ax_scatter.grid(alpha=0.2)

        ax_curve.plot(tau_centers, val_means, color="#bc4749", linewidth=2.0)
        ax_curve.scatter(tau_centers, val_means, s=24, color="#bc4749")
        ax_curve.set_title(f"{signal_label}\nBinned g(tau)")
        ax_curve.set_xlabel("Relative time gap tau")
        ax_curve.set_ylabel("Mean value difference")
        ax_curve.set_xlim(0.0, 1.0)
        ax_curve.grid(alpha=0.25)
        ax_curve.text(
            0.02,
            0.98,
            f"slope={slope:.4f}\nreg={adaptive_reg:.4f}\nbins={relation_bins}",
            transform=ax_curve.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )

        ax_delta.plot(np.arange(len(delta)), delta, color="#6a994e", linewidth=1.5)
        ax_delta.set_title(f"{signal_label}\nSharpness Delta_i")
        ax_delta.set_xlabel("Ref point index")
        ax_delta.set_ylabel("d_(2) - d_(1)")
        ax_delta.grid(alpha=0.25)
        ax_delta.text(
            0.02,
            0.98,
            f"median={summary[signal_name]['delta_median']:.4f}",
            transform=ax_delta.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )

        hist_bins: int | np.ndarray = 30
        if ratio_bin_edges is not None and len(ratio_bin_edges) >= 2:
            hist_bins = ratio_bin_edges
        ax_ratio.hist(valid_ratio_nonzero, bins=hist_bins, color="#f4a261", alpha=0.85)
        ax_ratio.set_title(f"{signal_label}\nRatio R_i")
        ax_ratio.set_xlabel("d_(2) / (d_(1) + eps)")
        ax_ratio.set_ylabel("Count")
        if ratio_xmax is not None and np.isfinite(ratio_xmax) and ratio_xmax > 0.0:
            ax_ratio.set_xlim(0.0, float(ratio_xmax))
        ax_ratio.grid(alpha=0.25)
        ax_ratio.text(
            0.02,
            0.98,
            f"median={summary[signal_name]['ratio_median']:.4f}\nshown: R_i>0\neps={eps:g}",
            transform=ax_ratio.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )

    fig.suptitle(
        f"{pair_title}\nReference: {ref_title}    Target: {target_title}\n"
        f"Relative time from 011_ epoch measure, value_mode={value_mode}, reg_scale={reg_scale}",
        fontsize=15,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--out-dir", default=str(ROOT_DIR / "9_cost_function_diagnostics"))
    p.add_argument("--reference-csv", default="")
    p.add_argument("--template-csv", default="")
    p.add_argument("--target-csv", default="")
    p.add_argument("--base-group", default="baseclean")
    p.add_argument("--target-group", default="baseblurring")
    p.add_argument(
        "--mode",
        choices=["template", "target", "both"],
        default="both",
        help="which comparison diagnostics to render",
    )
    p.add_argument(
        "--value-mode",
        choices=["abs", "squared"],
        default="abs",
        help="visualize absolute value gaps or squared value gaps",
    )
    p.add_argument("--relation-bins", type=int, default=24)
    p.add_argument("--scatter-max-points", type=int, default=12000)
    p.add_argument("--reg-scale", type=float, default=0.05)
    p.add_argument("--sharpness-eps", type=float, default=1e-9)
    p.add_argument(
        "--ratio-xmax",
        type=float,
        default=0.0,
        help="optional fixed x-axis max for the 5th-panel Ratio R_i histogram; 0 means autoscale",
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

    outputs: list[dict[str, object]] = []
    if args.mode in {"template", "both"}:
        out_path = out_dir / "baseclean_reference_vs_template_cost_diagnostic.png"
        summary = _plot_diagnostics(
            out_path=out_path,
            pair_title="Cost Function Diagnostic: Baseclean Reference vs Template",
            ref_title=reference_csv.stem,
            target_title=template_csv.stem,
            ref_measures=ref_measures,
            target_measures=template_measures,
            analysis=analysis,
            value_mode=args.value_mode,
            relation_bins=int(args.relation_bins),
            scatter_max_points=int(args.scatter_max_points),
            reg_scale=float(args.reg_scale),
            eps=float(args.sharpness_eps),
            ratio_xmax=float(args.ratio_xmax) if float(args.ratio_xmax) > 0.0 else None,
        )
        outputs.append(
            {
                "kind": "template",
                "target_csv": str(template_csv),
                "plot_png": str(out_path.resolve()),
                "summary": summary,
            }
        )

    if args.mode in {"target", "both"}:
        target_measures = run_measures[_run_key(analysis, target_spec)]
        out_path = out_dir / "baseclean_reference_vs_target_cost_diagnostic.png"
        summary = _plot_diagnostics(
            out_path=out_path,
            pair_title="Cost Function Diagnostic: Baseclean Reference vs Target",
            ref_title=reference_csv.stem,
            target_title=target_csv.stem,
            ref_measures=ref_measures,
            target_measures=target_measures,
            analysis=analysis,
            value_mode=args.value_mode,
            relation_bins=int(args.relation_bins),
            scatter_max_points=int(args.scatter_max_points),
            reg_scale=float(args.reg_scale),
            eps=float(args.sharpness_eps),
            ratio_xmax=float(args.ratio_xmax) if float(args.ratio_xmax) > 0.0 else None,
        )
        outputs.append(
            {
                "kind": "target",
                "target_csv": str(target_csv),
                "plot_png": str(out_path.resolve()),
                "summary": summary,
            }
        )

    manifest = {
        "root": str(root),
        "reference_csv": str(reference_csv),
        "template_csv": str(template_csv),
        "target_csv": str(target_csv),
        "mode": args.mode,
        "value_mode": args.value_mode,
        "relation_bins": int(args.relation_bins),
        "scatter_max_points": int(args.scatter_max_points),
        "reg_scale": float(args.reg_scale),
        "sharpness_eps": float(args.sharpness_eps),
        "ratio_xmax": float(args.ratio_xmax),
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
