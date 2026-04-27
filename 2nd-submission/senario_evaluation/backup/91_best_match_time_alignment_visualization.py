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
    return _load_module("scenario_best_match_alignment_mod", ROOT_DIR / "011_unbinned_ot_analysis.py")


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


def _best_match_series(
    ref_measure: dict[str, np.ndarray],
    target_measure: dict[str, np.ndarray],
    squared: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value_diff = _value_diff_matrix(ref_measure, target_measure, squared=squared)
    if value_diff.ndim != 2 or value_diff.shape[0] == 0 or value_diff.shape[1] == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float), value_diff

    best_idx = np.argmin(value_diff, axis=1).astype(int)
    best_time_gap = np.abs(ref_measure["t"] - target_measure["t"][best_idx])
    return best_idx, best_time_gap, value_diff


def _monotonic_violation_rate(best_idx: np.ndarray) -> float:
    if len(best_idx) <= 1:
        return float("nan")
    diffs = np.diff(best_idx)
    return float(np.mean(diffs < 0))


def _plot_best_match_alignment(
    out_path: Path,
    pair_title: str,
    ref_title: str,
    target_title: str,
    ref_measures: dict[str, dict[str, np.ndarray]],
    target_measures: dict[str, dict[str, np.ndarray]],
    value_mode: str,
) -> dict[str, dict[str, float]]:
    squared = value_mode == "squared"
    fig, axes = plt.subplots(nrows=len(SIGNAL_ORDER), ncols=2, figsize=(15, 3.6 * len(SIGNAL_ORDER)), constrained_layout=True)

    if len(SIGNAL_ORDER) == 1:
        axes = np.asarray([axes])

    summary: dict[str, dict[str, float]] = {}

    for row_idx, (signal_name, signal_label) in enumerate(SIGNAL_ORDER):
        ref_measure = ref_measures[signal_name]
        target_measure = target_measures[signal_name]
        best_idx, best_time_gap, value_diff = _best_match_series(ref_measure, target_measure, squared=squared)

        ax_match = axes[row_idx, 0]
        ax_gap = axes[row_idx, 1]
        i_axis = np.arange(len(best_idx))

        if len(best_idx):
            ax_match.plot(i_axis, best_idx, color="#355070", linewidth=1.4)
            ax_match.scatter(i_axis, best_idx, c=best_time_gap, s=12, cmap="magma", alpha=0.85)
        ax_match.set_title(f"{signal_label}\nBest Match Index j*(i)")
        ax_match.set_xlabel("Ref point index i")
        ax_match.set_ylabel("Target index j*")
        ax_match.grid(alpha=0.25)

        if len(best_time_gap):
            ax_gap.plot(i_axis, best_time_gap, color="#bc4749", linewidth=1.6)
            ax_gap.scatter(i_axis, best_time_gap, s=12, color="#bc4749", alpha=0.7)
        ax_gap.set_title(f"{signal_label}\nRelative Time Gap delta_t(i)")
        ax_gap.set_xlabel("Ref point index i")
        ax_gap.set_ylabel("|t_i - s_j*|")
        ax_gap.set_ylim(bottom=0.0, top=max(1.0, float(np.nanmax(best_time_gap)) if len(best_time_gap) else 1.0))
        ax_gap.grid(alpha=0.25)

        local_jump = np.abs(np.diff(best_idx)) if len(best_idx) > 1 else np.zeros(0, dtype=float)
        mean_gap = float(np.nanmean(best_time_gap)) if len(best_time_gap) else float("nan")
        median_gap = float(np.nanmedian(best_time_gap)) if len(best_time_gap) else float("nan")
        mean_jump = float(np.nanmean(local_jump)) if len(local_jump) else float("nan")
        violation_rate = _monotonic_violation_rate(best_idx)

        summary[signal_name] = {
            "ref_points": float(len(ref_measure["t"])),
            "target_points": float(len(target_measure["t"])),
            "mean_time_gap": mean_gap,
            "median_time_gap": median_gap,
            "max_time_gap": float(np.nanmax(best_time_gap)) if len(best_time_gap) else float("nan"),
            "mean_index_jump": mean_jump,
            "median_index_jump": float(np.nanmedian(local_jump)) if len(local_jump) else float("nan"),
            "monotonic_violation_rate": violation_rate,
            "value_diff_mean": float(np.nanmean(value_diff[np.arange(len(best_idx)), best_idx])) if len(best_idx) else float("nan"),
        }

        ax_match.text(
            0.02,
            0.98,
            f"mean jump={mean_jump:.3f}\nviolation={violation_rate:.3f}",
            transform=ax_match.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        ax_gap.text(
            0.02,
            0.98,
            f"mean={mean_gap:.4f}\nmedian={median_gap:.4f}",
            transform=ax_gap.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )

    fig.suptitle(
        f"{pair_title}\nReference: {ref_title}    Target: {target_title}\n"
        f"Best match based on value_mode={value_mode}, time gap uses relative time from 011_",
        fontsize=15,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--out-dir", default=str(ROOT_DIR / "91_best_match_alignment"))
    p.add_argument("--reference-csv", default="")
    p.add_argument("--template-csv", default="")
    p.add_argument("--target-csv", default="")
    p.add_argument("--base-group", default="baseclean")
    p.add_argument("--target-group", default="baseblurring")
    p.add_argument(
        "--mode",
        choices=["template", "target", "both"],
        default="both",
        help="which comparison plots to render",
    )
    p.add_argument(
        "--value-mode",
        choices=["abs", "squared"],
        default="abs",
        help="best match is chosen from absolute or squared value difference",
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
        out_path = out_dir / "baseclean_reference_vs_template_best_match_alignment.png"
        summary = _plot_best_match_alignment(
            out_path=out_path,
            pair_title="Best Match Alignment: Baseclean Reference vs Template",
            ref_title=reference_csv.stem,
            target_title=template_csv.stem,
            ref_measures=ref_measures,
            target_measures=template_measures,
            value_mode=args.value_mode,
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
        out_path = out_dir / "baseclean_reference_vs_target_best_match_alignment.png"
        summary = _plot_best_match_alignment(
            out_path=out_path,
            pair_title="Best Match Alignment: Baseclean Reference vs Target",
            ref_title=reference_csv.stem,
            target_title=target_csv.stem,
            ref_measures=ref_measures,
            target_measures=target_measures,
            value_mode=args.value_mode,
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
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
