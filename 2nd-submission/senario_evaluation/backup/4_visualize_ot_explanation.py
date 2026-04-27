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
CORE_COLS = ["core_0", "core_1", "core_2", "core_3"]
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
    return _load_module("scenario_analysis_mod", ROOT_DIR / "1_cpu_core_sorted_mem_baseclean_analysis.py")


def _default_csvs(root: Path) -> tuple[Path, Path, Path]:
    baseclean_csvs = sorted((root / "baseclean").glob("*.csv"))
    baseblurring_csvs = sorted((root / "baseblurring").glob("*.csv"))
    if len(baseclean_csvs) < 2:
        raise ValueError("Need at least 2 CSVs in senario_evaluation/baseclean.")
    if not baseblurring_csvs:
        raise ValueError("Need at least 1 CSV in senario_evaluation/baseblurring.")
    return baseclean_csvs[0], baseclean_csvs[1], baseblurring_csvs[0]


def _preprocess_logs(df: pd.DataFrame, analysis, bins: int) -> pd.DataFrame:
    required = ["epoch", "ts_unix", "cpu_per_core", "mem_percent"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()
    df["epoch"] = epoch_numeric.loc[df.index]
    df = df.loc[df["epoch"] != 9].copy()
    df["mem_percent"] = pd.to_numeric(df["mem_percent"], errors="coerce")
    df = df.loc[df["mem_percent"].notna()].copy()

    core_vals = df["cpu_per_core"].apply(analysis._parse_core_list)
    df = df.loc[core_vals.notna()].copy()
    sorted_vals = core_vals.apply(lambda v: sorted(v, reverse=True))
    for idx, col in enumerate(CORE_COLS):
        df[col] = sorted_vals.apply(lambda v: float(v[idx]) if idx < len(v) else float("nan"))
    df = df.loc[df[CORE_COLS].notna().all(axis=1)].copy()
    df["bins"] = int(bins)
    return df


def _build_run_shapes(df: pd.DataFrame, analysis, bins: int) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    processed = _preprocess_logs(df, analysis=analysis, bins=bins)
    per_run_shapes: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = {}

    for key, group in processed.groupby(["source_group", "source_label", "run_csv", "epoch"], dropna=False):
        run_key = key[:3]
        per_run_shapes.setdefault(run_key, {"mem": [], **{col: [] for col in CORE_COLS}})
        per_run_shapes[run_key]["mem"].append(analysis._epoch_shape_mean(group, "mem_percent", bins=bins))
        for col in CORE_COLS:
            per_run_shapes[run_key][col].append(analysis._epoch_shape_mean(group, col, bins=bins))

    run_summary: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for run_key, shapes in per_run_shapes.items():
        run_summary[run_key] = {}
        for signal_name, vectors in shapes.items():
            arr = np.vstack(vectors) if vectors else np.zeros((0, bins), dtype=float)
            run_summary[run_key][signal_name] = (
                arr.mean(axis=0).astype(float) if len(arr) else np.zeros(bins, dtype=float)
            )
    return run_summary


def _cdf_gap(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref_norm = np.asarray(ref, dtype=float)
    target_norm = np.asarray(target, dtype=float)
    if ref_norm.sum() > 0:
        ref_norm = ref_norm / ref_norm.sum()
    if target_norm.sum() > 0:
        target_norm = target_norm / target_norm.sum()
    return np.abs(np.cumsum(ref_norm) - np.cumsum(target_norm))


def _contrib_matrix(coupling: np.ndarray) -> np.ndarray:
    idx = np.arange(coupling.shape[0], dtype=float)
    return coupling * np.abs(idx[:, None] - idx[None, :])


def _plot_pair_figure(
    out_path: Path,
    pair_title: str,
    ref_title: str,
    target_title: str,
    ref_shapes: dict[str, np.ndarray],
    target_shapes: dict[str, np.ndarray],
    analysis,
) -> None:
    nrows = len(SIGNAL_ORDER)
    plot_items: list[dict[str, np.ndarray | float | str]] = []
    max_cdf_gap = 0.0
    max_coupling = 0.0
    max_contrib = 0.0
    max_ot = 0.0

    for signal_name, signal_label in SIGNAL_ORDER:
        ref_vec = np.asarray(ref_shapes[signal_name], dtype=float)
        target_vec = np.asarray(target_shapes[signal_name], dtype=float)
        cdf_gap = _cdf_gap(ref_vec, target_vec)
        coupling = analysis._ot_coupling_1d(ref_vec, target_vec)
        contrib = _contrib_matrix(coupling)
        scalar_ot = float(analysis._wasserstein_1d(ref_vec, target_vec))
        plot_items.append(
            {
                "signal_name": signal_name,
                "signal_label": signal_label,
                "cdf_gap": cdf_gap,
                "coupling": coupling,
                "contrib": contrib,
                "scalar_ot": scalar_ot,
            }
        )
        if len(cdf_gap):
            max_cdf_gap = max(max_cdf_gap, float(np.nanmax(cdf_gap)))
        if coupling.size:
            max_coupling = max(max_coupling, float(np.nanmax(coupling)))
        if contrib.size:
            max_contrib = max(max_contrib, float(np.nanmax(contrib)))
        if np.isfinite(scalar_ot):
            max_ot = max(max_ot, scalar_ot)

    cdf_ylim = max(max_cdf_gap * 1.05, 1e-6)
    coupling_vmax = max(max_coupling, 1e-12)
    contrib_vmax = max(max_contrib, 1e-12)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=3,
        figsize=(24, 3.8 * nrows),
        constrained_layout=True,
    )

    for row_idx, item in enumerate(plot_items):
        signal_label = str(item["signal_label"])
        cdf_gap = np.asarray(item["cdf_gap"], dtype=float)
        coupling = np.asarray(item["coupling"], dtype=float)
        contrib = np.asarray(item["contrib"], dtype=float)
        scalar_ot = float(item["scalar_ot"])

        ax_gap = axes[row_idx, 0]
        ax_gap.plot(np.arange(len(cdf_gap)), cdf_gap, color="#1f77b4", linewidth=2.0)
        ax_gap.fill_between(np.arange(len(cdf_gap)), cdf_gap, color="#1f77b4", alpha=0.25)
        ax_gap.set_xlim(0, len(cdf_gap) - 1)
        ax_gap.set_ylim(0.0, cdf_ylim)
        ax_gap.set_ylabel(signal_label)
        ax_gap.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax_gap.set_title(
            f"CDF Gap |CDF(ref)-CDF(target)|\nOT={scalar_ot:.4f} / max={max_ot:.4f}"
        )

        ax_coupling = axes[row_idx, 1]
        im1 = ax_coupling.imshow(
            coupling,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=coupling_vmax,
        )
        diag = np.arange(coupling.shape[0])
        ax_coupling.plot(diag, diag, color="white", linestyle="--", linewidth=0.8, alpha=0.9)
        ax_coupling.set_title("OT Coupling")
        ax_coupling.set_ylabel("Ref Bin")
        cbar1 = fig.colorbar(im1, ax=ax_coupling, fraction=0.046, pad=0.02)
        cbar1.ax.set_ylabel("Mass", rotation=90)

        ax_contrib = axes[row_idx, 2]
        im2 = ax_contrib.imshow(
            contrib,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=contrib_vmax,
        )
        ax_contrib.plot(diag, diag, color="white", linestyle="--", linewidth=0.8, alpha=0.9)
        ax_contrib.set_title("Transport Cost Contribution\nP[i,j] * |i-j|")
        cbar2 = fig.colorbar(im2, ax=ax_contrib, fraction=0.046, pad=0.02)
        cbar2.ax.set_ylabel("Contribution", rotation=90)

        if row_idx == nrows - 1:
            ax_gap.set_xlabel("Normalized Time Bin")
            ax_coupling.set_xlabel("Target Bin")
            ax_contrib.set_xlabel("Target Bin")

    fig.suptitle(f"{pair_title}\nReference: {ref_title}    Target: {target_title}", fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_DIR))
    p.add_argument("--out-dir", default=str(ROOT_DIR / "4_visualization"))
    p.add_argument("--bins", type=int, default=50)
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_csv, template_csv, target_csv = _default_csvs(root)
    analysis = _analysis_module()

    reference_spec = f"baseclean:reference={reference_csv.resolve()}"
    template_spec = f"baseclean:template={template_csv.resolve()}"
    target_spec = f"baseblurring:run={target_csv.resolve()}"

    df = analysis.load_logs(reference_spec, [template_spec, target_spec], device_id="scenario_device")
    run_shapes = _build_run_shapes(df=df, analysis=analysis, bins=int(args.bins))

    ref_group, ref_label, ref_path = analysis.parse_run_spec(reference_spec)
    template_group, template_label, template_path = analysis.parse_run_spec(template_spec)
    target_group, target_label, target_path = analysis.parse_run_spec(target_spec)
    ref_key = (ref_group, ref_label, str(ref_path))
    template_key = (template_group, template_label, str(template_path))
    target_key = (target_group, target_label, str(target_path))

    ref_shapes = run_shapes[ref_key]
    template_shapes = run_shapes[template_key]
    target_shapes = run_shapes[target_key]

    _plot_pair_figure(
        out_path=out_dir / "baseclean_reference_vs_template.png",
        pair_title="OT Explanation: Baseclean Reference vs Baseclean Template",
        ref_title=reference_csv.stem,
        target_title=template_csv.stem,
        ref_shapes=ref_shapes,
        target_shapes=template_shapes,
        analysis=analysis,
    )
    _plot_pair_figure(
        out_path=out_dir / "baseclean_reference_vs_baseblurring.png",
        pair_title="OT Explanation: Baseclean Reference vs Baseblurring Run",
        ref_title=reference_csv.stem,
        target_title=target_csv.stem,
        ref_shapes=ref_shapes,
        target_shapes=target_shapes,
        analysis=analysis,
    )

    manifest = {
        "reference_csv": str(reference_csv.resolve()),
        "template_csv": str(template_csv.resolve()),
        "target_csv": str(target_csv.resolve()),
        "bins": int(args.bins),
        "outputs": [
            str((out_dir / "baseclean_reference_vs_template.png").resolve()),
            str((out_dir / "baseclean_reference_vs_baseblurring.png").resolve()),
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
