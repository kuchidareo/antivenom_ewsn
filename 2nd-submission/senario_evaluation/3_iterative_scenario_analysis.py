from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent


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


def _plot_module():
    return _load_module("scenario_plot_mod", ROOT_DIR / "2_plot_baseclean_box.py")


def _slugify(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def _discover_cases(root: Path) -> dict[str, list[Path]]:
    cases: dict[str, list[Path]] = {}

    for name in ("baseclean", "baseblurring"):
        case_dir = root / name
        if case_dir.is_dir():
            csvs = sorted(case_dir.glob("*.csv"))
            if csvs:
                cases[name] = csvs

    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if scenario_dir.name in {"baseclean", "baseblurring", "__pycache__"}:
            continue
        for poison_type in ("clean", "blurring"):
            case_dir = scenario_dir / poison_type
            if case_dir.is_dir():
                csvs = sorted(case_dir.glob("*.csv"))
                if csvs:
                    cases[f"{scenario_dir.name}/{poison_type}"] = csvs

    return cases


def _make_spec(case_name: str, role: str, index: int, csv_path: Path) -> str:
    label = f"{role}_{index:02d}_{csv_path.stem}"
    return f"{case_name}:{label}={csv_path.resolve()}"


def _ordered_case_names(cases: dict[str, list[Path]], min_runs: int) -> list[str]:
    return [name for name in sorted(cases) if len(cases[name]) >= min_runs]


def _scenario_name(case_name: str) -> str:
    return case_name.split("/", 1)[0]


def _run_display_label(row: pd.Series) -> str:
    run_csv = Path(str(row["target_run_csv"]))
    return f"{row['target_group']}:{run_csv.stem}"


def _case_colors(case_names: list[str]) -> dict[str, Any]:
    scenario_names = []
    for case_name in case_names:
        scenario = _scenario_name(case_name)
        if scenario not in scenario_names:
            scenario_names.append(scenario)
    cmap = plt.cm.get_cmap("tab10", max(len(scenario_names), 1))
    return {scenario: cmap(i) for i, scenario in enumerate(scenario_names)}


def _plot_run_boxes(
    ax,
    df: pd.DataFrame,
    value_col: str,
    y_label: str,
    color_map: dict[str, Any],
) -> None:
    ordered_runs = (
        df[["target_group", "target_run_csv", "display_label"]]
        .drop_duplicates()
        .sort_values(["target_group", "target_run_csv"])
    )

    data = []
    labels = []
    colors = []
    for _, run_row in ordered_runs.iterrows():
        mask = (
            (df["target_group"] == run_row["target_group"])
            & (df["target_run_csv"] == run_row["target_run_csv"])
        )
        vals = df.loc[mask, value_col].astype(float).dropna().to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(str(run_row["display_label"]))
        colors.append(color_map[_scenario_name(str(run_row["target_group"]))])

    if not data:
        ax.set_axis_off()
        return

    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, alpha=0.7, edgecolor="black", linewidth=0.8)
    for median in bp["medians"]:
        median.set(color="black", linewidth=1.0)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)


def _plot_case_summary(
    base_case: str,
    df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
) -> None:
    plot_df = df.copy()
    plot_df["display_label"] = plot_df.apply(_run_display_label, axis=1)
    case_names = plot_df["target_group"].drop_duplicates().tolist()
    color_map = _case_colors(case_names)
    n_boxes = len(plot_df[["target_group", "target_run_csv"]].drop_duplicates())
    fig_w = max(24.0, n_boxes * 0.55)
    has_cosine = "cosine_similarity" in plot_df.columns and plot_df["cosine_similarity"].notna().any()
    nrows = 2 if has_cosine else 1
    fig_h = 10.0 if has_cosine else 5.5
    fig, axes = plt.subplots(nrows, 1, figsize=(fig_w, fig_h), squeeze=False)

    _plot_run_boxes(axes[0][0], plot_df, "ot_distance", "Magnitude score W", color_map)
    if has_cosine:
        _plot_run_boxes(axes[1][0], plot_df, "cosine_similarity", "Similarity score S", color_map)

    legend_handles = [
        Patch(facecolor=color_map[scenario], edgecolor="black", label=scenario)
        for scenario in color_map
    ]
    if legend_handles:
        axes[0][0].legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.002, 1.0),
            borderaxespad=0.0,
            title="Scenario",
        )

    fig.suptitle(f"{title_prefix}: {base_case}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_case_summary(
    base_case: str,
    cases: dict[str, list[Path]],
    bins: int,
) -> pd.DataFrame:
    analysis = _analysis_module()
    eligible_cases = _ordered_case_names(cases, min_runs=2)
    if base_case not in eligible_cases:
        raise ValueError(f"Base case '{base_case}' does not have at least 2 CSVs.")

    base_csvs = cases[base_case]
    reference_spec = _make_spec(base_case, "reference", 0, base_csvs[0])
    template_spec = _make_spec(base_case, "template", 1, base_csvs[1])

    run_specs: list[str] = []
    for case_name in eligible_cases:
        csvs = cases[case_name]
        start_idx = 2 if case_name == base_case else 0
        for idx, csv_path in enumerate(csvs[start_idx:], start=start_idx):
            run_specs.append(_make_spec(case_name, "run", idx, csv_path))

    if not run_specs:
        raise ValueError(f"No target runs found for base case '{base_case}'.")

    reference_group, reference_label, _ = analysis.parse_run_spec(reference_spec)
    template_group, template_label, _ = analysis.parse_run_spec(template_spec)
    parsed_runs = [analysis.parse_run_spec(spec) for spec in run_specs]
    target_keys = {(group, label) for group, label, _ in parsed_runs}

    df = analysis.load_logs(reference_spec, [template_spec, *run_specs], device_id="scenario_device")
    summary = analysis.build_summary(
        df=df,
        reference_group=reference_group,
        reference_label=reference_label,
        template_group=template_group,
        template_label=template_label,
        target_keys=target_keys,
        bins=bins,
    )
    summary["base_case"] = base_case
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_DIR))
    p.add_argument("--out-dir", default=str(ROOT_DIR / "iterative_analysis"))
    p.add_argument("--bins", type=int, default=50)
    p.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="minimum CSV count required for a case to be used as a base",
    )
    p.add_argument(
        "--title",
        default="Iterative Scenario Comparison",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _discover_cases(root)
    base_cases = _ordered_case_names(cases, min_runs=args.min_runs)
    if not base_cases:
        raise ValueError(f"No cases found in {root} with at least {args.min_runs} CSVs.")

    manifest_rows: list[dict[str, Any]] = []

    for base_case in base_cases:
        summary = _build_case_summary(base_case=base_case, cases=cases, bins=args.bins)
        csv_path = out_dir / f"{_slugify(base_case)}_compare.csv"
        summary.to_csv(csv_path, index=False)
        png_path = out_dir / f"{_slugify(base_case)}_box.png"
        _plot_case_summary(
            base_case=base_case,
            df=summary,
            out_path=png_path,
            title_prefix=args.title,
        )
        manifest_rows.append(
            {
                "base_case": base_case,
                "reference_csv": str(cases[base_case][0].resolve()),
                "template_csv": str(cases[base_case][1].resolve()),
                "summary_csv": str(csv_path),
                "plot_png": str(png_path),
                "num_target_rows": len(summary),
            }
        )
        print(f"Saved summary CSV: {csv_path}")
        print(f"Saved plot to: {png_path}")

    manifest_path = out_dir / "iterative_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved manifest CSV: {manifest_path}")


if __name__ == "__main__":
    main()
