from __future__ import annotations

from pathlib import Path
import argparse
import ast

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one 4-row CPU usage plot per logs_120 scenario. "
            "Each row shows the rank-sorted core usage over time."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs_120"),
        help="Root directory containing scenario CSV logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("1_ranked_cpu_core_timeseries/plots"),
        help="Directory where the generated plots will be written.",
    )
    return parser.parse_args()


def parse_cpu_per_core(value: object) -> list[float] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, (list, tuple)):
        return None
    try:
        return sorted((float(x) for x in parsed), reverse=True)
    except (TypeError, ValueError):
        return None


def discover_scenarios(logs_root: Path) -> list[Path]:
    return sorted({csv_path.parent for csv_path in logs_root.rglob("*.csv")})


def load_ranked_run(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ts_unix" not in df.columns or "cpu_per_core" not in df.columns:
        raise ValueError(f"Missing required columns in {csv_path}")

    df["ts_unix"] = pd.to_numeric(df["ts_unix"], errors="coerce")
    ranked = df["cpu_per_core"].apply(parse_cpu_per_core)
    valid_mask = df["ts_unix"].notna() & ranked.notna()
    df = df.loc[valid_mask, ["ts_unix"]].copy()
    ranked = ranked.loc[valid_mask]
    if df.empty:
        raise ValueError(f"No valid CPU samples in {csv_path}")

    elapsed = df["ts_unix"] - df["ts_unix"].iloc[0]
    ranked_frame = pd.DataFrame(
        ranked.tolist(),
        index=df.index,
        columns=[f"rank_{idx + 1}" for idx in range(len(ranked.iloc[0]))],
    )
    run_df = pd.concat([elapsed.rename("elapsed_sec"), ranked_frame], axis=1)
    return run_df.reset_index(drop=True)


def scenario_slug(logs_root: Path, scenario_dir: Path) -> str:
    return "_".join(scenario_dir.relative_to(logs_root).parts)


def plot_scenario(logs_root: Path, scenario_dir: Path, output_dir: Path) -> Path:
    csv_paths = sorted(scenario_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in {scenario_dir}")

    run_frames: list[tuple[str, pd.DataFrame]] = []
    for csv_path in csv_paths:
        run_frames.append((csv_path.stem, load_ranked_run(csv_path)))

    rank_columns = [f"rank_{idx}" for idx in range(1, 5)]
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(16, 12),
        sharex=True,
        constrained_layout=True,
    )

    title = scenario_slug(logs_root, scenario_dir)
    fig.suptitle(f"Ranked CPU Core Usage Over Time: {title}", fontsize=16)

    for rank_idx, (ax, rank_col) in enumerate(zip(axes, rank_columns, strict=True), start=1):
        for run_name, run_df in run_frames:
            if rank_col not in run_df.columns:
                continue
            ax.plot(
                run_df["elapsed_sec"],
                run_df[rank_col],
                linewidth=1.2,
                alpha=0.8,
                label=run_name,
            )
        ax.set_ylabel(f"Rank {rank_idx}\nCPU %")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time From Run Start (seconds)")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{title}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    logs_root = args.logs_root.resolve()
    output_dir = args.output_dir.resolve()

    scenario_dirs = discover_scenarios(logs_root)
    if not scenario_dirs:
        raise SystemExit(f"No CSV scenarios found under {logs_root}")

    generated_paths: list[Path] = []
    for scenario_dir in scenario_dirs:
        generated_paths.append(plot_scenario(logs_root, scenario_dir, output_dir))

    print(f"Generated {len(generated_paths)} scenario plots in {output_dir}")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
