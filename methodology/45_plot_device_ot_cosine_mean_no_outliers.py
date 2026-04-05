from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _robust_z(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or not np.isfinite(mad):
        return pd.Series([0.0] * len(x), index=x.index)
    return 0.6745 * (x - med) / mad


def _drop_outliers(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    group_cols = ["device_id", "poisoning_type", "poisoning_rate"]
    keep = pd.Series(True, index=df.index)

    for _, grp in df.groupby(group_cols, dropna=False):
        z_ot = _robust_z(grp["ot_distance"])
        z_cos = _robust_z(grp["cosine_similarity"])
        mask = (z_ot.abs() <= z_thresh) & (z_cos.abs() <= z_thresh)
        keep.loc[grp.index] = mask

    return df.loc[keep].copy()


def _agg_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["device_id", "poisoning_type", "poisoning_rate"], dropna=False)[
            value_col
        ]
        .agg(["mean"])
        .reset_index()
    )
    return grouped


def plot_per_device(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ot_stats = _agg_mean(df, "ot_distance")
    cos_stats = _agg_mean(df, "cosine_similarity")

    devices = sorted(df["device_id"].dropna().unique())
    for device_id in devices:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

        for ax, stats, title, y_label in [
            (axes[0], ot_stats, "OT Distance (mean ± std)", "OT Distance"),
            (axes[1], cos_stats, "Cosine Similarity (mean ± std)", "Cosine Similarity"),
        ]:
            sub = stats[stats["device_id"] == device_id].copy()
            for ptype, grp in sub.groupby("poisoning_type", dropna=False):
                grp = grp.sort_values("poisoning_rate")
                ax.plot(
                    grp["poisoning_rate"],
                    grp["mean"],
                    marker="o",
                    linewidth=1.5,
                    label=str(ptype),
                )
            ax.set_title(title)
            ax.set_xlabel("poisoning_rate")
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        axes[1].legend(title="poisoning_type", loc="best")
        fig.tight_layout()
        out_path = out_dir / f"device_{device_id}_ot_cosine.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "ot_distance" not in df.columns:
        df["ot_distance"] = 0.8 * df["core_ot_distance_mean"] + 0.2 * df[
            "mem_ot_distance_to_clean_005"
        ]
    if "cosine_similarity" not in df.columns:
        df["cosine_similarity"] = 0.8 * df["core_type_cosine_mean_to_clean010"] + 0.2 * df[
            "mem_type_cosine_to_clean010"
        ]
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="fl_main.csv")
    p.add_argument("--out-dir", type=str, default="4_figures")
    p.add_argument("--z-thresh", type=float, default=3.5)
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    required = [
        "device_id",
        "poisoning_type",
        "poisoning_rate",
        "core_ot_distance_mean",
        "mem_ot_distance_to_clean_005",
        "core_type_cosine_mean_to_clean010",
        "mem_type_cosine_to_clean010",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _ensure_derived_columns(df)
    df = _drop_outliers(df, z_thresh=float(args.z_thresh))

    plot_per_device(df, Path(args.out_dir))


if __name__ == "__main__":
    main()
