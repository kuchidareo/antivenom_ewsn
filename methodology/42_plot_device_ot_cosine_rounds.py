from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_per_device(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    devices = sorted(df["device_id"].dropna().unique())
    for device_id in devices:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

        for ax, value_col, title, y_label in [
            (axes[0], "ot_distance", "OT Distance (per round)", "OT Distance"),
            (axes[1], "cosine_similarity", "Cosine Similarity (per round)", "Cosine Similarity"),
        ]:
            sub = df[df["device_id"] == device_id].copy()
            for idx, (ptype, grp) in enumerate(sub.groupby("poisoning_type", dropna=False)):
                grp = grp.sort_values("poisoning_rate")
                x = grp["poisoning_rate"].astype(float).to_numpy()
                jitter = (idx - 0.5) * 0.01
                ax.scatter(
                    x + jitter,
                    grp[value_col],
                    s=18,
                    alpha=0.7,
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

    plot_per_device(df, Path(args.out_dir))


if __name__ == "__main__":
    main()
