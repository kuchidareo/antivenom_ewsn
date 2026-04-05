from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "ot_distance" not in df.columns:
        df["ot_distance"] = 0.8 * df["core_ot_distance_mean"] + 0.2 * df[
            "mem_ot_distance_to_clean_005"
        ]
    return df


def _plot_side_by_side(ax, df: pd.DataFrame, value_col: str, y_label: str) -> None:
    devices = ["112", "113", "114", "115", "116", "117", "118", "119", "121", "122"]
    data = []
    positions = []
    labels = []

    base_positions = np.arange(len(devices))
    offset = 0.18

    for i, device_id in enumerate(devices):
        clean_vals = df[
            (df["device_id"].astype(str) == device_id)
            & (df["poisoning_type"] == "clean")
        ][value_col].astype(float).to_numpy()

        poison_vals = df[
            (df["device_id"].astype(str) == device_id)
            & (df["poisoning_type"].isin(["blurring", "occlusion", "label-flip", "steganography"]))
            & (df["poisoning_rate"] == 0.30)
        ][value_col].astype(float).to_numpy()

        if len(clean_vals) > 0:
            data.append(clean_vals)
            positions.append(base_positions[i] - offset)
            labels.append(f"{device_id}\nclean")
        if len(poison_vals) > 0:
            data.append(poison_vals)
            positions.append(base_positions[i] + offset)
            labels.append(f"{device_id}\npoison 0.30")

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
    )

    for idx, box in enumerate(bp["boxes"]):
        label = labels[idx]
        if "clean" in label:
            box.set(facecolor="#f58518", alpha=0.6, edgecolor="black")
        else:
            box.set(facecolor="#4c78a8", alpha=0.6, edgecolor="black")

    ax.set_xlabel("Device ID")
    ax.set_ylabel(y_label)
    ax.set_xticks(base_positions)
    ax.set_xticklabels([str(i + 1) for i in range(len(devices))])

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#f58518",
            markeredgecolor="black",
            label="clean",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#4c78a8",
            markeredgecolor="black",
            label="poison30%-agg",
        ),
    ]
    return handles


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="fl_main.csv")
    p.add_argument("--out", type=str, default="4_figures/clean_vs_poison30_agg.png")
    p.add_argument("--legend-out", type=str, default="4_figures/clean_vs_poison30_agg_legend.png")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    required = [
        "device_id",
        "poisoning_type",
        "poisoning_rate",
        "core_ot_distance_mean",
        "mem_ot_distance_to_clean_005",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _ensure_derived_columns(df)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3), sharex=False)
    handles = _plot_side_by_side(ax, df, "ot_distance", "Magnitude score W")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")

    leg_fig = plt.figure(figsize=(4.5, 1.2))
    leg_fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="center", ncol=2)
    leg_fig.tight_layout()
    leg_path = Path(args.legend_out)
    leg_path.parent.mkdir(parents=True, exist_ok=True)
    leg_fig.savefig(leg_path, dpi=150, bbox_inches="tight")
    plt.close(leg_fig)
    print(f"Saved legend to: {leg_path}")


if __name__ == "__main__":
    main()
