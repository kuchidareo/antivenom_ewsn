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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="fl_main.csv")
    p.add_argument("--left-device", type=str, default="115")
    p.add_argument("--right-device", type=str, default="120")
    p.add_argument("--out", type=str, default="4_figures/device_compare_distance.png")
    p.add_argument("--legend-out", type=str, default="4_figures/device_compare_distance_legend.png")
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

    keep_types = {"clean", "blurring", "occlusion", "label-flip", "steganography"}
    df = df[df["poisoning_type"].isin(keep_types)].copy()

    left = df[df["device_id"].astype(str) == args.left_device].copy()
    right = df[df["device_id"].astype(str) == args.right_device].copy()

    for sub in (left, right):
        clean_mask = sub["poisoning_type"] == "clean"
        sub.loc[clean_mask, "poisoning_rate"] = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(4, 2), sharex=False)

    # Build combined axis: left device then right device with a gap
    left = left.copy()
    right = right.copy()
    left["panel"] = "left"
    right["panel"] = "right"
    combined = pd.concat([left, right], ignore_index=True)

    # Shift right device poisoning_rate positions by adding an offset index
    rates = sorted(combined["poisoning_rate"].dropna().unique())
    rate_to_idx = {r: i for i, r in enumerate(rates)}
    gap = 0.6
    combined["x_pos"] = combined["poisoning_rate"].map(rate_to_idx).astype(float)
    combined.loc[combined["panel"] == "right", "x_pos"] += (len(rates) + gap)

    # Plot using custom positions
    width = 0.32
    gap_inner = 0.07
    positions = []
    data = []
    labels = []
    colors = plt.cm.tab10.colors
    types = sorted(combined["poisoning_type"].dropna().unique())
    type_to_color = {t: colors[i % len(colors)] for i, t in enumerate(types)}

    for panel in ["left", "right"]:
        panel_df = combined[combined["panel"] == panel]
        for x_val in sorted(panel_df["x_pos"].dropna().unique()):
            sub_rate = panel_df[panel_df["x_pos"] == x_val]
            types_here = sorted(sub_rate["poisoning_type"].dropna().unique())
            n_types = max(len(types_here), 1)
            total_width = n_types * width + (n_types - 1) * gap_inner
            for t_idx, ptype in enumerate(types_here):
                offset = -total_width / 2 + t_idx * (width + gap_inner) + width / 2
                grp = sub_rate[sub_rate["poisoning_type"] == ptype]
                vals = grp["ot_distance"].astype(float).to_numpy()
                if len(vals) == 0:
                    continue
                positions.append(x_val + offset)
                data.append(vals)
                labels.append(ptype)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    for i, box in enumerate(bp["boxes"]):
        ptype = labels[i]
        color = type_to_color.get(ptype, colors[0])
        box.set(facecolor=color, alpha=0.6, edgecolor="black")

    ax.set_xlabel("Poisoning rate (%)")
    ax.set_ylabel("Magnitude score W")

    left_ticks = [rate_to_idx[r] for r in rates]
    right_ticks = [rate_to_idx[r] + len(rates) + gap for r in rates]
    ax.set_xticks(left_ticks + right_ticks)
    ax.set_xticklabels(
        [str(int(r * 100)) for r in rates] + [str(int(r * 100)) for r in rates]
    )
    # ax.set_title(...)

    # Vertical dashed line between panels
    mid = (max(left_ticks) + min(right_ticks)) / 2
    ax.axvline(mid, color="black", linestyle="--", linewidth=1)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")

    # Legend
    colors = plt.cm.tab10.colors
    types = sorted(df["poisoning_type"].dropna().unique())
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=colors[i % len(colors)],
            markeredgecolor="black",
            label=t,
        )
        for i, t in enumerate(types)
    ]
    leg_fig = plt.figure(figsize=(6.0, 1.2))
    leg_fig.legend(handles=handles, labels=types, ncol=5, loc="center")
    leg_fig.tight_layout()
    leg_path = Path(args.legend_out)
    leg_path.parent.mkdir(parents=True, exist_ok=True)
    leg_fig.savefig(leg_path, dpi=150, bbox_inches="tight")
    plt.close(leg_fig)
    print(f"Saved legend to: {leg_path}")


if __name__ == "__main__":
    main()
