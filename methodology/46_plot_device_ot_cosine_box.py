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
    if "cosine_similarity" not in df.columns:
        df["cosine_similarity"] = 0.8 * df["core_type_cosine_mean_to_clean010"] + 0.2 * df[
            "mem_type_cosine_to_clean010"
        ]
    return df


def _plot_box_grouped(ax, df: pd.DataFrame, value_col: str, title: str, y_label: str) -> None:
    rates = sorted(df["poisoning_rate"].dropna().unique())
    types = sorted(df["poisoning_type"].dropna().unique())

    n_types = max(len(types), 1)
    width = 0.15
    gap = 0.05
    total_width = n_types * width + (n_types - 1) * gap

    base_positions = np.arange(len(rates))
    positions = []
    data = []
    labels = []
    colors = plt.cm.tab10.colors

    for t_idx, ptype in enumerate(types):
        offset = -total_width / 2 + t_idx * (width + gap) + width / 2
        for r_idx, rate in enumerate(rates):
            grp = df[(df["poisoning_type"] == ptype) & (df["poisoning_rate"] == rate)]
            vals = grp[value_col].astype(float).to_numpy()
            if len(vals) == 0:
                continue
            positions.append(base_positions[r_idx] + offset)
            data.append(vals)
            labels.append((ptype, rate))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    for i, box in enumerate(bp["boxes"]):
        ptype, _ = labels[i]
        color = colors[types.index(ptype) % len(colors)]
        box.set(facecolor=color, alpha=0.6, edgecolor="black")

    ax.set_title(title)
    ax.set_xlabel("poisoning_rate")
    ax.set_ylabel(y_label)
    ax.set_xticks(base_positions)
    ax.set_xticklabels([str(r) for r in rates])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    handles = []
    for t_idx, ptype in enumerate(types):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=colors[t_idx % len(colors)],
                markeredgecolor="black",
                label=str(ptype),
            )
        )
    if handles:
        ax.legend(handles=handles, title="poisoning_type", loc="best")


def plot_per_device(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    devices = sorted(df["device_id"].dropna().unique())
    for device_id in devices:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
        sub = df[df["device_id"] == device_id].copy()

        _plot_box_grouped(
            axes[0],
            sub,
            "ot_distance",
            "OT Distance (box)",
            "OT Distance",
        )
        _plot_box_grouped(
            axes[1],
            sub,
            "cosine_similarity",
            "Cosine Similarity (box)",
            "Cosine Similarity",
        )

        fig.tight_layout()
        out_path = out_dir / f"device_{device_id}_ot_cosine_box.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


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
