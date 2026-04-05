from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def _display_label(row: pd.Series) -> str:
    group = str(row["target_group"])
    label = str(row["target_label"])
    return f"{group}:{label}" if group != label else label


def _plot_box(ax, df: pd.DataFrame, value_col: str, y_label: str) -> None:
    labels = df["display_label"].drop_duplicates().tolist()
    data = []
    keep_labels = []
    for label in labels:
        vals = df.loc[df["display_label"] == label, value_col].astype(float).dropna().to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        keep_labels.append(label)
    if not data:
        ax.set_axis_off()
        return
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    colors = plt.cm.tab10.colors
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=colors[i % len(colors)], alpha=0.6, edgecolor="black")
    ax.set_xticks(range(1, len(keep_labels) + 1))
    ax.set_xticklabels(keep_labels, rotation=25, ha="right")
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="senario_evaluation/baseclean_box.png")
    p.add_argument("--title", default="Baseclean Comparison")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    required = ["target_group", "target_label", "ot_distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["display_label"] = df.apply(_display_label, axis=1)

    has_cosine = "cosine_similarity" in df.columns and df["cosine_similarity"].notna().any()
    if has_cosine:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        _plot_box(axes[0], df, "ot_distance", "Magnitude score W")
        _plot_box(axes[1], df, "cosine_similarity", "Similarity score S")
        fig.suptitle(args.title)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        _plot_box(ax, df, "ot_distance", "Magnitude score W")
        fig.suptitle(args.title)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
