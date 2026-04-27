from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("rmse_status") == "ok"]


def _layer_columns(rows: list[dict[str, str]], prefix: str) -> list[str]:
    if not rows:
        return []
    return sorted(c for c in rows[0].keys() if c.startswith(prefix))


def _series(rows: list[dict[str, str]], col: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        raw = row.get(col, "")
        if raw in ("", None):
            continue
        vals.append(float(raw))
    return vals


def _style(bp, color: str) -> None:
    for patch in bp["boxes"]:
        patch.set(facecolor=color, edgecolor="black", alpha=0.7)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set(color="black", linewidth=1.0)
    for flier in bp["fliers"]:
        flier.set(marker="o", markersize=2.5, markerfacecolor=color, markeredgecolor="black", alpha=0.45)


def _plot_group(ax, clean_rows, blur_rows, prefix: str, title: str) -> None:
    cols = _layer_columns(clean_rows or blur_rows, prefix)
    positions = []
    data = []
    colors = []
    labels = []

    for idx, col in enumerate(cols):
        base = idx * 3
        clean_vals = _series(clean_rows, col)
        blur_vals = _series(blur_rows, col)
        positions.extend([base + 1, base + 2])
        data.extend([clean_vals, blur_vals])
        colors.extend(["#d4a017", "#2a6f97"])
        labels.append(col.replace(prefix, ""))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        manage_ticks=False,
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set(facecolor=colors[i], edgecolor="black", alpha=0.72)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set(color="black", linewidth=1.0)
    for i, flier in enumerate(bp["fliers"]):
        flier.set(marker="o", markersize=2.5, markerfacecolor=colors[i], markeredgecolor="black", alpha=0.45)

    tick_positions = [idx * 3 + 1.5 for idx in range(len(labels))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-csv", required=True)
    parser.add_argument("--blurring-csv", required=True)
    parser.add_argument("--out", default="figures/rmse_cosine_sparsity_per_layer_box.png")
    args = parser.parse_args()

    clean_rows = _load_rows(Path(args.clean_csv))
    blur_rows = _load_rows(Path(args.blurring_csv))

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.ravel()

    _plot_group(axes[0], clean_rows, blur_rows, "forward_rmse_", "Forward RMSE by Layer")
    _plot_group(axes[1], clean_rows, blur_rows, "backward_rmse_", "Backward RMSE by Layer")
    _plot_group(axes[2], clean_rows, blur_rows, "forward_cosine_", "Forward Cosine by Layer")
    _plot_group(axes[3], clean_rows, blur_rows, "backward_cosine_", "Backward Cosine by Layer")
    _plot_group(axes[4], clean_rows, blur_rows, "forward_sparsity_", "Forward Sparsity by Layer")
    _plot_group(axes[5], clean_rows, blur_rows, "backward_sparsity_", "Backward Sparsity by Layer")

    handles = [
        plt.Line2D([0], [0], color="#d4a017", lw=8, alpha=0.72),
        plt.Line2D([0], [0], color="#2a6f97", lw=8, alpha=0.72),
    ]
    fig.legend(handles, ["Clean", "Blurring30"], loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Clean vs Blurring30 by Layer", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"saved_plot={out_path}")


if __name__ == "__main__":
    main()
