from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


NON_METRIC_COLUMNS = {
    "perf_status",
    "perf_error",
    "perf_interval_ms",
    "perf_time_ms",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("perf_status") == "ok"]


def numeric_perf_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    cols = []
    for col in rows[0].keys():
        if not col.startswith("perf_") or col in NON_METRIC_COLUMNS:
            continue
        values = []
        for row in rows:
            raw = row.get(col, "")
            if raw in ("", None):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                values = []
                break
        if values:
            cols.append(col)
    return cols


def column_values(rows: list[dict[str, str]], col: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        raw = row.get(col, "")
        if raw in ("", None):
            continue
        vals.append(float(raw))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-csv", required=True)
    parser.add_argument("--blurring-csv", required=True)
    parser.add_argument("--out", default="2nd-submission/figures/perf_metrics_boxplots.png")
    args = parser.parse_args()

    clean_path = Path(args.clean_csv)
    blurring_path = Path(args.blurring_csv)
    out_path = Path(args.out)

    clean_rows = load_rows(clean_path)
    blurring_rows = load_rows(blurring_path)
    metrics = numeric_perf_columns(clean_rows)

    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4 * nrows, 8)))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        clean_vals = column_values(clean_rows, metric)
        blur_vals = column_values(blurring_rows, metric)
        bp = ax.boxplot(
            [clean_vals, blur_vals],
            tick_labels=["Clean", "Blurring30"],
            patch_artist=True,
            widths=0.55,
        )
        colors = ["#d4a017", "#2a6f97"]
        for i, patch in enumerate(bp["boxes"]):
            patch.set(facecolor=colors[i], edgecolor="black", alpha=0.72)
        for key in ("whiskers", "caps", "medians"):
            for artist in bp[key]:
                artist.set(color="black", linewidth=1.0)
        for i, flier in enumerate(bp["fliers"]):
            flier.set(marker="o", markersize=2.5, markerfacecolor=colors[i], markeredgecolor="black", alpha=0.45)
        ax.set_title(metric.replace("perf_", ""))
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle("Perf Metrics: Clean vs Blurring30", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"saved_plot={out_path}")


if __name__ == "__main__":
    main()
