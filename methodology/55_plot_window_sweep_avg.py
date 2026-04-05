from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="window_sweep.csv")
    p.add_argument("--out-dir", type=str, default="5_figures")
    p.add_argument("--windows", type=str, default="")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    required = ["window", "best_f1", "best_fpr", "best_fnr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary = (
        df.groupby("window", dropna=False)[["best_f1", "best_fpr", "best_fnr"]]
        .mean()
        .reset_index()
        .sort_values("window")
    )
    if args.windows.strip():
        keep = {int(s.strip()) for s in args.windows.split(",") if s.strip()}
        summary = summary[summary["window"].isin(keep)].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(1, 1, figsize=(3, 2))
    x_labels = [str(int(w)) for w in summary["window"].to_numpy()]
    x_pos = range(len(x_labels))
    ax1.bar(x_pos, summary["best_f1"], color="#4c78a8", alpha=0.6, edgecolor="black")
    ax1.set_xlabel("Monitoring Round Size")
    ax1.set_ylabel("Average F1")
    # ax1.set_title("Average F1 vs Window Size")
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xticks(list(x_pos))
    ax1.set_xticklabels(x_labels)
    fig1.tight_layout()
    out1 = out_dir / "window_sweep_avg_f1.png"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(3, 2))
    ax2.bar(
        [p - 0.18 for p in x_pos],
        summary["best_fpr"],
        width=0.36,
        label="FPR",
        color="#ff7f0e",
        alpha=0.6,
        edgecolor="black",
    )
    ax2.bar(
        [p + 0.18 for p in x_pos],
        summary["best_fnr"],
        width=0.36,
        label="FNR",
        color="#2ca02c",
        alpha=0.6,
        edgecolor="black",
    )
    ax2.set_xlabel("Monitoring Round Size")
    ax2.set_ylabel("Average Rate")
    # ax2.set_title("Average FPR/FNR vs Window Size")
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    # ax2.legend(loc="best")
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(x_labels)
    fig2.tight_layout()
    out2 = out_dir / "window_sweep_avg_fpr_fnr.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)

    leg_fig = plt.figure(figsize=(4.0, 1.0))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#ff7f0e",
            markeredgecolor="black",
            label="FPR",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#2ca02c",
            markeredgecolor="black",
            label="FNR",
        ),
    ]
    leg_fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="center", ncol=2)
    leg_fig.tight_layout()
    leg_path = out_dir / "window_sweep_avg_fpr_fnr_legend.png"
    leg_fig.savefig(leg_path, dpi=150, bbox_inches="tight")
    plt.close(leg_fig)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
