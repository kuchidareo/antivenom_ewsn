from __future__ import annotations

from pathlib import Path
import argparse

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="fl_main.csv")
    p.add_argument("--out", type=str, default="4_figures/ot_vs_similarity_scatter.png")
    p.add_argument("--device-id", type=str, default="114")
    p.add_argument("--legend-out", type=str, default="4_figures/ot_vs_similarity_scatter_legend.png")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    required = [
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

    poison_types = {"blurring", "occlusion", "steganography", "label-flip"}

    device_id = str(args.device_id).strip()
    if not device_id:
        raise ValueError("--device-id must be provided")

    sub = df[df["device_id"].astype(str) == device_id].copy()
    if sub.empty:
        raise ValueError(f"No data for device_id={device_id}")

    clean = sub[sub["poisoning_type"] == "clean"]
    poison = sub[
        (sub["poisoning_type"].isin(poison_types))
        & (sub["poisoning_rate"] == 0.30)
    ]

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(
        poison["ot_distance"],
        poison["cosine_similarity"],
        s=18,
        alpha=0.7,
        color="red",
        label="poisoned (0.30)",
    )
    ax.scatter(
        clean["ot_distance"],
        clean["cosine_similarity"],
        s=18,
        alpha=0.7,
        color="black",
        label="clean",
    )

    ax.set_xlabel("Magnitude Score W")
    ax.set_ylabel("Similarity Score S")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    # ax.legend(loc="best")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")

    leg_fig = plt.figure(figsize=(3.5, 1.0))
    handles = [
        plt.Line2D([0], [0], marker="o", color="red", linestyle="None", label="poison30%"),
        plt.Line2D([0], [0], marker="o", color="black", linestyle="None", label="clean"),
    ]
    leg_fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="center", ncol=2)
    leg_fig.tight_layout()
    leg_path = Path(args.legend_out)
    leg_path.parent.mkdir(parents=True, exist_ok=True)
    leg_fig.savefig(leg_path, dpi=150, bbox_inches="tight")
    plt.close(leg_fig)
    print(f"Saved legend to: {leg_path}")


if __name__ == "__main__":
    main()
