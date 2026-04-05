from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _make_presence_matrix(df: pd.DataFrame, device_id: str):
    sub = df[df["device_id"].astype(str) == device_id].copy()
    types = sorted(sub["poisoning_type"].dropna().unique())
    rates = sorted(sub["poisoning_rate"].dropna().unique())
    if not types or not rates:
        return np.zeros((0, 0)), types, rates

    mat = np.zeros((len(types), len(rates)), dtype=int)
    for i, ptype in enumerate(types):
        for j, rate in enumerate(rates):
            exists = not sub[
                (sub["poisoning_type"] == ptype)
                & (sub["poisoning_rate"] == rate)
            ].empty
            mat[i, j] = 1 if exists else 0
    return mat, types, rates


def plot_coverage(df: pd.DataFrame, title: str, out_path: Path) -> None:
    devices = sorted(df["device_id"].dropna().astype(str).unique())
    if not devices:
        return

    mats = []
    types_all = set()
    rates_all = set()
    for device_id in devices:
        mat, types, rates = _make_presence_matrix(df, device_id)
        mats.append((device_id, mat, types, rates))
        types_all.update(types)
        rates_all.update(rates)

    types_all = sorted(types_all)
    rates_all = sorted(rates_all)

    fig, axes = plt.subplots(len(devices), 1, figsize=(10, 2.4 * len(devices)))
    if len(devices) == 1:
        axes = [axes]

    for ax, (device_id, mat, types, rates) in zip(axes, mats):
        grid = np.zeros((len(types_all), len(rates_all)), dtype=int)
        for i, ptype in enumerate(types_all):
            for j, rate in enumerate(rates_all):
                if ptype in types and rate in rates:
                    ii = types.index(ptype)
                    jj = rates.index(rate)
                    grid[i, j] = mat[ii, jj]
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(types_all)))
        ax.set_yticklabels(types_all)
        ax.set_xticks(range(len(rates_all)))
        ax.set_xticklabels([str(r) for r in rates_all], rotation=45, ha="right")
        ax.set_ylabel(f"device {device_id}")
        for i in range(len(types_all)):
            for j in range(len(rates_all)):
                ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=7, color="black")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csvs",
        type=str,
        default="fl_main.csv,sl_main.csv,bg_main.csv,bg2_main.csv,finetune_main.csv,fl_synthetic_blur.csv",
        help="comma-separated list of csv files",
    )
    p.add_argument("--out-dir", type=str, default="0_figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in [s.strip() for s in args.csvs.split(",") if s.strip()]:
        path = Path(csv_path)
        if not path.exists():
            print(f"[skip] missing: {path}")
            continue
        df = pd.read_csv(path)
        required = ["device_id", "poisoning_type", "poisoning_rate"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[skip] {path} missing columns: {missing}")
            continue
        title = f"Coverage: {path.name}"
        out_path = out_dir / f"coverage_{path.stem}.png"
        plot_coverage(df, title, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
