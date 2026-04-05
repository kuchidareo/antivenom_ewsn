from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def _per_device_round(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "round" in numeric_cols:
        numeric_cols.remove("round")
    # Keep round/poisoning_rate numeric columns, but they should be identical per group
    agg = {c: "mean" for c in numeric_cols}
    non_numeric = [c for c in df.columns if c not in numeric_cols]
    for c in non_numeric:
        agg[c] = "first"
    return df.groupby("round", dropna=False, as_index=False).agg(agg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="fl_main.csv")
    p.add_argument("--out", type=str, default="fl_synthetic_blur.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-scale", type=float, default=0.01, help="std as fraction of column std")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    required = ["device_id", "poisoning_type", "poisoning_rate", "round"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    device_pairs = {
        "112": ("114", "115"),
        "113": ("115", "116"),
        "119": ("116", "117"),
        "121": ("117", "118"),
        "122": ("118", "114"),
    }

    rng = np.random.default_rng(args.seed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # We will override these explicitly
    for col in ["device_id", "poisoning_rate", "round"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    out_rows = []
    def _synthesize_block(block_df: pd.DataFrame, poisoning_type: str, poisoning_rate: float):
        for new_device, (dev_a, dev_b) in device_pairs.items():
            a_df = block_df[block_df["device_id"].astype(str) == dev_a]
            b_df = block_df[block_df["device_id"].astype(str) == dev_b]
            if a_df.empty or b_df.empty:
                continue

            a_round = _per_device_round(a_df)
            b_round = _per_device_round(b_df)

            merged = pd.merge(a_round, b_round, on="round", suffixes=("_a", "_b"))
            if merged.empty:
                continue

            pooled = pd.concat([a_df, b_df], ignore_index=True)
            col_std = pooled[numeric_cols].std(ddof=0).replace(0, 0.0)

            for _, row in merged.iterrows():
                out = {}
                for col in df.columns:
                    if col in numeric_cols:
                        val = 0.5 * (row[f"{col}_a"] + row[f"{col}_b"])
                        noise = rng.normal(0.0, args.noise_scale * col_std.get(col, 0.0))
                        out[col] = float(val + noise)
                    elif col == "device_id":
                        out[col] = new_device
                    elif col == "poisoning_type":
                        out[col] = poisoning_type
                    elif col == "poisoning_rate":
                        out[col] = poisoning_rate
                    elif col == "round":
                        out[col] = row["round"]
                    else:
                        out[col] = row.get(f"{col}_a", row.get(col, None))
                out_rows.append(out)

    # blurring 0.30
    blur = df[
        (df["poisoning_type"] == "blurring") & (df["poisoning_rate"] == 0.30)
    ].copy()
    _synthesize_block(blur, "blurring", 0.30)

    # clean (all rates merged by round)
    clean = df[df["poisoning_type"] == "clean"].copy()
    if not clean.empty:
        # set poison rate to 0.0 for synthetic clean rows
        _synthesize_block(clean, "clean", 0.0)

    out_df = pd.DataFrame(out_rows)
    combined = pd.concat([df, out_df], ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"Saved synthetic CSV to: {args.out}")


if __name__ == "__main__":
    main()
