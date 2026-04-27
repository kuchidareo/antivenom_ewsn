from __future__ import annotations

from pathlib import Path

import pandas as pd

from log_loader import load_logs


def compute_epoch_durations(df: pd.DataFrame) -> pd.DataFrame:
    if "epoch" not in df.columns or "ts_unix" not in df.columns:
        raise ValueError("Missing required columns: epoch, ts_unix")

    # Training rows only (epoch numeric).
    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()
    df["epoch"] = epoch_numeric.loc[df.index]

    # Per epoch duration = max(ts_unix) - min(ts_unix)
    per_epoch = (
        df.groupby(["device_id", "poisoning_type", "poison_frac", "epoch"], dropna=False)["ts_unix"]
        .agg(["min", "max"])
        .reset_index()
    )
    per_epoch["epoch_duration"] = per_epoch["max"] - per_epoch["min"]
    return per_epoch


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    per_epoch = compute_epoch_durations(df)

    summary = (
        per_epoch.groupby(["device_id", "poisoning_type", "poison_frac"], dropna=False)
        .agg(
            epochs=("epoch", "count"),
            epoch_duration_mean=("epoch_duration", "mean"),
            epoch_duration_median=("epoch_duration", "median"),
            epoch_duration_std=("epoch_duration", "std"),
        )
        .reset_index()
        .sort_values(["device_id", "poisoning_type", "poison_frac"])
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(summary)
