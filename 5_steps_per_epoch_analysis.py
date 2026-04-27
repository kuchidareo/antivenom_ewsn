from __future__ import annotations

from pathlib import Path

import pandas as pd

from log_loader import load_logs


def steps_per_epoch_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = ["epoch", "step", "ts_unix"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Training rows only (epoch numeric)
    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()
    df["epoch"] = epoch_numeric.loc[df.index]

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.loc[df["step"].notna()].copy()

    per_epoch = (
        df.groupby(["device_id", "poisoning_type", "poison_frac", "epoch"], dropna=False)
        .agg(
            steps=("step", lambda s: int(s.max() - s.min() + 1) if len(s) else 0),
            epoch_start=("ts_unix", "min"),
            epoch_end=("ts_unix", "max"),
        )
        .reset_index()
    )
    per_epoch["epoch_duration"] = per_epoch["epoch_end"] - per_epoch["epoch_start"]
    per_epoch["time_per_step"] = per_epoch["epoch_duration"] / per_epoch["steps"].replace(0, pd.NA)

    summary = (
        per_epoch.groupby(["device_id", "poisoning_type", "poison_frac"], dropna=False)
        .agg(
            epochs=("epoch", "count"),
            steps_mean=("steps", "mean"),
            steps_median=("steps", "median"),
            steps_std=("steps", "std"),
            epoch_duration_mean=("epoch_duration", "mean"),
            time_per_step_mean=("time_per_step", "mean"),
        )
        .reset_index()
        .sort_values(["device_id", "poisoning_type", "poison_frac"])
    )

    return summary


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    summary = steps_per_epoch_summary(df)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(summary)
