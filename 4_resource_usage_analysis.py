from __future__ import annotations

from pathlib import Path

import pandas as pd

from log_loader import load_logs


def resource_usage_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "cpu_percent",
        "mem_percent",
        "epoch",
        "disk_read_bytes",
        "disk_write_bytes",
        "disk_read_count",
        "disk_write_count",
        "disk_read_bytes_delta",
        "disk_write_bytes_delta",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Training rows only (epoch numeric).
    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()

    # Coerce cpu/mem/disk to numeric
    df["cpu_percent"] = pd.to_numeric(df["cpu_percent"], errors="coerce")
    df["mem_percent"] = pd.to_numeric(df["mem_percent"], errors="coerce")
    df["disk_read_bytes"] = pd.to_numeric(df["disk_read_bytes"], errors="coerce")
    df["disk_write_bytes"] = pd.to_numeric(df["disk_write_bytes"], errors="coerce")
    df["disk_read_count"] = pd.to_numeric(df["disk_read_count"], errors="coerce")
    df["disk_write_count"] = pd.to_numeric(df["disk_write_count"], errors="coerce")
    df["disk_read_bytes_delta"] = pd.to_numeric(df["disk_read_bytes_delta"], errors="coerce")
    df["disk_write_bytes_delta"] = pd.to_numeric(df["disk_write_bytes_delta"], errors="coerce")

    summary = (
        df.groupby(["device_id", "poisoning_type", "poison_frac"], dropna=False)
        .agg(
            cpu_mean=("cpu_percent", "mean"),
            mem_mean=("mem_percent", "mean"),
            disk_read_bytes_mean=("disk_read_bytes", "mean"),
            disk_write_bytes_mean=("disk_write_bytes", "mean"),
            disk_read_count_mean=("disk_read_count", "mean"),
            disk_write_count_mean=("disk_write_count", "mean"),
            disk_read_bytes_delta_mean=("disk_read_bytes_delta", "mean"),
            disk_write_bytes_delta_mean=("disk_write_bytes_delta", "mean"),
            rows=("cpu_percent", "count"),
        )
        .reset_index()
        .sort_values(["device_id", "poisoning_type", "poison_frac"])
    )

    return summary


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    summary = resource_usage_summary(df)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(summary)
