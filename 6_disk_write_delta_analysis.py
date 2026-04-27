from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from log_loader import load_logs


def _wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    """1D Wasserstein distance on a uniform grid in [0,1]."""
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.mean(np.abs(cdf_p - cdf_q)))


def _epoch_shape(group: pd.DataFrame, bins: int) -> np.ndarray:
    t0 = group["ts_unix"].min()
    t1 = group["ts_unix"].max()
    if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
        return np.zeros(bins, dtype=float)
    t = (group["ts_unix"] - t0) / (t1 - t0)
    values = group["disk_write_bytes_delta"].astype(float).values
    hist, _ = np.histogram(t, bins=bins, range=(0.0, 1.0), weights=values)
    return hist.astype(float)


def disk_write_shape_ot(df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    required = ["epoch", "disk_write_bytes_delta", "ts_unix"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Training rows only (epoch numeric)
    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()
    df["epoch"] = epoch_numeric.loc[df.index]

    df["disk_write_bytes_delta"] = pd.to_numeric(df["disk_write_bytes_delta"], errors="coerce")
    df = df.loc[df["disk_write_bytes_delta"].notna()].copy()

    per_epoch_shapes: dict[tuple, list[np.ndarray]] = {}
    for key, group in df.groupby(
        ["device_id", "poisoning_type", "poison_frac", "epoch"], dropna=False
    ):
        shape = _epoch_shape(group, bins=bins)
        per_epoch_shapes.setdefault(key[:3], []).append(shape)

    rows: list[dict[str, object]] = []
    for (device_id, poisoning_type, poison_frac), shapes in per_epoch_shapes.items():
        shapes_arr = np.vstack(shapes) if shapes else np.zeros((0, bins))
        mean_shape = shapes_arr.mean(axis=0) if len(shapes_arr) else np.zeros(bins)
        rows.append(
            {
                "device_id": device_id,
                "poisoning_type": poisoning_type,
                "poison_frac": poison_frac,
                "shape_mean": json.dumps(mean_shape.tolist()),
            }
        )

    summary = pd.DataFrame(rows)

    # Compute OT distance to clean per device/poison_frac (clean is reference)
    distances: list[dict[str, object]] = []
    for (device_id, poison_frac), g in summary.groupby(["device_id", "poison_frac"], dropna=False):
        clean_row = g[g["poisoning_type"] == "clean"]
        if clean_row.empty:
            continue
        clean_shape = np.array(json.loads(clean_row.iloc[0]["shape_mean"]))
        for _, row in g.iterrows():
            shape = np.array(json.loads(row["shape_mean"]))
            dist = _wasserstein_1d(clean_shape, shape)
            distances.append(
                {
                    "device_id": device_id,
                    "poison_frac": poison_frac,
                    "poisoning_type": row["poisoning_type"],
                    "ot_distance_to_clean": dist,
                }
            )

    # Also compare all groups to clean reference at poison_frac=0.05 (per device).
    ref_rows = summary[
        (summary["poisoning_type"] == "clean") & (summary["poison_frac"] == 0.05)
    ]
    ref_by_device = {
        row["device_id"]: np.array(json.loads(row["shape_mean"]))
        for _, row in ref_rows.iterrows()
    }
    for _, row in summary.iterrows():
        ref = ref_by_device.get(row["device_id"])
        if ref is None:
            continue
        shape = np.array(json.loads(row["shape_mean"]))
        dist = _wasserstein_1d(ref, shape)
        distances.append(
            {
                "device_id": row["device_id"],
                "poison_frac": row["poison_frac"],
                "poisoning_type": row["poisoning_type"],
                "ot_distance_to_clean_005": dist,
            }
        )

    dist_df = pd.DataFrame(distances).groupby(
        ["device_id", "poison_frac", "poisoning_type"], dropna=False
    ).first().reset_index()
    return dist_df


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    summary = disk_write_shape_ot(df, bins=50)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(summary)
