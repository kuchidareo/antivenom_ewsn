from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from log_loader import load_logs


def _wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.mean(np.abs(cdf_p - cdf_q)))


def _epoch_shape_mean(group: pd.DataFrame, value_col: str, bins: int) -> np.ndarray:
    t0 = group["ts_unix"].min()
    t1 = group["ts_unix"].max()
    if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
        return np.zeros(bins, dtype=float)
    t = (group["ts_unix"] - t0) / (t1 - t0)
    values = group[value_col].astype(float).values
    sums, _ = np.histogram(t, bins=bins, range=(0.0, 1.0), weights=values)
    counts, _ = np.histogram(t, bins=bins, range=(0.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
    return means.astype(float)


def cpu_mem_shape_ot(df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    required = ["epoch", "ts_unix", "cpu_percent", "mem_percent", "cpu_temp_c"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Training rows only (epoch numeric)
    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()
    df["epoch"] = epoch_numeric.loc[df.index]

    df["cpu_percent"] = pd.to_numeric(df["cpu_percent"], errors="coerce")
    df["mem_percent"] = pd.to_numeric(df["mem_percent"], errors="coerce")
    df["cpu_temp_c"] = pd.to_numeric(df["cpu_temp_c"], errors="coerce")
    df = df.loc[
        df["cpu_percent"].notna()
        & df["mem_percent"].notna()
        & df["cpu_temp_c"].notna()
    ].copy()

    per_epoch_shapes: dict[tuple, dict[str, list[np.ndarray]]] = {}
    for key, group in df.groupby(
        ["device_id", "poisoning_type", "poison_frac", "epoch"], dropna=False
    ):
        cpu_shape = _epoch_shape_mean(group, "cpu_percent", bins=bins)
        mem_shape = _epoch_shape_mean(group, "mem_percent", bins=bins)
        temp_shape = _epoch_shape_mean(group, "cpu_temp_c", bins=bins)
        per_epoch_shapes.setdefault(key[:3], {"cpu": [], "mem": [], "temp": []})
        per_epoch_shapes[key[:3]]["cpu"].append(cpu_shape)
        per_epoch_shapes[key[:3]]["mem"].append(mem_shape)
        per_epoch_shapes[key[:3]]["temp"].append(temp_shape)

    rows: list[dict[str, object]] = []
    for (device_id, poisoning_type, poison_frac), shapes in per_epoch_shapes.items():
        cpu_arr = np.vstack(shapes["cpu"]) if shapes["cpu"] else np.zeros((0, bins))
        mem_arr = np.vstack(shapes["mem"]) if shapes["mem"] else np.zeros((0, bins))
        temp_arr = np.vstack(shapes["temp"]) if shapes["temp"] else np.zeros((0, bins))
        cpu_mean_shape = cpu_arr.mean(axis=0) if len(cpu_arr) else np.zeros(bins)
        mem_mean_shape = mem_arr.mean(axis=0) if len(mem_arr) else np.zeros(bins)
        temp_mean_shape = temp_arr.mean(axis=0) if len(temp_arr) else np.zeros(bins)
        rows.append(
            {
                "device_id": device_id,
                "poisoning_type": poisoning_type,
                "poison_frac": poison_frac,
                "cpu_shape_mean": json.dumps(cpu_mean_shape.tolist()),
                "mem_shape_mean": json.dumps(mem_mean_shape.tolist()),
                "temp_shape_mean": json.dumps(temp_mean_shape.tolist()),
            }
        )

    summary = pd.DataFrame(rows)

    # Reference clean at poison_frac=0.05 per device.
    ref_rows = summary[
        (summary["poisoning_type"] == "clean") & (summary["poison_frac"] == 0.05)
    ]
    ref_by_device_cpu = {
        row["device_id"]: np.array(json.loads(row["cpu_shape_mean"]))
        for _, row in ref_rows.iterrows()
    }
    ref_by_device_mem = {
        row["device_id"]: np.array(json.loads(row["mem_shape_mean"]))
        for _, row in ref_rows.iterrows()
    }
    ref_by_device_temp = {
        row["device_id"]: np.array(json.loads(row["temp_shape_mean"]))
        for _, row in ref_rows.iterrows()
    }

    distances: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        ref_cpu = ref_by_device_cpu.get(row["device_id"])
        ref_mem = ref_by_device_mem.get(row["device_id"])
        ref_temp = ref_by_device_temp.get(row["device_id"])
        if ref_cpu is None or ref_mem is None or ref_temp is None:
            continue
        cpu_shape = np.array(json.loads(row["cpu_shape_mean"]))
        mem_shape = np.array(json.loads(row["mem_shape_mean"]))
        temp_shape = np.array(json.loads(row["temp_shape_mean"]))
        cpu_dist = _wasserstein_1d(ref_cpu, cpu_shape)
        mem_dist = _wasserstein_1d(ref_mem, mem_shape)
        temp_dist = _wasserstein_1d(ref_temp, temp_shape)
        distances.append(
            {
                "device_id": row["device_id"],
                "poison_frac": row["poison_frac"],
                "poisoning_type": row["poisoning_type"],
                "cpu_ot_distance_to_clean_005": cpu_dist,
                "mem_ot_distance_to_clean_005": mem_dist,
                "temp_ot_distance_to_clean_005": temp_dist,
            }
        )

    dist_df = pd.DataFrame(distances).sort_values(
        ["device_id", "poison_frac", "poisoning_type"]
    )
    return dist_df


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    summary = cpu_mem_shape_ot(df, bins=50)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(summary)
