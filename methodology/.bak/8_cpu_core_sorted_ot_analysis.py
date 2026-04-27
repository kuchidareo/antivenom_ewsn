from __future__ import annotations

from pathlib import Path
import json
import ast

import numpy as np
import pandas as pd

from log_loader import load_logs


def _parse_core_list(value: object) -> list[float] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)):
        return [float(x) for x in parsed]
    return None


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


def cpu_core_sorted_ot(df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    required = ["epoch", "ts_unix", "cpu_per_core"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Training rows only (epoch numeric)
    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()
    df["epoch"] = epoch_numeric.loc[df.index]

    core_cols = ["core_0", "core_1", "core_2", "core_3"]
    core_vals = df["cpu_per_core"].apply(_parse_core_list)
    df = df.loc[core_vals.notna()].copy()
    sorted_vals = core_vals.apply(lambda v: sorted(v, reverse=True))

    for i, col in enumerate(core_cols):
        df[col] = sorted_vals.apply(lambda v: float(v[i]) if i < len(v) else float("nan"))

    df = df.loc[df[core_cols].notna().all(axis=1)].copy()

    per_epoch_shapes: dict[tuple, dict[str, list[np.ndarray]]] = {}
    for key, group in df.groupby(
        ["device_id", "poisoning_type", "poison_frac", "epoch"], dropna=False
    ):
        per_epoch_shapes.setdefault(key[:3], {c: [] for c in core_cols})
        for col in core_cols:
            shape = _epoch_shape_mean(group, col, bins=bins)
            per_epoch_shapes[key[:3]][col].append(shape)

    rows: list[dict[str, object]] = []
    for (device_id, poisoning_type, poison_frac), shapes in per_epoch_shapes.items():
        mean_shapes = {}
        for col in core_cols:
            arr = np.vstack(shapes[col]) if shapes[col] else np.zeros((0, bins))
            mean_shapes[col] = arr.mean(axis=0) if len(arr) else np.zeros(bins)
        rows.append(
            {
                "device_id": device_id,
                "poisoning_type": poisoning_type,
                "poison_frac": poison_frac,
                "core0_shape_mean": json.dumps(mean_shapes["core_0"].tolist()),
                "core1_shape_mean": json.dumps(mean_shapes["core_1"].tolist()),
                "core2_shape_mean": json.dumps(mean_shapes["core_2"].tolist()),
                "core3_shape_mean": json.dumps(mean_shapes["core_3"].tolist()),
            }
        )

    summary = pd.DataFrame(rows)

    # Reference clean at poison_frac=0.05 per device.
    ref_rows = summary[
        (summary["poisoning_type"] == "clean") & (summary["poison_frac"] == 0.05)
    ]
    ref_by_device = {
        row["device_id"]: {
            "core_0": np.array(json.loads(row["core0_shape_mean"])),
            "core_1": np.array(json.loads(row["core1_shape_mean"])),
            "core_2": np.array(json.loads(row["core2_shape_mean"])),
            "core_3": np.array(json.loads(row["core3_shape_mean"])),
        }
        for _, row in ref_rows.iterrows()
    }

    distances: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        ref = ref_by_device.get(row["device_id"])
        if ref is None:
            continue
        dists = []
        for idx in range(4):
            shape = np.array(json.loads(row[f"core{idx}_shape_mean"]))
            dist = _wasserstein_1d(ref[f"core_{idx}"], shape)
            dists.append(dist)
        distances.append(
            {
                "device_id": row["device_id"],
                "poison_frac": row["poison_frac"],
                "poisoning_type": row["poisoning_type"],
                "core0_ot_distance_to_clean_005": dists[0],
                "core1_ot_distance_to_clean_005": dists[1],
                "core2_ot_distance_to_clean_005": dists[2],
                "core3_ot_distance_to_clean_005": dists[3],
                "core_ot_distance_mean": float(np.mean(dists)),
            }
        )

    dist_df = pd.DataFrame(distances).sort_values(
        ["device_id", "poison_frac", "poisoning_type"]
    )
    return dist_df


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    summary = cpu_core_sorted_ot(df, bins=50)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(summary)
