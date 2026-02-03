from __future__ import annotations

from pathlib import Path
import json
import ast

import math
from typing import Any


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


def _wasserstein_1d(p: np.ndarray, q: np.ndarray, bin_width: float = 1.0) -> float:
    """Discrete 1D Wasserstein-1 distance on an equally-spaced grid.

    With cost |i-j| in *bin units*, W1 = sum_k |CDF_p[k] - CDF_q[k]|.
    Multiply by `bin_width` to convert to physical units (e.g., time in [0,1]).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)) * float(bin_width))


# --- 1D OT coupling and feature extraction ---


def _normalize_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= 0:
        return x
    return x / s


def _ot_coupling_1d(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the 1D optimal transport coupling between discrete histograms p and q.

    Assumes p and q are nonnegative and sum to 1 (or close). Uses the monotone (greedy) algorithm
    valid for 1D with cost |i-j|.

    Returns an (n,n) coupling matrix G where G[i,j] is mass moved from bin i to bin j.
    """
    p = _normalize_nonneg(p)
    q = _normalize_nonneg(q)
    if float(p.sum()) == 0.0 or float(q.sum()) == 0.0:
        return np.zeros((len(p), len(q)), dtype=float)

    n = len(p)
    m = len(q)
    G = np.zeros((n, m), dtype=float)
    i = 0
    j = 0
    pi = float(p[0]) if n else 0.0
    qj = float(q[0]) if m else 0.0

    # Copy because we'll mutate
    p_work = p.copy()
    q_work = q.copy()

    while i < n and j < m:
        a = float(p_work[i])
        b = float(q_work[j])
        if a <= 0.0:
            i += 1
            continue
        if b <= 0.0:
            j += 1
            continue
        move = a if a < b else b
        G[i, j] += move
        p_work[i] -= move
        q_work[j] -= move
        if p_work[i] <= 1e-15:
            i += 1
        if q_work[j] <= 1e-15:
            j += 1

    # Normalize any small numerical drift
    s = float(G.sum())
    if s > 0:
        G /= s
    return G




def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if not np.isfinite(na) or not np.isfinite(nb) or na <= eps or nb <= eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


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

    # Precompute reference couplings (clean-to-clean) for each device/core
    ref_coupling_by_device: dict[object, dict[int, np.ndarray]] = {}
    for device_id, ref_shapes in ref_by_device.items():
        ref_coupling_by_device[device_id] = {}
        for idx in range(4):
            rs = np.asarray(ref_shapes[f"core_{idx}"], dtype=float)
            ref_coupling_by_device[device_id][idx] = _ot_coupling_1d(rs, rs)

    # Build clean-jitter template deltas from an additional clean run (poison_frac=0.10)
    clean010_rows = summary[
        (summary["poisoning_type"] == "clean") & (summary["poison_frac"] == 0.10)
    ]
    clean010_delta_lists: dict[object, dict[int, list[np.ndarray]]] = {}

    for _, row in clean010_rows.iterrows():
        device_id = row["device_id"]
        ref = ref_by_device.get(device_id)
        if ref is None or device_id not in ref_coupling_by_device:
            continue
        clean010_delta_lists.setdefault(device_id, {i: [] for i in range(4)})
        for idx in range(4):
            shape010 = np.array(json.loads(row[f"core{idx}_shape_mean"]), dtype=float)
            ref_shape = np.array(ref[f"core_{idx}"], dtype=float)
            gs_ref = ref_coupling_by_device[device_id][idx]
            gs010 = _ot_coupling_1d(ref_shape, shape010)
            clean010_delta_lists[device_id][idx].append((gs010 - gs_ref).ravel())

    clean010_delta_by_device: dict[object, dict[int, np.ndarray]] = {}
    for device_id, per_core in clean010_delta_lists.items():
        clean010_delta_by_device[device_id] = {}
        for idx, vecs in per_core.items():
            if vecs:
                clean010_delta_by_device[device_id][idx] = np.mean(np.vstack(vecs), axis=0)

    bin_width = 1.0 / float(bins)
    distances: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        ref = ref_by_device.get(row["device_id"])
        if ref is None:
            continue

        dists: list[float] = []
        cos_sims: list[float] = []

        # Per-core distance + cosine similarity vs template
        for idx in range(4):
            shape = np.array(json.loads(row[f"core{idx}_shape_mean"]), dtype=float)
            ref_shape = np.array(ref[f"core_{idx}"], dtype=float)

            dist = _wasserstein_1d(ref_shape, shape, bin_width=bin_width)
            dists.append(dist)

            # Coupling-delta (type) similarity to the device-specific clean(0.10) jitter template
            device_id = row["device_id"]
            gs_ref = ref_coupling_by_device[device_id][idx]
            gs = _ot_coupling_1d(ref_shape, shape)
            delta = (gs - gs_ref).ravel()
            tmpl = clean010_delta_by_device.get(device_id, {}).get(idx)
            cos_val = _cosine_similarity(delta, tmpl) if tmpl is not None else float("nan")
            cos_sims.append(cos_val)

        finite_cos = [v for v in cos_sims if np.isfinite(v)]
        cos_mean = float(np.mean(finite_cos)) if finite_cos else float("nan")

        distances.append(
            {
                "device_id": row["device_id"],
                "poison_frac": row["poison_frac"],
                "poisoning_type": row["poisoning_type"],
                "core0_ot_distance_to_clean_005": dists[0],
                "core1_ot_distance_to_clean_005": dists[1],
                "core2_ot_distance_to_clean_005": dists[2],
                "core3_ot_distance_to_clean_005": dists[3],
                "core0_type_cosine_to_clean010": cos_sims[0] if len(cos_sims) > 0 else float("nan"),
                "core1_type_cosine_to_clean010": cos_sims[1] if len(cos_sims) > 1 else float("nan"),
                "core2_type_cosine_to_clean010": cos_sims[2] if len(cos_sims) > 2 else float("nan"),
                "core3_type_cosine_to_clean010": cos_sims[3] if len(cos_sims) > 3 else float("nan"),
                "core_type_cosine_mean_to_clean010": cos_mean,
                "core_ot_distance_mean": float(np.mean(dists)),
                "core_ot_distance_mean_to_clean": float(np.mean(dists)),
            }
        )

    dist_df = pd.DataFrame(distances).sort_values(
        ["device_id", "poison_frac", "poisoning_type"]
    )
    return dist_df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--device-ids", type=str, default="", help="comma-separated device_id values to include (e.g., 114,115); empty means all")
    p.add_argument("--out", type=str, default="", help="if set, save the summary table as CSV to this path")
    args = p.parse_args()

    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    summary = cpu_core_sorted_ot(df, bins=int(args.bins))

    # Filter by device_id(s) if requested
    if args.device_ids.strip():
        wanted = {s.strip() for s in args.device_ids.split(",") if s.strip()}
        summary = summary[summary["device_id"].astype(str).isin(wanted)].copy()

    cols = [
        "device_id",
        "poisoning_type",
        "poison_frac",
        "core0_type_cosine_to_clean010",
        "core1_type_cosine_to_clean010",
        "core2_type_cosine_to_clean010",
        "core3_type_cosine_to_clean010",
        "core_type_cosine_mean_to_clean010",
        "core_ot_distance_mean_to_clean",
    ]
    cols = [c for c in cols if c in summary.columns]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)

    if summary.empty:
        print("No rows to display (summary is empty).")
    else:
        print(summary[cols].to_string(index=False))

    if args.out.strip():
        summary.to_csv(args.out, index=False)
        print(f"Saved CSV to: {args.out}")