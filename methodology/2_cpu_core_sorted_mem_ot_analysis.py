from __future__ import annotations

from pathlib import Path
import ast
import json
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


def _wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.mean(np.abs(cdf_p - cdf_q)))


def _normalize_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= 0:
        return x
    return x / s


def _ot_coupling_1d(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the 1D optimal transport coupling between discrete histograms p and q."""
    p = _normalize_nonneg(p)
    q = _normalize_nonneg(q)
    if float(p.sum()) == 0.0 or float(q.sum()) == 0.0:
        return np.zeros((len(p), len(q)), dtype=float)

    n = len(p)
    m = len(q)
    G = np.zeros((n, m), dtype=float)
    i = 0
    j = 0

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


def cpu_core_sorted_mem_ot(
    df: pd.DataFrame,
    bins: int = 50,
    per_epoch: bool = False,
    ref_clean_frac: float = 0.05,
    template_clean_frac: float = 0.10,
) -> pd.DataFrame:
    required = ["epoch", "ts_unix", "cpu_per_core", "mem_percent"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Training rows only (epoch numeric)
    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()
    df["epoch"] = epoch_numeric.loc[df.index]
    df = df.loc[df["epoch"] != 9].copy()

    df["mem_percent"] = pd.to_numeric(df["mem_percent"], errors="coerce")
    df = df.loc[df["mem_percent"].notna()].copy()

    core_cols = ["core_0", "core_1", "core_2", "core_3"]
    core_vals = df["cpu_per_core"].apply(_parse_core_list)
    df = df.loc[core_vals.notna()].copy()
    sorted_vals = core_vals.apply(lambda v: sorted(v, reverse=True))

    for i, col in enumerate(core_cols):
        df[col] = sorted_vals.apply(lambda v: float(v[i]) if i < len(v) else float("nan"))

    df = df.loc[df[core_cols].notna().all(axis=1)].copy()

    per_epoch_rows: list[dict[str, object]] = []
    per_device_shapes: dict[tuple, dict[str, list[np.ndarray]]] = {}
    for key, group in df.groupby(
        ["device_id", "poisoning_type", "poison_frac", "epoch"], dropna=False
    ):
        mem_shape = _epoch_shape_mean(group, "mem_percent", bins=bins)
        per_epoch_rows.append(
            {
                "device_id": key[0],
                "poisoning_type": key[1],
                "poison_frac": key[2],
                "epoch": key[3],
                "mem_shape_mean": json.dumps(mem_shape.tolist()),
            }
        )
        per_device_shapes.setdefault(
            key[:3],
            {"mem": [], **{c: [] for c in core_cols}},
        )
        per_device_shapes[key[:3]]["mem"].append(mem_shape)
        for col in core_cols:
            shape = _epoch_shape_mean(group, col, bins=bins)
            per_device_shapes[key[:3]][col].append(shape)
            per_epoch_rows[-1][f"{col.replace('_', '')}_shape_mean"] = json.dumps(
                shape.tolist()
            )

    rows: list[dict[str, object]] = []
    for (device_id, poisoning_type, poison_frac), shapes in per_device_shapes.items():
        mean_shapes = {}
        mem_arr = np.vstack(shapes["mem"]) if shapes["mem"] else np.zeros((0, bins))
        mean_shapes["mem"] = mem_arr.mean(axis=0) if len(mem_arr) else np.zeros(bins)
        for col in core_cols:
            arr = np.vstack(shapes[col]) if shapes[col] else np.zeros((0, bins))
            mean_shapes[col] = arr.mean(axis=0) if len(arr) else np.zeros(bins)
        rows.append(
            {
                "device_id": device_id,
                "poisoning_type": poisoning_type,
                "poison_frac": poison_frac,
                "mem_shape_mean": json.dumps(mean_shapes["mem"].tolist()),
                "core0_shape_mean": json.dumps(mean_shapes["core_0"].tolist()),
                "core1_shape_mean": json.dumps(mean_shapes["core_1"].tolist()),
                "core2_shape_mean": json.dumps(mean_shapes["core_2"].tolist()),
                "core3_shape_mean": json.dumps(mean_shapes["core_3"].tolist()),
            }
        )

    summary = pd.DataFrame(rows)
    summary_per_epoch = pd.DataFrame(per_epoch_rows)

    # Reference clean at poison_frac=ref_clean_frac per device.
    ref_rows = summary[
        (summary["poisoning_type"] == "clean")
        & (summary["poison_frac"] == ref_clean_frac)
    ]
    ref_by_device = {
        row["device_id"]: {
            "mem": np.array(json.loads(row["mem_shape_mean"])),
            "core_0": np.array(json.loads(row["core0_shape_mean"])),
            "core_1": np.array(json.loads(row["core1_shape_mean"])),
            "core_2": np.array(json.loads(row["core2_shape_mean"])),
            "core_3": np.array(json.loads(row["core3_shape_mean"])),
        }
        for _, row in ref_rows.iterrows()
    }

    ref_coupling_by_device: dict[object, dict[int, np.ndarray]] = {}
    for device_id, ref_shapes in ref_by_device.items():
        ref_coupling_by_device[device_id] = {}
        for idx in range(4):
            rs = np.asarray(ref_shapes[f"core_{idx}"], dtype=float)
            ref_coupling_by_device[device_id][idx] = _ot_coupling_1d(rs, rs)

    ref_coupling_mem_by_device: dict[object, np.ndarray] = {}
    for device_id, ref_shapes in ref_by_device.items():
        rs = np.asarray(ref_shapes["mem"], dtype=float)
        ref_coupling_mem_by_device[device_id] = _ot_coupling_1d(rs, rs)

    clean010_rows = summary[
        (summary["poisoning_type"] == "clean")
        & (summary["poison_frac"] == template_clean_frac)
    ]
    if clean010_rows.empty and ref_clean_frac == template_clean_frac:
        clean010_rows = summary[
            (summary["poisoning_type"] == "clean")
            & (summary["poison_frac"] == ref_clean_frac)
        ]
    elif clean010_rows.empty:
        clean010_rows = summary[
            (summary["poisoning_type"] == "clean")
            & (summary["poison_frac"] == ref_clean_frac)
        ]
    clean010_delta_lists: dict[object, dict[int, list[np.ndarray]]] = {}
    clean010_delta_mem_lists: dict[object, list[np.ndarray]] = {}

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

        mem010 = np.array(json.loads(row["mem_shape_mean"]), dtype=float)
        ref_mem = np.array(ref["mem"], dtype=float)
        gs_ref_mem = ref_coupling_mem_by_device.get(device_id)
        if gs_ref_mem is not None:
            gs010_mem = _ot_coupling_1d(ref_mem, mem010)
            clean010_delta_mem_lists.setdefault(device_id, []).append(
                (gs010_mem - gs_ref_mem).ravel()
            )

    clean010_delta_by_device: dict[object, dict[int, np.ndarray]] = {}
    for device_id, per_core in clean010_delta_lists.items():
        clean010_delta_by_device[device_id] = {}
        for idx, vecs in per_core.items():
            if vecs:
                clean010_delta_by_device[device_id][idx] = np.mean(np.vstack(vecs), axis=0)

    clean010_delta_mem_by_device: dict[object, np.ndarray] = {}
    for device_id, vecs in clean010_delta_mem_lists.items():
        if vecs:
            clean010_delta_mem_by_device[device_id] = np.mean(np.vstack(vecs), axis=0)

    distances: list[dict[str, Any]] = []
    source_df = summary_per_epoch if per_epoch else summary
    for _, row in source_df.iterrows():
        ref = ref_by_device.get(row["device_id"])
        if ref is None:
            continue
        dists: list[float] = []
        cos_sims: list[float] = []
        for idx in range(4):
            shape = np.array(json.loads(row[f"core{idx}_shape_mean"]))
            dist = _wasserstein_1d(ref[f"core_{idx}"], shape)
            dists.append(dist)
            device_id = row["device_id"]
            ref_shape = np.array(ref[f"core_{idx}"], dtype=float)
            gs_ref = ref_coupling_by_device[device_id][idx]
            gs = _ot_coupling_1d(ref_shape, shape)
            delta = (gs - gs_ref).ravel()
            tmpl = clean010_delta_by_device.get(device_id, {}).get(idx)
            cos_val = _cosine_similarity(delta, tmpl) if tmpl is not None else float("nan")
            cos_sims.append(cos_val)
        mem_shape = np.array(json.loads(row["mem_shape_mean"]))
        mem_dist = _wasserstein_1d(ref["mem"], mem_shape)
        device_id = row["device_id"]
        ref_mem = np.array(ref["mem"], dtype=float)
        gs_ref_mem = ref_coupling_mem_by_device.get(device_id)
        mem_cos = float("nan")
        if gs_ref_mem is not None:
            gs_mem = _ot_coupling_1d(ref_mem, mem_shape)
            mem_delta = (gs_mem - gs_ref_mem).ravel()
            mem_tmpl = clean010_delta_mem_by_device.get(device_id)
            mem_cos = _cosine_similarity(mem_delta, mem_tmpl) if mem_tmpl is not None else float("nan")
        finite_cos = [v for v in cos_sims if np.isfinite(v)]
        cos_mean = float(np.mean(finite_cos)) if finite_cos else float("nan")
        out = {
            "device_id": row["device_id"],
            "poison_frac": row["poison_frac"],
            "poisoning_type": row["poisoning_type"],
            "mem_ot_distance_to_clean_005": mem_dist,
            "mem_type_cosine_to_clean010": mem_cos,
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
        }
        if per_epoch and "epoch" in row:
            out["epoch"] = row["epoch"]
        distances.append(out)

    dist_df = pd.DataFrame(distances)
    if dist_df.empty:
        return dist_df
    return dist_df.sort_values(
        ["device_id", "poison_frac", "poisoning_type"] + (["epoch"] if per_epoch else [])
    )


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    summary = cpu_core_sorted_mem_ot(df, bins=50)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    cols = [
        "device_id",
        "poisoning_type",
        "poison_frac",
        "core0_type_cosine_to_clean010",
        "core1_type_cosine_to_clean010",
        "core2_type_cosine_to_clean010",
        "core3_type_cosine_to_clean010",
        "mem_type_cosine_to_clean010",
        "core_type_cosine_mean_to_clean010",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
