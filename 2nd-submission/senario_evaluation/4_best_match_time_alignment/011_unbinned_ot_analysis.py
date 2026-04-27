from __future__ import annotations

from pathlib import Path
import argparse
import ast
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from log_loader import load_logs, parse_run_spec


CORE_COLS = ["core_0", "core_1", "core_2", "core_3"]
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
DEFAULT_GAMMA = 0.1
DEFAULT_WINDOW_SIZE = 5
DEFAULT_PAD_MODE = "edge"
DEFAULT_USE_SHAPE_COST = True
DEFAULT_COST_FUNCTION = "time_distance_shape"


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


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if not np.isfinite(na) or not np.isfinite(nb) or na <= eps or nb <= eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _normalize_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= 0.0:
        return x
    return x / s


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")


def _pairwise_squared_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _ensure_2d(a)
    b = _ensure_2d(b)
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def _local_window_features(
    values: np.ndarray,
    window_size: int,
    z_normalize_window: bool,
    pad_mode: str,
) -> np.ndarray:
    x = _ensure_2d(values)
    n_points, n_dim = x.shape
    if n_points == 0:
        width = 2 * max(int(window_size), 0) + 1
        return np.zeros((0, width * n_dim), dtype=float)

    window_size = max(int(window_size), 0)
    window_width = 2 * window_size + 1
    if window_size == 0:
        windows = x[:, None, :]
    else:
        # At endpoints we use explicit edge padding, so every point still gets the same
        # 2 * window_size + 1 local context length instead of silently shrinking.
        padded = np.pad(x, ((window_size, window_size), (0, 0)), mode=pad_mode)
        windows = np.stack([padded[idx : idx + window_width, :] for idx in range(n_points)], axis=0)

    if z_normalize_window:
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        windows = (windows - mean) / std

    return windows.reshape(n_points, window_width * n_dim)


def _shape_features(
    measure: dict[str, Any],
    window_size: int,
    z_normalize_window: bool,
    pad_mode: str,
) -> np.ndarray:
    cache = measure.setdefault("_shape_cache", {})
    cache_key = (int(window_size), bool(z_normalize_window), str(pad_mode))
    if cache_key not in cache:
        cache[cache_key] = _local_window_features(
            measure["x"],
            window_size=int(window_size),
            z_normalize_window=bool(z_normalize_window),
            pad_mode=str(pad_mode),
        )
    return np.asarray(cache[cache_key], dtype=float)


def _delta_features(measure: dict[str, Any]) -> np.ndarray:
    cache = measure.setdefault("_delta_cache", None)
    if cache is None:
        x = _ensure_2d(measure["x"])
        if len(x) == 0:
            delta = np.zeros((0, 1), dtype=float)
        else:
            delta = np.zeros_like(x, dtype=float)
            if len(x) > 1:
                delta[1:] = x[1:] - x[:-1]
        measure["_delta_cache"] = delta
        cache = delta
    return np.asarray(cache, dtype=float)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["epoch", "ts_unix", "cpu_per_core", "mem_percent"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()
    df["epoch"] = epoch_numeric.loc[df.index]
    df = df.loc[df["epoch"] != 9].copy()
    df["ts_unix"] = pd.to_numeric(df["ts_unix"], errors="coerce")
    df["mem_percent"] = pd.to_numeric(df["mem_percent"], errors="coerce")
    df = df.loc[df["ts_unix"].notna() & df["mem_percent"].notna()].copy()

    core_vals = df["cpu_per_core"].apply(_parse_core_list)
    df = df.loc[core_vals.notna()].copy()
    sorted_vals = core_vals.apply(lambda v: sorted(v, reverse=True))
    for idx, col in enumerate(CORE_COLS):
        df[col] = sorted_vals.apply(lambda v: float(v[idx]) if idx < len(v) else float("nan"))
    df = df.loc[df[CORE_COLS].notna().all(axis=1)].copy()
    return df


def _time_width_weights(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    mids = 0.5 * (t[:-1] + t[1:])
    edges = np.concatenate(([0.0], mids, [1.0]))
    widths = np.diff(edges)
    widths = np.maximum(widths, 0.0)
    s = float(widths.sum())
    if s <= 0.0:
        return np.full(n, 1.0 / n, dtype=float)
    return widths / s


def _epoch_measure(group: pd.DataFrame, value_col: str) -> dict[str, np.ndarray]:
    sub = group.loc[group[value_col].notna()].sort_values("ts_unix")
    if sub.empty:
        return {
            "t": np.zeros(0, dtype=float),
            "x": np.zeros(0, dtype=float),
            "w": np.zeros(0, dtype=float),
            "_shape_cache": {},
            "_delta_cache": None,
        }

    ts = sub["ts_unix"].astype(float).to_numpy()
    values = sub[value_col].astype(float).to_numpy()
    finite_mask = np.isfinite(ts) & np.isfinite(values)
    ts = ts[finite_mask]
    values = values[finite_mask]
    if len(ts) == 0:
        return {
            "t": np.zeros(0, dtype=float),
            "x": np.zeros(0, dtype=float),
            "w": np.zeros(0, dtype=float),
            "_shape_cache": {},
            "_delta_cache": None,
        }

    t0 = float(ts.min())
    t1 = float(ts.max())
    if not np.isfinite(t0) or not np.isfinite(t1):
        return {
            "t": np.zeros(0, dtype=float),
            "x": np.zeros(0, dtype=float),
            "w": np.zeros(0, dtype=float),
            "_shape_cache": {},
            "_delta_cache": None,
        }
    if t1 <= t0:
        t_norm = np.full(len(ts), 0.5, dtype=float)
    else:
        t_norm = (ts - t0) / (t1 - t0)
    w = _time_width_weights(t_norm)
    return {
        "t": t_norm.astype(float),
        "x": values.astype(float),
        "w": w.astype(float),
        "_shape_cache": {},
        "_delta_cache": None,
    }


def _aggregate_run_measure(epoch_measures: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    valid = [m for m in epoch_measures if len(m["t"]) > 0]
    if not valid:
        return {
            "t": np.zeros(0, dtype=float),
            "x": np.zeros(0, dtype=float),
            "w": np.zeros(0, dtype=float),
            "_shape_cache": {},
            "_delta_cache": None,
        }

    epoch_scale = 1.0 / len(valid)
    t_all = np.concatenate([m["t"] for m in valid])
    x_all = np.concatenate([m["x"] for m in valid])
    w_all = np.concatenate([m["w"] * epoch_scale for m in valid])
    w_all = _normalize_nonneg(w_all)
    return {"t": t_all, "x": x_all, "w": w_all, "_shape_cache": {}, "_delta_cache": None}


def _cost_matrix(
    ref: dict[str, Any],
    target: dict[str, Any],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    window_size: int = DEFAULT_WINDOW_SIZE,
    use_shape_cost: bool = DEFAULT_USE_SHAPE_COST,
    cost_function: str = DEFAULT_COST_FUNCTION,
    z_normalize_window: bool = False,
    pad_mode: str = DEFAULT_PAD_MODE,
    return_terms: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    use_shape_cost = bool(use_shape_cost)
    cost_function = str(cost_function)
    if cost_function == "time_distance":
        use_shape_cost = False
    elif cost_function == "time_distance_shape":
        use_shape_cost = True
    elif cost_function != "time_distance_delta":
        raise ValueError(f"Unsupported cost function: {cost_function}")

    dt = ref["t"][:, None] - target["t"][None, :]
    time_cost = float(alpha) * (dt * dt)
    value_cost = float(beta) * _pairwise_squared_l2(ref["x"], target["x"])

    if use_shape_cost:
        # This term lowers cost when the local neighborhood also matches, so OT
        # prefers continuous local-shape agreement over isolated point matches.
        ref_phi = _shape_features(
            ref,
            window_size=int(window_size),
            z_normalize_window=bool(z_normalize_window),
            pad_mode=str(pad_mode),
        )
        target_phi = _shape_features(
            target,
            window_size=int(window_size),
            z_normalize_window=bool(z_normalize_window),
            pad_mode=str(pad_mode),
        )
        shape_cost = float(gamma) * _pairwise_squared_l2(ref_phi, target_phi)
    else:
        shape_cost = np.zeros((len(ref["t"]), len(target["t"])), dtype=float)

    if cost_function == "time_distance_delta":
        ref_delta = _delta_features(ref)
        target_delta = _delta_features(target)
        delta_cost = float(gamma) * _pairwise_squared_l2(ref_delta, target_delta)
    else:
        delta_cost = np.zeros((len(ref["t"]), len(target["t"])), dtype=float)

    total_cost = time_cost + value_cost + shape_cost + delta_cost
    if return_terms:
        return total_cost, {
            "time_cost": time_cost,
            "value_cost": value_cost,
            "shape_cost": shape_cost,
            "delta_cost": delta_cost,
            "cost_function": cost_function,
        }
    return total_cost


def _sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    reg: float,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> np.ndarray:
    a = _normalize_nonneg(a)
    b = _normalize_nonneg(b)
    if len(a) == 0 or len(b) == 0 or float(a.sum()) == 0.0 or float(b.sum()) == 0.0:
        return np.zeros((len(a), len(b)), dtype=float)

    reg = max(float(reg), 1e-6)
    scaled = -cost / reg
    scaled = scaled - np.max(scaled)
    K = np.exp(scaled)
    K = np.maximum(K, 1e-300)

    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max_iter):
        u_prev = u.copy()
        Kv = K @ v
        Kv = np.maximum(Kv, 1e-300)
        u = a / Kv
        KTu = K.T @ u
        KTu = np.maximum(KTu, 1e-300)
        v = b / KTu
        if np.max(np.abs(u - u_prev)) <= tol:
            break

    plan = (u[:, None] * K) * v[None, :]
    total = float(plan.sum())
    if total > 0.0:
        plan /= total
    return plan


def _transport_plan(
    ref: dict[str, Any],
    target: dict[str, Any],
    reg_scale: float = 0.05,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    window_size: int = DEFAULT_WINDOW_SIZE,
    use_shape_cost: bool = DEFAULT_USE_SHAPE_COST,
    cost_function: str = DEFAULT_COST_FUNCTION,
    z_normalize_window: bool = False,
    pad_mode: str = DEFAULT_PAD_MODE,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    if len(ref["t"]) == 0 or len(target["t"]) == 0:
        empty = np.zeros((len(ref["t"]), len(target["t"])), dtype=float)
        details = {
            "reg": float("nan"),
            "time_cost_mean": float("nan"),
            "value_cost_mean": float("nan"),
            "shape_cost_mean": float("nan"),
            "delta_cost_mean": float("nan"),
            "shape_cost_enabled": float(bool(use_shape_cost)),
            "cost_function": str(cost_function),
        }
        if return_details:
            return empty, np.zeros_like(empty), float("nan"), details
        return empty, np.zeros_like(empty), float("nan")

    cost, terms = _cost_matrix(
        ref,
        target,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        window_size=int(window_size),
        use_shape_cost=bool(use_shape_cost),
        cost_function=str(cost_function),
        z_normalize_window=bool(z_normalize_window),
        pad_mode=str(pad_mode),
        return_terms=True,
    )
    positive_cost = cost[cost > 0]
    if positive_cost.size:
        reg = max(float(np.median(positive_cost)) * reg_scale, 1e-3)
    else:
        reg = 1e-3
    plan = _sinkhorn(ref["w"], target["w"], cost=cost, reg=reg)
    ot_distance = float(np.sum(plan * cost)) if plan.size else float("nan")
    if not return_details:
        return plan, cost, ot_distance

    details = {
        "reg": float(reg),
        "time_cost_mean": float(np.sum(plan * terms["time_cost"])) if plan.size else float("nan"),
        "value_cost_mean": float(np.sum(plan * terms["value_cost"])) if plan.size else float("nan"),
        "shape_cost_mean": float(np.sum(plan * terms["shape_cost"])) if plan.size else float("nan"),
        "delta_cost_mean": float(np.sum(plan * terms["delta_cost"])) if plan.size else float("nan"),
        "shape_cost_enabled": float(bool(use_shape_cost and cost_function == "time_distance_shape")),
        "cost_function": str(cost_function),
    }
    return plan, cost, ot_distance, details


def _resample_matrix(mat: np.ndarray, out_rows: int = 32, out_cols: int = 32) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0:
        return np.zeros((out_rows, out_cols), dtype=float)
    if mat.shape == (out_rows, out_cols):
        return mat

    row_src = np.linspace(0.0, 1.0, num=mat.shape[0])
    row_dst = np.linspace(0.0, 1.0, num=out_rows)
    tmp = np.vstack([np.interp(row_dst, row_src, mat[:, j]) for j in range(mat.shape[1])]).T
    col_src = np.linspace(0.0, 1.0, num=mat.shape[1])
    col_dst = np.linspace(0.0, 1.0, num=out_cols)
    out = np.vstack([np.interp(col_dst, col_src, tmp[i, :]) for i in range(tmp.shape[0])])
    return out


def _plan_signature(plan: np.ndarray) -> np.ndarray:
    return _resample_matrix(plan, out_rows=32, out_cols=32).ravel()


def build_summary(
    df: pd.DataFrame,
    reference_group: str,
    reference_label: str,
    template_group: str,
    template_label: str,
    target_keys: set[tuple[str, str]] | None = None,
    bins: int = 0,
    reg_scale: float = 0.05,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    window_size: int = DEFAULT_WINDOW_SIZE,
    use_shape_cost: bool = DEFAULT_USE_SHAPE_COST,
    cost_function: str = DEFAULT_COST_FUNCTION,
    z_normalize_window: bool = False,
    pad_mode: str = DEFAULT_PAD_MODE,
) -> pd.DataFrame:
    del bins
    df = _prepare_df(df)

    per_epoch_rows: list[dict[str, object]] = []
    per_run_measures: dict[tuple[str, str, str], dict[str, dict[int, dict[str, np.ndarray]]]] = {}

    for key, group in df.groupby(["source_group", "source_label", "run_csv", "epoch"], dropna=False):
        source_group, source_label, run_csv, epoch = key
        meta = group.iloc[0]
        row: dict[str, object] = {
            "source_group": source_group,
            "source_label": source_label,
            "run_csv": run_csv,
            "epoch": epoch,
            "device_id": meta["device_id"],
            "poisoning_type": meta["poisoning_type"],
            "poison_frac": meta["poison_frac"],
            "ood_frac": meta.get("ood_frac", np.nan),
        }
        per_run_measures.setdefault(key[:3], {"mem": {}, **{c: {} for c in CORE_COLS}})
        mem_measure = _epoch_measure(group, "mem_percent")
        per_run_measures[key[:3]]["mem"][int(epoch)] = mem_measure
        row["mem_num_points"] = len(mem_measure["t"])
        row["mem_mass_sum"] = float(mem_measure["w"].sum()) if len(mem_measure["w"]) else 0.0
        for col in CORE_COLS:
            measure = _epoch_measure(group, col)
            per_run_measures[key[:3]][col][int(epoch)] = measure
            row[f"{col}_num_points"] = len(measure["t"])
            row[f"{col}_mass_sum"] = float(measure["w"].sum()) if len(measure["w"]) else 0.0
        per_epoch_rows.append(row)

    per_run_rows: list[dict[str, object]] = []
    for (source_group, source_label, run_csv), measures in per_run_measures.items():
        first = next(
            row
            for row in per_epoch_rows
            if row["source_group"] == source_group
            and row["source_label"] == source_label
            and row["run_csv"] == run_csv
        )
        row = {
            "source_group": source_group,
            "source_label": source_label,
            "run_csv": run_csv,
            "device_id": first["device_id"],
            "poisoning_type": first["poisoning_type"],
            "poison_frac": first["poison_frac"],
            "ood_frac": first["ood_frac"],
        }
        for signal_name in ["mem", *CORE_COLS]:
            measure = _aggregate_run_measure([measures[signal_name][epoch] for epoch in sorted(measures[signal_name])])
            row[f"{signal_name}_num_points"] = len(measure["t"])
            row[f"{signal_name}_mass_sum"] = float(measure["w"].sum()) if len(measure["w"]) else 0.0
        per_run_rows.append(row)

    summary = pd.DataFrame(per_run_rows)
    summary_per_epoch = pd.DataFrame(per_epoch_rows)

    ref_key = (reference_group, reference_label)
    ref_rows = summary[
        (summary["source_group"] == reference_group)
        & (summary["source_label"] == reference_label)
    ]
    if ref_rows.empty:
        raise ValueError(f"Reference run not found: {reference_group}:{reference_label}")
    if len(ref_rows) > 1:
        raise ValueError(f"Reference run is ambiguous: {reference_group}:{reference_label}")
    ref_row = ref_rows.iloc[0]
    ref_run_key = (reference_group, reference_label, str(ref_row["run_csv"]))
    ref_measures = {
        "mem": _aggregate_run_measure(
            [per_run_measures[ref_run_key]["mem"][epoch] for epoch in sorted(per_run_measures[ref_run_key]["mem"])]
        ),
        **{
            col: _aggregate_run_measure(
                [per_run_measures[ref_run_key][col][epoch] for epoch in sorted(per_run_measures[ref_run_key][col])]
            )
            for col in CORE_COLS
        },
    }

    template_rows = summary[
        (summary["source_group"] == template_group)
        & (summary["source_label"] == template_label)
    ]
    if template_rows.empty:
        raise ValueError(f"Template run not found: {template_group}:{template_label}")
    if len(template_rows) > 1:
        raise ValueError(f"Template run is ambiguous: {template_group}:{template_label}")
    template_row = template_rows.iloc[0]
    template_run_key = (template_group, template_label, str(template_row["run_csv"]))
    # The template run is still validated for interface compatibility, but this summary path
    # now focuses only on distance-based OT terms and does not compute similarity signatures.
    _ = template_run_key

    rows: list[dict[str, Any]] = []
    for _, row in summary_per_epoch.iterrows():
        row_key = (str(row["source_group"]), str(row["source_label"]))
        if row_key == ref_key:
            continue
        if target_keys is not None and row_key not in target_keys:
            continue

        run_key = (str(row["source_group"]), str(row["source_label"]), str(row["run_csv"]))
        epoch_idx = int(row["epoch"])
        dists: list[float] = []
        core_time_cost_means: list[float] = []
        core_value_cost_means: list[float] = []
        core_shape_cost_means: list[float] = []
        core_delta_cost_means: list[float] = []
        signal_term_details: dict[str, dict[str, float]] = {}

        for core_idx, col in enumerate(CORE_COLS):
            ref_measure = ref_measures[col]
            target_measure = per_run_measures[run_key][col][epoch_idx]
            plan, _, dist, detail = _transport_plan(
                ref_measure,
                target_measure,
                reg_scale=reg_scale,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                window_size=window_size,
                use_shape_cost=use_shape_cost,
                cost_function=cost_function,
                z_normalize_window=z_normalize_window,
                pad_mode=pad_mode,
                return_details=True,
            )
            dists.append(dist)
            core_time_cost_means.append(detail["time_cost_mean"])
            core_value_cost_means.append(detail["value_cost_mean"])
            core_shape_cost_means.append(detail["shape_cost_mean"])
            core_delta_cost_means.append(detail["delta_cost_mean"])
            signal_term_details[col] = detail

        ref_mem = ref_measures["mem"]
        target_mem = per_run_measures[run_key]["mem"][epoch_idx]
        mem_plan, _, mem_dist, mem_detail = _transport_plan(
            ref_mem,
            target_mem,
            reg_scale=reg_scale,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            window_size=window_size,
            use_shape_cost=use_shape_cost,
            cost_function=cost_function,
            z_normalize_window=z_normalize_window,
            pad_mode=pad_mode,
            return_details=True,
        )
        signal_term_details["mem"] = mem_detail

        rows.append(
            {
                "device_id": row["device_id"],
                "reference_group": reference_group,
                "reference_label": reference_label,
                "target_group": row["source_group"],
                "target_label": row["source_label"],
                "poisoning_type": row["poisoning_type"],
                "poison_frac": row["poison_frac"],
                "ood_frac": row["ood_frac"],
                "epoch": row["epoch"],
                "reference_run_csv": ref_row["run_csv"],
                "target_run_csv": row["run_csv"],
                "mem_ot_distance_to_baseclean": mem_dist,
                "mem_time_cost_mean_to_baseclean": mem_detail["time_cost_mean"],
                "mem_value_cost_mean_to_baseclean": mem_detail["value_cost_mean"],
                "mem_shape_cost_mean_to_baseclean": mem_detail["shape_cost_mean"],
                "mem_delta_cost_mean_to_baseclean": mem_detail["delta_cost_mean"],
                "core0_ot_distance_to_baseclean": dists[0],
                "core1_ot_distance_to_baseclean": dists[1],
                "core2_ot_distance_to_baseclean": dists[2],
                "core3_ot_distance_to_baseclean": dists[3],
                "core0_time_cost_mean_to_baseclean": signal_term_details["core_0"]["time_cost_mean"],
                "core1_time_cost_mean_to_baseclean": signal_term_details["core_1"]["time_cost_mean"],
                "core2_time_cost_mean_to_baseclean": signal_term_details["core_2"]["time_cost_mean"],
                "core3_time_cost_mean_to_baseclean": signal_term_details["core_3"]["time_cost_mean"],
                "core0_value_cost_mean_to_baseclean": signal_term_details["core_0"]["value_cost_mean"],
                "core1_value_cost_mean_to_baseclean": signal_term_details["core_1"]["value_cost_mean"],
                "core2_value_cost_mean_to_baseclean": signal_term_details["core_2"]["value_cost_mean"],
                "core3_value_cost_mean_to_baseclean": signal_term_details["core_3"]["value_cost_mean"],
                "core0_shape_cost_mean_to_baseclean": signal_term_details["core_0"]["shape_cost_mean"],
                "core1_shape_cost_mean_to_baseclean": signal_term_details["core_1"]["shape_cost_mean"],
                "core2_shape_cost_mean_to_baseclean": signal_term_details["core_2"]["shape_cost_mean"],
                "core3_shape_cost_mean_to_baseclean": signal_term_details["core_3"]["shape_cost_mean"],
                "core0_delta_cost_mean_to_baseclean": signal_term_details["core_0"]["delta_cost_mean"],
                "core1_delta_cost_mean_to_baseclean": signal_term_details["core_1"]["delta_cost_mean"],
                "core2_delta_cost_mean_to_baseclean": signal_term_details["core_2"]["delta_cost_mean"],
                "core3_delta_cost_mean_to_baseclean": signal_term_details["core_3"]["delta_cost_mean"],
                "core_ot_distance_mean_to_baseclean": float(np.mean(dists)),
                "core_time_cost_mean_to_baseclean": float(np.mean(core_time_cost_means)),
                "core_value_cost_mean_to_baseclean": float(np.mean(core_value_cost_means)),
                "core_shape_cost_mean_to_baseclean": float(np.mean(core_shape_cost_means)),
                "core_delta_cost_mean_to_baseclean": float(np.mean(core_delta_cost_means)),
                "ot_distance": 0.8 * float(np.mean(dists)) + 0.2 * mem_dist,
                "cost_alpha": float(alpha),
                "cost_beta": float(beta),
                "cost_gamma": float(gamma),
                "cost_window_size": int(window_size),
                "cost_use_shape": bool(use_shape_cost),
                "cost_function": str(cost_function),
                "cost_z_normalize_window": bool(z_normalize_window),
                "cost_reg_scale": float(reg_scale),
            }
        )

    return pd.DataFrame(rows).sort_values(["target_group", "target_label", "epoch"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="reference spec: group:label=/path/to.csv")
    p.add_argument("--template", required=True, help="template spec: group:label=/path/to.csv")
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="target spec: group:label=/path/to.csv; may be passed multiple times",
    )
    p.add_argument("--device-id", type=str, default="scenario_device")
    p.add_argument(
        "--bins",
        type=int,
        default=0,
        help="unused in this unbinned OT variant; kept for CLI compatibility",
    )
    p.add_argument("--out", type=str, default="")
    p.add_argument("--reg-scale", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    p.add_argument(
        "--cost-function",
        type=str,
        default=DEFAULT_COST_FUNCTION,
        choices=["time_distance", "time_distance_shape", "time_distance_delta"],
        help="select the OT cost: time+value, time+value+shape, or time+value+delta",
    )
    p.add_argument(
        "--use-shape-cost",
        dest="use_shape_cost",
        action="store_true",
        default=DEFAULT_USE_SHAPE_COST,
        help="add a local window-shape term so OT prefers local shape agreement over isolated point matches",
    )
    p.add_argument(
        "--no-shape-cost",
        dest="use_shape_cost",
        action="store_false",
        help="disable the local window-shape term and use only time and point-value costs",
    )
    p.add_argument(
        "--z-normalize-window",
        action="store_true",
        help="z-normalize each local window before shape comparison",
    )
    p.add_argument(
        "--pad-mode",
        type=str,
        default=DEFAULT_PAD_MODE,
        help="np.pad mode used for local windows at the endpoints",
    )
    args = p.parse_args()

    if not args.run:
        raise ValueError("At least one --run must be provided.")

    reference_group, reference_label, _ = parse_run_spec(args.reference)
    template_group, template_label, _ = parse_run_spec(args.template)
    parsed_runs = [parse_run_spec(spec) for spec in args.run]
    specs = [args.template, *args.run]
    target_keys = {(group, label) for group, label, _ in parsed_runs}

    df = load_logs(args.reference, specs, device_id=args.device_id)
    summary = build_summary(
        df=df,
        reference_group=reference_group,
        reference_label=reference_label,
        template_group=template_group,
        template_label=template_label,
        target_keys=target_keys,
        bins=int(args.bins),
        reg_scale=float(args.reg_scale),
        alpha=float(args.alpha),
        beta=float(args.beta),
        gamma=float(args.gamma),
        window_size=int(args.window_size),
        use_shape_cost=bool(args.use_shape_cost),
        cost_function=str(args.cost_function),
        z_normalize_window=bool(args.z_normalize_window),
        pad_mode=str(args.pad_mode),
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    if summary.empty:
        print("No rows to display (summary is empty).")
    else:
        print(summary.to_string(index=False))

    if args.out.strip():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"Saved CSV to: {args.out}")


if __name__ == "__main__":
    main()
