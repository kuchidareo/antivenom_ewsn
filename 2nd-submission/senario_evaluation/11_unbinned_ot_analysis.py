from __future__ import annotations

from pathlib import Path
import argparse
import ast
from typing import Any

import numpy as np
import pandas as pd

from log_loader import load_logs, parse_run_spec


CORE_COLS = ["core_0", "core_1", "core_2", "core_3"]


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
        return {"t": np.zeros(0, dtype=float), "x": np.zeros(0, dtype=float), "w": np.zeros(0, dtype=float)}

    ts = sub["ts_unix"].astype(float).to_numpy()
    values = sub[value_col].astype(float).to_numpy()
    finite_mask = np.isfinite(ts) & np.isfinite(values)
    ts = ts[finite_mask]
    values = values[finite_mask]
    if len(ts) == 0:
        return {"t": np.zeros(0, dtype=float), "x": np.zeros(0, dtype=float), "w": np.zeros(0, dtype=float)}

    t0 = float(ts.min())
    t1 = float(ts.max())
    if not np.isfinite(t0) or not np.isfinite(t1):
        return {"t": np.zeros(0, dtype=float), "x": np.zeros(0, dtype=float), "w": np.zeros(0, dtype=float)}
    if t1 <= t0:
        t_norm = np.full(len(ts), 0.5, dtype=float)
    else:
        t_norm = (ts - t0) / (t1 - t0)
    w = _time_width_weights(t_norm)
    return {"t": t_norm.astype(float), "x": values.astype(float), "w": w.astype(float)}


def _aggregate_run_measure(epoch_measures: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    valid = [m for m in epoch_measures if len(m["t"]) > 0]
    if not valid:
        return {"t": np.zeros(0, dtype=float), "x": np.zeros(0, dtype=float), "w": np.zeros(0, dtype=float)}

    epoch_scale = 1.0 / len(valid)
    t_all = np.concatenate([m["t"] for m in valid])
    x_all = np.concatenate([m["x"] for m in valid])
    w_all = np.concatenate([m["w"] * epoch_scale for m in valid])
    w_all = _normalize_nonneg(w_all)
    return {"t": t_all, "x": x_all, "w": w_all}


def _cost_matrix(ref: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> np.ndarray:
    dt = ref["t"][:, None] - target["t"][None, :]
    dx = ref["x"][:, None] - target["x"][None, :]
    return dt * dt + dx * dx


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
    ref: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    reg_scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(ref["t"]) == 0 or len(target["t"]) == 0:
        empty = np.zeros((len(ref["t"]), len(target["t"])), dtype=float)
        return empty, np.zeros_like(empty), float("nan")

    cost = _cost_matrix(ref, target)
    positive_cost = cost[cost > 0]
    if positive_cost.size:
        reg = max(float(np.median(positive_cost)) * reg_scale, 1e-3)
    else:
        reg = 1e-3
    plan = _sinkhorn(ref["w"], target["w"], cost=cost, reg=reg)
    ot_distance = float(np.sum(plan * cost)) if plan.size else float("nan")
    return plan, cost, ot_distance


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
    template_measures = {
        "mem": _aggregate_run_measure(
            [per_run_measures[template_run_key]["mem"][epoch] for epoch in sorted(per_run_measures[template_run_key]["mem"])]
        ),
        **{
            col: _aggregate_run_measure(
                [per_run_measures[template_run_key][col][epoch] for epoch in sorted(per_run_measures[template_run_key][col])]
            )
            for col in CORE_COLS
        },
    }

    ref_plan_signature_by_signal: dict[str, np.ndarray] = {}
    template_delta_by_signal: dict[str, np.ndarray] = {}
    for signal_name in ["mem", *CORE_COLS]:
        ref_measure = ref_measures[signal_name]
        ref_plan, _, _ = _transport_plan(ref_measure, ref_measure)
        ref_sig = _plan_signature(ref_plan)
        ref_plan_signature_by_signal[signal_name] = ref_sig
        template_plan, _, _ = _transport_plan(ref_measure, template_measures[signal_name])
        template_sig = _plan_signature(template_plan)
        template_delta_by_signal[signal_name] = template_sig - ref_sig

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
        cos_sims: list[float] = []

        for col in CORE_COLS:
            ref_measure = ref_measures[col]
            target_measure = per_run_measures[run_key][col][epoch_idx]
            plan, _, dist = _transport_plan(ref_measure, target_measure)
            dists.append(dist)
            delta = _plan_signature(plan) - ref_plan_signature_by_signal[col]
            cos_sims.append(_cosine_similarity(delta, template_delta_by_signal[col]))

        ref_mem = ref_measures["mem"]
        target_mem = per_run_measures[run_key]["mem"][epoch_idx]
        mem_plan, _, mem_dist = _transport_plan(ref_mem, target_mem)
        mem_delta = _plan_signature(mem_plan) - ref_plan_signature_by_signal["mem"]
        mem_cos = _cosine_similarity(mem_delta, template_delta_by_signal["mem"])
        finite_cos = [v for v in cos_sims if np.isfinite(v)]
        core_cos_mean = float(np.mean(finite_cos)) if finite_cos else float("nan")

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
                "core0_ot_distance_to_baseclean": dists[0],
                "core1_ot_distance_to_baseclean": dists[1],
                "core2_ot_distance_to_baseclean": dists[2],
                "core3_ot_distance_to_baseclean": dists[3],
                "core_ot_distance_mean_to_baseclean": float(np.mean(dists)),
                "mem_type_cosine_to_clean": mem_cos,
                "core0_type_cosine_to_clean": cos_sims[0],
                "core1_type_cosine_to_clean": cos_sims[1],
                "core2_type_cosine_to_clean": cos_sims[2],
                "core3_type_cosine_to_clean": cos_sims[3],
                "core_type_cosine_mean_to_clean": core_cos_mean,
                "ot_distance": 0.8 * float(np.mean(dists)) + 0.2 * mem_dist,
                "cosine_similarity": 0.8 * core_cos_mean + 0.2 * mem_cos
                if np.isfinite(core_cos_mean) and np.isfinite(mem_cos)
                else float("nan"),
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
