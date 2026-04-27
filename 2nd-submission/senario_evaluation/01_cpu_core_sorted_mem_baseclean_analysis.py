from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json
from typing import Any

import numpy as np
import pandas as pd

from log_loader import load_logs, parse_run_spec


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


def build_summary(
    df: pd.DataFrame,
    reference_group: str,
    reference_label: str,
    template_group: str,
    template_label: str,
    target_keys: set[tuple[str, str]] | None = None,
    bins: int = 50,
) -> pd.DataFrame:
    required = ["epoch", "ts_unix", "cpu_per_core", "mem_percent"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

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
    per_run_shapes: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = {}

    for key, group in df.groupby(["source_group", "source_label", "run_csv", "epoch"], dropna=False):
        source_group, source_label, run_csv, epoch = key
        meta = group.iloc[0]
        mem_shape = _epoch_shape_mean(group, "mem_percent", bins=bins)
        row = {
            "source_group": source_group,
            "source_label": source_label,
            "run_csv": run_csv,
            "epoch": epoch,
            "device_id": meta["device_id"],
            "poisoning_type": meta["poisoning_type"],
            "poison_frac": meta["poison_frac"],
            "ood_frac": meta.get("ood_frac", np.nan),
            "mem_shape_mean": json.dumps(mem_shape.tolist()),
        }
        per_run_shapes.setdefault(key[:3], {"mem": [], **{c: [] for c in core_cols}})
        per_run_shapes[key[:3]]["mem"].append(mem_shape)
        for col in core_cols:
            shape = _epoch_shape_mean(group, col, bins=bins)
            per_run_shapes[key[:3]][col].append(shape)
            row[f"{col.replace('_', '')}_shape_mean"] = json.dumps(shape.tolist())
        per_epoch_rows.append(row)

    per_run_rows: list[dict[str, object]] = []
    for (source_group, source_label, run_csv), shapes in per_run_shapes.items():
        first = next(
            row
            for row in per_epoch_rows
            if row["source_group"] == source_group
            and row["source_label"] == source_label
            and row["run_csv"] == run_csv
        )
        mem_arr = np.vstack(shapes["mem"]) if shapes["mem"] else np.zeros((0, bins))
        row = {
            "source_group": source_group,
            "source_label": source_label,
            "run_csv": run_csv,
            "device_id": first["device_id"],
            "poisoning_type": first["poisoning_type"],
            "poison_frac": first["poison_frac"],
            "ood_frac": first["ood_frac"],
            "mem_shape_mean": json.dumps(
                (mem_arr.mean(axis=0) if len(mem_arr) else np.zeros(bins)).tolist()
            ),
        }
        for col in core_cols:
            arr = np.vstack(shapes[col]) if shapes[col] else np.zeros((0, bins))
            row[f"{col.replace('_', '')}_shape_mean"] = json.dumps(
                (arr.mean(axis=0) if len(arr) else np.zeros(bins)).tolist()
            )
        per_run_rows.append(row)

    summary = pd.DataFrame(per_run_rows)
    summary_per_epoch = pd.DataFrame(per_epoch_rows)

    ref_rows = summary[
        (summary["source_group"] == reference_group)
        & (summary["source_label"] == reference_label)
    ]
    if ref_rows.empty:
        raise ValueError(f"Reference run not found: {reference_group}:{reference_label}")
    if len(ref_rows) > 1:
        raise ValueError(f"Reference run is ambiguous: {reference_group}:{reference_label}")
    ref_row = ref_rows.iloc[0]
    ref = {
        "mem": np.array(json.loads(ref_row["mem_shape_mean"])),
        "core_0": np.array(json.loads(ref_row["core0_shape_mean"])),
        "core_1": np.array(json.loads(ref_row["core1_shape_mean"])),
        "core_2": np.array(json.loads(ref_row["core2_shape_mean"])),
        "core_3": np.array(json.loads(ref_row["core3_shape_mean"])),
        "run_csv": ref_row["run_csv"],
    }

    ref_coupling = {idx: _ot_coupling_1d(ref[f"core_{idx}"], ref[f"core_{idx}"]) for idx in range(4)}
    ref_mem_coupling = _ot_coupling_1d(ref["mem"], ref["mem"])

    template_rows = summary[
        (summary["source_group"] == template_group)
        & (summary["source_label"] == template_label)
    ]
    if template_rows.empty:
        raise ValueError(f"Template run not found: {template_group}:{template_label}")
    if len(template_rows) > 1:
        raise ValueError(f"Template run is ambiguous: {template_group}:{template_label}")
    template_row = template_rows.iloc[0]

    template_delta_by_core: dict[int, np.ndarray] = {}
    for idx in range(4):
        template_shape = np.array(json.loads(template_row[f"core{idx}_shape_mean"]), dtype=float)
        ref_shape = np.asarray(ref[f"core_{idx}"], dtype=float)
        template_coupling = _ot_coupling_1d(ref_shape, template_shape)
        template_delta_by_core[idx] = (template_coupling - ref_coupling[idx]).ravel()

    template_mem_shape = np.array(json.loads(template_row["mem_shape_mean"]), dtype=float)
    template_mem_coupling = _ot_coupling_1d(np.asarray(ref["mem"], dtype=float), template_mem_shape)
    template_delta_mem = (template_mem_coupling - ref_mem_coupling).ravel()

    rows: list[dict[str, Any]] = []
    for _, row in summary_per_epoch.iterrows():
        row_key = (str(row["source_group"]), str(row["source_label"]))
        if row["source_group"] == reference_group and row["source_label"] == reference_label:
            continue
        if target_keys is not None and row_key not in target_keys:
            continue
        dists = []
        cos_sims = []
        for idx in range(4):
            shape = np.array(json.loads(row[f"core{idx}_shape_mean"]))
            dist = _wasserstein_1d(ref[f"core_{idx}"], shape)
            dists.append(dist)
            coupling = _ot_coupling_1d(ref[f"core_{idx}"], shape)
            delta = (coupling - ref_coupling[idx]).ravel()
            cos_val = _cosine_similarity(delta, template_delta_by_core[idx])
            cos_sims.append(cos_val)
        mem_shape = np.array(json.loads(row["mem_shape_mean"]))
        mem_dist = _wasserstein_1d(ref["mem"], mem_shape)
        mem_coupling = _ot_coupling_1d(ref["mem"], mem_shape)
        mem_delta = (mem_coupling - ref_mem_coupling).ravel()
        mem_cos = _cosine_similarity(mem_delta, template_delta_mem)
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
                "reference_run_csv": ref["run_csv"],
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
    p.add_argument("--bins", type=int, default=50)
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
