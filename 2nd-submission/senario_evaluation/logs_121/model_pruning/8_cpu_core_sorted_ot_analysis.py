from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def _extract_run_info(df: pd.DataFrame) -> dict[str, object]:
    for event in df.get("event", pd.Series(dtype=object)).dropna():
        text = str(event).strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        run_info = payload.get("run_info")
        if isinstance(run_info, dict):
            return run_info
    return {}


def _load_run(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    run_info = _extract_run_info(df)
    df["run_label"] = label
    df["source_csv"] = str(csv_path)
    df["poisoning_type"] = run_info.get("poison_type", label)
    df["poison_frac"] = float(run_info.get("poison_frac", 1.0))
    return df


def load_default_runs(root_dir: Path) -> pd.DataFrame:
    run_paths = {
        "baseclean": root_dir.parent / "baseclean" / "20260403_163855.csv",
        "clean": root_dir / "clean" / "20260403_151459.csv",
        "blurring": root_dir / "blurring" / "20260403_160017.csv",
    }
    missing = [str(path) for path in run_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required CSV(s): {missing}")
    frames = [_load_run(path, label) for label, path in run_paths.items()]
    return pd.concat(frames, ignore_index=True, sort=False)


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


def cpu_core_sorted_ot(df: pd.DataFrame, bins: int = 50, reference_label: str = "baseclean") -> pd.DataFrame:
    required = ["epoch", "ts_unix", "cpu_per_core", "run_label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()
    df["epoch"] = epoch_numeric.loc[df.index].astype(int)

    core_cols = ["core_0", "core_1", "core_2", "core_3"]
    core_vals = df["cpu_per_core"].apply(_parse_core_list)
    df = df.loc[core_vals.notna()].copy()
    sorted_vals = core_vals.apply(lambda v: sorted(v, reverse=True))

    for i, col in enumerate(core_cols):
        df[col] = sorted_vals.apply(lambda v: float(v[i]) if i < len(v) else float("nan"))

    df = df.loc[df[core_cols].notna().all(axis=1)].copy()

    per_epoch_shapes: dict[str, dict[str, list[np.ndarray]]] = {}
    for run_label, group in df.groupby("run_label", dropna=False):
        per_epoch_shapes.setdefault(run_label, {c: [] for c in core_cols})
        for _, epoch_group in group.groupby("epoch", dropna=False):
            for col in core_cols:
                shape = _epoch_shape_mean(epoch_group, col, bins=bins)
                per_epoch_shapes[run_label][col].append(shape)

    rows: list[dict[str, object]] = []
    for run_label, shapes in per_epoch_shapes.items():
        mean_shapes: dict[str, np.ndarray] = {}
        for col in core_cols:
            arr = np.vstack(shapes[col]) if shapes[col] else np.zeros((0, bins))
            mean_shapes[col] = arr.mean(axis=0) if len(arr) else np.zeros(bins)

        run_meta = df.loc[df["run_label"] == run_label].iloc[0]
        rows.append(
            {
                "run_label": run_label,
                "poisoning_type": run_meta["poisoning_type"],
                "poison_frac": run_meta["poison_frac"],
                "source_csv": run_meta["source_csv"],
                "core0_shape_mean": json.dumps(mean_shapes["core_0"].tolist()),
                "core1_shape_mean": json.dumps(mean_shapes["core_1"].tolist()),
                "core2_shape_mean": json.dumps(mean_shapes["core_2"].tolist()),
                "core3_shape_mean": json.dumps(mean_shapes["core_3"].tolist()),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    ref_row = summary.loc[summary["run_label"] == reference_label]
    if ref_row.empty:
        raise ValueError(f"Reference run_label '{reference_label}' not found.")
    ref_row = ref_row.iloc[0]

    ref_shapes = {
        f"core_{idx}": np.array(json.loads(ref_row[f"core{idx}_shape_mean"]), dtype=float)
        for idx in range(4)
    }

    distances: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        if row["run_label"] == reference_label:
            continue
        dists = []
        for idx in range(4):
            shape = np.array(json.loads(row[f"core{idx}_shape_mean"]), dtype=float)
            dists.append(_wasserstein_1d(ref_shapes[f"core_{idx}"], shape))
        distances.append(
            {
                "reference_run": reference_label,
                "target_run": row["run_label"],
                "poisoning_type": row["poisoning_type"],
                "poison_frac": row["poison_frac"],
                "source_csv": row["source_csv"],
                "core0_ot_distance_to_baseclean": dists[0],
                "core1_ot_distance_to_baseclean": dists[1],
                "core2_ot_distance_to_baseclean": dists[2],
                "core3_ot_distance_to_baseclean": dists[3],
                "core_ot_distance_mean": float(np.mean(dists)),
            }
        )

    return pd.DataFrame(distances).sort_values(["core_ot_distance_mean", "target_run"]).reset_index(drop=True)


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_default_runs(root_dir)
    summary = cpu_core_sorted_ot(df, bins=50)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(summary.to_string(index=False))
