from __future__ import annotations

from pathlib import Path
from typing import Iterable
import ast
import math

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


def _entropy_from_probs(probs: Iterable[float]) -> float:
    probs = [p for p in probs if p > 0]
    return -sum(p * math.log(p) for p in probs) if probs else float("nan")


def compute_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        core_vals = _parse_core_list(row.get("cpu_per_core"))
        if not core_vals:
            continue

        total = sum(core_vals)
        if total <= 0:
            continue
        sorted_vals = sorted(core_vals, reverse=True)
        max_margin = (sorted_vals[0] - sorted_vals[1]) / total if len(sorted_vals) > 1 else float("nan")
        argmax_core = int(max(range(len(core_vals)), key=lambda i: core_vals[i]))

        rows.append(
            {
                "device_id": row.get("device_id"),
                "poisoning_type": row.get("poisoning_type"),
                "poison_frac": row.get("poison_frac"),
                "epoch": row.get("epoch"),
                "argmax_core": argmax_core,
                "max_margin": max_margin,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    # Training rows only (epoch numeric).
    epoch_numeric = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()

    metrics_df = compute_core_metrics(df)

    # Aggregate per group to compare clean vs poisoned.
    rows: list[dict[str, object]] = []
    for (device_id, poisoning_type, poison_frac), g in metrics_df.groupby(
        ["device_id", "poisoning_type", "poison_frac"], dropna=False
    ):
        argmax_counts = g["argmax_core"].value_counts(normalize=True).sort_index()
        probs = argmax_counts.tolist()
        entropy = _entropy_from_probs(probs)
        num_cores = int(g["argmax_core"].max() + 1) if not g["argmax_core"].empty else 0
        normalized_entropy = entropy / math.log(num_cores) if num_cores > 1 else float("nan")
        hhi = sum(p * p for p in probs) if probs else float("nan")
        switch_rate = (
            (g["argmax_core"].diff() != 0).mean() if len(g) > 1 else float("nan")
        )

        # Mean run length of same argmax core
        runs = (g["argmax_core"].diff() != 0).cumsum()
        run_lengths = runs.value_counts()
        mean_run_length = run_lengths.mean() if not run_lengths.empty else float("nan")

        top_core_share = argmax_counts.max() if not argmax_counts.empty else float("nan")
        top2_share = argmax_counts.sort_values(ascending=False).head(2).sum() if len(argmax_counts) > 1 else float("nan")

        rows.append(
            {
                "device_id": device_id,
                "poisoning_type": poisoning_type,
                "poison_frac": poison_frac,
                "argmax_entropy": entropy,
                "argmax_entropy_norm": normalized_entropy,
                "argmax_hhi": hhi,
                "argmax_switch_rate": switch_rate,
                "argmax_mean_run_len": mean_run_length,
                "argmax_top_core_share": top_core_share,
                "argmax_top2_share": top2_share,
                "max_margin_mean": g["max_margin"].mean(),
            }
        )

    summary = pd.DataFrame(rows)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(summary)
