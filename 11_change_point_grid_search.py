from __future__ import annotations

from pathlib import Path
from typing import Sequence
import itertools
import importlib.util

import pandas as pd
from tqdm import tqdm

from log_loader import load_logs


def _load_change_point_module() -> object:
    module_path = Path(__file__).resolve().parent / "1_change_point_analyze.py"
    spec = importlib.util.spec_from_file_location("change_point_analyze", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid_search_best_setting(
    df: pd.DataFrame,
    *,
    base_features: Sequence[str],
    models: Sequence[str],
    penalties: Sequence[float],
    min_sizes: Sequence[int],
    ts_col: str = "ts_unix",
    epoch_col: str = "epoch",
) -> pd.DataFrame:
    """
    Grid search combinations to maximize separation between clean and non-clean
    per device, then average across devices (Option C).
    """
    feature_sets: list[list[str]] = []
    for r in range(1, len(base_features) + 1):
        for combo in itertools.combinations(base_features, r):
            feature_sets.append(list(combo))

    change_point_module = _load_change_point_module()

    rows: list[dict[str, object]] = []

    configs = [
        (features, model, penalty, min_size)
        for features in feature_sets
        for model in models
        for penalty in penalties
        for min_size in min_sizes
    ]

    for features, model, penalty, min_size in tqdm(configs, desc="Grid search", unit="config"):
        rates = change_point_module.average_cp_rate_per_epoch_by_group(
            df,
            features,
            ts_col=ts_col,
            epoch_col=epoch_col,
            model=model,
            penalty=penalty,
            min_size=min_size,
        )

        if rates["avg_change_point_rate_per_epoch"].isna().all():
            score = float("-inf")
        else:
            device_scores = []
            for _, g in rates.groupby("device_id"):
                clean_row = g[g["poisoning_type"] == "clean"]
                other_rows = g[g["poisoning_type"] != "clean"]
                if clean_row.empty or other_rows.empty:
                    continue
                clean_val = clean_row["avg_change_point_rate_per_epoch"].iloc[0]
                other_val = other_rows["avg_change_point_rate_per_epoch"].mean()
                if pd.isna(clean_val) or pd.isna(other_val):
                    continue
                device_scores.append(abs(clean_val - other_val))
            score = (
                float(sum(device_scores) / len(device_scores))
                if device_scores
                else float("-inf")
            )

        rows.append(
            {
                "features": features,
                "model": model,
                "penalty": penalty,
                "min_size": min_size,
                "score": score,
            }
        )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    change_point_module = _load_change_point_module()
    combined_df = load_logs(root_dir)

    base_features = ["cpu_percent", "cpu_temp_c", "mem_percent"]
    search_results = grid_search_best_setting(
        combined_df,
        base_features=base_features,
        # models=["l2", "rbf"],
        models=["rbf"],
        # penalties=[1.0, 2.0, 5.0, 10.0, 20.0],
        penalties=[1.0],
        min_sizes=[2, 3, 5, 8],
    )
    print(search_results.head(10))
