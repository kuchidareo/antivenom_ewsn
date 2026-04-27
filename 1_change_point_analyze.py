from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    import ruptures as rpt
except ImportError as exc:  # pragma: no cover - runtime check
    raise SystemExit(
        "Missing dependency: ruptures. Install with `pip install ruptures`."
    ) from exc

from log_loader import load_logs

def average_cp_rate_per_epoch_by_group(
    df: pd.DataFrame,
    analysis_features: Sequence[str],
    *,
    ts_col: str = "ts_unix",
    epoch_col: str = "epoch",
    model: str = "rbf",
    penalty: float = 1.0,
    min_size: int = 4,
) -> pd.DataFrame:
    """
    Compute average change-point rate per epoch for each (device_id, poisoning_type).

    Returns a DataFrame with:
    device_id, poisoning_type, avg_change_point_rate_per_epoch
    """
    if ts_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {ts_col}")
    if epoch_col not in df.columns:
        raise ValueError(f"Missing epoch column: {epoch_col}")

    # Training rows are those where the epoch column is numeric.
    epoch_numeric = pd.to_numeric(df[epoch_col], errors="coerce")
    df = df.loc[epoch_numeric.notna()].copy()

    rows: list[dict[str, object]] = []

    total_groups = df.groupby(["device_id", "poisoning_type"], dropna=False).ngroups
    for idx, ((device_id, poisoning_type), group) in enumerate(
        df.groupby(["device_id", "poisoning_type"], dropna=False), start=1
    ):
        group = group.reset_index(drop=True)
        data = group[list(analysis_features)].dropna()
        if len(data) < max(2 * min_size, 5):
            rows.append(
                {
                    "device_id": device_id,
                    "poisoning_type": poisoning_type,
                    "avg_change_point_rate_per_epoch": float("nan"),
                }
            )
            continue

        algo = rpt.Pelt(model=model, min_size=min_size).fit(data.values)
        cps = algo.predict(pen=penalty)
        cps = [cp for cp in cps if cp < len(group)]

        if not cps:
            rows.append(
                {
                    "device_id": device_id,
                    "poisoning_type": poisoning_type,
                    "avg_change_point_rate_per_epoch": float("nan"),
                }
            )
            continue

        cp_epochs = group.loc[cps, epoch_col].tolist()
        cp_counts = pd.Series(cp_epochs).value_counts()

        epoch_times = group.groupby(epoch_col)[ts_col].agg(["min", "max"])
        epoch_times["elapsed"] = epoch_times["max"] - epoch_times["min"]

        rates: list[float] = []
        for epoch_val, count in cp_counts.items():
            if epoch_val not in epoch_times.index:
                continue
            elapsed = epoch_times.loc[epoch_val, "elapsed"]
            if pd.isna(elapsed) or elapsed <= 0:
                continue
            rates.append(count / elapsed)

        avg_rate = float(sum(rates) / len(rates)) if rates else float("nan")
        rows.append(
            {
                "device_id": device_id,
                "poisoning_type": poisoning_type,
                "avg_change_point_rate_per_epoch": avg_rate,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    combined_df = load_logs(root_dir)
    analysis_features = [
        #"cpu_percent",
        "cpu_temp_c",
        # "mem_percent"
    ]
    model = "rbf"
    penalty = 1.0
    min_size = 3
    group_rates = average_cp_rate_per_epoch_by_group(
        combined_df,
        analysis_features,
        model=model,
        penalty=penalty,
        min_size=min_size,
    )
    print(group_rates)
