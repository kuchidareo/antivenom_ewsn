from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def _extract_run_info(event_series: pd.Series) -> dict:
    for raw in event_series.dropna().astype(str):
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "run_info" in parsed:
            return parsed.get("run_info") or {}
    return {}


def load_logs(root: Path) -> pd.DataFrame:
    """
    Read all CSVs under log/logs_batch_*/logs_batch/*/*.csv and return one DataFrame
    with device_id, poison_type, poison_frac, dataset, and all original CSV columns.
    """
    csv_paths = sorted(root.glob("log/logs_batch_*/logs_batch/*/*.csv"))
    frames: list[pd.DataFrame] = []

    for path in csv_paths:
        # Expected path example:
        # log/logs_batch_114/logs_batch/blurring/20260129_153025.csv
        parts = path.parts
        poisoning_type_dir = parts[-2]
        batch_dir = parts[-4]  # logs_batch_114
        device_id = batch_dir.split("_")[-1]

        df = pd.read_csv(path)
        run_info = _extract_run_info(df.get("event", pd.Series(dtype=str)))
        poisoning_type = run_info.get("poison_type") or poisoning_type_dir
        if poisoning_type == "none":
            poisoning_type = "clean"
        poison_frac = run_info.get("poison_frac")
        dataset = run_info.get("dataset")

        df.insert(0, "device_id", device_id)
        df.insert(1, "poisoning_type", poisoning_type)
        df.insert(2, "poison_frac", poison_frac)
        df.insert(3, "dataset", dataset)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["device_id", "poisoning_type", "poison_frac", "dataset"])

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    combined_df = load_logs(root_dir)
    print(combined_df.head())
    print(f"rows={len(combined_df)} cols={len(combined_df.columns)}")
