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


def load_logs(root: Path, log_dir: str = "log", check_schema: bool = False) -> pd.DataFrame:
    """
    Read all CSVs under {log_dir}/logs_*_*/logs_*/*/*.csv and return one DataFrame
    with device_id, poison_type, poison_frac, dataset, and all original CSV columns.
    """
    csv_paths = sorted(root.glob(f"{log_dir}/logs_*_*/logs_*/*/*.csv"))
    frames: list[pd.DataFrame] = []

    for path in csv_paths:
        # Expected path example:
        # log_x/logs_bg_114/logs_bg/blurring/20260203_130115.csv
        parts = path.parts
        poisoning_type_dir = parts[-2]
        batch_dir = parts[-4]  # logs_bg_114, logs_sl_119, logs_finetune_113, ...
        device_id = batch_dir.split("_")[-1]
        try:
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
        except:
            print(f'{path} is broken')

    if not frames:
        return pd.DataFrame(columns=["device_id", "poisoning_type", "poison_frac", "dataset"])

    combined = pd.concat(frames, ignore_index=True)
    print(combined)
    if check_schema:
        required = ["epoch", "ts_unix", "cpu_per_core", "mem_percent"]
        missing = [c for c in required if c not in combined.columns]
        if missing:
            print(f"[schema] missing columns: {missing}")
        else:
            print("[schema] all required columns present")
    return combined


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    combined_df = load_logs(root_dir)
    print(combined_df.head())
    print(f"rows={len(combined_df)} cols={len(combined_df.columns)}")
