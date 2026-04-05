from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "ot_distance" not in df.columns:
        df["ot_distance"] = 0.8 * df["core_ot_distance_mean"] + 0.2 * df[
            "mem_ot_distance_to_clean_005"
        ]
    if "cosine_similarity" not in df.columns:
        df["cosine_similarity"] = 0.8 * df["core_type_cosine_mean_to_clean010"] + 0.2 * df[
            "mem_type_cosine_to_clean010"
        ]
    return df


def _mwu(clean_vals: np.ndarray, poison_vals: np.ndarray) -> dict[str, float]:
    clean_vals = clean_vals[np.isfinite(clean_vals)]
    poison_vals = poison_vals[np.isfinite(poison_vals)]
    if len(clean_vals) == 0 or len(poison_vals) == 0:
        return {"u_stat": float("nan"), "p_value": float("nan")}
    res = mannwhitneyu(clean_vals, poison_vals, alternative="two-sided", method="auto")
    return {"u_stat": float(res.statistic), "p_value": float(res.pvalue)}


def _run(df: pd.DataFrame, dataset: str, devices: list[str]) -> pd.DataFrame:
    rows = []
    for device_id in devices:
        sub = df[df["device_id"].astype(str) == device_id]
        if sub.empty:
            continue
        clean = sub[sub["poisoning_type"] == "clean"]
        poison = sub[
            (sub["poisoning_type"] != "clean")
            & (sub["poisoning_rate"] == 0.30)
        ]
        for metric in ["ot_distance", "cosine_similarity"]:
            res = _mwu(clean[metric].astype(float).to_numpy(), poison[metric].astype(float).to_numpy())
            rows.append(
                {
                    "dataset": dataset,
                    "device_id": device_id,
                    "metric": metric,
                    "n_clean": int(clean[metric].notna().sum()),
                    "n_poisoned": int(poison[metric].notna().sum()),
                    "u_stat": res["u_stat"],
                    "p_value": res["p_value"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bg", type=str, default="bg_main.csv")
    p.add_argument("--bg2", type=str, default="bg2_main.csv")
    p.add_argument("--finetune", type=str, default="")
    p.add_argument("--devices", type=str, default="114,115")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    rows = []
    device_list = [s.strip() for s in args.devices.split(",") if s.strip()]
    for path, name in [(args.bg, "bg"), (args.bg2, "bg2")]:
        pth = Path(path)
        if not pth.exists():
            print(f"[skip] missing: {pth}")
            continue
        df = pd.read_csv(pth)
        required = [
            "device_id",
            "poisoning_type",
            "poisoning_rate",
            "core_ot_distance_mean",
            "mem_ot_distance_to_clean_005",
            "core_type_cosine_mean_to_clean010",
            "mem_type_cosine_to_clean010",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[skip] {pth} missing columns: {missing}")
            continue
        df = _ensure_derived_columns(df)
        rows.append(_run(df, name, device_list))

    if args.finetune.strip():
        pth = Path(args.finetune)
        if not pth.exists():
            print(f"[skip] missing: {pth}")
        else:
            df = pd.read_csv(pth)
            required = [
                "device_id",
                "poisoning_type",
                "poisoning_rate",
                "core_ot_distance_mean",
                "mem_ot_distance_to_clean_005",
                "core_type_cosine_mean_to_clean010",
                "mem_type_cosine_to_clean010",
            ]
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[skip] {pth} missing columns: {missing}")
            else:
                df = _ensure_derived_columns(df)
                rows.append(_run(df, "finetune", device_list))

    if not rows:
        print("No results.")
        return

    result = pd.concat(rows, ignore_index=True).sort_values(["dataset", "device_id", "metric"])
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(result.to_string(index=False))

    if args.out.strip():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"Saved CSV to: {args.out}")


if __name__ == "__main__":
    main()
