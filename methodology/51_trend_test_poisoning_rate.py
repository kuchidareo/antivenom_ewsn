from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, combine_pvalues


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


def _spearman_per_device(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (device_id, ptype), grp in df.groupby(["device_id", "poisoning_type"], dropna=False):
        for metric in ["ot_distance", "cosine_similarity"]:
            x = grp["poisoning_rate"].astype(float).to_numpy()
            y = grp[metric].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if len(x) < 2:
                rows.append(
                    {
                        "device_id": device_id,
                        "poisoning_type": ptype,
                        "metric": metric,
                        "n": int(len(x)),
                        "spearman_r": float("nan"),
                        "p_value": float("nan"),
                    }
                )
                continue
            r, p = spearmanr(x, y)
            rows.append(
                {
                    "device_id": device_id,
                    "poisoning_type": ptype,
                    "metric": metric,
                    "n": int(len(x)),
                    "spearman_r": float(r),
                    "p_value": float(p),
                }
            )
    return pd.DataFrame(rows).sort_values(["device_id", "poisoning_type", "metric"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="fl_main.csv")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--exclude-device", type=str, default="120")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

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
        raise ValueError(f"Missing required columns: {missing}")

    df = _ensure_derived_columns(df)
    df = df[df["poisoning_type"] != "clean"].copy()

    result = _spearman_per_device(df)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(result.to_string(index=False))

    exclude = {s.strip() for s in args.exclude_device.split(",") if s.strip()}
    fisher_rows: list[dict[str, object]] = []
    for (ptype, metric), grp in result.groupby(["poisoning_type", "metric"], dropna=False):
        use = grp[~grp["device_id"].astype(str).isin(exclude)]
        pvals = use["p_value"].astype(float).dropna().to_numpy()
        pvals = pvals[np.isfinite(pvals)]
        if len(pvals) == 0:
            fisher_rows.append(
                {
                    "poisoning_type": ptype,
                    "metric": metric,
                    "n_devices": 0,
                    "fisher_stat": float("nan"),
                    "fisher_p_value": float("nan"),
                }
            )
            continue
        stat, pval = combine_pvalues(pvals, method="fisher")
        fisher_rows.append(
            {
                "poisoning_type": ptype,
                "metric": metric,
                "n_devices": int(len(pvals)),
                "fisher_stat": float(stat),
                "fisher_p_value": float(pval),
            }
        )
    fisher_df = pd.DataFrame(fisher_rows).sort_values(
        ["poisoning_type", "metric"]
    )
    print("\nFisher combined p-values (excluding device_id(s): %s)" % ", ".join(sorted(exclude)))
    print(fisher_df.to_string(index=False))

    if args.out.strip():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"Saved CSV to: {args.out}")


if __name__ == "__main__":
    main()
