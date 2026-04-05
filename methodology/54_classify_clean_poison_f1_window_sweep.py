from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


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


def _best_threshold_by_j(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(scores)
    y_true = y_true[mask]
    scores = scores[mask]
    if scores.size == 0 or len(np.unique(y_true)) < 2:
        return {
            "threshold": float("nan"),
            "j": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "fpr": float("nan"),
            "fnr": float("nan"),
        }
    thresholds = np.unique(scores)
    best = {
        "threshold": float("nan"),
        "j": -1.0,
        "f1": -1.0,
        "precision": float("nan"),
        "recall": float("nan"),
        "fpr": float("nan"),
        "fnr": float("nan"),
    }
    for t in thresholds:
        y_pred = scores >= t
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        tn = float(np.sum((y_true == 0) & (y_pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        j = recall - fpr
        if j > best["j"]:
            best = {
                "threshold": float(t),
                "j": float(j),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "fpr": float(fpr),
                "fnr": float(fnr),
            }
    return best


def _eval_device_window(df: pd.DataFrame, window: int, grid_step: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, grp in df.groupby("device_id", dropna=False):
        grp = grp.sort_values("round").copy()
        grp["window_id"] = (grp["round"].astype(int) // int(window))
        agg = (
            grp.groupby([
                "device_id",
                "poisoning_type",
                "poisoning_rate",
                "window_id",
            ], dropna=False)
            .agg(
                label=("label", "max"),
                score_base_ot=("ot_distance", "mean"),
                score_base_cos=("cosine_similarity", "mean"),
            )
            .reset_index()
        )
        y_true = agg["label"].to_numpy()
        if len(np.unique(y_true)) < 2:
            rows.append(
                {
                    "device_id": device_id,
                    "window": int(window),
                    "n_clean": int(np.sum(y_true == 0)),
                    "n_poisoned": int(np.sum(y_true == 1)),
                    "best_w_ot": float("nan"),
                    "best_w_cos": float("nan"),
                    "best_threshold": float("nan"),
                    "best_j": float("nan"),
                    "best_f1": float("nan"),
                    "best_precision": float("nan"),
                    "best_recall": float("nan"),
                    "best_fpr": float("nan"),
                    "best_fnr": float("nan"),
                }
            )
            continue
        best = {
            "w_ot": float("nan"),
            "w_cos": float("nan"),
            "threshold": float("nan"),
            "j": -1.0,
            "f1": -1.0,
            "precision": float("nan"),
            "recall": float("nan"),
            "fpr": float("nan"),
            "fnr": float("nan"),
        }
        found = False
        step = float(grid_step)
        if step <= 0 or step > 1:
            raise ValueError("--grid-step must be in (0, 1].")
        w_ots = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
        for w_ot in w_ots:
            w_cos = 1.0 - w_ot
            scores = (
                w_ot * agg["score_base_ot"] + w_cos * (1.0 - agg["score_base_cos"])
            ).astype(float).to_numpy()
            trial = _best_threshold_by_j(y_true, scores)
            if np.isfinite(trial["j"]) and (trial["j"] > best["j"]):
                best = {
                    "w_ot": float(w_ot),
                    "w_cos": float(w_cos),
                    "threshold": trial["threshold"],
                    "j": trial["j"],
                    "f1": trial["f1"],
                    "precision": trial["precision"],
                    "recall": trial["recall"],
                    "fpr": trial["fpr"],
                    "fnr": trial["fnr"],
                }
                found = True
        if not found:
            best = {
                "w_ot": float("nan"),
                "w_cos": float("nan"),
                "threshold": float("nan"),
                "j": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "fpr": float("nan"),
                "fnr": float("nan"),
            }
        rows.append(
            {
                "device_id": device_id,
                "window": int(window),
                "n_clean": int(np.sum(y_true == 0)),
                "n_poisoned": int(np.sum(y_true == 1)),
                "best_w_ot": best["w_ot"],
                "best_w_cos": best["w_cos"],
                "best_threshold": best["threshold"],
                "best_j": best["j"],
                "best_f1": best["f1"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "best_fpr": best["fpr"],
                "best_fnr": best["fnr"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="fl_main.csv")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--grid-step", type=float, default=0.1)
    p.add_argument("--windows", type=str, default="1,2,3,4,5,6,7,8,9,10")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    required = [
        "device_id",
        "poisoning_type",
        "poisoning_rate",
        "round",
        "core_ot_distance_mean",
        "mem_ot_distance_to_clean_005",
        "core_type_cosine_mean_to_clean010",
        "mem_type_cosine_to_clean010",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _ensure_derived_columns(df)
    df = df.copy()
    df["label"] = (df["poisoning_type"] != "clean").astype(int)

    windows = [int(s.strip()) for s in args.windows.split(",") if s.strip()]
    if not windows:
        raise ValueError("--windows must list at least one integer")
    all_rows = []
    for w in windows:
        all_rows.append(_eval_device_window(df, w, float(args.grid_step)))
    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["device_id", "window"])

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
