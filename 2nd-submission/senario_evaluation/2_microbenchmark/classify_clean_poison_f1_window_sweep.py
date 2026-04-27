from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


FEATURE_ALIASES = {
    "core_ot_distance_mean_to_base": ["core_ot_distance_mean"],
    "mem_ot_distance_to_base": ["mem_ot_distance_to_clean_005"],
    "core_type_cosine_mean_to_ref": ["core_type_cosine_mean_to_clean010"],
    "mem_type_cosine_to_ref": ["mem_type_cosine_to_clean010"],
}


def _normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for canonical, aliases in FEATURE_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df[canonical] = df[alias]
                break
    return df


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "ot_distance" not in df.columns:
        df["ot_distance"] = (
            0.8 * df["core_ot_distance_mean_to_base"]
            + 0.2 * df["mem_ot_distance_to_base"]
        )
    if "cosine_similarity" not in df.columns:
        df["cosine_similarity"] = (
            0.8 * df["core_type_cosine_mean_to_ref"]
            + 0.2 * df["mem_type_cosine_to_ref"]
        )
    return df


def _ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "classification_label" in df.columns:
        labels = df["classification_label"].astype(str).str.strip().str.lower()
        valid = {"clean", "poisoned"}
        invalid = sorted(set(labels.dropna()) - valid)
        if invalid:
            raise ValueError(
                "classification_label must contain only 'clean' or 'poisoned'; "
                f"found: {invalid}"
            )
        df["label"] = (labels == "poisoned").astype(int)
    else:
        df["label"] = (df["poisoning_type"] != "clean").astype(int)
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


def _metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    mask = np.isfinite(scores)
    y_true = y_true[mask]
    scores = scores[mask]
    if scores.size == 0 or not np.isfinite(threshold):
        return {
            "j": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "fpr": float("nan"),
            "fnr": float("nan"),
        }
    y_pred = scores >= threshold
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "j": float(recall - fpr),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def _stratified_calibration_mask(
    agg: pd.DataFrame,
    calibration_fraction: float,
) -> np.ndarray:
    if not 0 < calibration_fraction < 1:
        raise ValueError("--calibration-fraction must be in (0, 1).")

    if "target_run_csv" in agg.columns:
        unit_col = "target_run_csv"
        group_cols = [unit_col]
        stratify_cols = ["label"]
        if "target_group" in agg.columns:
            group_cols.append("target_group")
            stratify_cols.append("target_group")
        unit_labels = agg.groupby(group_cols, dropna=False)["label"].max().reset_index()
        unit_labels = unit_labels.sort_values(group_cols)
        calibration_units: set[object] = set()
        for _, label_units in unit_labels.groupby(stratify_cols, dropna=False):
            units = list(label_units[unit_col])
            n_calibration = int(np.ceil(len(units) * calibration_fraction))
            if len(units) > 1:
                n_calibration = min(max(n_calibration, 1), len(units) - 1)
            calibration_units.update(units[:n_calibration])
        return agg[unit_col].isin(calibration_units).to_numpy()

    calibration_mask = np.zeros(len(agg), dtype=bool)
    for label in sorted(agg["label"].dropna().unique()):
        label_indices = list(agg.index[agg["label"] == label])
        n_calibration = int(np.ceil(len(label_indices) * calibration_fraction))
        if len(label_indices) > 1:
            n_calibration = min(max(n_calibration, 1), len(label_indices) - 1)
        calibration_mask[label_indices[:n_calibration]] = True
    return calibration_mask


def _empty_result_row(
    device_id: object,
    window: int,
    y_true: np.ndarray,
) -> dict[str, object]:
    return {
        "device_id": device_id,
        "window": int(window),
        "n_clean": int(np.sum(y_true == 0)),
        "n_poisoned": int(np.sum(y_true == 1)),
        "n_calibration_clean": 0,
        "n_calibration_poisoned": 0,
        "n_test_clean": int(np.sum(y_true == 0)),
        "n_test_poisoned": int(np.sum(y_true == 1)),
        "best_w_ot": float("nan"),
        "best_w_cos": float("nan"),
        "best_threshold": float("nan"),
        "calibration_j": float("nan"),
        "calibration_f1": float("nan"),
        "calibration_precision": float("nan"),
        "calibration_recall": float("nan"),
        "calibration_fpr": float("nan"),
        "calibration_fnr": float("nan"),
        "best_j": float("nan"),
        "best_f1": float("nan"),
        "best_precision": float("nan"),
        "best_recall": float("nan"),
        "best_fpr": float("nan"),
        "best_fnr": float("nan"),
    }


def _eval_device_window(
    df: pd.DataFrame,
    window: int,
    grid_step: float,
    calibration_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for device_id, grp in df.groupby("device_id", dropna=False):
        grp = grp.sort_values("round").copy()
        grp["window_id"] = (grp["round"].astype(int) // int(window))
        group_cols = [
            "device_id",
            "poisoning_type",
            "poisoning_rate",
            "window_id",
        ]
        if "target_run_csv" in grp.columns:
            group_cols.insert(1, "target_run_csv")
        if "target_group" in grp.columns:
            group_cols.insert(2 if "target_run_csv" in grp.columns else 1, "target_group")
        agg = (
            grp.groupby(group_cols, dropna=False)
            .agg(
                label=("label", "max"),
                score_base_ot=("ot_distance", "mean"),
                score_base_cos=("cosine_similarity", "mean"),
            )
            .reset_index()
        )
        y_true = agg["label"].to_numpy()
        if len(np.unique(y_true)) < 2:
            rows.append(_empty_result_row(device_id, window, y_true))
            continue

        calibration_mask = _stratified_calibration_mask(agg, calibration_fraction)
        test_mask = ~calibration_mask
        y_calibration = agg.loc[calibration_mask, "label"].to_numpy()
        y_test = agg.loc[test_mask, "label"].to_numpy()
        if len(np.unique(y_calibration)) < 2 or len(np.unique(y_test)) < 2:
            rows.append(_empty_result_row(device_id, window, y_test))
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
            trial = _best_threshold_by_j(y_calibration, scores[calibration_mask])
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
            rows.append(_empty_result_row(device_id, window, y_test))
            continue

        test_scores = (
            best["w_ot"] * agg.loc[test_mask, "score_base_ot"]
            + best["w_cos"] * (1.0 - agg.loc[test_mask, "score_base_cos"])
        ).astype(float).to_numpy()
        test = _metrics_at_threshold(y_test, test_scores, best["threshold"])
        rows.append(
            {
                "device_id": device_id,
                "window": int(window),
                "n_clean": int(np.sum(y_test == 0)),
                "n_poisoned": int(np.sum(y_test == 1)),
                "n_calibration_clean": int(np.sum(y_calibration == 0)),
                "n_calibration_poisoned": int(np.sum(y_calibration == 1)),
                "n_test_clean": int(np.sum(y_test == 0)),
                "n_test_poisoned": int(np.sum(y_test == 1)),
                "best_w_ot": best["w_ot"],
                "best_w_cos": best["w_cos"],
                "best_threshold": best["threshold"],
                "calibration_j": best["j"],
                "calibration_f1": best["f1"],
                "calibration_precision": best["precision"],
                "calibration_recall": best["recall"],
                "calibration_fpr": best["fpr"],
                "calibration_fnr": best["fnr"],
                "best_j": test["j"],
                "best_f1": test["f1"],
                "best_precision": test["precision"],
                "best_recall": test["recall"],
                "best_fpr": test["fpr"],
                "best_fnr": test["fnr"],
            }
        )
    return pd.DataFrame(rows)


def run_window_sweep(
    df: pd.DataFrame,
    windows: list[int],
    grid_step: float = 0.1,
    calibration_fraction: float = 0.5,
) -> pd.DataFrame:
    df = _normalize_feature_columns(df)

    required = [
        "device_id",
        "poisoning_type",
        "poisoning_rate",
        "round",
        "core_ot_distance_mean_to_base",
        "mem_ot_distance_to_base",
        "core_type_cosine_mean_to_ref",
        "mem_type_cosine_to_ref",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not windows:
        raise ValueError("windows must list at least one integer")

    df = _ensure_derived_columns(df)
    df = _ensure_label_column(df)

    all_rows = []
    for w in windows:
        all_rows.append(
            _eval_device_window(
                df,
                int(w),
                float(grid_step),
                calibration_fraction=float(calibration_fraction),
            )
        )
    result = pd.concat(all_rows, ignore_index=True)
    return result.sort_values(["device_id", "window"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="fl_main.csv")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--grid-step", type=float, default=0.1)
    p.add_argument("--windows", type=str, default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    windows = [int(s.strip()) for s in args.windows.split(",") if s.strip()]
    result = run_window_sweep(
        df=df,
        windows=windows,
        grid_step=float(args.grid_step),
        calibration_fraction=float(args.calibration_fraction),
    )

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
