from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

from log_loader import load_logs


def _load_module(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    # labels: 1 for poisoned, 0 for clean
    order = np.argsort(scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    # candidate thresholds between unique scores
    unique_scores = np.unique(scores_sorted)
    if len(unique_scores) == 1:
        return float(unique_scores[0]), 0.0

    best_thr = float(unique_scores[0])
    best_acc = -1.0

    for i in range(len(unique_scores) - 1):
        thr = (unique_scores[i] + unique_scores[i + 1]) / 2.0
        preds = (scores > thr).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return float(best_thr), float(best_acc)


def _metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
    }


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df = load_logs(root_dir)

    mod7 = _load_module(root_dir / "7_cpu_mem_ot_analysis.py", "cpu_mem_ot")
    mod8 = _load_module(root_dir / "8_cpu_core_sorted_ot_analysis.py", "cpu_core_ot")

    mem_cpu_df = mod7.cpu_mem_shape_ot(df, bins=50)
    core_df = mod8.cpu_core_sorted_ot(df, bins=50)

    features = (
        mem_cpu_df.merge(
            core_df,
            on=["device_id", "poisoning_type", "poison_frac"],
            how="inner",
        )
        .copy()
    )

    # Use mem OT + core OT mean as score
    features["score"] = (
        features["mem_ot_distance_to_clean_005"]
        + features["core_ot_distance_mean"]
    )

    # Label: clean=0, poisoned=1
    features["label"] = (features["poisoning_type"] != "clean").astype(int)

    features = features.dropna(subset=["score"])
    scores = features["score"].to_numpy()
    labels = features["label"].to_numpy()

    print("per_device_thresholds:")
    rows = []
    for device_id, g in features.groupby("device_id"):
        scores_d = g["score"].to_numpy()
        labels_d = g["label"].to_numpy()
        thr, best_acc = _best_threshold(scores_d, labels_d)
        preds = (scores_d > thr).astype(int)
        metrics = _metrics(preds, labels_d)
        rows.append(
            {
                "device_id": device_id,
                "threshold": thr,
                "accuracy": best_acc,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

    print(pd.DataFrame(rows).sort_values("device_id"))
    print(features[["device_id", "poisoning_type", "poison_frac", "score", "label"]])
