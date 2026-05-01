from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_runner():
    path = SCRIPT_DIR / "run_cost_function_microbenchmark.py"
    spec = importlib.util.spec_from_file_location("simple_c11_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simple c11 window-1 classification outputs.")
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else results_dir / "plots"
    runner = _load_runner()

    count = 0
    for classification_csv in sorted(results_dir.glob("*__c11__*_classification.csv")):
        stem = classification_csv.stem.removesuffix("_classification")
        points_csv = results_dir / f"{stem}_window1_points.csv"
        if not points_csv.exists():
            print(f"Skipping {classification_csv}: missing {points_csv}")
            continue
        points = pd.read_csv(points_csv)
        classification = pd.read_csv(classification_csv)
        out_path = out_dir / f"{stem}_window1.png"
        runner.plot_window1_threshold(points, classification, out_path)
        print(f"Saved plot: {out_path}")
        count += 1

    print(f"Saved {count} plots under: {out_dir}")


if __name__ == "__main__":
    main()
