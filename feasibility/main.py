"""Batch runner for feasibility experiments.

    Runs ml_running.py for:
- clean (poison-type=clean, from disk)
- blurring
- occlusion
- label-flip

Each run uses the same base hyperparams and writes logs into:
  <log-root>/<mode>/<timestamp>.csv

Example:
  python main.py --epochs 10 --data-root data --log-root logs_batch
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Experiment
    p.add_argument("--epochs", type=int, default=10)

    # Data
    p.add_argument("--dataset", type=str, default="kuchidareo/small_trashnet")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-root", type=str, default="data", help="Root containing blurring/occlusion/label-flip/")
    p.add_argument("--poison-frac", type=float, default=1.0)

    # Image / transforms
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--normalize", type=str, default="0.5", choices=["none", "0.5", "imagenet"])

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="Override device passed to ml_running.py")

    # Logging
    p.add_argument("--log-root", type=str, default="logs_batch")
    p.add_argument("--log-fps", type=float, default=1.0)

    # Subsampling
    p.add_argument("--train-frac", type=float, default=1.0, help="Fraction of train split to use (0..1)")
    p.add_argument("--test-frac", type=float, default=1.0, help="Fraction of test split to use (0..1)")

    # Controls
    p.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["clean", "blurring", "occlusion", "label-flip"],
        choices=["none", "clean", "blurring", "occlusion", "label-flip"],
        help="Which modes to run",
    )
    p.add_argument(
        "--prepare-data",
        action="store_true",
        help="If set, run data_preparing.py when poison data is missing",
    )
    p.add_argument(
        "--prepare-max-per-split",
        type=int,
        default=None,
        help="Optional cap per split when preparing poison data",
    )
    p.add_argument(
        "--background-script",
        type=str,
        default=None,
        help="Optional background workload script to run during training",
    )
    p.add_argument(
        "--background-warmup-sec",
        type=float,
        default=3.0,
        help="Warmup time before training starts (seconds)",
    )

    return p.parse_args()


def run_one(mode: str, args: argparse.Namespace) -> None:
    """Run one experiment mode via subprocess."""
    here = Path(__file__).resolve().parent
    ml_script = here / "ml_running.py"

    log_dir = Path(args.log_root) / ("clean" if mode in ("none", "clean") else mode)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        str(ml_script),
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--poison-type",
        mode,
        "--poison-frac",
        str(args.poison_frac),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--normalize",
        args.normalize,
        "--log-dir",
        str(log_dir),
        "--log-fps",
        str(args.log_fps),
        "--train-frac",
        str(args.train_frac),
        "--test-frac",
        str(args.test_frac),
    ]

    if args.config:
        cmd.extend(["--config", args.config])

    if args.img_size is not None:
        cmd.extend(["--img-size", str(args.img_size)])

    if args.device:
        cmd.extend(["--device", args.device])

    print("\n=== Running ===")
    print(" ".join(cmd))

    if args.background_script:
        cmd.extend(["--background-script", args.background_script])
    subprocess.run(cmd, check=True)


def _poison_data_missing(data_root: str) -> bool:
    base = Path(data_root)
    required = [
        base / "clean" / "metadata.jsonl",
        base / "blurring" / "metadata.jsonl",
        base / "occlusion" / "metadata.jsonl",
        base / "label-flip" / "metadata.jsonl",
    ]
    return any(not p.exists() for p in required)


def maybe_prepare_data(args: argparse.Namespace) -> None:
    if not _poison_data_missing(args.data_root):
        return
    if not args.prepare_data:
        missing = [
            str(p)
            for p in (
                Path(args.data_root) / "blurring" / "metadata.jsonl",
                Path(args.data_root) / "occlusion" / "metadata.jsonl",
                Path(args.data_root) / "label-flip" / "metadata.jsonl",
            )
            if not p.exists()
        ]
        raise FileNotFoundError(
            "Missing poison data. Run data_preparing.py or pass --prepare-data. "
            f"Missing: {', '.join(missing)}"
        )

    here = Path(__file__).resolve().parent
    prep_script = here / "data_preparing.py"
    cmd: List[str] = [
        sys.executable,
        str(prep_script),
        "--dataset",
        args.dataset,
        "--out",
        args.data_root,
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.prepare_max_per_split is not None:
        cmd.extend(["--max-per-split", str(args.prepare_max_per_split)])

    print("\n=== Preparing data ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    if any(m in ("clean", "blurring", "occlusion", "label-flip") for m in args.modes):
        maybe_prepare_data(args)

    bg_proc = None
    if args.background_script:
        bg_proc = subprocess.Popen(
            ["bash", args.background_script],
            start_new_session=True,
        )
        if args.background_warmup_sec > 0:
            import time

            time.sleep(float(args.background_warmup_sec))

    try:
        for mode in args.modes:
            run_one(mode, args)
    finally:
        if bg_proc is not None:
            try:
                bg_proc.terminate()
                bg_proc.wait(timeout=5.0)
            except Exception:
                try:
                    bg_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
