from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from ml_running import (
    CleanHFDataset,
    PoisonDiskDataset,
    SimpleCNN,
    TransformConfig,
    _find_image_column,
    _find_label_column,
    _get_class_names,
    build_splits,
    load_dataset,
    sample_n_dataset,
    set_seed,
    subsample_dataset,
)


REORDER_RE = re.compile(r"primitive,exec,cpu,reorder")
CONV_RE = re.compile(r"primitive,exec,cpu,convolution")
REORDER_TIME_RE = re.compile(r"primitive,exec,cpu,reorder.*?,([0-9.]+)$", re.M)
CONV_TIME_RE = re.compile(r"primitive,exec,cpu,convolution.*?,([0-9.]+)$", re.M)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    script_dir = Path(__file__).resolve().parent
    script_candidate = (script_dir / path).resolve()
    if script_candidate.exists():
        return script_candidate

    repo_candidate = (script_dir.parent / path).resolve()
    return repo_candidate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare oneDNN reorder behavior between clean and blurring30 ML runs."
    )
    p.add_argument("--dataset", type=str, default="kuchidareo/small_trashnet")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-root", type=str, default="2nd-submission/jit_compare_data")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--poison-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", type=str, default="0.5", choices=["none", "0.5", "imagenet"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="2nd-submission/reorder_compare_clean_vs_blurring30")
    p.add_argument("--worker-case", type=str, default=None, choices=[None, "clean", "blurring30"])
    p.add_argument("--worker-summary", type=str, default=None)
    return p.parse_args()


def build_train_dataset(
    *,
    dataset_name: str,
    config: str | None,
    data_root: Path,
    poison_type: str,
    poison_frac: float,
    img_size: int,
    normalize: str,
    seed: int,
) -> Tuple[torch.utils.data.Dataset, int]:
    tfm_cfg = TransformConfig(img_size=img_size, normalize=normalize)
    local_clean_dir = data_root / "clean"
    local_clean_meta = local_clean_dir / "metadata.jsonl"

    if local_clean_meta.exists():
        clean_train = PoisonDiskDataset(
            variant_dir=local_clean_dir,
            split_name="train",
            tfm_cfg=tfm_cfg,
        )
        labels = {int(label) for _, label in clean_train.items}
        n_classes = len(labels)
        train_split = "train"
    else:
        ds_dict = load_dataset(dataset_name, config) if config else load_dataset(dataset_name)
        train_split, _ = build_splits(ds_dict)
        train_hf = ds_dict[train_split]
        image_col = _find_image_column(train_hf.features)
        label_col = _find_label_column(train_hf.features)
        class_names = _get_class_names(train_hf, label_col)
        if class_names is not None:
            n_classes = len(class_names)
        else:
            labels = set(int(train_hf[i][label_col]) for i in range(min(len(train_hf), 5000)))
            n_classes = len(labels)
        clean_train = CleanHFDataset(train_hf, image_col=image_col, label_col=label_col, tfm_cfg=tfm_cfg)

    clean_train = subsample_dataset(clean_train, frac=1.0, seed=seed)

    if poison_type == "none":
        return clean_train, n_classes

    variant_dir = data_root / poison_type
    poison_train = PoisonDiskDataset(variant_dir=variant_dir, split_name=train_split, tfm_cfg=tfm_cfg)
    target_n = len(clean_train)
    poison_k = int(round(target_n * float(poison_frac)))
    poison_k = max(0, min(poison_k, len(poison_train)))
    clean_k = max(0, target_n - poison_k)

    poison_sampled = sample_n_dataset(poison_train, n=poison_k, seed=seed)
    clean_sampled = sample_n_dataset(clean_train, n=clean_k, seed=seed + 1)
    train_ds = torch.utils.data.ConcatDataset([clean_sampled, poison_sampled])
    return train_ds, n_classes


def take_n_batches(loader: Iterable[Tuple[torch.Tensor, torch.Tensor]], steps: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, batch in enumerate(loader):
        if i >= steps:
            break
        batches.append(batch)
    if not batches:
        raise RuntimeError("No batches were produced.")
    return batches


def run_case(
    *,
    case_name: str,
    poison_type: str,
    poison_frac: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    set_seed(args.seed)
    torch.set_num_threads(1)

    dataset, n_classes = build_train_dataset(
        dataset_name=args.dataset,
        config=args.config,
        data_root=resolve_repo_path(args.data_root),
        poison_type=poison_type,
        poison_frac=poison_frac,
        img_size=args.img_size,
        normalize=args.normalize,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batches = take_n_batches(loader, args.steps)

    device = torch.device(args.device)
    model = SimpleCNN(n_classes=n_classes, img_size=args.img_size).to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case_name}_onednn.log"
    profiler_path = out_dir / f"{case_name}_profiler.txt"

    profiler_tables: List[str] = []
    loss_trace: List[float] = []

    with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
        for step, (x, y) in enumerate(batches):
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True, acc_events=True) as prof:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optim.step()
            loss_trace.append(float(loss.item()))
            profiler_tables.append(
                f"step={step}\n"
                + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
            )

    profiler_path.write_text("\n\n".join(profiler_tables) + "\n", encoding="utf-8")

    return {
        "case": case_name,
        "poison_type": poison_type,
        "poison_frac": poison_frac,
        "steps": len(batches),
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "loss_trace": loss_trace,
        "onednn_log": str(log_path),
        "profiler_summary": str(profiler_path),
    }


def analyze_log(path: Path, steps: int) -> Dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    reorder_times = [float(m.group(1)) for m in REORDER_TIME_RE.finditer(text)]
    convolution_times = [float(m.group(1)) for m in CONV_TIME_RE.finditer(text)]
    reorder_count = len(reorder_times)
    convolution_count = len(convolution_times)
    return {
        "reorder_exec_count": reorder_count,
        "convolution_exec_count": convolution_count,
        "reorder_per_step": reorder_count / max(1, steps),
        "reorder_exec_time_ms_sum": sum(reorder_times),
        "reorder_exec_time_ms_avg": (sum(reorder_times) / reorder_count) if reorder_count else 0.0,
        "reorder_exec_time_ms_max": max(reorder_times) if reorder_times else 0.0,
        "convolution_exec_time_ms_sum": sum(convolution_times),
        "convolution_exec_time_ms_avg": (sum(convolution_times) / convolution_count) if convolution_count else 0.0,
        "convolution_exec_time_ms_max": max(convolution_times) if convolution_times else 0.0,
    }


def run_worker_subprocess(args: argparse.Namespace, case_name: str, poison_type: str, poison_frac: float) -> Dict[str, object]:
    out_dir = resolve_repo_path(args.out_dir)
    log_path = out_dir / f"{case_name}_onednn.log"
    summary_path = out_dir / f"{case_name}_worker_summary.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        args.dataset,
        "--data-root",
        str(resolve_repo_path(args.data_root)),
        "--batch-size",
        str(args.batch_size),
        "--img-size",
        str(args.img_size),
        "--steps",
        str(args.steps),
        "--poison-frac",
        str(poison_frac),
        "--seed",
        str(args.seed),
        "--normalize",
        args.normalize,
        "--device",
        args.device,
        "--out-dir",
        str(out_dir),
        "--worker-case",
        case_name,
        "--worker-summary",
        str(summary_path),
    ]
    if args.config:
        cmd.extend(["--config", args.config])

    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)

    worker_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    worker_summary.update(analyze_log(log_path, int(worker_summary["steps"])))
    worker_summary["poison_type"] = poison_type
    worker_summary["poison_frac"] = poison_frac
    worker_summary["onednn_log"] = str(log_path)
    summary_path.write_text(json.dumps(worker_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return worker_summary


def worker_main(args: argparse.Namespace) -> None:
    poison_type = "none" if args.worker_case == "clean" else "blurring"
    poison_frac = 1.0 if args.worker_case == "clean" else args.poison_frac
    result = run_case(
        case_name=args.worker_case,
        poison_type=poison_type,
        poison_frac=poison_frac,
        args=args,
    )
    if args.worker_summary:
        Path(args.worker_summary).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"worker_case": args.worker_case, "status": "ok", "steps": result["steps"]}, ensure_ascii=False))


def parent_main(args: argparse.Namespace) -> None:
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = run_worker_subprocess(args, "clean", "none", 1.0)
    blurring30 = run_worker_subprocess(args, "blurring30", "blurring", args.poison_frac)

    summary = {
        "config": {
            "dataset": args.dataset,
            "config": args.config,
            "data_root": str(resolve_repo_path(args.data_root)),
            "device": args.device,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "steps": args.steps,
            "normalize": args.normalize,
            "blurring_poison_frac": args.poison_frac,
            "seed": args.seed,
            "torch": torch.__version__,
        },
        "clean": clean,
        "blurring30": blurring30,
        "delta": {
            "reorder_exec_count": int(blurring30["reorder_exec_count"]) - int(clean["reorder_exec_count"]),
            "convolution_exec_count": int(blurring30["convolution_exec_count"]) - int(clean["convolution_exec_count"]),
            "reorder_per_step": float(blurring30["reorder_per_step"]) - float(clean["reorder_per_step"]),
            "reorder_exec_time_ms_sum": float(blurring30["reorder_exec_time_ms_sum"]) - float(clean["reorder_exec_time_ms_sum"]),
            "reorder_exec_time_ms_avg": float(blurring30["reorder_exec_time_ms_avg"]) - float(clean["reorder_exec_time_ms_avg"]),
            "convolution_exec_time_ms_sum": float(blurring30["convolution_exec_time_ms_sum"]) - float(clean["convolution_exec_time_ms_sum"]),
            "convolution_exec_time_ms_avg": float(blurring30["convolution_exec_time_ms_avg"]) - float(clean["convolution_exec_time_ms_avg"]),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"saved_clean_log={clean['onednn_log']}")
    print(f"saved_blurring30_log={blurring30['onednn_log']}")
    print(f"saved_summary={summary_path}")
    print(
        "reorder_exec_count "
        f"clean={clean['reorder_exec_count']} "
        f"blurring30={blurring30['reorder_exec_count']} "
        f"delta={summary['delta']['reorder_exec_count']}"
    )
    print(
        "reorder_exec_time_ms_sum "
        f"clean={clean['reorder_exec_time_ms_sum']:.3f} "
        f"blurring30={blurring30['reorder_exec_time_ms_sum']:.3f} "
        f"delta={summary['delta']['reorder_exec_time_ms_sum']:.3f}"
    )


def main() -> None:
    args = parse_args()
    if args.worker_case:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
