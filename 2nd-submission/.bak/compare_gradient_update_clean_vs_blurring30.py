from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from datasets import DatasetDict, load_dataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import ml_running


DEFAULT_TAUS: Tuple[float, ...] = (1e-2, 1e-4, 1e-6, 1e-8, 1e-12, 1e-20, 1e-30, 1e-37)
LOGSPACE_MIN_EXP = -45
LOGSPACE_MAX_EXP = 2
LOGSPACE_STEPS = 80
STEP_START_PREFIX = "ANALYSIS_STEP_START|"
STEP_END_PREFIX = "ANALYSIS_STEP_END|"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare clean ML vs blurring30 ML with per-step activation/gradient/update "
            "statistics and oneDNN primitive timing logs."
        )
    )
    p.add_argument("--dataset", type=str, default="kuchidareo/small_trashnet")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-root", type=str, default="jit_compare_data")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", type=str, default="0.5", choices=["none", "0.5", "imagenet"])
    p.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "mobilenet_v3_large"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--clean-source",
        type=str,
        default="hf",
        choices=["hf", "disk"],
        help="hf follows ml_running.py exactly; disk uses data_root/clean for fully local runs.",
    )
    p.add_argument("--train-frac", type=float, default=1.0)
    p.add_argument("--test-frac", type=float, default=1.0)
    p.add_argument("--poison-frac", type=float, default=0.3)
    p.add_argument("--cases", nargs="+", default=["clean", "blurring30"], choices=["clean", "blurring30"])
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline"],
        choices=["baseline", "hard-zero", "floor-up"],
        help="Gradient intervention conditions to run for each case.",
    )
    p.add_argument("--eps", type=float, default=1e-20, help="Threshold for hard-zero and floor-up.")
    p.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    p.add_argument("--out-dir", type=str, default="gradient_update_compare_clean_vs_blurring30")
    p.add_argument(
        "--histogram-steps",
        type=int,
        default=0,
        help="Save log-binned histogram rows for the first N steps only.",
    )
    p.add_argument(
        "--onednn-verbose",
        type=str,
        default="profile_exec",
        help="Value passed to ONEDNN_VERBOSE for worker runs.",
    )
    p.add_argument(
        "--set-dnnl-verbose",
        action="store_true",
        help="Also set DNNL_VERBOSE to the same value for older builds.",
    )

    # Worker-only args.
    p.add_argument("--worker-run-id", type=str, default=None)
    p.add_argument("--worker-case", type=str, default=None)
    p.add_argument("--worker-condition", type=str, default=None)
    p.add_argument("--worker-stats-csv", type=str, default=None)
    p.add_argument("--worker-steps-csv", type=str, default=None)
    p.add_argument("--worker-hist-jsonl", type=str, default=None)
    p.add_argument("--worker-summary-json", type=str, default=None)
    return p.parse_args()


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_local_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    direct = Path.cwd() / path
    if direct.exists():
        return direct.resolve()
    under_script = script_dir() / path
    if under_script.exists():
        return under_script.resolve()
    repo_relative = script_dir().parent / path
    if repo_relative.exists():
        return repo_relative.resolve()
    if path.parts and path.parts[0] == script_dir().name:
        return (script_dir().parent / path).resolve()
    return under_script.resolve()


def field_name_for_tau(tau: float) -> str:
    return f"p_abs_lt_{tau:.0e}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_datasets(args: argparse.Namespace, case_name: str) -> Tuple[Dataset, int]:
    if args.clean_source == "disk":
        return build_datasets_disk_only(args, case_name)
    return build_datasets_hf_plus_disk(args, case_name)


def build_datasets_hf_plus_disk(args: argparse.Namespace, case_name: str) -> Tuple[Dataset, int]:
    ds_dict: DatasetDict = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)
    train_split, test_split = ml_running.build_splits(ds_dict)
    if test_split == "__split_from_train__":
        ds_dict = ds_dict[train_split].train_test_split(test_size=0.2, seed=args.seed)
        train_split, test_split = "train", "test"

    train_hf = ds_dict[train_split]
    image_col = ml_running._find_image_column(train_hf.features)
    label_col = ml_running._find_label_column(train_hf.features)
    class_names = ml_running._get_class_names(train_hf, label_col)
    if class_names is not None:
        n_classes = len(class_names)
    else:
        labels = {int(train_hf[i][label_col]) for i in range(min(len(train_hf), 5000))}
        n_classes = len(labels)

    data_root = resolve_local_path(args.data_root)
    tfm_cfg = ml_running.TransformConfig(img_size=int(args.img_size), normalize=args.normalize)
    clean_train = ml_running.CleanHFDataset(train_hf, image_col=image_col, label_col=label_col, tfm_cfg=tfm_cfg)
    clean_train = ml_running.subsample_dataset(clean_train, frac=float(args.train_frac), seed=args.seed)

    target_n = len(clean_train)
    if case_name == "clean":
        return clean_train, n_classes

    if case_name != "blurring30":
        raise ValueError(f"Unsupported case: {case_name}")

    variant_dir = data_root / "blurring"
    poison_train = ml_running.PoisonDiskDataset(variant_dir=variant_dir, split_name=train_split, tfm_cfg=tfm_cfg)
    poison_train = ml_running.subsample_dataset(poison_train, frac=float(args.train_frac), seed=args.seed + 2)

    poison_k = int(round(target_n * float(args.poison_frac)))
    poison_k = max(0, min(poison_k, len(poison_train)))
    clean_k = max(0, target_n - poison_k)
    poison_sampled = ml_running.sample_n_dataset(poison_train, n=poison_k, seed=args.seed)
    clean_sampled = ml_running.sample_n_dataset(clean_train, n=clean_k, seed=args.seed + 1)
    train_ds = ConcatDataset([clean_sampled, poison_sampled])
    return train_ds, n_classes


def build_datasets_disk_only(args: argparse.Namespace, case_name: str) -> Tuple[Dataset, int]:
    data_root = resolve_local_path(args.data_root)
    train_split = "train"
    tfm_cfg = ml_running.TransformConfig(img_size=int(args.img_size), normalize=args.normalize)
    clean_dir = data_root / "clean"
    clean_train = ml_running.PoisonDiskDataset(variant_dir=clean_dir, split_name=train_split, tfm_cfg=tfm_cfg)
    clean_train = ml_running.subsample_dataset(clean_train, frac=float(args.train_frac), seed=args.seed)
    n_classes = count_classes_from_metadata(clean_dir, split_name=train_split)

    target_n = len(clean_train)
    if case_name == "clean":
        return clean_train, n_classes

    if case_name != "blurring30":
        raise ValueError(f"Unsupported case: {case_name}")

    variant_dir = data_root / "blurring"
    poison_train = ml_running.PoisonDiskDataset(variant_dir=variant_dir, split_name=train_split, tfm_cfg=tfm_cfg)
    poison_train = ml_running.subsample_dataset(poison_train, frac=float(args.train_frac), seed=args.seed + 2)

    poison_k = int(round(target_n * float(args.poison_frac)))
    poison_k = max(0, min(poison_k, len(poison_train)))
    clean_k = max(0, target_n - poison_k)
    poison_sampled = ml_running.sample_n_dataset(poison_train, n=poison_k, seed=args.seed)
    clean_sampled = ml_running.sample_n_dataset(clean_train, n=clean_k, seed=args.seed + 1)
    train_ds = ConcatDataset([clean_sampled, poison_sampled])
    return train_ds, n_classes


def count_classes_from_metadata(variant_dir: Path, split_name: str) -> int:
    meta_path = variant_dir / "metadata.jsonl"
    labels = set()
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            file_name = str(record.get("file", ""))
            if not file_name.startswith(f"{split_name}_"):
                continue
            labels.add(int(record["label"]))
    if not labels:
        raise RuntimeError(f"No labels found in {meta_path} for split {split_name}")
    return len(labels)


def build_loader(args: argparse.Namespace, case_name: str) -> Tuple[DataLoader, int]:
    train_ds, n_classes = build_datasets(args, case_name)
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(torch.device(args.device).type == "cuda"),
        generator=generator,
    )
    return loader, n_classes


def build_model(args: argparse.Namespace, n_classes: int) -> nn.Module:
    if args.model == "mobilenet_v3_large":
        model = ml_running.MobileNetV3Large(num_classes=n_classes)
    else:
        model = ml_running.SimpleCNN(n_classes=n_classes, img_size=int(args.img_size))
    return model.to(torch.device(args.device))


def tensor_stats_record(
    *,
    run_id: str,
    case_name: str,
    condition: str,
    eps: float,
    step: int,
    name: str,
    kind: str,
    x: torch.Tensor,
    taus: Sequence[float],
) -> Dict[str, Any]:
    flat = x.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        raise ValueError("tensor_stats_record received an empty tensor")
    absx = flat.abs()
    tiny = torch.finfo(torch.float32).tiny
    record: Dict[str, Any] = {
        "run_id": run_id,
        "case": case_name,
        "condition": condition,
        "eps": float(eps),
        "step": int(step),
        "tensor_name": name,
        "kind": kind,
        "numel": int(flat.numel()),
        "mean_abs": float(absx.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "max_abs": float(absx.max().item()),
        "zero_ratio": float((flat == 0).float().mean().item()),
        "subnormal_ratio": float(((absx > 0) & (absx < tiny)).float().mean().item()),
    }
    for tau in taus:
        record[field_name_for_tau(tau)] = float((absx < tau).float().mean().item())
    return record


def histogram_payload(x: torch.Tensor) -> Optional[Dict[str, Any]]:
    flat = x.detach().float().reshape(-1).cpu()
    absx = flat.abs()
    nz = absx[absx > 0]
    if nz.numel() == 0:
        return None
    bins = torch.logspace(LOGSPACE_MIN_EXP, LOGSPACE_MAX_EXP, steps=LOGSPACE_STEPS)
    hist = torch.histc(
        nz.clamp(min=float(bins[0].item()), max=float(bins[-1].item())),
        bins=LOGSPACE_STEPS,
        min=float(bins[0].item()),
        max=float(bins[-1].item()),
    )
    return {
        "hist": [float(v) for v in hist.tolist()],
        "hist_bins": [float(v) for v in bins.tolist()],
    }


class TensorAnalyzer:
    def __init__(
        self,
        *,
        model: nn.Module,
        run_id: str,
        case_name: str,
        condition: str,
        eps: float,
        taus: Sequence[float],
        histogram_steps: int,
    ) -> None:
        self.model = model
        self.run_id = run_id
        self.case_name = case_name
        self.condition = condition
        self.eps = eps
        self.taus = tuple(float(t) for t in taus)
        self.histogram_steps = max(0, int(histogram_steps))
        self.current_step = -1
        self.stats_rows: List[Dict[str, Any]] = []
        self.hist_rows: List[Dict[str, Any]] = []
        self._handles: List[Any] = []
        self._register()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def _record(self, name: str, kind: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None or self.current_step < 0 or not torch.is_tensor(tensor):
            return
        self.stats_rows.append(
            tensor_stats_record(
                run_id=self.run_id,
                case_name=self.case_name,
                condition=self.condition,
                eps=self.eps,
                step=self.current_step,
                name=name,
                kind=kind,
                x=tensor,
                taus=self.taus,
            )
        )
        if self.current_step < self.histogram_steps:
            hist = histogram_payload(tensor)
            if hist is not None:
                hist_row = {
                    "run_id": self.run_id,
                    "case": self.case_name,
                    "condition": self.condition,
                    "eps": float(self.eps),
                    "step": int(self.current_step),
                    "tensor_name": name,
                    "kind": kind,
                }
                hist_row.update(hist)
                self.hist_rows.append(hist_row)

    def record_named_tensor(self, name: str, kind: str, tensor: Optional[torch.Tensor]) -> None:
        self._record(name=name, kind=kind, tensor=tensor)

    def _register(self) -> None:
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            self._handles.append(module.register_forward_hook(self._make_forward_hook(name)))
            self._handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))

    def _make_forward_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            tensor = ml_running.TensorReferenceManager._first_tensor(output)
            self._record(name=name, kind="activation", tensor=tensor)

        return hook

    def _make_backward_hook(self, name: str):
        def hook(module: nn.Module, grad_input: Tuple[Any, ...], grad_output: Tuple[Any, ...]) -> None:
            tensor = ml_running.TensorReferenceManager._first_tensor(grad_output)
            self._record(name=name, kind="grad_out", tensor=tensor)

        return hook


def hard_zero_(x: torch.Tensor, eps: float) -> torch.Tensor:
    mask = x.abs() < eps
    x[mask] = 0
    return x


def floor_up_(x: torch.Tensor, eps: float) -> torch.Tensor:
    absx = x.abs()
    mask = (absx > 0) & (absx < eps)
    x[mask] = x[mask].sign() * eps
    return x


def apply_intervention_(condition: str, params: Iterable[Tuple[str, nn.Parameter]], eps: float) -> None:
    if condition == "baseline":
        return
    for _, param in params:
        if param.grad is None:
            continue
        if condition == "hard-zero":
            hard_zero_(param.grad, eps)
        elif condition == "floor-up":
            floor_up_(param.grad, eps)
        else:
            raise ValueError(f"Unsupported condition: {condition}")


def print_step_marker(prefix: str, run_id: str, step: int) -> None:
    print(f"{prefix}run_id={run_id}|step={step}", flush=True)


def parse_step_marker(line: str, prefix: str) -> Optional[Tuple[str, int]]:
    if not line.startswith(prefix):
        return None
    payload = line[len(prefix) :].strip()
    parts = dict(part.split("=", 1) for part in payload.split("|") if "=" in part)
    run_id = parts.get("run_id")
    step_text = parts.get("step")
    if run_id is None or step_text is None:
        return None
    return run_id, int(step_text)


def parse_onednn_exec_line(line: str) -> Optional[Dict[str, Any]]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 8:
        return None
    if parts[0] != "onednn_verbose":
        return None
    if parts[1] == "v1":
        if len(parts) < 9 or parts[2] != "primitive" or parts[3] != "exec":
            return None
        kind_idx = 5
        impl_idx = 6
    else:
        if parts[1] != "primitive" or parts[2] != "exec":
            return None
        kind_idx = 4
        impl_idx = 5
    try:
        exec_time_ms = float(parts[-1])
    except ValueError:
        return None
    return {
        "primitive_kind": parts[kind_idx],
        "primitive_impl": parts[impl_idx],
        "raw_line": line.rstrip("\n"),
        "exec_time_ms": exec_time_ms,
    }


def parse_onednn_log(log_path: Path, run_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    primitive_rows: List[Dict[str, Any]] = []
    current_step: Optional[int] = None
    seen_step_run: Optional[str] = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            start_marker = parse_step_marker(raw_line, STEP_START_PREFIX)
            if start_marker is not None:
                seen_step_run, current_step = start_marker
                continue
            end_marker = parse_step_marker(raw_line, STEP_END_PREFIX)
            if end_marker is not None:
                current_step = None
                seen_step_run = None
                continue
            parsed = parse_onednn_exec_line(raw_line)
            if parsed is None or current_step is None or seen_step_run != run_id:
                continue
            row = {
                "run_id": run_id,
                "step": int(current_step),
                "primitive_kind": parsed["primitive_kind"],
                "primitive_impl": parsed["primitive_impl"],
                "exec_time_ms": parsed["exec_time_ms"],
                "raw_line": parsed["raw_line"],
            }
            primitive_rows.append(row)

    by_kind = defaultdict(lambda: {"count": 0, "sum_exec_time_ms": 0.0})
    for row in primitive_rows:
        bucket = by_kind[row["primitive_kind"]]
        bucket["count"] += 1
        bucket["sum_exec_time_ms"] += float(row["exec_time_ms"])

    summary = {
        "run_id": run_id,
        "primitive_count_total": len(primitive_rows),
        "primitive_kind_summary": {
            kind: {
                "count": int(values["count"]),
                "sum_exec_time_ms": float(values["sum_exec_time_ms"]),
                "avg_exec_time_ms": (
                    float(values["sum_exec_time_ms"]) / float(values["count"]) if values["count"] else 0.0
                ),
            }
            for kind, values in sorted(by_kind.items())
        },
    }
    return primitive_rows, summary


def run_worker(args: argparse.Namespace) -> None:
    if not args.worker_run_id or not args.worker_case or not args.worker_condition:
        raise ValueError("worker args are required in worker mode")

    ml_running.set_seed(int(args.seed))
    loader, n_classes = build_loader(args, args.worker_case)
    model = build_model(args, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    criterion = nn.CrossEntropyLoss()
    analyzer = TensorAnalyzer(
        model=model,
        run_id=args.worker_run_id,
        case_name=args.worker_case,
        condition=args.worker_condition,
        eps=float(args.eps),
        taus=args.taus,
        histogram_steps=int(args.histogram_steps),
    )

    stats_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    device = torch.device(args.device)
    model.train()

    try:
        for step, (x, y) in enumerate(loader):
            if step >= int(args.steps):
                break

            analyzer.set_step(step)
            print_step_marker(STEP_START_PREFIX, args.worker_run_id, step)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            named_params = list(model.named_parameters())
            for name, param in named_params:
                if param.grad is not None:
                    analyzer.record_named_tensor(name=name, kind="weight_grad_raw", tensor=param.grad)

            apply_intervention_(args.worker_condition, named_params, float(args.eps))

            before_step = {
                name: param.detach().clone()
                for name, param in named_params
                if param.grad is not None
            }
            for name, param in named_params:
                if param.grad is not None:
                    analyzer.record_named_tensor(name=name, kind="weight_grad", tensor=param.grad)

            optimizer.step()

            for name, param in named_params:
                if name in before_step:
                    analyzer.record_named_tensor(name=name, kind="update", tensor=param.detach() - before_step[name])

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = float((preds == y).float().mean().item())

            step_rows.append(
                {
                    "run_id": args.worker_run_id,
                    "case": args.worker_case,
                    "condition": args.worker_condition,
                    "eps": float(args.eps),
                    "step": int(step),
                    "loss": float(loss.item()),
                    "acc": acc,
                    "batch_size": int(y.shape[0]),
                }
            )
            print_step_marker(STEP_END_PREFIX, args.worker_run_id, step)

        stats_rows.extend(analyzer.stats_rows)
    finally:
        analyzer.close()

    if args.worker_stats_csv:
        write_csv(Path(args.worker_stats_csv), stats_rows)
    if args.worker_steps_csv:
        write_csv(Path(args.worker_steps_csv), step_rows)
    if args.worker_hist_jsonl:
        hist_path = Path(args.worker_hist_jsonl)
        ensure_parent(hist_path)
        with open(hist_path, "w", encoding="utf-8") as f:
            for row in analyzer.hist_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.worker_summary_json:
        summary = {
            "run_id": args.worker_run_id,
            "case": args.worker_case,
            "condition": args.worker_condition,
            "eps": float(args.eps),
            "steps_recorded": len(step_rows),
            "stats_row_count": len(stats_rows),
            "hist_row_count": len(analyzer.hist_rows),
        }
        write_json(Path(args.worker_summary_json), summary)


def run_parent(args: argparse.Namespace) -> None:
    out_dir = resolve_local_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []

    for case_name in args.cases:
        for condition in args.conditions:
            run_suffix = "" if condition == "baseline" else f"_eps{args.eps:.0e}"
            run_id = f"{case_name}_{condition}{run_suffix}".replace("+", "")
            stats_csv = out_dir / f"{run_id}.tensor_stats.csv"
            steps_csv = out_dir / f"{run_id}.steps.csv"
            hist_jsonl = out_dir / f"{run_id}.histograms.jsonl"
            worker_summary_json = out_dir / f"{run_id}.worker_summary.json"
            onednn_log = out_dir / f"{run_id}.onednn.log"
            primitive_csv = out_dir / f"{run_id}.primitive_exec.csv"
            primitive_summary_json = out_dir / f"{run_id}.primitive_summary.json"

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset",
                args.dataset,
                "--data-root",
                str(resolve_local_path(args.data_root)),
                "--batch-size",
                str(args.batch_size),
                "--img-size",
                str(args.img_size),
                "--steps",
                str(args.steps),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--normalize",
                args.normalize,
                "--model",
                args.model,
                "--device",
                args.device,
                "--clean-source",
                args.clean_source,
                "--train-frac",
                str(args.train_frac),
                "--test-frac",
                str(args.test_frac),
                "--poison-frac",
                str(args.poison_frac),
                "--eps",
                str(args.eps),
                "--histogram-steps",
                str(args.histogram_steps),
                "--worker-run-id",
                run_id,
                "--worker-case",
                case_name,
                "--worker-condition",
                condition,
                "--worker-stats-csv",
                str(stats_csv),
                "--worker-steps-csv",
                str(steps_csv),
                "--worker-hist-jsonl",
                str(hist_jsonl),
                "--worker-summary-json",
                str(worker_summary_json),
            ]
            if args.config:
                cmd.extend(["--config", args.config])
            cmd.append("--taus")
            cmd.extend([str(tau) for tau in args.taus])

            env = os.environ.copy()
            env["ONEDNN_VERBOSE"] = str(args.onednn_verbose)
            if args.set_dnnl_verbose:
                env["DNNL_VERBOSE"] = str(args.onednn_verbose)

            with open(onednn_log, "w", encoding="utf-8") as log_file:
                subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT, env=env)

            primitive_rows, primitive_summary = parse_onednn_log(onednn_log, run_id)
            write_csv(primitive_csv, primitive_rows)
            write_json(primitive_summary_json, primitive_summary)

            run_summary = {
                "run_id": run_id,
                "case": case_name,
                "condition": condition,
                "eps": float(args.eps),
                "stats_csv": str(stats_csv),
                "steps_csv": str(steps_csv),
                "hist_jsonl": str(hist_jsonl),
                "onednn_log": str(onednn_log),
                "primitive_csv": str(primitive_csv),
                "primitive_summary_json": str(primitive_summary_json),
                "worker_summary_json": str(worker_summary_json),
                "primitive_count_total": primitive_summary["primitive_count_total"],
                "primitive_kind_summary": primitive_summary["primitive_kind_summary"],
            }
            runs.append(run_summary)

    combined_summary = {
        "dataset": args.dataset,
        "data_root": str(resolve_local_path(args.data_root)),
        "model": args.model,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "img_size": int(args.img_size),
        "steps": int(args.steps),
        "normalize": args.normalize,
        "poison_frac": float(args.poison_frac),
        "seed": int(args.seed),
        "taus": [float(t) for t in args.taus],
        "conditions": list(args.conditions),
        "cases": list(args.cases),
        "runs": runs,
    }
    write_json(out_dir / "summary.json", combined_summary)


def main() -> None:
    args = parse_args()
    if args.worker_run_id:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
