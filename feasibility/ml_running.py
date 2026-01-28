

"""Train a simple CNN on TrashNet (HF) with optional poisoning variants.

Data
- Clean data is loaded directly from Hugging Face dataset `kuchidareo/trashnet_small`.
- Poisoned data is loaded from the output of `data_preparing.py`, under:
    <data_root>/blurring/
    <data_root>/occlusion/
    <data_root>/label-flip/
  Each has JPEGs + metadata.jsonl.

Poisoning usage
- Choose one of: none | blurring | occlusion | label-flip
- Training dataset becomes: clean_train + sampled(poison_train, poison_frac)
  (i.e., concatenate clean + poison samples)

Logging (important)
- Writes ONE timestamped CSV: <log_dir>/<YYYYmmdd_HHMMSS>.csv
- Logs at 1 FPS by default.
- Captures system metrics via psutil (CPU, mem, process usage, IO, net).
- Best-effort CPU temperature (if available).
- Training loop marks train_start/train_end events and only logs while training
  (logger disabled during evaluation by default).

This file is designed to be extended heavily later.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Torch deps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# HF deps
from datasets import ClassLabel, Dataset as HFDataset, DatasetDict, Image as HFImage, load_dataset

# Image deps
from PIL import Image

# System metrics
import psutil
from tqdm import tqdm


# =============================
# Utilities: dataset schema detection
# =============================


def _find_image_column(features: Dict[str, Any]) -> str:
    if "image" in features and isinstance(features["image"], HFImage):
        return "image"
    for k, v in features.items():
        if isinstance(v, HFImage):
            return k
    raise ValueError("Could not find an image column (datasets.Image).")


def _find_label_column(features: Dict[str, Any]) -> str:
    if "label" in features:
        return "label"
    for k, v in features.items():
        if isinstance(v, ClassLabel):
            return k
    for candidate in ("labels", "category", "class", "target"):
        if candidate in features:
            return candidate
    raise ValueError("Could not find a label column.")


def _get_class_names(ds: HFDataset, label_col: str) -> Optional[List[str]]:
    feat = ds.features.get(label_col)
    if isinstance(feat, ClassLabel):
        return list(feat.names)
    return None


# =============================
# Transforms (minimal, no torchvision dependency)
# =============================


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # Convert PIL RGB to float tensor in [0,1], shape (C,H,W)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def resize_pil(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.BILINEAR)


def normalize_tensor(x: torch.Tensor, mode: str) -> torch.Tensor:
    # x in [0,1]
    if mode == "none":
        return x
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)[:, None, None]
        return (x - mean) / std
    # default: center to [-1,1] with mean=0.5 std=0.5
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device)[:, None, None]
    std = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device)[:, None, None]
    return (x - mean) / std


@dataclass(frozen=True)
class TransformConfig:
    img_size: int
    normalize: str = "0.5"  # one of: none | 0.5 | imagenet


def apply_transform(img: Image.Image, cfg: TransformConfig) -> torch.Tensor:
    img = img.convert("RGB")
    img = resize_pil(img, cfg.img_size)
    x = pil_to_tensor(img)
    x = normalize_tensor(x, "imagenet" if cfg.normalize == "imagenet" else ("none" if cfg.normalize == "none" else "0.5"))
    return x


# =============================
# Datasets
# =============================


class CleanHFDataset(Dataset):
    def __init__(
        self,
        ds: HFDataset,
        image_col: str,
        label_col: str,
        tfm_cfg: TransformConfig,
    ) -> None:
        self.ds = ds
        self.image_col = image_col
        self.label_col = label_col
        self.tfm_cfg = tfm_cfg

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ex = self.ds[idx]
        img_any = ex[self.image_col]
        if isinstance(img_any, dict) and "image" in img_any:
            img = img_any["image"]
        else:
            img = img_any
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(img)}")

        x = apply_transform(img, self.tfm_cfg)
        y = int(ex[self.label_col])
        return x, y


class PoisonDiskDataset(Dataset):
    """Loads poisoned samples from <variant>/metadata.jsonl and JPEGs.

    Notes
    - Filenames created by data_preparing.py include split prefix: <split>_<hash>.jpg
    - We filter rows by that prefix.
    """

    def __init__(
        self,
        variant_dir: Path,
        split_name: str,
        tfm_cfg: TransformConfig,
    ) -> None:
        self.variant_dir = variant_dir
        self.split_name = split_name
        self.tfm_cfg = tfm_cfg

        meta_path = self.variant_dir / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.jsonl not found: {meta_path}")

        items: List[Tuple[str, int]] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                fname = rec["file"]
                if not fname.startswith(f"{split_name}_"):
                    continue
                items.append((fname, int(rec["label"])))

        if not items:
            raise RuntimeError(f"No items found for split='{split_name}' in {meta_path}")

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fname, y = self.items[idx]
        img_path = self.variant_dir / fname
        img = Image.open(img_path)
        x = apply_transform(img, self.tfm_cfg)
        return x, y


def sample_poison_dataset(poison_ds: Dataset, poison_frac: float, seed: int) -> Dataset:
    if poison_frac <= 0:
        return torch.utils.data.Subset(poison_ds, [])
    if poison_frac >= 1:
        return poison_ds

    n = len(poison_ds)
    k = max(1, int(n * poison_frac))
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    idxs = idxs[:k]
    return torch.utils.data.Subset(poison_ds, idxs)


def subsample_dataset(ds: Dataset, frac: float, seed: int) -> Dataset:
    if frac >= 1.0:
        return ds
    if frac <= 0:
        return torch.utils.data.Subset(ds, [])
    n = len(ds)
    k = max(1, int(n * frac))
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    idxs = idxs[:k]
    return torch.utils.data.Subset(ds, idxs)


# =============================
# Model: simple CNN (3 conv, 3 fc)
# =============================


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int, img_size: int) -> None:
        super().__init__()

        # 3 conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Determine feature dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feats = self._forward_features(dummy)
            feat_dim = int(feats.view(1, -1).shape[1])

        # 3 fc layers
        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =============================
# Logger: 1 FPS system + training state to one CSV
# =============================


class MetricsCollector:
    def collect(self) -> Dict[str, Any]:
        raise NotImplementedError


class SystemCollector(MetricsCollector):
    def __init__(self, proc: psutil.Process) -> None:
        self.proc = proc

        # warm-up for cpu_percent
        psutil.cpu_percent(interval=None)
        self.proc.cpu_percent(interval=None)

    @staticmethod
    def _cpu_temperature_c() -> Optional[float]:
        # Best-effort: Linux via psutil or /sys; other OS may return None.
        try:
            temps = psutil.sensors_temperatures(fahrenheit=False)  # type: ignore[attr-defined]
            if temps:
                # Prefer common keys
                for key in ("coretemp", "k10temp", "cpu_thermal", "soc_thermal"):
                    if key in temps and temps[key]:
                        vals = [t.current for t in temps[key] if t.current is not None]
                        if vals:
                            return float(np.mean(vals))
                # Fallback: first available
                for entries in temps.values():
                    vals = [t.current for t in entries if t.current is not None]
                    if vals:
                        return float(np.mean(vals))
        except Exception:
            pass

        # /sys fallback
        try:
            base = Path("/sys/class/thermal")
            if base.exists():
                temps = []
                for p in base.glob("thermal_zone*/temp"):
                    try:
                        raw = p.read_text().strip()
                        if raw:
                            v = float(raw)
                            # Often millidegrees
                            if v > 1000:
                                v = v / 1000.0
                            temps.append(v)
                    except Exception:
                        continue
                if temps:
                    return float(np.mean(temps))
        except Exception:
            pass

        return None

    def collect(self) -> Dict[str, Any]:
        now = dt.datetime.now(dt.timezone.utc)

        # CPU
        cpu_total = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

        # Memory
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        # Process
        p_cpu = self.proc.cpu_percent(interval=None)
        try:
            p_mem = self.proc.memory_info().rss
        except Exception:
            p_mem = None

        temp_c = self._cpu_temperature_c()

        out: Dict[str, Any] = {
            "ts_iso": now.isoformat(),
            "ts_unix": now.timestamp(),
            "cpu_percent": cpu_total,
            "cpu_per_core": json.dumps(cpu_per_core),
            "mem_percent": vm.percent,
            "swap_percent": sm.percent,
            "proc_cpu_percent": p_cpu,
            "proc_rss": p_mem,
            "cpu_temp_c": temp_c,
        }

        return out


class CSVRunLogger:
    """One-file CSV logger with a 1 FPS background loop.

    - When disabled, does not write periodic rows.
    - You can still emit event rows via mark_event().
    - Training loop can update `state` (epoch/step/loss/etc.) and it will be included.

    Extensible: add more collectors later.
    """

    def __init__(self, out_csv: Path, fps: float = 1.0, collectors: Optional[List[MetricsCollector]] = None) -> None:
        self.out_csv = out_csv
        self.fps = fps
        self.collectors = collectors or []

        self._enabled = False
        self._stop = False
        self._thread: Optional[threading.Thread] = None

        self._state_lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "phase": "idle",
            "train_active": 0,
            "event": "",
            "epoch": None,
            "step": None,
            "global_step": None,
            "loss": None,
            "lr": None,
            "acc": None,
        }

        self._header_written = False
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

    def update_state(self, **kwargs: Any) -> None:
        with self._state_lock:
            self._state.update(kwargs)

    def enable(self) -> None:
        self._enabled = True
        self.update_state(train_active=1, phase="train")

    def disable(self) -> None:
        self._enabled = False
        self.update_state(train_active=0, phase="idle")

    def mark_event(self, name: str) -> None:
        # Emit an immediate row containing an event marker.
        self._write_row(event=name)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    def _loop(self) -> None:
        period = 1.0 / max(0.1, float(self.fps))
        next_t = time.time()
        while not self._stop:
            now = time.time()
            if now >= next_t:
                next_t = now + period
                if self._enabled:
                    self._write_row()
            time.sleep(0.01)

    def _collect_all(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for c in self.collectors:
            try:
                row.update(c.collect())
            except Exception as e:
                row["collector_error"] = f"{type(e).__name__}: {e}"
        with self._state_lock:
            row.update(self._state)
        return row

    def _write_row(self, **override: Any) -> None:
        row = self._collect_all()
        if override:
            row.update(override)

        # Write header lazily (union of keys encountered).
        # To keep it extensible, we maintain a growing header by rewriting nothing;
        # instead, we write with a stable superset header computed at first write.
        # If you add new keys later, extend this by storing header state.
        if not self._header_written:
            self._header = sorted(row.keys())  # type: ignore[attr-defined]
            with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._header, extrasaction="ignore")
                w.writeheader()
                w.writerow({k: row.get(k, "") for k in self._header})
            self._header_written = True
            return

        # Append
        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._header, extrasaction="ignore")  # type: ignore[attr-defined]
            w.writerow({k: row.get(k, "") for k in self._header})


# =============================
# Training / eval
# =============================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * int(y.shape[0])
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.shape[0])

    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    logger: CSVRunLogger,
    epoch: int,
    global_step_start: int,
) -> int:
    model.train()
    criterion = nn.CrossEntropyLoss()

    global_step = global_step_start

    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=True)
    for step, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = float((preds == y).float().mean().item())

        # Update logger state (will be sampled at 1 FPS)
        lr = float(optim.param_groups[0].get("lr", 0.0))
        logger.update_state(epoch=epoch, step=step, global_step=global_step, loss=float(loss.item()), lr=lr, acc=acc)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}", lr=f"{lr:.2e}")

        global_step += 1

    return global_step


# =============================
# Main
# =============================


def _timestamp_name() -> str:
    # Local time
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_img_size(ds: HFDataset, image_col: str, sample_n: int = 64) -> Optional[int]:
    # Try to infer a common square size from a small sample.
    sizes: List[int] = []
    n = min(len(ds), sample_n)
    for i in range(n):
        ex = ds[i]
        img_any = ex[image_col]
        if isinstance(img_any, dict) and "image" in img_any:
            img = img_any["image"]
        else:
            img = img_any
        if isinstance(img, Image.Image):
            w, h = img.size
            # If already square, prefer that.
            if w == h:
                sizes.append(w)
            else:
                sizes.append(int(round((w + h) / 2.0)))
    if not sizes:
        return None
    # mode
    vals, counts = np.unique(np.array(sizes), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def build_splits(ds_dict: DatasetDict) -> Tuple[str, str]:
    keys = list(ds_dict.keys())
    if "train" in keys and "test" in keys:
        return "train", "test"
    if "train" in keys and "validation" in keys:
        return "train", "validation"
    # fallback: first two splits
    if len(keys) >= 2:
        return keys[0], keys[1]
    # last resort: split train
    return keys[0], "__split_from_train__"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--dataset", type=str, default="kuchidareo/trashnet_small")
    p.add_argument("--config", type=str, default=None, help="HF dataset config name (if needed)")
    p.add_argument("--data-root", type=str, default="data", help="Root containing poisoning dirs")
    p.add_argument("--poison-type", type=str, default="none", choices=["none", "blurring", "occlusion", "label-flip"])
    p.add_argument("--poison-frac", type=float, default=1.0, help="Fraction of poison samples to add (0..1)")

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Image
    p.add_argument("--img-size", type=int, default=None, help="If not set, infer from dataset sample")
    p.add_argument("--normalize", type=str, default="0.5", choices=["none", "0.5", "imagenet"])

    # Logging
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--log-fps", type=float, default=1.0)

    # Subsampling
    p.add_argument("--train-frac", type=float, default=1.0, help="Fraction of train split to use (0..1)")
    p.add_argument("--test-frac", type=float, default=1.0, help="Fraction of test split to use (0..1)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    # Load HF dataset (clean)
    ds_dict = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)
    train_split, test_split = build_splits(ds_dict)

    if test_split == "__split_from_train__":
        # Split train into train/test
        ds_dict = ds_dict[train_split].train_test_split(test_size=0.2, seed=args.seed)
        train_split, test_split = "train", "test"

    train_hf = ds_dict[train_split]
    test_hf = ds_dict[test_split]

    image_col = _find_image_column(train_hf.features)
    label_col = _find_label_column(train_hf.features)
    class_names = _get_class_names(train_hf, label_col)

    # Determine number of classes
    if class_names is not None:
        n_classes = len(class_names)
    else:
        # best-effort
        labels = set(int(train_hf[i][label_col]) for i in range(min(len(train_hf), 5000)))
        n_classes = len(labels)

    # Image size
    img_size = args.img_size
    if img_size is None:
        inferred = infer_img_size(train_hf, image_col)
        img_size = inferred if inferred is not None else 224

    tfm_cfg = TransformConfig(img_size=int(img_size), normalize=args.normalize)

    # Build datasets
    clean_train = CleanHFDataset(train_hf, image_col=image_col, label_col=label_col, tfm_cfg=tfm_cfg)
    clean_test = CleanHFDataset(test_hf, image_col=image_col, label_col=label_col, tfm_cfg=tfm_cfg)

    # Optional subsampling (after inferring classes/img size)
    clean_train = subsample_dataset(clean_train, frac=float(args.train_frac), seed=args.seed)
    clean_test = subsample_dataset(clean_test, frac=float(args.test_frac), seed=args.seed + 1)

    train_ds: Dataset
    if args.poison_type == "none":
        train_ds = clean_train
    else:
        variant_dir = Path(args.data_root) / args.poison_type
        poison_train = PoisonDiskDataset(variant_dir=variant_dir, split_name=train_split, tfm_cfg=tfm_cfg)
        poison_train = subsample_dataset(poison_train, frac=float(args.train_frac), seed=args.seed + 2)
        poison_sampled = sample_poison_dataset(poison_train, poison_frac=float(args.poison_frac), seed=args.seed)
        train_ds = ConcatDataset([clean_train, poison_sampled])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        clean_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = SimpleCNN(n_classes=n_classes, img_size=int(img_size)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # Logger
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_csv = log_dir / f"{_timestamp_name()}.csv"

    proc = psutil.Process(os.getpid())
    collectors: List[MetricsCollector] = [SystemCollector(proc)]
    logger = CSVRunLogger(out_csv=out_csv, fps=float(args.log_fps), collectors=collectors)

    # Record static run info once (as an event row)
    logger.start()
    logger.update_state(
        phase="init",
        train_active=0,
        epoch=None,
        step=None,
        global_step=None,
        loss=None,
        lr=float(args.lr),
        acc=None,
    )
    logger.mark_event(
        json.dumps(
            {
                "run_info": {
                    "dataset": args.dataset,
                    "config": args.config,
                    "train_split": train_split,
                    "test_split": test_split,
                    "poison_type": args.poison_type,
                    "poison_frac": float(args.poison_frac),
                    "img_size": int(img_size),
                    "normalize": args.normalize,
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "seed": int(args.seed),
                    "train_frac": float(args.train_frac),
                    "test_frac": float(args.test_frac),
                    "device": str(device),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                }
            },
            ensure_ascii=False,
        )
    )

    print("=== Run ===")
    print(f"dataset={args.dataset} split=({train_split},{test_split}) poison={args.poison_type} frac={args.poison_frac}")
    print(f"img_size={img_size} normalize={args.normalize} n_classes={n_classes} device={device}")
    print(f"log_csv={out_csv}")

    # Training
    global_step = 0
    logger.enable()
    logger.mark_event("train_start")

    try:
        for epoch in range(int(args.epochs)):
            global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optim=optim,
                device=device,
                logger=logger,
                epoch=epoch,
                global_step_start=global_step,
            )

            # Mark end of each epoch
            logger.mark_event(f"epoch_end_{epoch}")

        # Mark end of full training
        logger.mark_event("train_end")
        logger.disable()

        # Evaluation (logger disabled by default)
        test_loss, test_acc = evaluate(model, test_loader, device=device)
        print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        # Record evaluation summary as an event row (does not resume 1FPS logging)
        logger.mark_event(json.dumps({"eval": {"loss": test_loss, "acc": test_acc}}, ensure_ascii=False))

    finally:
        logger.stop()


if __name__ == "__main__":
    main()
