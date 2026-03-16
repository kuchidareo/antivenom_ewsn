

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
- Captures interval metrics from `perf stat`.
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
import re
import shutil
import subprocess
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
from torchvision import models
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# HF deps
from datasets import ClassLabel, Dataset as HFDataset, DatasetDict, Image as HFImage, load_dataset

# Image deps
from PIL import Image

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


def sample_n_dataset(ds: Dataset, n: int, seed: int) -> Dataset:
    if n <= 0:
        return torch.utils.data.Subset(ds, [])
    total = len(ds)
    if n >= total:
        return ds
    rng = random.Random(seed)
    idxs = list(range(total))
    rng.shuffle(idxs)
    idxs = idxs[:n]
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


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =============================
# Logger: 1 FPS system + training state to one CSV
# =============================


class MetricsCollector:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def collect(self) -> Dict[str, Any]:
        raise NotImplementedError


class PerfCollector(MetricsCollector):
    DEFAULT_EVENTS: Tuple[str, ...] = (
        "cycles",
        "instructions",
        "branch-misses",
        "cache-misses",
        "cache-references",
        "context-switches",
        "cpu-clock",
        "cpu-migrations",
        "major-faults",
        "minor-faults",
        "page-faults",
        "task-clock",
        "duration_time",
        "L1-dcache-load-misses",
        "L1-dcache-loads",
        "L1-icache-load-misses",
        "dTLB-load-misses",
        "dTLB-store-misses",
        "iTLB-load-misses",
    )

    def __init__(
        self,
        pid: int,
        fps: float,
        events: Optional[Sequence[str]] = None,
        perf_bin: Optional[str] = None,
    ) -> None:
        self.pid = pid
        self.events = tuple(events or self.DEFAULT_EVENTS)
        self.event_set = set(self.events)
        self.interval_ms = max(100, int(round(1000.0 / max(0.1, float(fps)))))
        self.perf_bin = perf_bin or shutil.which("perf") or "perf"
        self._lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stop_requested = False
        self._pending_ts: Optional[str] = None
        self._pending_row: Dict[str, Any] = {}
        self._latest: Dict[str, Any] = self._empty_row()

    def _empty_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "perf_status": "init",
            "perf_error": "",
            "perf_interval_ms": self.interval_ms,
            "perf_time_ms": None,
        }
        for event in self.events:
            row[self._column_name(event)] = None
        return row

    @staticmethod
    def _column_name(event: str) -> str:
        normalized = event.lower().replace("-", "_")
        return f"perf_{normalized}"

    @staticmethod
    def _parse_value(raw: str) -> Optional[float]:
        value = raw.strip()
        if not value:
            return None
        lowered = value.lower()
        if lowered in {"<not counted>", "<not supported>", "<not available>", "nan"}:
            return None
        value = value.replace(" ", "")
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
        return parsed

    def start(self) -> None:
        if self._reader is not None:
            return
        if shutil.which(self.perf_bin) is None and self.perf_bin == "perf":
            with self._lock:
                self._latest["perf_status"] = "error"
                self._latest["perf_error"] = "perf binary not found"
            return

        cmd = [
            self.perf_bin,
            "stat",
            "-x,",
            "-I",
            str(self.interval_ms),
            "-e",
            ",".join(self.events),
            "-p",
            str(self.pid),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            with self._lock:
                self._latest["perf_status"] = "error"
                self._latest["perf_error"] = f"{type(exc).__name__}: {exc}"
            return

        with self._lock:
            self._latest["perf_status"] = "starting"
            self._latest["perf_error"] = ""

        self._stop_requested = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def stop(self) -> None:
        self._stop_requested = True
        proc = self._proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        if self._reader is not None:
            self._reader.join(timeout=5.0)
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        self._reader = None
        self._proc = None

    def _publish_pending_row(self) -> None:
        if self._pending_ts is None:
            return
        row = self._empty_row()
        row.update(self._pending_row)
        row["perf_status"] = "ok"
        row["perf_error"] = ""
        row["perf_time_ms"] = self._parse_value(self._pending_ts)
        with self._lock:
            self._latest = row
        self._pending_ts = None
        self._pending_row = {}

    def _handle_error_line(self, line: str) -> None:
        msg = line.strip()
        if not msg:
            return
        with self._lock:
            self._latest["perf_status"] = "error"
            self._latest["perf_error"] = msg

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return

        for raw_line in proc.stderr:
            line = raw_line.strip()
            if not line:
                continue
            parsed = self._parse_perf_line(line)
            if parsed is None:
                if "not supported" in line.lower() or "permission" in line.lower() or "failed" in line.lower():
                    self._handle_error_line(line)
                continue
            ts_key, event, value = parsed
            if self._pending_ts is None:
                self._pending_ts = ts_key
            elif ts_key != self._pending_ts:
                self._publish_pending_row()
                self._pending_ts = ts_key
            self._pending_row[self._column_name(event)] = value

        self._publish_pending_row()
        ret = proc.poll()
        if ret not in (None, 0) and not self._stop_requested:
            with self._lock:
                self._latest["perf_status"] = "error"
                if not self._latest.get("perf_error"):
                    self._latest["perf_error"] = f"perf exited with code {ret}"

    def _parse_perf_line(self, line: str) -> Optional[Tuple[str, str, Optional[float]]]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            return None
        event_idx = None
        for idx, part in enumerate(parts):
            if part in self.event_set:
                event_idx = idx
                break
        if event_idx is None or event_idx < 1:
            return None

        ts_key = parts[0]
        event = parts[event_idx]
        value = self._parse_value(parts[1])
        return ts_key, event, value

    def collect(self) -> Dict[str, Any]:
        now = dt.datetime.now(dt.timezone.utc)
        with self._lock:
            row = dict(self._latest)
        row["ts_iso"] = now.isoformat()
        row["ts_unix"] = now.timestamp()
        return row


class TensorReferenceManager:
    def __init__(
        self,
        model: nn.Module,
        root_dir: Path,
        dataset: str,
        model_name: str,
        poison_type: str,
    ) -> None:
        self.model = model
        self.root_dir = root_dir
        self.dataset = dataset
        self.model_name = model_name
        self.poison_type = poison_type
        self.reference_dir = self.root_dir / "gradient_reference"
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        dataset_key = self._sanitize_name(dataset.replace("/", "_"))
        self.reference_path = self.reference_dir / f"{self._sanitize_name(model_name)}_{dataset_key}.pt"

        self.layer_modules: List[Tuple[str, nn.Module]] = []
        self.forward_column_map: Dict[str, str] = {}
        self.backward_column_map: Dict[str, str] = {}
        self._forward_handles: List[Any] = []
        self._backward_handles: List[Any] = []
        self._current_forward: Dict[str, torch.Tensor] = {}
        self._current_backward: Dict[str, torch.Tensor] = {}
        self.reference_forward: Dict[str, torch.Tensor] = {}
        self.reference_backward: Dict[str, torch.Tensor] = {}
        self.reference_loaded = False
        self.reference_saved_this_run = False
        self.reference_layer_names: List[str] = []

        for name, module in self.model.named_modules():
            params = list(module.parameters(recurse=False))
            if not name or not params:
                continue
            self.layer_modules.append((name, module))
            safe = self._sanitize_name(name)
            self.forward_column_map[name] = f"forward_rmse_{safe}"
            self.backward_column_map[name] = f"backward_rmse_{safe}"

        self.reference_layer_names = [name for name, _ in self.layer_modules]
        self._load_reference()
        self._register_hooks()

    @staticmethod
    def _sanitize_name(name: str) -> str:
        sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        return sanitized.strip("_").lower() or "layer"

    @staticmethod
    def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                found = TensorReferenceManager._first_tensor(item)
                if found is not None:
                    return found
        if isinstance(obj, dict):
            for item in obj.values():
                found = TensorReferenceManager._first_tensor(item)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _vectorize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu().reshape(-1).clone()

    @staticmethod
    def _rmse(current: torch.Tensor, reference: torch.Tensor) -> Optional[float]:
        if current.numel() != reference.numel():
            return None
        diff = current - reference
        return float(torch.sqrt(torch.mean(diff * diff)).item())

    def _load_reference(self) -> None:
        if not self.reference_path.exists():
            return
        payload = torch.load(self.reference_path, map_location="cpu")
        self.reference_forward = {
            str(name): tensor.detach().float().cpu().reshape(-1)
            for name, tensor in payload.get("forward", {}).items()
            if torch.is_tensor(tensor)
        }
        self.reference_backward = {
            str(name): tensor.detach().float().cpu().reshape(-1)
            for name, tensor in payload.get("backward", {}).items()
            if torch.is_tensor(tensor)
        }
        if self.reference_forward or self.reference_backward:
            self.reference_loaded = True

    def _register_hooks(self) -> None:
        for name, module in self.layer_modules:
            self._forward_handles.append(module.register_forward_hook(self._make_forward_hook(name)))
            self._backward_handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))

    def close(self) -> None:
        for handle in self._forward_handles:
            handle.remove()
        for handle in self._backward_handles:
            handle.remove()
        self._forward_handles = []
        self._backward_handles = []

    def _make_forward_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            tensor = self._first_tensor(output)
            if tensor is None:
                return
            self._current_forward[name] = self._vectorize_tensor(tensor)

        return hook

    def _make_backward_hook(self, name: str):
        def hook(module: nn.Module, grad_input: Tuple[Any, ...], grad_output: Tuple[Any, ...]) -> None:
            tensor = self._first_tensor(grad_output)
            if tensor is None:
                return
            self._current_backward[name] = self._vectorize_tensor(tensor)

        return hook

    def reset_step(self) -> None:
        self._current_forward = {}
        self._current_backward = {}

    def empty_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for name in self.reference_layer_names:
            state[self.forward_column_map[name]] = None
            state[self.backward_column_map[name]] = None
        return state

    def maybe_save_reference(self) -> bool:
        if self.reference_loaded or self.reference_saved_this_run:
            return False
        if self.poison_type not in {"none", "clean"}:
            return False
        if not self._current_forward or not self._current_backward:
            return False
        payload = {
            "meta": {
                "dataset": self.dataset,
                "model": self.model_name,
                "poison_type": self.poison_type,
            },
            "forward": {name: tensor.clone() for name, tensor in self._current_forward.items()},
            "backward": {name: tensor.clone() for name, tensor in self._current_backward.items()},
        }
        torch.save(payload, self.reference_path)
        self.reference_forward = {name: tensor.clone() for name, tensor in self._current_forward.items()}
        self.reference_backward = {name: tensor.clone() for name, tensor in self._current_backward.items()}
        self.reference_loaded = True
        self.reference_saved_this_run = True
        return True

    def compute_rmse_state(self) -> Dict[str, Any]:
        state = self.empty_state()
        if not self.reference_loaded:
            return state
        for name in self.reference_layer_names:
            current_forward = self._current_forward.get(name)
            reference_forward = self.reference_forward.get(name)
            if current_forward is not None and reference_forward is not None:
                state[self.forward_column_map[name]] = self._rmse(current_forward, reference_forward)

            current_backward = self._current_backward.get(name)
            reference_backward = self.reference_backward.get(name)
            if current_backward is not None and reference_backward is not None:
                state[self.backward_column_map[name]] = self._rmse(current_backward, reference_backward)
        return state


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
        for collector in self.collectors:
            collector.start()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        for collector in self.collectors:
            collector.stop()

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
    reference_manager: TensorReferenceManager,
    epoch: int,
    global_step_start: int,
    bg_cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> int:
    model.train()
    criterion = nn.CrossEntropyLoss()

    global_step = global_step_start
    burst_steps: List[int] = []
    procs: List[subprocess.Popen] = []

    if bg_cfg is not None and int(bg_cfg.get("bursts_per_epoch", 0)) > 0:
        n_steps = len(loader)
        k = min(int(bg_cfg["bursts_per_epoch"]), max(n_steps, 1))
        rng = random.Random(seed + 1000 * epoch)
        if n_steps > 0:
            burst_steps = sorted(rng.sample(range(n_steps), k=k))
        else:
            burst_steps = [0] * k
        logger.mark_event(
            json.dumps(
                {"bg_bursts": {"epoch": epoch, "steps": burst_steps, "cfg": bg_cfg}},
                ensure_ascii=False,
            )
        )

    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=True)
    for step, (x, y) in enumerate(pbar):
        reference_manager.reset_step()
        if bg_cfg is not None and burst_steps and step in burst_steps:
            cmd = [
                "bash",
                str(bg_cfg["script"]),
                "--on-sec",
                str(bg_cfg["on_sec"]),
                "--threads",
                str(bg_cfg["threads"]),
            ]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                procs.append(proc)
                logger.mark_event(
                    json.dumps(
                        {"bg_burst_start": {"epoch": epoch, "step": step, "cmd": cmd}},
                        ensure_ascii=False,
                    )
                )
            except Exception as e:
                logger.mark_event(
                    json.dumps(
                        {
                            "bg_burst_error": {
                                "epoch": epoch,
                                "step": step,
                                "error": f"{type(e).__name__}: {e}",
                                "cmd": cmd,
                            }
                        },
                        ensure_ascii=False,
                    )
                )
        x = x.to(device)
        y = y.to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        reference_saved = reference_manager.maybe_save_reference()
        rmse_state = reference_manager.compute_rmse_state()
        optim.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = float((preds == y).float().mean().item())

        # Update logger state (will be sampled at 1 FPS)
        lr = float(optim.param_groups[0].get("lr", 0.0))
        logger.update_state(
            epoch=epoch,
            step=step,
            global_step=global_step,
            loss=float(loss.item()),
            lr=lr,
            acc=acc,
            **rmse_state,
        )
        if reference_saved:
            logger.mark_event(
                json.dumps(
                    {
                        "gradient_reference_saved": {
                            "path": str(reference_manager.reference_path),
                            "epoch": epoch,
                            "step": step,
                        }
                    },
                    ensure_ascii=False,
                )
            )
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}", lr=f"{lr:.2e}")

        global_step += 1

    # Terminate any remaining background bursts at epoch end
    for proc in procs:
        try:
            ret = proc.poll()
            if ret is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=2.0)
                logger.mark_event(
                    json.dumps(
                        {"bg_burst_end": {"epoch": epoch, "status": "terminated"}},
                        ensure_ascii=False,
                    )
                )
            else:
                logger.mark_event(
                    json.dumps(
                        {"bg_burst_end": {"epoch": epoch, "status": "exited", "code": int(ret)}},
                        ensure_ascii=False,
                    )
                )
        except Exception:
            pass

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
    p.add_argument("--dataset", type=str, default="kuchidareo/small_trashnet")
    p.add_argument("--config", type=str, default=None, help="HF dataset config name (if needed)")
    p.add_argument("--data-root", type=str, default="data", help="Root containing poisoning dirs")
    p.add_argument(
        "--poison-type",
        type=str,
        default="none",
        choices=["none", "clean", "blurring", "occlusion", "label-flip"],
    )
    p.add_argument("--poison-frac", type=float, default=1.0, help="Fraction of poison samples to add (0..1)")
    p.add_argument(
        "--background-script",
        type=str,
        default=None,
        help="Background workload script (per-epoch bursts)",
    )
    p.add_argument(
        "--background-bursts-per-epoch",
        type=int,
        default=0,
        help="Number of background bursts per epoch (0 disables)",
    )
    p.add_argument(
        "--background-burst-on-sec",
        type=int,
        default=10,
        help="ON duration per background burst (seconds)",
    )
    p.add_argument(
        "--background-burst-threads",
        type=int,
        default=0,
        help="Threads for background bursts (0=auto)",
    )
    p.add_argument(
        "--background-mode",
        type=str,
        default="epoch",
        choices=["epoch", "once"],
        help="Background run mode: epoch=bursts per epoch, once=single run for full training",
    )
    p.add_argument(
        "--background-once-sec",
        type=int,
        default=0,
        help="Duration of background run when mode=once (0=do not stop)",
    )

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
    p.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=["simple_cnn", "mobilenet_v3_large"],
    )

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
    target_n = len(clean_train)
    if args.poison_type == "none":
        train_ds = clean_train
    elif args.poison_type == "clean":
        variant_dir = Path(args.data_root) / "clean"
        clean_disk = PoisonDiskDataset(variant_dir=variant_dir, split_name=train_split, tfm_cfg=tfm_cfg)
        clean_disk = subsample_dataset(clean_disk, frac=float(args.train_frac), seed=args.seed + 2)
        train_ds = sample_n_dataset(clean_disk, n=target_n, seed=args.seed + 3)
    else:
        variant_dir = Path(args.data_root) / args.poison_type
        poison_train = PoisonDiskDataset(variant_dir=variant_dir, split_name=train_split, tfm_cfg=tfm_cfg)
        poison_train = subsample_dataset(poison_train, frac=float(args.train_frac), seed=args.seed + 2)

        poison_k = int(round(target_n * float(args.poison_frac)))
        poison_k = max(0, min(poison_k, len(poison_train)))
        clean_k = max(0, target_n - poison_k)

        poison_sampled = sample_n_dataset(poison_train, n=poison_k, seed=args.seed)
        clean_sampled = sample_n_dataset(clean_train, n=clean_k, seed=args.seed + 1)
        train_ds = ConcatDataset([clean_sampled, poison_sampled])

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
    if args.model == "mobilenet_v3_large":
        model = MobileNetV3Large(num_classes=n_classes).to(device)
    else:
        model = SimpleCNN(n_classes=n_classes, img_size=int(img_size)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    repo_root = Path(__file__).resolve().parent.parent
    reference_manager = TensorReferenceManager(
        model=model,
        root_dir=repo_root,
        dataset=args.dataset,
        model_name=args.model,
        poison_type=args.poison_type,
    )

    # Logger
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_csv = log_dir / f"{_timestamp_name()}.csv"

    collectors: List[MetricsCollector] = [PerfCollector(pid=os.getpid(), fps=float(args.log_fps))]
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
        **reference_manager.empty_state(),
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
                    "model": args.model,
                    "gradient_reference_path": str(reference_manager.reference_path),
                    "gradient_reference_loaded": bool(reference_manager.reference_loaded),
                    "background_script": args.background_script,
                    "background_bursts_per_epoch": int(args.background_bursts_per_epoch),
                    "background_burst_on_sec": int(args.background_burst_on_sec),
                    "background_burst_threads": int(args.background_burst_threads),
                    "background_mode": args.background_mode,
                    "background_once_sec": int(args.background_once_sec),
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
        bg_cfg: Optional[Dict[str, Any]] = None
        bg_proc: Optional[subprocess.Popen] = None
        if args.background_script:
            bg_cfg = {
                "script": str(args.background_script),
                "bursts_per_epoch": int(args.background_bursts_per_epoch),
                "on_sec": int(args.background_burst_on_sec),
                "threads": int(args.background_burst_threads),
            }
            if args.background_mode == "once":
                cmd = [
                    "bash",
                    str(args.background_script),
                    "--on-sec",
                    str(args.background_once_sec),
                    "--threads",
                    str(args.background_burst_threads),
                ]
                try:
                    bg_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logger.mark_event(json.dumps({"bg_once_start": {"cmd": cmd}}, ensure_ascii=False))
                except Exception as e:
                    logger.mark_event(
                        json.dumps(
                            {"bg_once_error": {"error": f"{type(e).__name__}: {e}", "cmd": cmd}},
                            ensure_ascii=False,
                        )
                    )
        for epoch in range(int(args.epochs)):
            global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optim=optim,
                device=device,
                logger=logger,
                reference_manager=reference_manager,
                epoch=epoch,
                global_step_start=global_step,
                bg_cfg=bg_cfg,
                seed=int(args.seed),
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
        if bg_proc is not None:
            try:
                ret = bg_proc.poll()
                if ret is None:
                    bg_proc.terminate()
                    try:
                        bg_proc.wait(timeout=3.0)
                    except Exception:
                        bg_proc.kill()
                logger.mark_event(json.dumps({"bg_once_end": {"status": "stopped"}}, ensure_ascii=False))
            except Exception:
                pass
        reference_manager.close()
        logger.stop()


if __name__ == "__main__":
    main()
