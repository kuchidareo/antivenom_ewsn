

"""Train a simple CNN on TrashNet (HF) with optional poisoning variants.

Data
- Clean data is loaded directly from Hugging Face dataset `kuchidareo/trashnet_small`.
- Poisoned data is loaded from the output of `data_preparing.py`, under:
    <data_root>/blurring/
    <data_root>/occlusion/
    <data_root>/label-flip/
  Each has JPEGs + metadata.jsonl.

Poisoning usage
- Choose one of: none | clean | blurring | occlusion | label-flip
- `none` loads clean HF training data directly.
- Other variants load the prepared directory under <data_root>/<variant>/ directly.
- `--poison-frac` is used only to validate prepared metadata when available.

Logging (important)
- Writes ONE timestamped CSV: <log_dir>/<YYYYmmdd_HHMMSS>.csv
- Logs at 1 FPS by default.
- Captures system metrics via psutil (CPU, mem, process usage, IO).
- Captures interval metrics from `perf stat` unless disabled.
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
import math
import os
import platform
import random
import re
import shutil
import subprocess
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Torch deps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


def save_model_grads(model: nn.Module, save_path: Path | str) -> None:
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().cpu().clone()
    torch.save(grads, save_path)


def _append_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


@dataclass(frozen=True)
class VogProbeSample:
    sample_id: int
    dataset_index: int
    x: torch.Tensor
    y: int


def build_vog_probe_samples(ds: Dataset, num_samples: int, seed: int) -> List[VogProbeSample]:
    n = len(ds)
    if n <= 0:
        raise ValueError("Cannot build a VoG probe set from an empty dataset.")

    k = min(max(1, int(num_samples)), n)
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    idxs = sorted(idxs[:k])

    samples: List[VogProbeSample] = []
    for sample_id, dataset_index in enumerate(idxs):
        item = ds[dataset_index]
        if not isinstance(item, tuple) or len(item) < 2:
            raise TypeError(f"Expected dataset item to be (x, y), got {type(item)}")
        x, y = item[0], item[1]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected VoG probe input tensor, got {type(x)}")
        samples.append(VogProbeSample(sample_id=sample_id, dataset_index=int(dataset_index), x=x.cpu(), y=int(y)))
    return samples


class VogGradientObserver:
    """Collects paper-style fixed-input gradient trajectories for VoG.

    The raw files allow exact post-run VoG:
        sum_j Var_t(g_j)
    for the same sample across observations. VoG uses input gradients dL/dx;
    parameter/layer gradients are intentionally not saved here.
    """

    SAMPLE_FIELDS: Tuple[str, ...] = ("sample_id", "dataset_index", "target")
    OBS_FIELDS: Tuple[str, ...] = (
        "observation_id",
        "epoch",
        "step",
        "global_step",
        "sample_id",
        "dataset_index",
        "target",
        "pred",
        "loss",
        "scope",
        "layer",
        "grad_norm",
        "energy",
        "num_grad_values",
        "grad_file",
    )
    SUMMARY_FIELDS: Tuple[str, ...] = (
        "sample_id",
        "dataset_index",
        "target",
        "scope",
        "layer",
        "observation_count",
        "first_global_step",
        "last_global_step",
        "vog_l2_population",
        "vog_l2_sample",
        "vog_total_variance_population",
        "vog_total_variance_sample",
        "vog_mean_variance_population",
        "vog_mean_variance_sample",
        "num_grad_values",
        "grad_norm_mean",
        "grad_norm_variance_population",
        "loss_mean",
        "loss_variance_population",
    )
    RANKING_FIELDS: Tuple[str, ...] = (
        "observation_id",
        "epoch",
        "step",
        "global_step",
        "sample_id",
        "dataset_index",
        "target",
        "observation_count",
        "vog_score",
        "rank",
        "rank_fraction",
        "top_percentile",
        "in_top_1pct",
        "in_top_5pct",
        "in_top_10pct",
    )
    STABILITY_FIELDS: Tuple[str, ...] = (
        "observation_id",
        "epoch",
        "step",
        "global_step",
        "top_percent",
        "top_k",
        "top_sample_ids",
        "previous_overlap_count",
        "previous_overlap_share",
        "previous_jaccard",
        "persistent_count",
        "persistent_share",
        "ever_top_count",
        "ever_top_share",
    )

    def __init__(
        self,
        *,
        out_dir: Path,
        samples: Sequence[VogProbeSample],
        device: torch.device,
        save_gradients: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.samples = list(samples)
        self.device = device
        self.save_gradients = bool(save_gradients)
        self.observation_id = 0

        self.grad_dir = self.out_dir / "gradients"
        self.observations_csv = self.out_dir / "vog_observations.csv"
        self.samples_csv = self.out_dir / "vog_probe_samples.csv"
        self.summary_csv = self.out_dir / "vog_summary.csv"
        self.rankings_csv = self.out_dir / "vog_rankings.csv"
        self.stability_csv = self.out_dir / "vog_top_stability.csv"
        self._global_norm_history: DefaultDict[int, List[float]] = defaultdict(list)
        self._loss_history: DefaultDict[int, List[float]] = defaultdict(list)
        self._top_history: DefaultDict[int, List[Set[int]]] = defaultdict(list)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.grad_dir.mkdir(parents=True, exist_ok=True)
        self._write_probe_samples()

    def _write_probe_samples(self) -> None:
        rows = [
            {"sample_id": s.sample_id, "dataset_index": s.dataset_index, "target": s.y}
            for s in self.samples
        ]
        if self.samples_csv.exists():
            self.samples_csv.unlink()
        _append_csv_rows(self.samples_csv, rows, self.SAMPLE_FIELDS)

    def observe(self, model: nn.Module, *, epoch: int, step: Optional[int], global_step: int) -> int:
        criterion = nn.CrossEntropyLoss()
        was_training = model.training
        model.eval()

        obs_id = self.observation_id
        obs_dir = self.grad_dir / f"obs_{obs_id:04d}_global_step_{int(global_step):06d}"
        if self.save_gradients:
            obs_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for sample in self.samples:
            model.zero_grad(set_to_none=True)

            x = sample.x.unsqueeze(0).to(self.device).detach().clone()
            x.requires_grad_(True)
            y = torch.tensor([sample.y], dtype=torch.long, device=self.device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            pred = int(logits.detach().argmax(dim=1).item())
            input_grad = x.grad.detach().cpu().float().clone() if x.grad is not None else torch.empty(0)
            input_energy = float((input_grad * input_grad).sum().item())
            input_numel = int(input_grad.numel())

            grad_file = ""
            if self.save_gradients:
                grad_path = obs_dir / f"sample_{sample.sample_id:04d}.pt"
                torch.save(
                    {
                        "observation_id": obs_id,
                        "epoch": int(epoch),
                        "step": None if step is None else int(step),
                        "global_step": int(global_step),
                        "sample_id": int(sample.sample_id),
                        "dataset_index": int(sample.dataset_index),
                        "target": int(sample.y),
                        "pred": pred,
                        "loss": float(loss.item()),
                        "input_grad": input_grad,
                    },
                    grad_path,
                )
                grad_file = str(grad_path)

            base_row = {
                "observation_id": obs_id,
                "epoch": int(epoch),
                "step": "" if step is None else int(step),
                "global_step": int(global_step),
                "sample_id": int(sample.sample_id),
                "dataset_index": int(sample.dataset_index),
                "target": int(sample.y),
                "pred": pred,
                "loss": float(loss.item()),
                "grad_file": grad_file,
            }
            self._global_norm_history[sample.sample_id].append(math.sqrt(input_energy))
            self._loss_history[sample.sample_id].append(float(loss.item()))
            rows.append(
                {
                    **base_row,
                    "scope": "input",
                    "layer": "",
                    "grad_norm": math.sqrt(input_energy),
                    "energy": input_energy,
                    "num_grad_values": input_numel,
                }
            )

        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

        _append_csv_rows(self.observations_csv, rows, self.OBS_FIELDS)
        ranking_info = self._log_rankings(obs_id=obs_id, epoch=epoch, step=step, global_step=global_step)
        self._print_progress(
            obs_id=obs_id,
            epoch=epoch,
            step=step,
            global_step=global_step,
            ranking_info=ranking_info,
        )
        self.observation_id += 1
        return obs_id

    def _log_rankings(
        self,
        *,
        obs_id: int,
        epoch: int,
        step: Optional[int],
        global_step: int,
    ) -> Dict[str, Any]:
        sample_by_id = {sample.sample_id: sample for sample in self.samples}
        scored: List[Tuple[int, float]] = []
        for sample in self.samples:
            values = self._global_norm_history.get(sample.sample_id, [])
            if len(values) >= 2:
                scored.append((sample.sample_id, float(np.var(values))))

        if not scored:
            return {}

        scored.sort(key=lambda item: (-item[1], item[0]))
        n = len(scored)
        thresholds = (1, 5, 10)
        top_sets: Dict[int, Set[int]] = {
            percent: {sample_id for sample_id, _ in scored[: max(1, int(math.ceil(n * percent / 100.0)))]}
            for percent in thresholds
        }

        ranking_rows: List[Dict[str, Any]] = []
        for rank_idx, (sample_id, score) in enumerate(scored, start=1):
            sample = sample_by_id[sample_id]
            top_percentile = 100.0 * rank_idx / max(n, 1)
            ranking_rows.append(
                {
                    "observation_id": obs_id,
                    "epoch": int(epoch),
                    "step": "" if step is None else int(step),
                    "global_step": int(global_step),
                    "sample_id": int(sample.sample_id),
                    "dataset_index": int(sample.dataset_index),
                    "target": int(sample.y),
                    "observation_count": len(self._global_norm_history[sample_id]),
                    "vog_score": score,
                    "rank": rank_idx,
                    "rank_fraction": rank_idx / max(n, 1),
                    "top_percentile": top_percentile,
                    "in_top_1pct": int(sample_id in top_sets[1]),
                    "in_top_5pct": int(sample_id in top_sets[5]),
                    "in_top_10pct": int(sample_id in top_sets[10]),
                }
            )
        _append_csv_rows(self.rankings_csv, ranking_rows, self.RANKING_FIELDS)

        stability_rows: List[Dict[str, Any]] = []
        stability_info: Dict[int, Dict[str, Any]] = {}
        for percent in thresholds:
            current = top_sets[percent]
            history = self._top_history[percent]
            previous = history[-1] if history else set()
            previous_overlap = current & previous
            previous_union = current | previous
            all_sets = history + [current]
            persistent = set.intersection(*all_sets) if all_sets else set()
            ever_top = set.union(*all_sets) if all_sets else set()
            top_k = max(1, int(math.ceil(n * percent / 100.0)))
            row = {
                "observation_id": obs_id,
                "epoch": int(epoch),
                "step": "" if step is None else int(step),
                "global_step": int(global_step),
                "top_percent": percent,
                "top_k": top_k,
                "top_sample_ids": " ".join(str(sample_id) for sample_id in sorted(current)),
                "previous_overlap_count": len(previous_overlap),
                "previous_overlap_share": (len(previous_overlap) / max(len(current), 1)) if history else "",
                "previous_jaccard": (len(previous_overlap) / max(len(previous_union), 1)) if history else "",
                "persistent_count": len(persistent),
                "persistent_share": len(persistent) / max(len(current), 1),
                "ever_top_count": len(ever_top),
                "ever_top_share": len(ever_top) / max(top_k, 1),
            }
            stability_rows.append(row)
            stability_info[percent] = row
            history.append(set(current))

        _append_csv_rows(self.stability_csv, stability_rows, self.STABILITY_FIELDS)
        return {
            "top_sample_id": scored[0][0],
            "top_score": scored[0][1],
            "stability": stability_info,
        }

    def _print_progress(
        self,
        *,
        obs_id: int,
        epoch: int,
        step: Optional[int],
        global_step: int,
        ranking_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        variances = [float(np.var(values)) for values in self._global_norm_history.values() if len(values) >= 2]
        step_text = "epoch_end" if step is None else str(step)

        if variances:
            if ranking_info:
                stability = ranking_info.get("stability", {})
                top10 = stability.get(10, {})
                overlap = top10.get("previous_overlap_share", "")
                overlap_text = "nan" if overlap == "" else f"{float(overlap):.3f}"
                print(
                    "[VoG-rank] "
                    f"epoch={epoch} step={step_text} obs={obs_id} "
                    f"top10_ids={top10.get('top_sample_ids', '')} "
                    f"prev_overlap={overlap_text} "
                    f"persistent={top10.get('persistent_count', 0)}/{top10.get('top_k', 0)}"
                )
        else:
            print(
                "[VoG-rank] "
                f"epoch={epoch} step={step_text} obs={obs_id} "
                "need_more_observations"
            )

    def summarize(self) -> None:
        if not self.save_gradients:
            return

        rows: List[Dict[str, Any]] = []
        sample_files: DefaultDict[int, List[Path]] = defaultdict(list)
        for path in sorted(self.grad_dir.glob("obs_*/sample_*.pt")):
            try:
                sample_id = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            sample_files[sample_id].append(path)

        for sample in self.samples:
            files = sample_files.get(sample.sample_id, [])
            if len(files) < 2:
                continue
            rows.extend(self._summarize_sample(sample, files))

        if self.summary_csv.exists():
            self.summary_csv.unlink()
        _append_csv_rows(self.summary_csv, rows, self.SUMMARY_FIELDS)

    def _summarize_sample(self, sample: VogProbeSample, files: Sequence[Path]) -> List[Dict[str, Any]]:
        layer_state: Dict[str, Dict[str, Any]] = {}
        grad_norms: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
        losses: List[float] = []
        global_steps: List[int] = []

        for path in files:
            rec = torch.load(path, map_location="cpu")
            losses.append(float(rec.get("loss", float("nan"))))
            global_steps.append(int(rec.get("global_step", -1)))

            input_grad = rec.get("input_grad")
            if isinstance(input_grad, torch.Tensor):
                vec = input_grad.detach().float().reshape(-1)
                energy = float((vec * vec).sum().item())
                grad_norms[("input", "")].append(math.sqrt(energy))
                state = layer_state.get("__input__")
                if state is None:
                    layer_state["__input__"] = {
                        "count": 1,
                        "mean": vec.clone(),
                        "m2": torch.zeros_like(vec),
                        "numel": int(vec.numel()),
                    }
                else:
                    state["count"] += 1
                    count = int(state["count"])
                    delta = vec - state["mean"]
                    state["mean"].add_(delta / count)
                    delta2 = vec - state["mean"]
                    state["m2"].add_(delta * delta2)

        rows: List[Dict[str, Any]] = []

        for key in sorted(layer_state):
            state = layer_state[key]
            count = int(state["count"])
            if count < 2:
                continue
            m2 = state["m2"]
            pop_total = float((m2 / count).sum().item())
            sample_total = float((m2 / (count - 1)).sum().item())
            numel = int(state["numel"])
            norms = grad_norms[("input", "")]
            rows.append(
                {
                    "sample_id": int(sample.sample_id),
                    "dataset_index": int(sample.dataset_index),
                    "target": int(sample.y),
                    "scope": "input",
                    "layer": "",
                    "observation_count": count,
                    "first_global_step": min(global_steps),
                    "last_global_step": max(global_steps),
                    "vog_l2_population": math.sqrt(max(pop_total, 0.0)),
                    "vog_l2_sample": math.sqrt(max(sample_total, 0.0)),
                    "vog_total_variance_population": pop_total,
                    "vog_total_variance_sample": sample_total,
                    "vog_mean_variance_population": pop_total / max(numel, 1),
                    "vog_mean_variance_sample": sample_total / max(numel, 1),
                    "num_grad_values": numel,
                    "grad_norm_mean": float(np.mean(norms)) if norms else "",
                    "grad_norm_variance_population": float(np.var(norms)) if len(norms) > 1 else "",
                    "loss_mean": float(np.nanmean(losses)) if losses else "",
                    "loss_variance_population": float(np.nanvar(losses)) if len(losses) > 1 else "",
                }
            )
            del state["mean"]
            del state["m2"]
        return rows


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
    """Loads prepared samples from <variant>/metadata.jsonl and JPEGs.

    Notes
    - Filenames created by data_preparing.py include split prefix: <split>_<hash>.jpg
    - We filter rows by that prefix.
    """

    def __init__(
        self,
        variant_dir: Path,
        split_name: str,
        tfm_cfg: TransformConfig,
        expected_poison_frac: Optional[float] = None,
    ) -> None:
        self.variant_dir = variant_dir
        self.split_name = split_name
        self.tfm_cfg = tfm_cfg

        meta_path = self.variant_dir / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.jsonl not found: {meta_path}")

        items: List[Tuple[str, int]] = []
        seen_poison_fracs: set[float] = set()
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                fname = rec["file"]
                if not fname.startswith(f"{split_name}_"):
                    continue
                poison_frac = rec.get("poison_frac")
                if poison_frac is not None:
                    try:
                        seen_poison_fracs.add(round(float(poison_frac), 6))
                    except (TypeError, ValueError):
                        pass
                items.append((fname, int(rec["label"])))

        if not items:
            raise RuntimeError(f"No items found for split='{split_name}' in {meta_path}")

        self.items = items
        self.prepared_poison_fracs = seen_poison_fracs

        if expected_poison_frac is not None and seen_poison_fracs:
            expected = round(float(expected_poison_frac), 6)
            if expected not in seen_poison_fracs:
                warnings.warn(
                    f"Prepared metadata poison_frac={sorted(seen_poison_fracs)} does not match "
                    f"requested --poison-frac={expected_poison_frac} for {meta_path}",
                    stacklevel=2,
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fname, y = self.items[idx]
        img_path = self.variant_dir / fname
        img = Image.open(img_path)
        x = apply_transform(img, self.tfm_cfg)
        return x, y


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
    def __init__(self, n_classes: int, img_size: int, dropout: float = 0.0) -> None:
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
        self.dropout1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(float(dropout))
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
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        from torchvision import models

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
        reader = self._reader
        if reader is not None:
            reader.join(timeout=3.0)
        proc = self._proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass
        self._reader = None
        self._proc = None

    def collect(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def _set_error(self, msg: str) -> None:
        with self._lock:
            self._latest["perf_status"] = "error"
            self._latest["perf_error"] = msg

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            self._set_error("perf process missing stderr")
            return
        try:
            for raw_line in proc.stderr:
                if self._stop_requested:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                parsed = self._parse_perf_line(line)
                if parsed is None:
                    continue
                ts_raw, event_name, value = parsed
                with self._lock:
                    if self._pending_ts is None:
                        self._pending_ts = ts_raw
                    if ts_raw != self._pending_ts:
                        self._flush_pending_locked()
                        self._pending_ts = ts_raw
                    self._pending_row[self._column_name(event_name)] = value
            with self._lock:
                self._flush_pending_locked()
        except Exception as exc:
            self._set_error(f"{type(exc).__name__}: {exc}")
        finally:
            ret = proc.poll()
            if ret is None:
                return
            if ret != 0 and not self._stop_requested:
                with self._lock:
                    self._latest["perf_status"] = "error"
                    if not self._latest.get("perf_error"):
                        self._latest["perf_error"] = f"perf exited with code {ret}"

    def _parse_perf_line(self, line: str) -> Optional[Tuple[str, str, Optional[float]]]:
        if line.startswith("#"):
            return None
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None
        ts_raw = parts[0]
        value_raw = parts[1]
        event_name = parts[2]
        if not ts_raw or event_name not in self.event_set:
            return None
        return ts_raw, event_name, self._parse_value(value_raw)

    def _flush_pending_locked(self) -> None:
        if self._pending_ts is None:
            return
        row = self._empty_row()
        row.update(self._pending_row)
        row["perf_status"] = "ok"
        row["perf_error"] = ""
        row["perf_time_ms"] = self._parse_value(self._pending_ts)
        self._latest = row
        self._pending_ts = None
        self._pending_row = {}


class SystemCollector(MetricsCollector):
    def __init__(self, proc: psutil.Process) -> None:
        self.proc = proc
        self._last_disk = None
        self._last_ts = None

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
        now_ts = now.timestamp()

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

        # Disk IO (system-wide counters)
        disk = psutil.disk_io_counters()
        if disk is not None:
            read_bytes = disk.read_bytes
            write_bytes = disk.write_bytes
            read_count = disk.read_count
            write_count = disk.write_count
        else:
            read_bytes = write_bytes = read_count = write_count = None

        # Deltas since last sample
        if self._last_disk is not None and self._last_ts is not None and disk is not None:
            dt_sec = max(now_ts - self._last_ts, 1e-6)
            read_bytes_delta = read_bytes - self._last_disk.read_bytes
            write_bytes_delta = write_bytes - self._last_disk.write_bytes
            read_bps = read_bytes_delta / dt_sec
            write_bps = write_bytes_delta / dt_sec
        else:
            read_bytes_delta = write_bytes_delta = None
            read_bps = write_bps = None

        self._last_disk = disk
        self._last_ts = now_ts

        out: Dict[str, Any] = {
            "ts_iso": now.isoformat(),
            "ts_unix": now_ts,
            "cpu_percent": cpu_total,
            "cpu_per_core": json.dumps(cpu_per_core),
            "mem_percent": vm.percent,
            "swap_percent": sm.percent,
            "proc_cpu_percent": p_cpu,
            "proc_rss": p_mem,
            "cpu_temp_c": temp_c,
            "disk_read_bytes": read_bytes,
            "disk_write_bytes": write_bytes,
            "disk_read_count": read_count,
            "disk_write_count": write_count,
            "disk_read_bytes_delta": read_bytes_delta,
            "disk_write_bytes_delta": write_bytes_delta,
            "disk_read_bps": read_bps,
            "disk_write_bps": write_bps,
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
            "batch_loss": None,
            "lr": None,
            "acc": None,
            "batch_acc": None,
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

    def log_batch(self, *, epoch: int, step: int, global_step: int, loss: float, lr: float, acc: float) -> None:
        self.update_state(
            epoch=epoch,
            step=step,
            global_step=global_step,
            loss=loss,
            batch_loss=loss,
            lr=lr,
            acc=acc,
            batch_acc=acc,
        )
        self._write_row(event="batch_end")

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
    epoch: int,
    global_step_start: int,
    save_grad: bool = False,
    grad_log_dir: Optional[Path] = None,
    vog_observer: Optional[VogGradientObserver] = None,
    vog_every_steps: int = 0,
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
        if save_grad and grad_log_dir is not None:
            grad_path = grad_log_dir / f"epoch_{epoch:04d}_step_{global_step:06d}.pt"
            save_model_grads(model, grad_path)
        optim.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = float((preds == y).float().mean().item())

        lr = float(optim.param_groups[0].get("lr", 0.0))
        logger.log_batch(
            epoch=epoch,
            step=step,
            global_step=global_step,
            loss=float(loss.item()),
            lr=lr,
            acc=acc,
        )
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}", lr=f"{lr:.2e}")

        global_step += 1
        if vog_observer is not None and int(vog_every_steps) > 0 and global_step % int(vog_every_steps) == 0:
            vog_observer.observe(model, epoch=epoch, step=step, global_step=global_step)
            model.train()

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
        choices=["none", "clean", "augmentation", "ood", "blurring", "occlusion", "steganography", "label-flip"],
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
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
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
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability after simple_cnn fc1/fc2; 0 disables dropout",
    )

    # Logging
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--log-fps", type=float, default=1.0)
    p.add_argument(
        "--grad-log-root",
        type=str,
        default="grad_logs",
        help="Root directory for saved gradient .pt files.",
    )
    p.add_argument(
        "--grad-reference-run",
        action="store_true",
        help="Save all epoch/step gradients and keep them.",
    )
    p.add_argument(
        "--vog-run",
        action="store_true",
        help="Enable fixed-input Variance of Gradients collection.",
    )
    p.add_argument(
        "--vog-log-root",
        type=str,
        default="vog_logs",
        help="Root directory for fixed-input VoG outputs.",
    )
    p.add_argument(
        "--vog-num-samples",
        type=int,
        default=8,
        help="Number of fixed probe samples used for VoG.",
    )
    p.add_argument(
        "--vog-seed",
        type=int,
        default=None,
        help="Seed for choosing fixed VoG probe samples. Defaults to --seed.",
    )
    p.add_argument(
        "--vog-probe-source",
        type=str,
        default="data-clean",
        choices=["data-clean", "train", "clean-train", "clean-test"],
        help="Dataset source for fixed VoG probe samples. Default uses <data-root>/clean.",
    )
    p.add_argument(
        "--vog-every-steps",
        type=int,
        default=0,
        help="Also collect VoG every N train steps. 0 means epoch boundaries only.",
    )
    p.add_argument(
        "--no-vog-at-train-start",
        action="store_false",
        dest="vog_at_train_start",
        help="Disable the initial VoG observation before the first optimizer step.",
    )
    p.set_defaults(vog_at_train_start=True)
    p.add_argument(
        "--no-vog-save-gradients",
        action="store_false",
        dest="vog_save_gradients",
        help="Only save VoG observation CSV; disables exact vector VoG summary.",
    )
    p.set_defaults(vog_save_gradients=True)
    p.add_argument(
        "--disable-perf",
        action="store_true",
        help="Disable perf-based metric collection",
    )

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
        prepared_train = PoisonDiskDataset(
            variant_dir=variant_dir,
            split_name=train_split,
            tfm_cfg=tfm_cfg,
            expected_poison_frac=float(args.poison_frac),
        )
        train_ds = subsample_dataset(prepared_train, frac=float(args.train_frac), seed=args.seed + 2)

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
        model = SimpleCNN(n_classes=n_classes, img_size=int(img_size), dropout=float(args.dropout)).to(device)
    if args.optimizer == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    else:
        optim = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # Logger
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = _timestamp_name()
    out_csv = log_dir / f"{run_name}.csv"
    grad_log_dir: Optional[Path] = None
    save_gradients = bool(args.grad_reference_run)
    if save_gradients:
        grad_log_dir = Path(args.grad_log_root) / run_name
        grad_log_dir.mkdir(parents=True, exist_ok=True)

    vog_observer: Optional[VogGradientObserver] = None
    vog_log_dir: Optional[Path] = None
    if bool(args.vog_run):
        vog_source = str(args.vog_probe_source)
        if vog_source == "data-clean":
            clean_variant_dir = Path(args.data_root) / "clean"
            if not clean_variant_dir.exists():
                raise FileNotFoundError(
                    f"VoG probe source requires clean prepared data at {clean_variant_dir}. "
                    "Set --data-root to the directory containing clean/."
                )
            vog_ds = PoisonDiskDataset(
                variant_dir=clean_variant_dir,
                split_name=train_split,
                tfm_cfg=tfm_cfg,
                expected_poison_frac=None,
            )
        elif vog_source == "clean-train":
            vog_ds = clean_train
        elif vog_source == "clean-test":
            vog_ds = clean_test
        else:
            vog_ds = train_ds
        vog_seed = int(args.seed if args.vog_seed is None else args.vog_seed)
        vog_samples = build_vog_probe_samples(vog_ds, num_samples=int(args.vog_num_samples), seed=vog_seed)
        vog_log_dir = Path(args.vog_log_root) / run_name
        vog_observer = VogGradientObserver(
            out_dir=vog_log_dir,
            samples=vog_samples,
            device=device,
            save_gradients=bool(args.vog_save_gradients),
        )

    proc = psutil.Process(os.getpid())
    collectors: List[MetricsCollector] = [SystemCollector(proc)]
    if not args.disable_perf:
        collectors.append(PerfCollector(pid=os.getpid(), fps=float(args.log_fps)))
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
                    "optimizer": args.optimizer,
                    "seed": int(args.seed),
                    "train_frac": float(args.train_frac),
                    "test_frac": float(args.test_frac),
                    "device": str(device),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "model": args.model,
                    "dropout": float(args.dropout),
                    "grad_reference_run": bool(args.grad_reference_run),
                    "grad_log_root": str(args.grad_log_root),
                    "grad_log_dir": str(grad_log_dir) if grad_log_dir is not None else None,
                    "vog_run": bool(args.vog_run),
                    "vog_log_root": str(args.vog_log_root),
                    "vog_log_dir": str(vog_log_dir) if vog_log_dir is not None else None,
                    "vog_num_samples": int(args.vog_num_samples),
                    "vog_seed": int(args.seed if args.vog_seed is None else args.vog_seed),
                    "vog_probe_source": str(args.vog_probe_source),
                    "vog_every_steps": int(args.vog_every_steps),
                    "vog_at_train_start": bool(args.vog_at_train_start),
                    "vog_save_gradients": bool(args.vog_save_gradients),
                    "disable_perf": bool(args.disable_perf),
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
    if grad_log_dir is not None:
        print(f"grad_log_dir={grad_log_dir}")
    if vog_log_dir is not None:
        print(f"vog_log_dir={vog_log_dir}")

    # Training
    global_step = 0
    logger.enable()
    logger.mark_event("train_start")

    try:
        bg_cfg: Optional[Dict[str, Any]] = None
        bg_proc: Optional[subprocess.Popen] = None
        if vog_observer is not None and bool(args.vog_at_train_start):
            logger.mark_event("vog_observation_train_start")
            vog_observer.observe(model, epoch=-1, step=None, global_step=global_step)
            model.train()
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
                epoch=epoch,
                global_step_start=global_step,
                save_grad=save_gradients,
                grad_log_dir=grad_log_dir,
                vog_observer=vog_observer,
                vog_every_steps=int(args.vog_every_steps),
                bg_cfg=bg_cfg,
                seed=int(args.seed),
            )
            if vog_observer is not None:
                logger.mark_event(json.dumps({"vog_observation_epoch_end": {"epoch": epoch}}, ensure_ascii=False))
                vog_observer.observe(model, epoch=epoch, step=None, global_step=global_step)
                model.train()

            test_loss, test_acc = evaluate(model, test_loader, device=device)
            print(f"epoch={epoch} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
            logger.mark_event(
                json.dumps(
                    {"epoch_eval": {"epoch": epoch, "loss": test_loss, "acc": test_acc}},
                    ensure_ascii=False,
                )
            )

            # Mark end of each epoch
            logger.mark_event(f"epoch_end_{epoch}")

        # Mark end of full training
        logger.mark_event("train_end")
        logger.disable()

        if vog_observer is not None:
            vog_observer.summarize()
            if vog_observer.summary_csv.exists():
                print(f"vog_summary_csv={vog_observer.summary_csv}")
            print(f"vog_observations_csv={vog_observer.observations_csv}")
            print(f"vog_rankings_csv={vog_observer.rankings_csv}")
            print(f"vog_top_stability_csv={vog_observer.stability_csv}")

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
        logger.stop()


if __name__ == "__main__":
    main()
