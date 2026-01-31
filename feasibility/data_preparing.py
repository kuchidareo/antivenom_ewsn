
"""Data preparation script.

Goal
- Download an image-classification dataset from Hugging Face.
- Apply 3 preprocessing variants (no augmentation intent; these are strong corruptions):
  1) blurring: very strong Gaussian blur on the whole image
  2) occlusion: draw a single black rectangular mask covering 10–50% of width/height
  3) label-flip: keep the image unchanged but flip the label to a random *different* class
- Save results to `data/` as JPEGs in a flat layout per variant:

    data/
      blurring/*.jpg
      blurring/metadata.jsonl
      occlusion/*.jpg
      occlusion/metadata.jsonl
      label-flip/*.jpg
      label-flip/metadata.jsonl

Each JSONL line contains at least:
  {"file": "<filename>.jpg", "label": <int>}

Example:
  python data_preparing.py --dataset kuchidareo/trashnet_small --out data --max-per-split 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Image as HFImage, load_dataset
from PIL import Image, ImageFilter
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    seed: int = 42

    # Blurring (very strong by default)
    blur_radius: float = 12.0

    # Occlusion: random black rectangle
    occlusion_min_frac: float = 0.10  # min fraction of width/height
    occlusion_max_frac: float = 0.50  # max fraction of width/height


VARIANTS = ("clean", "blurring", "occlusion", "label-flip")


# -----------------------------
# Helpers: dataset schema detection
# -----------------------------


def _find_image_column(features: Dict[str, Any]) -> str:
    if "image" in features and isinstance(features["image"], HFImage):
        return "image"
    for k, v in features.items():
        if isinstance(v, HFImage):
            return k
    raise ValueError(
        "Could not find an image column. Expected a datasets.Image feature (often named 'image')."
    )


def _find_label_column(features: Dict[str, Any]) -> str:
    if "label" in features:
        return "label"
    for k, v in features.items():
        if isinstance(v, ClassLabel):
            return k
    for candidate in ("labels", "category", "class", "target"):
        if candidate in features:
            return candidate
    raise ValueError(
        "Could not find a label column. Expected 'label' or a datasets.ClassLabel feature."
    )


def _get_class_names(ds: Dataset, label_col: str) -> Optional[List[str]]:
    feat = ds.features.get(label_col)
    if isinstance(feat, ClassLabel):
        return list(feat.names)
    return None


# -----------------------------
# Preprocess functions
# -----------------------------


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def apply_blurring(img: Image.Image, cfg: PreprocessConfig) -> Image.Image:
    # Whole-image strong blur
    return img.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))


def apply_occlusion(img: Image.Image, cfg: PreprocessConfig, rng: random.Random) -> Image.Image:
    # Single random black rectangle (mask), keeping image size unchanged.
    img = ensure_rgb(img)
    arr = np.array(img)

    h, w = arr.shape[:2]

    frac_w = rng.uniform(cfg.occlusion_min_frac, cfg.occlusion_max_frac)
    frac_h = rng.uniform(cfg.occlusion_min_frac, cfg.occlusion_max_frac)

    occ_w = max(1, int(w * frac_w))
    occ_h = max(1, int(h * frac_h))

    x0 = rng.randint(0, max(0, w - occ_w))
    y0 = rng.randint(0, max(0, h - occ_h))

    # Fill black
    arr[y0 : y0 + occ_h, x0 : x0 + occ_w, :] = 0
    return Image.fromarray(arr)


def flip_label_random_other(label: int, n_classes: int, rng: random.Random) -> int:
    """Pick a random label != label."""
    if n_classes <= 1:
        return label
    new_label = rng.randrange(n_classes)
    if new_label == label:
        new_label = (new_label + 1) % n_classes
    return new_label


# -----------------------------
# IO helpers
# -----------------------------


def stable_id(split: str, idx: int, variant: str) -> str:
    raw = f"{split}:{idx}:{variant}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def save_jpeg(img: Image.Image, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = ensure_rgb(img)
    img.save(path, format="JPEG", quality=quality, optimize=True)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# Main pipeline
# -----------------------------


def process_split(
    ds: Dataset,
    split_name: str,
    out_root: Path,
    cfg: PreprocessConfig,
    max_per_split: Optional[int] = None,
    image_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> None:
    features = ds.features
    image_col = image_col or _find_image_column(features)
    label_col = label_col or _find_label_column(features)

    class_names = _get_class_names(ds, label_col)
    if class_names is None:
        # Best-effort: infer classes from a subset of labels.
        labels = set()
        for i in range(min(len(ds), 5000)):
            labels.add(int(ds[i][label_col]))
        class_names = [str(x) for x in sorted(labels)]

    n_classes = len(class_names)

    # Deterministic RNG across the run given the seed.
    # We offset per split so different splits don't share the same random sequence.
    split_rng = random.Random(cfg.seed + (hash(split_name) % 10_000))

    n = len(ds)
    limit = min(n, max_per_split) if max_per_split is not None else n

    clean_meta = out_root / "clean" / "metadata.jsonl"
    blur_meta = out_root / "blurring" / "metadata.jsonl"
    occ_meta = out_root / "occlusion" / "metadata.jsonl"
    flip_meta = out_root / "label-flip" / "metadata.jsonl"

    for idx in tqdm(range(limit), desc=f"Processing {split_name}"):
        ex = ds[idx]

        img_any = ex[image_col]
        if isinstance(img_any, dict) and "image" in img_any:
            img = img_any["image"]
        else:
            img = img_any

        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image for column '{image_col}', got {type(img)}")

        img = ensure_rgb(img)

        label = int(ex[label_col])
        label_name = class_names[label] if 0 <= label < n_classes else str(label)

        # Create a per-sample RNG so label-flip/occlusion are deterministic per sample.
        sample_seed = int(hashlib.sha1(f"{cfg.seed}:{split_name}:{idx}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(sample_seed)

        # 0) clean (no change)
        v = "clean"
        file_id = stable_id(split_name, idx, v)
        filename = f"{split_name}_{file_id}.jpg"
        out_path = out_root / v / filename
        save_jpeg(img, out_path)
        append_jsonl(
            clean_meta,
            {
                "file": filename,
                "label": label,
                "split": split_name,
                "orig_label": label,
                "label_name": label_name,
            },
        )

        # 1) blurring
        v = "blurring"
        file_id = stable_id(split_name, idx, v)
        filename = f"{split_name}_{file_id}.jpg"
        out_path = out_root / v / filename
        out_img = apply_blurring(img, cfg)
        save_jpeg(out_img, out_path)
        append_jsonl(
            blur_meta,
            {
                "file": filename,
                "label": label,
                "split": split_name,
                "orig_label": label,
                "label_name": label_name,
            },
        )

        # 2) occlusion
        v = "occlusion"
        file_id = stable_id(split_name, idx, v)
        filename = f"{split_name}_{file_id}.jpg"
        out_path = out_root / v / filename
        out_img = apply_occlusion(img, cfg, rng=rng)
        save_jpeg(out_img, out_path)
        append_jsonl(
            occ_meta,
            {
                "file": filename,
                "label": label,
                "split": split_name,
                "orig_label": label,
                "label_name": label_name,
            },
        )

        # 3) label-flip (image unchanged)
        v = "label-flip"
        file_id = stable_id(split_name, idx, v)
        filename = f"{split_name}_{file_id}.jpg"
        out_path = out_root / v / filename
        new_label = flip_label_random_other(label, n_classes=n_classes, rng=rng)
        new_label_name = class_names[new_label] if 0 <= new_label < n_classes else str(new_label)
        save_jpeg(img, out_path)
        append_jsonl(
            flip_meta,
            {
                "file": filename,
                "label": new_label,
                "split": split_name,
                "orig_label": label,
                "orig_label_name": label_name,
                "label_name": new_label_name,
            },
        )


def load_hf_dataset(repo_id: str, config_name: Optional[str] = None) -> DatasetDict:
    if config_name:
        return load_dataset(repo_id, config_name)
    return load_dataset(repo_id)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="kuchidareo/trashnet_small")
    p.add_argument("--config", type=str, default=None, help="HF dataset config name (if needed)")
    p.add_argument("--out", type=str, default="data", help="Output root directory")
    p.add_argument("--max-per-split", type=int, default=None, help="Optional cap per split")

    p.add_argument("--seed", type=int, default=42)

    # Blurring
    p.add_argument("--blur-radius", type=float, default=12.0, help="Gaussian blur radius (very strong default)")

    # Occlusion
    p.add_argument("--occlusion-min-frac", type=float, default=0.10)
    p.add_argument("--occlusion-max-frac", type=float, default=0.50)

    return p.parse_args()


def _truncate_metadata_files(out_root: Path) -> None:
    # Ensure metadata.jsonl starts fresh on each run.
    for v in VARIANTS:
        meta = out_root / v / "metadata.jsonl"
        meta.parent.mkdir(parents=True, exist_ok=True)
        with open(meta, "w", encoding="utf-8"):
            pass


def main() -> None:
    args = parse_args()

    cfg = PreprocessConfig(
        seed=args.seed,
        blur_radius=args.blur_radius,
        occlusion_min_frac=args.occlusion_min_frac,
        occlusion_max_frac=args.occlusion_max_frac,
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Reset metadata files
    _truncate_metadata_files(out_root)

    ds_dict = load_hf_dataset(args.dataset, config_name=args.config)

    # Detect columns from the first available split
    first_split = next(iter(ds_dict.keys()))
    image_col = _find_image_column(ds_dict[first_split].features)
    label_col = _find_label_column(ds_dict[first_split].features)

    print(f"Loaded dataset: {args.dataset}")
    print(f"Splits: {list(ds_dict.keys())}")
    print(f"Detected columns: image='{image_col}', label='{label_col}'")
    print(f"Output: {out_root.resolve()}")
    print(f"Variants: {list(VARIANTS)}")

    for split_name, split_ds in ds_dict.items():
        process_split(
            split_ds,
            split_name=split_name,
            out_root=out_root,
            cfg=cfg,
            max_per_split=args.max_per_split,
            image_col=image_col,
            label_col=label_col,
        )

    print("Done.")


if __name__ == "__main__":
    main()
