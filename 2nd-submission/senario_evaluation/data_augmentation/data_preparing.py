
"""Data preparation script.

Goal
- Download an image-classification dataset from Hugging Face.
- Prepare disk-backed variants ahead of training.
- Save results to `data/` as JPEGs in a flat layout per variant:

    data/
      blurring/*.jpg
      blurring/metadata.jsonl
      occlusion/*.jpg
      occlusion/metadata.jsonl
      label-flip/*.jpg
      label-flip/metadata.jsonl
      steganography/*.jpg
      steganography/metadata.jsonl
      augmentation/*.jpg
      augmentation/metadata.jsonl

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
    poison_frac: float = 1.0

    # Blurring (very strong by default)
    blur_radius: float = 12.0

    # Occlusion: random black rectangle
    occlusion_min_frac: float = 0.10  # min fraction of width/height
    occlusion_max_frac: float = 0.50  # max fraction of width/height


VARIANTS = ("clean", "blurring", "occlusion", "label-flip", "steganography", "augmentation")


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
# Steganography helpers
# -----------------------------


STEGO_TEXT = "Lorem ipsum dolor sit amet. consectupidatat non proident, sunt in culpa qui offici."
STEGO_DELIM = "#####"


def message_to_binary(message):
    if isinstance(message, str):
        return "".join([format(ord(i), "08b") for i in message])
    if isinstance(message, (bytes, np.ndarray)):
        return [format(i, "08b") for i in message]
    if isinstance(message, (int, np.uint8)):
        return format(message, "08b")
    raise TypeError("Input type not supported.")


def hide_data(image: np.ndarray, secret_message: str) -> np.ndarray:
    n_bytes = image.shape[0] * image.shape[1] // 2
    if len(secret_message) > n_bytes:
        # Truncate to fit
        secret_message = secret_message[: max(1, n_bytes - len(STEGO_DELIM))]
    secret_message += STEGO_DELIM
    data_index = 0
    binary_secret_msg = message_to_binary(secret_message)
    data_len = len(binary_secret_msg)
    for values in image:
        for pixel in values:
            r, g, b = message_to_binary(pixel)
            if data_index < data_len:
                pixel[0] = int(binary_secret_msg[data_index] + r[1:], 2)
                data_index += 1
            if data_index < data_len:
                pixel[1] = int(binary_secret_msg[data_index] + g[1:], 2)
                data_index += 1
            if data_index < data_len:
                pixel[2] = int(binary_secret_msg[data_index] + b[1:], 2)
                data_index += 1
            if data_index >= data_len:
                break
        if data_index >= data_len:
            break
    return image


def show_data(image: np.ndarray) -> str:
    binary_data = ""
    for values in image:
        for pixel in values:
            r, g, b = message_to_binary(pixel)
            binary_data += r[0]
            binary_data += g[0]
            binary_data += b[0]

    all_bytes = [binary_data[i : i + 8] for i in range(0, len(binary_data), 8)]
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data.endswith(STEGO_DELIM):
            break
    return decoded_data[: -len(STEGO_DELIM)]


def apply_steganography(img: Image.Image) -> Image.Image:
    img = ensure_rgb(img)
    arr = np.array(img)
    stego = hide_data(arr.copy(), STEGO_TEXT)
    # Best-effort check
    try:
        assert STEGO_TEXT.startswith(show_data(stego))
    except Exception:
        pass
    return Image.fromarray(stego)


def build_augmentation_transform():
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "albumentations is required for the augmentation variant. "
            "Install it before running data_preparing.py for augmentation."
        ) from exc

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Rotate(limit=5, p=1.0),
                    A.Affine(
                        scale=(0.97, 1.03),
                        translate_percent=(-0.03, 0.03),
                        rotate=(-5, 5),
                        shear=(-2, 2),
                        p=1.0,
                    ),
                ],
                p=0.6,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
                    A.ImageCompression(quality_range=(60, 100), p=1.0),
                    A.RandomShadow(p=1.0),
                ],
                p=0.35,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.03, 0.08),
                hole_width_range=(0.03, 0.08),
                p=0.15,
            ),
        ]
    )


def apply_augmentation(img: Image.Image, sample_seed: int) -> Image.Image:
    img = ensure_rgb(img)
    tfm = build_augmentation_transform()
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32))
        transformed = tfm(image=np.array(img))
        return Image.fromarray(transformed["image"])
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


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

    # Choose an exact subset per split to poison so the prepared directory can be
    # loaded directly without additional clean/poison mixing at training time.
    all_indices = list(range(limit))
    variant_poisoned_indices: Dict[str, set[int]] = {}
    for variant in ("blurring", "occlusion", "label-flip", "steganography"):
        k = int(round(limit * cfg.poison_frac))
        k = max(0, min(k, limit))
        idxs = list(all_indices)
        random.Random(cfg.seed + (hash((split_name, variant)) % 10_000)).shuffle(idxs)
        variant_poisoned_indices[variant] = set(idxs[:k])

    clean_meta = out_root / "clean" / "metadata.jsonl"
    blur_meta = out_root / "blurring" / "metadata.jsonl"
    occ_meta = out_root / "occlusion" / "metadata.jsonl"
    flip_meta = out_root / "label-flip" / "metadata.jsonl"
    stego_meta = out_root / "steganography" / "metadata.jsonl"
    augmentation_meta = out_root / "augmentation" / "metadata.jsonl"

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
                "variant": "clean",
                "poison_type": "clean",
                "poison_applied": False,
                "poison_frac": 0.0,
                "orig_label": label,
                "label_name": label_name,
            },
        )

        # 1) blurring
        v = "blurring"
        file_id = stable_id(split_name, idx, v)
        filename = f"{split_name}_{file_id}.jpg"
        out_path = out_root / v / filename
        blur_applied = idx in variant_poisoned_indices[v]
        out_img = apply_blurring(img, cfg) if blur_applied else img
        save_jpeg(out_img, out_path)
        append_jsonl(
            blur_meta,
            {
                "file": filename,
                "label": label,
                "split": split_name,
                "variant": v,
                "poison_type": v,
                "poison_applied": blur_applied,
                "poison_frac": cfg.poison_frac,
                "orig_label": label,
                "label_name": label_name,
            },
        )

        # 2) occlusion
        v = "occlusion"
        file_id = stable_id(split_name, idx, v)
        filename = f"{split_name}_{file_id}.jpg"
        out_path = out_root / v / filename
        occ_applied = idx in variant_poisoned_indices[v]
        out_img = apply_occlusion(img, cfg, rng=rng) if occ_applied else img
        save_jpeg(out_img, out_path)
        append_jsonl(
            occ_meta,
            {
                "file": filename,
                "label": label,
                "split": split_name,
                "variant": v,
                "poison_type": v,
                "poison_applied": occ_applied,
                "poison_frac": cfg.poison_frac,
                "orig_label": label,
                "label_name": label_name,
            },
        )

        # 3) label-flip (image unchanged)
        v = "label-flip"
        file_id = stable_id(split_name, idx, v)
        filename = f"{split_name}_{file_id}.jpg"
        out_path = out_root / v / filename
        flip_applied = idx in variant_poisoned_indices[v]
        new_label = flip_label_random_other(label, n_classes=n_classes, rng=rng) if flip_applied else label
        new_label_name = class_names[new_label] if 0 <= new_label < n_classes else str(new_label)
        save_jpeg(img, out_path)
        append_jsonl(
            flip_meta,
            {
                "file": filename,
                "label": new_label,
                "split": split_name,
                "variant": v,
                "poison_type": v,
                "poison_applied": flip_applied,
                "poison_frac": cfg.poison_frac,
                "orig_label": label,
                "orig_label_name": label_name,
                "label_name": new_label_name,
            },
        )

        # 4) steganography
        v = "steganography"
        file_id = stable_id(split_name, idx, v)
        filename = f"{split_name}_{file_id}.jpg"
        out_path = out_root / v / filename
        stego_applied = idx in variant_poisoned_indices[v]
        out_img = apply_steganography(img) if stego_applied else img
        save_jpeg(out_img, out_path)
        append_jsonl(
            stego_meta,
            {
                "file": filename,
                "label": label,
                "split": split_name,
                "variant": v,
                "poison_type": v,
                "poison_applied": stego_applied,
                "poison_frac": cfg.poison_frac,
                "orig_label": label,
                "label_name": label_name,
            },
        )

        # 5) augmentation
        v = "augmentation"
        file_id = stable_id(split_name, idx, v)
        filename = f"{split_name}_{file_id}.jpg"
        out_path = out_root / v / filename
        out_img = apply_augmentation(img, sample_seed=sample_seed)
        save_jpeg(out_img, out_path)
        append_jsonl(
            augmentation_meta,
            {
                "file": filename,
                "label": label,
                "split": split_name,
                "variant": v,
                "poison_type": v,
                "poison_applied": True,
                "poison_frac": 1.0,
                "orig_label": label,
                "label_name": label_name,
                "preprocess_policy": "albumentations_flip_affine_brightness_noise_dropout",
            },
        )


def load_hf_dataset(repo_id: str, config_name: Optional[str] = None) -> DatasetDict:
    if config_name:
        return load_dataset(repo_id, config_name)
    return load_dataset(repo_id)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="kuchidareo/small_trashnet")
    p.add_argument("--config", type=str, default=None, help="HF dataset config name (if needed)")
    p.add_argument("--out", type=str, default="data", help="Output root directory")
    p.add_argument("--max-per-split", type=int, default=None, help="Optional cap per split")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--poison-frac",
        type=float,
        default=1.0,
        help="Fraction of each non-clean variant split to actually poison; the rest remain clean.",
    )

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
        poison_frac=max(0.0, min(1.0, float(args.poison_frac))),
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
    print(f"Poison frac: {cfg.poison_frac}")

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
