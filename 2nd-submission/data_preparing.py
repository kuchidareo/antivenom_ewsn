
"""Data preparation script.

Goal
- Download an image-classification dataset from Hugging Face.
- Apply preprocessing variants:
  1) blurring: very strong Gaussian blur on the whole image
  2) occlusion: draw a single black rectangular mask covering 50–80% of image area
  3) label-flip: keep the image unchanged but flip the label to a random *different* class
  4) steganography: hide a repeated payload in LSBs across a configurable image fraction
  5) augmentation: apply moderate image augmentation while preserving the label
  6) ood: replace a fraction of original images with Oxford Flowers images while
     keeping the original class label
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
      ood/*.jpg
      ood/metadata.jsonl

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

    # Occlusion: random black rectangle. Area fraction is clearer than sampling
    # width and height independently; 50-80% is a severe availability corruption.
    occlusion_min_area_frac: float = 0.50
    occlusion_max_area_frac: float = 0.80
    occlusion_min_aspect: float = 0.75
    occlusion_max_aspect: float = 1.33

    # Steganography: LSB payload strength. One bit plane over a small prefix was
    # too weak; two LSBs over half the channels is still visually subtle but much
    # more likely to create a measurable training signal.
    stego_payload_frac: float = 0.50
    stego_bit_planes: int = 2

    # OOD: fraction of positions replaced with external OOD images.
    ood_frac: float = 0.30


VARIANTS = ("clean", "blurring", "occlusion", "label-flip", "steganography", "augmentation", "ood")


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


def _resolve_matching_split(ds_dict: DatasetDict, split_name: str) -> str:
    if split_name in ds_dict:
        return split_name
    if split_name == "validation" and "test" in ds_dict:
        return "test"
    if split_name == "test" and "validation" in ds_dict:
        return "validation"
    if "train" in ds_dict:
        return "train"
    return next(iter(ds_dict.keys()))


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

    min_area = max(0.0, min(1.0, cfg.occlusion_min_area_frac))
    max_area = max(min_area, min(1.0, cfg.occlusion_max_area_frac))
    area_frac = rng.uniform(min_area, max_area)
    aspect = rng.uniform(cfg.occlusion_min_aspect, cfg.occlusion_max_aspect)

    target_area = max(1.0, area_frac * w * h)
    occ_w = max(1, min(w, int(round((target_area * aspect) ** 0.5))))
    occ_h = max(1, min(h, int(round((target_area / aspect) ** 0.5))))

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


def hide_data(
    image: np.ndarray,
    secret_message: str,
    payload_frac: float,
    bit_planes: int,
    rng: random.Random,
) -> np.ndarray:
    """Write a repeated payload into selected LSB channels across the image."""
    flat = image.reshape(-1)
    total_channels = flat.shape[0]
    payload_frac = max(0.0, min(1.0, payload_frac))
    bit_planes = max(1, min(4, int(bit_planes)))
    n_channels = max(1, int(round(total_channels * payload_frac)))

    selected = list(range(total_channels))
    rng.shuffle(selected)
    selected = selected[:n_channels]

    bits = message_to_binary(secret_message + STEGO_DELIM)
    if not bits:
        return image

    clear_mask = 0xFF ^ ((1 << bit_planes) - 1)
    data_index = 0
    for channel_idx in selected:
        payload = 0
        for _ in range(bit_planes):
            payload = (payload << 1) | int(bits[data_index % len(bits)])
            data_index += 1
        flat[channel_idx] = (int(flat[channel_idx]) & clear_mask) | payload

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


def apply_steganography(img: Image.Image, cfg: PreprocessConfig, rng: random.Random) -> Image.Image:
    img = ensure_rgb(img)
    arr = np.array(img)
    stego = hide_data(
        arr.copy(),
        STEGO_TEXT,
        payload_frac=cfg.stego_payload_frac,
        bit_planes=cfg.stego_bit_planes,
        rng=rng,
    )
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


def stable_seed(*parts: object) -> int:
    raw = ":".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha1(raw).hexdigest()[:8], 16)


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
    selected_variants: tuple[str, ...],
    max_per_split: Optional[int] = None,
    image_col: Optional[str] = None,
    label_col: Optional[str] = None,
    ood_ds: Optional[Dataset] = None,
    ood_image_col: Optional[str] = None,
    ood_label_col: Optional[str] = None,
    ood_dataset_name: str = "ood",
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

    n = len(ds)
    source_indices = list(range(n))
    random.Random(stable_seed(cfg.seed, split_name, "max_per_split")).shuffle(source_indices)
    if max_per_split is not None:
        source_indices = source_indices[: min(n, max_per_split)]
    limit = len(source_indices)

    # Choose an exact subset per split to poison so the prepared directory can be
    # loaded directly without additional clean/poison mixing at training time.
    # These indices are positions inside source_indices, not original dataset rows.
    all_positions = list(range(limit))
    variant_poisoned_positions: Dict[str, set[int]] = {}
    for variant in selected_variants:
        if variant == "clean":
            continue
        frac = cfg.ood_frac if variant == "ood" else cfg.poison_frac
        k = int(round(limit * frac))
        k = max(0, min(k, limit))
        positions = list(all_positions)
        random.Random(stable_seed(cfg.seed, split_name, variant)).shuffle(positions)
        variant_poisoned_positions[variant] = set(positions[:k])

    clean_meta = out_root / "clean" / "metadata.jsonl"
    blur_meta = out_root / "blurring" / "metadata.jsonl"
    occ_meta = out_root / "occlusion" / "metadata.jsonl"
    flip_meta = out_root / "label-flip" / "metadata.jsonl"
    stego_meta = out_root / "steganography" / "metadata.jsonl"
    augmentation_meta = out_root / "augmentation" / "metadata.jsonl"
    ood_meta = out_root / "ood" / "metadata.jsonl"

    ood_class_names: Optional[List[str]] = None
    ood_source_indices: List[int] = []
    ood_ptr = 0
    if "ood" in selected_variants:
        if ood_ds is None:
            raise ValueError("The 'ood' variant was selected, but no OOD dataset was provided.")
        ood_image_col = ood_image_col or _find_image_column(ood_ds.features)
        ood_label_col = ood_label_col or _find_label_column(ood_ds.features)
        ood_class_names = _get_class_names(ood_ds, ood_label_col)
        ood_source_indices = list(range(len(ood_ds)))
        random.Random(stable_seed(cfg.seed, split_name, "ood_source")).shuffle(ood_source_indices)
        if not ood_source_indices and variant_poisoned_positions.get("ood"):
            raise RuntimeError(f"No OOD source items available for split '{split_name}'")

    for pos, idx in enumerate(tqdm(source_indices, desc=f"Processing {split_name}")):
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
        rng = random.Random(stable_seed(cfg.seed, split_name, idx))

        if "clean" in selected_variants:
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

        if "blurring" in selected_variants:
            # 1) blurring
            v = "blurring"
            file_id = stable_id(split_name, idx, v)
            filename = f"{split_name}_{file_id}.jpg"
            out_path = out_root / v / filename
            blur_applied = pos in variant_poisoned_positions[v]
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

        if "occlusion" in selected_variants:
            # 2) occlusion
            v = "occlusion"
            file_id = stable_id(split_name, idx, v)
            filename = f"{split_name}_{file_id}.jpg"
            out_path = out_root / v / filename
            occ_applied = pos in variant_poisoned_positions[v]
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
                    "occlusion_min_area_frac": cfg.occlusion_min_area_frac,
                    "occlusion_max_area_frac": cfg.occlusion_max_area_frac,
                },
            )

        if "label-flip" in selected_variants:
            # 3) label-flip (image unchanged)
            v = "label-flip"
            file_id = stable_id(split_name, idx, v)
            filename = f"{split_name}_{file_id}.jpg"
            out_path = out_root / v / filename
            flip_applied = pos in variant_poisoned_positions[v]
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

        if "steganography" in selected_variants:
            # 4) steganography
            v = "steganography"
            file_id = stable_id(split_name, idx, v)
            filename = f"{split_name}_{file_id}.jpg"
            out_path = out_root / v / filename
            stego_applied = pos in variant_poisoned_positions[v]
            out_img = apply_steganography(img, cfg=cfg, rng=rng) if stego_applied else img
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
                    "stego_payload_frac": cfg.stego_payload_frac,
                    "stego_bit_planes": cfg.stego_bit_planes,
                },
            )

        if "augmentation" in selected_variants:
            # 5) augmentation
            v = "augmentation"
            file_id = stable_id(split_name, idx, v)
            filename = f"{split_name}_{file_id}.jpg"
            out_path = out_root / v / filename
            sample_seed = stable_seed(cfg.seed, split_name, idx, v)
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

        if "ood" in selected_variants:
            # 6) OOD replacement: same source position and label as clean, but
            # selected positions use an external image.
            v = "ood"
            file_id = stable_id(split_name, idx, v)
            filename = f"{split_name}_{file_id}.jpg"
            out_path = out_root / v / filename
            ood_applied = pos in variant_poisoned_positions[v]
            source_dataset = "trashnet"
            ood_source_label = None
            ood_source_label_name = None

            if ood_applied:
                assert ood_ds is not None
                assert ood_image_col is not None
                assert ood_label_col is not None
                ood_idx = ood_source_indices[ood_ptr % len(ood_source_indices)]
                ood_ptr += 1
                ood_ex = ood_ds[ood_idx]
                ood_img_any = ood_ex[ood_image_col]
                if isinstance(ood_img_any, dict) and "image" in ood_img_any:
                    ood_img = ood_img_any["image"]
                else:
                    ood_img = ood_img_any
                if not isinstance(ood_img, Image.Image):
                    raise TypeError(
                        f"Expected PIL.Image.Image for column '{ood_image_col}', got {type(ood_img)}"
                    )
                out_img = ensure_rgb(ood_img)
                source_dataset = ood_dataset_name
                try:
                    ood_source_label = int(ood_ex[ood_label_col])
                except (TypeError, ValueError):
                    ood_source_label = None
                if (
                    ood_source_label is not None
                    and ood_class_names is not None
                    and 0 <= ood_source_label < len(ood_class_names)
                ):
                    ood_source_label_name = ood_class_names[ood_source_label]
            else:
                out_img = img

            save_jpeg(out_img, out_path)
            append_jsonl(
                ood_meta,
                {
                    "file": filename,
                    "label": label,
                    "split": split_name,
                    "variant": v,
                    "poison_type": v,
                    "poison_applied": ood_applied,
                    "poison_frac": cfg.ood_frac,
                    "ood_frac": cfg.ood_frac,
                    "orig_label": label,
                    "label_name": label_name,
                    "source_dataset": source_dataset,
                    "is_ood": ood_applied,
                    "ood_source_label": ood_source_label,
                    "ood_source_label_name": ood_source_label_name,
                },
            )


def load_hf_dataset(repo_id: str, config_name: Optional[str] = None) -> DatasetDict:
    if config_name:
        return load_dataset(repo_id, config_name)
    return load_dataset(repo_id)


def build_splits(ds_dict: DatasetDict, seed: int) -> DatasetDict:
    keys = list(ds_dict.keys())
    if "train" in keys and ("test" in keys or "validation" in keys):
        return ds_dict
    if len(keys) >= 2:
        return ds_dict

    split_name = keys[0]
    split = ds_dict[split_name].train_test_split(test_size=0.2, seed=seed)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def parse_variants(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return VARIANTS

    aliases = {
        "label_flip": "label-flip",
        "label-flipping": "label-flip",
        "stego": "steganography",
        "aug": "augmentation",
        "data_augmentation": "augmentation",
        "out-of-distribution": "ood",
    }
    variants: List[str] = []
    for item in raw.split(","):
        name = aliases.get(item.strip().lower(), item.strip().lower())
        if not name:
            continue
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant '{name}'. Choose from {', '.join(VARIANTS)} or 'all'.")
        if name not in variants:
            variants.append(name)

    if not variants:
        raise ValueError("At least one variant must be selected.")
    return tuple(variants)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="kuchidareo/trashnet_small")
    p.add_argument("--config", type=str, default=None, help="HF dataset config name (if needed)")
    p.add_argument("--ood-dataset", type=str, default="Donghyun99/Oxford-Flower-102")
    p.add_argument("--ood-config", type=str, default=None, help="HF OOD dataset config name (if needed)")
    p.add_argument("--out", type=str, default="data", help="Output root directory")
    p.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Optional cap for train split; test/validation defaults to 20%% of this cap",
    )
    p.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variants to generate, e.g. 'occlusion,augmentation'. Use 'all' for every variant.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--poison-frac",
        type=float,
        default=1.0,
        help="Fraction of each non-clean variant split to actually poison; the rest remain clean.",
    )
    p.add_argument(
        "--ood-frac",
        type=float,
        default=0.3,
        help="Fraction of the OOD variant split replaced by external OOD images.",
    )

    # Blurring
    p.add_argument("--blur-radius", type=float, default=12.0, help="Gaussian blur radius (very strong default)")

    # Occlusion
    p.add_argument("--occlusion-min-area-frac", type=float, default=0.50)
    p.add_argument("--occlusion-max-area-frac", type=float, default=0.80)
    p.add_argument("--occlusion-min-aspect", type=float, default=0.75)
    p.add_argument("--occlusion-max-aspect", type=float, default=1.33)

    # Steganography
    p.add_argument("--stego-payload-frac", type=float, default=0.50)
    p.add_argument("--stego-bit-planes", type=int, default=2)

    return p.parse_args()


def _truncate_metadata_files(out_root: Path, selected_variants: tuple[str, ...]) -> None:
    # Ensure metadata.jsonl starts fresh on each run.
    for v in selected_variants:
        meta = out_root / v / "metadata.jsonl"
        meta.parent.mkdir(parents=True, exist_ok=True)
        with open(meta, "w", encoding="utf-8"):
            pass


def main() -> None:
    args = parse_args()
    selected_variants = parse_variants(args.variants)

    cfg = PreprocessConfig(
        seed=args.seed,
        poison_frac=max(0.0, min(1.0, float(args.poison_frac))),
        blur_radius=args.blur_radius,
        occlusion_min_area_frac=args.occlusion_min_area_frac,
        occlusion_max_area_frac=args.occlusion_max_area_frac,
        occlusion_min_aspect=args.occlusion_min_aspect,
        occlusion_max_aspect=args.occlusion_max_aspect,
        stego_payload_frac=args.stego_payload_frac,
        stego_bit_planes=args.stego_bit_planes,
        ood_frac=max(0.0, min(1.0, float(args.ood_frac))),
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Reset metadata files
    _truncate_metadata_files(out_root, selected_variants)

    ds_dict = build_splits(load_hf_dataset(args.dataset, config_name=args.config), seed=args.seed)
    ood_ds_dict = None
    if "ood" in selected_variants:
        ood_ds_dict = load_hf_dataset(args.ood_dataset, config_name=args.ood_config)

    # Detect columns from the first available split
    first_split = next(iter(ds_dict.keys()))
    image_col = _find_image_column(ds_dict[first_split].features)
    label_col = _find_label_column(ds_dict[first_split].features)
    ood_image_col = None
    ood_label_col = None
    if ood_ds_dict is not None:
        ood_first_split = next(iter(ood_ds_dict.keys()))
        ood_image_col = _find_image_column(ood_ds_dict[ood_first_split].features)
        ood_label_col = _find_label_column(ood_ds_dict[ood_first_split].features)

    print(f"Loaded dataset: {args.dataset}")
    if ood_ds_dict is not None:
        print(f"Loaded OOD dataset: {args.ood_dataset}")
    print(f"Splits: {list(ds_dict.keys())}")
    if ood_ds_dict is not None:
        print(f"OOD splits: {list(ood_ds_dict.keys())}")
    print(f"Detected columns: image='{image_col}', label='{label_col}'")
    if ood_ds_dict is not None:
        print(f"OOD columns: image='{ood_image_col}', label='{ood_label_col}'")
    print(f"Output: {out_root.resolve()}")
    print(f"Variants: {list(selected_variants)}")
    print(f"Poison frac: {cfg.poison_frac}")
    if "ood" in selected_variants:
        print(f"OOD frac: {cfg.ood_frac}")

    for split_name, split_ds in ds_dict.items():
        split_cap = args.max_per_split
        if args.max_per_split is not None and split_name != "train":
            split_cap = max(1, int(round(args.max_per_split * 0.2)))
        print(f"Preparing split '{split_name}' with max samples: {split_cap if split_cap is not None else 'all'}")
        process_split(
            split_ds,
            split_name=split_name,
            out_root=out_root,
            cfg=cfg,
            selected_variants=selected_variants,
            max_per_split=split_cap,
            image_col=image_col,
            label_col=label_col,
            ood_ds=ood_ds_dict[_resolve_matching_split(ood_ds_dict, split_name)] if ood_ds_dict is not None else None,
            ood_image_col=ood_image_col,
            ood_label_col=ood_label_col,
            ood_dataset_name=args.ood_dataset,
        )

    print("Done.")


if __name__ == "__main__":
    main()
