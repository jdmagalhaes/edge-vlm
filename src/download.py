#!/usr/bin/env python3
"""
One-time download of the Cauldron subsets (localized_narratives, tallyqa).

Each subset/split is written as:
    <CAULDRON_DIR>/<name>_<split>_images/000000.jpg ...
    <CAULDRON_DIR>/<name>_<split>.json     -> [{"image_path": ..., "texts": [{"user", "assistant"}, ...]}, ...]

That is the same record format dataset.py builds for COCO / GQA / VG, so the
training code treats every dataset identically.

Usage:
    python download.py             # download anything that is missing
"""

import argparse
import json
import logging
import os

from tqdm import tqdm

from config import CAULDRON_DIR, CAULDRON_SUBSETS, SAMPLES_PER_DATASET

log = logging.getLogger(__name__)


def records_path(name, split):
    return os.path.join(CAULDRON_DIR, f"{name}_{split}.json")


def images_dir(name, split):
    return os.path.join(CAULDRON_DIR, f"{name}_{split}_images")


def _write_json_atomic(records, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, path)  # never leaves a half-written file behind


def _clean_texts(texts):
    return [{"user": t["user"], "assistant": t["assistant"]} for t in texts]


def download_subset(name, split, skip, take):
    out_json = records_path(name, split)
    if os.path.exists(out_json):
        log.info("%s %s already present, skipping", name, split)
        return

    from datasets import load_dataset  # imported lazily: only needed here

    img_dir = images_dir(name, split)
    os.makedirs(img_dir, exist_ok=True)

    stream = load_dataset(
        "HuggingFaceM4/the_cauldron", name, streaming=True, split="train"
    ).skip(skip).take(take)

    records = []
    for idx, item in enumerate(tqdm(stream, total=take, desc=f"{name} {split}")):
        images = item["images"]
        image = images[0] if isinstance(images, list) else images
        image_path = os.path.join(img_dir, f"{idx:06d}.jpg")
        if not os.path.exists(image_path):
            image.convert("RGB").save(image_path, "JPEG", quality=95)
        records.append({"image_path": image_path, "texts": _clean_texts(item["texts"])})

    _write_json_atomic(records, out_json)
    log.info("Saved %d %s %s records", len(records), name, split)


def download_all():
    os.makedirs(CAULDRON_DIR, exist_ok=True)
    for name in CAULDRON_SUBSETS:
        n_val = SAMPLES_PER_DATASET["val"][name]
        n_train = SAMPLES_PER_DATASET["train"][name]
        # val = first n_val examples, train = the n_train examples after them
        download_subset(name, "val", skip=0, take=n_val)
        download_subset(name, "train", skip=n_val, take=n_train)


def migrate_legacy_caches():
    """Turn the old '<name>_<split>_cached_paths' HF datasets into JSON files."""
    from datasets import load_from_disk

    for name in CAULDRON_SUBSETS:
        for split in ("train", "val"):
            out_json = records_path(name, split)
            legacy = os.path.join(CAULDRON_DIR, f"{name}_{split}_cached_paths")
            if os.path.exists(out_json) or not os.path.exists(legacy):
                continue
            ds = load_from_disk(legacy)
            records = [
                {"image_path": r["image_path"], "texts": _clean_texts(r["texts"])}
                for r in tqdm(ds, desc=f"migrate {name} {split}")
            ]
            _write_json_atomic(records, out_json)
            log.info("Migrated %d records -> %s", len(records), out_json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_all()  # skips whatever already exists