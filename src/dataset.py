#!/usr/bin/env python3
"""
Dataset module for DotVLM.

Every source (Cauldron, COCO, GQA, VG) is turned into the same plain record:
    {"image_path": str, "texts": [{"user": str, "assistant": str}, ...]}
Records are sampled, then expanded into one (image, question, answer) sample
per QA pair. Image loading and tokenization happen per-sample in the workers
(__getitem__); the collate function only pads and stacks.

Cauldron data must be downloaded first:  python download.py
"""

import json
import logging
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config as cfg
from download import records_path as cauldron_records_path
from utils import load_vocab_mappings, remap_token_ids, smolvlm_tokenizer

log = logging.getLogger(__name__)

transform = transforms.Compose([
    transforms.Resize(size=269, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
])


# ============================================================================
# Building records (one generic path for every dataset)
# ============================================================================

# A record looks like this:
#
# {
#     "image_path": "/mnt/f/datasets/other_datasets/COCO/images/train2017/000000000009.jpg",
#     "texts": [
#         {"user": "What is in the image?", "assistant": "A person riding a horse."},
#         {"user": "What is the person wearing?", "assistant": "A helmet and a jacket."},
#         ... 
#     ]
# }

def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _index_images(*dirs):
    """{filename: full_path} for the given dirs. Earlier dirs win on duplicates.

    One directory listing instead of one os.path.exists() per image, which is
    what made the old path caching feel necessary on the slow /mnt/f mount.
    """
    index = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        with os.scandir(d) as it:
            for entry in it:
                index.setdefault(entry.name, entry.path)
    return index


def _records_from_grouped_json(json_path, image_index, filename_fn, split_fn=None):
    """Grouped QA json -> records. `split_fn` optionally selects a subset of groups."""
    groups = _read_json(json_path)
    if split_fn is not None:
        groups = split_fn(groups)

    records, missing = [], 0
    for qas in groups:
        if not qas:
            continue
        image_path = image_index.get(filename_fn(qas[0]["image_id"]))
        if image_path is None:
            missing += 1
            continue
        records.append({
            "image_path": image_path,
            "texts": [{"user": qa["user"], "assistant": qa["assistant"]} for qa in qas],
        })
    if missing:
        log.warning("%s: %d image groups had no image file", json_path, missing)
    return records


def _vg_split(split):
    """Seeded shuffle, then split by cfg.VG_TRAIN_RATIO."""
    def fn(groups):
        groups = list(groups)
        random.Random(cfg.SEED).shuffle(groups)
        cut = int(len(groups) * cfg.VG_TRAIN_RATIO)
        return groups[:cut] if split == "train" else groups[cut:]
    return fn


def build_records(name, split):
    """Return the full list of records for dataset `name` and `split`."""
    if name in cfg.CAULDRON_SUBSETS:
        path = cauldron_records_path(name, split)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing - run `python download.py` first")
        return _read_json(path)

    if name == "coco_captions":
        index = _index_images(os.path.join(cfg.COCO_IMAGES_DIR, f"{split}2017"))
        return _records_from_grouped_json(
            cfg.COCO_JSON[split], index, lambda i: f"{i:012d}.jpg")

    if name == "gqa":
        index = _index_images(cfg.GQA_IMAGES_DIR)
        return _records_from_grouped_json(
            cfg.GQA_JSON[split], index, lambda i: f"{i}.jpg")

    if name == "vgregqa":
        index = _index_images(
            os.path.join(cfg.VG_IMAGES_DIR, "VG_100K"),
            os.path.join(cfg.VG_IMAGES_DIR, "VG_100K_2"),
        )
        return _records_from_grouped_json(
            cfg.VG_JSON, index, lambda i: f"{i}.jpg", split_fn=_vg_split(split))

    raise ValueError(f"unknown dataset: {name}")


def load_records(split):
    """Sample N images from each dataset (seeded) and mix them together."""
    rng = random.Random(cfg.SEED)
    all_records = []
    for name, n in cfg.SAMPLES_PER_DATASET[split].items():
        records = build_records(name, split)
        rng.shuffle(records)
        records = records[:n]
        log.info("  %-22s %8d images", name, len(records))
        all_records.extend(records)
    rng.shuffle(all_records)
    log.info("  total %s images: %d", split, len(all_records))
    return all_records


# ============================================================================
# Dataset + collate
# ============================================================================

def clean_question(q):
    """Keep only the first line of the question to avoid things like "Keep it brief" instructions."""
    return q.split("\n")[0].strip()


class VQADataset(Dataset):
    """One item per (image, question, answer). Only a small (record, qa) index
    is held in memory; images are decoded and text tokenized on access."""

    def __init__(self, records, old_to_new, max_qa_pairs=10):
        self.records = records
        self.old_to_new = old_to_new
        self.index = np.array(
            [(i, j) for i, r in enumerate(records)
             for j in range(min(len(r["texts"]), max_qa_pairs))],
            dtype=np.int32,
        )
        log.info("  %d images -> %d QA samples", len(records), len(self.index))

    def __len__(self):
        return len(self.index)

    def _encode(self, text, max_len=None):
        ids = smolvlm_tokenizer.encode(text, add_special_tokens=False)
        if max_len is not None:
            ids = ids[:max_len]
        remapped = remap_token_ids(ids, self.old_to_new, unk_id=cfg.UNK_ID)
        return torch.as_tensor(remapped, dtype=torch.long)

    def _load(self, idx):
        rec_i, qa_i = self.index[idx]
        record = self.records[rec_i]
        qa = record["texts"][qa_i]

        image = transform(Image.open(record["image_path"]).convert("RGB"))
        question = self._encode(clean_question(qa["user"]))
        answer = self._encode(qa["assistant"], max_len=cfg.MAX_ANSWER_LEN - 1)

        bos = torch.tensor([cfg.BOS_ID])
        eos = torch.tensor([cfg.EOS_ID])
        return {
            "image": image,
            "question": question,
            "answer_input": torch.cat([bos, answer]),   # BOS + answer
            "answer_target": torch.cat([answer, eos]),  # answer + EOS
        }

    def __getitem__(self, idx):
        # A broken image shouldn't kill a training run: try the next few samples.
        for offset in range(5):
            j = (idx + offset) % len(self)
            try:
                return self._load(j)
            except Exception as e:
                log.warning("sample %d failed (%s), trying next", j, e)
        raise RuntimeError(f"5 consecutive samples failed starting at {idx}")


def collate_batch(batch):
    """Pad and stack. Returns CPU tensors; the training loop moves them."""
    def pad(key):
        return pad_sequence([b[key] for b in batch], batch_first=True,
                            padding_value=cfg.PAD_ID)

    return {
        "images": torch.stack([b["image"] for b in batch]),  # [B, 3, 256, 256]
        "questions": pad("question"),                        # [B, max_q_len]
        "answers_input": pad("answer_input"),                # [B, max_a_len]
        "answers_target": pad("answer_target"),              # [B, max_a_len]
    }


# ============================================================================
# Public entry point
# ============================================================================

def _make_loader(dataset, batch_size, train, num_workers):
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = train
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=train,
        collate_fn=collate_batch,
        drop_last=train,
        **kwargs,
    )


def load_all_datasets(max_qa_pairs=10, batch_size=64, train_workers=4, val_workers=2):
    """Build (trainloader, valloader)."""
    old_to_new, _ = load_vocab_mappings()

    log.info("Training data:")
    train_ds = VQADataset(load_records("train"), old_to_new, max_qa_pairs)
    log.info("Validation data:")
    val_ds = VQADataset(load_records("val"), old_to_new, max_qa_pairs)

    trainloader = _make_loader(train_ds, batch_size, train=True, num_workers=train_workers)
    valloader = _make_loader(val_ds, batch_size, train=False, num_workers=val_workers)
    log.info("train: %d batches | val: %d batches", len(trainloader), len(valloader))
    return trainloader, valloader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    trainloader, _ = load_all_datasets()
    batch = next(iter(trainloader))
    for k, v in batch.items():
        print(f"{k}: {tuple(v.shape)}")