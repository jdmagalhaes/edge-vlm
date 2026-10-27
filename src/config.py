"""Shared paths and settings for the DotVLM data pipeline."""

import os

import torch

# Root of all datasets.
DATA_ROOT = "/mnt/f/datasets"
CAULDRON_DIR = os.path.join(DATA_ROOT, "cauldron_subsets")
OTHERS_DIR = os.path.join(DATA_ROOT, "other_datasets")
JSON_DIR = "./json_data"

COCO_IMAGES_DIR = os.path.join(OTHERS_DIR, "COCO", "images")
GQA_IMAGES_DIR = os.path.join(OTHERS_DIR, "GQA", "images")
VG_IMAGES_DIR = os.path.join(OTHERS_DIR, "VGREGQA", "images")

COCO_JSON = {
    "train": os.path.join(JSON_DIR, "coco_grouped_train.json"),
    "val": os.path.join(JSON_DIR, "coco_grouped_val.json"),
}
GQA_JSON = {
    "train": os.path.join(JSON_DIR, "gqa_grouped_train.json"),
    "val": os.path.join(JSON_DIR, "gqa_grouped_val.json"),
}
VG_JSON = os.path.join(JSON_DIR, "vg_grouped.json")  # split 80/20 in code
VG_TRAIN_RATIO = 0.8

CHECKPOINT_PATH = "./checkpoints/best_model.pt"
CHECKPOINT_FT_PATH = "./checkpoints/best_model_ft.pt"
VOCAB_MAPPING_PATH = "./vocab/DotVLM_vocab_mapping_8k.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of images taken from each dataset (each image has several QA pairs).
SAMPLES_PER_DATASET = {
    "train": {
        "localized_narratives": 180_000,
        "tallyqa": 12_000,
        "coco_captions": 120_000,
        "gqa": 80_000,
        "vgregqa": 80_000,
    },
    "val": {
        "localized_narratives": 18_000,
        "tallyqa": 1_200,
        "coco_captions": 12_000,
        "gqa": 8_000,
        "vgregqa": 8_000,
    },
}

CAULDRON_SUBSETS = ("localized_narratives", "tallyqa")

# Tokens / lengths
PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3
MAX_ANSWER_LEN = 176  # max tokens in an answer, including EOS

SEED = 0