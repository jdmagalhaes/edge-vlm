#!/usr/bin/env python3

import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import config as cfg
from dataset import load_all_datasets
from model import DotVLM
from train import train_one_epoch, validate, set_seed, BATCH_SIZE

LEARNING_RATE = 2e-5
BACKBONE_LR = 5e-6
EPOCHS = 15
WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.05
PATIENCE = 3

INIT_CKPT = Path("checkpoints/best_model.pt")      # stage 1 weights (read only)
OUT_CKPT = Path("checkpoints/best_model_ft.pt")    # separate file, no overwrite


def main():
    print("Fine-tuning DotVLM on all datasets, but prioritizing localized_narratives by decreasing the number of QA pairs per image (from 10 to 3).")
    set_seed()
    device = cfg.DEVICE

    model = DotVLM(dropout=0.10).to(device)
    ckpt = torch.load(INIT_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    best_val_loss_stage1 = ckpt["val_loss"]
    del ckpt

    train_loader, val_loader = load_all_datasets(max_qa_pairs=3, batch_size=BATCH_SIZE)

    backbone = list(model.vision_encoder.model.parameters())
    ids = {id(p) for p in backbone}
    others = [p for p in model.parameters() if id(p) not in ids]

    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.PAD_ID, reduction="mean")
    baseline = validate(model, val_loader, loss_fn, device)
    print(f"Val loss of stage 1 on new data: {baseline:.4f}")
    val_loss = baseline
    saved = False
    optimizer = AdamW([
        {"params": backbone, "lr": BACKBONE_LR},
        {"params": others, "lr": LEARNING_RATE},
    ])

    warmup = WARMUP_EPOCHS * len(train_loader)
    total = EPOCHS * len(train_loader)

    def lr_mult(step):
        if step < warmup:
            return 0.01 + 0.99 * step / max(warmup, 1)
        p = min((step - warmup) / max(total - warmup, 1), 1.0)
        return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_mult)

    patience_counter = 0
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        va = validate(model, val_loader, loss_fn, device)
        print(f"FT epoch {epoch}/{EPOCHS} - train {tr:.4f} - val {va:.4f}")

        if va < val_loss:
            val_loss = va
            saved = True
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "train_loss": tr, "val_loss": va}, OUT_CKPT)
            print(f"Saved {OUT_CKPT}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    if saved:
        print(f"Best val loss: {val_loss:.4f} (baseline on new data: {baseline:.4f}; stage 1: {best_val_loss_stage1:.4f})")
    else:
        print(f"No improvement over baseline {baseline:.4f}")

if __name__ == "__main__":
    main()