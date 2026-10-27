#!/usr/bin/env python3

import math
import random
import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import config as cfg
from dataset import load_all_datasets
from model import DotVLM


BATCH_SIZE = 64
LEARNING_RATE = 2e-5
BACKBONE_LR = 2e-7
EPOCHS = 25
PATIENCE = 3
WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.05
CHECKPOINT_DIR = ".checkpoints/"
RESUME = True # Resume from the last checkpoint or not


def set_seed(seed: int = cfg.SEED) -> None:
    """Make data shuffling and model initialization repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_for_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Run the model for one batch and return loss plus valid token count."""
    images = batch["images"].to(device)
    questions = batch["questions"].to(device)
    answers_input = batch["answers_input"].to(device)
    answers_target = batch["answers_target"].to(device)

    logits = model(images, questions, answers_input)
    loss = loss_fn(
        logits.reshape(-1, logits.shape[-1]),
        answers_target.reshape(-1),
    )
    num_tokens = (answers_target != cfg.PAD_ID).sum().item()
    return loss, num_tokens


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return the average batch loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        optimizer.zero_grad()
        loss, num_tokens = loss_for_batch(model, batch, loss_fn, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        progress.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{total_loss / total_tokens:.4f}",
            lr=f"{optimizer.param_groups[1]['lr']:.2e}",
        )

    return total_loss / total_tokens if total_tokens else float("inf")


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model and return the average validation batch loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc="Validation", leave=False)
    for batch in progress:
        loss, num_tokens = loss_for_batch(model, batch, loss_fn, device)
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        progress.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{total_loss / total_tokens:.4f}",
        )

    return total_loss / total_tokens if total_tokens else float("inf")


def main() -> None:
    set_seed()
    device = cfg.DEVICE
    print(f"Using device: {device}")

    model = DotVLM(
        question_vocab_size=8000,
        answer_vocab_size=8000,
        vision_dim=512,
        num_queries=32,
        num_decoder_blocks=7,
        dropout=0.10,
    ).to(device)

    train_loader, val_loader = load_all_datasets(
        max_qa_pairs=10,
        batch_size=BATCH_SIZE,
    )

    backbone_params = list(model.vision_encoder.model.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_params}

    other_params = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids
    ]

    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.PAD_ID, reduction="mean")
    optimizer = AdamW(
        [
            {
                "params": backbone_params, 
                "lr": BACKBONE_LR
            },
            {
                "params": other_params,
                "lr": LEARNING_RATE
            },
        ]
    )
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    total_steps = EPOCHS * len(train_loader)

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return 0.01 + 0.99 * step / max(warmup_steps, 1)

        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_multiplier)
    print(
        f"Learning-rate schedule: {WARMUP_EPOCHS} warmup epoch, "
        f"cosine decay to {MIN_LR_RATIO:.0%} of each base rate"
    )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    resume_path = CHECKPOINT_DIR / "best_model.pt"
    if RESUME and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_loss = checkpoint["val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {resume_path} at epoch {checkpoint['epoch']} with val loss {best_val_loss:.4f}")
        del checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for epoch in tqdm(range(start_epoch, EPOCHS + 1), desc="Epochs"):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device
        )
        val_loss = validate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch}/{EPOCHS} - "
            f"train loss: {train_loss:.4f} - "
            f"val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved {checkpoint_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
