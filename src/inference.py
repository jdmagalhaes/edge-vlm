#!/usr/bin/env python3
import torch
from torchvision import transforms
from PIL import Image

from model import DotVLM
import config as cfg
import utils


def load_model(checkpoint_path=cfg.CHECKPOINT_FT_PATH, device=cfg.DEVICE):
    model = DotVLM()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded: epoch {checkpoint.get('epoch', 'unknown')}, "
          f"val loss: {checkpoint.get('val_loss', 'unknown')}")
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size=269, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
    ])
    return transform(image).unsqueeze(0)  # [1, 3, 256, 256]


def encode_question(question, tokenizer, old_to_new):
    # Must match VQADataset._encode: no special tokens, same unk handling.
    ids = tokenizer.encode(question, add_special_tokens=False)
    remapped = utils.remap_token_ids(ids, old_to_new, unk_id=cfg.UNK_ID)
    return torch.as_tensor(remapped, dtype=torch.long).unsqueeze(0)  # [1, seq_len]


def decode_answer(answer_token_ids, tokenizer, new_to_old):
    ids = answer_token_ids.tolist()
    if cfg.EOS_ID in ids:
        ids = ids[:ids.index(cfg.EOS_ID)]
    ids = [t for t in ids if t != cfg.PAD_ID]
    smolvlm_ids = [new_to_old.get(t, tokenizer.unk_token_id) for t in ids]
    return tokenizer.decode(smolvlm_ids, skip_special_tokens=True)


def generate_answer(model, image_tensor, question_tensor, device=cfg.DEVICE):
    image_tensor = image_tensor.to(device)
    question_tensor = question_tensor.to(device)
    with torch.no_grad():
        out = model.generate(
            image_tensor,
            question_tensor,
            max_length=cfg.MAX_ANSWER_LEN,
            bos_id=cfg.BOS_ID,
            eos_id=cfg.EOS_ID,
            temperature=0.7,
            top_k=5,  # greedy
        )
    return out.squeeze(0)


def answer_question(image_path, question, model, tokenizer, old_to_new, new_to_old, device=cfg.DEVICE):
    image_tensor = preprocess_image(image_path)
    question_tensor = encode_question(question, tokenizer, old_to_new)
    answer_ids = generate_answer(model, image_tensor, question_tensor, device)
    return decode_answer(answer_ids, tokenizer, new_to_old)


def main():
    print("Loading model...")
    model = load_model()
    tokenizer = utils.smolvlm_tokenizer
    old_to_new, new_to_old = utils.load_vocab_mappings()

    image_path = "test_data/dog1.jpg" # Set your own image path here
    question = "Describe this image."

    answer = answer_question(image_path, question, model, tokenizer, old_to_new, new_to_old)
    print("Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()