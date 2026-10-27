#!/usr/bin/env python3

import json
import torch
from transformers import AutoTokenizer
import config as cfg

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    return tokenizer

smolvlm_tokenizer = load_tokenizer()

def load_vocab_mappings(vocab_file=cfg.VOCAB_MAPPING_PATH):
    with open(vocab_file, 'r') as f:
        vocab_data = json.load(f)
        old_to_new = {int(k): v for k, v in vocab_data['old_to_new'].items()}
        new_to_old = {int(k): v for k, v in vocab_data['new_to_old'].items()}
    return old_to_new, new_to_old

def remap_token_ids(smolvlm_token_ids, old_to_new, unk_id=cfg.UNK_ID):
    
    if isinstance(smolvlm_token_ids, torch.Tensor):
        smolvlm_token_ids = smolvlm_token_ids.tolist()

    remap = [old_to_new.get(tid, unk_id) for tid in smolvlm_token_ids]

    return torch.tensor(remap, dtype=torch.long)

if __name__ == "__main__":
    old_to_new, new_to_old = load_vocab_mappings()
    question = "What color is the car?\nPlease answer in one word."
    cleaned_question = question.split('\n')[0].strip()
    print("Original Question: ", question)
    print("Cleaned Question: ", cleaned_question)
    encode_question = smolvlm_tokenizer.encode(cleaned_question)
    print("SmolVLM Encoded Question IDs: ", encode_question)
    print("Remapped Question IDs: ", remap_token_ids(encode_question, old_to_new))



    print("UNK token:", smolvlm_tokenizer.unk_token, smolvlm_tokenizer.unk_token_id)
    print("Remapping UNK token ID:", remap_token_ids([smolvlm_tokenizer.unk_token_id], old_to_new))