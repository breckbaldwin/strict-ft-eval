#!/usr/bin/env python3
"""Standard LoRA fine-tuning for JSON extraction.

Uses PEFT (HuggingFace) for LoRA — no custom adapter code.

Usage:
    # Local (0.5B, CPU/MPS)
    python src/train.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --data data/Restaurants_1_train.jsonl --epochs 5 --device cpu

    # RunPod (7B/32B, CUDA)
    python src/train.py --model Qwen/Qwen2.5-7B-Instruct \
        --data data/Restaurants_1_train.jsonl --epochs 10 --device cuda
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class JsonExtractionDataset(Dataset):
    """Dataset of (prompt, target_json) pairs for causal LM training."""

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 2048):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(data_path) as f:
            for line in f:
                rec = json.loads(line)
                self.examples.append(rec)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        target = ex["target_json"]

        # Tokenize prompt and target
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            target_ids = target_ids + [eos_id]

        input_ids = prompt_ids + target_ids

        # Truncate if needed
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        # Labels: -100 for prompt tokens (don't compute loss), actual ids for target
        labels = [-100] * len(prompt_ids) + target_ids
        if len(labels) > self.max_seq_len:
            labels = labels[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(F.pad(b["input_ids"], (0, pad_len), value=0))
        labels.append(F.pad(b["labels"], (0, pad_len), value=-100))
        mask = torch.ones(len(b["input_ids"]), dtype=torch.long)
        attention_mask.append(F.pad(mask, (0, pad_len), value=0))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def train_epoch(model, dataloader, optimizer, device, epoch, grad_clip=1.0):
    """Train one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        # Count non-masked tokens for reporting
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    elapsed = time.time() - start
    mean_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    print(f"  Epoch {epoch}: loss={mean_loss:.4f} "
          f"tokens={total_tokens} time={elapsed:.1f}s")
    return mean_loss


def main():
    parser = argparse.ArgumentParser(description="Standard LoRA fine-tuning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", required=True, help="Training JSONL file")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-targets", default="q_proj,v_proj",
                        help="Comma-separated LoRA target modules")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-prefix", default="lora")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"LoRA targets: {args.lora_targets}")
    print(f"Device: {args.device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).to(args.device)

    # Apply LoRA
    target_modules = [m.strip() for m in args.lora_targets.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    dataset = JsonExtractionDataset(args.data, tokenizer, args.max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    print(f"Training examples: {len(dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    ckpt_dir = Path(args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, args.device, epoch)

        # Save checkpoint every epoch
        ckpt_path = ckpt_dir / f"{args.checkpoint_prefix}_epoch{epoch}"
        model.save_pretrained(str(ckpt_path))
        print(f"  Saved {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
