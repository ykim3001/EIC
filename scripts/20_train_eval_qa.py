#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
from typing import Dict, List, Union, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

# Try your custom GPT-2 QA head
USE_CUSTOM_GPT2_QA = False
try:
    from src.modeling.gpt2_qa import GPT2QuantForQA  # your repo module
    USE_CUSTOM_GPT2_QA = True
except Exception:
    GPT2QuantForQA = None  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--train_steps", type=int, default=3500)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--eval_samples", type=int, default=1000)
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--force_fallback", action="store_true",
                   help="Ignore custom class and use a generic QA model (DistilBERT).")
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    return torch.device("cuda" if choice == "cuda" and torch.cuda.is_available() else "cpu")


def decide_backend(model_name: str, force_fallback: bool) -> Tuple[str, str, str]:
    """
    Decide which (model_class, model_ckpt, tokenizer_ckpt) to use.
    If we fall back, BOTH model and tokenizer switch to DistilBERT.
    """
    use_custom = (not force_fallback) and USE_CUSTOM_GPT2_QA and ("gpt2" in model_name.lower())
    if use_custom:
        # Use your custom GPT-2 QA with a GPT-2 tokenizer
        return "custom_gpt2qa", model_name, model_name
    else:
        # Generic QA fallback (DistilBERT) with matching tokenizer
        return "distilbert_qa", "distilbert-base-uncased", "distilbert-base-uncased"


def ensure_pad_token(tokenizer: AutoTokenizer):
    if tokenizer.pad_token_id is None:
        # For GPT-2 style tokenizers, reuse EOS as PAD
        tokenizer.pad_token = tokenizer.eos_token


def attach_backbone_if_possible(model, base_backbone):
    for attr in ("backbone", "transformer", "gpt2", "base_model"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, base_backbone)
                return True
            except Exception:
                pass
    return False


def build_model(model_kind: str, model_ckpt: str, device: torch.device):
    if model_kind == "custom_gpt2qa":
        # Robust constructor for custom class that may not implement from_pretrained
        cfg = AutoConfig.from_pretrained(model_ckpt)
        base_causal = AutoModelForCausalLM.from_pretrained(model_ckpt)
        backbone = getattr(base_causal, "transformer", getattr(base_causal, "base_model", base_causal))

        if hasattr(GPT2QuantForQA, "from_pretrained"):
            try:
                model = GPT2QuantForQA.from_pretrained(model_ckpt)
                model.to(device)
                return model
            except Exception as e:
                print(f"[WARN] GPT2QuantForQA.from_pretrained failed: {e}. Trying manual wiring...")

        tried = []
        try:
            model = GPT2QuantForQA(cfg, backbone=backbone)
            model.to(device); return model
        except Exception as e:
            tried.append(f"(config, backbone): {e}")
        try:
            model = GPT2QuantForQA(cfg)
            if not attach_backbone_if_possible(model, backbone):
                print("[WARN] Could not auto-attach backbone after (config) init.")
            model.to(device); return model
        except Exception as e:
            tried.append(f"(config): {e}")
        try:
            model = GPT2QuantForQA()
            if not attach_backbone_if_possible(model, backbone):
                print("[WARN] Could not auto-attach backbone after no-arg init.")
            model.to(device); return model
        except Exception as e:
            tried.append(f"(): {e}")

        raise RuntimeError("Failed to construct GPT2QuantForQA:\n  - " + "\n  - ".join(tried))

    else:
        print("[WARN] Falling back to generic QA model 'distilbert-base-uncased'.")
        model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
        model.to(device)
        return model


def preprocess_squad(tokenizer, max_len: int, doc_stride: int):
    squad = load_dataset("squad")

    def _prep(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_len,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        start_positions, end_positions = [], []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            ans = examples["answers"][sample_idx]
            if len(ans["answer_start"]) == 0:
                start_positions.append(0); end_positions.append(0); continue

            start_char = ans["answer_start"][0]
            end_char = start_char + len(ans["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # find context token span
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1: idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1: idx += 1
            context_end = idx - 1

            if context_start > context_end:
                start_positions.append(0); end_positions.append(0); continue

            s_pos = e_pos = 0
            found = False
            for t in range(context_start, context_end + 1):
                s, e = offsets[t]
                if s <= start_char < e: s_pos = t
                if s < end_char <= e:
                    e_pos = t; found = True; break
            if not found:
                start_positions.append(0); end_positions.append(0)
            else:
                start_positions.append(s_pos); end_positions.append(e_pos)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    print("Map: train…")
    train_dataset = squad["train"].map(
        _prep, batched=True, remove_columns=squad["train"].column_names, desc="Map train"
    )
    print("Map: eval…")
    eval_dataset = squad["validation"].map(
        _prep, batched=True, remove_columns=squad["validation"].column_names, desc="Map eval"
    )
    return train_dataset, eval_dataset



def to_device(batch: Dict[str, Union[torch.Tensor, List, int]], device: torch.device):
    out = {}
    # move tensors / make label tensors on device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif k in ("start_positions", "end_positions"):
            out[k] = torch.as_tensor(v, dtype=torch.long, device=device)
        else:
            out[k] = v
    # clamp labels into [0, seq_len-1] to avoid OOB when doc span mapping fails
    if "input_ids" in out:
        seq_len = out["input_ids"].size(1)
        for name in ("start_positions", "end_positions"):
            if name in out:
                out[name] = out[name].clamp_(0, seq_len - 1)
    return out


def main():
    args = parse_args()

    # Optional: quieter TF logs (those cuDNN/cuBLAS “already registered” lines are harmless)
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    set_seed(args.seed)
    device = resolve_device(args.device)

    # Decide backend first, so tokenizer matches the model
    model_kind, model_ckpt, tokenizer_ckpt = decide_backend(args.model_name, args.force_fallback)

    # Load tokenizer that MATCHES the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt, use_fast=True)
    ensure_pad_token(tokenizer)

    # Build model
    model = build_model(model_kind, model_ckpt, device)

    # Datasets (tokenized with the matching tokenizer)
    train_ds, eval_ds = preprocess_squad(tokenizer, args.max_len, args.doc_stride)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=default_data_collator, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds.select(range(min(args.eval_samples, len(eval_ds)))),
        batch_size=min(args.batch_size * 2, 32),
        shuffle=False, collate_fn=default_data_collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.train_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- Train ----
    model.train()
    step = 0
    accum = args.grad_accum
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        for batch in train_loader:
            batch = to_device(batch, device)
            outputs = model(**batch)
            (outputs.loss / accum).backward()

            if (step + 1) % accum == 0:
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 100 == 0 or step == total_steps:
                print(f"[train] step {step}/{total_steps} | loss={outputs.loss.item():.4f}")
            if step >= total_steps:
                break

    # ---- Eval ----
    model.eval()
    eval_loss_sum, eval_count = 0.0, 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = to_device(batch, device)
            out = model(**batch)
            eval_loss_sum += out.loss.item(); eval_count += 1

    print(f"[eval] batches={eval_count} | avg_loss={eval_loss_sum / max(1, eval_count):.4f}")


if __name__ == "__main__":
    main()
