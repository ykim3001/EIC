import os, sys, time, math, yaml, itertools
from dataclasses import dataclass
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (AutoTokenizer, get_linear_schedule_with_warmup,
                          DataCollatorForLanguageModeling)

from ..utils.logging import set_seed, get_logger
from ..modeling.gpt2_patch import GPT2QuantModel

def load_squad_text_dataset(path="./data/squad"):
    ds = load_from_disk(path)
    def _fmt(example):
        q = example.get("question","").strip()
        c = example.get("context","").strip()
        ans_list = example.get("answers",{}).get("text", [])
        a = (ans_list[0] if ans_list else "").strip()
        example["text"] = f"question: {q}\ncontext: {c}\nanswer: {a}\n"
        return example
    ds = ds.map(_fmt, remove_columns=[c for c in ds["train"].column_names if c != "text"])
    return ds

def tokenize_ds(ds, tokenizer, max_len=512):
    def _tok(batch):
        return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_len,
        padding="max_length",
        )
    return ds.map(_tok, batched=True, remove_columns=["text"])

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")

def load_yaml(p): 
    with open(p, "r") as f: 
        return yaml.safe_load(f)

def train_switchable(model, dataloader, profiles, total_steps, lr, warmup, wd, grad_accum, device, logger, clip=1.0):
    model.to(device)
    model.train()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    step = 0
    data_iter = itertools.cycle(dataloader)
    prof_cycle = itertools.cycle(profiles)

    t0 = time.time()
    while step < total_steps:
        pname, prof = next(prof_cycle)
        model.set_bits_profile(prof)

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            batch = next(data_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            out = model(**batch)             
            logits = out.logits               
            labels = batch["labels"]         

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            (loss / grad_accum).backward()
            loss_accum += float(loss.detach())

        clip_grad_norm_(model.parameters(), clip)
        opt.step(); sch.step()
        step += 1

        if step % 20 == 0 or step == total_steps:
            dt = time.time() - t0
            logger.info(f"step {step}/{total_steps} | profile={pname} | loss={loss_accum:.4f} | tok/s ~ (n/a) | elapsed {dt:.1f}s")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/last.pt")
    logger.info("Saved checkpoint to ./checkpoints/last.pt")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", type=str, required=True)
    ap.add_argument("--train_cfg", type=str, required=True)
    args = ap.parse_args()

    base = load_yaml(args.base_cfg)
    train = load_yaml(args.train_cfg)

    logger = get_logger("eic")
    set_seed(base.get("seed", 42))

    model_name = base.get("model_name", "gpt2")
    max_len = base.get("max_seq_len", 512)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    ds = load_squad_text_dataset("./data/squad")
    ds_tok = tokenize_ds(ds, tok, max_len=max_len)
    frac = float(os.environ.get("SUBSET_FRAC", "0.2"))  # 20%
    n = int(len(ds_tok["train"]) * frac)
    ds_tok["train"] = ds_tok["train"].select(range(n))
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    bs = train["train"].get("batch_size_per_device", 4)
    dl = DataLoader(
        ds_tok["train"],
        batch_size=bs,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=0,          
        pin_memory=False,     
    )

    prof_files = train["switchable"]["configs"]
    profiles = []
    for pf in prof_files:
        p = load_yaml(pf)
        name = p.get("name") or os.path.splitext(os.path.basename(pf))[0]
        prof = p.get("per_layer_bits", p)
        prof["name"] = name
        profiles.append((name, prof))

    init_prof = profiles[0][1]
    model = GPT2QuantModel(model_name, init_prof)

    device = pick_device()
    logger.info(f"device={device} | model={model_name} | steps={train['train']['total_steps']} | profiles={[n for n,_ in profiles]}")

    train_switchable(
        model=model,
        dataloader=dl,
        profiles=profiles,
        total_steps=train["train"]["total_steps"],
        lr=train["train"].get("lr", 2e-4),
        warmup=train["train"].get("warmup_steps", 100),
        wd=train["train"].get("weight_decay", 0.01),
        grad_accum=train["train"].get("grad_accum_steps", 2),
        device=device,
        logger=logger,
        clip=train["train"].get("clip_grad", 1.0),
    )

if __name__ == "__main__":
    main()