# src/training/loop_switchable.py
import os, sys, time, yaml, itertools
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer, get_linear_schedule_with_warmup,
    default_data_collator
)
from ..utils.logging import set_seed, get_logger
from ..modeling.gpt2_qa import GPT2QuantForQA

def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")

def build_features(tokenizer, max_len, doc_stride):
    # train set: create start/end positions; drop offset_mapping
    def prepare_train(ex):
        q = [s.strip() for s in ex["question"]]
        c = ex["context"]
        enc = tokenizer(
            q, c,
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = enc.pop("overflow_to_sample_mapping")
        offsets = enc["offset_mapping"]
        seq_ids = [enc.sequence_ids(i) for i in range(len(enc["input_ids"]))]

        sp, ep = [], []
        for i, off in enumerate(offsets):
            cls_idx = 0
            ex_idx = sample_map[i]
            ans = ex["answers"][ex_idx]
            if len(ans["answer_start"]) == 0:
                sp.append(cls_idx); ep.append(cls_idx); continue
            s_char = ans["answer_start"][0]
            e_char = s_char + len(ans["text"][0])
            ids = seq_ids[i]
            k = 0
            while k < len(ids) and ids[k] != 1: k += 1
            c_start = k
            while k < len(ids) and ids[k] == 1: k += 1
            c_end = k - 1
            if not (off[c_start][0] <= s_char and off[c_end][1] >= e_char):
                sp.append(cls_idx); ep.append(cls_idx)
            else:
                si, ei = c_start, c_end
                while si <= c_end and off[si][0] <= s_char: si += 1
                while ei >= c_start and off[ei][1] >= e_char: ei -= 1
                sp.append(si - 1); ep.append(ei + 1)
        enc["start_positions"] = sp
        enc["end_positions"]   = ep
        enc.pop("offset_mapping", None)   # important: not needed for training
        return enc

    # val set: keep offset_mapping masked to context (for postprocess)
    def prepare_val(ex):
        q = [s.strip() for s in ex["question"]]
        c = ex["context"]
        enc = tokenizer(
            q, c,
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = enc.pop("overflow_to_sample_mapping")
        enc["example_id"] = [ex["id"][i] for i in sample_map]
        for i in range(len(enc["input_ids"])):
            ids = enc.sequence_ids(i)
            enc["offset_mapping"][i] = [
                (o if ids[k] == 1 else None)
                for k, o in enumerate(enc["offset_mapping"][i])
            ]
        return enc
    return prepare_train, prepare_val

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", type=str, required=True)
    ap.add_argument("--train_cfg", type=str, required=True)
    args = ap.parse_args()

    base = load_yaml(args.base_cfg)
    train = load_yaml(args.train_cfg)

    logger = get_logger("eic-span")
    set_seed(base.get("seed", 42))

    model_name = base.get("model_name", "gpt2")
    max_len    = int(base.get("max_seq_len", 384))
    doc_stride = int(base.get("doc_stride", 128))

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # for extractive QA

    ds = load_dataset("squad")
    prep_train, prep_val = build_features(tok, max_len, doc_stride)
    train_features = ds["train"].map(prep_train, batched=True, remove_columns=ds["train"].column_names)
    # (Optional) sub-sample for speed during dev:
    frac = float(os.environ.get("SUBSET_FRAC", "1.0"))
    if frac < 1.0:
        n = int(len(train_features) * frac)
        train_features = train_features.select(range(n))

    collator = default_data_collator
    bs = int(train["train"].get("batch_size_per_device", 4))

    dl = DataLoader(
        train_features,
        batch_size=bs,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )

    # Build profiles
    prof_files = train["switchable"]["configs"]
    profiles = []
    for pf in prof_files:
        y = load_yaml(pf)
        name = y.get("name") or os.path.splitext(os.path.basename(pf))[0]
        prof = y.get("per_layer_bits", y)
        prof["name"] = name
        profiles.append((name, prof))

    # Model: quant backbone + span head
    init_prof = profiles[0][1]
    model = GPT2QuantForQA(model_name=model_name, profile=init_prof)
    model.backbone.model.config.pad_token_id = tok.pad_token_id

    device = pick_device()
    model.to(device)
    logger.info(f"device={device} | model={model_name} | steps={train['train']['total_steps']} | profiles={[n for n,_ in profiles]}")

    # Optim & sched
    total_steps = int(train["train"]["total_steps"])
    lr          = float(train["train"].get("lr", 3e-5))
    wd          = float(train["train"].get("weight_decay", 0.01))
    warmup      = int(train["train"].get("warmup_steps", 100))
    grad_accum  = int(train["train"].get("grad_accum_steps", 1))
    clip        = float(train["train"].get("clip_grad", 1.0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    step = 0
    data_iter = itertools.cycle(dl)
    prof_cycle = itertools.cycle(profiles)
    t0 = time.time()

    use_fp16 = str(base.get("use_fp16", False)).lower() in ("1","true","yes") and device.type=="cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    model.train()
    while step < total_steps:
        pname, prof = next(prof_cycle)
        model.set_bits_profile(prof)

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(grad_accum):
            batch = next(data_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_fp16):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"],
                )
                loss = out.loss / grad_accum

            scaler.scale(loss).backward()
            loss_accum += float(loss.detach()) * grad_accum

        if clip and clip > 0:
            scaler.unscale_(opt)
            clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt); scaler.update()
        sch.step()

        step += 1
        if step % 20 == 0 or step == total_steps:
            dt = time.time() - t0
            logger.info(f"step {step}/{total_steps} | profile={pname} | loss={loss_accum:.4f} | elapsed {dt:.1f}s")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/last.pt")
    logger.info("Saved checkpoint to ./checkpoints/last.pt")

if __name__ == "__main__":
    main()
