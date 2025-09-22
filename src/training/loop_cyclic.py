# src/training/loop_cyclic.py
import os, math, yaml, argparse, time, itertools
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, default_data_collator
from datasets import load_dataset
from ..modeling.gpt2_qa import GPT2QuantForQA

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_features(tokenizer, max_len, doc_stride):
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
        enc.pop("offset_mapping", None)
        return enc
    return prepare_train

def train_cyclic(base_cfg_path, train_cfg_path, verbose=1):
    base = load_yaml(base_cfg_path)
    train_cfg = load_yaml(train_cfg_path)

    model_name = base.get("model_name", "gpt2")
    max_len    = int(base.get("max_seq_len", 384))
    doc_stride = int(base.get("doc_stride", 128))
    seed = int(base.get("seed", 42))
    use_fp16 = str(base.get("use_fp16", False)).lower() in ("1","true","yes")

    total_steps = int(train_cfg["train"].get("total_steps", 1000))
    lr = float(train_cfg["train"].get("lr", 3e-5))
    warmup_steps = int(train_cfg["train"].get("warmup_steps", 100))
    weight_decay = float(train_cfg["train"].get("weight_decay", 0.01))
    batch_size = int(train_cfg["train"].get("batch_size_per_device", 4))
    grad_accum = int(train_cfg["train"].get("grad_accum_steps", 1))
    clip_grad = float(train_cfg["train"].get("clip_grad", 1.0))

    cyc = train_cfg.get("cyclic", {})
    cycle_every = int(cyc.get("cycle_every_steps", 100))
    order = list(cyc.get("order", []))  # names must match yaml 'name' or filename stem

    switch = train_cfg.get("switchable", {})
    cfg_paths = switch.get("configs", ["configs/bitwidth_uniform.yaml", "configs/bitwidth_mixed_small.yaml"])
    profiles = {}
    for p in cfg_paths:
        y = load_yaml(p)
        name = y.get("name") or os.path.splitext(os.path.basename(p))[0]
        profiles[name] = y

    # sanity check names
    for nm in order:
        if nm not in profiles:
            raise ValueError(f"Profile '{nm}' not found among {list(profiles.keys())}")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    device = pick_device()
    set_seed(seed)

    # data
    ds = load_dataset("squad")
    prep_train = build_features(tok, max_len, doc_stride)
    split = ds["train"].map(prep_train, batched=True, remove_columns=ds["train"].column_names)

    subset_frac = float(os.getenv("SUBSET_FRAC", "1.0"))
    if subset_frac < 1.0:
        n = max(1, int(len(split) * subset_frac))
        split = split.select(range(n))

    dl = DataLoader(
        split,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    # model
    first_prof = dict(profiles[order[0]].get("per_layer_bits", profiles[order[0]]))
    first_prof["name"] = order[0]
    model = GPT2QuantForQA(model_name, first_prof)
    model.backbone.model.config.pad_token_id = tok.pad_token_id
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_fp16 and device.type == "cuda"))

    it = itertools.cycle(dl)
    step = 0
    t0 = time.time()
    model.train()

    while step < total_steps:
        cycle_idx = (step // cycle_every) % len(order)
        active = order[cycle_idx]
        bits = profiles[active].get("per_layer_bits", profiles[active])
        bits = dict(bits); bits["name"] = active
        model.set_bits_profile(bits)

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            batch = next(it)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(use_fp16 and device.type=="cuda")):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"],
                )
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            loss_accum += float(loss.detach()) * grad_accum

        scaler.unscale_(opt)
        if clip_grad and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()

        step += 1
        if (step % 20 == 0) or (step == total_steps):
            elapsed = time.time() - t0
            print(f"[INFO] step {step}/{total_steps} | active_profile={active} | loss={loss_accum:.4f} | elapsed {elapsed:.1f}s", flush=True)

    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = "./checkpoints/last_cyclic.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Saved checkpoint to {ckpt_path}", flush=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", type=str, default="configs/gpt2_base.yaml")
    ap.add_argument("--train_cfg", type=str, default="configs/training.yaml")
    ap.add_argument("--verbose", type=int, default=1)
    return ap.parse_args()

def main():
    args = parse_args()
    train_cyclic(args.base_cfg, args.train_cfg, verbose=args.verbose)

if __name__ == "__main__":
    main()
