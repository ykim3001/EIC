import os, math, yaml, argparse, time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
from datasets import load_from_disk
from ..modeling.gpt2_patch import GPT2QuantModel

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_dataloader(tok, max_len, batch_size, subset_frac=1.0):
    ds = load_from_disk("./data/squad")
    split = ds["train"]

    def fmt(ex):
        q = ex["question"]
        c = ex["context"]
        a = ex["answers"]["text"][0] if ex.get("answers", {}).get("text") else ""
        ex["text"] = f"question: {q}\ncontext: {c}\nanswer: {a}\n"
        return ex

    split = split.map(fmt, remove_columns=[c for c in split.column_names if c != "text"])

    if subset_frac < 1.0:
        n = max(1, int(len(split) * subset_frac))
        split = split.select(range(n))

    def tok_map(batch):
        return tok(batch["text"], truncation=True, max_length=max_len)

    split = split.map(tok_map, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    dl = DataLoader(
        split,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,      
        pin_memory=False,
        drop_last=True,
    )
    return dl

def cross_entropy_from_logits(logits, labels):
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

def train_cyclic(base_cfg_path, train_cfg_path, verbose=1):
    base = load_yaml(base_cfg_path)
    train_cfg = load_yaml(train_cfg_path)

    model_name = base.get("model_name", "gpt2")
    max_len = int(base.get("max_seq_len", 512))
    seed = int(base.get("seed", 42))
    use_fp16 = str(base.get("use_fp16", False)).lower() in ("1", "true", "yes")

    total_steps = int(train_cfg["train"].get("total_steps", 1000))
    lr = float(train_cfg["train"].get("lr", 2e-4))
    warmup_steps = int(train_cfg["train"].get("warmup_steps", 100))
    weight_decay = float(train_cfg["train"].get("weight_decay", 0.01))
    batch_size = int(train_cfg["train"].get("batch_size_per_device", 2))
    grad_accum = int(train_cfg["train"].get("grad_accum_steps", 1))
    clip_grad = float(train_cfg["train"].get("clip_grad", 1.0))

    cyc = train_cfg.get("cyclic", {})
    cycle_every = int(cyc.get("cycle_every_steps", 100))
    order = list(cyc.get("order", ["uniform8", "mixed8648"]))

    switch = train_cfg.get("switchable", {})
    cfg_paths = switch.get("configs", ["configs/bitwidth_uniform.yaml", "configs/bitwidth_mixed_small.yaml"])
    profiles = {}
    for p in cfg_paths:
        y = load_yaml(p)
        name = y.get("name") or os.path.splitext(os.path.basename(p))[0]
        profiles[name] = y

    for nm in order:
        if nm not in profiles:
            raise ValueError(f"Profile '{nm}' not found among {list(profiles.keys())}")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = pick_device()
    set_seed(seed)

    model = GPT2QuantModel(model_name, {"default_weight_bits": 8, "default_activation_bits": 8})
    model.to(device)

    if verbose:
        print(f"[INFO] device={device.type} | model={model_name} | total_steps={total_steps} | "
              f"cycle_every={cycle_every} | order={order}", flush=True)

    # Data
    subset_frac = float(os.getenv("SUBSET_FRAC", "1.0"))
    dl = build_dataloader(tok, max_len, batch_size, subset_frac=subset_frac)
    it = iter(dl)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_fp16 and device.type == "cuda"))

    step = 0
    losses = []
    t0 = time.time()

    while step < total_steps:
        cycle_idx = (step // cycle_every) % len(order)
        active = order[cycle_idx]
        bits = profiles[active].get("per_layer_bits", profiles[active])
        bits = dict(bits); bits["name"] = active
        model.set_bits_profile(bits)

        for _ in range(grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = {k: v.to(device) for k, v in batch.items()}  # input_ids, attention_mask, labels

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(use_fp16 and device.type != "cpu")):
                out = model(**batch)
                loss = cross_entropy_from_logits(out.logits, batch["labels"]) / grad_accum

            scaler.scale(loss).backward()
            losses.append(loss.item())

        if clip_grad and clip_grad > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sch.step()

        step += 1
        if verbose and (step % 20 == 0 or step == total_steps):
            avg = sum(losses[-20:]) / max(1, min(20, len(losses)))
            elapsed = time.time() - t0
            print(f"[INFO] step {step}/{total_steps} | active_profile={active} | loss={avg:.4f} | elapsed {elapsed:.1f}s",
                  flush=True)

    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = "./checkpoints/last_cyclic.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Saved checkpoint to {ckpt_path}", flush=True)

# CLI
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
