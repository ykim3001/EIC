import os, re, math, random, yaml, argparse, string, torch
from typing import List, Dict
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from ..modeling.gpt2_patch import GPT2QuantModel

def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def choose_device():
    force = os.getenv("EVAL_DEVICE", "").lower()
    if force == "cpu": return torch.device("cpu")
    if force == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if force == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def set_seed(s):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

KEYBOARD_NEIGHBORS = {
    "a":"qwsz", "s":"awedxz", "d":"serfcx", "f":"drtgcv", "g":"ftyhbv", "h":"gyujnb",
    "j":"huikmn", "k":"jiolm,", "l":"kop;.", "q":"was", "w":"qeasd", "e":"wsdr",
    "r":"edft", "t":"rfgy", "y":"tghu", "u":"yjh i", "i":"uok", "o":"ipkl", "p":"ol;",
}

def _rnd_char():
    return random.choice(string.ascii_lowercase)

def attack_swap(text: str) -> str:
    tokens = list(text)
    idxs = [i for i in range(len(tokens)-1) if tokens[i].isalnum() and tokens[i+1].isalnum()]
    if not idxs: return text
    i = random.choice(idxs)
    tokens[i], tokens[i+1] = tokens[i+1], tokens[i]
    return "".join(tokens)

def attack_delete(text: str) -> str:
    idxs = [i for i,c in enumerate(text) if c.isalnum()]
    if not idxs: return text
    i = random.choice(idxs)
    return text[:i] + text[i+1:]

def attack_keyboard_typo(text: str) -> str:
    idxs = [i for i,c in enumerate(text.lower()) if c in KEYBOARD_NEIGHBORS]
    if not idxs: return text
    i = random.choice(idxs)
    orig = text[i]
    neigh = KEYBOARD_NEIGHBORS[orig.lower()]
    repl = random.choice(neigh)
    repl = repl.upper() if orig.isupper() else repl
    return text[:i] + repl + text[i+1:]

ATTACKS = {
    "swap": attack_swap,
    "delete": attack_delete,
    "kbtypo": attack_keyboard_typo,
}

def apply_attacks(text: str, strength: int = 1) -> str:
    out = text
    for _ in range(strength):
        fn = random.choice(list(ATTACKS.values()))
        out = fn(out)
    return out

def build_eval_dl(tok, max_len: int, split, n: int, batch_size: int):
    split = split.select(range(min(n, len(split))))
    def fmt(ex):
        q, c = ex["question"], ex["context"]
        a = ex["answers"]["text"][0] if ex.get("answers",{}).get("text") else ""
        ex["text"] = f"question: {q}\ncontext: {c}\nanswer: {a}\n"
        return ex
    split = split.map(fmt, remove_columns=[c for c in split.column_names if c!="text"])
    def tok_map(batch):
        return tok(batch["text"], truncation=True, max_length=max_len)
    split = split.map(tok_map, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    return DataLoader(split, batch_size=batch_size, shuffle=False,
                      collate_fn=collator, num_workers=0, pin_memory=False)

def build_eval_dl_with_perturb(tok, max_len: int, split, n: int, batch_size: int, strength: int, seed: int):
    rng = random.Random(seed)
    inds = list(range(min(n, len(split))))
    subset = split.select(inds)
    def fmt_adv(ex):
        q, c = ex["question"], ex["context"]
        q2 = apply_attacks(q, strength=strength)
        c2 = apply_attacks(c, strength=strength)
        a = ex["answers"]["text"][0] if ex.get("answers",{}).get("text") else ""
        ex["text"] = f"question: {q2}\ncontext: {c2}\nanswer: {a}\n"
        return ex
    subset = subset.map(fmt_adv, remove_columns=[c for c in subset.column_names if c!="text"])
    def tok_map(batch):
        return tok(batch["text"], truncation=True, max_length=max_len)
    subset = subset.map(tok_map, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      collate_fn=collator, num_workers=0, pin_memory=False)

@torch.no_grad()
def ppl_over_dl(model, dl, device):
    nll, tok_ct = 0.0, 0
    for batch in dl:
        batch = {k: v.to(device) for k,v in batch.items()}
        out = model(**batch)
        logits = out.logits[..., :-1, :].contiguous()
        labels = batch["labels"][..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        nll += loss.item()
        tok_ct += (labels != -100).sum().item()
    ppl = math.exp(nll / max(tok_ct,1))
    return ppl, tok_ct

def set_profile(model, prof_dict: Dict):
    bits = prof_dict.get("per_layer_bits", prof_dict)
    bits = dict(bits); bits["name"] = prof_dict.get("name", bits.get("name","custom"))
    model.set_bits_profile(bits)

def randomize_bits_per_batch(model, profiles: Dict[str,Dict]):
    name = random.choice(list(profiles.keys()))
    set_profile(model, profiles[name])
    return name

@torch.no_grad()
def calibrate(model, dl, steps=30, device=None):
    old = os.getenv("FORCE_WQ","0"); os.environ["FORCE_WQ"]="1"
    model.train()
    it = iter(dl)
    for _ in range(min(steps, len(dl))):
        try: b = next(it)
        except StopIteration: break
        b = {k:v.to(device) for k,v in b.items()}
        _ = model(**b)
    os.environ["FORCE_WQ"]=old
    model.eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="./checkpoints/last.pt")
    ap.add_argument("--profiles", type=str, nargs="+",
                    default=["configs/bitwidth_uniform.yaml","configs/bitwidth_mixed_small.yaml"])
    ap.add_argument("--max_eval", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--attack_strength", type=int, default=2)   # number of atomic edits per sample
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = choose_device()

    base = load_yaml("configs/gpt2_base.yaml")
    model_name = base.get("model_name","gpt2")
    max_len = int(base.get("max_seq_len",512))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = GPT2QuantModel(model_name, {"default_weight_bits":8, "default_activation_bits":8})
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # load profiles
    profs = {}
    for p in args.profiles:
        y = load_yaml(p)
        nm = y.get("name") or os.path.splitext(os.path.basename(p))[0]
        y["name"] = nm
        profs[nm] = y
    assert "uniform8" in profs, "Expect a 'uniform8' profile"
    non_uniform = [k for k in profs if k != "uniform8"]

    # data
    ds = load_from_disk("./data/squad")
    dev = ds["validation"]

    # build loaders
    dl_clean = build_eval_dl(tok, max_len, dev, args.max_eval, args.batch_size)
    dl_adv   = build_eval_dl_with_perturb(tok, max_len, dev, args.max_eval, args.batch_size, strength=args.attack_strength, seed=args.seed)

    # calibration
    # uniform
    set_profile(model, profs["uniform8"])
    calibrate(model, dl_clean, steps=50, device=device)
    ppl_u_clean, toks = ppl_over_dl(model, dl_clean, device)
    ppl_u_adv, _ = ppl_over_dl(model, dl_adv, device)

    mixed_name = non_uniform[0] if non_uniform else "uniform8"
    set_profile(model, profs[mixed_name])
    calibrate(model, dl_clean, steps=50, device=device)
    ppl_m_clean, _ = ppl_over_dl(model, dl_clean, device)
    ppl_m_adv, _ = ppl_over_dl(model, dl_adv, device)

    for nm in profs:
        set_profile(model, profs[nm])
        calibrate(model, dl_clean, steps=20, device=device)

    @torch.no_grad()
    def ppl_random(dl):
        nll, tok_ct = 0.0, 0
        for batch in dl:
            used = randomize_bits_per_batch(model, profs)
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch)
            logits = out.logits[..., :-1, :].contiguous()
            labels = batch["labels"][..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            nll += loss.item()
            tok_ct += (labels != -100).sum().item()
        return math.exp(nll / max(tok_ct,1)), tok_ct

    ppl_r_clean, _ = ppl_random(dl_clean)
    ppl_r_adv,   _ = ppl_random(dl_adv)

    def pct(a,b):  
        return 100.0 * (b - a) / max(1e-9, a)

    print("\n=== Robustness (Perplexity; lower is better) ===")
    print(f"clean | uniform8   : {ppl_u_clean:.2f}")
    print(f"clean | {mixed_name:<10}: {ppl_m_clean:.2f}")
    print(f"clean | random     : {ppl_r_clean:.2f}")
    print("---")
    print(f"adv   | uniform8   : {ppl_u_adv:.2f}  (Δ {pct(ppl_u_clean, ppl_u_adv):+.1f}% vs clean)")
    print(f"adv   | {mixed_name:<10}: {ppl_m_adv:.2f}  (Δ {pct(ppl_m_clean, ppl_m_adv):+.1f}% vs clean)")
    print(f"adv   | random     : {ppl_r_adv:.2f}  (Δ {pct(ppl_r_clean, ppl_r_adv):+.1f}% vs clean)")

if __name__ == "__main__":
    main()
