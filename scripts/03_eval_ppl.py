import os, math, torch, argparse, yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from src.modeling.gpt2_patch import GPT2QuantModel

def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def pick_device():
    force = os.getenv("EVAL_DEVICE", "").lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if force == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")  # safe default

@torch.no_grad()
def calibrate(model, dl, steps=50, device=None):
    old_force = os.getenv("FORCE_WQ", "0")
    os.environ["FORCE_WQ"] = "1"   
    model.train()
    it = iter(dl)
    for _ in range(min(steps, len(dl))):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(**batch)  
    os.environ["FORCE_WQ"] = old_force
    model.eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", type=str, default="configs/")
    ap.add_argument("--max_eval", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--calib_steps", type=int, default=50)
    args = ap.parse_args()

    base = load_yaml("configs/gpt2_base.yaml")
    model_name = base.get("model_name", "gpt2")
    max_len = int(base.get("max_seq_len", 512))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Model + weights
    model = GPT2QuantModel(model_name, {"default_weight_bits": 8, "default_activation_bits": 8})
    sd = torch.load("./checkpoints/last.pt", map_location="cpu")
    model.load_state_dict(sd, strict=False)
    device = pick_device()
    model.to(device).eval()

    # Build evaluation dataset from SQuAD dev
    ds = load_from_disk("./data/squad")["validation"]
    N = min(args.max_eval, len(ds))
    ds = ds.select(range(N))

    def fmt(ex):
        q = ex["question"]
        c = ex["context"]
        a = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        ex["text"] = f"question: {q}\ncontext: {c}\nanswer: {a}\n"
        return ex

    ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "text"])

    def tok_map(batch):
        return tok(batch["text"], truncation=True, max_length=max_len)

    ds = ds.map(tok_map, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collator, num_workers=0, pin_memory=False)

    # Gather profiles
    if os.path.isdir(args.profiles):
        prof_paths = [os.path.join(args.profiles, fn)
                      for fn in sorted(os.listdir(args.profiles))
                      if fn.endswith(".yaml") and ("uniform" in fn or "mixed" in fn or "bitwidth" in fn)]
    else:
        prof_paths = [args.profiles]

    for pf in prof_paths:
        y = load_yaml(pf)
        name = y.get("name") or os.path.splitext(os.path.basename(pf))[0]
        bits = y.get("per_layer_bits", y)
        bits = dict(bits); bits["name"] = name
        model.set_bits_profile(bits)

        calibrate(model, dl, steps=args.calib_steps, device=device)

        b0 = model.model.transformer.h[0]
        print(f"{name}: qkv ({b0.attn.c_attn.w_bits},{b0.attn.c_attn.a_bits}) "
              f"mlp_in ({b0.mlp.c_fc.w_bits},{b0.mlp.c_fc.a_bits})")

        nll_sum, tok_count = 0.0, 0
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}  # input_ids, attention_mask, labels
                out = model(**batch)
                logits = out.logits[..., :-1, :].contiguous()
                labels = batch["labels"][..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                nll_sum += loss.item()
                tok_count += (labels != -100).sum().item()

        ppl = math.exp(nll_sum / max(tok_count, 1))
        print(f"[{name}] ppl={ppl:.2f}  tokens={tok_count}")

if __name__ == "__main__":
    main()
