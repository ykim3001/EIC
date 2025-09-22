import os, re, string, argparse, yaml, torch
from datasets import load_from_disk
from transformers import AutoTokenizer, GenerationConfig
from ..modeling.gpt2_patch import GPT2QuantModel

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def choose_device():
    force = os.getenv("EVAL_DEVICE", "").lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if force == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())

def exact_match(pred: str, golds) -> int:
    p = normalize(pred)
    if isinstance(golds, (list, tuple)):
        return int(any(p == normalize(g) for g in golds))
    return int(p == normalize(golds))

EXEMPLAR = (
    "Answer the question in 1-3 words. If unanswerable, say 'unknown'.\n"
    "question: Who wrote Hamlet?\n"
    "context: William Shakespeare wrote many plays.\n"
    "answer: William Shakespeare\n\n"
)

def make_prompt(q: str, c: str) -> str:
    return (
        "Answer the question in 1-3 words. If unanswerable, say 'unknown'.\n"
        f"question: {q}\ncontext: {c}\nanswer:"
    )

@torch.no_grad()
def eval_profile(model, tok, ds, profile_name: str, profile_cfg: dict, max_examples: int, device):
    from math import ceil
    import evaluate, re
    gen_cfg = GenerationConfig(
        max_new_tokens=6,   # shorter == faster
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # set bits & eval
    bits = dict(profile_cfg.get("per_layer_bits", profile_cfg))
    bits["name"] = profile_name
    model.set_bits_profile(bits)
    model.eval()

    # Fewer examples for speed
    n = min(max_examples, len(ds))
    ctx_cap = int(os.getenv("EVAL_CTX_CAP", "96"))

    # Pre-build prompts
    prompts = []
    qids, golds_all = [], []
    for i in range(n):
        ex = ds[i]
        q, c = ex["question"], ex["context"]
        golds = ex.get("answers", {}).get("text", [""])
        qid = ex.get("id", str(i))
        enc_c = tok(c, add_special_tokens=False)
        c_short = tok.decode(enc_c["input_ids"][:ctx_cap])
        prompt = (
            "Answer the question in 1-3 words. If unanswerable, say 'unknown'.\n"
            "question: Who wrote Hamlet?\n"
            "context: William Shakespeare wrote many plays.\n"
            "answer: William Shakespeare\n\n"
            f"Answer the question in 1-3 words. If unanswerable, say 'unknown'.\n"
            f"question: {q}\ncontext: {c_short}\nanswer:"
        )
        prompts.append(prompt)
        qids.append(qid)
        golds_all.append(golds)

    # Batch tokenize
    bs = 32  # tune based on VRAM
    preds = []
    for b in range(0, len(prompts), bs):
        chunk = prompts[b:b+bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
        # GPT-2 generate
        out = model.model.generate(**enc, generation_config=gen_cfg)
        # Slice out the generated tails
        input_lens = enc["input_ids"].shape[1]
        gens = tok.batch_decode(out[:, input_lens:], skip_special_tokens=True)
        for g in gens:
            pred = g.split("\n")[0].strip()
            pred = re.split(r"(?:question:|context:|answer:)", pred)[0].strip()
            preds.append(pred)

    # Compute SQuAD metric once
    squad_metric = evaluate.load("squad")
    pred_records = [{"id": qids[i], "prediction_text": preds[i]} for i in range(len(preds))]
    ref_records  = [{"id": qids[i], "answers": {"text": golds_all[i], "answer_start": [0]*len(golds_all[i])}}
                    for i in range(len(preds))]
    scores = squad_metric.compute(predictions=pred_records, references=ref_records)
    em, f1 = scores["exact_match"], scores["f1"]
    print(f"[{profile_name}] EM={em:.2f}  F1={f1:.2f}  (n={len(preds)})")
    return {"profile": profile_name, "n": len(preds), "em": em, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="./checkpoints/last.pt")
    ap.add_argument("--profiles", type=str, default="configs/")  
    ap.add_argument("--max_eval", type=int, default=int(os.getenv("MAX_EVAL", "100")))
    args = ap.parse_args()

    base = load_yaml("configs/gpt2_base.yaml")
    model_name = base.get("model_name", "distilgpt2")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # model + weights
    model = GPT2QuantModel(model_name, {"default_weight_bits": 8, "default_activation_bits": 8})
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)

    device = choose_device()
    model.to(device)

    # dataset (dev split)
    ds = load_from_disk("./data/squad")["validation"]

    # collect profile yamls
    prof_paths = []
    if os.path.isdir(args.profiles):
        for fn in sorted(os.listdir(args.profiles)):
            if fn.endswith(".yaml") and ("uniform" in fn or "mixed" in fn or "bitwidth" in fn):
                prof_paths.append(os.path.join(args.profiles, fn))
    else:
        prof_paths = [args.profiles]

    results = []
    for pth in prof_paths:
        y = load_yaml(pth)
        name = y.get("name") or os.path.splitext(os.path.basename(pth))[0]
        res = eval_profile(model, tok, ds, name, y, max_examples=args.max_eval, device=device)
        results.append(res)

    # summary
    print("\n=== Summary ===")
    hdr = "profile        EM     F1      n"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        em = f"{r['em']:.2f}" if r["em"] is not None else "n/a"
        f1 = f"{r['f1']:.2f}" if r["f1"] is not None else "n/a"
        print(f"{r['profile']:<13} {em:>6}  {f1:>6}  {r['n']:>5}")

if __name__ == "__main__":
    main()
