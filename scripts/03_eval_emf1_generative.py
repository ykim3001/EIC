import os, argparse, json, re, string
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

metric = evaluate.load("squad")

EXEMPLAR = (
    "Answer the question in 1-3 words. If unanswerable, say 'unknown'.\n"
    "question: Who wrote Hamlet?\n"
    "context: William Shakespeare wrote many plays.\n"
    "answer: William Shakespeare\n\n"
)

def make_prompt(q, c):
    return (
        "Answer the question in 1-3 words. If unanswerable, say 'unknown'.\n"
        f"question: {q}\ncontext: {c}\nanswer:"
    )

_rm_articles = re.compile(r"\b(a|an|the)\b")
def normalize(text):
    if text is None: return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _rm_articles.sub(" ", text)
    text = " ".join(text.split())
    return text

def snap_to_context(pred, context):
    p = normalize(pred)
    if not p: return pred
    c_norm = normalize(context)
    # exact substring
    if p in c_norm:
        return pred.strip()
    toks = p.split()
    for L in range(min(5, len(toks)), 0, -1):
        for i in range(len(toks)-L+1):
            ng = " ".join(toks[i:i+L])
            if ng in c_norm:
                return " ".join(pred.split()[i:i+L]).strip()
    return pred.strip()

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="./checkpoints/last.pt")
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--max_eval", type=int, default=500)
    ap.add_argument("--device", type=str, default=os.getenv("EVAL_DEVICE", "cpu"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if os.path.exists(args.ckpt):
        payload = torch.load(args.ckpt, map_location="cpu")
        sd = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(args.device).eval()

    ds = load_dataset("squad", split="validation[:{}]".format(args.max_eval))

    preds, refs = [], []
    for ex in ds:
        q, c = ex["question"], ex["context"]
        # build prompt with exemplar
        prompt = EXEMPLAR + make_prompt(q, c)
        inputs = tok(prompt, return_tensors="pt").to(args.device)
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            eos_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = gen.split("\n")[0].strip()  
        gen = gen.split("question:")[0].strip()  

        snapped = snap_to_context(gen, c)
        preds.append({"id": ex["id"], "prediction_text": snapped})
        refs.append({"id": ex["id"], "answers": {"text": ex["answers"]["text"], "answer_start": ex["answers"]["answer_start"]}})

    scores = metric.compute(predictions=preds, references=refs)
    print(json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()
