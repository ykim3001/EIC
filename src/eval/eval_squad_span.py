# src/eval/eval_squad_span.py
import os, argparse, json, numpy as np, torch, yaml
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator
from ..modeling.gpt2_qa import GPT2QuantForQA 

# custom collate keeps offset_mapping/example_id as lists
# def qa_collate(features):
#     import torch
#     batch = {
#         "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
#     }
#     if "attention_mask" in features[0]:
#         batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
#     # keep these as python lists (contain None)
#     batch["offset_mapping"] = [f["offset_mapping"] for f in features]
#     batch["example_id"]     = [f["example_id"] for f in features]
#     return batch

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def build_val_features(tok, max_len=384, doc_stride=128):
    def prep(ex):
        q = [s.strip() for s in ex["question"]]
        c = ex["context"]
        enc = tok(
            q, c,
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sm = enc.pop("overflow_to_sample_mapping")
        enc["example_id"] = [ex["id"][i] for i in sm]
        for i in range(len(enc["input_ids"])):
            ids = enc.sequence_ids(i)
            enc["offset_mapping"][i] = [
                (o if ids[k]==1 else None) for k,o in enumerate(enc["offset_mapping"][i])
            ]
        return enc
    return prep

def postprocess(examples, features, start_logits, end_logits, n_best=20, max_ans=30):
    per_ex = {}
    for i, ex_id in enumerate(features["example_id"]):
        per_ex.setdefault(ex_id, []).append(i)
    preds, refs = [], []
    for ex in examples:
        ex_id, ctx, ans = ex["id"], ex["context"], ex["answers"]
        best, score = "", -1e9
        for idx in per_ex.get(ex_id, []):
            s, e, off = start_logits[idx], end_logits[idx], features["offset_mapping"][idx]
            si = np.argsort(s)[-n_best:][::-1]
            ei = np.argsort(e)[-n_best:][::-1]
            for i in si:
                for j in ei:
                    if j < i or j-i+1 > max_ans: continue
                    if off[i] is None or off[j] is None: continue
                    sc = s[i] + e[j]
                    if sc > score:
                        a, b = off[i][0], off[j][1]
                        best, score = ctx[a:b], sc
        preds.append({"id": ex_id, "prediction_text": best})
        refs.append({"id": ex_id, "answers": ans})
    import evaluate
    return evaluate.load("squad").compute(predictions=preds, references=refs)

def main():
    import argparse, os, json, re
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)           # e.g., ./checkpoints/last.pt
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--profile", type=str, required=True)        # yaml path for the profile you want to eval
    ap.add_argument("--max_eval", type=int, default=10570)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    # ---- load target profile yaml ----
    profile_cfg = load_yaml(args.profile)
    prof = profile_cfg.get("per_layer_bits", profile_cfg)
    prof = dict(prof)
    prof["name"] = profile_cfg.get("name") or os.path.splitext(os.path.basename(args.profile))[0]

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # extractive QA

    # ---- build model (init with anything; we'll set profile next) ----
    from ..modeling.gpt2_qa import GPT2QuantForQA
    model = GPT2QuantForQA(args.model_name, {"default_weight_bits": 8, "default_activation_bits": 8})

    # ---- discover adapters in checkpoint, pre-create them, then load ----
    sd = torch.load(args.ckpt, map_location="cpu")

    # find adapter names like "...lora_A.<NAME>.weight"
    keys_blob = "\n".join(sd.keys())
    adapter_names = sorted(set(re.findall(r"lora_[AB]\.([^.]+)\.weight", keys_blob)))
    print("adapters in ckpt:", adapter_names, flush=True)

    # create each adapter once so load_state_dict matches keys
    for name in adapter_names:
        model.set_bits_profile({"name": name})

    # load weights
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", missing, flush=True)
    print("unexpected:", unexpected, flush=True)

    # activate the profile we're evaluating (enables its LoRA + bits)
    model.set_bits_profile(prof)

    try:
        print("active adapter:", getattr(model.backbone.model, "active_adapter", None))
        from peft import PeftModel
        if isinstance(model.backbone.model, PeftModel):
            print("all adapters:", list(model.backbone.model.peft_config.keys()))
    except Exception as e:
        print("adapter introspection skipped:", e)

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ---- dataset & features ----
    ds = load_dataset("squad")
    n = min(args.max_eval, len(ds["validation"]))
    examples = ds["validation"].select(range(n))
    feats = examples.map(
        build_val_features(tok),  # this must be defined above (as in your file)
        batched=True,
        remove_columns=examples.column_names,
        desc="Tokenizing eval"
    )

    # ---- custom collator (keeps offset_mapping/example_id as lists) ----
    def qa_collate(features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        }
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        batch["offset_mapping"] = [f["offset_mapping"] for f in features]
        batch["example_id"]     = [f["example_id"] for f in features]
        return batch

    dl = DataLoader(feats, batch_size=16, shuffle=False, collate_fn=qa_collate)

    # ---- forward + collect logits ----
    all_s, all_e, all_offsets, all_ids = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            out = model(**inputs)  # QuestionAnsweringModelOutput
            all_s.append(out.start_logits.detach().cpu().numpy())
            all_e.append(out.end_logits.detach().cpu().numpy())
            all_offsets.extend(batch["offset_mapping"])
            all_ids.extend(batch["example_id"])

    start_logits = np.concatenate(all_s, axis=0)
    end_logits   = np.concatenate(all_e, axis=0)

    # minimal "features" view for postprocess
    features_view = {"example_id": all_ids, "offset_mapping": all_offsets}

    # ---- postprocess to EM/F1 ----
    metrics = postprocess(  # this must be defined above (as in your file)
        examples=examples,
        features=features_view,
        start_logits=start_logits,
        end_logits=end_logits,
        n_best=20,
        max_ans=30,
    )
    print(metrics)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    main()
