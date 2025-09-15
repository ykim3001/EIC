import os, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
import evaluate

from src.modeling.gpt2_qa import GPT2QuantForQA

def qa_collate(features):
    keys = ("input_ids", "attention_mask", "start_positions", "end_positions")
    batch = {}
    for k in keys:
        if k in features[0] and features[0][k] is not None:
            batch[k] = torch.tensor([f[k] for f in features])
    return batch

def add_pad_token_if_missing(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def prepare_features(tokenizer, examples, max_len=384, doc_stride=128):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    enc = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=None,
    )

    sample_map = enc.pop("overflow_to_sample_mapping")
    offset_mapping = enc["offset_mapping"]
    seq_ids_list = [enc.sequence_ids(i) for i in range(len(enc["input_ids"]))]
    example_ids = [examples["id"][sample_map[i]] for i in range(len(enc["input_ids"]))]

    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_mapping):
        cls_index = 0

        answers = examples["answers"][sample_map[i]]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        sequence_ids = seq_ids_list[i]

        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_idx = context_start
            end_idx = context_end
            while start_idx <= context_end and offsets[start_idx][0] <= start_char:
                start_idx += 1
            while end_idx >= context_start and offsets[end_idx][1] >= end_char:
                end_idx -= 1
            start_positions.append(start_idx - 1)
            end_positions.append(end_idx + 1)

    enc["start_positions"] = start_positions
    enc["end_positions"] = end_positions
    enc["example_id"] = example_ids
    enc["sequence_ids"] = seq_ids_list
    return enc

@torch.no_grad()
def postprocess(pred_start, pred_end, features, examples, tokenizer, max_answer_len=30):
    metric = evaluate.load("squad")

    features_per_example = {}
    for i, ex_id in enumerate(features["example_id"]):
        features_per_example.setdefault(ex_id, []).append(i)

    predictions, references = [], []
    for ex_idx, ex_id in enumerate(examples["id"]):
        if ex_id not in features_per_example:
            continue
        best_text = ""
        best_score = -1e9
        context = examples["context"][ex_idx]
        for feat_idx in features_per_example[ex_id]:
            starts = pred_start[feat_idx]
            ends = pred_end[feat_idx]
            offset_mapping = features["offset_mapping"][feat_idx]
            sequence_ids = features["sequence_ids"][feat_idx]

            for i in np.argsort(starts)[-20:]:
                for j in np.argsort(ends)[-20:]:
                    if i > j or (j - i + 1) > max_answer_len:
                        continue
                    if sequence_ids[i] != 1 or sequence_ids[j] != 1:
                        continue
                    start_char = offset_mapping[i][0]
                    end_char = offset_mapping[j][1]
                    text = context[start_char:end_char]
                    score = starts[i] + ends[j]
                    if score > best_score:
                        best_score = score
                        best_text = text

        predictions.append({"id": ex_id, "prediction_text": best_text})
        references.append({"id": ex_id, "answers": examples["answers"][ex_idx]})

    return metric.compute(predictions=predictions, references=references)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--train_steps", type=int, default=1500)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--device", type=str, default=os.getenv("TRAIN_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--eval_samples", type=int, default=1000)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    uniform8 = {"default_weight_bits": 8, "default_activation_bits": 8}
    model = GPT2QuantForQA(args.model_name, uniform8).to(args.device)
    model.backbone.model.config.pad_token_id = tokenizer.pad_token_id

    add_pad_token_if_missing(tokenizer)

    for p in model.backbone.parameters():
        p.requires_grad_(False)         
    for p in model.qa_outputs.parameters():
        p.requires_grad_(True)

    optim = torch.optim.AdamW(
        [
            {"params": [p for n,p in model.named_parameters() if p.requires_grad and "qa_outputs" not in n], "lr": 5e-5},
            {"params": model.qa_outputs.parameters(), "lr": 5e-4},
        ],
        weight_decay=0.01,
    )

    warmup_steps = max(100, int(0.1 * args.train_steps))
    scheduler = get_linear_schedule_with_warmup(optim, warmup_steps, args.train_steps)
    ds_train = load_dataset("squad", split="train")
    ds_eval = load_dataset("squad", split=f"validation[:{args.eval_samples}]")

    def _prep(examples):
        return prepare_features(tokenizer, examples, max_len=args.max_len, doc_stride=args.doc_stride)

    train_feats = ds_train.map(_prep, batched=True, remove_columns=ds_train.column_names)
    eval_feats = ds_eval.map(_prep, batched=True, remove_columns=ds_eval.column_names)

    train_dl = DataLoader(train_feats, batch_size=args.batch_size, shuffle=True, collate_fn=qa_collate)
    eval_dl  = DataLoader(eval_feats,  batch_size=args.batch_size, shuffle=False, collate_fn=qa_collate)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_updates = args.train_steps
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=total_updates)

    model.train()
    step, it = 0, iter(train_dl)
    while step < total_updates:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)
        for k in ("input_ids", "attention_mask", "start_positions", "end_positions"):
            batch[k] = batch[k].to(args.device)
        out = model(**batch)
        loss = out.loss / args.grad_accum
        loss.backward()
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); scheduler.step(); optim.zero_grad()
        if (step + 1) % 100 == 0:
            print(f"[train] step {step+1}/{total_updates} | loss={loss.item()*args.grad_accum:.4f}")
        step += 1

        if step == 300:  
            for p in model.backbone.parameters():
                p.requires_grad_(True)

    model.eval()
    all_start, all_end = [], []
    for batch in eval_dl:
        for k in ("input_ids", "attention_mask"):
            batch[k] = batch[k].to(args.device)
        with torch.no_grad():
            out = model(**batch)
        all_start.extend(out.start_logits.detach().cpu().numpy())
        all_end.extend(out.end_logits.detach().cpu().numpy())

    scores = postprocess(
        np.array(all_start, dtype=np.float32),
        np.array(all_end, dtype=np.float32),
        features=eval_feats,   
        examples=ds_eval,
        tokenizer=tokenizer,
    )
    print(json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()
