import argparse, os, numpy as np, torch, torch.nn as nn
import transformers, evaluate
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoConfig, AutoModel,
                          TrainingArguments, Trainer, default_data_collator)

class GPT2SpanQA(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = getattr(self.config, "n_embd", self.config.hidden_size)
        self.qa_outputs = nn.Linear(hidden, 2)

    def forward(self, input_ids=None, attention_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        logits = self.qa_outputs(seq)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = (loss_fct(start_logits, start_positions) +
                    loss_fct(end_logits, end_positions)) / 2
        return transformers.modeling_outputs.QuestionAnsweringModelOutput(
            loss=loss, start_logits=start_logits, end_logits=end_logits
        )

def build_features(tokenizer, max_len, doc_stride):
    def prepare_train(examples):
        q = [s.strip() for s in examples["question"]]
        c = examples["context"]
        enc = tokenizer(q, c, truncation="only_second", max_length=max_len,
                        stride=doc_stride, return_overflowing_tokens=True,
                        return_offsets_mapping=True, padding="max_length")
        sample_map = enc.pop("overflow_to_sample_mapping")
        offsets = enc["offset_mapping"]
        seq_ids = [enc.sequence_ids(i) for i in range(len(enc["input_ids"]))]

        sp, ep = [], []
        for i, off in enumerate(offsets):
            cls_idx = 0
            ex_idx = sample_map[i]
            ans = examples["answers"][ex_idx]
            if len(ans["answer_start"]) == 0:
                sp.append(cls_idx); ep.append(cls_idx); continue
            s_char = ans["answer_start"][0]; e_char = s_char + len(ans["text"][0])
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
        enc["start_positions"] = sp; enc["end_positions"] = ep
        return enc

    def prepare_val(examples):
        q = [s.strip() for s in examples["question"]]; c = examples["context"]
        enc = tokenizer(q, c, truncation="only_second", max_length=max_len,
                        stride=doc_stride, return_overflowing_tokens=True,
                        return_offsets_mapping=True, padding="max_length")
        sample_map = enc.pop("overflow_to_sample_mapping")
        enc["example_id"] = [examples["id"][i] for i in sample_map]
        for i in range(len(enc["input_ids"])):
            ids = enc.sequence_ids(i)
            enc["offset_mapping"][i] = [
                (o if ids[k] == 1 else None) for k, o in enumerate(enc["offset_mapping"][i])
            ]
        return enc
    return prepare_train, prepare_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--save_dir", default="runs/baseline_fp32")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("squad")
    prep_train, prep_val = build_features(tokenizer, args.max_len, args.doc_stride)
    tokenized_train = ds["train"].map(prep_train, batched=True, remove_columns=ds["train"].column_names)
    tokenized_val   = ds["validation"].map(prep_val, batched=True, remove_columns=ds["validation"].column_names)

    model = GPT2SpanQA(args.model_name)
    model.backbone.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        learning_rate=args.lr,
        weight_decay=args.wd,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        logging_steps=100,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        report_to=[],
        seed=args.seed,
        fp16=args.fp16,
        remove_unused_columns=False,
        save_total_limit=1,
        save_strategy="epoch",
        evaluation_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

if __name__ == "__main__":
    main()
