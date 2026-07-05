# One trainer for every model with config held identical across all models so results are comparable.
#
# Effective batch size is constant at 32.
# 5 epochs, LR 2e-5, 6% warmup, weight decay 0.01.
# Regression head (num_labels=1); labels cast to float32.
# Evaluation once per epoch; the best checkpoint by dev Spearman is kept.
# A timing probe estimates the full-run cost from the first N steps.

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Value
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint

from config import PROCESSED_DIR, MODELS_DIR
from evaluate import compute_metrics

EFFECTIVE_BATCH = 32
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01


@dataclass
class RunConfig:
    name: str
    model_name: str
    max_length: int = 128
    epochs: int = 5
    seed: int = 42
    hf_repo: Optional[str] = None
    per_device_override: Optional[int] = None


def _resolve_batch(n_gpus: int, per_device_override: Optional[int]):
    per_device = per_device_override or max(1, EFFECTIVE_BATCH // n_gpus)
    denom = per_device * n_gpus
    assert EFFECTIVE_BATCH % denom == 0, (
        f"effective batch {EFFECTIVE_BATCH} not divisible by "
        f"per_device({per_device}) x n_gpus({n_gpus})")
    return per_device, EFFECTIVE_BATCH // denom


def _make_dataset(df, tokenizer, max_length):
    ds = Dataset.from_pandas(
        df[["joke", "score"]].rename(columns={"score": "labels"}),
        preserve_index=False)
    ds = ds.map(lambda b: tokenizer(b["joke"], truncation=True,
                                    max_length=max_length), batched=True)
    return ds.cast_column("labels", Value("float32"))


def _hf_metrics(eval_pred):
    preds, labels = eval_pred
    return compute_metrics(labels, preds)


def train(cfg: RunConfig, probe_steps: Optional[int] = None, push: bool = False):
    set_seed(cfg.seed)
    probing = probe_steps is not None
    n_gpus = max(1, torch.cuda.device_count())
    per_device, grad_accum = _resolve_batch(n_gpus, cfg.per_device_override)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=1, problem_type="regression")

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    dev_df = pd.read_csv(PROCESSED_DIR / "dev.csv")
    train_ds = _make_dataset(train_df, tokenizer, cfg.max_length)
    dev_ds = _make_dataset(dev_df, tokenizer, cfg.max_length)

    steps_per_epoch = int(np.ceil(len(train_df) / EFFECTIVE_BATCH))
    total_steps = steps_per_epoch * cfg.epochs
    print(f"[{cfg.name}] {n_gpus} GPU(s) | per_device={per_device} "
          f"grad_accum={grad_accum} -> effective={EFFECTIVE_BATCH} | "
          f"{steps_per_epoch:,} steps/epoch x {cfg.epochs} = {total_steps:,} updates")

    out_dir = str(MODELS_DIR / cfg.name)
    args = TrainingArguments(
        output_dir=out_dir,
        hub_model_id=cfg.hf_repo,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        seed=cfg.seed,
        fp16=True,
        eval_strategy="no" if probing else "epoch",
        save_strategy="no" if probing else "epoch",
        save_total_limit=2,
        logging_steps=200,
        load_best_model_at_end=not probing,
        metric_for_best_model="spearman",
        greater_is_better=True,
        report_to="none",
        max_steps=probe_steps if probing else -1,
    )

    collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=dev_ds, compute_metrics=_hf_metrics,
                      data_collator=collator)

    if probing:
        t0 = time.time()
        trainer.train()
        rate = probe_steps / (time.time() - t0)
        print(f"[probe] {rate:.2f} updates/s -> full-run ETA "
              f"~{total_steps / rate / 3600:.1f}h for {total_steps:,} updates")
        return None

    last = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    if last:
        print("resuming from", last)
    trainer.train(resume_from_checkpoint=last)
    print("best dev metrics:", trainer.evaluate())

    if push and cfg.hf_repo:
        trainer.push_to_hub(f"{cfg.name}: {cfg.epochs} epochs, seed {cfg.seed}")
        tokenizer.push_to_hub(cfg.hf_repo)
        print("pushed ->", f"https://huggingface.co/{cfg.hf_repo}")
    return trainer
