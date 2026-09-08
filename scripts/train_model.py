#!/usr/bin/env python
"""Единый тренер MoMedA_ChData: EN-роутер и агенты-специалисты.

Полный файнтюн Qwen2.5-3B-Instruct (unsloth с откатом на transformers),
ChatML-токенизация, лосс только по ответу ассистента (labels=-100 вне его).
Рецепт и гиперпараметры — отработанный train.py соседнего проекта
AI_Dev_Qwen2.5-3B_Diagnosis-training (модели med-cot-3b / med-routing-3b).

Цели (--data | --specialty, ровно одна):
  --data data/router          EN-роутер, 14 классов  -> models/med-router-en-3b
  --specialty Терапевт        агент-специалист       -> models/med-spec-<slug>-3b

Запуск:
  python scripts/train_model.py --data data/router
  python scripts/train_model.py --specialty Терапевт [--epochs 1]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from specialties import BY_NAME  # noqa: E402
from utils import load_config, project_path  # noqa: E402

CFG = load_config()
BASE = str(project_path(CFG["base_model"]).resolve())
# общий кэш скомпилированных ядер unsloth (рядом с базовой моделью)
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION",
                      str(Path(BASE).parent / "unsloth_compiled_cache"))

HP = {
    "router": {"max_len": 1024, "epochs": 2.0, "batch": 2, "accum": 8, "lr": 2e-5},
    # batch 1×accum 16 (рецепт cot соседнего проекта): при max_len 2048 батч из
    # двух длинных примеров даёт пик логитов ~1.2 ГБ и OOM на 24 ГБ карте —
    # Отоларинголог упал на 230/268 шаге 2026-09-06
    "specialty": {"max_len": 2048, "epochs": 1.0, "batch": 1, "accum": 16, "lr": 2e-5},
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chat_ids(tokenizer, messages, add_generation_prompt=False):
    """ChatML-id через рендер в строку: apply_chat_template(tokenize=True) в
    transformers 5.x возвращает BatchEncoding, а unsloth-токенизатор — список;
    строковый рендер + add_special_tokens=False одинаков во всех версиях."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def tokenize(examples, tokenizer, max_len):
    input_ids, labels = [], []
    for messages in examples["messages"]:
        full = chat_ids(tokenizer, messages)
        prompt = chat_ids(tokenizer, messages[:-1], add_generation_prompt=True)
        ids = full[:max_len]
        lab = [-100] * min(len(prompt), len(ids)) + ids[len(prompt):]
        if all(l == -100 for l in lab):
            continue
        input_ids.append(ids)
        labels.append(lab)
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in feats:
            ids, lab = f["input_ids"], f["labels"]
            pad = n - len(ids)
            batch["input_ids"].append(ids + [self.pad_id] * pad)
            batch["attention_mask"].append([1] * len(ids) + [0] * pad)
            batch["labels"].append(lab + [-100] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


class ETACallback(TrainerCallback):
    """Прогноз полного времени обучения по фактической скорости первых шагов."""

    def __init__(self, total_steps: int):
        self.total = total_steps
        self.t0 = None
        self.steps_done = 0

    def on_step_end(self, args, state, control, **kw):
        self.steps_done = state.global_step
        if self.t0 is None:
            self.t0 = time.time()
            return
        if state.global_step in (10, 50, 100, 250):
            dt = (time.time() - self.t0) / (state.global_step - 1)
            eta = dt * (self.total - state.global_step)
            print(f">>> ETA: шаг {state.global_step}/{self.total}, "
                  f"{dt:.1f} с/шаг, осталось ~{eta / 60:.0f} мин", flush=True)


def unsloth_selfcheck() -> bool:
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_engine.py")],
                       capture_output=True, text=True, timeout=900)
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(f">>> самотест unsloth: {tail[-1] if tail else 'нет вывода'}")
    return r.returncode == 0


def load_model(max_len: int, engine: str):
    use_unsloth = engine == "unsloth" or (engine == "auto" and unsloth_selfcheck())
    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE, max_seq_length=max_len, dtype=torch.bfloat16,
            load_in_4bit=False, full_finetuning=True)
        print(">>> движок: unsloth (полный файнтюн)")
        return model, tokenizer

    import transformers as _tf
    major = int(_tf.__version__.split(".")[0])
    kw = {"dtype" if major >= 5 else "torch_dtype": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    model.enable_input_require_grads()
    print(f">>> движок: transformers {_tf.__version__} (полный файнтюн)")
    return model, tokenizer


def resolve_target(args) -> tuple[str, Path, Path, str]:
    """-> (режим гиперпараметров, train.jsonl, val.jsonl, run_name)."""
    if bool(args.data) == bool(args.specialty):
        raise SystemExit("укажи ровно одну цель: --data data/router или --specialty <Имя>")
    if args.data:
        d = project_path(args.data)
        if "chief" in str(d).lower():
            # мастер-агент: длинные примеры — гиперпараметры как у специалистов
            return "specialty", d / "train.jsonl", d / "val.jsonl", "med-chief-3b"
        return "router", d / "train.jsonl", d / "val.jsonl", "med-router-en-3b"
    spec = BY_NAME.get(args.specialty)
    if spec is None:
        raise SystemExit(f"неизвестная специальность: {args.specialty}; "
                         f"доступны: {', '.join(BY_NAME)}")
    d = project_path(CFG["data_specialties"]) / spec.name
    return "specialty", d / "train.jsonl", d / "val.jsonl", f"med-spec-{spec.slug}-3b"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="каталог датасета (data/router)")
    ap.add_argument("--specialty", default=None, help="имя специальности (Терапевт)")
    ap.add_argument("--engine", choices=["auto", "unsloth", "transformers"], default="auto")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    mode, train_path, val_path, run_name = resolve_target(args)
    hp = dict(HP[mode])
    for k, v in [("epochs", args.epochs), ("batch", args.batch),
                 ("accum", args.accum), ("max_len", args.max_len)]:
        if v is not None:
            hp[k] = v
    if args.run_name:
        run_name = args.run_name

    print(f">>> цель: {run_name}; train={train_path.name}@{train_path.parent}, "
          f"hp={hp}, base={BASE}")

    model, tokenizer = load_model(hp["max_len"], args.engine)
    model.config.use_cache = False

    raw = Dataset.from_list(load_jsonl(train_path))
    raw_val = Dataset.from_list(load_jsonl(val_path))
    kw = dict(tokenizer=tokenizer, max_len=hp["max_len"])
    ds = raw.map(tokenize, batched=True, fn_kwargs=kw, remove_columns=raw.column_names)
    ds_val = raw_val.map(tokenize, batched=True, fn_kwargs=kw, remove_columns=raw_val.column_names)
    print(f"train: {len(ds)} (из {len(raw)}), val: {len(ds_val)} (из {len(raw_val)})")

    steps_per_epoch = max(1, len(ds) // (hp["batch"] * hp["accum"]))
    total_steps = int(steps_per_epoch * hp["epochs"])

    out_dir = project_path(CFG["runs_dir"]) / run_name
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["batch"],
        gradient_accumulation_steps=hp["accum"],
        per_device_eval_batch_size=hp["batch"],
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=hp["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        max_grad_norm=1.0,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        eval_dataset=ds_val,
        data_collator=Collator(tokenizer.pad_token_id),
        callbacks=[ETACallback(total_steps)],
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    merged_dir = project_path(CFG["models_dir"]) / run_name
    model.config.use_cache = True
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"ГОТОВО: полная модель сохранена в {merged_dir}")


if __name__ == "__main__":
    main()
