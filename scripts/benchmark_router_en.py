#!/usr/bin/env python
"""Тест роутера: точность по классам и источникам, путаницы."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from specialties import SPECIALTIES  # noqa: E402
from utils import load_config, read_jsonl, write_jsonl  # noqa: E402

CLASSES = [s.name for s in SPECIALTIES]
ROUTE_RE = re.compile(r"Рекомендуемый\s+специалист\s*:\s*([^\n.]+)")
MAX_NEW = 24


def extract_class(text: str) -> str | None:
    m = ROUTE_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1).strip().lower().replace("ё", "е")
    for c in CLASSES:
        if c.lower().replace("ё", "е") in raw or raw in c.lower():
            return c
    return None


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default=None, help="путь к test.jsonl (по умолчанию data/router)")
    ap.add_argument("--usmle", type=int, default=0,
                    help="дополнительно псевдо-разметить N виньеток MedQA-USMLE")
    ap.add_argument("--limit", type=int, default=0, help="ограничить test (для быстрого прогона)")
    args = ap.parse_args()

    cfg = load_config()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False)
    FastLanguageModel.for_inference(model)

    def route(messages: list[dict]) -> tuple[str | None, str]:
        # ответы ассистента из messages исключаем: строим промпт только по
        # system+user, иначе золотой ответ попадает в контекст и модель
        # продолжает текст после него
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_class(text), text

    # --- test.jsonl ---
    test_path = Path(args.test) if args.test else (PROJECT_ROOT / cfg["data_router"] / "test.jsonl")
    rows = read_jsonl(test_path)
    if args.limit:
        rows = rows[: args.limit]
    print(f"test: {len(rows)} примеров из {test_path}")

    strict = soft = 0
    by_class = collections.defaultdict(lambda: [0, 0])       # класс -> [верно, всего]
    by_src = collections.defaultdict(lambda: [0, 0])         # источник -> [верно, всего]
    by_class_src = collections.defaultdict(lambda: [0, 0])   # (класс, источник) -> [верно, всего]
    confusions = collections.Counter()
    raw_errors: list[tuple[str, str | None, str]] = []       # (золото, предсказание, сырой ответ)
    for r in tqdm(rows, desc="router test"):
        pred, raw = route(r["messages"])
        gold = r["label"]
        src = r.get("source", "?")
        ok_soft = pred == gold
        soft += ok_soft
        by_class[gold][1] += 1
        by_class_src[(gold, src)][1] += 1
        by_src[src][1] += 1
        if ok_soft:
            by_class[gold][0] += 1
            by_class_src[(gold, src)][0] += 1
            by_src[src][0] += 1
        else:
            confusions[(gold, pred)] += 1
            if len(raw_errors) < 10:
                raw_errors.append((gold, pred, raw))
        by_class[gold][1] += 1
        by_class_src[(gold, src)][1] += 1
        by_src[src][1] += 1
        if ok_soft:
            by_class[gold][0] += 1
            by_class_src[(gold, src)][0] += 1
            by_src[src][0] += 1
        else:
            confusions[(gold, pred)] += 1

    n = len(rows)
    lines = ["# Бенчмарк EN-роутера", "",
             f"Модель: `{args.model}`; test: {n} примеров",
             "",
             f"**Точность: {soft / max(n, 1) * 100:.1f}%** ({soft}/{n})", "",
             "## Точность по классам × источникам", "",
             "| Класс | всего | точность | " + " | ".join(sorted(set(r.get('source', '?') for r in rows))) + " |",
             "|---" * (4 + len(set(r.get('source', '?') for r in rows))) + "|"]
    for c in CLASSES:
        tot, ok = by_class[c][1], by_class[c][0]
        if not tot:
            continue
        srcs = sorted(set(r.get("source", "?") for r in rows))
        cells = []
        for s in srcs:
            a, b = by_class_src[(c, s)]
            cells.append(f"{a}/{b}" if b else "—")
        lines.append(f"| {c} | {tot} | {ok / tot * 100:.0f}% | " + " | ".join(cells) + " |")
    lines += ["", "## Точность по источникам (диагностика шума разметки)", "",
              "| Источник | Точность |", "|---|---|"]
    for s, (a, b) in sorted(by_src.items(), key=lambda x: -x[1][1]):
        lines.append(f"| {s} | {a}/{b} = {a / max(b, 1) * 100:.0f}% |")
    lines += ["", "## Топ-20 ошибок (золото → предсказание)", ""]
    for (g, p), c in confusions.most_common(20):
        lines.append(f"- {g} → {p}: {c}")
    if raw_errors:
        lines += ["", "## Сырые ответы на ошибках (первые 10, для диагностики формата)", ""]
        for g, p, raw in raw_errors:
            lines.append(f"- золото **{g}**, извлечено `{p}`; сырой ответ: `{raw[:140]}`")

    report = PROJECT_ROOT / cfg["data_processed"] / "ROUTER_BENCHMARK.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n[ok] отчёт: {report}")

    # --- USMLE псевдо-разметка ---
    if args.usmle:
        medqa = []
        for split in ["train", "test"]:
            p = PROJECT_ROOT / cfg["data_raw"] / "en" / "medqa" / f"phrases_no_exclude_{split}.jsonl"
            if p.exists():
                medqa.extend(read_jsonl(p))
        step = max(1, len(medqa) // args.usmle)
        sample = medqa[::step][: args.usmle]
        sys_prompt = ("You are a medical triage router. Read the patient case in English and "
                      "select the single most appropriate specialist. Answer with exactly one "
                      "line in the format «Рекомендуемый специалист: <специалист>.», choosing "
                      f"one of: {', '.join(CLASSES)}.")
        out_rows = []
        for r in tqdm(sample, desc="usmle pseudo-label"):
            opts = r.get("options") or {}
            opt_block = "\n".join(f"{k}. {v}" for k, v in opts.items())
            messages = [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": (r["question"] + "\n" + opt_block).strip()}]
            pred, _ = route(messages)
            if pred:
                out_rows.append({"specialty": pred, "question": r["question"], "options": opts,
                                 "answer_idx": r.get("answer_idx"), "answer": r.get("answer")})
        write_jsonl(PROJECT_ROOT / cfg["data_processed"] / "usmle_pseudo_labeled.jsonl", out_rows)
        cnt = collections.Counter(r["specialty"] for r in out_rows)
        print(f"[ok] USMLE псевдо-разметка: {len(out_rows)} виньеток; распределение: {dict(cnt.most_common())}")


if __name__ == "__main__":
    main()
