#!/usr/bin/env python
"""Оценка агента: формат, чистота EN, MCQ-точность; опционально USMLE."""
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

from specialties import BY_NAME  # noqa: E402
from utils import load_config, read_jsonl, project_path  # noqa: E402

DIAG_RE = re.compile(r"###\s*(Предварительный\s*диагноз|Preliminary\s*[Dd]iagnosis)")
# модель дрейфует между маркерами «Ответ:» (обучение) и «Answer:» (инструкция EN) — берём оба
ANSWER_RE = re.compile(r"(?:Ответ|Answer)\s*:\s*([A-F])", re.I)
CYR_RE = re.compile(r"[а-яА-ЯёЁ]")
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
ALLOWED_RU = ("рассуждение", "предварительный", "диагноз", "ответ")  # маркеры блоков


def lang_pure(text: str) -> bool:
    """Кириллица допустима только в маркерах блоков; CJK — недопустимы."""
    if CJK_RE.search(text):
        return False
    residual = "\n".join(line for line in text.splitlines()
                         if not any(w in line.lower() for w in ALLOWED_RU))
    return not CYR_RE.search(residual)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specialty", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--usmle", default=None, help="usmle_pseudo_labeled.jsonl для внешнего теста")
    ap.add_argument("--usmle-limit", type=int, default=50)
    args = ap.parse_args()

    spec = BY_NAME.get(args.specialty)
    if spec is None:
        raise SystemExit(f"неизвестная специальность: {args.specialty}")

    cfg = load_config()
    from unsloth import FastLanguageModel

    model_dir = project_path(cfg["models_dir"]) / f"med-spec-{spec.slug}-3b"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir), max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False)
    FastLanguageModel.for_inference(model)

    def generate(messages):
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        out = model.generate(**ids, max_new_tokens=768, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def evaluate(rows, title, report):
        stats = collections.Counter()
        samples = []
        answers_path = project_path(cfg["runs_dir"]) / f"eval_answers_{spec.slug}_{title.split()[0].lower()}.jsonl"
        answers_file = answers_path.open("w", encoding="utf-8")
        for r in tqdm(rows, desc=title):
            answer = generate(r["messages"])
            answers_file.write(json.dumps({"answer": answer,
                                           "gold": r.get("answer_letters", "")},
                                          ensure_ascii=False) + "\n")
            fmt = bool(DIAG_RE.search(answer))
            pure = lang_pure(answer)
            stats["n"] += 1
            stats["format_ok"] += fmt
            stats["lang_pure"] += pure
            gold = r.get("answer_letters")
            if gold and fmt:
                m = ANSWER_RE.search(answer)
                stats["mcq_total"] += 1
                if m and m.group(1).upper() == gold.upper():
                    stats["mcq_ok"] += 1
            if len(samples) < 8:
                samples.append((r["messages"][1]["content"][:300], answer[:600]))
        answers_file.close()
        n = max(stats["n"], 1)
        report += [f"## {title} (n={stats['n']})", "",
                   f"- формат `### Предварительный диагноз`: {stats['format_ok']}/{stats['n']} "
                   f"({stats['format_ok'] / n * 100:.0f}%)",
                   f"- чистый EN (без CJK/кир. вне маркеров): {stats['lang_pure']}/{stats['n']} "
                   f"({stats['lang_pure'] / n * 100:.0f}%)"]
        if stats["mcq_total"]:
            report.append(f"- MCQ-точность: {stats['mcq_ok']}/{stats['mcq_total']} "
                          f"({stats['mcq_ok'] / stats['mcq_total'] * 100:.0f}%)")
        report.append("")
        for i, (q, a) in enumerate(samples, 1):
            report += [f"### Сэмпл {i}", f"**Вопрос:** {q}", "", f"**Ответ:** {a}", "", "---", ""]
        return report

    report = [f"# Оценка агента: {spec.name} (`{model_dir.name}`)", ""]
    test = read_jsonl(project_path(cfg["data_specialties"]) / spec.name / "test.jsonl")
    if args.limit:
        test = test[: args.limit]
    report = evaluate(test, "Внутренний тест (data/specialties)", report)

    if args.usmle:
        usmle_rows = []
        for r in read_jsonl(Path(args.usmle)):
            if r.get("specialty") != spec.name:
                continue
            opts = r.get("options") or {}
            opt_block = "\n".join(f"{k}. {v}" for k, v in opts.items())
            sys_prompt = (f"You are a doctor — {spec.names_en[0]}. Study the patient's history "
                          "and examination results and formulate a preliminary diagnosis. "
                          "Answer in English only.")
            usmle_rows.append({"messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": (r["question"] + "\n" + opt_block).strip()},
                {"role": "assistant", "content": ""},
            ], "answer_letters": r.get("answer_idx", "")})
            if len(usmle_rows) >= args.usmle_limit:
                break
        if usmle_rows:
            report = evaluate(usmle_rows, f"Внешний тест USMLE-псевдо ({args.usmle})", report)
        else:
            report += ["## Внешний тест USMLE", "", "виньеток этой специальности не найдено", ""]

    out = project_path(cfg["runs_dir"]) / f"eval_med-spec-{spec.slug}-3b.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"[ok] отчёт: {out}")


if __name__ == "__main__":
    main()
