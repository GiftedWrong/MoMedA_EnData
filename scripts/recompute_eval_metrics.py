#!/usr/bin/env python
"""Пересчёт метрик по сохранённым ответам — без GPU."""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from specialties import SPECIALTIES  # noqa: E402

# дрейф маркеров: русский оригинал + английские варианты заголовка диагноза
DIAG_RE = re.compile(
    r"###\s*(?:Предварительный\s*диагноз|Preliminary\s*[Dd]iagnosis|Answer|Диагноз)")
ANSWER_RE = re.compile(r"(?:Ответ|Answer)\s*:\s*\**\s*([A-F])", re.I)
CYR_RE = re.compile(r"[а-яА-ЯёЁ]")
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
ALLOWED_RU = ("рассуждение", "предварительный", "диагноз", "ответ", "ответ:")


def lang_pure(text: str) -> bool:
    if CJK_RE.search(text):
        return False
    residual = "\n".join(line for line in text.splitlines()
                         if not any(w in line.lower() for w in ALLOWED_RU))
    return not CYR_RE.search(residual)


def main() -> None:
    rows = {}
    for path in sorted(glob.glob(str(PROJECT_ROOT / "runs" / "eval_answers_*.jsonl"))):
        stem = Path(path).stem.replace("eval_answers_", "")
        slug, section = stem.rsplit("_", 1)
        fmt = mcq_ok = mcq_cov = pure = n = 0
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            answer, gold = r.get("answer", ""), r.get("gold", "")
            n += 1
            ok_fmt = bool(DIAG_RE.search(answer))
            fmt += ok_fmt
            pure += lang_pure(answer)
            if gold:
                m = ANSWER_RE.search(answer)
                if m:
                    mcq_cov += 1
                    mcq_ok += m.group(1).upper() == gold.upper()
        rows.setdefault(slug, {})[section] = (n, fmt, pure, mcq_ok, mcq_cov)

    lines = ["# Сводные метрики агентов (пересчёт по сохранённым ответам)", "",
             "MCQ — точность среди ответов, где буква распознана (покрытие в скобках);",
             "формат — любой из дрейф-вариантов заголовка диагноза (### Предварительный диагноз /",
             "Preliminary diagnosis / Answer); EN — без кириллицы/CJK вне маркеров.", "",
             "| Агент | вн. формат | вн. EN | вн. MCQ | USMLE формат | USMLE EN | USMLE MCQ |",
             "|---|---|---|---|---|---|---|"]
    for spec in SPECIALTIES:
        r = rows.get(spec.slug, {})
        def pct(sec, idx):
            v = r.get(sec)
            return f"{v[idx] / v[0] * 100:.0f}%" if v and v[0] else "—"
        def mcq(sec):
            v = r.get(sec)
            if not v or not v[4]:
                return "—"
            return f"{v[3] / v[4] * 100:.0f}% (n={v[4]})"
        lines.append(f"| {spec.name} | {pct('внутренний', 1)} "
                     f"| {pct('внутренний', 2)} | {mcq('внутренний')} "
                     f"| {pct('внешний', 1)} | {pct('внешний', 2)} | {mcq('внешний')} |")
    out = PROJECT_ROOT / "runs" / "FINAL_METRICS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[ok] {out}")


if __name__ == "__main__":
    main()
