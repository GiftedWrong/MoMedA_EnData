#!/usr/bin/env python
"""Готовим наборы 14 специалистов: чистка, разбиение, экспорт в messages-формат."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from specialties import (SPECIALTIES, map_medmcqa, pseudo_label, PSEUDO_KEYWORDS)  # noqa: E402
from utils import md5, write_jsonl, load_config  # noqa: E402

SEED = 42
MIN_Q, MIN_A, MAX_PAIR = 20, 50, 16000
VAL_SHARE, TEST_SHARE = 0.05, 0.05
LETTERS = "ABCD"

SYSTEM_PROMPT = (
    "You are a doctor — {spec_en}. Study the patient's history and examination "
    "results and formulate a preliminary diagnosis. Answer in English only.\n"
    "Ты — врач-{spec_ru}. Изучи анамнез и результаты обследований и сформулируй "
    "предварительный диагноз. Отвечай только на английском языке."
)
DIAG_HEADER = "### Предварительный диагноз"
REASON_HEADER = "### Рассуждение"

# унификация формата (--uniform-format): у консультаций выделяем строку
# диагноза из собственных формулировок врача; без совпадения — как есть
_DIAG_LINE_RE = re.compile(
    r"[^.!?]*(?:\bsuggests?\b|\bindicates?\b|indicative of|due to|caused by|"
    r"consistent with|points? to|sign of|may (?:have|be|due)|might (?:have|be)|"
    r"could (?:be|have|indicate)|is likely|are likely|probably|most likely|"
    r"diagnos\w+ (?:as|of|is|with)|suffering from|chances? of|probability of|"
    r"strong suspicion)[^.!?]*[.!?]", re.I)


def wrap_consult(answer: str) -> str:
    """Консультация → единый шаблон: рассуждение = полный ответ, диагноз =
    предложение с диагностической формулировкой врача (если найдено)."""
    m = _DIAG_LINE_RE.search(answer)
    if not m:
        return answer
    line = m.group(0).strip()
    if len(line) < 25:
        return answer
    return f"{REASON_HEADER}\n{answer}\n\n{DIAG_HEADER}\n{line}"


def clean(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return "\n".join(line.strip() for line in text.strip().splitlines()).strip()


def load_medmcqa(raw: Path) -> list[dict]:
    rows, unmapped = [], collections.Counter()
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(raw / "medmcqa" / "data" / f"{split}-00000-of-00001.parquet")
        for r in df.itertuples(index=False):
            spec = map_medmcqa(r.subject_name, r.topic_name)
            if not spec:
                unmapped[r.subject_name] += 1
                continue
            q = clean(r.question)
            opts = {LETTERS[i]: clean(x) for i, x in enumerate([r.opa, r.opb, r.opc, r.opd])}
            ans_letter = LETTERS[int(r.cop)] if 0 <= int(r.cop) < 4 else ""
            ans_text = opts.get(ans_letter, "")
            if len(q) < MIN_Q or not ans_text:
                continue
            opt_block = "\n".join(f"{k}. {v}" for k, v in opts.items() if v)
            expl = clean(getattr(r, "exp", "") or "")
            rows.append({
                "source": "medmcqa", "kind": "exam", "specialty": spec,
                "question_en": q + "\n" + opt_block,
                "answer_text": ans_text, "answer_letters": ans_letter,
                "explanation_en": expl if len(expl) >= MIN_A else "",
            })
    print(f"      MedMCQA: {len(rows)} виньеток; несмаплено по предметам: {dict(unmapped.most_common(6))}")
    return rows


def load_medquad(raw: Path) -> list[dict]:
    """MedQuAD → консультации. Онкология: 1_CancerGov_QA (729 QA — реальные данные
    репозитория; крупные папки ADAM/Drugs распространяются без ответов из-за
    копирайта, см. readme.txt). Неврология: 6_NINDS_QA (институт NINDS, ~1.1k QA).
    NIDDK/NHLBI — резерв, их специальности уже покрыты MedMCQA."""
    folder_map = {"1_CancerGov_QA": "Онколог", "6_NINDS_QA": "Невролог"}
    rows = []
    for folder, specialty in folder_map.items():
        n = 0
        for path in glob.glob(str(raw / "medquad" / folder / "**" / "*.xml"), recursive=True):
            try:
                tree = ET.parse(path)
            except ET.ParseError:
                continue
            for qa in tree.iter("QAPair"):
                q = clean("".join(qa.findtext("Question") or "").strip())
                a = clean("".join(qa.findtext("Answer") or "").strip())
                if len(q) >= MIN_Q and len(a) >= MIN_A and len(q) + len(a) <= MAX_PAIR:
                    rows.append({"source": "medquad", "kind": "consult", "specialty": specialty,
                                 "question_en": q, "answer_text": a})
                    n += 1
        print(f"      MedQuAD {folder}: {n} QA -> {specialty}")
    return rows


def load_hcm_pseudo(raw: Path, per_specialty_limit: int = 20000) -> list[dict]:
    """HealthCareMagic + словарная псевдо-разметка (только специальности из PSEUDO_KEYWORDS).

    per_specialty_limit — потолок псевдо-примеров на специальность; поднимается
    для режима максимального объёма (--hcm-limit), понижается при шуме разметки.
    Отбираются строки с наибольшим счётом словаря."""
    df = pd.read_parquet(raw / "hcm" / "data" / "train-00000-of-00001-5e7cb295b9cff0bf.parquet")
    by_spec: dict[str, list[tuple[float, dict]]] = {name: [] for name in PSEUDO_KEYWORDS}
    for r in df.itertuples(index=False):
        inp, out = clean(r.input), clean(r.output)
        if len(inp) < MIN_Q or len(out) < MIN_A or len(inp) + len(out) > MAX_PAIR:
            continue
        text = inp + "\n" + out[:400]
        best, best_score = None, 0.0
        for name, rules in PSEUDO_KEYWORDS.items():
            score = 0.0
            for rx, w in rules:
                hits = len(rx.findall(text))
                if hits:
                    score += w * min(hits, 2)
            if score >= 3 and score > best_score:
                best, best_score = name, score
        if best is not None and len(by_spec[best]) < per_specialty_limit:
            by_spec[best].append((best_score, {"source": "hcm_pseudo", "kind": "consult",
                                               "specialty": best, "question_en": inp,
                                               "answer_text": out, "pseudo_score": best_score}))
    rows = []
    for name, items in by_spec.items():
        items.sort(key=lambda x: -x[0])
        rows.extend(it for _, it in items)
        print(f"      HCM-псевдо {name}: {len(items)}")
    return rows


def to_messages(row: dict, spec_en: str, spec_ru: str) -> dict:
    system = SYSTEM_PROMPT.format(spec_en=spec_en, spec_ru=spec_ru.lower())
    if row["kind"] == "exam":
        answer = row["answer_text"]
        if row.get("explanation_en"):
            assistant = f"{REASON_HEADER}\n{row['explanation_en']}\n\n{DIAG_HEADER}\nAnswer: {row['answer_letters']}. {answer}"
        else:
            assistant = f"{DIAG_HEADER}\nAnswer: {row['answer_letters']}. {answer}"
    else:
        assistant = row["answer_text"]
    ex = {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": row["question_en"]},
        {"role": "assistant", "content": assistant},
    ]}
    if row["kind"] == "exam":
        ex["answer_letters"] = row.get("answer_letters", "")
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-specialty", type=int, default=4000,
                    help="целевой объём на специальность; 100000+ = взять всё доступное")
    ap.add_argument("--hcm-limit", type=int, default=20000,
                    help="потолок HCM-псевдо на специальность")
    ap.add_argument("--uniform-format", action="store_true",
                    help="у консультаций выделить блок «### Предварительный диагноз» "
                         "из формулировок врача (нивелирует неоднородность стилей)")
    args = ap.parse_args()

    cfg = load_config()
    raw = PROJECT_ROOT / cfg["data_raw"] / "en"
    proc_dir = PROJECT_ROOT / cfg["data_processed"]
    spec_dir = PROJECT_ROOT / cfg["data_specialties"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    print("[1/3] Чтение источников ...")
    rows = load_medmcqa(raw) + load_medquad(raw) + load_hcm_pseudo(raw, args.hcm_limit)

    by_spec: dict[str, list[dict]] = collections.defaultdict(list)
    seen: dict[str, set] = collections.defaultdict(set)
    for r in rows:
        h = md5(r["question_en"])
        if h in seen[r["specialty"]]:
            continue
        seen[r["specialty"]].add(h)
        by_spec[r["specialty"]].append(r)

    print("[2/3] Пулы и экспорт ...")
    report = ["# Отчёт подготовки данных по специалистам (EN-ветка)", "",
              f"per-specialty={args.per_specialty}, seed={SEED}", "",
              "| Специалист | medmcqa | medquad | hcm_pseudo | с объяснением | train/val/test |",
              "|---|---|---|---|---|---|"]

    # приоритет источников при сборке пула
    order = ["medmcqa", "medquad", "hcm_pseudo"]
    for spec in SPECIALTIES:
        pool = []
        remain = args.per_specialty
        for src in order:
            if remain <= 0:
                break
            part = [r for r in by_spec.get(spec.name, []) if r["source"] == src]
            if src == "hcm_pseudo":  # псевдо-разметку перемешиваем, берём сколько нужно
                rng.shuffle(part)
            take = part[:remain]
            pool.extend(take)
            remain -= len(take)
        rng.shuffle(pool)
        if args.uniform_format:
            for r in pool:
                if r["kind"] == "consult":
                    r["answer_text"] = wrap_consult(r["answer_text"])
        n = len(pool)
        n_test, n_val = max(1, int(n * TEST_SHARE)), max(1, int(n * VAL_SHARE))
        test, val, train = pool[:n_test], pool[n_test:n_test + n_val], pool[n_test + n_val:]

        write_jsonl(proc_dir / spec.name / "pool.jsonl", pool)
        sdir = spec_dir / spec.name
        spec_en = " / ".join(spec.names_en)
        write_jsonl(sdir / "train.jsonl", [to_messages(r, spec_en, spec.name) for r in train])
        write_jsonl(sdir / "val.jsonl", [to_messages(r, spec_en, spec.name) for r in val])
        write_jsonl(sdir / "test.jsonl", [to_messages(r, spec_en, spec.name) for r in test])

        c = collections.Counter(r["source"] for r in pool)
        expl = sum(1 for r in pool if r.get("explanation_en"))
        thin = " ⚠️ тонкий" if n < 2000 else ""  # порог абсолютный, а не от цели
        report.append(f"| {spec.name}{thin} | {c['medmcqa']} | {c['medquad']} | {c['hcm_pseudo']} "
                      f"| {expl} ({expl / max(n, 1) * 100:.0f}%) | {len(train)}/{len(val)}/{len(test)} |")
        print(f"      {spec.name:18s} {len(train):5d}/{len(val):3d}/{len(test):3d} "
              f"(mcqa {c['medmcqa']}, medquad {c['medquad']}, hcm {c['hcm_pseudo']}, expl {expl})")

    (proc_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[3/3] Отчёт: {proc_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
