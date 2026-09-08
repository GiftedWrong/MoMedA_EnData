#!/usr/bin/env python
"""Шаг 4. Датасет EN-роутера на 14 классов → data/router/.

Вход — текст случая на английском (вопрос MedMCQA / клиническая заметка
MTSamples / «жалобы» DDXPlus из декодированных улик / консультация
HealthCareMagic с псевдо-меткой). Выход ассистента — строка формата
экосистемы роутера: «Рекомендуемый специалист: <Класс>.» (классы — русские
имена из таксономии 14 специалистов).

Балансировка: cap --per-class (по умолчанию 1500), при нехватке берётся
доступное (Проктолог и др. — меньше). Сплит 90/5/5, дедуп, seed 42.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from specialties import (SPECIALTIES, map_medmcqa, MTSAMPLES_MAP,  # noqa: E402
                         map_ddx_pathology, pseudo_label)
from utils import md5, write_jsonl, load_config  # noqa: E402

SEED = 42
VAL_SHARE, TEST_SHARE = 0.05, 0.05
MAX_TEXT = 2500  # обрезка длинных заметок для роутера

ROUTER_SYSTEM = (
    "You are a medical triage router. Read the patient case in English and select "
    "the single most appropriate specialist. Answer with exactly one line in the "
    "format «Рекомендуемый специалист: <специалист>.», choosing one of: "
    "{classes}.\n"
    "Ты — медицинский ассистент-маршрутизатор. Прочитай случай на английском и "
    "выбери одного специалиста. Отвечай одной строкой строго в формате "
    "«Рекомендуемый специалист: <специалист>.» из списка: {classes}."
)


def clean(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return "\n".join(line.strip() for line in text.strip().splitlines()).strip()


def texts_medmcqa(raw: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = collections.defaultdict(list)
    for split in ["train", "validation"]:
        df = pd.read_parquet(raw / "medmcqa" / "data" / f"{split}-00000-of-00001.parquet")
        for r in df.itertuples(index=False):
            spec = map_medmcqa(r.subject_name, r.topic_name)
            q = clean(r.question)
            if spec and len(q) >= 20:
                out[spec].append(q)
    return out


def texts_mtsamples(raw: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = collections.defaultdict(list)
    df = pd.read_csv(raw / "mtsamples" / "mtsamples.csv")
    for r in df.itertuples(index=False):
        spec = MTSAMPLES_MAP.get(str(r.medical_specialty).strip())
        text = clean(r.description) + "\n" + clean(r.transcription)
        if spec and len(text) >= 50:
            out[spec].append(text[:MAX_TEXT])
    return out


def texts_ddxplus(raw: Path, per_class_cap: int = 600) -> dict[str, list[str]]:
    """DDXPlus: улики (коды E_*) → текст вопросов-ответов пациента; метка — PATHOLOGY."""
    ev = json.loads((raw / "ddxplus" / "release_evidences.json").read_text(encoding="utf-8"))
    out: dict[str, list[str]] = collections.defaultdict(list)
    cap = {"Терапевт": per_class_cap // 2}  # дефолтная метка — не даём затопить класс
    for chunk in pd.read_csv(raw / "ddxplus" / "train.csv",
                             usecols=["AGE", "SEX", "PATHOLOGY", "EVIDENCES"],
                             chunksize=25_000):
        for r in chunk.itertuples(index=False):
            spec = map_ddx_pathology(r.PATHOLOGY)
            if len(out[spec]) >= cap.get(spec, per_class_cap):
                continue
            codes = re.findall(r"E_\d+(?:_V_\d+)?", str(r.EVIDENCES))
            parts = []
            for code in codes[:15]:
                base = code.split("_V_")[0]
                meta = ev.get(base, {})
                qtxt = meta.get("question_en") or meta.get("name_en") or base
                val = code.split("_V_")[1] if "_V_" in code else None
                vtxt = (meta.get("value_meaning", {}).get(val, {}) or {}).get("en", "")
                parts.append(f"{qtxt}" + (f" — {vtxt}" if vtxt else ""))
            if not parts:
                continue
            sex = "male" if str(r.SEX) == "M" else "female"
            text = f"Patient: {int(r.AGE)} y.o. {sex}. " + " ".join(parts)
            out[spec].append(text)
        if all(len(v) >= per_class_cap for v in out.values()):
            break
    return out


def texts_hcm(raw: Path, per_class_cap: int = 400) -> dict[str, list[str]]:
    out: dict[str, list[str]] = collections.defaultdict(list)
    df = pd.read_parquet(raw / "hcm" / "data" / "train-00000-of-00001-5e7cb295b9cff0bf.parquet")
    for r in df.itertuples(index=False):
        inp = clean(r.input)
        if len(inp) < 50:
            continue
        spec = pseudo_label(inp)
        if spec and len(out[spec]) < per_class_cap:
            out[spec].append(inp[:MAX_TEXT])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=1500)
    args = ap.parse_args()

    cfg = load_config()
    raw = PROJECT_ROOT / cfg["data_raw"] / "en"
    rng = random.Random(SEED)

    print("[1/3] Сбор текстов по классам ...")
    sources = [
        ("medmcqa", texts_medmcqa(raw), 900),
        ("mtsamples", texts_mtsamples(raw), 500),
        ("ddxplus", texts_ddxplus(raw), 600),
        ("hcm_pseudo", texts_hcm(raw), 400),
    ]

    classes = [s.name for s in SPECIALTIES]
    system = ROUTER_SYSTEM.format(classes=", ".join(classes))

    report = ["# Отчёт датасета роутера (EN, 14 классов)", "",
              f"per-class cap = {args.per_class}, seed = {SEED}", "",
              "| Класс | " + " | ".join(name for name, _, _ in sources) + " | train/val/test |",
              "|---" * (len(sources) + 2) + "|"]

    rows_by_class: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    seen: set[str] = set()
    for src_name, texts, cap_src in sources:
        for spec, items in texts.items():
            rng.shuffle(items)
            taken = 0
            for t in items:
                if taken >= cap_src or len(rows_by_class[spec]) >= args.per_class:
                    break
                h = md5(t)
                if h in seen:
                    continue
                seen.add(h)
                rows_by_class[spec].append((src_name, t))
                taken += 1

    out_dir = PROJECT_ROOT / cfg.get("data_router", "data/router")
    train_rows, val_rows, test_rows = [], [], []
    for spec in classes:
        rows = rows_by_class.get(spec, [])
        rng.shuffle(rows)
        n = len(rows)
        n_test, n_val = max(1, int(n * TEST_SHARE)), max(1, int(n * VAL_SHARE))
        test, val, train = rows[:n_test], rows[n_test:n_test + n_val], rows[n_test + n_val:]
        for split_rows, bucket in [(train, train_rows), (val, val_rows), (test, test_rows)]:
            for src_name, text in split_rows:
                bucket.append({"messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": f"Рекомендуемый специалист: {spec}."},
                ], "label": spec, "source": src_name})
        c = collections.Counter(s for s, _ in rows)
        report.append(f"| {spec} | " + " | ".join(str(c.get(name, 0)) for name, _, _ in sources)
                      + f" | {len(train)}/{len(val)}/{len(test)} |")
        print(f"   {spec:18s} " + " ".join(f"{name}={c.get(name, 0)}" for name, _, _ in sources)
              + f"  итого {n}")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    total = len(train_rows) + len(val_rows) + len(test_rows)
    report += ["", f"Всего: {total} ({len(train_rows)}/{len(val_rows)}/{len(test_rows)})", "",
               "Пропорции источников: " + json.dumps(
                   dict(collections.Counter(r["source"] for r in train_rows)), ensure_ascii=False)]
    (PROJECT_ROOT / cfg["data_processed"] / "ROUTER_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print(f"[2/3] Роутер: {total} примеров -> {out_dir}")
    print(f"[3/3] Отчёт: {PROJECT_ROOT / cfg['data_processed'] / 'ROUTER_REPORT.md'}")


if __name__ == "__main__":
    main()
