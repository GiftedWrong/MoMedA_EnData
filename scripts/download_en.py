#!/usr/bin/env python
"""Шаг 1. Скачивание англоязычных источников в data/raw/en/ и печать их схем.

Источники (v2, англоязычная ветка):
  medmcqa   — openlifescienceai/medmcqa (Apache-2.0), ~183k экзаменационных
              виньеток с объяснениями; метки subject_name (21 предмет) и
              topic_name (~2400 тем) — основа разбиения по специалистам;
  mtsamples — tchebonenko/MedicalTranscriptions (CC0, зеркало Kaggle
              tboyle10/medicaltranscriptions), 4 999 реальных клинических
              заметок с метками ~40 специальностей — для роутера;
  medquad   — git clone abachaa/MedQuAD (NLM/NIH), 47 457 QA с 12 сайтов NIH;
              подмножество NCI (cancer.gov) ~13k — онколог;
  medqa     — GBaker/MedQA-USMLE-4-options (CC-BY 4.0), 11.4k клинических
              виньеток — eval-резерв;
  hcm       — lavita/ChatDoctor-HealthCareMagic-100k, 100k реальных
              консультаций; ЯВНОЙ ЛИЦЕНЗИИ НЕТ — только исследование;
  ddxplus   — aai530-group6/ddxplus (CC-BY), 1.3M синтетических пациентов:
              evidences + дифдиагнозы — аугментация роутера, резерв
              главного агента. Тяжёлый — качать отдельно: --source ddxplus.

Идемпотентен; после скачивания печатает схемы и распределения, пишет
data/raw/en/SCHEMAS.md и data/raw/en/SOURCES.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils import load_config  # noqa: E402

HF_SOURCES = {
    "medmcqa": ("openlifescienceai/medmcqa", ["*.json", "*.jsonl", "*.parquet", "*.csv"]),
    "mtsamples": ("tchebonenko/MedicalTranscriptions", ["*.json", "*.jsonl", "*.parquet", "*.csv", "*.zip"]),
    "medqa": ("GBaker/MedQA-USMLE-4-options", ["*.json", "*.jsonl", "*.parquet", "*.csv", "*.zip"]),
    "hcm": ("lavita/ChatDoctor-HealthCareMagic-100k", ["*.json", "*.jsonl", "*.parquet", "*.csv"]),
    "ddxplus": ("aai530-group6/ddxplus", ["*.json", "*.jsonl", "*.parquet", "*.csv", "*.zip"]),
}
MEDQUAD_GIT = "https://github.com/abachaa/MedQuAD.git"

SOURCES_MD = """# Источники данных (англоязычная ветка)

Скачано: {today}

| Источник | Репозиторий | Лицензия | Использование |
|---|---|---|---|
| MedMCQA | openlifescienceai/medmcqa | Apache-2.0 | виньетки специалистов + роутер |
| MTSamples | tchebonenko/MedicalTranscriptions (ориг. Kaggle tboyle10) | CC0 | роутер (реальные заметки, метки специальностей) |
| MedQuAD | github.com/abachaa/MedQuAD (NLM/NIH) | CC-style; контент NIH в осн. public domain | Онколог (NCI/cancer.gov ~13k) + доборы |
| MedQA-USMLE-4 | GBaker/MedQA-USMLE-4-options | CC-BY 4.0 | eval-резерв |
| HealthCareMagic-100k | lavita/ChatDoctor-HealthCareMagic-100k | **явной лицензии нет** | только некоммерческое исследование |
| DDXPlus | aai530-group6/ddxplus (ориг. Zenodo/GitHub mila-iqia) | CC-BY | аугментация роутера, резерв главного агента |
"""


def download_hf(name: str, raw: Path) -> Path:
    from huggingface_hub import snapshot_download

    repo, patterns = HF_SOURCES[name]
    dest = raw / name
    if dest.exists() and any(dest.rglob("*")) and not (dest / ".incomplete").exists():
        print(f"[skip] {name} уже скачан")
        return dest
    (dest / ".incomplete").parent.mkdir(parents=True, exist_ok=True)
    (dest / ".incomplete").touch()
    print(f"[down] {repo} -> {dest} ...")
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest),
                      allow_patterns=patterns + ["README.md"])
    (dest / ".incomplete").unlink()
    return dest


def download_medquad(raw: Path) -> Path:
    dest = raw / "medquad"
    if (dest / ".git").exists() or any(dest.rglob("*.xml")):
        print("[skip] medquad уже скачан")
        return dest
    print(f"[down] {MEDQUAD_GIT} -> {dest} ...")
    subprocess.run(["git", "clone", "--depth", "1", MEDQUAD_GIT, str(dest)], check=True)
    return dest


def _iter_jsonish(path: Path):
    """Читает json/jsonl/parquet/csv, возвращает список dict (для небольших файлов)."""
    import pandas as pd

    suf = path.suffix.lower()
    if suf == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data
    if suf == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    if suf == ".csv":
        return pd.read_csv(path).to_dict("records")
    return []


def describe(name: str, dest: Path) -> str:
    lines = [f"## {name}", ""]
    files = sorted(p for p in dest.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".parquet", ".csv"}
                   and ".incomplete" not in p.name and ".cache" not in p.parts)
    if not files:
        lines.append("файлов данных не найдено")
        return "\n".join(lines)
    for path in files:
        rel = path.relative_to(dest)
        try:
            if path.stat().st_size > 600e6:  # DDXPlus: не грузим целиком в память
                lines.append(f"- `{rel}`: {path.stat().st_size / 1e6:.0f} МБ (большой, схема по первым строкам)")
                with path.open(encoding="utf-8", errors="replace") as f:
                    first = f.readline()
                lines.append("```json")
                lines.append(first[:1200])
                lines.append("```")
                continue
            rows = _iter_jsonish(path)
            if not rows:
                continue
            lines.append(f"- `{rel}`: {len(rows)} записей")
            first = rows[0] if isinstance(rows, list) else None
            if isinstance(first, dict):
                lines.append(f"  поля: {list(first.keys())}")
                sample = json.dumps(first, ensure_ascii=False, default=str)[:900]
                lines.append("  пример:")
                lines.append("  ```json")
                lines.append("  " + sample)
                lines.append("  ```")
                for key, cap in (("subject_name", 40), ("topic_name", 80), ("medical_specialty", 45)):
                    if key in first:
                        cnt = collections.Counter(str(r.get(key, "")) for r in rows)
                        lines.append(f"  распределение '{key}' (уникальных {len(cnt)}, топ-{cap}):")
                        for v, c in cnt.most_common(cap):
                            lines.append(f"    {v}: {c}")
        except Exception as e:  # схемы печатаются best-effort
            lines.append(f"- `{rel}`: ошибка чтения {type(e).__name__}: {e}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=[*HF_SOURCES, "medquad"], default=None,
                    help="скачать только указанный источник (по умолчанию все, кроме ddxplus)")
    ap.add_argument("--all", action="store_true", help="включая тяжёлый ddxplus")
    args = ap.parse_args()

    cfg = load_config()
    raw = PROJECT_ROOT / cfg["data_raw"] / "en"
    raw.mkdir(parents=True, exist_ok=True)

    todo = []
    if args.source:
        todo = [args.source]
    else:
        todo = ["medmcqa", "mtsamples", "medquad", "medqa", "hcm"] + (["ddxplus"] if args.all else [])

    for name in todo:
        if name == "medquad":
            download_medquad(raw)
        else:
            download_hf(name, raw)

    sections = []
    for name in todo:
        p = raw / name
        if p.exists():
            sections.append(describe(name, p))
    report = "\n\n".join(sections) if sections else "ничего не скачано"
    out = raw / "SCHEMAS.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[ok] Схемы: {out}")

    (raw / "SOURCES.md").write_text(SOURCES_MD.format(today=date.today().isoformat()), encoding="utf-8")
    print(f"[ok] {raw / 'SOURCES.md'}")


if __name__ == "__main__":
    main()
