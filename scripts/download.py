#!/usr/bin/env python
"""Этап 1. Скачивание источников в data/raw/ и печать их реальных схем.

Источники:
  - Toyhom Chinese-medical-dialogue-data (792k QA-пар, 6 отделений) —
    полное зеркало ticoAg/Chinese-medical-dialogue (JSON, ~635 МБ).
  - CMB (FreedomIntelligence, Apache-2.0): CMB-Exam train/val/test (merge-файлы)
    и CMB-Clin-qa (74 многоходовых клинических случая).

Скрипт идемпотентен: существующие файлы не перекачиваются.
После скачивания печатает схему и распределения каждого источника
и пишет data/raw/SOURCES.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils import load_config  # noqa: E402

TOYHOM_REPO = "ticoAg/Chinese-medical-dialogue"
TOYHOM_FILE = "data/train_0001_of_0001.json"

CMB_REPO = "FreedomIntelligence/CMB"
CMB_FILES = {
    "cmb_train.json": "CMB-Exam/CMB-train/CMB-train-merge.json",
    "cmb_val.json": "CMB-Exam/CMB-val/CMB-val-merge.json",
    "cmb_test.json": "CMB-Exam/CMB-test/CMB-test-choice-question-merge.json",
    "cmb_clin.json": "CMB-Clin-qa.json",
}


def download_all(raw_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths: dict[str, Path] = {}
    targets = {"toyhom.json": (TOYHOM_REPO, TOYHOM_FILE), **{k: (CMB_REPO, v) for k, v in CMB_FILES.items()}}
    for local_name, (repo, fname) in targets.items():
        dest = raw_dir / local_name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {dest.name} уже существует ({dest.stat().st_size / 1e6:.1f} МБ)")
        else:
            print(f"[down] {repo}/{fname} -> {dest.name} ...")
            got = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
            dest.write_bytes(Path(got).read_bytes())
            print(f"[ ok ] {dest.name}: {dest.stat().st_size / 1e6:.1f} МБ")
        paths[local_name] = dest
    return paths


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def describe_toyhom(path: Path) -> str:
    data = load_json(path)
    lines = [f"## Toyhom (ticoAg зеркало): {path.name}", ""]
    if isinstance(data, dict):
        lines.append(f"Корневой объект — dict, ключи: {list(data.keys())[:10]}")
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                lines.append(f"- '{k}': list[{len(v)}], поля примера: {list(v[0].keys())}")
                data = v
                break
    rows = data
    lines.append(f"Всего записей: {len(rows)}")
    if rows:
        lines.append(f"Поля: {list(rows[0].keys())}")
        # ищем поле-метку отделения
        for key in ("department", "科室", "depart", "category"):
            if key in rows[0]:
                cnt = collections.Counter(str(r.get(key, "")).strip() for r in rows)
                lines.append(f"Распределение '{key}' (топ-30):")
                for val, c in cnt.most_common(30):
                    lines.append(f"  {val}: {c}")
                break
        lines.append("")
        lines.append("Пример записи:")
        lines.append("```json")
        lines.append(json.dumps(rows[0], ensure_ascii=False, indent=2)[:1200])
        lines.append("```")
    return "\n".join(lines)


def describe_cmb_exam(path: Path) -> str:
    data = load_json(path)
    rows = data if isinstance(data, list) else data.get("data", data)
    # merge-файлы CMB-Exam бывают вложены: {'exam_type': [...]} или список с exam_type у каждой записи
    lines = [f"## CMB-Exam: {path.name}", ""]
    if isinstance(rows, dict):
        lines.append(f"Корневой объект — dict, ключи: {list(rows.keys())[:10]}")
        flat = []
        for k, v in rows.items():
            if isinstance(v, list):
                lines.append(f"- '{k}': list[{len(v)}]")
                flat.extend(v)
        rows = flat
    lines.append(f"Всего записей: {len(rows)}")
    if rows:
        lines.append(f"Поля: {list(rows[0].keys())}")
        for key in ("exam_type", "exam_class", "exam_subject"):
            if key in rows[0]:
                cnt = collections.Counter(str(r.get(key, "")).strip() for r in rows)
                lines.append(f"Распределение '{key}' (уникальных: {len(cnt)}; топ-40):")
                for val, c in cnt.most_common(40):
                    lines.append(f"  {val}: {c}")
        lines.append("")
        lines.append("Пример записи:")
        lines.append("```json")
        lines.append(json.dumps(rows[0], ensure_ascii=False, indent=2)[:1500])
        lines.append("```")
    return "\n".join(lines)


def describe_cmb_clin(path: Path) -> str:
    data = load_json(path)
    lines = [f"## CMB-Clin: {path.name}", ""]
    lines.append(f"Тип корня: {type(data).__name__}")
    rows = data if isinstance(data, list) else list(data.values())[0] if data else []
    lines.append(f"Записей: {len(rows)}")
    if rows:
        lines.append(f"Поля: {list(rows[0].keys()) if isinstance(rows[0], dict) else 'n/a'}")
        lines.append("```json")
        lines.append(json.dumps(rows[0], ensure_ascii=False, indent=2)[:1500])
        lines.append("```")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect-only", action="store_true", help="не скачивать, только показать схемы")
    args = ap.parse_args()

    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["data_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.inspect_only:
        download_all(raw_dir)

    sections = [
        describe_toyhom(raw_dir / "toyhom.json"),
        describe_cmb_exam(raw_dir / "cmb_train.json"),
        describe_cmb_exam(raw_dir / "cmb_val.json"),
        describe_cmb_clin(raw_dir / "cmb_clin.json"),
    ]
    report = "\n\n".join(sections)
    out = raw_dir / "SCHEMAS.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[ok] Схемы сохранены: {out}")

    sources = raw_dir / "SOURCES.md"
    sources.write_text(
        "# Источники данных\n\n"
        f"- **Toyhom Chinese-medical-dialogue-data** — HF-зеркало `{TOYHOM_REPO}`, файл `{TOYHOM_FILE}` "
        f"(локально `toyhom.json`). Оригинал: github.com/Toyhom/Chinese-medical-dialogue-data. "
        "~792 тыс. QA-пар, 6 отделений. Краулинг сайтов онлайн-консультаций, явной лицензии нет — "
        "**только некоммерческое исследование**.\n"
        f"- **CMB** — `{CMB_REPO}` (Apache-2.0), скачано {date.today().isoformat()}: "
        "CMB-Exam train/val/test (merge) + CMB-Clin-qa. Статья: NAACL 2024, FreedomIntelligence.\n",
        encoding="utf-8",
    )
    print(f"[ok] {sources}")


if __name__ == "__main__":
    main()
