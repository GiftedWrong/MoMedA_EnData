#!/usr/bin/env python
"""Данные мастера: случай + мнения специалистов → итоговый диагноз."""
from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from specialties import map_ddx_pathology, SPECIALTIES  # noqa: E402
from utils import load_config, write_jsonl, md5  # noqa: E402

CFG = load_config()
SEED = 42

CHIEF_SYSTEM = (
    "You are the senior physician of a multi-agent medical system. Several "
    "specialists have given their preliminary diagnoses for the case. Weigh "
    "their opinions against the patient's findings and formulate the final "
    "diagnosis. Answer in English only.\n"
    "Ты — старший врач мультиагентной системы. Взвесь предварительные диагнозы "
    "специалистов против данных случая и сформулируй итоговый диагноз. "
    "Отвечай только на английском языке."
)


def decode_evidences(ev_json: dict, row) -> tuple[str, list[str]]:
    codes = re.findall(r"E_\d+(?:_V_\d+)?", str(row.EVIDENCES))
    parts, short = [], []
    for code in codes[:15]:
        base = code.split("_V_")[0]
        meta = ev_json.get(base, {})
        q = meta.get("question_en") or base
        val = code.split("_V_")[1] if "_V_" in code else None
        vtxt = (meta.get("value_meaning", {}).get(val, {}) or {}).get("en", "")
        parts.append(f"{q}" + (f" — {vtxt}" if vtxt else ""))
        short.append(q.lower().replace("do you have ", "").replace("?", "").strip())
    sex = "male" if str(row.SEX) == "M" else "female"
    return f"Patient: {int(row.AGE)} y.o. {sex}. " + " ".join(parts), short


def build_example(rng: random.Random, vignette: str, evid_short: list[str],
                  pathology: str, differential: list[tuple[str, float]]) -> dict | None:
    # топ-4 гипотез по вероятности; истину добавляем с P=0.5, если её нет
    hyps = [d for d in differential if d[0] != pathology][:4]
    if rng.random() < 0.5 or not hyps:
        hyps.append((pathology, round(rng.uniform(0.15, 0.55), 3)))
    rng.shuffle(hyps)

    opinions = []
    for disease, p in hyps:
        spec = map_ddx_pathology(disease)
        opinions.append(f"- {disease} (preliminary diagnosis of a {spec}, "
                        f"estimated likelihood {p:.2f})")
    return finalize_example(vignette, opinions, pathology,
                            clues="; ".join(evid_short[:3]),
                            reasoning=None)


def finalize_example(case: str, opinions: list[str], gold: str,
                     clues: str = "", reasoning: str | None = None) -> dict:
    """Сборка пары мастера: случай + мнения → итоговый диагноз."""
    user = (f"Case findings:\n{case}\n\nSpecialists' preliminary diagnoses:\n"
            + "\n".join(opinions) + "\n\nFormulate the final diagnosis.")
    if not reasoning:
        alts = [o.lstrip("- ").split(" (")[0] for o in opinions
                if gold.lower() not in o.lower()]
        runner = alts[0] if alts else "the alternatives"
        reasoning = (f"Among the preliminary diagnoses, {gold} best explains the "
                     f"key findings ({clues or 'the clinical picture'}). "
                     f"The alternative {runner} fits less well. Weighing the "
                     f"specialists' opinions against the case, the final diagnosis "
                     f"is {gold}.")
    assistant = f"### Рассуждение\n{reasoning}\n\n### Итоговый диагноз\n{gold}"
    return {"messages": [
        {"role": "system", "content": CHIEF_SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ], "gold": gold}


# --- MedMCQA: дистракторы = конфликтующие предварительные, exp = рассуждение ---
def build_mcqa_examples(n: int, rng: random.Random) -> list[dict]:
    from specialties import map_medmcqa

    rows = []
    for split in ("train", "validation"):
        df = pd.read_parquet(PROJECT_ROOT / CFG["data_raw"] / "en" / "medmcqa" /
                             "data" / f"{split}-00000-of-00001.parquet")
        for r in df.itertuples(index=False):
            spec = map_medmcqa(r.subject_name, r.topic_name)
            if not spec or len(str(r.question)) < 60:
                continue
            opts = {i: str(v).strip() for i, v in
                    enumerate([r.opa, r.opb, r.opc, r.opd]) if str(v).strip()}
            gold_idx = int(r.cop)
            if gold_idx not in opts:
                continue
            rows.append({"spec": spec, "question": str(r.question).strip(),
                         "opts": opts, "gold": opts[gold_idx],
                         "exp": str(r.exp).strip() if str(r.exp) not in ("nan", "") else ""})
    rng.shuffle(rows)
    by_class = collections.Counter()
    out = []
    cap = max(120, n // 6)
    for r in rows:
        if len(out) >= n:
            break
        if by_class[r["spec"]] >= cap:
            continue
        by_class[r["spec"]] += 1
        distractors = [v for k, v in r["opts"].items() if v != r["gold"]]
        rng.shuffle(distractors)
        opinions = [f"- {r['gold']} (preliminary diagnosis of a {r['spec']})"] + \
                   [f"- {d} (differing preliminary diagnosis)" for d in distractors[:3]]
        reasoning = None
        if len(r["exp"]) >= 80:
            reasoning = r["exp"][:900]
        out.append(finalize_example(r["question"], opinions, r["gold"],
                                    reasoning=reasoning))
    return out


# --- MedQA-USMLE: без метки специальности — мнения без атрибуции ---
def build_usmle_examples(n: int, rng: random.Random) -> list[dict]:
    lines = []
    for split in ("train", "test"):
        p = PROJECT_ROOT / CFG["data_raw"] / "en" / "medqa" / f"phrases_no_exclude_{split}.jsonl"
        if p.exists():
            lines += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    rng.shuffle(lines)
    out = []
    for r in lines[:n]:
        opts = {k: str(v).strip() for k, v in (r.get("options") or {}).items()
                if str(v).strip()}
        gold = str(r.get("answer", "")).strip()
        if not gold or gold not in opts.values() or len(str(r["question"])) < 60:
            continue
        distractors = [v for v in opts.values() if v != gold]
        rng.shuffle(distractors)
        opinions = [f"- {gold} (preliminary diagnosis)"] + \
                   [f"- {d} (differing preliminary diagnosis)" for d in distractors[:2]]
        out.append(finalize_example(str(r["question"]).strip(), opinions, gold))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=15000, help="пар из DDXPlus")
    ap.add_argument("--n-mcqa", type=int, default=7000, help="пар из MedMCQA (все 14 спец.)")
    ap.add_argument("--n-usmle", type=int, default=2000, help="пар из MedQA-USMLE")
    args = ap.parse_args()
    rng = random.Random(SEED)

    raw = PROJECT_ROOT / CFG["data_raw"] / "en" / "ddxplus"
    ev = json.loads((raw / "release_evidences.json").read_text(encoding="utf-8"))
    df = pd.read_csv(raw / "train.csv",
                     usecols=["AGE", "SEX", "PATHOLOGY", "EVIDENCES",
                              "DIFFERENTIAL_DIAGNOSIS"])
    print(f"[1/3] DDXPlus: {len(df)} строк, беру {args.n} ...")
    rows, seen = [], set()
    for row in df.sample(frac=1.0, random_state=SEED).itertuples(index=False):
        if len(rows) >= args.n:
            break
        try:
            diff = ast.literal_eval(str(row.DIFFERENTIAL_DIAGNOSIS))
        except (ValueError, SyntaxError):
            continue
        vignette, short = decode_evidences(ev, row)
        h = md5(vignette)
        if h in seen or len(vignette) < 60:
            continue
        seen.add(h)
        ex = build_example(rng, vignette, short, row.PATHOLOGY, diff)
        if ex is not None:
            rows.append(ex)

    print(f"[2/3] MedMCQA: до {args.n_mcqa} пар (дистракторы = мнения, exp = рассуждение) ...")
    rows += build_mcqa_examples(args.n_mcqa, rng)
    print(f"[3/3] MedQA-USMLE: до {args.n_usmle} пар ...")
    rows += build_usmle_examples(args.n_usmle, rng)

    rng.shuffle(rows)
    n = len(rows)
    n_test, n_val = max(1, int(n * 0.05)), max(1, int(n * 0.05))
    test, val, train = rows[:n_test], rows[n_test:n_test + n_val], rows[n_test + n_val:]
    out = PROJECT_ROOT / "data" / "chief"
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)
    print(f"[ok] train/val/test = {len(train)}/{len(val)}/{len(test)}; всего {n}")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
