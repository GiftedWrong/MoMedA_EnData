#!/usr/bin/env python
"""Сквозной прогон: RU-жалоба → /api/case → RU-ответ, метрики конвейера."""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from specialties import map_ddx_pathology  # noqa: E402
from utils import load_config  # noqa: E402

CFG = load_config()
SEED = 42
TRANSLATE_SYSTEM = (
    "You are a precise medical translator. Translate the patient's message from "
    "English to Russian. Preserve all medical terms, drug names, dosages, units "
    "and numbers exactly. Output only the translation, in Russian."
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def decode_evidences(ev_json: dict, row) -> str:
    codes = re.findall(r"E_\d+(?:_V_\d+)?", str(row.EVIDENCES))
    parts = []
    for code in codes[:15]:
        base = code.split("_V_")[0]
        meta = ev_json.get(base, {})
        q = meta.get("question_en") or base
        val = code.split("_V_")[1] if "_V_" in code else None
        vtxt = (meta.get("value_meaning", {}).get(val, {}) or {}).get("en", "")
        parts.append(f"{q}" + (f" — {vtxt}" if vtxt else ""))
    sex = "male" if str(row.SEX) == "M" else "female"
    return f"Patient: {int(row.AGE)} y.o. {sex}. " + " ".join(parts)


def prepare_complaints(n: int, cache: Path) -> list[dict]:
    if cache.exists():
        rows = [json.loads(l) for l in cache.open(encoding="utf-8")]
        print(f"[skip] кэш жалоб: {len(rows)} шт. ({cache})")
        return rows

    import pandas as pd

    ev = json.loads((PROJECT_ROOT / CFG["data_raw"] / "en" / "ddxplus" /
                     "release_evidences.json").read_text(encoding="utf-8"))
    df = pd.read_csv(PROJECT_ROOT / CFG["data_raw"] / "en" / "ddxplus" / "train.csv",
                     usecols=["AGE", "SEX", "PATHOLOGY", "EVIDENCES"])
    rng = random.Random(SEED)
    df = df.sample(frac=1.0, random_state=SEED)
    picked = []
    per_class = collections.Counter()
    CAP = max(4, n // 8)  # не даём одному классу занять весь набор
    for row in df.itertuples(index=False):
        spec = map_ddx_pathology(row.PATHOLOGY)
        if not spec or per_class[spec] >= CAP or len(picked) >= n:
            if len(picked) >= n:
                break
            continue
        per_class[spec] += 1
        picked.append({"specialty_gold": spec, "pathology": row.PATHOLOGY,
                       "text_en": decode_evidences(ev, row)})
        if len(picked) >= n:
            break

    print(f"[1/3] Перевод {len(picked)} жалоб в RU (одна загрузка переводчика)...")
    import torch
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=str((PROJECT_ROOT / CFG["base_model"]).resolve()),
        max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False)
    FastLanguageModel.for_inference(model)
    with torch.inference_mode():
        for i, r in enumerate(picked):
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": TRANSLATE_SYSTEM},
                 {"role": "user", "content": r["text_en"]}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            r["text_ru"] = tok.decode(out[0][ids["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
            if (i + 1) % 25 == 0:
                print(f"      {i + 1}/{len(picked)}")
    del model, tok
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    with cache.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] жалобы сохранены: {cache}")
    return picked


def post_case(port: int, text: str, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/case", data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--out", default=None,
                    help="имя отчёта в runs/ (по умолчанию E2E_REPORT.md)")
    args = ap.parse_args()

    cache = PROJECT_ROOT / CFG["data_processed"] / "e2e_ru_complaints.jsonl"
    complaints = prepare_complaints(args.n, cache)
    complaints.sort(key=lambda r: r["specialty_gold"])  # меньше смен агента

    print(f"[2/3] Сквозной прогон {len(complaints)} случаев через /api/case ...")
    # построчный дамп сырых ответов: датасет для внешнего судьи (judge_medgemma)
    # и разбора ошибок; пишем с flush — прогресс по строкам, сбой не теряет хвост
    dump_path = PROJECT_ROOT / CFG["runs_dir"] / \
        (Path(args.out or "E2E_REPORT.md").stem + "_raw.jsonl")
    dump = dump_path.open("w", encoding="utf-8")
    results = []
    t_start = time.time()
    for i, c in enumerate(complaints):
        t0 = time.time()
        try:
            r = post_case(args.port, c["text_ru"])
            ok_route = r["specialty"] == c["specialty_gold"]
            # консилиум: сервер может вернуть specialties (top-1 + top-2 при
            # малой марже роутера); без этого поля вырождается в [specialty]
            specs = r.get("specialties") or [r["specialty"]]
            ok_recall2 = c["specialty_gold"] in specs
            consult = bool(r.get("consult"))
            ans = r.get("answer_ru", "")
            diag_block = ("Итоговый диагноз" in ans) or ("Предварительный диагноз" in ans)
            cjk = bool(_CJK_RE.search(ans))
            latin_leak = len(_LATIN_RE.findall(ans)) / max(len(ans), 1) > 0.10
            row = {"gold": c["specialty_gold"], "pred": r["specialty"],
                   "ok": ok_route, "ok2": ok_recall2, "consult": consult,
                   "diag": diag_block, "cjk": cjk,
                   "latin": latin_leak, "sec": time.time() - t0,
                   "trace": r["trace"], "answer_ru": ans,
                   "text_ru": c["text_ru"], "pathology": c["pathology"]}
            results.append(row)
            dump.write(json.dumps({
                "gold": c["specialty_gold"], "pathology": c["pathology"],
                "text_ru": c["text_ru"], "case_en": r.get("case_en"),
                "specialty": r["specialty"], "specialties": specs,
                "consult": consult,
                "opinions": r.get("opinions"),
                "preliminary_en": r.get("preliminary_en"),
                "final_en": r.get("final_en"), "answer_ru": ans,
                "latency_ms": r.get("latency_ms"),
                "trace_ms": {t["step"]: t["ms"] for t in r.get("trace", [])},
            }, ensure_ascii=False) + "\n")
            dump.flush()
        except Exception as e:
            results.append({"gold": c["specialty_gold"], "pred": None, "ok": False,
                            "error": str(e), "sec": time.time() - t0})
        done = i + 1
        if done % 10 == 0:
            acc = sum(x["ok"] for x in results) / done
            print(f"      {done}/{len(complaints)} · маршрут {acc*100:.0f}% · "
                  f"~{(time.time()-t_start)/done:.1f}с/случай")

    dump.close()
    print(f"[ok] сырой дамп: {dump_path}")

    print("[3/3] Отчёт ...")
    n = len(results)
    ok = sum(x["ok"] for x in results)
    ok2 = sum(x.get("ok2") for x in results)
    consults = [x for x in results if x.get("consult")]
    plain = [x for x in results if not x.get("consult")]
    diag = sum(x.get("diag") for x in results)
    cjk = sum(x.get("cjk") for x in results)
    latin = sum(x.get("latin") for x in results)
    secs = [x["sec"] for x in results]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")  # noqa: E731
    steps = collections.defaultdict(list)
    for x in results:
        for t in x.get("trace", []):
            steps[t["step"]].append(t["ms"])
    by_class = collections.defaultdict(lambda: [0, 0, 0])
    for x in results:
        by_class[x["gold"]][2] += 1
        by_class[x["gold"]][0] += x["ok"]
        by_class[x["gold"]][1] += x.get("ok2", x["ok"])

    lines = ["# Сквозной e2e-бенчмарк (RU жалоба → /api/case → RU ответ)", "",
             f"Случаев: {n} (DDXPlus evidences, переведены в RU offline, seed {SEED})",
             f"Сервер: 127.0.0.1:{args.port}", "",
             f"**Маршрутизация end-to-end (top-1): {ok}/{n} = {ok/n*100:.1f}%**",
             f"**Recall@2 (золото среди вызванных специалистов): {ok2}/{n} = {ok2/n*100:.1f}%**",
             f"- консилиумов (2 специалиста): {len(consults)}/{n} "
             f"({len(consults)/max(n,1)*100:.0f}%)",
             f"- блок «Предварительный диагноз» в RU-ответе: {diag}/{n} ({diag/n*100:.0f}%)",
             f"- CJK-утечки в ответе: {cjk}/{n}; латиница >10%: {latin}/{n}",
             f"- латентность: медиана {med(secs):.0f} с · средняя {sum(secs)/n:.0f} с"
             + (f" · без консилиума {med([x['sec'] for x in plain]):.0f} с"
                f" · с консилиумом {med([x['sec'] for x in consults]):.0f} с"
                if consults and plain else ""),
             "", "## Латентность по шагам (медиана)", ""]
    for sname, vals in steps.items():
        lines.append(f"- {sname}: {sorted(vals)[len(vals)//2]/1000:.1f} с")
    lines += ["", "## Маршрутизация по классам (top-1 / recall@2)", ""]
    for cls, (a, a2, b) in sorted(by_class.items()):
        lines.append(f"- {cls}: {a}/{b} ({a/max(b,1)*100:.0f}%) · "
                     f"recall@2 {a2}/{b} ({a2/max(b,1)*100:.0f}%)")
    lines += ["", "## Сэмплы (по одному на класс)", ""]
    seen = set()
    for x in results:
        if x["gold"] in seen or "answer_ru" not in x:
            continue
        seen.add(x["gold"])
        lines += [f"### {x['gold']} → {x['pred']} ({'✓' if x['ok'] else '✗'})",
                  f"**Жалоба (RU):** {x['text_ru'][:250]}", "",
                  f"**Ответ:** {x['answer_ru'][:500]}", "", "---", ""]
    out = PROJECT_ROOT / CFG["runs_dir"] / (args.out or "E2E_REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
