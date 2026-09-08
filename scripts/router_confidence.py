#!/usr/bin/env python
"""Калибровка уверенности роутера: p(class) скорингом 14 кандидатов.

Роутер генеративный и в рантайме отдаёт только argmax-строку. Здесь для
каждого кейса test-сплита считаем средний logprob на токен каждого из 14
ответов «Рекомендуемый специалист: <Класс>.» (teacher forcing, один батч
на кейс) — получаем p(class), маржу top1−top2 и recall@2. Это основа
триггера консилиума: маржа ниже порога → вызвать второго специалиста.

Триггер-метрика — лог-шансы ln(p1/p2), «top-1 во сколько раз вероятнее
top-2»: у вероятностной маржи p1−p2 сжатый диапазон и порог нелинеен.

Выход: runs/router_confidence/test_scored.jsonl (пишем построчно, прогресс
по mtime) + отчёт data/processed/ROUTER_CONFIDENCE.md с таблицей
«порог → доля консилиумов → покрытие» и рекомендуемым порогом.
--aggregate-only пересчитывает отчёт из сохранённого скоринга без GPU.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from specialties import SPECIALTIES  # noqa: E402
from utils import load_config, read_jsonl, project_path  # noqa: E402

CLASSES = [s.name for s in SPECIALTIES]
CANDIDATES = [f"Рекомендуемый специалист: {c}." for c in CLASSES]
MAX_PROMPT = 1900   # потолок длины промпта; длинные обрезаем слева (редкость)
SEED = 42


def resolve_model(models_dir: Path) -> str:
    """v2 приоритет, v1 фолбэк — как в inference/server.py."""
    v2 = models_dir / "med-router-en-v2-3b"
    return str(v2 if v2.is_dir() else models_dir / "med-router-en-3b")


@torch.inference_mode()
def score_candidates(model, prompt_ids: torch.Tensor, cand_ids: list[torch.Tensor],
                     pad_id: int) -> list[float]:
    """Средний logprob на токен для каждого кандидата. Нормировка длины
    обязательна: «Терапевт» и «Гастроэнтеролог» токенизируются по-разному.
    Кандидаты токенизируем отдельно и прицепляем к промпту — ровно те
    продолжения, которые могла породить генерация после chat-template."""
    plen = prompt_ids.shape[1]
    seqs = [torch.cat([prompt_ids, c], dim=1) for c in cand_ids]
    width = max(s.shape[1] for s in seqs)
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : s.shape[1]] = s[0]
        attn[i, : s.shape[1]] = 1
    input_ids, attn = input_ids.to("cuda"), attn.to("cuda")
    logits = model(input_ids, attention_mask=attn).logits
    scores = []
    for i, s in enumerate(seqs):
        clen = s.shape[1] - plen
        # токены кандидата [plen, plen+clen) предсказываются позициями [plen-1, plen+clen-1)
        step = logits[i, plen - 1: plen - 1 + clen].float()
        lsm = F.log_softmax(step, dim=-1)
        tgt = input_ids[i, plen: plen + clen]
        scores.append(lsm.gather(-1, tgt.unsqueeze(-1)).mean().item())
    return scores


def median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="модель (по умолчанию v2 → v1, как в сервере)")
    ap.add_argument("--test", default=None, help="путь к test.jsonl (по умолчанию data/router)")
    ap.add_argument("--limit", type=int, default=0, help="ограничить test (для быстрого прогона)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="пересчитать отчёт из сохранённого test_scored.jsonl, без GPU")
    ap.add_argument("--target-consult", type=float, default=0.30,
                    help="целевая доля кейсов с консилиумом при выборе порога")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    cfg = load_config()

    out_dir = PROJECT_ROOT / "runs" / "router_confidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "test_scored.jsonl"
    model_path = args.model or resolve_model(project_path(cfg["models_dir"]))

    truncated = 0
    results = []
    if args.aggregate_only and scored_path.exists():
        results = [json.loads(l) for l in scored_path.open(encoding="utf-8") if l.strip()]
        print(f"aggregate-only: {len(results)} строк из {scored_path}")
    else:
        # чистый transformers, без unsloth: инференс-правило проекта
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        model.eval().to("cuda")

        test_path = Path(args.test) if args.test else project_path(cfg["data_router"]) / "test.jsonl"
        rows = read_jsonl(test_path)
        if args.limit:
            rows = rows[: args.limit]
        print(f"model: {model_path}\ntest: {len(rows)} примеров из {test_path}")

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        cand_ids = [tok(c, add_special_tokens=False, return_tensors="pt")["input_ids"]
                    for c in CANDIDATES]
        cand_lens = [c.shape[1] for c in cand_ids]

        with scored_path.open("w", encoding="utf-8") as fout:
            for r in tqdm(rows, desc="scoring", mininterval=5.0):
                # ассистента из промпта исключаем: иначе золото утекает в контекст
                prompt_msgs = [m for m in r["messages"] if m["role"] != "assistant"]
                prompt = tok.apply_chat_template(prompt_msgs, tokenize=False,
                                                 add_generation_prompt=True)
                prompt_ids = tok(prompt, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"]
                if prompt_ids.shape[1] > MAX_PROMPT:
                    prompt_ids = prompt_ids[:, -MAX_PROMPT:]
                    truncated += 1
                mean_lp = score_candidates(model, prompt_ids, cand_ids, pad_id)
                probs = F.softmax(torch.tensor(mean_lp), dim=0).tolist()
                order = sorted(range(len(CLASSES)), key=lambda i: -probs[i])
                i1, i2 = order[:2]
                # санити-метрика без нормировки длины: sum = mean × n_токенов
                argmax_sum = CLASSES[max(range(len(CLASSES)),
                                         key=lambda i: mean_lp[i] * cand_lens[i])]
                gold = r["label"]
                rec = {
                    "gold": gold, "source": r.get("source", "?"),
                    "top1": CLASSES[i1], "p1": round(probs[i1], 4),
                    "top2": CLASSES[i2], "p2": round(probs[i2], 4),
                    "margin": round(probs[i1] - probs[i2], 4),
                    "ok1": CLASSES[i1] == gold,
                    "in_top2": gold in (CLASSES[i1], CLASSES[i2]),
                    "argmax_sum": argmax_sum,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                results.append(rec)

    # триггер-метрика: лог-шансы top-1 против top-2
    for r in results:
        r["lm"] = math.log(r["p1"] / max(r["p2"], 1e-9))

    n = len(results)
    acc1 = sum(r["ok1"] for r in results) / n
    acc1_sum = sum(r["argmax_sum"] == r["gold"] for r in results) / n
    rec2 = sum(r["in_top2"] for r in results) / n
    margins = sorted(r["lm"] for r in results)
    corr_m = sorted(r["lm"] for r in results if r["ok1"])
    wrong_m = sorted(r["lm"] for r in results if not r["ok1"])

    # порог → «консилиум» (top-1 + top-2 при марже ≤ порога) → покрытие
    # тонкая сетка в рабочем диапазоне 0.01–0.30, дальше грубая (там насыщение)
    grid = [0.01 * k for k in range(1, 31)] + [0.05 * k for k in range(7, 41)]
    table = []
    for t in grid:
        share = sum(r["lm"] <= t for r in results) / n
        cov = sum(r["in_top2"] if r["lm"] <= t else r["ok1"]
                  for r in results) / n
        table.append((t, share, cov))
    # порог под целевую долю консилиумов; если таких нет (мелкий прогон) —
    # максимум покрытия при минимальной доле
    fitting = [row for row in table if row[1] <= args.target_consult]
    best = max(fitting if fitting else table, key=lambda row: (row[2], -row[1]))

    by_class = collections.defaultdict(list)
    for r in results:
        by_class[r["gold"]].append(r)

    pairs = collections.Counter()
    pairs_saved = collections.Counter()
    for r in results:
        pairs[(r["top1"], r["top2"])] += 1
        if not r["ok1"] and r["in_top2"]:
            pairs_saved[(r["top1"], r["top2"])] += 1

    q = lambda p: margins[int(p * (n - 1))]  # noqa: E731
    lines = [
        "# Калибровка уверенности роутера (триггер консилиума)", "",
        f"Модель: `{Path(model_path).name}`; test: {n} примеров (обрезано слева "
        f"{truncated}); скоринг — средний logprob/токен 14 кандидатов, softmax → p(class).", "",
        f"**Top-1 (скоринг): {acc1 * 100:.1f}%** — санити против 76.6% жадной генерации "
        f"(ROUTER_BENCHMARK.md); argmax без нормировки длины: {acc1_sum * 100:.1f}%",
        f"**Recall@2: {rec2 * 100:.1f}%** (прирост к top-1: {100 * (rec2 - acc1):+.1f} п.п.)", "",
        "## Маржа ln(p1/p2)", "",
        f"- квантили 10/25/50/75/90%: {q(.10):.2f} / {q(.25):.2f} / {q(.50):.2f} / "
        f"{q(.75):.2f} / {q(.90):.2f}",
        f"- медиана у верных top-1: {median(corr_m):.2f} ({len(corr_m)} шт.), "
        f"у ошибочных: {median(wrong_m):.2f} ({len(wrong_m)} шт.) — чем ниже маржа у "
        "ошибок, тем лучше маржа отделяет сомнительные кейсы", "",
        "## Порог → консилиум → покрытие", "",
        "Консилиум = при марже ≤ порога вызвать top-1 и top-2; покрытие = золото "
        "среди вызванных.", "",
        "| Порог (ln p1/p2) | Доля консилиумов | Покрытие | Прирост к top-1 |",
        "|---|---|---|---|",
    ]
    for t, share, cov in table:
        mark = " **← рекомендация**" if t == best[0] else ""
        lines.append(f"| {t:.2f} | {share * 100:.0f}% | {cov * 100:.1f}% | "
                     f"{100 * (cov - acc1):+.1f} п.п.{mark} |")
    lines += [
        "", f"## Рекомендуемый порог: **{best[0]:.2f}** (ln p1/p2)", "",
        f"Доля консилиумов {best[1] * 100:.0f}%, покрытие {best[2] * 100:.1f}% "
        f"(+{100 * (best[2] - acc1):.1f} п.п. к top-1; потолок recall@2 = "
        f"{rec2 * 100:.1f}%). Целевая доля консилиумов была ≤ "
        f"{args.target_consult * 100:.0f}%. Для config.yaml: `consult_margin: {best[0]:.2f}`.", "",
        "## По классам", "",
        "| Класс | n | top-1 | recall@2 | медианная маржа |",
        "|---|---|---|---|---|",
    ]
    for c in CLASSES:
        rs = by_class.get(c, [])
        if not rs:
            continue
        a1 = sum(r["ok1"] for r in rs) / len(rs)
        r2 = sum(r["in_top2"] for r in rs) / len(rs)
        lines.append(f"| {c} | {len(rs)} | {a1 * 100:.0f}% | {r2 * 100:.0f}% | "
                     f"{median([r['lm'] for r in rs]):.2f} |")
    lines += ["", "## Частые пары top-1 + top-2 (приор статических правил)", "",
              "«Спасает» = золото было во втором месте, консилиум покрыл бы ошибку.", ""]
    for (t1, t2), cnt in pairs.most_common(15):
        if cnt < 5:
            continue
        lines.append(f"- {t1} + {t2}: {cnt} (спасает {pairs_saved.get((t1, t2), 0)})")

    report = project_path(cfg["data_processed"]) / "ROUTER_CONFIDENCE.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "threshold.json").write_text(json.dumps({
        "margin_ln": best[0], "consult_share": round(best[1], 4),
        "coverage": round(best[2], 4), "top1": round(acc1, 4),
        "recall2": round(rec2, 4), "n": n,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\ntop-1 {acc1 * 100:.1f}% | recall@2 {rec2 * 100:.1f}% | "
          f"порог {best[0]:.2f} ln-маржа (консилиумы {best[1] * 100:.0f}%, "
          f"покрытие {best[2] * 100:.1f}%)")
    print(f"[ok] отчёт: {report}\n[ok] порог: {out_dir / 'threshold.json'}")


if __name__ == "__main__":
    main()
