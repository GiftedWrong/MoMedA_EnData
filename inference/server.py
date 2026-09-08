#!/usr/bin/env python
"""Инференс-сервер: RU случай → RU ответ.

Перевод только на границах: вход RU→EN (MiLMMT), выход EN→RU (base Qwen).
Внутри всё на английском: роутер → агент → мастер (med-chief-3b, при его
отсутствии — проход базовой Qwen). Qwen-переводчик транзиентный: грузится
на шаг и сразу выгружается; --translator-ttl задаёт окно кэша по простою.

Запуск: .venv/bin/python -m inference.server --port 8010
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from specialties import SPECIALTIES, BY_NAME  # noqa: E402
from utils import load_config, project_path  # noqa: E402

CFG = load_config()
BASE_MODEL = str(project_path(CFG["base_model"]).resolve())
_models_dir = project_path(CFG["models_dir"])
# роутер: v2 (переобучен на расширенном DDX-маппинге) с фолбэком на v1
_ROUTER_V2 = _models_dir / "med-router-en-v2-3b"
ROUTER_MODEL = str(_ROUTER_V2 if _ROUTER_V2.is_dir()
                   else _models_dir / "med-router-en-3b")
CHIEF_MODEL = _models_dir / "med-chief-3b"
CHIEF_AVAILABLE = CHIEF_MODEL.is_dir()
NLLB_MODEL = Path("/home/sgv/Desktop/Dev/AI_Dev/models/nllb-200-distilled-600M")
MILMMT_MODEL = Path("/home/sgv/Desktop/Dev/AI_Dev/models/MiLMMT-46-1B")


class MilmmtTranslator:
    """MiLMMT-1B — вход RU→EN: лучшее качество из протестированного.
    Сырой промпт без чат-ролей (формат из карточки). Выход не доверяем:
    ломает структуру — там работает Qwen."""

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(MILMMT_MODEL))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(MILMMT_MODEL), dtype=torch.bfloat16).eval().to("cuda")

    def translate(self, text: str, direction: str = "ru2en") -> str | None:
        src, tgt = ("Russian", "English") if direction == "ru2en" else ("English", "Russian")
        prompt = f"Translate this from {src} to {tgt}:\n{src}: {text}\n{tgt}:"
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=768, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        full = self.tok.decode(out[0], skip_special_tokens=True)
        result = full.split(f"{tgt}:")[-1].strip()
        return result or None

CHIEF_SYSTEM = (
    "You are the senior physician of a multi-agent medical system. Several "
    "specialists have given their preliminary diagnoses for the case. Weigh "
    "their opinions against the patient's findings and formulate the final "
    "diagnosis. Answer in English only.\n"
    "Ты — старший врач мультиагентной системы. Взвесь предварительные диагнозы "
    "специалистов против данных случая и сформулируй итоговый диагноз. "
    "Отвечай только на английском языке."
)

TRANSLATE_OUT = (
    "Ты — точный медицинский переводчик. Переведи заключение старшего врача с "
    "английского на русский, сохраняя структуру (заголовки «### Рассуждение» и "
    "«### Итоговый диагноз» оставь как есть). Точно передавай диагноз, термины, "
    "дозировки и цифры. Пиши только по-русски, без иероглифов и латиницы. "
    "Выведи только перевод."
)

CLASSES = [s.name for s in SPECIALTIES]
ROUTE_RE = re.compile(r"Рекомендуемый\s+специалист\s*:\s*([^\n.]+)")

ROUTER_SYSTEM = (
    "You are a medical triage router. Read the patient case in English and select "
    "the single most appropriate specialist. Answer with exactly one line in the "
    "format «Рекомендуемый специалист: <специалист>.», choosing one of: "
    f"{', '.join(CLASSES)}.\n"
    "Ты — медицинский ассистент-маршрутизатор. Прочитай случай на английском и "
    "выбери одного специалиста. Отвечай одной строкой строго в формате "
    "«Рекомендуемый специалист: <специалист>.» из списка: " + f"{', '.join(CLASSES)}."
)

TRANSLATE_INTO_EN = (
    "You are a precise medical translator. Translate the patient's message from "
    "Russian to English. Preserve all medical terms, drug names, dosages, units "
    "and numbers exactly. Output only the translation."
)

# v1 слот мастер-агента: оформление итога + перевод EN→RU одной генерацией
OUTPUT_PASS = (
    "Ты — врач-куратор мультиагентной системы. На основе заключения "
    "специалиста-консультанта (на английском) сформируй итоговый ответ на русском "
    "языке строго по структуре:\n"
    "1) Краткое заключение (2–3 предложения)\n"
    "2) Предварительный диагноз\n"
    "3) Рекомендации (обследования, к кому обратиться, срочность)\n"
    "Точно сохраняй диагноз, термины, дозировки и цифры из заключения. "
    "Ничего не добавляй от себя. Пиши только по-русски: никаких иероглифов, "
    "латинских слов и англицизмов — все термины переводи на русский."
)

AGENT_SYSTEM = (
    "You are a doctor — {spec_en}. Study the patient's history and examination "
    "results and formulate a preliminary diagnosis. Answer in English only.\n"
    "Ты — врач-{spec_ru}. Изучи анамнез и результаты обследований и сформулируй "
    "предварительный диагноз. Отвечай только на английском языке."
)

# канонизация дрейфа маркеров в выдаче агента
_CANON = [
    (re.compile(r"###\s*Rationale\b", re.I), "### Рассуждение"),
    (re.compile(r"###\s*Reasoning\b", re.I), "### Рассуждение"),
    (re.compile(r"###\s*Preliminary\s+[Dd]iagnosis\b"), "### Предварительный диагноз"),
    (re.compile(r"###\s*Answer\b", re.I), "### Предварительный диагноз"),
    (re.compile(r"###\s*Диагноз\b"), "### Предварительный диагноз"),
]
# MCQ-артефакты: «Ответ: D. Текст» / «Answer: B) Текст» — буква долой, текст остаётся
_MCQ_PREFIX = re.compile(r"^(?:Ответ|Answer)\s*:\s*[A-F]\s*[).:]?\s*", re.I | re.M)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def canonicalize_agent_output(text: str) -> str:
    text = _CJK_RE.sub("", text)
    for rx, repl in _CANON:
        text = rx.sub(repl, text)
    text = _MCQ_PREFIX.sub("", text)
    return text.strip()


_RU_OUT_CLEAN = [
    (re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+"), " "),  # CJK и полноразрядная пунктуация
    (re.compile(r"\s{2,}"), " "),
]


def clean_russian_output(text: str) -> str:
    """Защитная зачистка русских выдач переводчика/мастера от CJK-утечек."""
    for rx, repl in _RU_OUT_CLEAN:
        text = rx.sub(repl, text)
    return text.strip()


def vram_used_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


# ---------------------------------------------------------------- модели

def _load_vanilla(path: str):
    """Грузим чистым transformers. unsloth здесь запрещён: патчит классы
    глобально, и повторные загрузки падают. Он живёт только в тренере."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    model.eval().to("cuda")
    return model, tok


class NllbTranslator:
    """NLLB-600M — запасной входной переводчик, если MiLMMT не завёлся.
    Выход тоже портит («Reasoning»→«Разумство») — только вход."""

    LANGS = {"ru2en": ("rus_Cyrl", "eng_Latn"), "en2ru": ("eng_Latn", "rus_Cyrl")}

    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tok = {}
        for src, _ in self.LANGS.values():
            self.tok[src] = AutoTokenizer.from_pretrained(str(NLLB_MODEL), src_lang=src)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(NLLB_MODEL), dtype=torch.float16).eval().to("cuda")

    def translate(self, text: str, direction: str = "ru2en") -> str | None:
        src, tgt = self.LANGS[direction]
        tok = self.tok[src]
        ids = tok(text, return_tensors="pt").to("cuda")
        bos = tok.convert_tokens_to_ids(tgt)
        with torch.inference_mode():
            out = self.model.generate(**ids, forced_bos_token_id=bos,
                                      max_new_tokens=768)
        return tok.batch_decode(out, skip_special_tokens=True)[0]


class TransientTranslator:
    """Базовая Qwen как переводчик: грузится на переводный шаг, освобождает VRAM.
    ttl=0 — строгая выгрузка после каждого использования; ttl>0 — окно кэша."""

    def __init__(self, ttl: int = 0):
        self.ttl = ttl
        self.model = None
        self.tokenizer = None
        self.last_used = 0.0
        self.loads = 0

    def ensure(self):
        if self.model is None:
            self.model, self.tokenizer = _load_vanilla(BASE_MODEL)
            self.loads += 1
        self.last_used = time.time()

    def release(self, force=False):
        if self.model is None:
            return
        if not force and self.ttl > 0 and time.time() - self.last_used < self.ttl:
            return
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()  # с чистым transformers безопасно

    def maybe_expire(self):
        if self.model is not None and self.ttl > 0 and \
                time.time() - self.last_used >= self.ttl:
            self.release(force=True)


class ModelHub:
    """Роутер резидентно; агент LRU=1; переводчик транзиентно.
    Всё грузится чистым transformers: import unsloth в этом процессе запрещён —
    он глобально патчит классы моделей, из-за чего перезагрузки и смешивание
    падают (illegal memory access / AttributeError apply_qkv)."""

    def __init__(self, translator_ttl: int = 0):
        self.gpu_lock = threading.Lock()
        self.router, self.router_tok = _load_vanilla(ROUTER_MODEL)
        # входная граница: MiLMMT (лучшее качество) → NLLB (резерв) → Qwen (транзиентно)
        self.input_mt = None
        for cls, path, name in [(MilmmtTranslator, MILMMT_MODEL, "MiLMMT"),
                                (NllbTranslator, NLLB_MODEL, "NLLB")]:
            if path.is_dir():
                try:
                    self.input_mt = cls()
                    self.input_mt_name = name
                    break
                except Exception as e:
                    print(f"[warn] {name} недоступен ({e})")
        self.agent_slug = None
        self.agent = None
        self.agent_tok = None
        self.agent_loads = 0
        self.chief = None
        self.chief_tok = None
        self.translator = TransientTranslator(translator_ttl)
        self.peak_vram = vram_used_mb()

    def agent_for(self, slug: str):
        if self.agent_slug != slug:
            self.release_agent()
            path = str(project_path(CFG["models_dir"]) / f"med-spec-{slug}-3b")
            self.agent, self.agent_tok = _load_vanilla(path)
            self.agent_slug = slug
            self.agent_loads += 1
        self.peak_vram = max(self.peak_vram, vram_used_mb())
        return self.agent, self.agent_tok

    def release_agent(self):
        """Выгрузка агента (LRU): перед выходной границей — иначе роутер+chief+
        агент+переводчик = 24.8 ГБ не влезают."""
        if self.agent is None:
            return
        self.agent = None
        self.agent_tok = None
        self.agent_slug = None
        gc.collect()
        torch.cuda.empty_cache()  # с чистым transformers безопасно

    def ensure_chief(self):
        """Мастер-агент: ленивая загрузка, далее резидентно (нужен каждому случаю)."""
        if self.chief is None and CHIEF_AVAILABLE:
            self.chief, self.chief_tok = _load_vanilla(str(CHIEF_MODEL))
        return self.chief is not None

    def note_vram(self):
        self.peak_vram = max(self.peak_vram, vram_used_mb())


HUB: ModelHub | None = None


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict], max_new: int = 768) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def extract_specialty(text: str) -> str | None:
    m = ROUTE_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1).strip().lower().replace("ё", "е")
    for c in CLASSES:
        if c.lower().replace("ё", "е") in raw:
            return c
    return None


# ---------------------------------------------------------------- приложение

class CaseIn(BaseModel):
    text: str


class TranslateIn(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global HUB
    HUB = ModelHub(translator_ttl=app.state.ttl)
    if app.state.ttl > 0:
        def janitor():
            while True:
                time.sleep(5)
                with HUB.gpu_lock:
                    HUB.translator.maybe_expire()
        threading.Thread(target=janitor, daemon=True).start()
    yield


app = FastAPI(title="MoMedA inference", lifespan=lifespan)
# TTL переводчика: env (для запуска через `uvicorn inference.server:app`) или --translator-ttl
app.state.ttl = int(os.environ.get("MOMEDA_TRANSLATOR_TTL", "0"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "router": os.path.basename(ROUTER_MODEL),
        "input_mt": getattr(HUB, "input_mt_name", "Qwen(fallback)"),
        "chief": ("loaded" if HUB.chief is not None
                  else "available" if CHIEF_AVAILABLE else "not_trained"),
        "agent_current": HUB.agent_slug,
        "agent_loads": HUB.agent_loads,
        "translator": "loaded" if HUB.translator.model is not None else "released",
        "translator_loads": HUB.translator.loads,
        "translator_ttl_s": HUB.translator.ttl,
        "vram_used_mb": vram_used_mb(),
        "vram_peak_mb": HUB.peak_vram,
    }


@app.post("/api/translate")
def translate(req: TranslateIn, direction: str = "ru2en"):
    system = TRANSLATE_INTO_EN if direction == "ru2en" else (
        "Ты — точный медицинский переводчик. Переведи текст с английского на "
        "русский. Точно сохраняй диагнозы, термины, дозировки и цифры. "
        "Пиши только по-русски, без иероглифов и латиницы. "
        "Выведи только перевод.")
    with HUB.gpu_lock:
        HUB.translator.ensure()
        try:
            result = generate(HUB.translator.model, HUB.translator.tokenizer,
                              [{"role": "system", "content": system},
                               {"role": "user", "content": req.text}])
        finally:
            HUB.translator.release()
            HUB.note_vram()
    result = clean_russian_output(result) if direction == "en2ru" else result
    return {"direction": direction, "text": result}


@app.post("/api/route")
def route(req: CaseIn):
    with HUB.gpu_lock:
        answer = generate(HUB.router, HUB.router_tok,
                          [{"role": "system", "content": ROUTER_SYSTEM},
                           {"role": "user", "content": req.text}], max_new=24)
        HUB.note_vram()
    return {"raw": answer, "specialty": extract_specialty(answer)}


@app.post("/api/agent")
def agent(req: CaseIn, specialty: str):
    spec = BY_NAME.get(specialty)
    if spec is None:
        raise HTTPException(400, f"неизвестная специальность: {specialty}")
    system = AGENT_SYSTEM.format(spec_en=" / ".join(spec.names_en),
                                 spec_ru=spec.name.lower())
    with HUB.gpu_lock:
        model, tok = HUB.agent_for(spec.slug)
        answer = generate(model, tok,
                          [{"role": "system", "content": system},
                           {"role": "user", "content": req.text}])
        HUB.note_vram()
    return {"specialty": spec.name, "output_en": canonicalize_agent_output(answer)}


@app.post("/api/case")
def case(req: CaseIn):
    trace: list[dict] = []

    def step(name, fn):
        t0 = time.time()
        result = fn()
        trace.append({"step": name, "ms": int((time.time() - t0) * 1000),
                      "vram_after_mb": vram_used_mb()})
        return result

    # 1. Вход: RU→EN резидентным MT; не завёлся — Qwen с выгрузкой после.
    def _translate_in():
        if HUB.input_mt is not None:
            try:
                return HUB.input_mt.translate(req.text, "ru2en")
            except Exception:
                pass
        with HUB.gpu_lock:
            HUB.translator.ensure()
            try:
                return generate(HUB.translator.model, HUB.translator.tokenizer,
                                [{"role": "system", "content": TRANSLATE_INTO_EN},
                                 {"role": "user", "content": req.text}])
            finally:
                HUB.translator.release(force=True)
    case_en = step("translate_ru2en", _translate_in)

    # 2. Роутер (резидент, EN)
    def _route():
        with HUB.gpu_lock:
            raw = generate(HUB.router, HUB.router_tok,
                           [{"role": "system", "content": ROUTER_SYSTEM},
                            {"role": "user", "content": case_en}], max_new=24)
            HUB.note_vram()
            return raw
    route_raw = step("route", _route)
    specialty = extract_specialty(route_raw)
    if specialty is None:
        raise HTTPException(422, {"error": "роутер не выдал специальность",
                                  "router_raw": route_raw, "trace": trace})
    spec = BY_NAME[specialty]

    # 3. Агент-специалист (lazy LRU=1, EN)
    def _agent():
        model, tok = HUB.agent_for(spec.slug)
        system = AGENT_SYSTEM.format(spec_en=" / ".join(spec.names_en),
                                     spec_ru=spec.name.lower())
        return generate(model, tok,
                        [{"role": "system", "content": system},
                         {"role": "user", "content": case_en}])
    preliminary_en = canonicalize_agent_output(step("agent", _agent))

    # 4. Слот мастер-агента: v2 — med-chief-3b (обученный синтез); v1 — проход
    # базовой Qwen «оформление+перевод» (пока chief не обучен).
    # Агента выгружаем ДО загрузки мастера — иначе не влезаем по VRAM.
    with HUB.gpu_lock:
        HUB.release_agent()
    if HUB.ensure_chief():

        def _chief():
            diag_m = re.search(r"###\s*Предварительный\s*диагноз\s*\n(.+)",
                               preliminary_en, re.S)
            diag_line = (diag_m.group(1).strip().split("\n")[0][:200]
                         if diag_m else preliminary_en[:200])
            reasoning_m = re.search(r"###\s*Рассуждение\s*\n(.+?)(?=\n###|\Z)",
                                    preliminary_en, re.S)
            notes = (reasoning_m.group(1).strip()[:600] + "…") if reasoning_m else ""
            user = (f"Case findings:\n{case_en}\n\n"
                    f"Specialists' preliminary diagnoses:\n"
                    f"- {diag_line} (preliminary diagnosis of a {specialty})\n"
                    + (f"Specialist's supporting notes: {notes}\n" if notes else "")
                    + "\nFormulate the final diagnosis.")
            with HUB.gpu_lock:
                return generate(HUB.chief, HUB.chief_tok,
                                [{"role": "system", "content": CHIEF_SYSTEM},
                                 {"role": "user", "content": user}])
        final_en = canonicalize_agent_output(step("chief", _chief))

        def _translate_out():
            with HUB.gpu_lock:
                HUB.translator.ensure()
                try:
                    return generate(HUB.translator.model, HUB.translator.tokenizer,
                                    [{"role": "system", "content": TRANSLATE_OUT},
                                     {"role": "user", "content": final_en}],
                                    max_new=1024)
                finally:
                    HUB.translator.release(force=True)
        answer_ru = clean_russian_output(step("translate_en2ru", _translate_out))
    else:
        def _output():
            with HUB.gpu_lock:
                HUB.translator.ensure()
                try:
                    return generate(HUB.translator.model, HUB.translator.tokenizer,
                                    [{"role": "system", "content": OUTPUT_PASS},
                                     {"role": "user", "content":
                                      f"Специальность: {spec.name}\n"
                                      f"Заключение специалиста (EN):\n{preliminary_en}"}],
                                    max_new=1024)
                finally:
                    HUB.translator.release()
        answer_ru = clean_russian_output(step("master_v1_translate_en2ru", _output))
    HUB.note_vram()

    return {
        "specialty": specialty,
        "case_en": case_en,
        "preliminary_en": preliminary_en,
        "final_en": final_en if HUB.chief is not None else None,
        "master": "chief" if HUB.chief is not None else "base_v1",
        "answer_ru": answer_ru,
        "latency_ms": sum(t["ms"] for t in trace),
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--translator-ttl", type=int, default=app.state.ttl,
                    help="сек кэширования переводчика (0 — выгрузка сразу; "
                         "env MOMEDA_TRANSLATOR_TTL)")
    args = ap.parse_args()
    import uvicorn
    app.state.ttl = args.translator_ttl
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
