# MoMedA_ChData

Мультиагентная медицинская система: маршрутизатор выбирает специалиста,
14 агентов-специалистов выдают предварительные диагнозы, мастер-агент
формулирует итоговый диагноз. Все модели — локальные файнтюны Qwen2.5-3B,
перевод на границах — локальный. Обучение и инференс — одна RTX 3090 (24 ГБ).

## Архитектура

```
RU случай → [MiLMMT-1B: RU→EN] → роутер (med-router-en-v2-3b, 14 классов)
→ агент med-spec-<slug>-3b (LRU=1): ### Рассуждение + ### Предварительный диагноз (EN)
→ мастер med-chief-3b (резидент): ### Рассуждение + ### Итоговый диагноз (EN)
→ [base Qwen: EN→RU] → RU-ответ
```

VRAM: роутер 6.2 + chief 6.2 + MiLMMT 2.0 резидентно; агент 6.2 (LRU, выгружается
до выхода); Qwen-переводчик 6.2 (только выход). Пик 19.8 ГБ.

## Этапы и результаты

### 1. Данные (06–07.09)
- Источники: MedMCQA (Apache-2.0), MTSamples (CC0), MedQuAD (NIH), MedQA-USMLE
  (CC-BY), HealthCareMagic-100k (без лицензии — исследование), DDXPlus (CC-BY).
- Наборы: 14 специальностей 113.7k примеров (`data/specialties/`), роутер 17.6k
  (`data/router/`), мастер 24k = DDXPlus 13.5k + MedMCQA 8.1k + USMLE 2.4k
  (`data/chief/`), USMLE-псевдо 2k для eval, RU-жалобы 175 (`e2e_ru_complaints.jsonl`).
- Покрытие: 12/14 классов из MedMCQA напрямую, Уролог/Онколог/Невролог/Гастро/
  Дерматолог/Проктолог/Травматолог — HCM-псевдо (валидировано роутером: 93%).

### 2. Обучение (07.09)
| Модель | Данные | Результат |
|---|---|---|
| med-router-en-3b (v1) | 17k, старый DDX-маппинг | 72.7% тест; на новых классах DDX 0–4% — переподготовлен |
| med-router-en-v2-3b | 17.6k, расширенный DDX-маппинг (7 классов) | **76.6%**, DDX-жалобы 96%, HCM 96% |
| med-spec-<14> | 113.7k суммарно, макс. объём, 1 эпоха | MCQ на своих виньетках 50–85%, USMLE 44–79%, EN ~100% |
| med-spec-<7> v2 (унификация) | те же + блок диагноза у 44% консультаций | единый формат «### Рассуждение + ### Предварительный диагноз» |
| med-chief-3b | 24k синтез «мнения → итоговый диагноз» | train 0.556 / eval 0.355; шаг 3–6 с в конвейере |

Отчёты: `runs/FINAL_METRICS.md`, `runs/FINAL_REPORT.md`, `data/processed/ROUTER_BENCHMARK.md`.

### 3. Инференс (07–08.09)
- Сервер `inference/server.py` (FastAPI, порт 8010): `/api/case`, `/api/route`,
  `/api/agent`, `/api/translate`, `/api/health`. Всё локально, без внешних сервисов.
- Выбор входного переводчика — сравнительный тест трёх моделей на жалобах:
  **MiLMMT-46-1B > NLLB-600M > opus-mt** (только MiLMMT верно переводит
  «стул»→stool, «назначили»→prescribed; термины/дозировки без искажений).
  Выход EN→RU — только base Qwen: все малые модели ломают структуру ответа
  («Reasoning»→«Выводы», «groin»→«локоть»). Фолбэк-цепочка входа:
  MiLMMT → NLLB → Qwen.
- E2e-бенчмарк (175 RU-жалоб DDXPlus, золото по расширенному маппингу):

| Метрика | v1 (07.09) | v2 (08.09) |
|---|---|---|
| Маршрутизация | 30.9% | **79.4%** |
| по классам | — | Невролог 100, Онколог 96, Травматолог 96, ЛОР 84, Терапевт 80, Хирург 52, Гастро 48 |
| Латентность (медиана) | 37 с | 30 с (финальная конфигурация с MiLMMT: ~21 с) |
| Вход RU→EN | Qwen 7 с | MiLMMT ~1 с |
| CJK-утечки | 0 | 0 |

Причина провала v1 и фикс: роутер v1 обучался до расширения DDX-маппинга
(все DDX-жалобы — Терапевт/ЛОР); переобучение на новом маппинге дало +48.5 п.п.

### 4. Инженерные решения
- Сервер — чистый transformers: import unsloth глобально патчит классы моделей,
  повторные загрузки при 3+ экземплярах падают (illegal memory access /
  apply_qkv); unsloth только в train_model.py.
- Переводчик и агент не живут в VRAM одновременно (роутер+chief+агент+
  переводчик = 24.8 ГБ > 23.5) — принудительные выгрузки в конвейере.
- Агент выгружается до загрузки chief; empty_cache — безопасен на transformers.
- v1-модели 7 переобученных агентов — в `models/backup_v1/`.

## Запуск

```bash
.venv/bin/python -m inference.server --port 8010        # сервер (переводчик транзиентен)
MOMEDA_TRANSLATOR_TTL=600 .venv/bin/uvicorn inference.server:app --port 8010

curl -s localhost:8010/api/case -H 'Content-Type: application/json' \
     -d '{"text": "Неделю болит горло, температура 38,5, налёт на миндалинах"}'
curl -s localhost:8010/api/health

# воспроизведение данных
.venv/bin/python scripts/download_en.py --all
.venv/bin/python scripts/prepare_specialties.py --per-specialty 100000 --hcm-limit 20000 --uniform-format
.venv/bin/python scripts/prepare_router.py
.venv/bin/python scripts/prepare_chief.py
# обучение
.venv/bin/python scripts/train_model.py --data data/router        # или --specialty Терапевт / --data data/chief
# оценка
.venv/bin/python scripts/benchmark_router_en.py --model models/med-router-en-v2-3b
.venv/bin/python scripts/eval_specialist.py --specialty Терапевт --limit 100 \
    --usmle data/processed/usmle_pseudo_labeled.jsonl
.venv/bin/python scripts/benchmark_e2e.py --n 200                 # RU-жалобы кэшированы
```

## Файлы

```
models/          med-router-en-{v2,}-3b · med-chief-3b · med-spec-<14> · backup_v1/<7>
data/specialties/  наборы 14 специалистов (train/val/test)
data/router/ data/chief/    наборы роутера и мастера
data/raw/en/     исходники 6 датасетов + SCHEMAS.md + SOURCES.md
data/processed/  отчёты подготовки, usmle_pseudo_labeled, e2e_ru_complaints
runs/            FINAL_REPORT.md · FINAL_METRICS.md · E2E_REPORT.md (+_v1) · ROUTER_BENCHMARK.md
scripts/         download_en · prepare_* · train_model · eval_specialist · benchmark_*
inference/       server.py
```

## Ограничения
Маршрутизация Гастро 48% / Хирург 52% (взаимная путаница классов); выходной
перевод только Qwen; Проктолог — тонкий класс (1.1k примеров).
