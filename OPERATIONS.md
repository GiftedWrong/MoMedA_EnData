# OPERATIONS — управление системой MoMedA_EnData


---

## 1. Инференс-сервер

### Запуск

```bash
.venv/bin/python -m inference.server --port 8010
```

в фоне, с логом:

```bash
nohup .venv/bin/python -m inference.server --port 8010 > server.log 2>&1 &
```

с окном кэша выходного переводчика (экономит ~10 с на случае при серии
запросов; 0 = выгружать сразу после каждого использования):

```bash
MOMEDA_TRANSLATOR_TTL=600 nohup .venv/bin/python -m inference.server --port 8010 > server.log 2>&1 &
# или флагом:
.venv/bin/python -m inference.server --port 8010 --translator-ttl 600
```

Старт занимает ~60 с (роутер + MiLMMT грузятся при старте, мастер — при
первом случае). Готовность: `curl -s localhost:8010/api/health`.

### Остановка

```bash
kill $(pgrep -f "python -m inference.server")
kill -9 <PID>                                        # если не помогло
pgrep -f "inference.server" || echo "остановлен"     # проверка
```

### Проверка состояния

```bash
curl -s localhost:8010/api/health | python3 -m json.tool
```

| Поле | Смысл |
|---|---|
| `router` | какой роутер загружен (v2 приоритетнее, v1 — фолбэк) |
| `consult_margin` | порог консилиума ln(p1/p2); 0 — выключено. Env `MOMEDA_CONSULT_MARGIN` перекрывает config.yaml |
| `input_mt` | входной переводчик: MiLMMT (норма) / NLLB / Qwen(fallback) |
| `chief` | loaded / available / not_trained |
| `agent_current`, `agent_loads` | текущий LRU-агент и сколько раз грузились агенты |
| `translator`, `translator_loads` | выходной Qwen-переводчик: в памяти или выгружен |
| `vram_used_mb`, `vram_peak_mb` | текущая и пиковая VRAM процесса |

Норма в покое: ~14–15 ГБ (роутер+мастер+MiLMMT), пик за случай ≤ 20 ГБ.

### Эндпоинты

**POST /api/case — полный конвейер** (основной):

```bash
curl -s -m 300 localhost:8010/api/case -H 'Content-Type: application/json' \
     -d '{"text": "Неделю болит горло, температура 38,5, налёт на миндалинах"}'
```

Ответ: `specialty` (top-1), `specialties` (при малой марже роутера —
top-1 + top-2, тогда же `consult: true`), `case_en`, `preliminary_en`
(агент top-1), `final_en` (мастер, взвешивает все мнения),
`answer_ru`, `master` («chief»/«base_v1»), `latency_ms`, `trace` — тайминги и
VRAM каждого шага. Латентность 15–40 с (первый случай новой специальности
дороже — загрузка агента ~30 с; консилиум добавляет ~5–30 с — второй агент).
Ошибка 422 — роутер не выдал класс (текст
слишком короткий/не медицинский); при включённом консилиуме скоринг-топ-1
страхует от 422.

**POST /api/route — только маршрутизация** (EN-текст на вход):

```bash
curl -s localhost:8010/api/route -H 'Content-Type: application/json' \
     -d '{"text": "Patient: 45 y.o. male. Sore throat, fever 38.5, tonsillar exudate."}'
```

**POST /api/agent?specialty=<Имя> — прямой вызов агента** (EN-текст):

```bash
curl -s 'localhost:8010/api/agent?specialty=Невролог' -H 'Content-Type: application/json' \
     -d '{"text": "Patient: 60 y.o. female. Progressive weakness in both legs..."}'
```

Имена: Терапевт, Хирург, Гинеколог, Педиатр, Уролог, Онколог, Невролог,
Дерматолог, Офтальмолог, Отоларинголог, Стоматолог, Проктолог,
Гастроэнтеролог, Травматолог.

**POST /api/translate?direction=ru2en|en2ru — изолированный перевод**
(Qwen, полный цикл загрузки/выгрузки, ~20–30 с):

```bash
curl -s 'localhost:8010/api/translate?direction=ru2en' -H 'Content-Type: application/json' \
     -d '{"text": "Принимаю омепразол 20 мг два раза в день"}'
```

---

## 2. Мониторинг


- `tail -f server.log` — access-лог и трейсбеки сервера.

- Счётчик обработанных случаев: `grep -c "POST /api/case" server.log`.

---

## 3. Обучение

Единый тренер — `scripts/train_model.py` (unsloth, полный файнтюн bf16,
batch 1×accum 16, 1–2 эпохи, чекпоинты каждые 250 шагов, `--resume`).

```bash
# роутер (после prepare_router.py)
.venv/bin/python scripts/train_model.py --data data/router --run-name med-router-en-v2-3b

# один агент (после prepare_specialties.py)
.venv/bin/python scripts/train_model.py --specialty Терапевт

# мастер (после prepare_chief.py)
.venv/bin/python scripts/train_model.py --data data/chief

# все 14 агентов последовательно, с оценкой каждого
bash scripts/train_all.sh

# унификация 7 HCM-агентов (регенерация данных + бэкап v1 + переобучение)
bash scripts/unify_agents.sh
```

Полезные флаги: `--epochs`, `--batch`, `--accum`, `--max-len`, `--resume`,
`--engine unsloth|transformers`. Модели пишутся в `models/<run-name>/`,
логи — `runs/<run-name>/`, время — от 6 минут (Проктолог) до ~2.5 ч (Терапевт).

⚠️ Перед обучением остановить сервер (см. раздел 1). После падения по OOM —
перезапустите с теми же аргументами и `--resume`.

Откат агента на v1: `mv models/med-spec-<slug>-3b models/med-spec-<slug>-3b-v2 && mv models/backup_v1/med-spec-<slug>-3b models/`.

---

## 4. Данные

Все скрипты идемпотентны и детерминированы (seed 42); перегенерация даёт
те же наборы. Требуется `data/raw/en/` (если пусто — `scripts/download_en.py --all`).

```bash
.venv/bin/python scripts/prepare_specialties.py --per-specialty 100000 \
    --hcm-limit 20000 --uniform-format      # наборы 14 агентов (макс. объём)
.venv/bin/python scripts/prepare_chief.py --n 15000 --n-mcqa 7000 --n-usmle 2000
.venv/bin/python scripts/prepare_router.py  # 17.6k, расширенный DDX-маппинг
```

Маппинги классов — `scripts/specialties.py`; изменение маппинга требует
перегенерации затронутых наборов и переобучения.

---

## 5. Оценка

```bash
# роутер: точность по классам × источникам, путаницы
.venv/bin/python scripts/benchmark_router_en.py --model models/med-router-en-v2-3b

# калибровка уверенности роутера: recall@2, порог консилиума (~10 мин GPU)
.venv/bin/python scripts/router_confidence.py
.venv/bin/python scripts/router_confidence.py --aggregate-only   # пересчёт отчёта без GPU

# агент: формат, EN-чистота, MCQ; USMLE — внешний тест
.venv/bin/python scripts/eval_specialist.py --specialty Терапевт --limit 100 \
    --usmle data/processed/usmle_pseudo_labeled.jsonl --usmle-limit 100

# пересчёт метрик всех агентов по сохранённым ответам (без GPU)
.venv/bin/python scripts/recompute_eval_metrics.py

# сквозной прогон конвейера (сервер должен работать; RU-жалобы кэшированы)
.venv/bin/python scripts/benchmark_e2e.py --n 200
```

Отчёты: `runs/FINAL_METRICS.md`, `runs/E2E_REPORT.md` (+`_v1`,
+`E2E_CONSULT_REPORT.md`), `data/processed/ROUTER_BENCHMARK.md`,
`data/processed/ROUTER_CONFIDENCE.md`, `runs/CONSULT_REPORT.md`,
`runs/eval_med-spec-*.md`.

---

## 6. Неисправности

| Симптом | Причина | Действие |
|---|---|---|
| `health` не отвечает | сервер грузится (~60 с) или упал | `tail server.log`; при трейсбеке — перезапустить |
| Порт 8010 занят | предыдущий экземпляр жив | `pgrep -f inference.server` → kill по PID |
| OOM при старте сервера | работает обучение | остановить обучение, перезапустить сервер |
| `illegal memory access` в server.log | в процессе есть unsloth (не должно быть) или битый чекпоинт | перезапустить сервер; проверить, что inference/ не импортирует unsloth |
| 422 на /api/case | роутер не выдал специальность | проверить текст запроса; посмотреть `router_raw` в теле ошибки |
| Обучение OOM на длинном батче | batch > 1 при max_len 2048 | `--batch 1 --accum 16` |
| Обучение падает сразу | GPU занята сервером/другим процессом | `nvidia-smi` → остановить конкурента |
| Русский ответ с иероглифами | сбой анти-CJK чистки | перезапустить; сохранить пример в багу |

---

## 7. Каталоги и вес

```
models/          ~87 ГБ (в git не входит): роутеры, мастер, 14 агентов, backup_v1
data/raw/en/     ~1.1 ГБ источников (не в git)
data/specialties|router|chief/   регенерируемые наборы (не в git)
runs/            отчёты и сырые ответы оценок (в git)
```
