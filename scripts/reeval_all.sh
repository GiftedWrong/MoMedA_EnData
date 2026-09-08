#!/bin/bash
# Переоценка всех специалистов исправленным eval_specialist.py:
# MCQ-метрика принимает «Ответ:» и «Answer:», генерация 768 токенов,
# полные ответы сохраняются в runs/eval_answers_*.jsonl.
# Ждёт завершения переобучения Отоларинголога (PID 607425), затем 13 агентов.
cd /home/sgv/Desktop/Dev/AI_Dev/MoMedA_ChData || exit 1
PROG=train_all_progress.log

while kill -0 607425 2>/dev/null; do sleep 60; done
echo "--- RE-EVAL серия (фикс метрики MCQ) старт $(date '+%F %T')" >> "$PROG"

for spec in Терапевт Хирург Гинеколог Педиатр Уролог Онколог Невролог Дерматолог Офтальмолог Проктолог Стоматолог Гастроэнтеролог Травматолог; do
  .venv/bin/python scripts/eval_specialist.py --specialty "$spec" --limit 100 \
      --usmle data/processed/usmle_pseudo_labeled.jsonl --usmle-limit 100 \
      > "eval_${spec}.log" 2>&1
  echo "--- RE-EVAL $spec rc=$? $(date '+%F %T')" >> "$PROG"
done
echo "=== RE-EVAL ЗАВЕРШЁН $(date '+%F %T')" >> "$PROG"
