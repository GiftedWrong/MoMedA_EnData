#!/bin/bash
# Последовательное обучение всех 14 специалистов на максимальном объёме
# и бенчмарк каждого сразу после обучения. Логи: train_<Спец>.log /
# eval_<Спец>.log; прогресс очереди: train_all_progress.log
cd /home/sgv/Desktop/Dev/AI_Dev/MoMedA_EnData || exit 1
PROG=train_all_progress.log
echo "=== СТАРТ ОЧЕРЕДИ $(date '+%F %T')" >> "$PROG"

# порядок: от малого к большому — быстрый контроль конвейера на Проктологе
SPECS=(Проктолог Отоларинголог Травматолог Невролог Дерматолог Офтальмолог Педиатр Уролог Онколог Гинеколог Стоматолог Хирург Гастроэнтеролог Терапевт)

for spec in "${SPECS[@]}"; do
  echo "--- ОБУЧЕНИЕ $spec старт $(date '+%F %T')" >> "$PROG"
  .venv/bin/python scripts/train_model.py --specialty "$spec" > "train_${spec}.log" 2>&1
  rc=$?
  echo "--- ОБУЧЕНИЕ $spec rc=$rc финиш $(date '+%F %T')" >> "$PROG"
  if [ $rc -ne 0 ]; then
    echo "!!! $spec обучение упало (rc=$rc), см. train_${spec}.log — пропускаю eval" >> "$PROG"
    continue
  fi
  echo "--- EVAL $spec старт $(date '+%F %T')" >> "$PROG"
  .venv/bin/python scripts/eval_specialist.py --specialty "$spec" --limit 100 \
      --usmle data/processed/usmle_pseudo_labeled.jsonl --usmle-limit 100 \
      > "eval_${spec}.log" 2>&1
  echo "--- EVAL $spec rc=$? финиш $(date '+%F %T')" >> "$PROG"
done
echo "=== ОЧЕРЕДЬ ЗАВЕРШЕНА $(date '+%F %T')" >> "$PROG"
