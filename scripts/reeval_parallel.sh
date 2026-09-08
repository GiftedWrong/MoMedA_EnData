#!/bin/bash
# Параллельная переоценка агентов: 3 воркера (по ~8 ГБ на модель = 23.7/24 ГБ),
# разнесённый старт против пиков загрузки, автоповтор упавших последовательно.
# Ждёт завершения eval Отоларинголога (PID 607425), чтобы не превысить VRAM.
cd /home/sgv/Desktop/Dev/AI_Dev/MoMedA_ChData || exit 1
PROG=train_all_progress.log
FAILED=/tmp/reeval_failed.txt
: > "$FAILED"

while kill -0 607425 2>/dev/null; do sleep 30; done
echo "--- RE-EVAL параллельный (3 воркера) старт $(date '+%F %T')" >> "$PROG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPECS=(Терапевт Хирург Гинеколог Педиатр Уролог Онколог Невролог Дерматолог Офтальмолог Проктолог Стоматолог Гастроэнтеролог Травматолог)

printf '%s\n' "${SPECS[@]}" | xargs -P 3 -d '\n' -I{} bash -c '
  spec="{}"
  sleep $((RANDOM % 25))
  cd /home/sgv/Desktop/Dev/AI_Dev/MoMedA_ChData || exit 1
  .venv/bin/python scripts/eval_specialist.py --specialty "$spec" --limit 100 \
      --usmle data/processed/usmle_pseudo_labeled.jsonl --usmle-limit 100 \
      > "eval_${spec}.log" 2>&1
  rc=$?
  echo "--- RE-EVAL[P] $spec rc=$rc $(date "+%F %T")" >> train_all_progress.log
  [ $rc -ne 0 ] && echo "$spec" >> /tmp/reeval_failed.txt
'

# упавшие (OOM на пике) — повтор последовательно на свободной GPU
if [ -s "$FAILED" ]; then
  echo "--- RE-EVAL повтор упавших ($(wc -l < $FAILED) шт.) $(date '+%F %T')" >> "$PROG"
  while read -r spec; do
    [ -z "$spec" ] && continue
    .venv/bin/python scripts/eval_specialist.py --specialty "$spec" --limit 100 \
        --usmle data/processed/usmle_pseudo_labeled.jsonl --usmle-limit 100 \
        > "eval_${spec}.log" 2>&1
    echo "--- RE-EVAL[R] $spec rc=$? $(date '+%F %T')" >> "$PROG"
  done < "$FAILED"
fi
echo "=== RE-EVAL ЗАВЕРШЁН $(date '+%F %T')" >> "$PROG"
