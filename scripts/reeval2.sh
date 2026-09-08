#!/bin/bash
# Чистый эксперимент «2 воркера»: переоценка оставшихся 10 агентов параллелью двойкой.
# Запускается наблюдателем после завершения текущей тройки. rc-строки: RE-EVAL[2].
cd /home/sgv/Desktop/Dev/AI_Dev/MoMedA_ChData || exit 1
PROG=train_all_progress.log
FAILED=/tmp/reeval2_failed.txt
: > "$FAILED"

echo "--- RE-EVAL[2 воркера] старт $(date '+%F %T')" >> "$PROG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPECS=(Педиатр Уролог Онколог Невролог Дерматолог Офтальмолог Проктолог Стоматолог Гастроэнтеролог Травматолог)

printf '%s\n' "${SPECS[@]}" | xargs -P 2 -d '\n' -I{} bash -c '
  spec="{}"
  sleep $((RANDOM % 10))
  cd /home/sgv/Desktop/Dev/AI_Dev/MoMedA_ChData || exit 1
  .venv/bin/python scripts/eval_specialist.py --specialty "$spec" --limit 100 \
      --usmle data/processed/usmle_pseudo_labeled.jsonl --usmle-limit 100 \
      > "eval_${spec}.log" 2>&1
  rc=$?
  echo "--- RE-EVAL[2] $spec rc=$rc $(date "+%F %T")" >> train_all_progress.log
  [ $rc -ne 0 ] && echo "$spec" >> /tmp/reeval2_failed.txt
'

if [ -s "$FAILED" ]; then
  echo "--- RE-EVAL[2] повтор упавших ($(wc -l < $FAILED) шт.) $(date '+%F %T')" >> "$PROG"
  while read -r spec; do
    [ -z "$spec" ] && continue
    .venv/bin/python scripts/eval_specialist.py --specialty "$spec" --limit 100 \
        --usmle data/processed/usmle_pseudo_labeled.jsonl --usmle-limit 100 \
        > "eval_${spec}.log" 2>&1
    echo "--- RE-EVAL[2R] $spec rc=$? $(date '+%F %T')" >> "$PROG"
  done < "$FAILED"
fi
echo "=== RE-EVAL2 ЗАВЕРШЁН $(date '+%F %T')" >> "$PROG"
