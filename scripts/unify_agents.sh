#!/bin/bash
# Унификация агентов: перегенерация наборов с --uniform-format (консультации
# получают блок «### Предварительный диагноз») и переобучение 7 hcm-тяжёлых
# специалистов. v1-модели бэкапятся в models/backup_v1/.
cd /home/sgv/Desktop/Dev/AI_Dev/MoMedA_EnData || exit 1
PROG=train_uniform_progress.log
echo "=== УНИФИКАЦИЯ старт $(date '+%F %T')" >> "$PROG"

echo "--- регенерация данных (--uniform-format) $(date '+%T')" >> "$PROG"
.venv/bin/python scripts/prepare_specialties.py --per-specialty 100000 \
    --hcm-limit 20000 --uniform-format >> "$PROG" 2>&1
echo "--- данные rc=$? $(date '+%T')" >> "$PROG"

mkdir -p models/backup_v1
SPECS=(Уролог Онколог Невролог Дерматолог Проктолог Гастроэнтеролог Травматолог)
for spec in "${SPECS[@]}"; do
  slug=$(.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); from specialties import BY_NAME; print(BY_NAME['$spec'].slug)")
  if [ -d "models/med-spec-${slug}-3b" ] && [ ! -d "models/backup_v1/med-spec-${slug}-3b" ]; then
    mv "models/med-spec-${slug}-3b" "models/backup_v1/med-spec-${slug}-3b"
    echo "--- бэкап v1: $slug" >> "$PROG"
  fi
  echo "--- ОБУЧЕНИЕ $spec старт $(date '+%T')" >> "$PROG"
  .venv/bin/python scripts/train_model.py --specialty "$spec" > "train_${spec}.log" 2>&1
  echo "--- ОБУЧЕНИЕ $spec rc=$? финиш $(date '+%T')" >> "$PROG"
done
echo "=== УНИФИКАЦИЯ ЗАВЕРШЕНА $(date '+%F %T')" >> "$PROG"
