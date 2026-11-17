#!/bin/bash
# Bước 30, Brugada (pipeline lọc chung): lần huấn luyện r = 1..4 với SEED_OFFSET = 1000·r. Lần 0 = bước 27.
# Chạy: bash analysis_v30/run30_brugada.sh <h27> <thư mục làm việc> <thư mục log>
set -u
H=$1; W=$2; L=$3
cd "$(dirname "$0")/.."
for r in 1 2 3 4; do
  export SEED_OFFSET=$((1000*r))
  V5=$W/v5_r$r; V8=$W/v8_r$r; RS=$W/res_r$r; mkdir -p $V5 $V8 $RS
  if [ ! -f $RS/v8_bootstrap_pooled.csv ]; then
    python3 concepts_v5/compare_v5.py $H $V5 0 2 > $L/r${r}_v5_0.log 2>&1 &
    python3 concepts_v5/compare_v5.py $H $V5 1 2 > $L/r${r}_v5_1.log 2>&1 &
    wait
    python3 analysis_v8/v8_core.py $H $V5 $V8 0 2 > $L/r${r}_v8_0.log 2>&1 &
    python3 analysis_v8/v8_core.py $H $V5 $V8 1 2 > $L/r${r}_v8_1.log 2>&1 &
    wait
    python3 analysis_v8/v8_analyze.py $V8 $RS > $L/r${r}_an.log 2>&1
  fi
  echo "r$r xong $(date -u +%H:%M)" >> $L/run30.progress
done
echo DONE >> $L/run30.progress
