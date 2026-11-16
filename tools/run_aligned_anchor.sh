#!/bin/bash
# 对齐后的锚定复跑：A1(dot/SGD) A2(dot/Adagrad) A4(DCN→本仓仅 dot/Adagrad)，各 REP 轮 300 步
# 用法: tools/run_aligned_anchor.sh <运行目录(已 sync)> <DATA_DIR> [repeats]
# 环境: STEPS(默认 300) WARMUP(默认 20) NPROC(默认 8) EXTRA(追加参数)
set -u
DIR=${1:?运行目录}; DATA=${2:?DATA_DIR}; REP=${3:-3}
STEPS=${STEPS:-300}; WARMUP=${WARMUP:-20}; NPROC=${NPROC:-8}; EXTRA=${EXTRA:-}
OUT="$DIR/anchor_aligned_$(date +%m%d_%H%M).txt"
: > "$OUT"
cd "$DIR" || exit 2
echo "STEPS=$STEPS WARMUP=$WARMUP REP=$REP DATA=$DATA EXTRA=$EXTRA start=$(date +%H:%M:%S)" >> "$OUT"

run_one() {  # $1=tag $2=global_bs $3=dense_optim
  local tag=$1 bs=$2 opt=$3 log="$DIR/anchor_aligned_${tag}.log"
  echo "=== $tag start $(date +%H:%M:%S) bs=$bs dense_optim=$opt ===" >> "$OUT"
  torchrun --nproc_per_node="$NPROC" --master_port=$((32000 + RANDOM % 500)) dlrm_v2_train.py \
    --batch-size "$bs" --epochs 1 --eval-every 0 --warmup "$WARMUP" --steps "$STEPS" \
    --data-dir "$DATA" --train-days 0,1,2,3,4,5 --test-days 6 --dense-optim "$opt" $EXTRA \
    > "$log" 2>&1
  local rc=$?
  echo "$tag rc=$rc" >> "$OUT"
  grep -aE "inter_dim|top_dims|dense_optim|median|step_ms|auc" "$log" | tail -6 >> "$OUT"
}

for i in $(seq 1 "$REP"); do
  for cfg in "A1 2048 sgd" "A2 16384 adagrad" "A4 131072 adagrad"; do
    set -- $cfg
    run_one "${1}_r${i}" "$2" "$3"
    sleep 3
  done
done
echo "ANCHOR_DONE $(date +%H:%M:%S)" >> "$OUT"
