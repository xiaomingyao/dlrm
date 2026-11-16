#!/bin/bash
# 对齐后的「优化配置」锚定复跑：与 README §2.1 的三行配置一致
#   A1 = local 256(global 2,048)  统一表 + EBC
#   A2 = local 2,048(global 16,384) 统一表 + EC + dedup + Adagrad(dense)
#   A4 = local 16,384(global 131,072) 统一表 + EC + dedup + Adagrad(dense)
# 用法: tools/run_aligned_opt.sh <运行目录(已 sync)> <DATA_DIR> [repeats]
set -u
DIR=${1:?运行目录}; DATA=${2:?DATA_DIR}; REP=${3:-3}
STEPS=${STEPS:-300}; WARMUP=${WARMUP:-20}; NPROC=${NPROC:-8}
OUT="$DIR/anchor_opt_$(date +%m%d_%H%M).txt"
: > "$OUT"
cd "$DIR" || exit 2
echo "STEPS=$STEPS WARMUP=$WARMUP REP=$REP DATA=$DATA start=$(date +%H:%M:%S)" >> "$OUT"

run_one() {  # $1=tag $2=global_bs $3=dense_optim $4=extra flags
  local tag="$1"
  local bs="$2"
  local opt="$3"
  local extra="$4"
  local log="$DIR/anchor_opt_${tag}.log"
  : > "$log"
  echo "=== $tag start $(date +%H:%M:%S) bs=$bs dense_optim=$opt extra=[$extra] ===" >> "$OUT"
  # shellcheck disable=SC2086
  torchrun --nproc_per_node="$NPROC" --master_port=$((33000 + RANDOM % 500)) dlrm_v2_train.py \
    --batch-size "$bs" --epochs 1 --eval-every 0 --warmup "$WARMUP" --steps "$STEPS" \
    --data-dir "$DATA" --train-days 0,1,2,3,4,5 --test-days 6 --dense-optim "$opt" $extra \
    > "$log" 2>&1
  echo "$tag rc=$?" >> "$OUT"
  grep -aE "inter_dim|top_dims|dense_optim|step_time_ms|test_auc" "$log" | tail -5 >> "$OUT"
}

for i in $(seq 1 "$REP"); do
  run_one "A1_r${i}" 2048 sgd "--unified-emb"
  run_one "A2_r${i}" 16384 adagrad "--unified-emb --ec-mode --index-dedup"
  run_one "A4_r${i}" 131072 adagrad "--unified-emb --ec-mode --index-dedup"
  sleep 3
done
echo "ANCHOR_OPT_DONE $(date +%H:%M:%S)" >> "$OUT"
