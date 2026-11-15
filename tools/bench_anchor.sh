#!/bin/bash
# 锚定基准：只跑 A100 有对应档、且本实现能跑的 A1 / A2
#   A1 = local 256  (global 2,048)   dot  |  A100 8× = 204,800 sps
#   A2 = local 2,048(global 16,384)  dot  |  A100 8× = 925,696 sps
# 用法: tools/bench_anchor.sh <运行目录(已 sync)> <DATA_DIR> [repeats] [额外参数...]
# 环境: NPROC(默认8) TORCHRUN(默认 torchrun) EXTRA_DAYS(默认 0,1,2,3,4,5) TEST_DAYS(默认 6)
set -eu
DIR=${1:?运行目录}
DATA=${2:?DATA_DIR}
REP=${3:-3}
shift 3 2>/dev/null || shift $#
EXTRA="$*"
NPROC=${NPROC:-8}
TORCHRUN=${TORCHRUN:-torchrun}
A1_A100=204800
A2_A100=925696
STEPS=${STEPS:-300}
WARMUP=${WARMUP:-20}

cd "$DIR"
tmp=$(mktemp -d)
summary="$tmp/summary.txt"
: > "$summary"

run_one() {  # $1=tag $2=global_bs
  local tag=$1 bs=$2 log="$tmp/$tag.log"
  $TORCHRUN --nproc_per_node="$NPROC" --master_port=$((31000 + RANDOM % 500)) dlrm_v2_train.py \
    --batch-size "$bs" --epochs 1 --eval-every 0 --warmup "$WARMUP" --steps "$STEPS" \
    --data-dir "$DATA" --train-days "${EXTRA_DAYS:-0,1,2,3,4,5}" --test-days "${TEST_DAYS:-6}" \
    $EXTRA > "$log" 2>&1 || { echo "FAIL $tag (see $log)"; return 1; }
  local med sps auc
  med=$(grep -aoE 'median=[0-9.]+' "$log" | tail -1 | cut -d= -f2)
  auc=$(grep -a 'test_auc' "$log" | tail -1 | awk '{print $3}')
  [ -n "$med" ] || { echo "FAIL $tag: 无 median"; return 1; }
  sps=$(python3 -c "print(f'{$bs/($med/1000):.0f}')")
  echo "$tag $med $sps ${auc:-NA}" >> "$summary"
  echo "  $tag  median=${med} ms  sps=$sps  auc=${auc:-NA}"
}

for cfg in "A1 2048 $A1_A100" "A2 16384 $A2_A100"; do
  set -- $cfg; tag=$1; bs=$2; ref=$3
  echo "== $tag (global $bs) x $REP =="
  for i in $(seq 1 "$REP"); do run_one "${tag}_r$i" "$bs" || true; done
done

echo
echo "== 汇总（median of medians）=="
python3 - "$summary" "$A1_A100" "$A2_A100" <<'PY'
import sys, statistics as st, collections
path, a1, a2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = collections.defaultdict(list); auc = {}
for line in open(path):
    p = line.split()
    if len(p) < 4: continue
    tag = p[0].split('_')[0]; rows[tag].append(float(p[1])); auc[tag] = p[3]
ref = {'A1': a1, 'A2': a2}
bs = {'A1': 2048, 'A2': 16384}
print(f"{'档':4s}{'n':>3s}{'median(ms)':>12s}{'sps':>12s}{'A100':>10s}{'达成率':>9s}  AUC")
ok = True
for tag in ('A1', 'A2'):
    if tag not in rows:
        print(f"{tag:4s} 缺数据"); ok = False; continue
    m = st.median(rows[tag]); sps = bs[tag] / (m / 1000)
    print(f"{tag:4s}{len(rows[tag]):>3d}{m:>12.3f}{sps:>12.0f}{ref[tag]:>10.0f}{sps/ref[tag]*100:>8.1f}%  {auc.get(tag,'NA')}")
print("\n判据：A1/A2 均须出现且 n≥3；达成率 ≥200% 才算满足阶段一目标")
sys.exit(0 if ok else 1)
PY
