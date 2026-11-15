#!/bin/bash
# 定期发布：从运行节点收集 训练日志摘要 + profile 关键数据 → 脱敏闸门 → commit/push
#
# 用法: NODE_SSH=<节点> CTR=<容器> tools/publish.sh ["commit message"]
# 环境变量: NODE_SSH（必填）、CTR（必填）、REMOTE_TMP（默认 /tmp）、KEEP_FULL_RE（默认 ^un_）
#       GITHUB_TOKEN 从 ~/.hermes/.env 读取（不写入仓库、不写入 git config）
set -eu
MSG=${1:-"dlrm: 发布训练日志与 profile 摘要"}
NODE_SSH=${NODE_SSH:?需要设置 NODE_SSH（节点 ssh 别名）}
CTR=${CTR:?需要设置 CTR（容器名）}
RTMP=${REMOTE_TMP:-/tmp}
# 只保留「已验证有效」优化项的完整日志；其余只留摘要行（KB 级），避免仓库膨胀
KEEP_FULL_RE=${KEEP_FULL_RE:-^un_}
ROOT=$(cd "$(dirname "$0")/.." && pwd)          # dlrm/
REPO=$(cd "$ROOT/.." && pwd)
LOGS="$ROOT/results/logs"
mkdir -p "$LOGS"

# 日志里必然出现 venv / 数据盘 / CANN 的绝对路径，统一替换成占位再落盘
sanitize() { sed -E 's#/opt/buildtools/[^ ]*#<VENV>#g; s#/mnt/paas/[^ ]*#<DATA_DIR>#g; s#/usr/local/Ascend[^ ]*#<CANN>#g'; }

pull() {  # $1=远端文件 $2=本地文件
  ssh "$NODE_SSH" "docker exec $CTR cat '$1'" > "$2" 2>/dev/null || { echo "skip $1"; rm -f "$2"; }
}

echo "== 1. 训练日志：摘要行（+ 有效优化项保留压缩全量）=="
collect_one() {   # $1 = 远端路径
  local f=$1 base tmp
  base=$(basename "$f" .log); tmp="$LOGS/.$base.tmp"
  pull "$f" "$tmp"
  [ -s "$tmp" ] || return 0
  grep -aE 'step_time_ms|throughput|RESULT:|test_auc|test_acc|\[unified\]|world_size' "$tmp" \
    | sanitize | head -20 > "$LOGS/$base.summary.txt"
  if echo "$base" | grep -qE "$KEEP_FULL_RE" && [ "$(stat -c%s "$tmp")" -le 8388608 ]; then
    sanitize < "$tmp" | gzip -c > "$LOGS/$base.log.gz"
  fi
  rm -f "$tmp"
}
for pat in an_ un_ p1a_ sw_ smoke_ uni_; do
  for f in $(ssh "$NODE_SSH" "docker exec $CTR bash -lc 'ls -S $RTMP/${pat}*.log 2>/dev/null | head -6'" | tr -d '\r'); do
    [ -n "$f" ] && collect_one "$f"
  done
done
echo "  摘要 $(ls "$LOGS"/*.summary.txt 2>/dev/null | wc -l) 份，全量 $(ls "$LOGS"/*.log.gz 2>/dev/null | wc -l) 份"

echo "== 2. profile：算子 top 表（若存在 op_statistic）=="
ssh "$NODE_SSH" "docker exec $CTR bash -lc 'ls $RTMP/prof_*/PROF_*/mindstudio_profiler_output/op_statistic_*.csv 2>/dev/null | head -1'" \
  | tr -d '\r' | while read -r op; do
  [ -n "$op" ] || continue
  pull "$op" "$LOGS/.op_raw.csv"
  [ -s "$LOGS/.op_raw.csv" ] || continue
  python3 - "$LOGS/.op_raw.csv" "$ROOT/profiling/op_top_k.csv" <<'PY'
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, encoding="utf-8", errors="ignore")))
if not rows:
    sys.exit(0)
key_op = next((k for k in rows[0] if "op" in k.lower() and "type" in k.lower()), list(rows[0])[0])
key_t = next((k for k in rows[0] if "time" in k.lower()), list(rows[0])[-1])
key_n = next((k for k in rows[0] if "count" in k.lower()), None)
def num(v):
    try: return float(str(v).replace(",", ""))
    except Exception: return 0.0
agg = {}
for r in rows:
    op = r[key_op]
    a = agg.setdefault(op, [0.0, 0])
    a[0] += num(r[key_t]); a[1] += int(num(r[key_n])) if key_n else 1
top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:25]
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["op_type", "total_time_ms", "count"])
    for op, (t, c) in top:
        w.writerow([op, round(t, 3), c])
print(f"op_top_k: {len(top)} 行 → {dst}")
PY
  rm -f "$LOGS/.op_raw.csv"
done

echo "== 3. 脱敏闸门 =="
python3 "$ROOT/tools/scrub_check.py" "$ROOT" || { echo "命中敏感信息，已中止发布"; exit 1; }

echo "== 4. commit + push =="
cd "$REPO"
git add -A
git diff --cached --quiet && { echo "无变更，跳过"; exit 0; }
git -c user.name=xiaomingyao -c user.email=xiaomingyao@users.noreply.github.com commit -q -m "$MSG"
T=$(grep '^GITHUB_TOKEN=' /root/.hermes/.env | cut -d= -f2)
AUTH=$(printf 'x-access-token:%s' "$T" | base64 -w0)
git -c http.extraHeader="Authorization: Basic $AUTH" push -q origin main
echo "pushed: $(git log --oneline -1)"
