#!/bin/bash
# 单向同步：本仓 dlrm/ → NPU 节点运行目录（仓是真源，节点只读副本）
# 用法: tools/sync_to_node.sh <目标目录>   或   DLRM_NODE_DIR=<目标目录> tools/sync_to_node.sh
set -eu
DEST=${1:-${DLRM_NODE_DIR:-}}
if [ -z "$DEST" ]; then
  echo "ERROR: 需要目标目录。用法: tools/sync_to_node.sh /path/on/node  或 export DLRM_NODE_DIR=..." >&2
  exit 2
fi
SRC=$(cd "$(dirname "$0")/.." && pwd)
echo "src = $SRC"
echo "dst = $DEST"
mkdir -p "$DEST/tools"

FILES="dlrm_v2_train.py dlrm_v2_model.py criteo_dataset.py"
TOOLS="p2_no_step_sync.py parse_msprof.py"

for f in $FILES; do cp -a "$SRC/$f" "$DEST/$f"; done
for f in $TOOLS; do cp -a "$SRC/tools/$f" "$DEST/tools/$f"; done

# 身份闸门：两侧 md5 必须逐一对齐
fail=0
for f in $FILES; do
  a=$(md5sum "$SRC/$f" | cut -d' ' -f1); b=$(md5sum "$DEST/$f" | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "MISMATCH $f $a != $b"; fail=1; }
  printf '%-24s %s\n' "$f" "$b"
done
for f in $TOOLS; do
  a=$(md5sum "$SRC/tools/$f" | cut -d' ' -f1); b=$(md5sum "$DEST/tools/$f" | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "MISMATCH tools/$f $a != $b"; fail=1; }
  printf '%-24s %s\n' "tools/$f" "$b"
done
[ "$fail" = 0 ] && echo "SYNC OK（两侧 md5 全部一致）" || { echo "SYNC FAILED" >&2; exit 1; }
