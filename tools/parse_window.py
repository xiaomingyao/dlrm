#!/usr/bin/env python3
"""解析 msprof task_time，按训练窗口输出单步预算（多 stream 取并集）。

用法: python3 parse_window.py <prof_root> [steps]

窗口定义：含 'lookup' 的 kernel 的首个 start 到最后一个 stop（每步必现）。
输出：window、wall/步、busy 并集/步、idle/步、按 kernel 的 top 明细。
"""
import csv, glob, os, sys, collections

root = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20

cands = sorted(glob.glob(os.path.join(root, "**", "task_time_*.csv"), recursive=True))
if not cands:
    sys.exit(f"未找到 task_time*.csv（root={root}）；msprof 需用 --application= 启动目标进程")
path = cands[0]
print(f"file = {path}")

with open(path, newline="") as f:
    r = csv.DictReader(f)
    cols = r.fieldnames
    rows = list(r)
print("cols =", cols)


def pick(*names):
    for n in names:
        for c in cols:
            if c.strip().lower() == n:
                return c
    return None


c_dev = pick("device_id")
c_name = pick("kernel_name") or pick("name")
c_type = pick("kernel_type")
c_dur = pick("task_time(us)", "task_time", "duration_time", "dur")
c_start = pick("task_start(us)", "start_time(us)", "start_time", "ts")
c_stop = pick("task_stop(us)", "stop_time(us)", "stop_time")
if not (c_dev and c_dur and c_start):
    sys.exit(f"列不匹配: {cols}")

recs = []
for row in rows:
    try:
        dev = int(row[c_dev]); dur = float(row[c_dur]); st = float(row[c_start])
    except (ValueError, TypeError, KeyError):
        continue
    sp = float(row[c_stop]) if c_stop and row.get(c_stop) else st + dur
    recs.append((dev, row.get(c_name, "N/A") or "N/A", row.get(c_type, "") or "", dur, st, sp))

if not recs:
    sys.exit("无有效任务行")

dev = collections.Counter(r[0] for r in recs).most_common(1)[0][0]
sel = [r for r in recs if r[0] == dev]
print(f"device = {dev}  任务数 = {len(sel)}")

lk = [r for r in sel if "lookup" in r[1].lower()]
if lk:
    t0 = min(r[4] for r in lk); t1 = max(r[5] for r in lk)
    print(f"窗口用 lookup kernel 定义：{len(lk)} 次")
else:
    t0 = min(r[4] for r in sel); t1 = max(r[5] for r in sel)
    print("窗口退回全量首末时间")
win = [r for r in sel if r[4] >= t0 and r[5] <= t1]
wall_ms = (t1 - t0) / 1000.0

# busy 并集
ivs = sorted((r[4], r[5]) for r in win)
merged = []
for s, e in ivs:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
busy_ms = sum(e - s for s, e in merged) / 1000.0

print(f"\n窗口 {wall_ms:.1f} ms / {steps} 步 = {wall_ms/steps:.3f} ms/步")
print(f"busy {busy_ms:.1f} ms = {busy_ms/steps:.3f} ms/步 ({busy_ms/wall_ms*100:.1f}%)")
print(f"idle {wall_ms-busy_ms:.1f} ms = {(wall_ms-busy_ms)/steps:.3f} ms/步 ({(wall_ms-busy_ms)/wall_ms*100:.1f}%)\n")

agg = collections.defaultdict(lambda: [0.0, 0])
for r in win:
    k = r[1] if r[1] != "N/A" else f"<{r[2]}>"
    agg[k][0] += r[3]; agg[k][1] += 1

print(f"{'kernel':52s}{'ms/step':>10s}{'%wall':>8s}{'calls':>8s}{'us/call':>9s}")
for k, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:16]:
    print(f"{k[:50]:52s}{t/1000/steps:10.3f}{t/1000/wall_ms*100:8.1f}{c/steps:8.1f}{t/c:9.0f}")

by_type = collections.defaultdict(float)
for r in win:
    by_type[r[2] or "NA"] += r[3]
print("\n按 kernel_type（求和，非并集）：")
for k, t in sorted(by_type.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k:24s}{t/1000/steps:9.3f} ms/步")

# ---- 分类并集：计算 / 通信 / 等待 ----
def classify(ktype: str, kname: str) -> str:
    k = (ktype or "").upper(); n = (kname or "").upper()
    if "WAIT" in k or "WAIT" in n:
        return "wait"
    if any(x in k for x in ("UBDMA", "SDMA", "RDMA", "HCCL")):
        return "comm"
    if "AI_CPU" in k or "AICPU" in k:
        return "aicpu"
    if any(x in k for x in ("AI_CORE", "AI_VECTOR", "MIX_", "AIV", "AIC")):
        return "compute"
    return "other"

buckets = collections.defaultdict(list)
for r in win:
    buckets[classify(r[2], r[1])].append((r[4], r[5]))

def union_ms(iv):
    if not iv: return 0.0
    iv = sorted(iv); m = []
    for s, e in iv:
        if m and s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return sum(e - s for s, e in m) / 1000.0

print("\n分类并集（同一时刻多类可重叠，占比之和可 >100%）：")
for k in ("compute", "comm", "aicpu", "wait", "other"):
    ms = union_ms(buckets.get(k, []))
    if ms:
        print(f"  {k:8s}{ms/steps:8.3f} ms/步  {ms/wall_ms*100:6.1f}% wall")
