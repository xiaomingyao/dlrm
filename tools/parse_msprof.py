import sys
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""msprof 产物分析：区分初始化 vs 稳态，算通信占比"""
import csv, glob, os
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else "mindstudio_profiler_output"

def load(name):
    f = glob.glob(os.path.join(D, name))[0]
    with open(f) as fh:
        return list(csv.DictReader(fh))

tasks = load("task_time_*.csv")
comm_ops = {"hcom_alltoall_AicpuKernel","hcom_allReduce_AicpuKernel","hcom_allGather_AicpuKernel",
            "hcom_reduceScatter_AicpuKernel","hcom_broadcast_AicpuKernel","hcom_alltoallv_AicpuKernel",
            "HcclLaunchAicpuKernel"}

# 时间跨度
starts = [float(r["task_start(us)"]) for r in tasks]
stops  = [float(r["task_stop(us)"]) for r in tasks]
t0, t1 = min(starts), max(stops)
wall = t1 - t0
print(f"时间跨度: {t0:.0f} -> {t1:.0f} us   wall = {wall/1000:.1f} ms")
print(f"任务数: {len(tasks)}")
print()

# 按 "是否通信" 汇总
def cls(name):
    n = name
    if n in comm_ops or "hcom_" in n or "AllToAll" in n or "AllReduce" in n or "AllGather" in n:
        return "通信"
    if "CommInit" in n or "ChannelInit" in n or "ThreadInit" in n:
        return "通信初始化"
    return "计算/I-O"

g = defaultdict(lambda: [0, 0.0])
for r in tasks:
    k = cls(r["kernel_name"])
    g[k][0] += 1
    g[k][1] += float(r["task_time(us)"])

tot_busy = sum(v[1] for v in g.values())
print("=" * 72)
print("【全程 busy 时间按类别】（注：多 stream 并行，busy 之和可 > wall）")
print("=" * 72)
for k, (n, t) in sorted(g.items(), key=lambda x: -x[1][1]):
    print(f"  {k:<12} n={n:>6}  busy={t/1000:>10.1f}ms  ratio={t/tot_busy*100:>6.2f}%  (占wall {t/wall*100:>6.1f}%)")
print(f"  {'合计':<12} {'':>8}  busy={tot_busy/1000:>10.1f}ms")
print()

# 找出最大单笔任务（初始化嫌疑）
print("=" * 72)
print("【耗时最长的 8 个单任务】")
print("=" * 72)
for r in sorted(tasks, key=lambda x: -float(x["task_time(us)"]))[:8]:
    print(f"  {r['kernel_name'][:55]:<55} {float(r['task_time(us)'])/1000:>8.2f}ms  @{float(r['task_start(us)'])/1000:.0f}ms")
print()

# 稳态：剔除首 30% 时间窗（初始化集中在前段）
cut = t0 + (t1 - t0) * 0.30
steady = [r for r in tasks if float(r["task_start(us)"]) >= cut]
g2 = defaultdict(lambda: [0, 0.0])
for r in steady:
    k = cls(r["kernel_name"])
    g2[k][0] += 1
    g2[k][1] += float(r["task_time(us)"])
tot2 = sum(v[1] for v in g2.values())
smin = min(float(r["task_start(us)"]) for r in steady)
smax = max(float(r["task_stop(us)"]) for r in steady)
swall = smax - smin
print("=" * 72)
print(f"【稳态窗口】 起于 {cut/1000:.1f}ms（剔除前 30%），窗口 {swall/1000:.1f} ms，任务 {len(steady)}")
print("=" * 72)
for k, (n, t) in sorted(g2.items(), key=lambda x: -x[1][1]):
    print(f"  {k:<12} n={n:>6}  busy={t/1000:>10.1f}ms  ratio={t/tot2*100:>6.2f}%  (占wall {t/swall*100:>6.1f}%)")
print(f"  {'合计':<12} {'':>8}  busy={tot2/1000:>10.1f}ms")

# 稳态里通信明细
print()
print("【稳态窗口内通信算子明细】")
cg = defaultdict(lambda: [0, 0.0])
for r in steady:
    if cls(r["kernel_name"]) == "通信":
        cg[r["kernel_name"]][0] += 1
        cg[r["kernel_name"]][1] += float(r["task_time(us)"])
for k, (n, t) in sorted(cg.items(), key=lambda x: -x[1][1]):
    print(f"  {k:<34} n={n:>4}  total={t/1000:>9.2f}ms  avg={t/n/1000:>7.2f}ms")
