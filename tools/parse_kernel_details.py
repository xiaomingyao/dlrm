#!/usr/bin/env python3
"""解析 torch_npu.profiler 导出的 kernel_details.csv（比 msprof 稳，不被 dynamic_emb 打断）。

用法：python3 parse_kernel_details.py <PROF_DIR> [steps]
  <PROF_DIR> 下需含 *kernel_details*.csv（profiler 导出目录）

输出：
  1) 窗口（设备上首末 kernel 之间）与步数、每步 wall
  2) 分类并集：compute / comm / aicpu / wait（同一时刻可重叠，占比和可 >100%）
  3) 按 kernel 名的 top 15：每步时长、调用次数、单次耗时
"""
import csv
import glob
import os
import sys

CLASS_MAP = (
    ("wait", ("WAIT", "NOTIFY")),
    ("comm", ("UBDMA", "SDMA", "RDMA", "HCCL", "ALLTOALL", "ALLREDUCE", "REDUCESCATTER")),
    ("aicpu", ("AI_CPU", "AICPU")),
    ("compute", ("AI_CORE", "AI_VECTOR", "MIX_", "AIV", "AIC", "CUBE", "VECTOR")),
)


def classify(name: str, ktype: str) -> str:
    blob = (name + " " + ktype).upper()
    for cls, keys in CLASS_MAP:
        if any(k in blob for k in keys):
            return cls
    return "other"


def union_ms(intervals):
    if not intervals:
        return 0.0
    intervals.sort()
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged) / 1000.0


def main():
    root = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    path = sorted(glob.glob(os.path.join(root, "**", "*kernel_details*.csv"), recursive=True))
    if not path:
        sys.exit("未找到 kernel_details*.csv，检查 profiler export_type 是否包含 text")
    rows = list(csv.DictReader(open(path[0])))

    def pick(*names):
        for n in names:
            if n in rows[0]:
                return n
        return None

    c_name = pick("Name", "name")
    c_type = pick("Type", "type")
    c_start = pick("Start Time(us)", "Start Time", "start_time")
    c_dur = pick("Duration(us)", "Duration", "duration")
    c_dev = pick("Device_id", "Device ID", "device_id")

    if c_dev:
        from collections import Counter
        dev = Counter(r[c_dev] for r in rows).most_common(1)[0][0]
        rows = [r for r in rows if r[c_dev] == dev]

    data = []
    for r in rows:
        try:
            s = float(r[c_start]); d = float(r[c_dur])
        except (TypeError, ValueError):
            continue
        data.append((s, s + d, r.get(c_name, ""), r.get(c_type, ""), d))

    if not data:
        sys.exit("kernel_details 里没有可用的时长数据")

    t0 = min(x[0] for x in data)
    t1 = max(x[1] for x in data)
    wall = (t1 - t0) / 1000.0
    print(f"来源      : {path[0]}")
    print(f"窗口      : {wall:.1f} ms / {steps} 步 = {wall / steps:.2f} ms/步")
    print(f"kernel 数 : {len(data)}（{len(data) / steps:.0f}/步）\n")

    buckets = {}
    for s, e, nm, ty, d in data:
        buckets.setdefault(classify(nm, ty), []).append((s, e))
    print("分类并集（可重叠，占比和可 >100%）：")
    for cls in ("compute", "comm", "aicpu", "wait", "other"):
        ms = union_ms(buckets.get(cls, []))
        if ms:
            print(f"  {cls:8s}{ms / steps:8.3f} ms/步  {ms / wall * 100:6.1f}% wall")

    agg, cnt = {}, {}
    for s, e, nm, ty, d in data:
        key = (ty, nm[:58])
        agg[key] = agg.get(key, 0.0) + d / 1000.0
        cnt[key] = cnt.get(key, 0) + 1
    print(f"\n{'type':10s}{'name':44s}{'ms/step':>9s}{'calls/step':>11s}{'us/call':>9s}")
    for (ty, nm), ms in sorted(agg.items(), key=lambda kv: -kv[1])[:15]:
        print(f"{ty[:9]:10s}{nm[:42]:44s}{ms / steps:9.3f}{cnt[(ty, nm)] / steps:11.1f}"
              f"{ms * 1000 / cnt[(ty, nm)]:9.0f}")


if __name__ == "__main__":
    main()
