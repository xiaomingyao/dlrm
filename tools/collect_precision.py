#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收集精度/吞吐日志里的 test_auc / test_acc / loss rel_change / 步数，输出机器可读 JSON。

数据源（全部落盘，不手抄）：
  results/logs/*.summary.txt             —— 每轮 20 行摘要（150–300 步锚定跑）
  /root/dlrm-artifacts_v1_20261116/logs/*.log.gz —— 含满 epoch 跑（1,354 / 169 / 10,832 步）

输出：results/precision_runs.json
"""
import glob, gzip, json, re

ART = "/root/dlrm-artifacts_v1_20261116/logs/*.log.gz"
SUM = "/root/dlrm/results/logs/*.summary.txt"
OUT = "/root/dlrm/results/precision_runs.json"

GROUPS = [
    (r"^an_A[12]_c_", "26 表 + EBC（基线）"),
    (r"^an_A[12]_rr_", "26 表 + EBC + roundrobin"),
    (r"^un_A[12]_u_", "统一表 + EBC"),
    (r"^ec2_A[12]_dd_", "统一表 + EC + dedup"),
    (r"^sw_b2_", "统一表 + EC + dedup + 同步节奏（A1）"),
    (r"^sw_b16_", "统一表 + EC + dedup + 同步节奏（A2）"),
    (r"^sw_a2_", "统一表 + EBC + overlap / tq2 探针"),
    (r"^p1a_b32$", "统一表 + EC + dedup · 32k"),
    (r"^p1a_rr16384$", "统一表 + EC + dedup · 16k roundrobin"),
    (r"^util", "统一表 + EC + dedup · 满 epoch 采集"),
    (r"^hbm_run$", "统一表 + EC + dedup · 满 epoch 采集"),
    (r"^a4_r3$", "统一表 + EC + dedup · A4 满 epoch"),
    (r"^smoke", "探针（smoke，步数不足）"),
]


def group_of(name):
    for pat, g in GROUPS:
        if re.search(pat, name):
            return g
    return "未分组"


def parse(text):
    def g(pat, cast=str):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None
    return {
        "steps": g(r"total steps\s*=\s*(\d+)", int),
        "med_ms": g(r"step_time_ms\s*=\s*mean=[\d.]+ median=([\d.]+)", float),
        "sps": g(r"throughput\s*=\s*([\d.]+)", float),
        "auc": g(r"test_auc\s*=\s*([\d.]+)", float),
        "acc": g(r"test_acc\s*=\s*([\d.]+)", float),
        "acc_base": g(r"全负基线\s*([\d.]+)", float),
        "loss_tail": g(r"loss_tail\d+\s*=\s*([\d.]+)", float),
        "loss_rel": g(r"loss rel_change=(-?[\d.]+%)"),
        "result": g(r"RESULT:\s*(\w+)"),
        "prep_ms": g(r"prep_ms\s*=\s*median=([\d.]+)", float),
    }


def main():
    runs = {}
    for f in glob.glob(SUM):
        name = f.split("/")[-1].replace(".summary.txt", "")
        runs[name] = dict(parse(open(f, errors="replace").read()), src=f)
    for f in glob.glob(ART):
        name = f.split("/")[-1].replace(".log.gz", "")
        if name in runs:
            continue
        txt = gzip.open(f, "rt", errors="replace").read()
        runs[name] = dict(parse(txt), src=f)
    out = []
    for name, d in sorted(runs.items()):
        d["run"] = name
        d["group"] = group_of(name)
        out.append(d)
    json.dump({"source": [SUM, ART], "runs": out}, open(OUT, "w"), ensure_ascii=False, indent=1)
    for d in out:
        print("%-16s %-38s steps=%-6s med=%-7s auc=%-7s acc=%-7s base=%s loss=%s %s"
              % (d["run"], d["group"], d["steps"], d["med_ms"], d["auc"], d["acc"],
                 d["acc_base"], d["loss_rel"], d["result"]))
    print("\nwrote", OUT, len(out), "runs")


if __name__ == "__main__":
    main()
