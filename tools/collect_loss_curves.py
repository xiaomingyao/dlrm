#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽取训练曲线（逐 step loss/acc）与每轮终评指标，供报告画精度曲线。

数据源：
  /root/dlrm-artifacts_v1_20261116/logs/*.log.gz   全量训练日志（脱敏 gzip）
  日志自带的逐 step 行：  ep0 step   250/1354  loss=0.129354 acc=0.9692  16.97 ms
输出：results/loss_curves.json
"""
import glob, gzip, json, re

SRC = "/root/dlrm-artifacts_v1_20261116/logs/*.log.gz"
OUT = "/root/dlrm/results/loss_curves.json"
PICK = ["an_A2_c_r1", "un_A2_u_r1", "ec2_A2_dd_r1", "util3_run", "a4_r3", "an_A1_c_r1", "un_A1_u_r1"]
STEP_RE = re.compile(r"ep(\d+) step\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+acc=([\d.]+)")


def parse(path):
    pts = []
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                ep, st, tot, loss, acc = m.groups()
                pts.append({"epoch": int(ep), "step": int(st), "total": int(tot),
                            "loss": float(loss), "acc": float(acc)})
    return pts


def main():
    out = {}
    for name in PICK:
        p = "/root/dlrm-artifacts_v1_20261116/logs/%s.log.gz" % name
        try:
            pts = parse(p)
        except FileNotFoundError:
            print("MISS", name)
            continue
        out[name] = {"points": pts, "n": len(pts),
                     "steps": sorted({q["step"] for q in pts}),
                     "total_steps": pts[-1]["total"] if pts else None}
        print("%-14s n=%-3d total=%-5s steps=%s" % (name, len(pts), out[name]["total_steps"],
                                                    out[name]["steps"][:12]))
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
