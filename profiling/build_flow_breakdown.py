#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 profiling/A{2,4}_ec_dedup 的 kernel_details.csv 生成「按训练流程」的耗时分解。

口径（写进报告）：
  1) 每步按 kernel 名 + 步内先后顺序切 6 段：S1 稀疏前向 / S2 稠密前向 / S3 损失 /
     S4 反向 / S5 优化器 / S6 指标，另加 host 取数（不在 step 窗口内）。
  2) 段内一律用**区间并集**（多流重叠不重复计），不用 Duration 直接求和。
  3) 步时窗口取 step_trace_time.csv 的 Stage 字段；窗口 − 并集 = 空隙。
  4) 自证：Σ段并集 ≈ Computing + Communication(Not Overlapped)；Stage = Computing + Comm + Free。

输出：profiling/flow_breakdown_A2_A4.json（供报告直接读取，不手抄数字）
"""
import csv, json, re, statistics
from collections import defaultdict

PROF = "/root/dlrm/profiling"
OUT = PROF + "/flow_breakdown_A2_A4.json"
TIERS = [("A2", "A2_ec_dedup", 16384), ("A4", "A4_ec_dedup", 131072)]

# 类别规则：按 band 给 (正则, 类别)，先匹配先归属；未匹配 → 其它
RULES = {
    "S1": [
        (r"^hcom_|^HcclLaunch", "① 通信 · id 交换"),
        (r"hkv|insert_or_assign|find_and_update|find_ptr_with_digest|load_from_pointer|"
         r"initialize_with_index|IndexSelect_GatherV2|aclnnGather_GatherV2", "③ HKV 查表 / 建表"),
        (r"Unique2|NonzeroV2|aclnnIndex_Index|IndexPut|maskselect|LogicalNot|get_table_range_op|"
         r"get_new_length|block_bucketsize|aclnnCumsum|Stack_Pack|Adds_|ReduceSum|InplaceFillScalar|"
         r"InplaceZero|Transpose|BroadcastTo|TensorMove|Muls_Mul|device_timestamp|InplaceCopy_Cast",
         "② id 去重 / 长度计算"),
    ],
    "S2": [
        (r"^HcclLaunch", "③ 通信发射"),
        (r"Addmm|MatMulV3|BatchMatMul", "① 矩阵乘（MLP + 交互）"),
        (r"Relu|ConcatD|Triu|SelectV2|Cast|InplaceZero|InplaceFillScalar", "② 激活 / 拼接 / 整形"),
    ],
    "S3": [(r".*", "BCE 前向")],
    "S4": [
        (r"^hcom_|^HcclLaunch", "③ 梯度通信"),
        (r"IndexAdd", "② 稀疏梯度累加"),
        (r".*", "① 梯度计算（MLP / 交互 / 规约）"),
    ],
    "S5": [(r"rowwise_adagrad", "① 稀疏 HKV 更新"), (r".*", "② 稠密参数更新")],
    "S6": [(r".*", "指标 / 收尾")],
}
BAND_NAME = {"S1": "S1 稀疏前向", "S2": "S2 稠密前向", "S3": "S3 损失",
             "S4": "S4 反向", "S5": "S5 优化器", "S6": "S6 指标"}


def load_kernels(p):
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8", errors="replace")))
    for r in rows:
        r["s"] = float(str(r["Start Time(us)"]).strip())
        r["d"] = float(str(r["Duration(us)"]).strip())
        r["e"] = r["s"] + r["d"]
    return rows


def union_ms(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return sum(e - s for s, e in out) / 1000.0


def ktype(n):
    return re.sub(r"_612_\d+(_\d+)?$", "", n)


def cat_of(band, name):
    for pat, cat in RULES[band]:
        if re.search(pat, name):
            return cat
    return "④ 其它"


def do_tier(tag, dirn, global_bs):
    base = "%s/%s/" % (PROF, dirn)
    rows = load_kernels(base + "kernel_details.csv")
    stg = {}
    for r in csv.DictReader(open(base + "step_trace_time.csv", newline="", encoding="utf-8", errors="replace")):
        stg[r["Step"]] = {k: float(str(v).strip()) for k, v in r.items()
                          if k != "Device_id" and str(v).strip() != ""}
    by = defaultdict(list)
    for r in rows:
        by[r["Step Id"]].append(r)
    steps = sorted(by, key=int)
    acc = defaultdict(lambda: defaultdict(list))       # band -> metric -> [per-step]
    cat_ms = defaultdict(lambda: defaultdict(list))    # band -> cat -> [per-step sum]
    op_ms = defaultdict(lambda: defaultdict(list))     # band -> ktype -> [per-step sum]
    op_cnt = defaultdict(lambda: defaultdict(list))
    walls = []
    for st in steps:
        rs = sorted(by[st], key=lambda r: r["s"])
        walls.append(stg[st]["Stage"] / 1000.0)
        i_d = next(i for i, r in enumerate(rs) if r["Name"].startswith("aclnnAddmm_MatMulV3Common"))
        i_l = next(i for i, r in enumerate(rs) if r["Name"].startswith("aclnnBinaryCrossEntropyWithLogits_"))
        i_b = next(i for i, r in enumerate(rs) if "BinaryCrossEntropyWithLogitsBackward" in r["Name"])
        i_o = next(i for i, r in enumerate(rs) if ("rowwise_adagrad" in r["Name"] or "ForeachAddList" in r["Name"]))
        i_m = next((i for i, r in enumerate(rs) if r["Name"].startswith("aclnnSigmoid_Sigmoid")), len(rs))
        bands = [("S1", rs[:i_d]), ("S2", rs[i_d:i_l]), ("S3", rs[i_l:i_b]),
                 ("S4", rs[i_b:i_o]), ("S5", rs[i_o:i_m]), ("S6", rs[i_m:])]
        for b, v in bands:
            acc[b]["span"].append((v[-1]["e"] - v[0]["s"]) / 1000.0)
            acc[b]["busy"].append(union_ms([(r["s"], r["e"]) for r in v]))
            acc[b]["comm"].append(union_ms([(r["s"], r["e"]) for r in v
                                            if re.search(r"^hcom_|^HcclLaunch", r["Name"])]))
            # 类别/算子也用区间并集：多流重叠处不重复计（类别之间仍可重叠）
            iv_cat, iv_op, per_cnt = defaultdict(list), defaultdict(list), defaultdict(int)
            for r in v:
                c = cat_of(b, r["Name"])
                iv_cat[c].append((r["s"], r["e"]))
                iv_op[ktype(r["Name"])].append((r["s"], r["e"]))
                per_cnt[ktype(r["Name"])] += 1
            for c, iv in iv_cat.items():
                cat_ms[b][c].append(union_ms(iv))
            for n, iv in iv_op.items():
                op_ms[b][n].append(union_ms(iv))
                op_cnt[b][n].append(per_cnt[n])
    wall = statistics.mean(walls)
    out = {"tier": tag, "global_bs": global_bs, "steps": len(steps),
           "first_step": steps[0], "last_step": steps[-1],
           "wall_ms": wall,
           "computing": statistics.mean([stg[s]["Computing"] / 1000.0 for s in steps]),
           "comm_no_ovl": statistics.mean([stg[s]["Communication(Not Overlapped)"] / 1000.0 for s in steps]),
           "free": statistics.mean([stg[s]["Free"] / 1000.0 for s in steps]),
           "bands": {}, "band_order": ["S1", "S2", "S3", "S4", "S5", "S6"]}
    for b in out["band_order"]:
        m = {k: statistics.mean(v) for k, v in acc[b].items()}
        span, busy, comm = m["span"], m["busy"], m["comm"]
        cats = []
        for c in sorted(cat_ms[b], key=lambda c: -statistics.mean(cat_ms[b][c])):
            mv = statistics.mean(cat_ms[b][c])
            ops = []
            for n in op_ms[b]:
                if cat_of(b, n) != c:
                    continue
                ops.append({"name": n, "ms": statistics.mean(op_ms[b][n]),
                            "calls": statistics.mean(op_cnt[b][n])})
            ops.sort(key=lambda x: -x["ms"])
            cats.append({"cat": c, "ms": mv, "ops": ops})
        out["bands"][b] = {"name": BAND_NAME[b], "span": span, "busy": busy, "comm": comm,
                           "compute": busy - comm, "gap": span - busy, "cats": cats}
    return out


def main():
    res = {"source": "profiling/{A2,A4}_ec_dedup/kernel_details.csv + step_trace_time.csv",
           "method": "kernel 名 + 步内顺序切段；段内区间并集；窗口=Stage",
           "tiers": {}}
    for tag, dirn, bs in TIERS:
        res["tiers"][tag] = do_tier(tag, dirn, bs)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    # 自证打印
    for tag, t in res["tiers"].items():
        busy = sum(b["busy"] for b in t["bands"].values())
        span = sum(b["span"] for b in t["bands"].values())
        print("%s wall=%.2f Σspan=%.2f Σbusy=%.2f (Computing+Comm=%.2f) idle=%.2f"
              % (tag, t["wall_ms"], span, busy, t["computing"] + t["comm_no_ovl"],
                 t["wall_ms"] - busy))
        for b in t["band_order"]:
            d = t["bands"][b]
            print("   %-10s span=%6.2f (%5.1f%%) busy=%6.2f comm=%5.2f comp=%5.2f gap=%6.2f"
                  % (d["name"], d["span"], d["span"] / t["wall_ms"] * 100, d["busy"],
                     d["comm"], d["compute"], d["gap"]))
            for c in d["cats"]:
                print("      %-28s %6.3f" % (c["cat"], c["ms"]))


if __name__ == "__main__":
    main()
