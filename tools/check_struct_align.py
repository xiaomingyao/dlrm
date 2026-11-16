#!/usr/bin/env python3
# coding=utf-8
"""结构对齐自检：本仓 DLRMv2 vs 上游 torchrec.models.dlrm.DLRM。

判定项（全部要 PASS 才算对齐）：
  1. 稠密 bottom 层形状一致（13→512→256→128）
  2. dot 交互输出维度一致，且逐元素数值一致（严格上三角、同序）
  3. top 首层 in_features 一致（128 + C(27,2) = 479）
  4. top 层形状一致（479→1024→1024→512→256→1）
  5. 稠密侧参数量一致

用法: python tools/check_struct_align.py [--emb-dim 128] [--bot 512-256] [--top 1024-1024-512-256]
"""
import argparse
import sys

import torch

try:
    import torch_npu  # noqa: F401  某些 torchrec 构建要求先注册 PrivateUse1
except Exception:  # noqa: BLE001
    pass

import torchrec
from torchrec.models.dlrm import DLRM

from dlrm_v2_model import DLRMv2, build_sparse_configs, dot_interaction

NUM_SPARSE, NUM_DENSE = 26, 13


def shapes(mod):
    return [tuple(p.shape) for p in mod.parameters()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--bot", default="512-256")
    ap.add_argument("--top", default="1024-1024-512-256")
    a = ap.parse_args()
    bot = [int(x) for x in a.bot.split("-")]
    top = [int(x) for x in a.top.split("-")]

    # ---- 上游参考实现 ----
    up_ebc = torchrec.EmbeddingBagCollection(
        device=torch.device("meta"),
        tables=build_sparse_configs(NUM_SPARSE, [1000] * NUM_SPARSE, a.emb_dim, prefix="ref"),
    )
    up = DLRM(
        embedding_bag_collection=up_ebc,
        dense_in_features=NUM_DENSE,
        dense_arch_layer_sizes=bot + [a.emb_dim],
        over_arch_layer_sizes=top + [1],
    )

    # ---- 本仓实现 ----
    ours = DLRMv2(
        num_dense=NUM_DENSE,
        sparse_configs=build_sparse_configs(NUM_SPARSE, [1000] * NUM_SPARSE, a.emb_dim, prefix="ours"),
        emb_dim=a.emb_dim,
        bot_dims=bot,
        top_dims=top,
    )

    results = []

    # 1 稠密 bottom 形状
    up_bot = shapes(up.dense_arch)
    our_bot = [s for s in shapes(ours.bot_l)]
    results.append(("稠密 bottom 形状", up_bot == our_bot, f"{up_bot} vs {our_bot}"))

    # 2 交互层：维度 + 数值
    torch.manual_seed(0)
    b, f = 8, NUM_SPARSE + 1
    T = torch.randn(b, f, a.emb_dim)          # 第 0 位是真值 dense embedding，1.. 是 26 个稀疏向量
    up_int = up.inter_arch(T[:, 0, :], T[:, 1:, :])
    up_vec = up_int[:, a.emb_dim:]
    our_vec = dot_interaction(T)
    results.append(("交互维度", up_vec.shape == our_vec.shape, f"{tuple(up_vec.shape)} vs {tuple(our_vec.shape)}"))
    diff = (up_vec - our_vec).abs().max().item() if up_vec.shape == our_vec.shape else float("nan")
    results.append(("交互数值（max|Δ|）", diff < 1e-6, f"{diff:.3e}"))

    # 3 top 首层 in_features
    up_in = shapes(up.over_arch)[0][1]
    our_in = shapes(ours.top_l)[0][1]
    results.append(("top 首层 in_features", up_in == our_in, f"{up_in} vs {our_in}"))

    # 4 top 形状
    up_top = shapes(up.over_arch)
    our_top = shapes(ours.top_l)
    results.append(("top 形状", up_top == our_top, f"{up_top} vs {our_top}"))

    # 5 稠密侧参数量
    up_n = sum(p.numel() for p in up.parameters() if "sparse_arch" not in _name_of(up, p))
    our_n = sum(p.numel() for p in ours.parameters())
    results.append(("稠密侧参数量", up_n == our_n, f"{up_n:,} vs {our_n:,}"))

    print(f"{'判定项':22}{'结果':6}细节")
    ok = True
    for name, good, detail in results:
        ok &= bool(good)
        print(f"{name:22}{'PASS' if good else 'FAIL':6}{detail}")
    print("\n总体:", "PASS —— 与 torchrec.models.dlrm.DLRM 对齐" if ok else "FAIL —— 存在结构差异")
    return 0 if ok else 1


def _name_of(model, param):
    for n, p in model.named_parameters():
        if p is param:
            return n
    return ""


if __name__ == "__main__":
    sys.exit(main())
