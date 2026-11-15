# coding=utf-8
"""Criteo (Kaggle DAC / Terabyte 同源) 真实数据加载器 —— 给 RecSDK v2 DLRM 用。

数据来源：$DLRM_DATA_DIR/dac_sample_day_{i}_processed.npz
  X_cat (N, 26) float64   —— 类别特征，已是整数 rank（0 基）
  X_int (N, 13) int32     —— 稠密特征，原版 Criteo 就是整数计数
  y     (N,)    int32     —— 0/1 标签，正例率约 22%

划分：day 0-5 训练、day 6 测试（按天切分，避免同日数据泄漏）。
全部数据常驻内存（10 万行 × 39 特征 ≈ 40MB），不做 cache 文件、不碰 caffe2 老管线。

关键约束：
  - X_cat 每列的 cardinality 不同（2 ~ 7546），必须逐列取 max+1 作为表行数。
  - 表行数取 2 的幂向上对齐，HKV 动态表对非 2 幂容量会有内部对齐放大。
  - 索引一律 int64，KeyedJaggedTensor 要求。
"""

import os

import numpy as np
import torch
import torch_npu  # noqa: F401  必须在 torchrec 之前导入，torchrec 依赖 torch_npu 的私有后端
import torchrec

import os as _os
DATA_DIR_DEFAULT = _os.environ.get("DLRM_DATA_DIR", "./data/criteo_pt")
NUM_FIELDS = 26
NUM_DENSE = 13


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def load_days(data_dir, days, verbose=True):
    """加载指定 day 的 npz，返回 (X_int, X_cat, y)。"""
    Xi, Xc, Y = [], [], []
    for d in days:
        f = os.path.join(data_dir, "dac_sample_day_%d_processed.npz" % d)
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        z = np.load(f)
        Xi.append(z["X_int"])
        Xc.append(z["X_cat"])
        Y.append(z["y"])
        if verbose:
            print("  loaded day%d: N=%d y_mean=%.4f" % (d, len(z["y"]), float(z["y"].mean())),
                  flush=True)
    return (np.concatenate(Xi, 0), np.concatenate(Xc, 0), np.concatenate(Y, 0))


class CriteoDataset:
    """把 Criteo 全量数据放在内存里，按 batch 切分。

    X_cat 保持 float64 存储（npz 原样），取 batch 时转 int64。
    不预先生成 .npz cache —— 10 万行规模下预处理耗时可忽略，
    而 cache 会让「数据从哪来」变得不可追溯。
    """

    def __init__(self, data_dir=DATA_DIR_DEFAULT, train_days=(0, 1, 2, 3, 4, 5),
                 test_days=(6,), verbose=True):
        self.data_dir = data_dir
        if verbose:
            print("[data] loading train days %s" % (list(train_days),), flush=True)
        self.Xi_tr, self.Xc_tr, self.y_tr = load_days(data_dir, train_days, verbose)
        if verbose:
            print("[data] loading test days %s" % (list(test_days),), flush=True)
        self.Xi_te, self.Xc_te, self.y_te = load_days(data_dir, test_days, verbose)

        # 真实 cardinality（不 2 幂对齐，保留原始值用于报告）
        self.raw_cardinalities = []
        for f in range(NUM_FIELDS):
            cmax = int(max(self.Xc_tr[:, f].max(), self.Xc_te[:, f].max()))
            self.raw_cardinalities.append(cmax + 1)
        # 实际表行数由 align_cardinalities() 按 batch_size 决定
        self.cardinalities = list(self.raw_cardinalities)

        if verbose:
            print("[data] train N=%d  test N=%d" % (len(self.y_tr), len(self.y_te)), flush=True)
            print("[data] cardinalities (2-pow aligned): %s" % self.cardinalities, flush=True)
            print("[data] total table rows       = %d" % sum(self.cardinalities), flush=True)
            print("[data] total table bytes @128d = %.2f GB" %
                  (sum(self.cardinalities) * 128 * 4 / 1024 ** 3), flush=True)

    def align_cardinalities(self, batch_size):
        """把每张表的行数向上对齐到 batch_size 的整数倍。

        ⚠ dynamic_emb 的硬约束（batched_dynamicemb_function.py:192 附近）：
            num_embeddings % batch_size == 0
          否则 lookup kernel 内部 accum_D 累加错位、越界，表现为 **进程挂死**
          （AICore 0%、CPU 100%+ 忙等），不是抛异常。

        Criteo 的 field 8/19 真实 cardinality 只有 3，远小于 batch_size，
        必须对齐。对齐代价：387K -> 377K 行仅 +2.6%（batch=1024）。
        填充行永远不会被 lookup 到（id < raw_cardinality），是纯占位。
        """
        aligned = []
        for c in self.raw_cardinalities:
            a = ((c + batch_size - 1) // batch_size) * batch_size
            aligned.append(max(a, batch_size))
        self.cardinalities = aligned
        return aligned

    def enable_unified_ids(self, vocab_sizes):
        """为统一表模式准备 per-feature id 偏移（各特征落在互不重叠的行区间）。

        返回总行数（未对齐）；调用方负责向上取 2 的幂后建表。
        """
        off = np.zeros(NUM_FIELDS, dtype=np.int64)
        acc = 0
        for f in range(NUM_FIELDS):
            off[f] = acc
            acc += int(vocab_sizes[f])
        self.offsets = torch.from_numpy(off)
        self.unified_total = acc
        return acc

    def num_batches(self, split, batch_size, drop_last=True):
        n = len(self.y_tr) if split == "train" else len(self.y_te)
        return n // batch_size if drop_last else (n + batch_size - 1) // batch_size

    def iter_batches(self, split, batch_size, rank=0, world_size=1, device="cpu",
                     shuffle=True, epoch=0, drop_last=True):
        """按 rank 切分 + 可选 shuffle 地迭代 batch。

        多卡：先把整个 split 按 batch 切成 global batch，再将每个 global batch 的
        行按 rank 均分（等价于 DDP 的 sample 并行语义），保证各 rank 看到不重叠数据。
        """
        if split == "train":
            Xi, Xc, Y = self.Xi_tr, self.Xc_tr, self.y_tr
        else:
            Xi, Xc, Y = self.Xi_te, self.Xc_te, self.y_te
        n = len(Y)

        nb = n // batch_size if drop_last else (n + batch_size - 1) // batch_size
        order = np.arange(n)
        if shuffle and split == "train":
            rng = np.random.RandomState(1234 + epoch)
            rng.shuffle(order)

        local_bs = batch_size // world_size
        for bi in range(nb):
            idx = order[bi * batch_size:(bi + 1) * batch_size]
            # global batch 均分给各 rank
            lo = rank * local_bs
            hi = lo + local_bs
            idx = idx[lo:hi]
            if len(idx) == 0:
                continue

            # 稠密特征 log1p 变换：Criteo 原始计数跨度 0 ~ 1.6e6（col4 max=1618112,
            # 均值 1.6e4），直接送 MLP 会让 logits 爆到 1e2~1e13 并立刻 NaN。
            # 官方 dlrm_data_pytorch.py 同样用 log1p 处理 dense 特征。
            xi_np = np.log1p(Xi[idx].astype(np.float64))
            xi = torch.from_numpy(xi_np.astype(np.float32))
            xc = torch.from_numpy(Xc[idx].astype(np.int64))   # (B, 26)
            yy = torch.from_numpy(Y[idx].astype(np.float32))

            if device != "cpu":
                xi = xi.to(device)
                yy = yy.to(device)

            # KeyedJaggedTensor：26 个特征 × 每样本 1 个值，按特征拼接
            vals = [xc[:, f].contiguous() for f in range(NUM_FIELDS)]
            values = torch.cat(vals, 0)
            # 统一表模式：feature-major 布局（NUM_FIELDS × local_bs）+ per-feature 偏移
            if getattr(self, "offsets", None) is not None:
                off = self.offsets.to(values.device)
                values = (values.view(NUM_FIELDS, local_bs) + off.unsqueeze(1)).reshape(-1)
            if device != "cpu":
                values = values.to(device)
            # ⚠ lengths 必须是 int64（torch.long）**且** 与 values 同 device。
            #   实测（probe_table_size.py，2026-10-31）：
            #     dtype=int32  -> dynamic_emb lookup kernel 内部挂死（AICore 0%、
            #                     CPU 100% 忙等、不抛异常、无日志，极难定位）
            #     dtype=long   -> 0.06s 正常通过
            #   曾在 lengths 放到 NPU(int32) 时遇到 507035，误判为"必须留 CPU"，
            #   实际根因是 dtype 而非 device。留在 CPU 会因为 device 不一致而报
            #   另一个错，所以这里统一成 long + 同 device。
            lengths = torch.ones(local_bs * NUM_FIELDS, dtype=torch.long,
                                 device=values.device)
            keys = ["cate_%d" % f for f in range(NUM_FIELDS)]
            kt = torchrec.KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)

            yield bi, xi, kt, yy


if __name__ == "__main__":
    ds = CriteoDataset()
    for bi, x, kt, y in ds.iter_batches("train", 1024, epoch=0):
        print("batch", bi, "dense", tuple(x.shape), "values", tuple(kt.values().shape),
              "label_mean", float(y.mean()))
        print("  kt lengths[:26]", kt.lengths()[:26].tolist())
        print("  a few ids per field:",
              [int(kt.values()[i * 1024]) for i in range(0, 26, 5)])
        assert x.shape[0] == 1024 and y.shape[0] == 1024
        assert kt.values().shape[0] == 1024 * 26
        break
    print("OK")
