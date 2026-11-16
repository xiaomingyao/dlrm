# coding=utf-8
"""DLRM 模型 —— RecSDK v2 (torch_rec_v2 / dynamic_emb) 实现。

结构与官方 recsdk-7.1.RC1/examples/dlrm/model/model.py 严格对齐：
  bottom_stack : dense(13) -> 512 -> 256 -> emb_dim  (relu, L1 正则 1e-2)
  dot 交互     : matmul(x, x^T) 取严格上三角(不含对角) 成对点积 -> C(n_vec,2) 维
  top_stack    : (emb_dim + C(n_vec,2)) -> 1024 -> 1024 -> 512 -> 256 -> 1

对齐基准：torchrec.models.dlrm.DLRM（facebookresearch/dlrm torchrec_dlrm 用的就是它）
  - InteractionArch: triu_indices(F, F, offset=1)，F = num_sparse + 1 = 27 -> 351 维
  - DenseArch: 13 -> 512 -> 256 -> 128，每层带 relu（含末层）
  - OverArch : (128 + 351) -> 1024 -> 1024 -> 512 -> 256 -> 1，末层无激活

稀疏侧用 TorchRec EmbeddingBagCollection，由 dynamic_emb 提供 NPU 上的
DynamicEmbeddingBagCollectionSharder（HKV 动态稀疏表），
不复用 nn.EmbeddingBag —— 后者反向 _embedding_bag_backward 在 torch_npu 上无实现。
"""

import os

import torch
import torch.nn as nn
import torchrec


# ----------------------------------------------------------------------------
# 稠密侧
# ----------------------------------------------------------------------------
class DenseMLP(nn.Module):
    """bottom_stack / top_stack 共用的 MLP 构造器。

    初始化对齐 zoo 的 tf.variance_scaling_initializer(mode="fan_avg", distribution="normal")：
      tf fan_avg normal -> std = sqrt(scale / ((fan_in + fan_out) / 2))，scale=1.0
    PyTorch 默认的 kaiming_uniform_(a=sqrt(5)) 实际 std ≈ 1/sqrt(fan_in)，
    比 fan_avg 小约 sqrt(2) 倍，会把 logits 压在 0 附近、梯度极弱。
    """

    def __init__(self, dims, last_activation=True, l1_lambda=1e-2):
        super().__init__()
        layers = []
        lin_idx = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            # 对齐 tf.variance_scaling_initializer(mode="fan_avg", distribution="normal")
            fan_avg = (dims[i] + dims[i + 1]) / 2.0
            std = (1.0 / fan_avg) ** 0.5
            with torch.no_grad():
                lin.weight.normal_(0.0, std)
                lin.bias.zero_()
            layers.append(lin)
            lin_idx.append(len(layers) - 1)
            is_last = i == len(dims) - 2
            if not is_last or last_activation:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._lin_idx = lin_idx
        # L1 正则沿 zoo 的 kernel_regularizer=l1_regularizer(1e-2)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        return self.net(x)

    def l1_penalty(self):
        """只对第一个 Linear 的权重施加 L1，与 zoo 的 kernel_regularizer 语义一致。"""
        w = self.net[self._lin_idx[0]].weight
        return self.l1_lambda * w.abs().sum()


def dot_interaction(x):
    """DLRM 特征交互：x (B, F, D) -> (B, C(F,2))，严格上三角（不含对角）。

    对齐 torchrec.models.dlrm.InteractionArch（上游 torchrec_dlrm 的唯一 dot 实现）：
      xactions = matmul(x, x^T)                                  # (B,F,F)
      idx      = triu_indices(F, F, offset=1)                     # 严格上三角，C(F,2) 对
      return xactions[:, idx[0], idx[1]]                          # (B, C(F,2))
    与旧实现的差别：旧版把含对角上三角置零后展平成 F*F 维（F=27 时 729 维，
    其中 378 维恒为 0），与上游的 351 维不一致，导致 top 首层宽度对不上。
    """
    if x.dim() != 3:
        raise ValueError(f"dot_interaction 期望 (B,F,D)，得到 {tuple(x.shape)}")
    b, f, _ = x.shape
    idx = torch.triu_indices(f, f, offset=1, device=x.device)
    xactions = torch.bmm(x, x.transpose(1, 2))
    return xactions[:, idx[0], idx[1]]


# ----------------------------------------------------------------------------
# 稀疏侧 + 整体模型
# ----------------------------------------------------------------------------
class DLRMv2(nn.Module):
    """DLRM：dense MLP + TorchRec EBC + dot 交互 + top MLP。

    稀疏侧用 EmbeddingBagCollection，供 DistributedModelParallel 切分成
    HKV 动态稀疏表（RecSDK v2 的 dynamic_emb sharder）。
    """

    def __init__(self, num_dense, sparse_configs, emb_dim, bot_dims, top_dims,
                 num_sparse=None, ec_mode=False, overlap_dense=False):
        super().__init__()
        self.ec_mode = bool(ec_mode)
        self.overlap_dense = bool(overlap_dense)
        self._side = None
        self.num_dense = num_dense
        # 统一表模式下一张表承载全部特征，特征数必须外部给定（见 build_unified_sparse_config）
        self.num_sparse = len(sparse_configs) if num_sparse is None else int(num_sparse)
        self.emb_dim = emb_dim
        # n_vec = num_sparse + 1（dense embedding 向量也算一个"特征"）
        self.n_vec = self.num_sparse + 1
        # 严格上三角成对点数 = C(n_vec, 2)，与 torchrec.models.dlrm.InteractionArch 一致
        self.inter_dim = self.n_vec * (self.n_vec - 1) // 2

        # bottom: dense -> emb_dim
        self.bot_l = DenseMLP([num_dense] + list(bot_dims) + [emb_dim], last_activation=True)
        # top: [emb_dim, inter_dim] -> 1（最后一层不带激活）
        self.top_l = DenseMLP([emb_dim + self.inter_dim] + list(top_dims) + [1],
                              last_activation=False)

        # meta device 上建表，交给 planner 决定落到哪
        if self.ec_mode:
            # EC：不做 pooling，每个值返回自己的向量 ⇒ 单特征(每样本 26 值) 时输出仍是 (B,26,D)
            self.ec = torchrec.EmbeddingCollection(
                device=torch.device("meta"),
                tables=sparse_configs,
            )
        else:
            self.ebc = torchrec.EmbeddingBagCollection(
                device=torch.device("meta"),
                tables=sparse_configs,
            )

    def forward(self, dense_x, sparse_features):
        """
        dense_x        : (B, num_dense) float
        sparse_features: KeyedJaggedTensor
        """
        if self.overlap_dense:
            # bottom MLP 只依赖稠密特征，可与稀疏侧 all2all/查表并行；用独立 stream 发起，
            # 在拼接前用 event 对齐。设备空转 64% 的场景里，这段能盖住部分通信时间。
            import torch as _t
            if self._side is None:
                self._side = _t.npu.Stream()
            _ev0 = _t.npu.Event()
            _ev0.record(_t.npu.current_stream())
            with _t.npu.stream(self._side):
                self._side.wait_event(_ev0)
                _bot = self.bot_l(dense_x)
                _ev1 = _t.npu.Event()
                _ev1.record(self._side)
        else:
            _bot = None
            _ev1 = None

        if self.ec_mode:
            ec_out = self.ec(sparse_features)
            # EC 返回 KeyedTensor：.values() 是 {feature: tensor}（单特征时只有一个）
            _v = ec_out.values()
            if isinstance(_v, dict):                  # KeyedTensor: {feature: JaggedTensor}
                _v = next(iter(_v.values()))
            elif not hasattr(_v, "view") and hasattr(_v, "__iter__") and not isinstance(_v, tuple):
                try:
                    _v = next(iter(_v))               # dict_values
                except StopIteration:
                    pass
            if hasattr(_v, "values") and not isinstance(_v, torch.Tensor):
                _v = _v.values()                      # JaggedTensor → 扁平 (B*num_sparse, D)
            flat = _v
            # flat: (B*num_sparse, emb_dim)，行主序
        else:
            ebc_out = self.ebc(sparse_features)
            flat = ebc_out.values()                             # (B, num_sparse*emb_dim)
        emb_out = flat.view(-1, self.num_sparse, self.emb_dim)

        if _bot is not None:
            torch.npu.current_stream().wait_event(_ev1)         # 等 side stream 上的 bottom MLP
            dense_emb = _bot
        else:
            dense_emb = self.bot_l(dense_x)                     # (B, D)
        feats = torch.cat([dense_emb.unsqueeze(1), emb_out], dim=1)  # (B, F, D)
        inter = dot_interaction(feats)                          # (B, F*F)
        top_in = torch.cat([dense_emb, inter], dim=1)           # (B, D + F*F)
        logits = self.top_l(top_in).squeeze(1)                  # (B,)
        return logits


def build_sparse_configs(num_sparse, num_embeddings_per_table, emb_dim, prefix="t"):
    """构造每个稀疏特征一张表的 EmbeddingBagConfig 列表。"""
    return [
        torchrec.EmbeddingBagConfig(
            name="%s_%d" % (prefix, i),
            embedding_dim=emb_dim,
            num_embeddings=num_embeddings_per_table[i],
            feature_names=["cate_%d" % i],
        )
        for i in range(num_sparse)
    ]


def build_unified_sparse_config(num_sparse, total_rows, emb_dim, name="t_unified"):
    """统一表：26 个特征共用一张表（id 由数据侧加 per-feature 偏移）。

    动机（实测 A2 档 42.90 ms/步）：每表一次 lookup_backward_v2，24 次 × 615 us
    = 15.05 ms，占 35%。合并成一张表后每步只剩 1 次查表，且 collective 分组数
    同步减少。等价性：各特征 id 落在互不重叠的行区间，行间无耦合，数学上与原
    26 表完全一致（仅随机初始化顺序不同）。
    """
    return [
        torchrec.EmbeddingBagConfig(
            name=name,
            embedding_dim=emb_dim,
            num_embeddings=int(total_rows),
            feature_names=["cate_%d" % i for i in range(num_sparse)],
        )
    ]


def build_unified_ec_config(num_sparse, total_rows, emb_dim, name="t_unified", feature="cate_all"):
    """统一表 + EC 模式：num_sparse 个特征打包成 1 个特征（每样本 num_sparse 个值）。

    动机（stage1 实测 A2 档每步）：UBDMA 264 次/步、NOTIFY_WAIT 343 次/步 —— all2all 被
    26 个特征切碎。打包成 1 个特征后分块数应降到个位数；EC 路径还支持 use_index_dedup，
    按唯一 id 收发，进一步压缩通信与查表行数。
    """
    return [
        torchrec.EmbeddingConfig(
            name=name,
            embedding_dim=emb_dim,
            num_embeddings=int(total_rows),
            feature_names=[feature],
        )
    ]
