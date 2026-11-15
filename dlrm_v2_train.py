#!/usr/bin/env python3
# coding=utf-8
"""DLRM 训练入口 —— RecSDK v2 (torch_rec_v2 / dynamic_emb)。

关键差异（相对官方 dlrm_s_pytorch.py）：
  稀疏侧走 EmbeddingBagCollection + DynamicEmbeddingBagCollectionSharder，
  由 dynamic_emb 提供 NPU 上的 HKV 动态稀疏表算子，
  不经过 nn.EmbeddingBag 的 _embedding_bag_backward（torch_npu 无该实现，会回退 CPU）。

用法（单卡冒烟）：
  torchrun --nproc_per_node=1 dlrm_v2_train.py --steps 30

用法（8 卡）：
  torchrun --nproc_per_node=8 dlrm_v2_train.py --steps 200
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import torchrec
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamic_emb.distributed.embeddingbag import DynamicEmbeddingBagCollectionSharder
from dynamic_emb.distributed.embedding import DynamicEmbeddingCollectionSharder
from dynamic_emb.distributed.planner.enumerators import DynamicEmbeddingEnumerator
from dynamic_emb.distributed.planner.planners import DynamicEmbeddingShardingPlanner
from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import ShardingType

from criteo_dataset import CriteoDataset, _next_pow2
from dlrm_v2_model import (DLRMv2, build_sparse_configs, build_unified_sparse_config,
                           build_unified_ec_config)


def make_kt(batch_size, num_sparse, vocab_sizes, device, seed=None):
    """构造 KeyedJaggedTensor：每个稀疏特征 multi-hot=1（对齐 Criteo 单值类别特征）。"""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    values = []
    lengths = []
    for i in range(num_sparse):
        ids = torch.randint(0, vocab_sizes[i], (batch_size,), generator=g, dtype=torch.long)
        values.append(ids)
        lengths.append(torch.ones(batch_size, dtype=torch.long))
    values = torch.cat(values).to(device)
    lengths = torch.cat(lengths).to(device)
    keys = ["cate_%d" % i for i in range(num_sparse)]
    return torchrec.KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)


def make_batch(batch_size, num_sparse, num_dense, vocab_sizes, device, seed):
    """构造**可学**的合成 batch，使 loss 能被观察到下降。

    纯随机 label 与特征无关，任何模型都只能学到 ln2≈0.693，看不出收敛。
    这里让 label 由特征真实决定，且保证信号强度足以在数十步内体现：

        只用每张表前 1000 个 id 的「前后半段」做区分，
        hot_i = 1 if id_i < vocab_i * HOT_FRAC else -1
        score = mean_i(hot_i) * GAIN + dense[:,0] - dense[:,1]
        label = 1 if score > 0

    之前失败的原因记录（避免重复踩）：
      - 若 label 用 `id < vocab/2` 且 id 取遍全表，正负样本各覆盖一半 id 且
        训练/测试同分布，梯度在表内互相抵消，loss 永远停在 0.693。
      - 必须把判别信息集中在少数 id 段上，HKV 才能学到"这些 id 是正例"。
    """
    HOT_FRAC = 0.5    # 一半 id 为正、一半为负，保证标签平衡
    GAIN = 3.0

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # 稀疏 id：散布全表（模拟真实长尾）
    ids_list = [
        torch.randint(0, vocab_sizes[i], (batch_size,), generator=g, dtype=torch.long)
        for i in range(num_sparse)
    ]
    values = torch.cat(ids_list).to(device)
    lengths = torch.ones(batch_size * num_sparse, dtype=torch.long, device=device)
    keys = ["cate_%d" % i for i in range(num_sparse)]
    kt = torchrec.KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)

    # dense
    dense_x = torch.randn(batch_size, num_dense, generator=g).to(device)

    # 标签：**只**由稀疏特征决定。
    # 不能让 dense 参与 —— 一旦 dense 权重压倒稀疏信号，模型会优先拟合 dense，
    # 但 bottom MLP 容量有限又拟合不好，结果两头都学不动（loss 恒 0.693）。
    score = torch.zeros(batch_size)
    for i, ids in enumerate(ids_list):
        thr = max(1, int(vocab_sizes[i] * HOT_FRAC))
        hot = torch.where(ids < thr, 1.0, -1.0)
        score += hot * (1.0 if i % 2 == 0 else -1.0)
    score = score * GAIN / num_sparse
    label = (score > 0).float().to(device)

    return dense_x, kt, label


def build_planner(world_size, compute_device, eb_configs, batch_size, hbm_bytes):
    """每张表都用 ROW_WISE + dynamic_emb（HKV 动态表）。

    初始化用 UNIFORM [-1/sqrt(n), 1/sqrt(n)]，对齐 zoo 里
    np.random.uniform(low=-sqrt(1/n), high=sqrt(1/n))。
    不能用 DEBUG 模式 —— 那是测试用的调试值生成器，会产出 1e9 量级的异常数。

    ⚠ 关键参数是 local_hbm_for_values（每卡本地 HBM 字节数），默认 0
    会导致表无法驻留、lookup 全返回 0（loss 恒等于 ln2）。
    global_hbm_for_values 留 0，由 planner 按「值字节×行数(对齐2的幂)×列数」自动推算。
    """
    constraints = {}
    for cfg in eb_configs:
        bound = 1.0 / (cfg.num_embeddings ** 0.5)
        constraints[cfg.name] = DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            compute_kernels=["fused"],
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                local_hbm_for_values=hbm_bytes,
                global_hbm_for_values=0,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.UNIFORM,
                    lower=-bound,
                    upper=bound,
                ),
                safe_check_mode=DynamicEmbCheckMode.WARNING,
            ),
        )
    topology = Topology(world_size=world_size, compute_device=compute_device)
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology, batch_size=batch_size, constraints=constraints
    )
    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs, topology=topology, constraints=constraints,
        enumerator=enumerator,
    )


def parse_dims(s):
    return [int(x) for x in s.split("-") if x]


def main():
    p = argparse.ArgumentParser("DLRM RecSDK v2")
    # 规模
    p.add_argument("--num-sparse", type=int, default=26, help="稀疏表数（Criteo=26）")
    p.add_argument("--num-dense", type=int, default=13, help="稠密特征数（Criteo=13）")
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=100000, help="每张表行数")
    p.add_argument("--bot-dims", type=str, default="512-256", help="bottom MLP 隐层")
    p.add_argument("--top-dims", type=str, default="1024-1024-512-256", help="top MLP 隐层")
    # 训练
    p.add_argument("--batch-size", type=int, default=1024, help="全局 batch")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5, help="统计时跳过的前若干步")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--hbm-gb", type=float, default=4.0, help="每张表在 HBM 里预留的容量")
    p.add_argument("--unified-emb", action="store_true",
                   help="统一表：26 个特征共用一张表（id 加 per-feature 偏移），每步只 1 次查表")
    p.add_argument("--probe-graph", action="store_true",
                   help="可行性探针：尝试用 NPUGraph 捕获一步训练并 replay 3 次")
    p.add_argument("--probe-graph-at", type=int, default=3, help="第几步开始捕获（默认 3）")
    p.add_argument("--ec-mode", action="store_true",
                   help="EC 模式：26 特征打包成 1 个特征（每样本 26 值），配合 --unified-emb 使用")
    p.add_argument("--index-dedup", action="store_true",
                   help="EC 模式下按唯一 id 收发（use_index_dedup）")
    p.add_argument("--unified-align", choices=["pow2", "batch"], default="pow2",
                   help="统一表行数对齐方式：pow2=向上取 2 的幂（旧行为）；batch=对齐到每卡本地 batch")
    p.add_argument("--data-dir", type=str,
                   default=os.environ.get("DLRM_DATA_DIR", "./data/criteo_pt"))
    p.add_argument("--data", type=str, default="criteo", choices=["criteo", "synth"],
                   help="criteo=真实 Kaggle DAC 数据；synth=合成可学数据（调试用）")
    p.add_argument("--train-days", type=str, default="0,1,2,3,4,5")
    p.add_argument("--test-days", type=str, default="6")
    p.add_argument("--epochs", type=int, default=1,
                   help="criteo 模式下的 epoch 数；用了 --steps>0 则以 steps 为准")
    p.add_argument("--eval-every", type=int, default=0,
                   help="每 N 个 epoch 跑一次测试集；0=只在最后跑")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print-sharding", action="store_true")
    p.add_argument("--dist-type", type=str, default="",
                   choices=["", "continuous", "roundrobin"],
                   help="RW 分发的 key->rank 映射方式，留空=库默认(continuous)")
    p.add_argument("--frozen-data", action="store_true",
                   help="P0探针：冻结一个 batch 常驻 NPU，循环内不重新加载数据")
    p.add_argument("--csv", type=str, default="", help="把每步耗时写到 csv")
    args = p.parse_args()

    # ---- 分布式初始化 ----
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local_rank)
    device = torch.device("npu:%d" % local_rank)

    # ⚠ 不能用 hccl 做 object collective。
    # torchrec planner 的 collective_plan() 内部用 dist.broadcast_object_list()
    # 广播 ShardingPlan（Python 对象）。走 HCCL 时 torch_npu 的
    # HCCLUtils::create_config 会直接失败：
    #   RuntimeError: create_config:...HCCLUtils.cpp:140 device error type 0, code 19
    # 解法：主 PG 用 hccl（张量通信/all2all），另建一个 cpu:gloo 的 PG 专供
    # object collective；并把 gloo PG 设为默认 device 组。
    if world_size > 1:
        # PG 布局（2026-10-31 实测定型）：
        #   默认 PG = HCCL —— DDP 的 allreduce 与 planner 的 all2all 都是 NPU 张量，
        #     必须由 HCCL 承载。反例：init(gloo) 默认 + new_group(hccl) 会让 DDP
        #     拿到 gloo 组，报 "No backend type associated with device type npu"。
        #   子组 gloo = 专供 **CPU 张量**的 object collective：
        #     - planner.collective_plan() 的 broadcast_object_list()
        #     - DDP._verify_param_shape_across_processes() 元数据广播
        #     这两者对 CPU 张量做集合通信，落在 HCCL 上必崩 code 19。
        dist.init_process_group(backend="hccl")
        _hccl_pg = dist.GroupMember.WORLD
        _cpu_pg = dist.new_group(backend="gloo")
    else:
        dist.init_process_group(backend="hccl", rank=0, world_size=1)
        _hccl_pg = dist.GroupMember.WORLD
        # 单卡无跨进程 object collective，但 collective_plan 仍要求传一个
        # 合法 ProcessGroup（传 None 会 raise ValueError: pg can be dist.ProcessGroup）
        _cpu_pg = dist.GroupMember.WORLD

    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    # ---- 数据 ----
    # local_bs_pre：每张表的 num_embeddings 需按 local batch 对齐（见 align_cardinalities）
    local_bs_pre = args.batch_size // world_size
    # 真实 Criteo 每张表的 cardinality 不同（2~7546），必须逐表配置，
    # 不能像合成数据那样统一用 --vocab-size，否则小表会浪费大量 HBM、大表会越界。
    ds = None
    if args.data == "criteo":
        train_days = tuple(int(x) for x in args.train_days.split(",") if x)
        test_days = tuple(int(x) for x in args.test_days.split(",") if x)
        ds = CriteoDataset(args.data_dir, train_days=train_days,
                           test_days=test_days, verbose=is_main)
        # ⚠ 必须按 batch_size 对齐：dynamic_emb 要求 num_embeddings % batch_size == 0，
        # 否则 lookup kernel 内部累加错位 → 进程挂死（不抛异常，极难排查）。
        vocab_sizes = ds.align_cardinalities(local_bs_pre)
        if is_main:
            bad = [(f, c) for f, c in enumerate(vocab_sizes) if c % local_bs_pre != 0]
            print("[align] batch_size=%d  aligned rows=%d  raw rows=%d  (+%.1f%%)  violations=%d"
                  % (local_bs_pre, sum(vocab_sizes), sum(ds.raw_cardinalities),
                     100.0 * (sum(vocab_sizes) / sum(ds.raw_cardinalities) - 1), len(bad)),
                  flush=True)
        if len(vocab_sizes) != args.num_sparse:
            raise ValueError("数据字段数 %d != --num-sparse %d"
                             % (len(vocab_sizes), args.num_sparse))
        num_dense_actual = ds.Xi_tr.shape[1]
        if num_dense_actual != args.num_dense:
            raise ValueError("稠密特征数 %d != --num-dense %d"
                             % (num_dense_actual, args.num_dense))
    else:
        vocab_sizes = [args.vocab_size] * args.num_sparse

    if args.ec_mode and not args.unified_emb:
        raise SystemExit("--ec-mode 必须与 --unified-emb 同用（单特征 + 统一表）")
    ds.single_feature = bool(args.ec_mode)
    if args.unified_emb:
        total_rows = ds.enable_unified_ids(vocab_sizes)
        if args.unified_align == "pow2":
            rows_aligned = _next_pow2(total_rows)
        else:
            a = local_bs_pre
            rows_aligned = ((total_rows + a - 1) // a) * a
        if args.ec_mode:
            eb_configs = build_unified_ec_config(args.num_sparse, rows_aligned, args.emb_dim)
        else:
            eb_configs = build_unified_sparse_config(
                args.num_sparse, rows_aligned, args.emb_dim)
        log("[unified] 26 表合 1：raw=%d 对齐后=%d（%s，+%.1f%%）"
            % (total_rows, rows_aligned, args.unified_align,
               100.0 * (rows_aligned / total_rows - 1)))
    else:
        eb_configs = build_sparse_configs(args.num_sparse, vocab_sizes, args.emb_dim)

    # ---- planner ----
    # local_hbm_for_values 是「每张表、每卡」的本地 HBM 配额，不是总量。
    # 26 张表共用 hbm_gb 的总预算，需要均分，否则会超出单卡 HBM（96GB）。
    # ⚠ planner 的 batch_size 必须是**每卡 local batch**，不是全局 batch。
    # 传全局会让 enumerator 算出 8 倍于实际的 per-rank 表容量，导致
    # num_embeddings//batch_size 的整除约束按错误的 batch 校验。
    if args.unified_emb:
        per_table_bytes = int(args.hbm_gb * 1024 ** 3)          # 单表，整包预算
    else:
        per_table_bytes = int(args.hbm_gb * 1024 ** 3 / args.num_sparse)
    planner = build_planner(world_size, device.type, eb_configs, local_bs_pre, per_table_bytes)

    fused_params = {
        "output_dtype": SparseType.FP32,
        "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        "learning_rate": args.lr,
        "eps": 1e-10,
    }
    if args.dist_type:
        fused_params["dist_type"] = args.dist_type
        log("[dist_type] =", args.dist_type)
    if args.ec_mode:
        sharder = DynamicEmbeddingCollectionSharder(
            fused_params=fused_params, use_index_dedup=bool(args.index_dedup))
        log("[ec-mode] 单特征打包 + use_index_dedup=%s" % bool(args.index_dedup))
    else:
        sharder = DynamicEmbeddingBagCollectionSharder(fused_params=fused_params)

    model_ebc = DLRMv2(
        num_dense=args.num_dense,
        sparse_configs=eb_configs,
        num_sparse=args.num_sparse,
        ec_mode=args.ec_mode,
        emb_dim=args.emb_dim,
        bot_dims=parse_dims(args.bot_dims),
        top_dims=parse_dims(args.top_dims),
    )

    # ⚠ collective_plan 内部走 broadcast_object_list（CPU 张量）⇒ 必须用 gloo 子组，
        #   用 dist.GroupMember.WORLD(HCCL) 会崩 code 19。
    _eb_module = model_ebc.ec if args.ec_mode else model_ebc.ebc
    plan = planner.collective_plan(_eb_module, [sharder], _cpu_pg)

    # collective_plan 对传入的 EBC 生成的 key 是 ""（视作顶层）；
    # 但 DMP 会从 DLRMv2 顶层遍历，子模块 ebc 的路径是 "ebc"。
    # 两者必须对齐，否则 DMP 会在顶层用 DLRMv2 类型去查 _sharder_map → KeyError。
    _eb_key = "ec" if args.ec_mode else "ebc"
    if "" in plan.plan:
        plan.plan[_eb_key] = plan.plan.pop("")

    if is_main and args.print_sharding:
        for ckey, plans in plan.plan.items():
            print("collection:", ckey)
            for tname, tp in plans.items():
                print("  ", tname, tp)

    model = DistributedModelParallel(
        module=model_ebc,
        device=device,
        sharders=[sharder],
        plan=plan,
    )

    # ---- 优化器：稠密参数用 SGD（稀疏参数已在 sharder 的 fused optimizer 里）----
    dense_params = [p_ for n, p_ in model.named_parameters() if "ebc" not in n]
    optimizer = torch.optim.SGD(dense_params, lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    log("=== DLRM RecSDK v2 ===")
    log("world_size   =", world_size)
    log("num_sparse   =", args.num_sparse, " num_dense =", args.num_dense)
    log("emb_dim      =", args.emb_dim)
    log("n_vec        =", args.num_sparse + 1, " inter_dim =", (args.num_sparse + 1) ** 2)
    log("bot_dims     =", [args.num_dense] + parse_dims(args.bot_dims) + [args.emb_dim])
    log("top_dims     =", [args.emb_dim + (args.num_sparse + 1) ** 2] + parse_dims(args.top_dims) + [1])
    log("dense params =", sum(p_.numel() for p_ in dense_params))
    log("batch(global)=", args.batch_size, " local =", args.batch_size // world_size)
    log("steps        =", args.steps)

    local_bs = args.batch_size // world_size

    # ---- 训练 / 评估 ----
    def run_eval():
        """测试集评估：返回 (loss, acc, auc)。

        Criteo 正例率 22.6%，"全预测为负"的 acc 就有 77.4% —— acc 在此不可用，
        必须看 AUC（排序能力，随机=0.5）。AUC 用秩统计手算，避免依赖 sklearn。
        """
        model.eval()
        all_p = []
        all_y = []
        tot_loss = 0.0
        tot_n = 0.0
        with torch.no_grad():
            for _bi, xi, kt_, yy in ds.iter_batches(
                    "test", args.batch_size, rank=rank, world_size=world_size,
                    device=device, shuffle=False, epoch=0, drop_last=True):
                lg = model(xi, kt_)
                l = criterion(lg, yy)
                n = yy.numel()
                tot_n += n
                tot_loss += float(l.item()) * n
                all_p.append(torch.sigmoid(lg).float().cpu())
                all_y.append(yy.cpu())
        model.train()
        p = torch.cat(all_p)
        y = torch.cat(all_y)
        acc = float(((p > 0.5).float() == y).float().mean())
        # AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        order = torch.argsort(p)
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(1, len(p) + 1, dtype=torch.float64)
        npos = float(y.sum())
        nneg = float(len(y) - npos)
        if npos > 0 and nneg > 0:
            auc = float((ranks[y > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg))
        else:
            auc = float("nan")
        return tot_loss / max(tot_n, 1.0), acc, auc

    model.train()
    step_times = []
    losses = []
    accs = []
    step_global = 0

    if ds is not None:
        # 真实数据：按 epoch × 全量 batch 训练
        epochs = args.epochs
        nb = ds.num_batches("train", args.batch_size, drop_last=True)
        total_steps = epochs * nb
        log("train batches/epoch = %d   total steps = %d" % (nb, total_steps))
        stop = False
        _frozen = None
        _gen = None
        if args.frozen_data:
            # P0 探针：预取一个 batch 的原始 numpy 切片，转成 NPU 常驻 tensor。
            # 复用同一个 tensor 对象，循环内零数据准备开销。
            _gen = ds.iter_batches("train", args.batch_size, rank=rank,
                                   world_size=world_size, device=device,
                                   shuffle=False, epoch=0, drop_last=True)
            _bi0, _xi0, _kt0, _yy0 = next(_gen)
            _frozen = (_xi0, _kt0.values(), _kt0.lengths(), _yy0,
                       list(_kt0.keys()))
            if is_main:
                print("[frozen-data] batch cached on NPU: xi=%s values=%s"
                      % (tuple(_xi0.shape), tuple(_kt0.values().shape)), flush=True)
        for ep in range(epochs):
            for bi, xi, kt_, label in ds.iter_batches(
                    "train", args.batch_size, rank=rank, world_size=world_size,
                    device=device, shuffle=True, epoch=ep, drop_last=True):
                if args.frozen_data:
                    _fxi, _fv, _fl, _fy, _fkeys = _frozen
                    xi = _fxi
                    kt_ = torchrec.KeyedJaggedTensor(keys=_fkeys, values=_fv, lengths=_fl)
                    label = _fy
                t0 = time.time()
                logits = model(xi, kt_)
                loss = criterion(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.probe_graph and step_global == args.probe_graph_at:
                    log("[probe-graph] 开始捕获（第 %d 步）" % step_global)
                    try:
                        _g = torch.npu.NPUGraph()
                        torch.npu.synchronize()
                        with torch.npu.graph(_g):
                            _lo = model(xi, kt_)
                            _l = criterion(_lo, label)
                            optimizer.zero_grad()
                            _l.backward()
                            optimizer.step()
                        torch.npu.synchronize()
                        log("[probe-graph] CAPTURE_OK，开始 replay")
                        _t0 = time.time()
                        for _ in range(3):
                            _g.replay()
                        torch.npu.synchronize()
                        log("[probe-graph] REPLAY %.3f ms/step" % ((time.time() - _t0) / 3 * 1000))
                    except Exception as _e:   # noqa: BLE001 —— 探针就是要报出原始异常
                        log("[probe-graph] CAPTURE_FAIL %s: %s"
                            % (type(_e).__name__, str(_e)[:400]))
                    stop = True
                    break

                torch.npu.synchronize()
                dt = time.time() - t0
                step_times.append(dt)
                losses.append(loss.item())

                with torch.no_grad():
                    acc = float(((torch.sigmoid(logits) > 0.5).float() == label)
                                .float().mean().item())
                accs.append(acc)

                if is_main and (step_global < 3 or (step_global + 1) % 50 == 0):
                    print("ep%d step %5d/%d  loss=%.6f acc=%.4f  %.2f ms"
                          % (ep, bi + 1, nb, loss.item(), acc, dt * 1000), flush=True)

                step_global += 1
                if args.steps > 0 and step_global >= args.steps:
                    stop = True
                    break
            # ⚠ eval 必须**所有 rank 都跑**：DMP 模型 forward 含跨 rank all2all，
            #   若只有 rank0 执行，rank0 会一直等其余 rank 配对 ⇒ 双向死锁
            #   （症状：其余 rank 报 TASK_NOTIFY_WAIT waiting peer rank:0，
            #     rank0 用户态空转 CPU~97%、无报错）。
            if args.eval_every and (ep + 1) % args.eval_every == 0:
                el, ea, eu = run_eval()
                if is_main:
                    print("[eval] epoch %d  test_loss=%.6f test_acc=%.4f test_auc=%.4f"
                          % (ep + 1, el, ea, eu), flush=True)
            if stop:
                log("early stop at --steps %d" % args.steps)
                break
    else:
        # 合成数据（调试用，保持原有行为）
        for step in range(args.steps):
            t0 = time.time()
            dense_x, kt, label = make_batch(
                local_bs, args.num_sparse, args.num_dense, vocab_sizes, device,
                seed=args.seed * 100000 + step,
            )
            logits = model(dense_x, kt)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.npu.synchronize()
            dt = time.time() - t0
            step_times.append(dt)
            losses.append(loss.item())
            accs.append(float(((torch.sigmoid(logits) > 0.5).float() == label)
                              .float().mean().item()))

            if is_main and (step < 3 or (step + 1) % 10 == 0 or step == args.steps - 1):
                print("step %4d  loss=%.6f  acc=%.4f  %.2f ms"
                      % (step + 1, loss.item(), accs[-1], dt * 1000), flush=True)

    # ---- 汇总 ----
    if world_size > 1:
        dist.barrier()

    if args.csv:
        try:
            with open(args.csv, "w") as _f:
                _f.write("step,loss,acc,ms\n")
                for _i, (_t, _l) in enumerate(zip(step_times, losses)):
                    _f.write("%d,%.6f,%s,%.4f\n" % (_i + 1, _l,
                              ("%.6f" % accs[_i]) if _i < len(accs) else "", _t * 1000))
            if is_main:
                print("[csv] wrote %s" % args.csv, flush=True)
        except Exception as _e:
            print("[csv] fail %s" % _e, flush=True)
    steady = step_times[args.warmup:]
    if is_main:
        import statistics as st
        print()
        print("=== RESULT ===")
        print("steps_total   = %d" % len(step_times))
        print("steady_steps  = %d (skip warmup %d)" % (len(steady), args.warmup))
        if steady:
            m = st.mean(steady)
            med = st.median(steady)
            print("step_time_ms  = mean=%.3f median=%.3f min=%.3f max=%.3f std=%.3f"
                  % (m * 1000, med * 1000, min(steady) * 1000, max(steady) * 1000,
                     st.pstdev(steady) * 1000))
            print("throughput    = %.1f samples/s (global)" % (args.batch_size / m))
        # loss / acc：首末 + 前 100 步均值 vs 末 100 步均值（比单点更稳）
        print("loss_first    = %.6f" % losses[0])
        print("loss_last     = %.6f" % losses[-1])
        print("loss_delta    = %.6f" % (losses[-1] - losses[0]))
        k = min(100, max(1, len(losses) // 2))
        lm_head = st.mean(losses[:k])
        lm_tail = st.mean(losses[-k:])
        print("loss_head%d    = %.6f" % (k, lm_head))
        print("loss_tail%d    = %.6f" % (k, lm_tail))
        print("loss_improve  = %.6f (tail-head)" % (lm_tail - lm_head))
        print("acc_first     = %.4f" % accs[0])
        print("acc_last      = %.4f" % accs[-1])
        print("acc_head%d     = %.4f" % (k, st.mean(accs[:k])))
        print("acc_tail%d     = %.4f" % (k, st.mean(accs[-k:])))

        # 每步明细导出（供报告画图；仅 rank0 落盘）
        try:
            import os as _os
            _csv = _os.environ.get("STEP_CSV", "")
            if _csv:
                with open(_csv, "w") as _f:
                    _f.write("step,loss,acc,ms\n")
                    for _i in range(len(step_times)):
                        _f.write("%d,%.6f,%.6f,%.4f\n"
                                 % (_i + 1, losses[_i], accs[_i],
                                    step_times[_i] * 1000.0))
                print("step_csv      = %s" % _csv)
        except Exception as _e:
            print("step_csv_error = %r" % (_e,))

        # 判据：loss_tail 明显低于 loss_head（用相对降幅 >1% 过滤噪声）
        rel = (lm_tail - lm_head) / max(abs(lm_head), 1e-9)
        ok = rel < -0.01
        print("RESULT: %s (loss rel_change=%.2f%%)"
              % ("PASS" if ok else "FAIL(loss未实质下降)", rel * 100))

    # ---- 最终 test eval：必须所有 rank 执行（DMP 前向含跨 rank all2all）----
    # 仅打印由 is_main 门控。此前该段误置于 is_main 块内导致 8 卡退出前死锁。
    if ds is not None:
        el, ea, eu = run_eval()
        if is_main:
            print("test_loss     = %.6f" % el)
            print("test_acc      = %.4f  (全负基线 %.4f)"
                  % (ea, float((ds.y_te == 0).mean())))
            print("test_auc      = %.4f  (随机=0.5)" % eu)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
