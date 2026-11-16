# dlrm950 —— 昇腾 950 上的 DLRM（HKV 动态表）

> 本仓为 DLRM 在昇腾 950DT 上的独立仓库（性能优化 SSOT）。运行环境、数据放置、同步流程见 `OPERATIONS.md`。

基于 RecSDK 自研稀疏侧（`dynamic_emb` + HKV 动态表）的 DLRM v2 实现。**本目录的目标不是"能跑"，而是把昇腾 950 做成 A100 的可替代训练平台**，因此按四条能力逐一给出目标、当前实测与差距。

## 0. 唯一真源（SSOT）

**本目录（GitHub `torch-examples/dlrm/`）是 dlrm950 的唯一真源**。集群里的运行目录（历史路径 `<数据盘>/dlrm950/v2/`）自此**只读**，不再直接修改。

| 事项 | 规则 |
|---|---|
| 改代码 | 只改本仓 → commit → 用 `tools/sync_to_node.sh <节点目录>` 单向推到运行目录（脚本内置两侧 md5 闸门，不一致即退出 1） |
| 跑实验 | 用同步后的节点副本运行；实验脚本/参数写在 commit message 或 `results/` 里，不留在节点上 |
| 历史产物 | 节点上已有的日志、csv、profile 是历史证据，保留原样、不回写本仓 |
| 节点缺文件 | 以本仓为准补齐；不允许在节点上"就地改一版"再想同步回来 |


## 项目目标与现状

| # | 目标 | 状态 | 关键数字 | 差距 / 下一步判据 |
|---|---|---|---|---|
| 1 | **移植**：DLRM 在 950 上可训练 | ✅ 单机 8 die 跑通 | 300 步无中断，loss 单调下降，test AUC 0.7525（150 步链路值） | — |
| 2 | **高性能**：对比 8×A100 更优 | 🟡 进行中 | 16384 档 42.90 ms / 378,798 sps = A100 的 **41%**；2048 档 24.26 ms / 82,988 sps = **41%** | 目标 ≤17.7 ms（超过 A100 925,696 sps）；先做 P1/P3 |
| 3 | **线性扩展**：单机 8 die → 多机 | 🔴 未验证 | 仅单机 8 die 有数据 | 判据：8→16 die 扩展比 ≥1.6（效率 ≥80%），见 §3 |
| 4 | **高性能 HKV** | 🟡 已量化，未优化 | 单步 `lookup_backward_v2_kernel` 15.05 ms（35%），24 次/步、615 µs/次 | 目标 ≤9 ms；判据见 §2.4 P1 |

A100 数据取自 facebook `torchrec_dlrm` README 的 Preliminary Training Results（8×A100 40GB、Criteo 1TB、collective emb 104.54 GiB、dot 交互）；本机用同一份 Criteo 1TB 日志的 112.3 M 行切片 + MLPerf 词表（≈104 GiB）复测。

## 目录

| 文件 | 说明 |
|---|---|
| `dlrm_v2_train.py` | 训练入口：建表（HKV 分片规划）、数据加载、训练/评测循环、逐步耗时统计 |
| `dlrm_v2_model.py` | 模型：bottom MLP(512-256) + 26 表 embedding + dot 交互 + top MLP(1024-1024-512-256) |
| `criteo_dataset.py` | 数据：读取 `dac_sample_day_{i}_processed.npz`，按 2 的幂对齐表容量 |
| `tools/p2_no_step_sync.py` | 生成"按 N 步同步"的版本（`--sync-every`），用于验证同步节奏 |
| `tools/parse_msprof.py` | 解析 msprof `op_summary*.csv`，按训练窗口输出每步预算 |

## 1. 移植（已达成）

| 项 | 内容 |
|---|---|
| 环境 | CANN 9.1.0 · Ascend950DT ×8 die（单 die 96 GiB HBM）· torch 2.7.1+cpu / torch_npu 2.7.1.post4 · torchrec 1.2.0+npu · fbgemm_ascend 1.2.0 · `dynamic_emb` |
| 启动 | `torchrun --nproc_per_node=8`，HCCL 建组 |
| 已解决的三个坑 | ① HCCL 不支持 object collective → planner 的 `broadcast_object_list` 走 gloo 子组；② 表容量需 `% batch_size == 0` 且向上对齐 2 的幂；③ `--dist-type` 决定 key→rank 映射（默认 continuous） |
| 验收 | 300 步无中断、loss 单调下降、失败即 `RESULT: FAIL` |

## 2. 高性能（进行中）

### 2.1 现状对 A100

口径：8 die、Criteo 1TB 切片（112.3 M 行）、每档 n=3 取 median、warmup 20 步、无 profiler；A100 为官方 8×A100 40GB 同表数据。

| 档 | global batch | 每卡本地 | 最佳配置 | 纯计算 median | 端到端¹ | 对 A100（端到端） | 2× 目标 | 差距 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| A1 | 2,048 | 256 | 统一表 + **EBC** | 6.70 ms | **7.36 ms** | **135.8%** | 5.00 ms | 32% |
| A2 | 16,384 | 2,048 | 统一表 + **EC + dedup** | 15.12 ms | **16.58 ms** | **106.8%** | 8.85 ms | 47% |
| A4 | 131,072 | 16,384 | 统一表 + **EC + dedup** | 56.60 ms | **62.30 ms** | **176.8%**¹ | 55.07 ms | 13% |

¹ A4 是 A100 官方第 4 档（DCN v2 + Adagrad，1,190,128 sps）；本实现只有 dot 交互，算力需求更低，故该行的百分比不是同模型对比，只作规模参考。

端到端 = 步时 + host 取数（A1 0.67 ms / A2 1.53 ms / A4 5.7 ms）。历史表中「纯计算 median」不含取数，与 A100 官方口径不完全对齐，故另列一列，避免高估；两档在端到端口径下仍都高于 A100 一倍线。


扩展曲线（每卡本地 batch 放大，统一表 + EC + dedup，150 步/次）：

| global batch | 每卡本地 | 端到端 ms | 吞吐 | 说明 |
|---:|---:|---:|---:|---|
| 2,048 | 256 | 7.36 | 278,111 sps | A1 锚，EBC 更优 |
| 16,384 | 2,048 | 16.23 | 1,009,000 sps | A2 锚 |
| 32,768 | 4,096 | 24.29 | 1,349,000 sps | 无 A100 对应档 |
| 65,536 | 8,192 | 36.76 | 1,783,000 sps | 无 A100 对应档 |
| 131,072 | 16,384 | 62.30 | 2,104,000 sps | A4 档（对照口径见脚注） |

结论：950 的优势在**大 batch**——固定开销被摊薄后到 A100 的 1.77×（端到端）；小 batch 受单步开销限制只有 1.36×。
`TASK_QUEUE_ENABLE=2` 在 A2 有 +2.7%~7% 的小幅收益，在 A4 略负，不设默认。原始数据见 `results/dlrm950_batch_scaling_anchor.txt`。

为什么按档选配置：EC + `use_index_dedup` 在大 batch 下把唯一 id 从 42.6 万压到约 12.6 万（A2 −19.8%）；小 batch 下排序开销大于通信节省（A1 9.89 ms vs EBC 6.78 ms）。EC 不带 dedup 时 A2 反而 62.4 ms，说明**收益来自 dedup，不是 EC 本身**。

进展：优化前 A1 21.19 ms（47.2%）、A2 40.27 ms（44.0%）；统一表 A1 −68.0%、A2 −49.5%；EC+dedup 再让 A2 −19.8%。原始数据见 `results/dlrm950_ec_dedup_anchor.txt`、`results/dlrm950_unified_emb_anchor.txt`。

硬件利用率（A2 实测，宿主采样）：AICore 2.9%、AIVector 0.6%、带宽 0%，折算约 17.7 TFLOP/s、19 GB/s ⇒ **瓶颈不是算力，而是每步有效工作量太小**（见 `profiling/stage1_utilization.md`）。

### 2.2 单步预算（msprof，16384 档）

| 项 | 每步 | 占 wall | 说明 |
|---|---:|---:|---|
| idle（设备空转） | 12.80 ms | 29.8% | 非设备执行时间 |
| HKV 反向 lookup | 15.05 ms | 35.1% | 24 次/步，615 µs/次 |
| HCCL 合计 | 9.51 ms | 22.2% | reduceScatter 3.57 / alltoall 2.17 / launch 1.95 |
| 其余 device | 5.54 ms | 12.9% | MatMul 1.31 / pooling 0.54 / Sort 0.52 |

### 2.3 已完成实验：同步节奏（P2）

| 配置 | sync 间隔 | median 步时 | 相对同脚本对照 |
|---|---:|---:|---:|
| 16,384 | 1（原始） | 45.23 ms | — |
| 16,384 | 2 / 4 / 8 | 44.06 / 44.38 / 44.92 ms | −2.6% / −1.9% / −0.7% |
| 16,384 | 50 | **67.43 ms** | **+49%（更差）** |
| 2,048 | 1（原始） | 23.69 ms | — |
| 2,048 | 4 | **22.00 ms** | **−7.1%** |

结论：per-step 同步不是 idle 的来源（1/2/4/8 差异 <1.2 ms）；但同步间隔拉到 50 步会让 host 超前下发、队列积压，反而慢 49% 且抖动变大（std 10.9 ms）。**"去掉同步" ≠ "流水起来"**。

### 2.4 下一步（按实测收益排序）

| 编号 | 动作 | 现状→目标 | 判据 |
|---|---|---|---|
| P1 | HKV 反向 lookup：`--dist-type` 对比、bucket 容量、batched-backward 能否替代 24 次逐表调用 | 15.05 → 9.0 ms | 该 kernel ≤9 ms 且调用次数 ≤8 |
| P3 | 训练步 aclgraph / npugraph_ex 捕获（静态 shape、无控制流） | idle 12.80 → 6.0 ms | 捕获成功且步时不退化 |
| P4 | HCCL：合并 6 次 launch（1.95 ms 纯启动开销）、allReduce+allGather 合一 | 9.51 → 6.0 ms | HCCL ≤6 ms |
| P5 | dense bf16 + L1 融合 | 1.31 → 0.8 ms | 最后做，收益小 |

目标终态：**26.5 ms / 617,332 sps（A100 的 67%）**；若每样本项同时压到 1.0 µs，可到 25.3 ms / 648,000 sps（70%）。

## 3. 线性扩展（未验证）

| 项 | 现状 | 方案 | 判据 |
|---|---|---|---|
| 单机 | 8 die 已验证 | — | — |
| 多机 | 未验证 | `torchrun --nnodes=N --node_rank=i --master_addr=...`；HCCL over RoCE/UB 需先确认网卡与 `ub_verbs` 配置 | 8→16 die 扩展比 ≥1.6；16→32 ≥1.7 |
| 风险 | — | ① planner 用 gloo 广播，跨机需 gloo 可达；② `global_hbm_for_values=0` 时按全局规模推算，跨机要复核 HBM 预算；③ 分片计划随 world_size 变化，容量整除约束需重算 | — |

## 4. 高性能 HKV（已量化）

机制：`DynamicEmbeddingBagCollectionSharder` 只物化被访问的 key，`local_hbm_for_values` 给每表本地 HBM 配额，`global_hbm_for_values=0` 由 planner 推算；容量取 `max+1` 向上对齐 2 的幂。

已排除的两个猜测（实测）：

| 猜测 | 实验 | 结果 |
|---|---|---|
| 数据搬运是瓶颈 | `--frozen-data`（一个 batch 常驻 NPU） | 16384 省 0.75 ms、2048 省 1.56 ms → 数据侧 ≤1.6 ms |
| HBM 配额不足导致 spill | `--hbm-gb 4 → 16` | 16384 省 0.87 ms、2048 无变化 → 不是热集问题 |

当前瓶颈：`lookup_backward_v2_kernel` 24 次/步 × 615 µs。优化方向与判据见 §2.4 P1。

## 5. 数据格式

`dac_sample_day_{i}_processed.npz`，每天一个文件：

| 键 | dtype | 形状 | 说明 |
|---|---|---|---|
| `X_int` | int32 | (N, 13) | 数值特征，**已做 `log(max(x,0)+3)`** |
| `X_cat` | float64 | (N, 26) | 类别特征（已按词表取模） |
| `y` | int32 | (N,) | 标签 |

```bash
export DLRM_DATA_DIR=/path/to/npz_dir      # 默认 ./data/criteo_pt
```

1 TB 切片的构造（parquet → npy → npz）见 `results/dlrm_950_criteo1tb_vs_gpu.txt` §1。

### 数据落盘位置（重要）

数据集必须放在**宿主机挂载盘**，不要留在容器 overlay 里（`docker rm` 会连数据一起删）：

```
<宿主数据盘>/dlrm950/data/dlrm1tb/
├── parquet/      # 原始 parquet（day=2015-02-15，250 文件，11.4 GiB）
├── npy24/        # 24 shard（112.3 M 行，16.7 GiB）
│   └── dac_sample_day_{0..6}_processed.npz   # 训练/测试实际读取（6.1 GiB）
└── npy/ npy_probe/
```

容器以 `-v <宿主数据盘>:<容器内数据根>` 整块挂载后，`--data-dir` 直接指 `…/dlrm1tb/npy24`。
迁移核对判据：两侧 `find -printf "%P %s"` 逐行 `diff` 为 0 行，且 npz 文件 md5 一致。

## 6. 运行

```bash
torchrun --nproc_per_node=8 dlrm_v2_train.py \
  --batch-size 16384 --epochs 1 --eval-every 0 --warmup 20 --steps 300 \
  --data-dir $DLRM_DATA_DIR --train-days 0,1,2,3,4,5 --test-days 6
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch-size` | 1024 | **全局** batch（每 die = global/8） |
| `--hbm-gb` | 4.0 | 每张表 HBM 配额（26 表均分） |
| `--dist-type` | continuous | key→rank 映射，可选 `roundrobin` |
| `--frozen-data` | 关 | 冻结一个 batch 常驻 NPU，隔离数据侧 |
| `--csv` | 空 | 逐步耗时落盘 |

`tools/`：`p2_no_step_sync.py` 生成带 `--sync-every` 的版本；`parse_msprof.py` 解析 msprof 单步预算。

## 7. 已知限制

- 交互类型只有 dot，无 DCN 分支；
- 稠密优化器写死 SGD（A100 的 2048 档用 Adagrad，口径不同）；
- loader 把整天 npz 读进内存，10^8 行级别需分天切片；
- 多机未验证。

## 8. 脱敏说明

入库版本数据路径走 `DLRM_DATA_DIR` 或 `./data` 占位，代码与文档不含内网地址、主机名、账号信息。
