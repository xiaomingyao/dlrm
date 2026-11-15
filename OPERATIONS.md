# OPERATIONS —— dlrm/ 的目标与操作规约

> 本文件固化优化目标、锚定用例与操作流程。与 README 冲突时以本文件为准（README 面向读者，本文件面向执行）。

## 0. 真源与改动流

| 事项 | 规则 |
|---|---|
| 代码真源 | **本仓 `dlrm/`**；集群运行目录是只读副本 |
| 改码 | 只改本仓 → commit → `bash dlrm/tools/sync_to_node.sh <节点目录>`（两侧 md5 必须一致才允许继续） |
| 跑实验 | 一律用同步后的节点副本；**禁止在节点上就地改代码** |
| 结果 | 原始数据回写 `results/`，实验参数写进 commit message |
| 敏感信息 | 不写内网 IP / 主机名 / 账号 / 数据盘绝对路径；用 `DLRM_DATA_DIR`、`<CTR>`、`<DATA_DIR>` 占位 |

## 1. 目标

| # | 能力 | 判据 |
|---|---|---|
| 1 | 移植 | 8 die 跑通、300 步无中断、AUC 收敛（不用 acc 判） |
| 2 | **高性能：2× A100（阶段一硬指标）** | §2 两个锚定档，每档吞吐 ≥ 2× 同档 A100 |
| 3 | 线性扩展 | 8→16 die 扩展比 ≥1.6（效率 ≥80%），16→32 ≥1.7 |
| 4 | 高性能 HKV | 单步 lookup 反向 ≤3 ms（现状 15.05 ms） |

阶段一目标值：global 2048 档 **≥409,600 sps（≤5.00 ms/步）**、global 16384 档 **≥1,851,392 sps（≤8.85 ms/步）**。

## 2. 锚定用例（唯一合法对比矩阵）

只对比 A100 官方 README 的四行，不新增、不外推、不跨档相除：

| 编号 | local BS（global） | 交互/优化器 | A100 8× | 本实现 |
|---|---|---:|---:|---|
| **A1** | 256（2,048） | Dot / SGD | 204,800 sps（10.00 ms） | ✅ 可测（dot + 稀疏 Adagrad + 稠密 SGD） |
| **A2** | 2,048（16,384） | Dot / Adagrad | 925,696 sps（17.70 ms） | ✅ 可测（优化器差异需标注） |
| A3 | 2,048（16,384） | DCN v2 / Adagrad | 901,120 sps（18.18 ms） | ❌ 不可测：本实现无 DCN |
| A4 | 16,384（131,072） | DCN v2 / Adagrad | 1,190,128 sps（110.13 ms） | ❌ 不可测：本实现无 DCN |

规则：

1. **只有 A1/A2 进对比表**；A3/A4 只在「限制」里点名不可测及原因。
2. 每档报告 median 步时 + 全局吞吐（global batch ÷ 步时）+ 达成率（本实现 ÷ A100）。
3. 数据：Criteo 1TB 切片（`--train-days 0,1,2,3,4,5 --test-days 6`，24.5 M 行）+ MLPerf 词表；口径必须在表下注明。
4. 每档 **n≥3 次独立进程**，取 median of medians；单次差异 <3% 视为噪声，不得定性。
5. 禁止 A1 与 A2 相除；禁止把其它 global batch 的数字塞进 A 行。

## 3. 度量口径

| 项 | 规定 |
|---|---|
| 步时 | 每档 300 步、剔前 20 步、取 **median**（mean 受偶发尖峰污染，实测 max 675 ms） |
| 吞吐 | 全局 samples/s = global batch ÷ median 步时 |
| 本地 batch | 脚本 `--batch-size` 为**全局**语义（每 die = global/8）；跨实现比较前确认每卡本地 batch 相等 |
| profiler | 只用于归因；**吞吐结论只能引无 profiler 的 runner**（插桩扰动实测 1.67×） |
| 精度 | 只认 AUC + loss 单调；**禁用 acc**（全负基线 0.9679 > 实测 0.9674） |
| 合成数据 | 只验链路，不做精度验收 |
| 归因 | 未采 profiler 的一律标「推测」 |

## 4. 优化路线（按实测预算）

A2 档每步 42.90 ms：idle 12.80 + HKV 15.05 + HCCL 9.51 + 其余 device 5.54。
分解：**固定项 22.0 ms ≈ idle + HCCL；变长项 1.273 µs/样本 ≈ HKV 0.92 + 其余 0.34**。

| WP | 动作 | 门禁 |
|---|---|---|
| **B** | 融合查表：26 表 × N 行一次 batched lookup/backward（包内已有 `BatchedDynamicEmbeddingTablesV2`，未接入 EBC 路径） | 单步 lookup kernel ≤2 次且该项 ≤5 ms；A2 ≤38 ms |
| **A** | 把融合路径接进 sharder / 自写 sharded module | 再降 ≥2 ms，A1/A2 AUC 波动 ≤0.01 |
| **C** | 图捕获（aclgraph）+ 稠密 bf16 | idle ≤6 ms；A2 ≤30 ms |
| **D** | 合并 collective：6 次 → ≤2 次 | HCCL ≤6 ms |
| **E** | int32 sparse + 合批 H2D | 变长项 ≤1.0 µs/样本 |

**已验证无效，不要再做**：逐步同步（1/2/4/8 差 <1.2 ms；间隔 50 步 +49%）、加大 `--hbm-gb`（省 0.87 ms）、`--frozen-data`（省 0.75 ms）、单纯调大 batch（b32 63.49 ms 只验证线性模型，无 A100 对应档）。

## 5. 标准操作

```bash
# 1) 同步（唯一改码出口）
bash dlrm/tools/sync_to_node.sh <节点目录>

# 2) 锚定基准（A1 + A2 各 N 次）
bash dlrm/tools/bench_anchor.sh <节点目录> <DATA_DIR> [repeats]

# 3) 单档手动跑（global batch 语义）
<VENV>/torchrun --nproc_per_node=8 dlrm_v2_train.py \
  --batch-size 16384 --epochs 1 --eval-every 0 --warmup 20 --steps 300 \
  --data-dir $DLRM_DATA_DIR --train-days 0,1,2,3,4,5 --test-days 6

# 4) 归因（按需）
python3 tools/parse_msprof.py <msprof 输出目录>
```

判据：`bench_anchor.sh` 输出必须有 `A1`、`A2` 两行，每行带 `n=`、`median`、`sps`、`vs A100`。

## 6. 红线

1. 用 acc 判精度；2. 跨 batch / 跨本地 batch 相除；3. 用 A3/A4 的 A100 值算达成率；4. 把 profiler 步时当吞吐；5. 在节点上改代码；6. 提交含内网信息的文件；7. 用一次测量定性 <3% 的差异。

## 7. 状态快照

| 日期 | A1（global 2048） | A2（global 16384） | 说明 |
|---|---|---|---|
| 2026-11-15 | 24.26 ms / 82,988 sps（41%） | 42.90 ms / 378,798 sps（41%） | 优化前基线；A2 同日另测 42.26，属会话波动 |
| 2026-11-15 | — | `--dist-type roundrobin` 41.31 ms / 393,494 sps | 单次，待 n=3 复核 |
