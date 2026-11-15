# dlrm950 —— HKV 动态表的 DLRM 实现

RecSDK 自研稀疏侧（`dynamic_emb` + HKV 动态表）的 DLRM v2 实现，用于和 `recsdk-examples/` 里的 TorchRec 静态 TBE 版本做同机对照。

## 目录

| 文件 | 说明 |
|---|---|
| `dlrm_v2_train.py` | 训练入口：建表（HKV 分片规划）、数据加载、训练/评测循环、逐步耗时统计 |
| `dlrm_v2_model.py` | 模型：bottom MLP(512-256) + 26 表 embedding + dot 交互 + top MLP(1024-1024-512-256) |
| `criteo_dataset.py` | 数据：读取 `dac_sample_day_{i}_processed.npz`，按 2 的幂对齐表容量 |
| `tools/p2_no_step_sync.py` | 生成"去逐步同步"版本（每 N 步一次 `npu.synchronize()`，用于纠正计时口径） |
| `tools/parse_msprof.py` | 解析 msprof `op_summary*.csv`，按训练窗口输出每步预算 |

## 依赖

```
torch 2.7.1+cpu · torch_npu 2.7.1.post4
torchrec 1.2.0+npu · dynamic_emb · fbgemm_ascend 1.2.0
CANN 9.1.0 · Ascend950DT（8 die / 单 die 96 GiB HBM）
```

## 数据

输入为 npz，每天一个文件 `dac_sample_day_{i}_processed.npz`，含三件套：

| 键 | dtype | 形状 | 说明 |
|---|---|---|---|
| `X_int` | int32 | (N, 13) | 数值特征，**已做过 `log(max(x,0)+3)`**，本实现不再变换 |
| `X_cat` | float64 | (N, 26) | 类别特征（已取模到词表基数；float64 是该 loader 的读入约定） |
| `y` | int32 | (N,) | 标签 |

路径由环境变量给出：

```bash
export DLRM_DATA_DIR=/path/to/npz_dir     # 默认 ./data/criteo_pt
```

真实 Criteo 1TB 切片的构造方式见 `results/dlrm_950_criteo1tb_vs_gpu.txt` §1（parquet → `day_i_{dense,sparse,labels}.npy` → npz）。

## 运行（单机 8 die）

```bash
torchrun --nproc_per_node=8 dlrm_v2_train.py \
  --batch-size 16384 --epochs 1 --eval-every 0 --warmup 20 --steps 300 \
  --data-dir $DLRM_DATA_DIR --train-days 0,1,2,3,4,5 --test-days 6
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch-size` | 1024 | **全局** batch（每 die = global/8） |
| `--hbm-gb` | 4.0 | 每张表在 HBM 的配额（26 表均分） |
| `--dist-type` | 库默认(continuous) | key→rank 映射，可选 `roundrobin` |
| `--frozen-data` | 关 | 冻结一个 batch 常驻 NPU，用于隔离数据侧开销 |
| `--csv` | 空 | 逐步耗时落盘 |

## 实现要点

1. **稀疏侧用 HKV 动态表**：`dynamic_emb.DynamicEmbeddingBagCollectionSharder` + `DynamicEmbeddingShardingPlanner`，`local_hbm_for_values` 给本地配额，`global_hbm_for_values=0` 由 planner 推算；只物化被访问的 key。
2. **容量约束**：表容量取 `max+1` 向上对齐到 2 的幂，并要求 `num_embeddings % batch_size == 0`。
3. **优化器**：稀疏 fused `EXACT_ROWWISE_ADAGRAD`，稠密 `SGD`（无开关）。
4. **交互类型固定 dot**：按 TF zoo 对齐——`matmul(x, xᵀ)` 取上三角不含对角；`fan_avg` normal 初始化；L1=1e-2 只加首层。
5. **分布式**：HCCL 建组，planner 内部 `broadcast_object_list` 走 gloo 子组（HCCL 不支持 object collective）。

## 实测性能（8 die，Criteo 1TB 切片，MLPerf 词表）

| global batch | 步时(median) | 吞吐 | 对 8×A100 基线 |
|---:|---:|---:|---:|
| 2,048 | 24.26 ms | 82,988 sps | 40.5% |
| 16,384 | 42.90 ms | 378,798 sps | 40.9% |

单步预算（`--batch-size 16384`，msprof）：idle 12.80 ms(29.8%) + HKV 反向 lookup 15.05 ms(35.1%) + HCCL 9.51 ms(22.2%) + 其余 5.54 ms。
详见 `results/dlrm_950_criteo1tb_vs_gpu.txt` 与 `results/dlrm950_prof_stage0.txt`。

## 已知限制

- 交互类型只有 dot，无 DCN 分支；
- 稠密优化器写死 SGD，与 A100 基线里 2048 档的 Adagrad 不同口径；
- 数据 loader 会把整个 npz 读进内存，10^8 行级别需要分天切片；
- 表容量与 batch_size 的整除约束在改 batch 时会触发容量对齐。

## 脱敏说明

入库版本已把集群内的数据路径改为 `DLRM_DATA_DIR` 环境变量 / `./data` 占位，代码中不含内网地址、主机名或账号信息。
