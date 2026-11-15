# profiling —— 性能诊断数据与报告

本目录只放**性能诊断**产物（分解数据、原始明细、诊断结论）。口径与判据见 `../OPERATIONS.md`；优化前后的对比数字在 `../results/`。

## 索引

| 文件 | 内容 | 采集口径 |
|---|---|---|
| `stage0_a2_16384.md` | A2 档（global 16,384）单步分解诊断：空转 / 稀疏反向 / 通信 / 稠密，含归因与已排除项 | msprof 包 8 rank、20 步；rank0/device_0 |
| `stage0_a2_16384.csv` | 上表的机器可读版（项、每步 ms、占 wall、调用次数、单次耗时） | 同上 |
| `hccl_a2_16384.csv` | HCCL 分项（reduceScatter / alltoall / HcclLaunch / allReduce / allGather / alltoallv） | 同上 |
| `stage1_unified_a2_16384.md` | 统一表后同档分解：查表 15.05→3.41 ms，空转+等待占 80%，通信并集仅 1.69 ms |
| `op_top_k.csv` | 算子 top25（time/count），由 `tools/publish.sh` 自动从 msprof `op_statistic` 提取 | 
| `fit_fixed_vs_var.md` | 「固定项 + 每样本项」线性拟合（A1/A2 本轮 n=3 实测），用于判断优化该打哪一轴 | 无 profiler 的 runner |

## 复现

```bash
# 采集（容器内才有 msprof；路径以该机 CANN 安装为准）
msprof --output=<OUT> --ai-core=on --aicpu=on --model-execution=on \
  <VENV>/torchrun --nproc_per_node=8 dlrm_v2_train.py \
  --batch-size 16384 --epochs 1 --warmup 20 --steps 20 --eval-every 0 \
  --data-dir $DLRM_DATA_DIR

# 解析成单步预算表
python3 ../tools/parse_msprof.py <OUT>
```

## 三条读数纪律

1. **profiler 有扰动**（实测步时 62.8 vs 生产 37.5 ms，1.67×）⇒ 只引**比例**，绝对耗时一律取无 profiler 的 runner。
2. **窗口必须切在训练段**：用每步必现的 `lookup_backward_v2` 首末时间戳框窗口，否则会把建表/planner 的 init 任务算进来（`allocate_bucket_*` 曾单轮 66.7 万次）。
3. **busy 用「步数 × 实测步时」归一**，不要用「末任务 − 首任务」（含 eval 空隙会算出 3% 这种荒谬值）。
