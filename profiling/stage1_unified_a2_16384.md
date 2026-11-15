# 阶段 1 诊断：统一表后的 A2 档（global 16,384）

采集：`msprof --ai-core=on --aicpu=on --model-execution=on`，`--application=` 方式启动 8 rank、20 步、warmup 3；解析 rank0/device_0 的 `task_time`。
窗口：首末 `lookup_backward_v2_kernel` 之间；窗口步时 **25.2 ms**（profiler 扰动后，生产同配置为 20.34 ms）。

## 单步分解（并集口径）

| 类别 | 每步 | 占 wall | 说明 |
|---|---:|---:|---|
| **空转（无任何 task）** | **12.02 ms** | **47.7%** | 设备没任务可执行，host 下发/队列断档 |
| **等待（NOTIFY_WAIT / DAVID_EVENT_WAIT）** | **8.15 ms** | **32.3%** | 已下发但等对端/等事件 |
| 计算（AI_CORE / AI_VECTOR / MIX） | 6.62 ms | 26.3% | 含查表反向 3.41 ms |
| AICPU | 2.07 ms | 8.2% | HCCL 调度 + 动态表 AICPU 算子 |
| 通信数据搬运（UBDMA / SDMA 并集） | 1.69 ms | 6.7% | 注意：按类求和是 8.38 ms，并集只有 1.69 ms（多流重叠） |

（各类可重叠，占比之和 >100%。）

## 与优化前（26 表）对比

| 项 | 26 表（A2） | 统一表（A2） | 变化 |
|---|---:|---:|---:|
| `lookup_backward_v2` | 15.05 ms（24 次 × 615 µs） | **3.41 ms（1 次 × 3413 µs）** | −77% |
| 单次耗时 | 615 µs | 3413 µs | 变大（一次处理 42.6 万 key），但总时长大幅下降 |
| 空转占比 | 29.8% | **47.7%** | 占比升高（其他项缩了），绝对 12.80 → 12.02 ms |
| 通信 | 9.51 ms（6–7 次） | 并集 1.69 ms | 显著下降 |

## 结论

1. **查表不再是瓶颈**：从 15.05 ms 降到 3.41 ms，占 wall 13.5%。继续优化查表（如 ③ dedup）单看这一项上限只有 ~2–3 ms。
2. **空转 + 等待 = 20.2 ms（80%）**：这一步的瓶颈已从「算子太慢」变成「设备在等」，即 host 逐步下发与同步点。⇒ 下一步优先 **图捕获 / 减少每步 kernel 数**，不是再压算子。
3. **通信数据量很小**（并集 1.69 ms）⇒ 合并 collective 的收益上限 <1.7 ms，优先级下调；此前「HCCL 占 22%」的判断只对 26 表结构成立，**已随统一表失效**。
4. 计算 6.62 ms 里查表 3.41 + MatMul 约 1.3 + pooling/其它约 1.9，稠密侧收益上限仍 <1.5 ms。

## 复现

```bash
msprof --output=<OUT> --ai-core=on --aicpu=on --model-execution=on \
  --application="<VENV>/torchrun --nproc_per_node=8 dlrm_v2_train.py \
    --batch-size 16384 --epochs 1 --eval-every 0 --warmup 3 --steps 20 \
    --data-dir $DLRM_DATA_DIR --train-days 0,1,2,3,4,5 --test-days 6 --unified-emb"
python3 ../tools/parse_window.py <OUT> 20
```

⚠️ msprof 必须用 `--application=` 启动目标，直接把命令当位置参数只会采到 launcher（实测只产出 `api_statistic`，没有 `task_time`）。
