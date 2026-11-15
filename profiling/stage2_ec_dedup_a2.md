# 阶段 2 诊断：统一表 + EC + dedup（A2 档）

采集：`torch_npu.profiler`（rank0，wait=1/warmup=2/active=20），导出 `kernel_details.csv`；
窗口 **21.51 ms/步**（含 profiler 扰动；生产实测 16.86 ms + 取数 1.16 ms）。
解析工具：`tools/parse_kernel_details.py`。

> 为什么不用 msprof：同一配置下 `msprof --model-execution=on` 会让 rank2 `SIGSEGV`（exitcode -11），
> 无 model-execution 时又只产出 `api_statistic`。`torch_npu.profiler` 能稳定采到，且多出
> `cube_utilization` / `aiv_vec_ratio` 等利用率列。

## 1. 每步构成（173 个 kernel/步，dedup 前约 536）

| 类别 | 每步 | 占 wall | 明细 |
|---|---:|---:|---|
| **通信合计** | **5.44 ms** | **25%** | alltoallv 1.82（4 次）/ alltoall 1.75（1 次）/ allReduce 0.64（1 次）/ HcclLaunch 1.23（6 次，纯发射） |
| 计算合计 | ≈ 2.3 ms | 11% | MatMulV3 1.375（22 次）/ Sort 0.139 / InplaceIndexAdd 0.107 / hkv 0.081 / BatchMatMul 0.078 / ReduceSum 0.072 / 其它 ~0.45 |
| **Wait + 空隙** | **≈ 13.8 ms** | **64%** | `Wait Time` 字段合计 17.86 ms/步（多流求和，83% wall） |

## 2. 关键变化（相对阶段 1）

| 项 | 阶段 1（EBC） | 阶段 2（EC+dedup） | 说明 |
|---|---:|---:|---|
| 查表反向 `lookup_backward_v2` | 3.41 ms（1 次） | **0.081 ms（hkv 1 次）** | 唯一 id 从 42.6 万降到约 12.6 万 |
| 每步 kernel 数 | ~536 | **173** | dedup 后长尾算子大幅减少 |
| 通信 | 并集 1.69 ms | 5.44 ms | 口径不同（此表按 kernel 求和、含发射），但通信占比从 7% 升到 **25%** |

## 3. 硬件利用率：算子本身很饱满，机器整体很空

| 指标 | 值 | 解读 |
|---|---:|---|
| MatMulV3 `cube_utilization` | **均值 91.3%，峰值 99.5%** | 稠密矩阵乘在 Cube 上几乎打满 |
| MatMulV3 `aic_mac_ratio` | 0.77 | MAC 占比高，非搬运瓶颈 |
| 宿主采样 AICore / AIVector | 2.9% / 0.6% | 整机大部分时间没有任务 |
| `Wait Time` | 83% wall | 任务在队列里等前驱/等对端 |

结论：**不是算子写得慢，而是每步喂给设备的活太少**（2048 行/卡、单步 16 ms、173 个 kernel）。
稠密算子在 Cube 上 91% 效率，说明继续优化算子没有空间；要提速只能压缩「等」和「发射」。

## 4. 对 2× 目标的意义

A2 当前 16.31 ms（含取数 17.47 ms），2× 目标 8.85 ms，需再降 46%。
按本表结构，能动的只有：

| 项 | 上限 | 可行性 |
|---|---:|---|
| 通信与计算重叠（双流） | 最多省 ~5 ms | 中，需改 DMP 调用顺序，风险可控 |
| 合并 collective（6 次发射 → ≤2） | ~1.2 ms | 中，需改分组方式 |
| 消除 wait 空隙（rank 间负载均衡） | 理论最大 13.8 ms，实际能拿 2–4 ms | 低，需均衡 dedup 后各 rank 的 id 数 |
| 算子优化 | **≈0** | Cube 已 91%，无空间 |

即：即使三项全做到，乐观估计 16.31 → 10–11 ms（约 1.7× A100），**2× 在 2048 行/卡的结构下难以达到**；
若允许放大每卡本地 batch（无 A100 对照档），固定开销被摊薄，2× 才有可能——这属于「超参换性能」，需与验收口径确认。

## 5. 复现

```bash
torchrun --nproc_per_node=8 dlrm_v2_train.py ... --unified-emb --ec-mode --index-dedup \
  --pyprof /tmp/pyprof_ec --pyprof-steps 20
python3 tools/parse_kernel_details.py /tmp/pyprof_ec 20
```

⚠️ 该配置下 msprof 会段错误，别用 msprof 采这一步；用 `--pyprof`。
