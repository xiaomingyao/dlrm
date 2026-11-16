# RecSDK DLRM 与 facebookresearch/dlrm（torchrec_dlrm）的差异

对比基准：

| 侧 | 来源 | commit |
|---|---|---|
| 上游 | `facebookresearch/dlrm` → `torchrec_dlrm/` | `b631a99`（2024-06-07） |
| RecSDK | `recsdk-examples/preview/dlrm/`（等价于 `develop/dlrm` 打上 `dlrm_npu.patch`） | RecSDK `43a0097` |

## 1. 规模

| 文件 | 上游 | RecSDK | 差异 |
|---|---:|---:|---|
| `dlrm_main.py` | 727 行 | 987 行 | **+345 / −109** |
| `multi_hot.py` | 175 | 226 | +66 / −19 |
| `data/multi_hot_criteo.py` | 306 | 285 | +18 / −38（多为格式化，实质改动小） |
| `data/dlrm_dataloader.py` | 169 | 169 | 1 处（`pin_memory` 固定 False） |
| `lr_scheduler.py` | 48 | 48 | 1 处（`sys.exit` → `raise ValueError`） |
| `aws_component.py` | 43 | 44 | 类型标注 + `USER` 环境变量判空 |

仅 RecSDK 有：`ec_dcnv2.py`（184 行，DCN + embedding cache 变体）、`run.sh`、`dlrm_patch.patch`、`generate_data.py`（在 `develop/dlrm/`）。
仅上游有：`Dockerfile`、`requirements.txt`、`scripts/`、`tests/`、`README.MD`、md5 校验清单 —— RecSDK 侧不提供容器与测试脚手架。

## 2. 关键差异（按影响排序）

| # | 维度 | 上游 | RecSDK | 影响 |
|---|---|---|---|---|
| 1 | **设备后端** | 只走 CUDA：`torch.device(f"cuda:{rank}")` + `torch.cuda.set_device` | `import torch_npu` + `torch_npu.npu.is_available()` 分支，NPU 上 `set_device` | 平台适配的总开关 |
| 2 | **稀疏算子来源** | `fbgemm_gpu`（CUDA 版 split TBE） | `fbgemm_ascend`：`torch.ops.load_library(LIB_FBGEMM_NPU_API_SO_PATH)` 加载 `*_py_a5.so` / `*_py_a2a3.so`；还需 `source fbgemm_ascend/env_setup.sh` 注册 AscendC 自定义算子（A5 侧 37 个，含 `block_bucketize_sparse_features`） | **不注册就会在第一步 `block_bucketize_sparse_features` 报 aclnn 缺失** |
| 3 | **嵌入 cache / EC** | 无 | 新增 `ec_dcnv2.py`（`DlrmDcnEc`）+ `torchrec_embcache_embedding` 的 `EmbCache*Sharder/Pipeline`；由 `WITH_EMBCACHE` / `USE_EC` / `USE_MOD_BUCKETIZE` / `EMBCACHE_SIZE_ON_DEVICE_MEM` 控制 | 走 CPU/device 分层缓存的大表方案，属 NPU 侧新增能力 |
| 4 | **分片与 planner** | `EmbeddingShardingPlanner(topology, batch_size, storage_reservation=…5%)` | 显式 `ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])`，**去掉** `storage_reservation` 与 `Topology.local_world_size` | 本机 torchrec（Ascend fork）强制要求这两项，且只支持 `(row_wise, fused)` 组合 |
| 5 | **训练循环结构** | 单个 `_train(...)` 长参数列表 + `_evaluate` | 重构为 `@dataclass TrainParam` / `TrainValTestParam` 承载参数；加 `tqdm` 进度条、`logging` 统一输出、按迭代数打印 lr、`limit_*_batches` 显式截断 | 可读性与可观测性改造，语义等价 |
| 6 | **指标 API** | `metrics.AUROC(compute_on_step=False, num_classes=2)` | `metrics.AUROC("binary")` | 适配 torchmetrics 新版签名（上游该调用在新版会报错，正是 `b631a99` 的修复方向） |
| 7 | **多档开关** | 无 | 新增环境变量：`WITH_TORCHREC`、`WITH_HYBRID_TORCHREC`、`WITH_EMBCACHE`、`USE_EC`、`SPARSE_OPTIM_NUM`、`LIB_FBGEMM_NPU_API_SO_PATH` | 一条代码支持「纯 torchrec / hybrid_torchrec / embcache」多种嵌入实现 |
| 8 | **多热与数据管道** | `MultiHotCriteoIterDataPipe` 原版 | 同类实现，`pin_memory` 强制 False（NPU H2D 路径）、精简注释与格式 | 功能等价，NPU 上避免锁页内存开销 |
| 9 | **启动方式** | `torchx run … dist.ddp`（`aws_component.py`） | 保留 torchx 入口，另给 `run.sh`（`torchx run -s local_cwd dist.ddp -j 1x8`）+ `generate_data.py` 造随机数据 | 开箱可跑 |

## 3. 结论

- **模型数学与网络结构未改**：DLRM 的 dense 塔、sparse 表、DCN/交互层、loss 定义一致，差异集中在设备后端、稀疏算子来源、分片规划与工程化。
- **RecSDK 侧是「框架级替换」而非「移植补丁」**：把 fbgemm_gpu 的 CUDA 算子换成 fbgemm_ascend 的 AscendC 算子，并把 torchrec 的分片 planner 收窄到 row_wise + fused。
- **对本机 950 的实际约束**（见 §4）：`fbgemm_ascend` 的 A5 算子必须通过 `env_setup.sh` 注册；`rec_cust_ops` 缺失只影响依赖它的样例（DLRM 本身不依赖）。

## 4. 在 Ascend950DT 上的实测（2026-11-15）

原始输出：`results/dlrm_950_smoke.txt`；兼容性改动：`patches/dlr950_torchrec_compat.patch`（3 处，均为 torchrec API 差异，非硬件相关）。

| 阶段 | 结果 |
|---|---|
| 算子注册 | `ASCEND_CUSTOM_OPP_PATH` 注册 **37 个 A5 自定义算子**，含 `block_bucketize_sparse_features` |
| 分片规划 | 成功：26 张表全部 `RW`（row_wise）+ `fused`，HBM 估算 0.157 GB |
| 训练 | 3 步跑通：`1.59 → 2.76 → 3.70 it/s`，`Train avg speed: 3.7019 its/s` |
| 验证/测试 | 跑通：`Test AUROC: 0.506836`（随机数据，≈0.5 符合预期） |

> 口径：这是**功能冒烟**（`--batch_size 64`、`--embedding_dim 16`、`--num_embeddings 100000`、随机数据、单 die），不构成性能结论；MLPerf 口径的吞吐需按 `run.sh` 的真实数据集与 8 die 配置另测。
