# DLRM 等传统搜推示例在 Ascend 950 上的支持现状

结论先行：**DLRM 已在 Ascend950DT 上实测跑通**（训练 3 步 + 测试评估，见 §5）；官方文档只承诺 Atlas A2，且需要两件 950 侧的准备：`fbgemm_ascend` 自定义算子注册（37 个 A5 算子）+ 3 处 torchrec API 兼容改动。逐项证据如下。

## 1. 官方口径

`recsdk-examples/develop/README.md` 原文：

> 本模型迁移依赖特定版本的 CANN、PyTorch、驱动和固件……**仅支持昇腾平台（Atlas A2 训练系列产品）**

全文（含 DLRM / DIN / model_zoo 各 README）检索 `950`、`910_95`、`A5`、`Atlas A3`：真实命中 0 处，只有 patch 文件里 commit hash 的十六进制子串被正则误捕。⇒ 上游没有声明过 950 支持。

## 2. 代码层：平台无关

`recsdk-examples/develop/dlrm/dlrm_npu.patch` 里设备选择写成：

```
-        device: torch.device = torch.device(f"cuda:{rank}")
-        torch.cuda.set_device(device)
+    elif torch_npu.npu.is_available():
```

即只判断「是不是 NPU」，不判断具体 SOC；模型、算子调用全部走 RecSDK-Torch（`fbgemm_ascend` + `torchrec`）。**结论：能否在 950 上跑，取决于依赖包的算子产物，而不是示例代码。**

## 3. 950 实测（Ascend950DT，CANN 9.1.0 / torch 2.7.1+cpu / torch_npu 2.7.1.post4）

原始输出：`results/check_env_950.txt`（脚本 `scripts/check_env.py`）。

| 检查项 | 实测 |
|---|---|
| SOC / 规模 | `Ascend950DT`，8 die，单 die HBM 98,304 MB |
| 驱动 / 固件 | 25.1.rc2 / 7.0.9.100.203 |
| `torch` / `torch_npu` | 2.7.1+cpu / 2.7.1.post4（齐） |
| `torchrec` / `fbgemm_gpu` / `fbgemm_ascend` | 1.2.0+npu / 1.2.0+cpu / 1.2.0（齐） |
| `hybrid_torchrec` / `torchrec_embcache` / `rec_cust_ops` | **均缺失**（`ModuleNotFoundError`） |
| `fbgemm_ascend` 算子产物 | 292 个 `.so` 中 **175 个**含本机 SOC 字符串；`opp/` 下有 **A2 / A3 / A5** 三套目录，A5 即 950 系列 |

`fbgemm_ascend` 内置算子示例（A5 目录下）：`lfu_cache_insert_byte`、`run_length_encode`、`group_index_select_dim0`、`select_dim1_to_permute` 等，`.so` 内同时含 `Ascend910 / Ascend910B / Ascend910_93 / Ascend950` 字符串 ⇒ **算子包已按 950 编译过**。

## 4. 由此得到的三条判定

1. **DLRM 主体路径（dense + 稀疏查表 + DCN）在 950 上已实测跑通**：torchrec 1.2.0+npu + fbgemm_ascend 1.2.0（A5 算子）在场，注册算子后训练与评估均正常。
2. **DLRM 的 EC（embcache）路径在 950 上尚未可用**：需要 `torchrec_embcache` + `hybrid_torchrec`，本环境两者皆缺（上游 `run.sh` 里 `USE_EC=0` 正是默认关闭）。
3. **DIN 在 950 上不可用（架构限制，非算子问题）**：上游明确「该迁移样例只支持在 x86 上运行」，而 950 宿主为 aarch64（`results/check_env_950.txt` 的 OS 行）。

## 5. 实测步骤与结果（2026-11-15）

复现命令（容器内，`preview/dlrm` 目录）：

```bash
# 1) 注册 AscendC 自定义算子（关键，缺这步会在第一步 bucketize 报 aclnn 缺失）
source $(python -c "import fbgemm_ascend; print(fbgemm_ascend.env_setup_path())")
export LIB_FBGEMM_NPU_API_SO_PATH=$(python -c "import fbgemm_ascend,os;print(os.path.join(os.path.dirname(fbgemm_ascend.__file__),'fbgemm_ascend_py_a5.so'))")
export USE_EC=0 WITH_EMBCACHE=0 WITH_TORCHREC=1

# 2) 打 3 处 torchrec API 兼容补丁（非硬件相关）
patch -p1 < ../../patches/dlr950_torchrec_compat.patch

# 3) 随机数据冒烟（单 die）
torchrun --nproc_per_node=1 dlrm_main.py --epochs 1 --batch_size 64 \
  --limit_train_batches 3 --limit_val_batches 1 --limit_test_batches 1 \
  --num_embeddings 100000 --embedding_dim 16 --dense_arch_layer_sizes 32,16 \
  --over_arch_layer_sizes 64,32,1 --interaction_type dcn --dcn_num_layers 2 \
  --dcn_low_rank_dim 32 --learning_rate 0.01 --validation_freq_within_epoch 10
```

结果（`results/dlrm_950_smoke.txt`）：

| 阶段 | 实测 |
|---|---|
| 算子注册 | 37 个 A5 算子注册成功（含 `block_bucketize_sparse_features`） |
| 分片规划 | 26 张表全部 `RW` + `fused`，HBM 0.157 GB |
| 训练 | 3 步：1.59 → 2.76 → 3.70 it/s，`Train avg speed: 3.7019 its/s` |
| 测试 | `Test AUROC: 0.506836`（随机数据，≈0.5 符合预期） |

**3 处 torchrec API 兼容改动**（`patches/dlr950_torchrec_compat.patch`，全部是版本差异，不是 950 特有问题）：

1. `ParameterConstraints` 必须显式给 `compute_kernels=["fused"]`（本机 fork 的 `__post_init__` 校验）；
2. `Topology` 不接受 `local_world_size`（本机 fork 要求为 None）；
3. `EmbeddingShardingPlanner` 不接受 `storage_reservation`，且 `constraints` 必传。

## 6. 仍未验证

- [ ] 真实 Criteo 数据集 / MLPerf 口径吞吐（`run.sh` 的 `GLOBAL_BATCH_SIZE=16384`、8 die）
- [ ] EC / embcache 路径（需 `torchrec_embcache` + `hybrid_torchrec`，本环境未装）
- [ ] `rec_cust_ops` 依赖的样例（当前环境无此包；DLRM 主体不依赖）
- [ ] model_zoo / deepCTR / fuxictr 等其他示例在 950 上的冒烟

> 口径提醒：§5 是**功能冒烟**（batch 64、emb_dim 16、随机数据、单 die），不含性能结论。

## 7. 吞吐（真实数据，2026-11-15）

数据：Criteo Kaggle `train.txt` 45,840,617 行（真实数据，单热），26 表合计 33.4 M 行；`--mmap_mode` 读 npy。
8 die，`--limit_train_batches` 截断，步时取 median。完整口径与命令见 `results/dlrm_950_real_criteo_bench.txt`。

| 全局 batch（每卡本地） | 步时 | 吞吐 |
|---|---:|---:|
| 16384（2048） | 258.4 ms | 63,406 sps |
| 1024（128） | 259.5 ms | 3,945 sps |
| 256（32） | 270 ms | 947 sps |

关键结论：**步时与 batch 几乎无关（±4%）**，且把词表从 33.4 M 行压到 0.21 M 行步时仅从 259.5 ms 变到 286 ms ⇒ 每步约 255–285 ms 的固定开销主导，与表规模、batch 均无关（推测为 26 张表各自的 bucketize + all2all，待 msprof 证实）。

与同机 `dlrm950`（RecSDK v2 / dynamic_emb HKV）对照（每卡本地 batch 对齐）：

| 全局 batch | 本示例（静态 TBE + fbgemm_ascend） | dlrm950（HKV） | 比值 |
|---|---:|---:|---:|
| 1024 | 259.5 ms → 3,945 sps | 23.667 ms → 43,267 sps | 11.0× |
| 16384 | 258.4 ms → 63,406 sps | 55.366 ms → 295,943 sps | 4.7× |

口径边界：两侧数据规模（45.8 M vs 10 万行）、表规模（33.4 M vs 基数 ≤7546）、dense 结构（MLPerf dcn vs TF zoo 对齐）都不同，**不能表述为「HKV 比 fbgemm TBE 快 N 倍」**。
