# 上游来源与许可

## 来源

| 项 | 值 |
|---|---|
| 仓库 | `https://gitcode.com/Ascend/RecSDK` |
| 分支 | `develop_torch_benchmark` |
| commit | `43a0097`（`!3270 merge ops into develop_torch_benchmark`） |
| 取用路径 | `torch_examples_benchmark/`（`develop/` 与 `preview/` 两个子目录） |
| 取用方式 | `git clone --depth 1 --filter=blob:none --sparse -b develop_torch_benchmark`，再 `git sparse-checkout set torch_examples_benchmark` |

## 本仓做的改动

1. **剔除生成式推荐**：删除 `develop/gr`、`develop/gr_nv`、`preview/gr`、`preview/recsys_gr`、`preview/wukong`，只保留传统搜推。
2. **结构归一**：上游示例统一放在 `recsdk-examples/` 下，原相对路径不变（例如 `recsdk-examples/develop/dlrm/run.sh`）。
3. **补充**：`README.md`（示例清单与复现前提）、`docs/950-support.md`（950 支持性证据）、`scripts/check_env.py`（环境探测）、`results/check_env_950.txt`（950 实测输出）。

除上述外，`recsdk-examples/` 下的文件未做内容修改。

## 许可

示例代码版权归 Ascend / 各开源模型作者所有，沿用上游仓库的许可（见上游 `LICENSE`）。本仓仅作归档整理与平台适配探测，不改变原始许可条款。

## 各示例的上游模型与固定版本

| 示例 | 上游开源实现 | 固定 commit |
|---|---|---|
| DLRM | `facebookresearch/dlrm`（`torchrec_dlrm/`） | `b631a99`（2024-06-07） |
| DIN | `alibaba/TorchEasyRec` | `9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e`（2025-05-30） |
| MMoE / ETA / DeepCTR 系列 | `shenweichen/DeepCTR` 等 | 见各 `README.md` |
| FuxiCTR | FuxiCTR | 见 `recsdk-examples/preview/fuxictr/README.md` |
| TorchEasyRec 子模型 | `alibaba/TorchEasyRec` | 同上 |
