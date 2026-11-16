# 训练日志原件（results/logs/raw/）

DLRM@950DT 全量训练日志，**脱敏后 gzip**，与 `../*.summary.txt`（20 行摘要）对应同一批实验。
导出包：`dlrm950_artifacts_v1_20261116.tar.gz`（含 profiling CSV + 锚定汇总，未入库的 21 GB msprof 原始采集见报告 §10）。

## 取值

```bash
zgrep -E 'step_time_ms|throughput|test_auc' results/logs/raw/<文件>.log.gz
```

每份日志都有自证行：`step_time_ms = ... median=...`、`throughput = ... samples/s (global) :: test_auc = ...`

## 分组

| 组 | 文件 | 实验 |
|---|---|---|
| 统一表锚定 | `un_A1_u_r1..r3`、`un_A2_u_r1..r3` | 26 表合 1（rows=268,435,456），n=3 |
| EC+dedup 锚定 | `ec2_A2_dd_r1..r3`、`ec2_A1_dd_r1..r3`、`ec2.txt` | EmbeddingCollection + index dedup，n=3 |
| EC 变体消融 | `ec_a2_ec`（无 dedup）、`ec_a2_ec_dd`、`ec_a1_ec`、`ec.txt` | 证明收益来自 dedup |
| batch 扩展 | `bs_b32k`、`bs_b64k`、`bs_b128k`、`bs.txt` | 32k / 64k / 128k |
| A4 档 | `a4_r3`、`a4_r4`、`a4.txt` | global 131,072 |
| TASK_QUEUE A/B | `ab_a2_{base,tq2}_r1..r3`、`ab_a4_{base,tq2}_r1..r2`、`ab.txt` | 交错多轮 |
| A1 附加 | `a1tq_{base,ovl,tq2}`、`a1tq.txt` | A1 上试 tq2 / overlap |
| HCCL 开关 | `hc_base`、`hc_base2`、`hc_aiv`、`hc_buf` | AIV / BUFFSIZE=1024 |
| 取数固化 | `prep_A1_ebc`、`prep_A2_ec`、`prep.txt`、`preproc_criteo` | 加载期 log1p + dtype |
| 诊断探针 | `pr_frozen2048`、`pr_frozen16384`、`pr_hbm16_2048`、`pr_hbm16_16384` | 取数上限 / hbm-gb |
| 同步节奏 sweep | `sw_b2_s2/s4`、`sw_b16_s2/s4/s8`、`sw_a2_*`、`sw_a4_*` | sync 间隔 / overlap 组合 |
| 利用率与带宽 | `util_run`、`util2_run`、`util3_run`、`util4_run`、`util.txt`、`bwprobe`、`bw2`、`hbm_run` | npu-smi 采样 + 带宽标定 |

## 口径

步时 = forward+backward+step，每步 synchronize 后计时；取数（prep）单列。统计取 median、剔除前 20 步。多数为 150/200/300 步截断；`util3_run`、`util4_run`、`hbm_run` 为满 epoch（1,354 步）。profiler 采集有 1.19–1.30× 扰动，绝对步时一律引无 profiler 的日志。

## 脱敏

venv / 数据盘 / CANN 路径、内网与环回 IP、主机名、容器名均已占位化（`<VENV>` / `<DATA_DIR>` / `<CANN>` / `<IP>` / `<HOST>` / `<CTR>`）。
