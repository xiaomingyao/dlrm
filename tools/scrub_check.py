#!/usr/bin/env python3
"""发布前敏感信息闸门：扫描仓库内所有文本文件，命中即失败（exit 1）。

用法: python3 tools/scrub_check.py [仓库根目录]
规则与替换建议见下方 PATTERNS。允许清单：results/ 与 profiling/ 的 `.json` 结构数据不受影响。
"""
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATTERNS = [
    # 排除固件/内核等 5 段版本号（如 7.0.9.100.203）与非法段值；环回地址不算敏感
    ("IPv4",        r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}(?![\d.])"),
    ("工号/账号",    r"\bx0\d{7}\b"),
    ("内网数据盘",   r"/mnt/paas"),
    ("CANN 安装路径", r"/usr/local/Ascend"),
    ("venv 绝对路径", r"/opt/buildtools"),
    ("SSH 私钥",     r"id_(?:ed25519|rsa)"),
    ("主机名后缀",   r"\b(?:novalocal|localdomain|hyperinstance)\b"),
    ("容器名",       r"\b(?:zeus-950|onetrans-950|vllm-\d+|clearml-)\S*"),
    ("云端 token",   r"\b(?:ghp_|github_pat_|AKIA[0-9A-Z]{16})"),
    ("ssh 短别名",   r"\bssh (?:122|950-111|111)\b"),
]

SKIP_DIRS = {".git", "__pycache__", "node_modules"}
# 规则文件自身包含模式串，扫描时跳过
SKIP_FILES = {"tools/scrub_check.py", "tools/publish.sh"}
SKIP_EXT = {".npz", ".npy", ".npy.gz", ".png", ".jpg", ".pdf", ".gz", ".zip", ".so", ".bin"}

hits = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if os.path.splitext(fn)[1].lower() in SKIP_EXT:
            continue
        path = os.path.join(dirpath, fn)
        if os.path.relpath(path, ROOT) in SKIP_FILES:
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for name, pat in PATTERNS:
            for m in re.finditer(pat, text):
                snippet = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
                hits.append((os.path.relpath(path, ROOT), name, m.group(0), snippet))

if hits:
    print(f"扫描命中 {len(hits)} 处，禁止提交：")
    seen = set()
    for rel, name, val, snip in hits:
        key = (rel, name, val)
        if key in seen:
            continue
        seen.add(key)
        print(f"  [{name}] {rel}: {val!r}   … {snip[:90]}")
    print("\n处理方式：替换为占位（<DATA_DIR> / <CTR> / <NODE> / 10.0.0.1）后重新扫描")
    sys.exit(1)

print(f"scrub_check: 干净（扫描 {ROOT}）")
