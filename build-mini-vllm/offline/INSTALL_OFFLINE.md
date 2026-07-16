# 离线安装说明（INSTALL_OFFLINE）

本课程按两个层级交付离线能力（对应 Prompt 第四章）。

## Level A —— 零第三方依赖离线学习（默认，随时可用）

核心课程是**纯 Python 标准库**实现，不需要 PyTorch、numpy、Pillow，也不需要 pip。
只要有 Python 3.11+ 和一个浏览器，就能完全离线学习：

```bash
# 1) 校验离线资源齐全（HTML/CSS/JS/tokenizer/config/workload/manifest）
python3 course.py verify-offline-bundle

# 2) 环境自检（附加离线检查）
python3 course.py doctor --offline

# 3) 启动网页（仅本机，无需网络）
python3 course.py serve
#    或直接双击 docs/index.html 离线打开（少数需 localhost 的功能除外）

# 4) 运行第一个机制模拟器
python3 course.py run 0
python3 course.py check 0
```

**锁定环境**（Level B 严格验收所用）：

| 项目          | 值                    |
|---------------|-----------------------|
| 操作系统      | macOS 14+             |
| 架构          | Apple Silicon / arm64 |
| Python        | 3.11                  |
| 内存下限      | 8 GB                  |
| 计算下限      | CPU（MPS 仅可选加速） |

## Level B —— 完整依赖离线安装（可选）

即使核心不需要第三方包，我们仍提供标准的「本地 wheel + require-hashes」安装流程，
以便你在受控环境里演练真正的离线安装，并为将来加入可选 PyTorch 加速做准备：

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python3 -m pip install \
  --no-index \
  --find-links offline/wheels/macos-arm64 \
  --require-hashes \
  -r offline/requirements-lock.txt
```

由于 `offline/requirements-lock.txt` 目前没有必需依赖，这条命令会**立即成功**
（installed 0 packages），并且 `--no-index` 保证**绝不回退到公网**。

## 可选加速：PyTorch（不属于核心必需项）

轨道 B 的 tiny model 默认用纯 Python 实现，可在仅 CPU 上离线运行。若你希望用真实
PyTorch 张量加速（例如用 MPS）：

1. 在**有网络**的机器上，为锁定环境（macOS 14+ / arm64 / cpython-3.11）下载 wheel：

   ```bash
   python3 -m pip download torch \
     --dest offline/wheels/macos-arm64 \
     --only-binary=:all: --platform macosx_14_0_arm64 \
     --python-version 3.11 --implementation cp
   ```

2. 计算 SHA-256 并登记到 `offline/manifests/wheels.json` 的 `optional`（或转入 `required`）：

   ```bash
   shasum -a 256 offline/wheels/macos-arm64/torch-*.whl
   ```

3. 之后即可完全离线安装。运行时仍**禁止联网**（不下载模型、不用 HF Hub）。

> **WARNING：** 本课程运行时严格禁止任何网络请求（远程 API / CDN / 在线模型 / 在线字体）。
> `python3 course.py verify-offline-bundle` 会扫描并阻止这些情况。

## 删除与重建环境

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.11 -m venv .venv && source .venv/bin/activate
# 核心无需 pip install 任何东西即可运行
python3 course.py doctor --offline
```
