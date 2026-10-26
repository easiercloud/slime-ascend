# 昇腾（NPU）快速使用

本文档从搭建环境开始，说明 Slime main@2026-10-22 的 NPU 环境、补丁构建和训练入口。

## 硬件支持说明

本页的 Dockerfile 和安装脚本面向 **Atlas 800T A3、Linux ARM64（aarch64）**，使用 CANN 9.0.0 和 Python 3.12。训练侧使用 MegatronAdaptor 与 TransformerEngineNPU。

## 组件版本

| 组件 | A3 ARM64 构建版本 |
| ---- | ---------------- |
| 基础镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-ubuntu22.04-py3.12`（`linux/arm64`） |
| Python / CANN | 3.12 / 9.0.0 |
| torch / torch_npu / torchvision | 2.10.0 / 2.10.0.post4 / 0.25.0 |
| Megatron-LM | `4b4acac9a1d28ea6829c8d4f566d75698a21249d` |
| MegatronAdaptor | `f707a3b6eb9981532c3419c08ea40a8bb5962af5` |
| TransformerEngineNPU | `2b8529f94c3fd10497bf772bdc5b448b7ed4659a` |
| Megatron-Bridge | `35b4ebfc486fb15dcc0273ceea804c3606be948a` |
| mbridge | `89eb10887887bc74853f89a4de258c0702932a1c` |
| SGLang | `v0.5.21`（`e00930c5489053f26d86b179cee0d087f846acbb`） |
| sgl-kernel-npu / torch-memory-saver / DeepEP | 使用 `py312-cann9.0-a3` 预编译包；仅从 `9765e27532b8f854717eba1d3fb9a0bb97a0c887` 重建已打补丁的 torch-memory-saver |
| sglang-router / memfabric-hybrid | `0.3.2` / `1.0.8` |
| transformers / triton-ascend | `5.12.1` / `3.2.1` |
| nvidia-modelopt | `0.46.1` |
| Slime submodule | `main@2026-10-22` (`18868a79b0b083a093798a94ed0f8cb4f71b9c14`) |

CANN 9.0.0 / Python 3.12 / A3 ARM64 使用 [sgl-kernel-npu 预编译包](https://gitcode.com/gcw_8jFUVqc5/sglang-kernel-npu-builds/releases/download/py312-cann9.0-a3/sgl-kernel-npu-torch2.10.0-py312-cann9.0-a3-aarch64.zip)。仅对仓库中的 torch-memory-saver 补丁从固定源码重新打包。版本及补丁路径见 [Dockerfile](../../docker/Dockerfile.a3.ubuntu22.04.cann900.latest)、[quick_install.sh](../../scripts/quick_install.sh) 和 [MANIFEST.toml](../../patches/MANIFEST.toml)。本组合尚未完成整镜构建验证。

## 使用指南

模型与训练配置（训推分离、colocate、全异步）见 [特性与模型支持](../supported_models.md)。

## 环境配置

### 方式一：Docker

从本仓库根目录构建 A3 ARM64 环境：

```bash
cd {slime-ascend-root-path}
docker build --platform linux/arm64 \
  -f docker/Dockerfile.a3.ubuntu22.04.cann900.latest \
  -t slime-ascend:9.0.0-a3-ubuntu22.04-py3.12-main-20261022 .
```

```bash
docker run -it --rm \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v "$(pwd)":/workspace/slime-ascend \
  -w /workspace/slime-ascend \
  slime-ascend:9.0.0-a3-ubuntu22.04-py3.12-main-20261022
```

Dockerfile 按分步格式安装 SGLang、PyTorch、NPU 内核、mbridge、Megatron-Bridge、Megatron-LM、MegatronAdaptor、TransformerEngineNPU 和 triton-ascend；对存在的 third_party 补丁逐个执行 `git am`，最后运行 `build.sh` 回放 Slime 补丁并安装项目。构建上下文须包含仓库 Git 元数据与 Slime submodule。

### 方式二：本地安装脚本

在 A3 ARM64、CANN 9.0.0 环境中安装 Python 3.12：

```bash
conda create -n slime-ascend python=3.12
conda activate slime-ascend
git clone --recurse-submodules https://gitcode.com/Ascend/slime-ascend.git
cd slime-ascend
bash scripts/quick_install.sh
```

脚本与 Dockerfile 使用相同的源码提交和 third_party 补丁。脚本开始时检查 Python 3.12、CANN 9.0.0 与 aarch64。其他架构和设备需要独立的版本组合与验证。

### 补丁构建（build.sh）

`build.sh` 根据 [MANIFEST.toml](../../patches/MANIFEST.toml) 检查补丁文件和 SHA256，然后在 Slime main@2026-10-22 子模块上按顺序应用补丁：

```bash
bash build.sh --dry-run
bash build.sh
```

`build.sh` 会清理并重置 `slime/` 子模块的工作区；运行前请保存该目录内自己的修改。

## 训练与排查

训练入口见仓库中的 `scripts/run-*.sh`。运行前用 `python --version`、`npu-smi info` 检查环境。补丁冲突时核对 [MANIFEST.toml](../../patches/MANIFEST.toml) 的基线与实际文件，再检查 `git am --show-current-patch=diff`。
