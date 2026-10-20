# 昇腾 NPU 快速开始

本页对应 Slime v0.4.0 的 Ascend 适配，训练侧使用 MegatronAdaptor 和 TransformerEngineNPU。以下 A3 x86_64 组合采用 Python 3.12；A2 和 aarch64 需选用相应的 CANN、torch_npu 与内核包，并在目标机器上验证。

## 版本与前提

| 组件 | A3 x86_64 配置 |
| --- | --- |
| 基础镜像 | Ubuntu 22.04，`quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12` |
| Python | **3.12** |
| CANN | 9.1.0 |
| PyTorch / torch_npu | 2.10.0 / 2.10.0.post4 |
| SGLang / sgl-kernel-npu | v0.5.15.post1 / 2026.9.0.post10 |
| Megatron-LM | 4b4acac9 |
| MegatronAdaptor / TransformerEngineNPU | f707a3b6 / a717fcda |
| slime submodule | v0.4.0 (`0b0c277d`) |

Python 版本必须为 3.12。安装脚本会在开始时检查版本，并校验所安装 wheel 的 cp312 标记。精确依赖以 [quick_install.sh](../../scripts/quick_install.sh) 为准，Slime 补丁顺序以 [MANIFEST.toml](../../patches/MANIFEST.toml) 为准。

## Docker 安装（A3）

从本仓库根目录构建，Dockerfile 使用同一份安装脚本和补丁清单，不另行维护第二套补丁应用命令：

```bash
git clone --recurse-submodules <slime-ascend-repo-url>
cd slime-ascend
docker build -f docker/Dockerfile.a3.ubuntu22.04.cann91.py312 \
  -t slime-ascend:slime-v0.4.0-cann9.1-py3.12 .
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
  slime-ascend:slime-v0.4.0-cann9.1-py3.12
```

Docker 构建依次运行 [quick_install.sh](../../scripts/quick_install.sh) 和 [build.sh](../../build.sh)。安装脚本对 SGLang、Megatron-LM、Megatron-Bridge、mbridge 及 torch-memory-saver 应用 [third_party](../../patches/third_party) 中实际存在的补丁；`build.sh` 根据 [MANIFEST.toml](../../patches/MANIFEST.toml) 应用 [npu-compat](../../patches/npu-compat) 和 [upstream-fixes](../../patches/upstream-fixes) 中登记的 Slime 补丁。Slime v0.4.0 已自带 NPU accelerator，仓库补丁仅保留 Ray NPU 资源、MegatronAdaptor 和经验证仍需的兼容逻辑。

## 本地安装

在具有匹配驱动和 CANN 9.1.0 的机器上创建 Python 3.12 环境：

```bash
conda create -n slime-ascend python=3.12
conda activate slime-ascend
git clone --recurse-submodules <slime-ascend-repo-url>
cd slime-ascend
export CANN_INSTALL_PATH=/usr/local/Ascend
export NPU_DEVICE=A3
bash scripts/quick_install.sh
```

`quick_install.sh` 根据设备类型与机器架构选择 Python 3.12 / CANN 9.1.0 的 sgl-kernel-npu 包。网络受限时，可用 `SGL_KERNEL_NPU_URL` 指向同规格的本地镜像地址。安装脚本和 Dockerfile 引用的补丁均需存在于对应文件夹；不要把不存在的补丁路径加入命令。

`build.sh` 会清理并重置 `slime/` 子模块的工作区，然后应用补丁。请只在没有未保存子模块修改的检出目录中运行；查看预期顺序可先执行 `bash build.sh --dry-run`。

## 训练与排查

训练入口见 [特性与模型支持](../supported_models.md) 和仓库中的 `scripts/run-*.sh`。运行前用 `python --version` 确认 Python 3.12，并用 `npu-smi info` 确认设备可见。遇到补丁冲突时，先核对 [MANIFEST.toml](../../patches/MANIFEST.toml) 中的基线 commit 和实际补丁文件，再检查 `git am --show-current-patch=diff` 的输出。
