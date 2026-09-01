#!/bin/bash
set -euxo pipefail
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
NPU_DEVICE=${NPU_DEVICE:=A3}

# Pinned dependency contract for Slime v0.3.2 / Ascend validation.
CANN_VERSION=9.0.0
SGLANG_COMMIT=0.5.15.post1
SGLANG_BASE_COMMIT=0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
SGLANG_ROUTER_PACKAGE=sglang-router==0.3.2
# SGLang v0.5.15.post1's NPU CI pins this exact release.
SGL_KERNEL_NPU_VERSION=2026.6.2
# 2026.6.2 release 内 sgl_kernel_npu wheel 的 metadata 尚为 2026.6.1。
SGL_KERNEL_NPU_WHEEL_VERSION=2026.6.1
SGL_KERNEL_NPU_FIX_COMMIT=9765e27532b8f854717eba1d3fb9a0bb97a0c887
MEMFABRIC_HYBRID_VERSION=1.0.8

MEGATRON_COMMIT=1dcf0dafa
MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
MEGATRON_ADAPTOR_COMMIT=f707a3b6eb9981532c3419c08ea40a8bb5962af5
MEGATRON_BRIDGE_COMMIT=35b4ebfc486fb15dcc0273ceea804c3606be948a
TRANSFORMER_ENGINE_NPU_COMMIT=47d604491f2fb6f284fa470352b127ca20e5eaea
MegatronAdaptor_COMMIT=${MEGATRON_ADAPTOR_COMMIT}
TransformerEngineNPU_COMMIT=${TRANSFORMER_ENGINE_NPU_COMMIT}
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

echo "1. install SGLang from source"
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v${SGLANG_COMMIT}
test "$(git rev-parse HEAD)" = "${SGLANG_BASE_COMMIT}"
mv python/pyproject.toml python/pyproject.toml.backup
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e python[all_npu]
cd ..

echo "2. install torch & torch_npu & other basic packages"
pip install torch==2.10.0 torch_npu==2.10.0.post4 torchvision==0.25.0 setuptools==81.0.0
pip install "memfabric-hybrid==${MEMFABRIC_HYBRID_VERSION}"
pip install "${SGLANG_ROUTER_PACKAGE}" --force-reinstall
python -c "import sglang_router"

echo "3.install sgl-kernel-npu from release whl"
KERNEL_ARCH=$(uname -m)
case "${KERNEL_ARCH}" in
    x86_64|aarch64) ;;
    *) echo "unsupported sgl-kernel-npu architecture: ${KERNEL_ARCH}" >&2; exit 1 ;;
esac
case "${NPU_DEVICE}" in
    A3) KERNEL_DEVICE=a3 ;;
    A2) KERNEL_DEVICE=910b ;;
    *) echo "unsupported NPU_DEVICE: ${NPU_DEVICE}" >&2; exit 1 ;;
esac
SGL_KERNEL_NPU_URL="https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGL_KERNEL_NPU_VERSION}/sgl-kernel-npu-${SGL_KERNEL_NPU_VERSION}-torch2.10.0-py311-cann9.0.0-${KERNEL_DEVICE}-${KERNEL_ARCH}.zip"
wget --no-check-certificate "${SGL_KERNEL_NPU_URL}"
unzip sgl-kernel-npu*.zip
pip install torch_memory_saver*.whl
pip install sgl_kernel_npu*.whl
pip install deep_ep*.whl
# For using deep_ep: Link to the deep_ep_cpp.*.so file
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so
# Avoid "ln -s" failure due to existing deep_ep symlink which breaks "cd -", hence split it out separately.
cd -

# (Optional) Confirm whether the import can be successfully
python -c "import deep_ep; print(deep_ep.__path__)"

echo "3.1 rebuild torch-memory-saver with the registered lifecycle patch"
git clone https://github.com/sgl-project/sgl-kernel-npu.git sgl-kernel-npu-src
cd sgl-kernel-npu-src
git checkout ${SGL_KERNEL_NPU_FIX_COMMIT}
git apply ../slime-ascend/patches/third_party/sgl-kernel-npu/0001-fix-torch-memory-saver-allocation-lifecycle.patch
git apply ../slime-ascend/patches/third_party/sgl-kernel-npu/0002-fix-tms-sglang-cuda-graph-api-and-preload-mode.patch
cd contrib/torch_memory_saver/python
python3 setup.py clean --all
python3 setup.py bdist_wheel
pip install --force-reinstall dist/torch_memory_saver*.whl
cd ../../../..

echo "4. install mbridge & Megatron-Bridge"
git clone https://github.com/ISEEKYAN/mbridge.git 
cd mbridge && git checkout ${MBRIDGE_COMMIT}
pip install -e . && cd ..

git clone https://github.com/fzyzcjy/Megatron-Bridge.git
cd Megatron-Bridge && git checkout ${MEGATRON_BRIDGE_COMMIT} && cd ..
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
pip install -e . && cd ..

echo "5. install MA&TE"
git clone https://gitcode.com/Ascend/MegatronAdaptor.git && \
cd MegatronAdaptor/ && git checkout ${MegatronAdaptor_COMMIT} && \
pip install -e . && cd ..
git clone https://gitcode.com/Ascend/TransformerEngineNPU.git&& \
cd TransformerEngineNPU/ && git checkout ${TransformerEngineNPU_COMMIT} && \
pip install -e . && cd ..

echo "6. install triton_ascend"
pip uninstall triton triton_ascend -y
pip install triton_ascend==3.2.1

echo "7. install slime-ascend"
cd slime-ascend/
pip install -e . 

echo "8. apply npu patches"
cd ../sglang
git am ../slime-ascend/patches/third_party/sglang/*
cd ../Megatron-LM/
git am ../slime-ascend/patches/third_party/megatron/*
cd ../Megatron-Bridge/
git am ../slime-ascend/patches/third_party/megatron-bridge/*
cd ../mbridge/
git am ../slime-ascend/patches/third_party/mbridge/*
# Apply TransformerEngineNPU patches (fixed path - was TransformerEngineNPU, should be transformer_engine_npu)
cd ../TransformerEngineNPU/
git am ../slime-ascend/patches/third_party/transformer_engine_npu/*
# Slime v0.3.2 当前没有 MegatronAdaptor 补丁条目。
echo "9. return to slime-ascend directory"
cd ../slime-ascend/

echo "9. build slime"
bash build.sh
# Create folder to store logs
mkdir -p logs
