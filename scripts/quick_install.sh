#!/bin/bash
set -euxo pipefail
python -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Python 3.12 is required, got {sys.version}"'
grep -q '^version=9.0.0$' /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
test "$(uname -m)" = aarch64
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
python "${REPO_ROOT}/scripts/check_patch_manifest.py"
pip install --upgrade pip packaging setuptools==80.10.2
DEPS_ROOT=${SLIME_ASCEND_DEPS_ROOT:-"$(dirname "${REPO_ROOT}")"}

# Pinned to the CANN 9.0.0 / Python 3.12 A3 aarch64 image.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd "${DEPS_ROOT}"

echo "0. install PyTorch and torch_npu"
pip install torch==2.10.0 torchvision==0.25.0
pip install torch_npu==2.10.0.post4

echo "1. install SGLang from source"
git clone --depth 1 --branch v0.5.15.post1 https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.15.post1
test "$(git rev-parse HEAD)" = 0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
git am "${REPO_ROOT}/patches/third_party/sglang/0001-slime-v0.5.15.post1-npu.patch" \
    "${REPO_ROOT}/patches/third_party/sglang/0002-tms-adapter-forward-cuda-graph-arguments.patch"
mv python/pyproject.toml python/pyproject.toml.backup
mv python/pyproject_npu.toml python/pyproject.toml
# Keep SGLang's NPU metadata on the Megatron-Bridge tested transformers pin.
sed -i 's/transformers==5.12.1/transformers==5.3.0/' python/pyproject.toml
grep -q '"transformers==5.3.0"' python/pyproject.toml
pip install -e python[all_npu]
cd ..

echo "2. install torch & torch_npu & other basic packages"
pip install pybind11==3.0.3 transformers==5.3.0 triton-ascend==3.2.1
pip install memfabric-hybrid==1.0.8
pip install sglang-router==0.3.2 --force-reinstall
python -c "import sglang_router"

echo "3. install CANN 9.0 / Python 3.12 A3 aarch64 prebuilt kernel wheels"
cd /tmp
wget --no-check-certificate https://gitcode.com/gcw_8jFUVqc5/sglang-kernel-npu-builds/releases/download/py312-cann9.0-a3/sgl-kernel-npu-torch2.10.0-py312-cann9.0-a3-aarch64.zip -O sgl-kernel-npu-a3.zip
unzip sgl-kernel-npu-a3.zip
pip install torch_memory_saver*.whl
pip install sgl_kernel_npu*.whl
pip install deep_ep*.whl
cd "$(pip show deep-ep | awk '/^Location:/ {print $2}')"
ln -s deep_ep/deep_ep_cpp*.so

echo "3a. rebuild only the patched torch-memory-saver wheel"
git clone https://github.com/sgl-project/sgl-kernel-npu.git /tmp/sgl-kernel-npu-src
cd /tmp/sgl-kernel-npu-src
git checkout 9765e27532b8f854717eba1d3fb9a0bb97a0c887
git am "${REPO_ROOT}"/patches/third_party/sgl-kernel-npu/0001-fix-torch-memory-saver-allocation-lifecycle.patch
git am "${REPO_ROOT}"/patches/third_party/sgl-kernel-npu/0002-fix-tms-sglang-cuda-graph-api-and-preload-mode.patch
cd contrib/torch_memory_saver/python
python3 setup.py clean --all
python3 setup.py bdist_wheel
pip install --force-reinstall dist/torch_memory_saver*.whl
cd "${DEPS_ROOT}"

echo "3.1 rebuild torch-memory-saver with the registered lifecycle patch"
git clone https://github.com/sgl-project/sgl-kernel-npu.git sgl-kernel-npu-src
cd sgl-kernel-npu-src
git checkout ${SGL_KERNEL_NPU_FIX_COMMIT}
git am "${REPO_ROOT}"/patches/third_party/sgl-kernel-npu/0001-fix-torch-memory-saver-allocation-lifecycle.patch
git am "${REPO_ROOT}"/patches/third_party/sgl-kernel-npu/0002-fix-tms-sglang-cuda-graph-api-and-preload-mode.patch
cd contrib/torch_memory_saver/python
python3 setup.py clean --all
python3 setup.py bdist_wheel
pip install --force-reinstall dist/torch_memory_saver*.whl
cd ../../../..

echo "4. install mbridge & Megatron-Bridge"
git clone https://github.com/ISEEKYAN/mbridge.git 
cd mbridge && git checkout 89eb10887887bc74853f89a4de258c0702932a1c
git am "${REPO_ROOT}/patches/third_party/mbridge/0001-fix-device-type.patch"
pip install -e . && cd ..

git clone https://github.com/fzyzcjy/Megatron-Bridge.git
cd Megatron-Bridge && git checkout 35b4ebfc486fb15dcc0273ceea804c3606be948a
git am "${REPO_ROOT}/patches/third_party/megatron-bridge/0001-ascend-megatron-bridge-patch-for-slime-v0.3.2.patch" \
    "${REPO_ROOT}/patches/third_party/megatron-bridge/0002-adapts-to-transformer-5.3.0-for-VL-model.patch" \
    "${REPO_ROOT}/patches/third_party/megatron-bridge/0003-megatron-Bridge-for-Qwen.patch"
cd ..
pip install 'nvidia-modelopt[torch]==0.46.1' --no-build-isolation

git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
cd Megatron-LM/ && git checkout 4b4acac9a1d28ea6829c8d4f566d75698a21249d && \
git am "${REPO_ROOT}/patches/third_party/megatron/megatron-npu-v040-rebased.patch" && \
pip install -e . && cd ..

echo "5. install MA&TE"
git clone https://gitcode.com/Ascend/MegatronAdaptor.git && \
cd MegatronAdaptor/ && git checkout f707a3b6eb9981532c3419c08ea40a8bb5962af5 && \
pip install -e . && cd ..
git clone https://gitcode.com/Ascend/TransformerEngineNPU.git&& \
cd TransformerEngineNPU/ && git checkout 2b8529f94c3fd10497bf772bdc5b448b7ed4659a && \
pip install -e . && cd ..

echo "6. install triton_ascend"
pip uninstall triton triton_ascend -y
pip install triton-ascend==3.2.1 \
    --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ \
    --trusted-host triton-ascend.osinfra.cn --no-cache-dir

echo "7. install slime-ascend"
# Keep the submodule on the official v0.4.0 commit and replay the registered
# format-patch series before installing the Python project.
cd "${REPO_ROOT}"
bash build.sh
cd slime/
pip install -e .

echo "8. installation complete"
cd "${REPO_ROOT}"
# Create folder to store logs
mkdir -p logs
