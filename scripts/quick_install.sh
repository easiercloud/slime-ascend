# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -euxo pipefail

python -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Python 3.12 is required, got {sys.version}"'
grep -q '^version=9.1.0$' /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
test "$(uname -m)" = aarch64
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
python "${REPO_ROOT}/scripts/check_patch_manifest.py"
pip install --upgrade pip packaging setuptools==80.10.2
DEPS_ROOT=${SLIME_ASCEND_DEPS_ROOT:-"$(dirname "${REPO_ROOT}")"}

# Pinned to the CANN 9.1.0 / Python 3.12 A3 aarch64 image.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd "${DEPS_ROOT}"

echo "0. install PyTorch and torch_npu"
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.10.0.post6 \
    --extra-index-url https://ascend.devcloud.huaweicloud.com/pypi/simple/

echo "1. install SGLang from source"
git clone --depth 1 --branch v0.5.21 https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.21
test "$(git rev-parse HEAD)" = e00930c5489053f26d86b179cee0d087f846acbb
for patch in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch sglang-deterministic.patch; do
    git apply "${REPO_ROOT}/patches/third_party/sglang/${patch}"
done
git add -A
git -c user.name=slime-sync -c user.email=slime-sync@localhost commit -m 'Apply Slime v0.5.21 SGLang integration patches'
git am "${REPO_ROOT}/patches/third_party/sglang/0001-v0.5.21-tms-forward-graph-args.patch"
mv python/pyproject.toml python/pyproject.toml.backup
mv python/pyproject_npu.toml python/pyproject.toml
# SGLang v0.5.21 imports transformers.image_processing_backends (5.12.1).
grep -q '"transformers==5.12.1"' python/pyproject.toml
pip install -e python[all_npu]
cd ..

echo "2. install torch & torch_npu & other basic packages"
pip install pybind11==3.0.3 transformers==5.12.1 \
    https://github.com/triton-lang/triton-ascend/releases/download/v3.2.1/triton_ascend-3.2.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
pip install \
    https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/v1.2.1/20260923.4/memfabric_hybrid-1.2.1-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl \
    memfabric-zbal==1.1.3
pip install sglang-router==0.3.2 --force-reinstall
python -c "import sglang_router"

echo "3. install CANN 9.1 / Python 3.12 A3 aarch64 prebuilt kernel wheels"
cd /tmp
wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.9.0.post11/sgl-kernel-npu-2026.9.0.post11-torch2.10.0-py312-cann9.1.0-a3-aarch64.zip -O sgl-kernel-npu-a3.zip
echo "1fec018f7842f505cdf13555a0bd948e4f5b6c8b0ef2c6a4dc800d1df8373e97  sgl-kernel-npu-a3.zip" | sha256sum -c -
unzip sgl-kernel-npu-a3.zip
pip install torch_memory_saver*.whl
pip install sgl_kernel_npu*.whl attentions*.whl
pip install deep_ep*.whl
cd "$(pip show deep-ep | awk '/^Location:/ {print $2}')"
ln -s deep_ep/deep_ep_cpp*.so

echo "3a. rebuild only the patched torch-memory-saver wheel"
git clone https://github.com/sgl-project/sgl-kernel-npu.git /tmp/sgl-kernel-npu-src
cd /tmp/sgl-kernel-npu-src
git checkout 9765e27532b8f854717eba1d3fb9a0bb97a0c887
git am "${REPO_ROOT}"/patches/third_party/sgl-kernel-npu/0001-fix-torch-memory-saver-allocation-lifecycle.patch
git am "${REPO_ROOT}"/patches/third_party/sgl-kernel-npu/0002-fix-tms-sglang-cuda-graph-api-and-preload-mode.patch
git am "${REPO_ROOT}"/patches/third_party/sgl-kernel-npu/0003-fix-tms-unaligned-allocations-and-nested-regions.patch
cd contrib/torch_memory_saver/python
python3 setup.py clean --all
python3 setup.py bdist_wheel
pip install --force-reinstall dist/torch_memory_saver*.whl
cd "${DEPS_ROOT}"

echo "4. install mbridge & Megatron-Bridge"
git clone https://github.com/ISEEKYAN/mbridge.git
cd mbridge
git checkout 89eb10887887bc74853f89a4de258c0702932a1c
git am "${REPO_ROOT}/patches/third_party/mbridge/0001-fix-device-type.patch"
pip install -e .
cd ..

git clone https://github.com/fzyzcjy/Megatron-Bridge.git
cd Megatron-Bridge
git checkout 35b4ebfc486fb15dcc0273ceea804c3606be948a
git am "${REPO_ROOT}/patches/third_party/megatron-bridge/0001-ascend-megatron-bridge-patch-for-slime-v0.3.2.patch" \
    "${REPO_ROOT}/patches/third_party/megatron-bridge/0002-adapts-to-transformer-5.3.0-for-VL-model.patch" \
    "${REPO_ROOT}/patches/third_party/megatron-bridge/0003-megatron-Bridge-for-Qwen.patch"
cd ..
pip install 'nvidia-modelopt[torch]==0.46.1' --no-build-isolation

git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
cd Megatron-LM/
git checkout 4b4acac9a1d28ea6829c8d4f566d75698a21249d
git am "${REPO_ROOT}/patches/third_party/megatron/megatron-npu-v040-rebased.patch" \
    "${REPO_ROOT}/patches/third_party/megatron/0002-ascend-async-checkpoint-worker.patch"
pip install -e .
cd ..

echo "5. install MA&TE"
git clone https://gitcode.com/Ascend/MegatronAdaptor.git
cd MegatronAdaptor/
git checkout eb0e5043438dd753099432bc3b3d8e5916ed1844
git am "${REPO_ROOT}/patches/third_party/megatron-adaptor/0001-async-preload-pickle.patch"
pip install -e .
cd ..

git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
cd TransformerEngineNPU/
git checkout 2b8529f94c3fd10497bf772bdc5b448b7ed4659a
pip install -e .
cd ..

echo "6. install triton_ascend"
pip uninstall triton triton_ascend -y
pip install \
    https://github.com/triton-lang/triton-ascend/releases/download/v3.2.1/triton_ascend-3.2.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl \
    --no-cache-dir

echo "7. install slime-ascend"
# Keep the submodule on the pinned upstream main commit and replay the registered
# format-patch series before installing the Python project.
cd "${REPO_ROOT}"
bash build.sh
cd slime/
pip install -e .

echo "8. installation complete"
cd "${REPO_ROOT}"
# Create folder to store logs
mkdir -p logs
