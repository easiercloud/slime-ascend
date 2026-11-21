#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 "$script_dir/check_pinned_runtime.py"
python3 "$script_dir/smoke_qwen3_random.py" --check-devices
export OMP_NUM_THREADS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
python3 "$script_dir/run_slime_grpo.py" --output-root "${1:?Usage: run_slime_grpo_8npu.sh OUTPUT_ROOT}"
