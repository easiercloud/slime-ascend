#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${FSDP_PYTHON:?Set FSDP_PYTHON to the absolute training venv Python path}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local Qwen3-0.6B checkpoint directory}"
export SLIME_SOURCE_DIR=${SLIME_SOURCE_DIR:-$repo_root/slime}
export PYTHONPATH="$SLIME_SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export SLIME_MIN_PORT=${SLIME_MIN_PORT:-50600}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60400-60499}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61400-61499}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
unset LD_PRELOAD
output=${FSDP_OUTPUT_DIR:-$repo_root/logs/fsdp2/checkpoint-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$output"
output=$(cd "$output" && pwd)
load_args=()
if [[ -n "${FSDP_CHECKPOINT_LOAD:-}" ]]; then
  load_args+=(--fsdp-checkpoint-load "$FSDP_CHECKPOINT_LOAD")
fi
set +e
timeout --signal=TERM --kill-after=20s "${FSDP_CHECKPOINT_TIMEOUT_SECONDS:-2400}s" \
  "${BASE_PYTHON:-python}" "$repo_root/examples/fsdp2/run-fsdp-init.py" \
  --train-backend fsdp --fsdp-provider fsdpturbo --fsdp-checkpoint-check \
  --hf-checkpoint "$HF_CHECKPOINT" --fsdp-python-executable "$FSDP_PYTHON" \
  --fsdp-init-report "$output/result.json" --fsdp-checkpoint-save "$output/checkpoints" \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 --num-gpus-per-node 4 \
  --rollout-batch-size 2 --n-samples-per-prompt 4 --global-batch-size 4 --micro-batch-size 2 \
  --lr 1e-4 --rollout-temperature 0.8 --distributed-backend hccl --distributed-timeout-minutes 10 \
  "${load_args[@]}" > "$output/run.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$output/exit-code.txt"
tail -n 20 "$output/run.log"
exit "$result"
