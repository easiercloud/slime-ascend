#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FSDP_PYTHON=${FSDP_PYTHON:-$(command -v "${BASE_PYTHON:-python}")}
train_devices=${FSDP_TRAIN_DEVICES:-2}
rollout_devices=${FSDP_ROLLOUT_DEVICES:-2}
engine_devices=${FSDP_DEVICES_PER_ENGINE:-$rollout_devices}
total_devices=$((train_devices + rollout_devices))
model_args=()
if [[ -n "${FSDP_MODEL_CONFIG:-}" ]]; then
  model_args+=(--fsdp-random-init --fsdp-model-config "$FSDP_MODEL_CONFIG")
  if [[ -n "${HF_CHECKPOINT:-}" ]]; then model_args+=(--hf-checkpoint "$HF_CHECKPOINT"); fi
else
  : "${HF_CHECKPOINT:?Set HF_CHECKPOINT or FSDP_MODEL_CONFIG to initialize the model}"
  model_args+=(--hf-checkpoint "$HF_CHECKPOINT")
fi
export SLIME_SOURCE_DIR=${SLIME_SOURCE_DIR:-$repo_root/slime}
export PYTHONPATH="$SLIME_SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s , 0 "$((total_devices - 1))")}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export SLIME_MIN_PORT=${SLIME_MIN_PORT:-30000}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60400-60499}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61400-61499}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
unset LD_PRELOAD
output=${FSDP_OUTPUT_DIR:-$repo_root/logs/fsdp2/online-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$output"
output=$(cd "$output" && pwd)
: "${FSDP_SAVE_DIR:?Set the trainer checkpoint root}"
: "${FSDP_WEIGHT_DIR:?Set a shared inference snapshot directory across restarts}"
resume_args=()
if [[ -n "${FSDP_LOAD_DIR:-}" ]]; then resume_args+=(--load "$FSDP_LOAD_DIR"); fi
set +e
timeout --signal=TERM --kill-after=20s "${FSDP_ONLINE_TIMEOUT_SECONDS:-2400}s" \
  "${BASE_PYTHON:-python}" "$repo_root/examples/fsdp2/run-fsdp-init.py" \
  --train-backend fsdp --fsdp-provider fsdpturbo --fsdp-sync-train \
  --fsdp-python-executable "$FSDP_PYTHON" \
  --fsdp-init-report "$output/result.json" "${model_args[@]}" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$train_devices" --num-gpus-per-node "$total_devices" \
  --rollout-num-gpus "$rollout_devices" --rollout-num-gpus-per-engine "$engine_devices" \
  --advantage-estimator "${FSDP_ADVANTAGE_ESTIMATOR:-grpo}" \
  --rollout-batch-size "${FSDP_ROLLOUT_BATCH_SIZE:-2}" --n-samples-per-prompt "${FSDP_SAMPLES_PER_PROMPT:-4}" \
  --global-batch-size "${FSDP_GLOBAL_BATCH_SIZE:-8}" --micro-batch-size "${FSDP_MICRO_BATCH_SIZE:-1}" \
  --lr 1e-4 --fsdp-weight-decay 0 --rollout-temperature 1 --distributed-backend hccl --distributed-timeout-minutes 10 \
  --num-rollout "${FSDP_NUM_ROLLOUT:-1}" --save "$FSDP_SAVE_DIR" --save-interval 1 "${resume_args[@]}" --rollout-max-response-len 96 --rollout-top-p 1 --rollout-top-k -1 \
  --prompt-data "${FSDP_PROMPT_DATA:-$repo_root/examples/fsdp2/resume-qa.jsonl}" --input-key prompt --label-key label \
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}' --rm-type f1 \
  --update-weight-mode full --update-weight-transport disk --update-weight-disk-dir "$FSDP_WEIGHT_DIR" \
  --update-weight-disk-keep-files --flush-cache-interval 1 --save-debug-rollout-data "$output/rollout-{rollout_id}.pt" \
  --sglang-device npu --sglang-attention-backend ascend --sglang-mem-fraction-static 0.4 \
  --sglang-context-length 2048 --sglang-max-total-tokens 8192 --sglang-max-running-requests 8 \
  --sglang-disable-cuda-graph --sglang-disable-radix-cache --sglang-router-port "${FSDP_ROUTER_PORT:-18377}" "$@" \
  > "$output/run.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$output/exit-code.txt"
tail -n 35 "$output/run.log"
exit "$result"
