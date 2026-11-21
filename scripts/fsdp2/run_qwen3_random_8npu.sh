#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for cann_script in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
  if [ -f "$cann_script" ]; then
    source "$cann_script"
    break
  fi
done
python3 "$script_dir/smoke_qwen3_random.py" --check-devices
run_root=${1:?Usage: run_qwen3_random_8npu.sh OUTPUT_ROOT}
mkdir -p -- "$run_root"
run_dir=$(mktemp -d "$run_root/qwen3-fsdp8-XXXXXXXX")
export OMP_NUM_THREADS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 "$script_dir/compat_faccessat2.py" python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 \
  "$script_dir/smoke_qwen3_random.py" --output-dir "$run_dir" 2>&1 | tee "$run_dir/torchrun.log"
test -s "$run_dir/summary.json"
printf 'Runtime smoke report: %s\n' "$run_dir/summary.json"
