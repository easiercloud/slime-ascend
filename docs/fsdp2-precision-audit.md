# FSDP2 precision and weight-export audit

The fixed-rollout runner can inspect the actual training state rather than
relying on configured dtype names:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 scripts/fsdp2/check_pinned_runtime.py
python3 scripts/fsdp2/run_slime_grpo.py \
  --output-root /path/to/persistent/runs \
  --rollouts 2 --precision-audit --operator-audit
```

Run this only in an existing pinned environment with eight explicitly allocated
Ascend devices. The default configuration is randomly initialized two-layer
Qwen3 with the original hidden/head/vocabulary dimensions and 1,630,556,672
parameters. Build the managed Slime patches before running the command.

The optional audit checks all 25 parameters on all eight ranks. Each current
microbatch must have observed BF16 module computation, FP32 sensitive operators,
BF16 parameter all-gather and FP32 gradient reduce-scatter. Step reports inspect
actual FP32 master shards, gradients before and after clipping, Adam tensors,
finite values and nonzero updates. Each exported version compares every live
master shard against the gathered FP32 state, safetensors bytes and independently
cold-loaded HF parameters. Missing phases, ranks, parameters or receipts fail
the runner.

Native FSDP may retain a full BF16 compute view after a no-grad forward. The
actor now calls the public reshard API after old-policy snapshots and export
probes, restoring canonical FP32 master shards before inspecting them. This
preserves the optimizer's original parameter objects and avoids treating a
temporary compute view as resident master storage. A real target-environment
regression also confirmed that the existing HF state-dict export preserved the
nontrivial FP32 master values; the initial audit failure did not establish HF
checkpoint corruption.

Validation used CANN 9.1, torch 2.10 and torch_npu 2.10.0.post6. Fresh replay of
all nine Slime patches and 19 CPU tests passed. Real small and original-dimension
Ray eight-rank runs each completed two updates with captured main exit 0.
For each run, versions 0/1/2 passed 600 rank-shard comparisons and 75 full-tensor
saved/cold-loaded comparisons. The portable one-device boundary regression is
`scripts/fsdp2/precision/reshard_case.py`; it requires an explicitly selected
single visible NPU for its NPU mode.

These acceptance runs use fixed tokens and rewards. They establish the observed
dtype and weight-handoff contracts for those runs. Independent numerical
references, online SGLang comparisons and optimizer/RNG recovery are separate
experiments. HF exports contain model weights and do not provide production
optimizer checkpoint recovery.
