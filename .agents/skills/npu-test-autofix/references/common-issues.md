# Common Issues

Apply a recipe only when its positive signatures match and its exclusions have
been checked. Record the recipe name and evidence in the issue ledger.

## PPO Fails After Graph Batch Capture Because of Signal Registration

- Category: `ENVIRONMENT`
- Applies to: Ascend PPO/ST processes using CANN graph capture with TMS or another signal-handling component.
- Positive signatures:
  - The rollout engine reaches `Capturing batches` or `Capture npu graph begin`.
  - An assertion or abnormal termination follows during or immediately after capture.
  - The process uses TMS or another component that conflicts with CANN trace signal registration.
- Check first:
  - `printenv ASCEND_COREDUMP_SIGNAL`
  - Inspect the first concrete error before the final `AssertionError`.
  - Confirm the launch environment reaches both the test process and Ray workers.
- Exclusions:
  - Do not use this recipe when the preceding error is NPU OOM.
  - Do not use it for `aclnnNonzeroV2`, `stream is captured`, or another explicit ACL/operator error.
  - Do not use it for HF/Megatron argument validation assertions.
- Minimal fix:

  ```bash
  export ASCEND_COREDUMP_SIGNAL=none
  ```

  Set it once in the shared launch environment and Ray runtime environment.
  Preserve an explicit user-provided value.
- Why: CANN trace signal registration can conflict with another signal handler.
  The value `none` disables the trace coredump signal set.
- Verification:
  - Confirm the variable is visible in the parent process and Ray runtime env.
  - Rerun only the affected PPO/ST case.
  - Capture must complete without the prior assertion or abnormal termination.
- Side effect: CANN trace will not generate `stackcore_tracer_*.txt` for covered
  signals. Record this diagnostic tradeoff in the issue.
- Local evidence: `scripts/ascend_script/run-qwen3-8B-npu-colocate.sh`.
- Vendor reference: Huawei Ascend CANN environment variable documentation for
  `ASCEND_COREDUMP_SIGNAL`.
- Status: `VERIFIED`

## Distributed Checkpoint BytesIO Error From Parallel-Layout Mismatch

- Category: `ASSET`
- Applies to: Megatron distributed checkpoints, especially MoE checkpoints
  containing `_extra_state` and sharded metadata.
- Positive signatures:
  - `TypeError: object of type '_io.BytesIO' has no len()`
  - The stack is loading or validating distributed checkpoint
    `_extra_state`/sharded metadata.
  - The checkpoint save topology differs from the requested load topology.
- Known example:
  - Saved as `tp=1, pp=4, ep=1, world_size=4`.
  - Loaded by Qwen3-30B-A3B R3 as `tp=4, pp=1, ep=8, world_size=8`.
- Checks:
  - Record the checkpoint's original TP, PP, EP, and world size from its
    conversion command, manifest, or distributed checkpoint metadata.
  - Extract the requested `--tensor-model-parallel-size`,
    `--pipeline-model-parallel-size`, `--expert-model-parallel-size`, and
    launch world size from the test.
  - Confirm the first checkpoint error occurs in `_extra_state` or sharded
    metadata handling; do not diagnose from the final `BytesIO` exception alone.
- Root cause: the checkpoint sharding/extra-state layout is incompatible with
  the load topology. The `BytesIO` exception is a secondary metadata failure.
- Minimal fixes, in preference order:
  1. Load using the checkpoint's original TP/PP/EP/world-size configuration.
  2. Convert the original HF checkpoint again using the exact topology required
     by the test and store it as a distinct compatible checkpoint.
- Do not:
  - Add `len()` handling to `BytesIO`.
  - Skip `_extra_state` or sharded metadata validation.
  - Rename or blindly reuse the incompatible checkpoint directory.
- Verification:
  - Record both saved and requested topology in the issue ledger.
  - Load the checkpoint with the selected compatible topology.
  - Confirm distributed metadata and `_extra_state` load successfully before
    starting the full PPO training step.
- Cache rule: convert once per model and compatible parallel layout, not merely
  once per model name. If origin topology is unknown, ask the user and do not
  mark the checkpoint reusable.
- Status: `USER_VERIFIED`

## Long-Response PPO Actor Logprob/Entropy OOM

- Category: `NPU_ADAPTATION` with secondary `RESOURCE` tag.
- Applies to: Slime PPO tests using dynamic batching with long responses and a
  large vocabulary.
- Positive signatures:
  - rollout, value, advantage, return, and old log-probability stages complete;
  - `actor_train` starts;
  - the traceback reaches `policy_loss_function`,
    `calculate_log_probs_and_entropy`, and `vocab_parallel_softmax`;
  - the failing allocation is consistent with a `[tokens, vocab_shard]`
    temporary, for example 4.35 GiB for Qwen3-4B TP2.
- Exclusions: do not use this recipe for actor initialization OOM, rollout
  engine OOM, FlashAttention tiling failures, or missing transport packages.
- Minimal verified Qwen3-4B TP2/CP2 configuration:

  ```text
  --max-tokens-per-gpu 8192
  --log-probs-chunk-size 2048
  ```

- Why: the token budget splits dynamic training microbatches, while logprob
  chunking bounds the largest token-by-vocabulary temporary without removing
  PPO computations.
- Verification: actor PPO training must progress beyond the old failure and
  the complete test must exit successfully.
- Status: `USER_VERIFIED`
