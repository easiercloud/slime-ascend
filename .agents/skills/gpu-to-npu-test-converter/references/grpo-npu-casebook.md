# GRPO GPU-to-NPU Casebook

Use these verified cases as diagnostic references, not unconditional patches.
Start from the earliest concrete error, match the positive signature, check the
exclusions, and apply the smallest relevant change. Preserve the source test's
behavior contract.

## Case A: Qwen3-8B GRPO colocate

### A1. Short smoke plan creates zero scheduler steps

- Scope: launch configuration only; no product source change.
- Positive signature:
  - Megatron fails while constructing `OptimizerParamScheduler`.
  - The assertion is `assert self.lr_decay_steps > 0`.
  - The available rollout samples are fewer than `global_batch_size`.
- Verified failing values:

  ```text
  num_rollout=1
  rollout_batch_size=4
  n_samples_per_prompt=1
  global_batch_size=16
  ```

- Root cause: the short smoke workload cannot form a complete global batch, so
  the derived training/scheduler plan has zero effective decay steps.
- Minimal verified fix:

  ```text
  global_batch_size: 16 -> 4
  ```

  Keep the existing constant LR style if already present; adding
  `--lr-decay-style constant` is not the fix when the failing command already
  uses it.
- Invariant to check:

  ```text
  rollout_batch_size * n_samples_per_prompt >= global_batch_size
  ```

  Also check divisibility required by the current Slime/Megatron version.
- Exclusions:
  - Do not modify scheduler source code.
  - Do not use this recipe for a nonzero-step checkpoint mismatch.
- Verification: require at least one complete rollout, actor optimizer step,
  checkpoint save when configured, and successful launcher/Ray exit.

### A2. Resume rejects a changed training plan

- Scope: checkpoint-resume policy only; no product source change.
- Positive signature:

  ```text
  OptimizerParamScheduler: class input value 12 and checkpointvalue 4
  for total number of iterations do not match
  ```

- Root cause: the Slime checkpoint contains actor, optimizer, scheduler, and
  iteration state. Changing `num_rollout` from 1 to 3 changes the newly derived
  total scheduler iterations, while the loaded checkpoint still records the old
  plan. The values `4` and `12` are iteration-plan values, not model sizes; the
  model remains Qwen3-8B.
- Minimal verified fix when the changed plan is intentional:

  ```bash
  --override-opt-param-scheduler
  ```

  Continue loading model/optimizer/data state, but use the current launch plan
  for scheduler parameters.
- Do not add the override automatically:
  - A fresh run with no Slime training checkpoint does not need it.
  - An unchanged resume plan does not need it.
  - If the user intends to reproduce the old schedule exactly, restore the old
    rollout/batch/scheduler arguments instead.
  - If the user intends weights-only initialization, use the framework's
    weights-only/no-optimizer restore mode rather than hiding all resume
    mismatches with this flag.
- Verification: confirm the expected checkpoint iteration is loaded, subsequent
  optimizer steps execute, and a new checkpoint can be saved.

### A3. GRPO smoke success is not training-effectiveness success

`n_samples_per_prompt=1` can make within-prompt GRPO advantages and gradients
zero. It is acceptable only when the behavior contract is process smoke or
end-to-end integration. Do not use it for a test that asserts meaningful GRPO
learning. Preserve multiple samples per prompt in such tests, or explicitly
record the semantic reduction.

## Case B: Qwen3-VL-8B Geo3K multi-turn GRPO

The verified smoke topology used 8 Ascend 910B3 devices: 4 actor devices, 4
rollout devices, TP2/DP2/PP1/CP1, BSHD, micro batch 1, three rollouts, 4 prompts
times 8 samples, and global batch 32. Treat these as a conservative validation
baseline, not universal performance settings.

### B1. Actor-to-SGLang visual weight name conversion

- Category: product code.
- Positive signature:

  ```text
  ValueError: Unknown parameter name:
  module.module.vision_model.patch_embed.proj.weight
  ```

  or another `module.module.vision_model.*` parameter during weight update.
- Check first:
  - Inspect the actual Actor module class and parameter names.
  - Compare every converted name and shape with the target HF/SGLang index.
  - Check for duplicate conversion function definitions where a later
    definition shadows the intended visual branch.
- Verified root cause: this Qwen3-VL Bridge constructs the visual tower directly
  in HF layout. The active converter lacked the visual branch.
- Minimal verified behavior:

  ```text
  module.module.vision_model.* -> model.visual.*
  ```

  Keep the visual tensor unchanged; do not reorder QKV/MLP merely because
  another Qwen-VL converter does. Also map the top-level language embedding,
  final norm, and LM head names required by the local layout.
- Exclusions:
  - Do not reuse this prefix-only mapping unless the target visual tower is
    confirmed to be HF-layout compatible.
  - Do not hardcode hidden sizes, layer counts, or developer paths.
- Focused verification: validate all visual names and shapes against the target
  checkpoint index before launching; then require weight update completion and
  rollout entry.

### B2. Fast image processor exceeds the NPU tensor-rank limit

- Category: NPU launch adaptation.
- Positive signature:

  ```text
  AclNN_Parameter_Error(EZ1001):
  The self tensor cannot be larger than 8 dimensions
  ```

  with traceback in a Qwen-VL fast image processor `patches.reshape` or
  `aclnnInplaceCopy`.
- Minimal verified fix:

  ```bash
  --sglang-disable-fast-image-processor
  ```

- Why: use the CPU/PIL slow image path and avoid the unsupported high-rank NPU
  temporary tensor. Image/sample semantics remain unchanged; throughput may
  decrease.
- Exclusion: do not apply when the first failing operator is unrelated to image
  preprocessing.

### B3. Megatron Core and Bridge RoPE keyword mismatch

- Category: dependency compatibility source adaptation.
- Positive signature:

  ```text
  Qwen3VLTextRotaryEmbedding.forward() got an unexpected keyword argument
  'packed_seq'
  ```

  The caller may also pass `cp_group`.
- Minimal verified fix: wrap the Bridge RoPE `forward` so it accepts
  `packed_seq`, `packed_seq_params`, and `cp_group`, and omits only the
  unsupported compatibility keywords when calling the original implementation.
- Boundary: this was verified with CP1, where RoPE uses already prepared
  position IDs. Do not silently discard `cp_group` for CP greater than 1 without
  a semantic test. Remove the wrapper if the target Bridge already supports the
  current Core signature.

### B4. THD packed FlashAttention crashes or stalls

- Category: NPU kernel/configuration adaptation.
- Positive signatures:
  - Ascend plog identifies `FlashAttentionScore_*` with
    `CCU instruction address check error` or error `507015`.
  - Alternatively, the THD packed ref-forward stalls with live ranks and no
    AICore work after smaller token limits avoid the crash.
- Intermediate finding: reducing `max_tokens_per_gpu` from 16384 to 4096 avoided
  the immediate 16K kernel crash but did not eliminate the later packed
  collective stall.
- Minimal verified stable configuration:

  ```bash
  --qkv-format bshd
  --micro-batch-size 1
  ```

  Remove the THD dynamic-batch arguments for this baseline.
- Exclusions:
  - Do not label generic HTTP 503, OOM, or Python exceptions as this kernel
    issue.
  - Do not claim 4096 alone is the final fix.
- Tradeoff: BSHD padding reduces throughput but preserves the GRPO objective.

### B5. Boolean advanced-index assignment triggers NonZero MTE error

- Category: NPU operator source adaptation.
- Positive signature:

  ```text
  aclnnNonzeroV2
  The write address of the MTE instruction is out of range
  ```

  with traceback at vocab-parallel cross entropy boolean assignments such as:

  ```python
  masked_target[target_mask] = 0
  predicted_logits[target_mask] = 0.0
  ```

- Minimal verified NPU-only fix: replace those assignments with mathematically
  equivalent `torch.where` expressions. Preserve all other cross-entropy
  calculations and return values.
- Exclusions:
  - Do not globally replace boolean indexing.
  - Recheck the helper signature after Megatron upgrades.
  - If the target torch-npu/CANN version no longer reproduces the operator
    failure, omit the override.

### B6. Parallel topology, batch, KL, and sequence-parallel stability

Use these changes only after matching their evidence:

| Failure/evidence | Verified minimal response |
| --- | --- |
| BSHD smoke stalls across a larger rollout/global-batch split | Align one rollout to one global batch: 4 prompts × 8 samples = global batch 32. |
| TP4 collective stalls; TP1 parameter plus DDP grad buffer OOMs a 61 GiB device | Use TP2/DP2 as the conservative 4-actor-device topology. |
| `--use-kl-loss` with `--kl-loss-coef 0.00` produces NaN loss/gradients through `0 * NaN` | Remove `--use-kl-loss`; do not compute a zero-weight KL term. |
| Ascend dump locates a `4096 x 12288` Slice MTE out-of-range write in sequence halving | Remove `--sequence-parallel` for the verified TP2 baseline. |

Do not convert production-scale GPU values into these smoke values without
recording the resource reduction and confirming that the test's assertions do
not require the original scale.

## Case C: Qwen3-4B long-response disaggregate PPO

Treat these as two independent failures and verify them in order. Removing the
transport failure does not prove PPO training is healthy.

### C1. GPU NIXL transport is unavailable in the NPU runtime

- Category: NPU test configuration / dependency difference.
- Positive signature: `_split_train_data_by_dp` calls
  `ray.put(..., _tensor_transport="nixl")`, followed by
  `ModuleNotFoundError: No module named 'nixl'` or an import failure in
  `nixl._api`.
- Check first: verify `import nixl`, `from nixl._api import nixl_agent,
  nixl_agent_config`, and package metadata in the exact test environment.
- Minimal verified fix when NIXL is not part of the NPU test contract: remove
  `--rollout-data-transport nixl` from the NPU case and use Slime's default
  object-store transport.
- Exclusions: do not apply this to a test whose purpose is NIXL or to external
  PD/Mooncake transport. Do not create a fake `nixl` module.
- Verification: rollout data reaches the trainer and value, advantage, return,
  reference log-probability, and old actor log-probability stages execute.

### C2. Actor policy logprob/entropy vocab softmax OOM

- Category: NPU memory configuration.
- Positive signature: PPO reaches `actor_train`, then fails through
  `policy_loss_function` -> `calculate_log_probs_and_entropy` ->
  `vocab_parallel_softmax`, with a multi-GiB allocation such as 4.35 GiB.
- Cause: long responses and dynamic batching create a large
  `[tokens, vocab_shard]` temporary during actor logprob/entropy recomputation.
- Minimal verified configuration for the Qwen3-4B TP2/CP2 case:

  ```text
  --max-tokens-per-gpu 8192
  --log-probs-chunk-size 2048
  ```

  The first setting bounds dynamic microbatch tokens; the second computes
  logprob/entropy in token chunks to reduce peak temporary memory.
- Preserve response length, PPO phases, sample count, assertions, and topology
  unless separate evidence requires changing them. Chunking trades additional
  launches for lower peak memory; it does not reduce the computation contract.
- Verification: require rollout, reward, critic/value, advantage/return, actor
  old log-probs, actor optimizer work, and clean test completion.

## Cross-case decision order

1. Separate environment failures from training failures. A Ray GCS
   `Operation not permitted` socket error is execution permission, not model
   code.
2. Fix argument invariants and intentional checkpoint-plan mismatches before
   editing source.
3. For VLM rollout failures, separate image preprocessing, weight-name
   conversion, compatibility signatures, and NPU kernel/operator failures.
4. Use focused static/name/shape/operator checks before another expensive
   end-to-end run.
5. Require a complete behavior signal: rollout, actor optimizer step, weight
   synchronization where applicable, and clean launcher/Ray success.
