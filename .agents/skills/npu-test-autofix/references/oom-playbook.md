# NPU OOM Diagnosis and Repair Playbook

Use this playbook only after the earliest concrete failure is confirmed to be
memory-related. `NPU out of memory` is a symptom class, not a root cause. First
identify the failing phase and largest allocation, then select the smallest
change that preserves the test contract.

## Evidence required before tuning

Record the following in the issue ledger:

- test id, model, precision, device type/count, and available memory;
- DP/TP/PP/CP/EP topology and actor/rollout device split;
- response/total length statistics, batch sizes, dynamic token budget, and
  micro-batch size;
- earliest OOM traceback, failing operator, requested allocation, and last
  completed behavior phase;
- allocator state when available: allocated, reserved, free, and non-framework
  device memory;
- whether the failure is deterministic and whether an earlier configuration
  reached a later phase.

Do not diagnose from a final Ray exception when a worker log contains an
earlier operator or allocator failure. Change one root-cause group per attempt.

## Phase classification

| Phase | Typical evidence | First controls to inspect | Do not confuse with |
| --- | --- | --- | --- |
| Rollout startup/capture | SGLang model load, KV cache, graph capture, memory pool | rollout TP, static memory fraction, running requests, graph capture, token limits | Actor training OOM |
| Actor/critic initialization | `MegatronTrainRayActor.init` before a training batch | actor device split, TP/PP/EP, optimizer state, precision, offload | Policy-loss temporary OOM |
| Attention/RoPE forward | FlashAttention/TND/BSHD operator and sequence shapes | sequence format, CP metadata, microbatch tokens, kernel support | Allocator fragmentation |
| Policy logprob/entropy | policy loss -> logprob/entropy -> vocab softmax | dynamic token budget, `log-probs-chunk-size` | Actor initialization OOM |
| Backward/optimizer | backward or `optimizer.step`/FusedAdam allocation | micro batch, recompute, gradient/optimizer precision, offload | Forward logits temporary |
| Weight update/checkpoint | gather, reshard, serialization, distributed checkpoint | update mode, shard topology, staging buffers, placement | Missing checkpoint |
| Ray/host memory | plasma spill, host OOM killer, serialization allocation | transport, object lifetime, object-store and host capacity | NPU allocator OOM |

If the phase cannot be established, add diagnostics and keep the case NG. Do
not apply a recipe based only on model name or allocation size.

## Repair decision order

1. Remove an invalid dependency/configuration only when it is outside the
   behavior contract. Example: optional GPU NIXL transport is not required by
   an NPU PPO algorithm test.
2. Bound the allocation nearest the failing operator. Prefer logprob chunking
   for token-by-vocabulary temporaries, microbatch/token limits for batch-scaled
   activations, and parallel sharding for persistent model/optimizer state.
3. Preserve model, dataset schema, response length, PPO phases, assertions,
   sample count, and distributed invariants unless the issue record justifies
   why a reduction keeps the contract reachable.
4. Run a focused test proving the old signature disappeared and the next
   required phase completed. A later OOM is a new root cause and issue record.
5. Return the case to the bounded suite queue. Focused PASS evidence does not
   replace the full synchronization release gate.

## Runtime upgrade exposes a later lifecycle failure

When upgrading `torch_npu` removes an allocator OOM and the same case later
fails in `torch_memory_saver.resume()`, create two issue records. Do not claim
that the new exception is an OOM regression:

- `torch_npu 2.10.0 -> 2.10.0.post4` removed a verified Qwen3-4B colocate actor
  initialization OOM in the current environment;
- the later `cpu_backup should not be nullptr` signature came from missing
  per-allocation pause state in TMS and required an independent native fix.

Read [tms-lifecycle.md](tms-lifecycle.md) for signatures, the `is_paused` state
machine, exclusions, portable verification, and deployment requirements.

## Known recipe: long-response PPO policy logprob/entropy

Positive signature:

- rollout, reward, critic/value, advantage/return, reference log-probs, and old
  actor log-probs complete;
- actor training starts;
- traceback enters `policy_loss_function`,
  `calculate_log_probs_and_entropy`, and `vocab_parallel_softmax`;
- the multi-GiB allocation matches `[microbatch tokens, vocab shard]`.

Verified Qwen3-4B TP2/CP2 configuration:

```text
--max-tokens-per-gpu 8192
--log-probs-chunk-size 2048
```

The token budget splits dynamic microbatches. Chunking computes the same token
log-probabilities and entropy in smaller blocks, trading more launches for a
lower peak temporary. This does not apply to actor initialization, rollout KV
cache, attention tiling, optimizer-step, or host-memory failures.

Verification level: `USER_VERIFIED`. Evidence:
`.npu-test-autofix/issues/ISSUE-008-ppo-policy-logprob-oom.md`.

## Knowledge promotion rules

- `OBSERVED`: signature and hypothesis exist; no successful repair yet.
- `USER_VERIFIED`: user reports a complete successful rerun, without a
  preserved auditable success log.
- `LOG_VERIFIED`: a preserved log proves required phases and clean exit for one
  environment/topology.
- `REGRESSION_VERIFIED`: repeated or multi-topology coverage passes.

Promote an issue resolution only when it has positive signatures, exclusions,
minimal change, preserved contract, exact success signals, and a verification
level. Keep case-specific paths and attempt history in the issue ledger.
