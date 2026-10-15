# MoE DeepEP and THD CP/TP Alignment

Use this reference for large MoE training cases that combine expert, context,
and tensor parallelism. A delayed NPU 507015 and a log-probability mismatch can
come from different earlier defects; preserve the first concrete evidence.

## Verified topology and scope

- Model/case: Qwen3-30B-A3B NPU training.
- Actor topology: TP4 + CP2 + EP8.
- SGLang: DeepEP remains enabled.
- Actor default: standard `alltoall`; Actor DeepEP remains opt-in for focused
  compatibility validation.
- Verification level: `USER_VERIFIED` for the final full case. Preserve a
  dedicated success log in the next run before upgrading to `LOG_VERIFIED`.

## Stage 1: Actor DeepEP empty-token rank

### Positive signature

Some Actor EP ranks receive no real routed tokens. DeepEP returns a placeholder
hidden state with shape `[1, H]` while routing metadata has zero rows, then
Megatron fails with:

```text
Permute not possible:
inp.size(0) (1) must match routing_map.size(0) (0)
```

Because NPU work is asynchronous, the complete run may surface the same defect
later as an illegal-address `507015`. Do not diagnose only from the final stream
synchronization failure; search earlier rank logs for the shape mismatch.

### Minimal verified isolation

Decouple rollout and training dispatch backends:

```text
SGLang: --sglang-moe-a2a-backend deepep --sglang-deepep-mode auto
Actor:  --moe-token-dispatcher-type alltoall
```

In the maintained NPU case:

```text
SLIME_TEST_USE_DEEPEP=1          # SGLang, default enabled
SLIME_TEST_USE_ACTOR_DEEPEP=0    # Actor, default disabled
```

Set `SLIME_TEST_USE_ACTOR_DEEPEP=1` only for a focused Actor DeepEP rerun. This
is a component-boundary workaround, not proof that the Actor DeepEP empty-token
contract is fixed upstream.

### Exclusions

- Do not disable SGLang DeepEP when its rollout path is healthy.
- Do not classify generic MoE HCCL buffer errors or nonempty routing shape
  mismatches as this issue.
- Do not claim 507015 alone proves DeepEP; require the earlier routing evidence.

## Stage 2: CP2+TP4 tail padding corrupts response alignment

### Positive signature

After Actor alltoall removes the dispatch crash:

- reference and actor forward complete;
- short sequences have matching log probabilities;
- long sequences diverge severely, for example Megatron `-12.56` versus
  SGLang `-0.274`;
- the mismatch concentrates on the final sample of dynamic microbatches.

This pattern excludes a general weight mismatch or a broad CP arithmetic error.

### Root cause

THD packs every real sequence with `2 * cp_size` alignment for mirrored CP
partitioning. The faulty TP alignment extended the final real sequence with an
extra `2 * CP` tail block. That changed the physical mirrored chunk size of the
last sequence, while response extraction still derived offsets from its real
length. Logits and target tokens therefore referred to different positions.

Dynamic batching makes each microbatch's last real sample susceptible.

### Required layout invariant

Keep logical and physical padding distinct:

1. Pad every real sequence independently to `2 * cp_size`.
2. Compute the smallest global tail needed so each CP-local packed length is
   divisible by TP.
3. Represent that tail as a separate physical-only sequence with zero real
   tokens; never extend a real sequence for TP alignment.
4. Append an unchanged logical `cu_seqlens` endpoint and a larger physical
   `cu_seqlens_padded` endpoint for the padding-only sequence.
5. Use the same physical layout and CP partition indices for tokens and every
   token-level tensor, including loss masks.
6. Assert `packed_tokens / cp_size` is divisible by `tp_size`.

The verified helper computes:

```text
sequence_alignment = 2 * cp_size
local_alignment = lcm(2, tp_size)
global_alignment = cp_size * local_alignment
tail_pad = -physical_total mod global_alignment
```

### Exclusions

- Do not attach TP tail padding to the final real sequence.
- Do not change `get_responses()` offsets to match corrupted physical padding;
  response semantics must remain based on real token lengths.
- Do not accept “forward completed” as validation; compare token-aligned
  SGLang and Megatron log probabilities on long sequences.

## Verification ladder

Run increasingly broad checks:

1. Single long sequence: response length 4809; SGLang `-0.18879`, Megatron
   `-0.18978`; training completes.
2. Sixteen long sequences: 15 dynamic microbatches; SGLang `-0.26798`,
   Megatron `-0.26999`; forward, backward, and optimizer complete.
3. Full 30B case: complete AIME eval; two 16-sample, 8192-token rollouts; two
   Actor training and weight-update cycles; successful Ray job exit.

For a new version, preserve per-microbatch real/padded boundaries and log the
maximum SGLang/Megatron discrepancy. Re-run CP-only, TP-only, and CP+TP cases so
the padding-only sequence does not regress existing THD coverage.
