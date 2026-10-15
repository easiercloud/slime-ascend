# Torch Memory Saver Allocation Lifecycle

Use this reference for Ascend colocate tests that offload Actor memory during
rollout and later restore it. Keep allocator/runtime OOM and TMS lifecycle
failures separate even when they occur in consecutive runs of the same case.

## Verified problem sequence

### Stage 1: Actor initialization OOM

- Environment: Qwen3-4B, Ascend colocate Actor/SGLang, `torch_npu 2.10.0`.
- Phase: SGLang had reserved rollout/KV-cache memory while Megatron DDP created
  parameter/gradient buffers.
- Signature: Actor initialization attempted a 7.49 GiB allocation with roughly
  2.66 GiB free and failed before useful training.
- Verified environment change: upgrade `torch_npu` to `2.10.0.post4`.
- Result: the initialization OOM disappeared and execution reached rollout.

This is version-scoped evidence, not a universal instruction to upgrade every
machine. Confirm CANN/PyTorch compatibility and reproduce the same allocator
signature before applying it. Record exact package versions in every run.

### Stage 2: TMS resume failure after rollout

After the runtime upgrade, the case completed initialization, weight update,
streaming generation, in-flight abort, and partial-sample recycling. Actor
wake-up then failed through:

```text
Actor wake_up
  -> torch_memory_saver.resume()
  -> cpu_backup should not be nullptr
  -> pybind11 / PyGILState_Check secondary crash
```

The pybind11/GIL failure is cleanup damage after the native exception; it is
not an independent root cause.

## Root cause

The affected `torch_memory_saver 0.0.8` native metadata tracked whether an
allocation enabled CPU backup and stored `cpu_backup`, but did not track whether
that allocation had actually been paused.

In a colocate lifecycle:

```text
allocate Actor tensor A
pause A -> backup/unmap A
allocate communication/rollout tensor B while paused
resume all registered allocations
```

B was active and had never executed `pause()`, so it had no CPU backup. The old
`resume()` nevertheless attempted to restore B and asserted that its backup
must exist. Repeated pause/resume calls had the same ambiguity.

## Native state-machine fix

Add `bool is_paused` to every allocation metadata record and maintain these
invariants:

```text
malloc:  is_paused = false
pause:   if already paused, skip; otherwise backup/unmap, then true
resume:  if not paused, skip; otherwise map/restore, then false
free:    if paused, do not unmap/free physical memory a second time
```

Also check the return value from `aclrtMapMem`. The verified source baseline was
SGLang commit `2a87cda308d3c5e38c6a0da74c2146146efbe625`; the modified files were:

- `contrib/torch_memory_saver/csrc/core.h`: add `AllocationMetadata::is_paused`;
- `contrib/torch_memory_saver/csrc/core.cpp`: initialize and enforce the state
  in `malloc`, `pause`, `resume`, and `free`.

Do not treat `disable_grad_buffers_cpu_backup` or
`disable_param_buffers_cpu_backup` as the fix. Those options change which
buffers are backed up but do not repair lifecycle classification.

## Slime integration boundaries

- Check a TMS nested region only after `_ensure_initialized()` and through
  `_impl._binary_wrapper.cdll.tms_get_interesting_region()`; do not read a
  nonexistent top-level `_binary_wrapper`.
- On NPU, keep SGLang `enable_memory_saver` disabled by default when Actor TMS
  owns colocate memory. Enable it only through an explicit environment override
  and validate that two managers do not control the same allocation lifecycle.
- Do not disable streaming, partial rollout, abort/recycle behavior, or Actor
  wake-up to make the test pass; those phases are the regression contract.

## Portable verification

Run the bundled regression on an idle allocated NPU after installing the
candidate TMS build:

```bash
python3 scripts/check_tms_pause_lifecycle.py hook
```

The test allocates A, pauses, allocates B from a separate pool while paused,
resumes, and verifies both tensors. Then run a colocate training case and
require all of these signals:

- Actor initialization and first `update_weights` complete;
- streaming generation aborts in-flight requests and recycles partial samples;
- Actor `wake_up`/TMS resume completes without `cpu_backup should not be nullptr`;
- all actor microbatches train and report loss/grad norm;
- a subsequent `update_weights` completes and the launcher exits cleanly.

Evidence levels:

- TMS state fix in Qwen3-4B PPO: `LOG_VERIFIED`; the preserved log shows
  `wake_up_time=9.62s`, actor train 8/8, and a later successful weight update.
- Qwen3-4B streaming partial rollout: `USER_VERIFIED` end-to-end PASS; preserve
  a dedicated success log in the next full run to upgrade it to `LOG_VERIFIED`.

## Deployment requirement

The verified native fix was installed from a locally rebuilt
`torch_memory_saver 0.0.8` wheel. Version `0.0.8` alone does not prove that the
wheel contains `is_paused`; inspect the source/build provenance or run the
portable regression.

Do not depend on `/tmp/slime-tms-fix-src` or copy a binary wheel blindly across
Python, architecture, CANN, or `torch_npu` combinations. On a new machine,
apply the state-machine change to the compatible locked source, rebuild the
wheel, record its checksum and build/runtime versions, run the focused
regression, and then run the full Slime case.
