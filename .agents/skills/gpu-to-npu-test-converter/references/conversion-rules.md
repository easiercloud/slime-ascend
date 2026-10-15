# Slime GPU-to-NPU conversion rules

Use this checklist selectively. Confirm each mapping against current source code and the nearest maintained NPU test.

## Preserve first

- Keep the feature under test, assertions, phase ordering, input and label meanings, reward mode, checkpoint semantics, and expected sample counts.
- Keep the source’s direct-execution entry point when CI runs the file with Python.
- Keep top-level static constants used by CI and resource inventory, including `MODEL_NAME`, `MODEL_TYPE`, and `NUM_GPUS`.
- Separate hardware-required changes from performance tuning. Explain every batch-size, parallelism, sequence-length, or memory change.

## Device and runtime mappings

| GPU concern | NPU action |
| --- | --- |
| SGLang defaults to CUDA | Add `--sglang-device npu` when the selected runtime supports and requires it. |
| NCCL-specific configuration | Use the repository’s HCCL path or helper; do not globally replace arbitrary strings. |
| `torch.cuda` calls | Use the project’s device abstraction or a verified `torch_npu` equivalent. Preserve synchronization and memory semantics. |
| CUDA allocator settings | Use a verified NPU allocator setting only when required; prefer caller-provided environment values. |
| CUDA graph tuning | Verify SGLang NPU support and choose a bounded value based on evidence. It is tuning, not a semantic conversion. |
| Distributed interfaces | Pass HCCL/Gloo/socket values through the established launcher or `extra_env_vars`; do not overwrite caller configuration without need. |
| Visible devices | Respect an existing `ASCEND_RT_VISIBLE_DEVICES`; do not assume eight devices when `NUM_GPUS` differs. |
| Ray propagation | Ensure required Ascend variables reach Ray workers through the shared launch path. |

Set `ASCEND_COREDUMP_SIGNAL=none` through the npu-test-autofix preparation unless the caller supplied a value. Avoid large per-test blocks of duplicated environment assignments when a shared helper already handles them.

## Assets and preparation

- Replace online `hf download` and dataset downloads with canonical environment-root paths described by `npu-test-autofix/references/test-assets.md`.
- Validate required child paths and fail with an actionable message when missing.
- Reuse the shared compatible Megatron conversion cache. Do not reconvert once per test.
- Preserve `input_key`, `label_key`, chat-template behavior, and dataset split. A locally stored dataset with a different schema is not an equivalent substitute.
- Never embed company CI paths, credentials, proxies, certificate values, or public endpoints.

## Resource adaptation

- Preserve logical invariants such as divisibility among world size, TP, PP, CP, EP, rollout engines, and batch sizes.
- Reduce response length, batch size, samples per prompt, or model parallelism only when the tested behavior remains reachable.
- Keep `NUM_GPUS` equal to the actual allocation requested by the launcher and CI.
- Put exceptionally long or resource-heavy tests in `nightly_CI` based on repository policy, not merely because they initially fail.

## Unsafe conversions

Stop and report a blocker when:

- The tested CUDA kernel or backend has no NPU implementation.
- Equivalent numerical behavior or tolerance cannot be justified.
- The required local model or dataset is unavailable.
- Conversion would require deleting assertions, skipping the core path, or replacing it with a mock that changes the test’s purpose.
- Hardware allocation, CANN, HCCL, or a required service is unavailable for validation.

## Static audit patterns

Inspect the new target for at least:

```bash
rg -n 'torch\.cuda|cuda:|nccl|hf download|hf_download_dataset|/root/(models|datasets)|/home/|pkill|ray stop|ASCEND_RT_VISIBLE_DEVICES' <target>
```

Every match must be removed or justified. Process termination is never an automatic conversion step; obtain user authorization before terminating shared processes.
