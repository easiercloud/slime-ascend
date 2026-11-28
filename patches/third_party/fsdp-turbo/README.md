# FSDPTurbo accelerator backport

This patch retains FSDPTurbo commit
`262f5ba4ae266d62b01c7e8a2e2bc46d46d1a0c3` and backports only the
`fsdp_turbo/utils/device.py` changes from these upstream commits:

- `46dd1d1e87170a2180ceab4c3e5623f34304fe2e`: prefer backend implementations
  over native `torch.accelerator` C bindings in torch 2.9 and later.
- `cef4e9948f273a7e194ee4343511bc0a2240dbb7`: delegate NPU memory APIs to
  the NPU allocator.

The resulting local reference commit is
`20409188e34fd44a5bd1d2908d00333525f4fb54`. Its changed function has the
same AST as the upstream implementation after `cef4e994`; only the long
upstream comment is shortened. No model, configuration, dependency or Docker
code changes. This patch does not contain a Docker compatibility helper.

The intended Slime environment remains the repository's
`docker/Dockerfile.a2.ubuntu22.04.cann910.latest`: CANN 9.1.0, Python 3.12,
torch 2.10.0, torch_npu 2.10.0.post6, Transformers 5.12.1 and Triton-Ascend
3.2.1. Do not install current FSDPTurbo main over it: current main requires
newer Transformers and Triton-Ascend. This backport is a candidate for that
environment; matching metadata does not establish NPU or training correctness.

Apply the patch to a clean checkout of the base commit using `git am`.
When reproducing it, verify the base and patch hash; the new commit ID can
depend on the applying committer's timestamp.

Validation completed locally: whitespace checks, upstream function AST
equivalence, and mocked NPU/CUDA accelerator delegation. The unmodified base
fails the NPU stream/event/prebound-memory checks, so the regression check
distinguishes the fix. Actual torch 2.10 NPU operations and eight-rank FSDP training have now
completed three optimizer updates with exit code 0 in the new Docker 26
container (2026-11-21, run qwen3-fsdp8-YE0LC1Ba). Slime rollout and GRPO
integration require separate validation.

SHA-256:

- Patch: `f3be7c40b138e09ebedb09d12ee06e4a21046860da44080297bec461b50cef31`
- Patched `device.py`: `31a15a15e01a0cbbff7b18acc2a32b16f0d3ddf49195871f274d069c908b7ba3`
- Full-index Git diff of `device.py`: `c7fd908a224041f0849ce71f73684244abad54c2a62d23db846c7b07798370c5`
