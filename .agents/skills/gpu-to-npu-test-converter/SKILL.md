---
name: gpu-to-npu-test-converter
description: Convert GPU, CUDA, or NCCL tests from any reinforcement-learning or distributed training framework into behavior-equivalent Ascend NPU tests. Preserve assertions, test contracts, data semantics, distributed invariants, and CI conventions while adapting devices, launchers, assets, and runtime configuration. Use for Slime, VERL, or another framework when asked to port, migrate, generate, copy, or batch-convert tests from GPU/CUDA to NPU/Ascend/CANN/HCCL.
---

# GPU to NPU Test Converter

Convert a selected test into the smallest behavior-equivalent NPU counterpart.
Keep framework-neutral conversion policy separate from framework profiles.

## Select a framework profile

Resolve the repository root, source tests, target scope, and framework before
editing. Use `slime`, `verl`, or `generic`:

```bash
python3 .agents/skills/npu-test-autofix/scripts/detect_framework.py \
  --repo-root . --framework auto
```

Read [references/framework-profiles.md](references/framework-profiles.md) and
[references/generic-conversion-rules.md](references/generic-conversion-rules.md)
for every conversion. Only for the `slime` profile, also read the Slime-specific
[references/conversion-rules.md](references/conversion-rules.md),
[references/grpo-npu-casebook.md](references/grpo-npu-casebook.md), and Slime
asset reference in `../npu-test-autofix/references/test-assets.md`.

Do not apply a Slime, VERL, Ray, Megatron, SGLang, vLLM, FSDP, or DeepSpeed
recipe to another framework without a matching error signature and source-code
evidence.

## Required inputs

- Build the complete source-to-target candidate list before editing a requested
  directory or batch.
- Infer test class and target location from the selected profile and nearby NPU
  tests. Do not assume `tests/tests_npu`, `UT/ST`, `MODEL_NAME`, or `NUM_GPUS`
  exists outside Slime.
- Discover models, datasets, configuration files, launchers, and services from
  the test and its helpers. Never guess asset paths or download missing assets.
- Inspect a maintained NPU analogue in the same framework when one exists.

## Workflow

1. Inventory and contract:
   - Record source, target, framework profile, test class, launcher, world-size
     topology, models, datasets, services, and closest NPU analogue.
   - Record the behavior contract: capability, phases, assertions, expected
     outputs, checkpoint semantics, sample counts, failure conditions, and
     distributed invariants.
   - Keep a per-case feature description derived from implementation and
     helpers, not its filename.

2. Adapt minimally:
   - Preserve assertions, branches, input meaning, reward semantics, data
     schema, and CI intent.
   - Apply only evidenced device, collective, launcher, asset, and resource
     mappings from the selected profile.
   - Prefer environment-configured asset roots and framework configuration over
     developer paths or repeated test-local setup.
   - Record every intentional reduction of model size, batch size, sequence
     length, sample count, parallelism, or timeout and why the contract remains
     reachable.

3. Static review:
   - Compile/import-check when feasible.
   - Audit CUDA-only APIs, NCCL-only settings, public downloads, hardcoded asset
     paths, destructive cleanup, unconditional environment overrides, and
     profile-incompatible launch arguments.
   - Compare source and target side by side; account for every semantic
     difference.

4. Hand off:
   - Invoke `npu-test-autofix` with the same framework profile and repository
     root.
   - Start with asset and test inventory. Use `--include` for focused runs;
     never bypass the runner or call an unexecuted case PASS.
   - Let `npu-test-autofix` own baseline execution, diagnosis, issue records,
     bounded retries, and the execution report.
   - When handoff resolves a migration-specific dependency or resource
     difference, add a concise casebook rule only after execution succeeds.
     Link to autofix evidence instead of duplicating logs, and do not promote
     test-specific tuning to every converted case.

## Deliverables

Report the framework profile and evidence, source-to-target map, behavior
contract, intentional differences, models/datasets/services/topology, static
checks, exact execution commands, and validation status for every selected
case. Write user-facing deliverables in Chinese; preserve commands, paths,
identifiers, log text, and machine-readable keys when translation would break
them.

Do not call a conversion complete before static review succeeds. Do not call it
NPU-validated until the target actually executes successfully.
