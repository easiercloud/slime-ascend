---
name: slime-upstream-sync
description: Review and adapt upstream Slime changes for slime-ascend while preserving NPU support and using the repository's existing UT, ST, and Nightly CI cases.
---

# Slime upstream sync

Use this single-file procedure when updating slime-ascend from upstream Slime or preparing an NPU compatibility change. Read the repository's current `HANDOFF.md`, Git status, patch manifest, Dockerfile, quick installer, and existing NPU tests before editing. Record the exact starting commits and keep user changes intact.

## Scope and baseline

1. Identify the current slime-ascend base branch, the Slime submodule SHA, and the candidate upstream Slime SHA or tag. Fetch only the refs needed for comparison. Check `git merge-base`, `git log --left-right`, and `git diff --name-status` for the exact range.
2. Group upstream changes by runtime behavior, dependencies, patches, documentation, and tests. Inspect changed code before deciding which features or bug fixes apply to NPU. Do not copy every upstream GPU feature or GPU test by default.
3. Generate the changed upstream test inventory with python scripts/upstream_test_map.py --base OLD --target NEW. Review each GPU test candidate against existing NPU cases and record gaps before changing tests. For the current NPU CI scope, use the UT, ST, and `tests/tests_npu/nightly_CI` cases already present in slime-ascend. Add or alter test cases only for an explicitly requested gap or a concrete regression. Avoid version-suffixed duplicate UT files when an existing case covers the behavior.
4. Keep training on MegatronAdaptor plus TransformerEngineNPU. Preserve existing NPU behavior and check the actual dependency pins against the maintained Dockerfile, `scripts/quick_install.sh`, and `docs/get_started/quick_start.md`. For the A3 ARM64 CANN 9.0 image, the documented Python version is 3.12.
5. Before changing patch applications, compare each path with `patches/MANIFEST.toml` and the files on disk. Remove obsolete patch applications after checking that upstream already supplies the behavior. Keep needed NPU changes and verify that every referenced patch exists.

## Work sequence

1. Create or use an isolated feature branch. Capture `git status --short`, target refs, changed paths, and the planned migration in `HANDOFF.md`.
2. Migrate small, reviewable groups of changes. Update source, patch manifest, installer, Dockerfile, and the existing Quick Start together when dependency or patch behavior changes. Use the real wheel filename and architecture; avoid placeholder bundle paths.
3. Inventory required model weights, Megatron checkpoints, datasets, and NPU resources before device tests. Check known local asset roots and caches. If paths remain unresolved, report the precise missing assets while continuing work that does not need them.
4. When validation is requested, run the repository's existing UT first, then ST, then Nightly CI cases. Read the current runner scripts before invoking them because some scripts allocate NPUs and clean up processes. Save commands, environment pins, exit codes, and log paths.
5. On a shared host, programmatically assert that every selected physical NPU has no running process immediately before each device test; abort when any selected NPU is occupied. Printing npu-smi output alone is insufficient. Avoid the broad process cleanup in tests/tests_npu/st/st_run.sh; invoke selected cases directly and clean only owned resources. On failure, inspect the full traceback and relevant runtime logs, identify the failing component, make the smallest code or configuration correction, and rerun the affected case. Distinguish a missing asset or infrastructure failure from a code regression.
6. After each material change, update `HANDOFF.md` with the commit baseline, files changed, evidence, unresolved items, and next step. Review the final diff against the target branch and ensure the branch has a merge base with it. Resolve PR conflicts in the feature branch before pushing.
7. Push only the authorized feature branch. Never merge into `dev` or another protected branch without explicit authorization.

## Repository entry points

- Patch inventory: `patches/MANIFEST.toml`, `scripts/check_patch_manifest.py`, `build.sh`.
- Environment: maintained `docker/Dockerfile*`, `scripts/quick_install.sh`, `docs/get_started/quick_start.md`.
- Existing tests: `tests/tests_npu/ut`, `tests/tests_npu/st`, `tests/tests_npu/nightly_CI`, and their repository CI runners.
- Run record: `HANDOFF.md`.

This skill has no companion scripts, child skills, generated state files, or external templates.
