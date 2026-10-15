---
name: npu-test-autofix
description: Run, classify, diagnose, and iteratively repair Ascend NPU tests for any reinforcement-learning or distributed training framework, including Slime and VERL. Preserve test contracts while performing framework-profile-aware asset inventory, runtime preflight, baseline execution, issue tracking, and at most five repair rounds. Use when asked to execute NPU tests, improve NPU pass rates, investigate failed NPU cases, validate GPU-to-NPU conversions, or produce a test/fix audit report.
---

# NPU Test Autofix

Run framework-profile-aware NPU validation from the selected repository root.
The generic core owns discovery, classification, execution records, issue
ledger, bounded retries, and reporting. Profiles own framework launchers,
assets, dependency preparation, and failure recipes.

## Select a framework profile

Before preparation or execution, detect or explicitly select `slime`, `verl`,
or `generic`:

```bash
python3 .agents/skills/npu-test-autofix/scripts/detect_framework.py \
  --repo-root . --framework auto \
  --output .npu-test-autofix/framework-profile.json
```

Read [references/framework-profiles.md](references/framework-profiles.md) and
the selected profile section before running commands. Read
[references/failure-policy.md](references/failure-policy.md) before modifying
code and [references/issue-ledger.md](references/issue-ledger.md) before
diagnosing baseline failures or writing the final report. Read
[references/common-issues.md](references/common-issues.md) only when its
signature matches. Read [references/network-execution.md](references/network-execution.md)
before network-dependent preparation or tests.
Read [references/oom-playbook.md](references/oom-playbook.md) whenever the first
concrete failure is an NPU/host OOM, allocator failure, or memory-related idle
timeout. Classify the failing phase before changing memory parameters; actor
initialization and policy-logprob temporary OOMs require different recipes.
Read [references/tms-lifecycle.md](references/tms-lifecycle.md) when a colocate
test fails in `torch_memory_saver.pause()`/`resume()`, reports
`cpu_backup should not be nullptr`, or changes behavior after a `torch_npu`
upgrade. Treat runtime OOM removal and the newly exposed TMS lifecycle failure
as separate root causes.
Read [references/moe-thd-cp-tp.md](references/moe-thd-cp-tp.md) when a MoE case
with combined EP/CP/TP reports an empty-token DeepEP permute error, delayed
507015, or long-sequence SGLang/Megatron log-probability divergence. Diagnose
Actor dispatch and THD sequence layout separately.

For `slime`, additionally read `references/test-assets.md` and use the Slime
preparation scripts. For `verl` or `generic`, do not run Slime-specific
Megatron conversion, `command_utils.py`, SGLang, or Ray preparation unless the
repository imports and requires it.

## Inputs and preflight

- Resolve test roots, Python command, test class/runner, per-case timeouts,
  framework launcher, required local assets, services, NPU devices, and CANN.
- Default to recursive discovery below user-selected test roots; use repository
  conventions, not Slime directory names, to classify tests.
- Inventory assets before asking for paths. Accept framework-neutral root
  variables or explicit per-asset mappings; never guess, substitute, or
  download missing assets.
- When required assets remain unresolved after inventory, actively ask the user
  for their local roots or explicit paths and wait for the answer. Unset
  variables and failed local search are discovery results, not permission to
  finalize cases as missing-asset.
- Preserve user changes and do not stop shared processes without authorization.

## Workflow

1. Inventory:
   - Write `.npu-test-autofix/framework-profile.json` and an asset requirement
     inventory suitable for the selected profile.
   - For Slime, use `prepare_test_resources.py --inventory`. For VERL and
     generic repositories, inspect test/config references and record unresolved
     dynamic metadata explicitly rather than forcing Slime static constants.
   - Show the unresolved requirements to the user and ask for the local asset
     locations. For Slime, request the HF weights root, datasets root, and an
     existing Megatron checkpoint root or confirmation to use the conversion
     cache. Validate the response and ask a focused follow-up for remaining
     gaps. Stop affected execution while waiting in full mode.
   - Mark an asset unavailable only when the user confirms it is unavailable,
     supplied paths remain invalid after follow-up, or the user explicitly
     requested non-interactive execution.

2. Preflight:
   - Check Linux, Python, `npu-smi`, CANN, selected devices, repository status,
     profile dependencies, services, launch command, and asset readability.
   - Apply only the selected profile's environment preparation. Propagate NPU
     variables through the framework's actual worker launcher.

3. Baseline:
   - Generate a complete discovered-case inventory before code fixes.
   - Execute the framework's collection command for Python tests and the
     profile-approved command for scripts or distributed jobs.
   - Keep `PASS`, `NG`, timeout, skipped, missing-asset, and unexecuted states
     distinct. PASS rate is `PASS / baseline discovered tests * 100`.

4. Diagnose and repair:
   - Group NG cases by evidence-backed root cause: product, NPU adaptation,
     test, launcher, environment, dependency, asset, resource, or flaky/timeout.
   - Inspect the source GPU counterpart and selected profile before edits.
   - Create stable issue records, state a hypothesis and expected log change,
     make the smallest justified fix, and record every attempt.
   - Never weaken assertions, hide exceptions, delete tests, or label an
     unsupported path as passing.
   - Match the earliest concrete signature against the issue ledger and
     references before proposing a fix. Reuse an OOM recipe only when its
     phase, operator/tensor signature, topology, and exclusions match.

5. Rerun and report:
   - Rerun only the latest NG set through round 5 at most.
   - Audit issue coverage before finalizing.
   - Write a Chinese final report with framework profile, environment fingerprint,
     per-test contract, baseline/final counts, issue evidence, changes, and
     unresolved owner/action. Do not omit PASS, NG, skipped, or unexecuted cases.
   - After a verified repair, update the issue record first. Promote only the
     reusable diagnosis to a reference, including signatures, exclusions,
     minimal fix, preserved contract, and verification level. A one-off tuning
     value must not become a global default.

## Generic runner boundary

`scripts/run_test_round.py` is a generic discovery/recording runner for common
Python and shell tests. When VERL or another framework needs a nonstandard
collector or launcher, keep the result schema and issue ledger, but invoke the
profile-approved command rather than forcing a Slime test layout. Add a new
profile with evidence and tests before making it the default.
