# Failure Analysis and Fix Policy

Read this file before changing code in response to NG tests.

## Classification

Assign every NG case one primary category:

- `PRODUCT_CODE`: implementation defect shared by supported environments.
- `NPU_ADAPTATION`: CUDA/NCCL/device assumptions, unsupported operators, HCCL setup, NPU memory behavior, or Ascend-specific launch logic.
- `TEST_SCRIPT`: invalid fixture, stale API, incorrect NPU expectation, or test harness defect.
- `WORKFLOW_SCRIPT`: discovery, preparation, conversion, runner, or reporting automation defect.
- `ENVIRONMENT`: missing dependency, CANN mismatch, unavailable device, port conflict, permissions, or service startup.
- `DEPENDENCY`: missing, incompatible, or incorrectly resolved Python/source dependency.
- `ASSET`: missing/inaccessible model, checkpoint, tokenizer, or dataset.
- `RESOURCE`: insufficient NPU/host memory, disk, or device capacity.
- `TIMEOUT_FLAKY`: nondeterministic failure or timeout without a proven code defect.

Do not use `ENVIRONMENT` as a convenient bucket. Cite exact log evidence and preflight results.

## Fix Boundaries

Allowed:

- Correct product or NPU adaptation code using evidence from logs and source.
- Update an NPU test when the upstream interface changed or an old expectation is invalid for supported NPU behavior.
- Adjust NPU launch configuration, topology, memory settings, configured paths, and HCCL/device setup.
- Add focused diagnostics or regression tests that clarify a root cause.

Forbidden:

- Translate a nonzero exit code into PASS.
- Catch and discard failures only to make a test pass.
- Remove assertions, reduce expected quality, or skip test logic without a documented semantic reason.
- Delete or rename a failing test to remove it from discovery.
- Overwrite unrelated user changes or use destructive Git commands.
- Continue beyond round 5.

## Iteration Discipline

Before each edit, identify a falsifiable root-cause hypothesis. State which log signature should disappear and what successful behavior should appear.

Prefer one shared fix for a root-cause group. Avoid unrelated refactors while tests are NG.

Apply the smallest change that can test the hypothesis. Limit the edit to the
necessary files and symbols, preserve unrelated user changes, and reject
formatting churn, opportunistic cleanup, broad renames, dependency upgrades, or
refactors that are not required for the observed failure.

If a failure is external, record the exact prerequisite and leave the case NG until it executes successfully. At finalization, classify it separately and mark it `SKIPPED_AFTER_MAX_ROUNDS`, never PASS.

## Change Record Template

```markdown
# Round N Changes

## Root Cause Group

- Issue ID:
- Cases:
- Category:
- Evidence:
- Hypothesis:

## Modifications

- File/symbol:
- Behavioral change:
- NPU-specific justification:
- Risk:
- Why this is the minimum sufficient change:

## Expected Verification

- Log signature removed:
- Expected successful signal:
- Commands:

## Diff Summary

- git diff --stat:
- Concise diff notes:
```
