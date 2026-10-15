# Issue Ledger

Maintain one record per distinct root cause under
`.npu-test-autofix/issues/ISSUE-NNN.md`. Reuse the same issue ID when one
root cause affects multiple tests or survives multiple rounds.

Every baseline NG test must appear in at least one issue record. A test may
appear in multiple records only when logs prove independent failures.

## Primary Categories

- `PRODUCT_CODE`: defect in Slime behavior shared across supported devices.
- `NPU_ADAPTATION`: Ascend device, operator, HCCL, topology, or NPU launch adaptation.
- `TEST_SCRIPT`: defect in a UT/ST test, fixture, assertion, or NPU test adaptation.
- `WORKFLOW_SCRIPT`: defect in discovery, preparation, conversion, runner, or reporting automation.
- `ENVIRONMENT`: CANN, permissions, ports, services, shell setup, or machine state.
- `DEPENDENCY`: missing, incompatible, or incorrectly resolved Python/source dependency.
- `ASSET`: missing or invalid model, checkpoint, tokenizer, or dataset.
- `RESOURCE`: insufficient NPU/host memory, disk, device count, or other capacity.
- `TIMEOUT_FLAKY`: nondeterminism or timeout without a proven defect in another category.

Assign one primary category. Add secondary tags when useful, but do not hide a
code or script defect under `ENVIRONMENT`.

## Record Template

```markdown
# ISSUE-NNN: Short title

- Category:
- Secondary tags:
- Status: OPEN | RESOLVED | UNRESOLVED | EXTERNAL_BLOCKED | SKIPPED_AFTER_MAX_ROUNDS
- First seen: round N
- Cases:
- Failure signature:
- Evidence/logs:
- Root-cause group:

## Attempts

### Round N / Attempt M

- Hypothesis:
- Evidence supporting hypothesis:
- Minimal change scope:
- Files/symbols changed:
- Exact behavioral change:
- Commands run:
- Result:
- New evidence:
- Decision: keep | revise | revert-own-change | blocked

## Final Status

- Outcome:
- Resolution evidence:
- Remaining blocker:
- Why attempts did not resolve it:
- Recommended owner/action:
- Residual risk:
```

Record failed attempts as well as successful ones. If an attempted change is
disproved, remove only that change when it is safe to distinguish from
pre-existing user work, and record the removal as `revert-own-change`.

## Retention and cleanup

Keep the active ledger focused on issues that affect runtime behavior, NPU
correctness, dependencies, resources, or reproducibility. A path, launcher,
fixture, or test-only import issue that has been verified fixed may remain in a
historical report, but should be removed from the active issue directory after
the affected test passes. Do not carry resolved test-path records into later
rounds or reports unless they are needed for audit provenance.

Retain concise records for functional fixes such as allocator behavior, TMS
lifecycle, graph-capture compatibility, and phase-specific OOMs until the
corresponding patch has been promoted or explicitly rejected.

## Minimal Change Gate

Before every edit:

1. State one falsifiable hypothesis and the expected log change.
2. Name the smallest files and symbols needed to test it.
3. Check the existing diff and preserve unrelated user changes.
4. Avoid formatting, renaming, dependency upgrades, and refactors unrelated to the hypothesis.

After every edit:

1. Inspect the focused diff and remove accidental churn.
2. Run the narrowest meaningful verification before a broad rerun.
3. Keep the change only when evidence supports it.
4. Append the outcome to the issue record before starting another attempt.

## Final Inventory

The final report must contain category totals and two issue tables:

- Resolved: issue ID, category, affected tests, root cause, minimal fix, verification.
- Unresolved: issue ID, category, affected tests, attempts made, latest evidence,
  blocker, why unresolved, and recommended owner/action.

If the baseline has no NG tests, state that no issue records were required and
that no failure-driven code changes were made.
