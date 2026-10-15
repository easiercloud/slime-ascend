---
name: slime-upstream-sync
description: Orchestrate portable, fail-closed synchronization from upstream THUDM/slime into slime-ascend, including upstream change planning, GPU-to-NPU test conversion, NPU asset inventory and validation, bounded autofix, patch-only persistence, clean-tree replay verification, Chinese audit reports, and gated publishing. Use when asked to upgrade, synchronize, continuously follow, validate, trial-run, migrate, deploy, or audit a slime-ascend upstream-sync workflow on one or more Ascend machines.
---

# Slime Upstream Sync

Run the synchronization as a staged protocol. Keep semantic decisions in the
two child skills and keep Git, schemas, counts, manifests, replay, and release
gates deterministic.

## Resolve the installation

Set `SKILL_ROOT` to this skill directory and `REPO_ROOT` to the outer
slime-ascend repository. Require these sibling skills:

- `../gpu-to-npu-test-converter/SKILL.md`
- `../npu-test-autofix/SKILL.md`

Read both child `SKILL.md` files completely before conversion or validation.
Do not copy their rules into this skill or silently continue when either is
missing.

Read [references/contracts.md](references/contracts.md) before a sync run. Read
[references/deployment.md](references/deployment.md) when installing or moving
the workflow to another machine.
Read [references/full-suite-validation.md](references/full-suite-validation.md)
when converting or validating the complete upstream test suite.
Read [references/patch-playbook.md](references/patch-playbook.md) when a patch
fails to apply or an NPU test matches a recorded failure signature. Confirm the
current source and runtime signature before reusing a fix.

Use `scripts/run_sync.py` as the only orchestration runtime. The legacy
`tools/upstream_sync/` paths, if present, are compatibility wrappers and must
not contain a second implementation. Use `assets/upstream-sync.env.example`
and `assets/slime-upstream-sync.cron` as deployment templates.

## Preflight

Before materializing any patch or changing the live dependency environment,
resolve the last validated Slime SHA and the target SHA. Run
`scripts/collect_change_assessment.py` over that exact range and preserve
`.upstream-sync/change-assessment.json`. If the SHAs differ, inspect all commits
and changed paths and complete the generated semantic assessment for:

- supported capability additions, removals, defaults, CLI/configuration and
  behavior-contract changes;
- upstream third-party dependency, source/fork, wheel and build-coordinate
  changes;
- test and GPU CI additions, modifications, deletions, renames and
  parametrization/classification changes.

This assessment happens before dependency migration, inventory conversion or
testing. An empty semantic assessment for a changed SHA range is invalid even
when the path classifier found no obvious user-facing feature.

Run the portable doctor before changing a worktree:

```bash
python3 "$SKILL_ROOT/scripts/doctor.py" \
  --repo-root "$REPO_ROOT" \
  --state-dir "$SLIME_SYNC_STATE_DIR" \
  --work-root "$SLIME_SYNC_WORK_ROOT" \
  --check-sync-layout \
  --check-upstream-dependencies \
  --require-npu \
  --json-output "$SLIME_SYNC_STATE_DIR/doctor-latest.json"
```

Use `--require-assets` for a full run. Separate hard environment blockers from
migration findings. Missing commands, repository layout, NPU allocation,
required assets, disk capacity, credentials, or external services block the
affected execution because no meaningful test can run. A dependency baseline
mismatch, stale patch, or patch apply failure is instead a provisional
`PATCH_DECISION_REQUIRED`: preserve it in the doctor and patch audit, but do
not stop inventory, conversion, independent patch application, or the complete
best-effort test baseline. Never print secret values. Preserve the doctor JSON
as an audit artifact.

An unset asset environment variable or an unsuccessful local search is not yet
proof that an asset is unavailable. After generating the complete asset
inventory, if any required model, tokenizer, Megatron checkpoint, or dataset
path remains unresolved, actively ask the user for the local asset locations
before deciding the ST execution state. For Slime, request
`SLIME_TEST_HF_WEIGHTS_ROOT`, `SLIME_TEST_DATASETS_ROOT`, and either an existing
`SLIME_TEST_MEGATRON_WEIGHTS_ROOT` or confirmation that checkpoints should be
converted into the run's local cache. Include the unresolved model and dataset
list, or link the run-specific inventory that contains the complete list.

Wait for the user's answer, validate every supplied root and required relative
path, and ask a focused follow-up for any remaining gaps. Do not finalize the
run, mark cases `UNEXECUTED_MISSING_ASSET`, or treat assets as a proven blocker
merely because paths were not preconfigured. That classification is allowed
only after the user explicitly confirms the assets are unavailable, supplied
paths remain invalid after follow-up, or the user explicitly requested a
non-interactive run. Inventory, conversion, dependency migration, and UT work
that does not require those assets may continue while waiting, but affected ST
must not start.

The dependency check is a release gate. Treat the target
`slime/docker/Dockerfile` as the authoritative SGLang and Megatron baseline;
do not use the previous slime-ascend release pins as the source of truth. The
doctor must verify that `scripts/quick_install.sh`, every maintained NPU
Dockerfile, and the live editable dependency repositories match that baseline.
For patched repositories, matching means the required upstream commit is an
ancestor of the live HEAD, not that package metadata happens to report the same
version. For an editable SGLang install, also require the imported module and
`direct_url.json` to resolve to that verified worktree, require module and
distribution versions to agree, and require the generated version commit marker
to identify the live HEAD. SGLang's dynamic version may use a newer repository
tag as its display base; record that label, but determine slime compatibility
from the required upstream ancestor plus the verified NPU patch stack.

Apply this rule to the complete upstream slime environment, not only package
version strings. Compare repository URL/fork, commit or tag, wheel URL/build
identity, extras, native artifact coordinates, and upstream runtime assertions.
For example, slime's `sglang-router==0.3.2` custom wheel is not equivalent to a
generic PyPI `0.3.2` wheel when slime requires `"slime"` in
`sglang_router.__version__`. Any difference must be synchronized to upstream
before later tests.

Classify every upstream dependency as exactly one of:

- `UPSTREAM_EXACT`: install the same source/artifact and preserve its runtime
  assertions.
- `NPU_EQUIVALENT`: replace only the hardware-specific implementation, record
  the upstream dependency, NPU replacement, reason, compatibility evidence,
  and validation command.
- `CUDA_ONLY_NOT_APPLICABLE`: prove that no imported or executed NPU path needs
  it; a package being CUDA-related is not sufficient evidence by itself.

An unclassified dependency, generic wheel substituted for a slime fork, or
missing upstream assertion is a dependency mismatch and must trigger an actual
environment update. Do not waive it as an expected NPU difference.

Record a complete component inventory in the final doctor artifact. Git-backed
components must include source URL, required base commit, patched live HEAD and
dirty state for Slime, SGLang, Megatron-LM, MegatronAdaptor (MA), mbridge,
Megatron-Bridge and TransformerEngineNPU (TE). Record SGL Kernel NPU, router,
torch, torch-npu, DeepEP, TorchMemorySaver, MemFabric and Triton Ascend package
versions/build sources. Record CANN Toolkit/runtime and ATB versions, build
identity and any vendor-exposed commit; use a documented null rather than
inventing a Git commit for binary packages that do not expose one.

When the baseline changes, migrate in this order before running product tests:

1. Rebase each affected third-party patch onto the new upstream dependency and
   record the decision and compatibility evidence.
2. Update `scripts/quick_install.sh`, maintained NPU Dockerfiles, wheel/kernel
   coordinates, and user-facing quick-start installation commands together.
3. Reinstall or update the live dependency source trees and rebuild native NPU
   artifacts. Do not merely edit version strings or package metadata.
4. Rerun doctor with `--check-upstream-dependencies`. A remaining mismatch or
   patch conflict keeps the migration/release gate NG, but does not by itself
   block the diagnostic test baseline. It blocks publishing only if it remains
   unresolved after patch classification, replay, and verification.

## Workflow

1. Lock, plan, and synchronize dependencies:
   - Acquire one host-level lock.
   - Resolve the outer repository SHA, previous successful upstream SHA, target
     upstream SHA, patch-manifest hash, legacy-overlay hash, and child-skill hashes.
   - Generate and review the initial change assessment before applying patches.
     Use its feature, dependency and test deltas to define migration and
     conversion scope.
   - Work only in a disposable clone. Do not mutate the deployment checkout.
   - Resolve the target slime dependency contract before conversion or product
     tests. If any version, commit, fork, wheel build, runtime assertion, NPU
     kernel, or related native stack differs from the installer or live
     container, perform the upgrade;
     reporting the mismatch is not completion.
   - Rebase or replace affected third-party patches, update every maintained NPU
     Dockerfile, `scripts/quick_install.sh`, and quick-start installation text,
     then update the live editable repositories and rebuild native artifacts.
   - Dependency patch, build, and install failures are repair work. Iterate them
     using the patch decision protocol; do not stop at `patch does not apply`.
   - Rerun doctor after installation. All `dependency_installer_*` and
     `dependency_runtime_*` checks must pass before step 2. The result contract
     rejects an environment artifact that does not prove this final state.
   - NPU-equivalent packages are part of that wildcard gate. In particular,
     doctor must validate `sgl-kernel-npu` release/build coordinates and
     required official/backported source assertions, plus the exact
     `memfabric-hybrid` pin required by SGLang's NPU Dockerfile for Ascend PD;
     checking SGLang, Megatron, and router alone is incomplete.

2. Materialize:
   - Validate `patches/MANIFEST.toml` with
     `scripts/patch_manifest.py validate`.
   - Require exact agreement between the manifest and patch directories.
   - Apply `target = "slime"` patches in manifest order. If a historical
     `overlay/` exists, copy it only as a read-only compatibility input. Keep
     third-party patches in the same authoritative manifest, but apply them only
     to the named dependency repository through the installation workflow.
   - Persist every durable source or test change in `patches/`, including new
     files represented by `/dev/null` new-file diffs. Register every patch in
     `patches/MANIFEST.toml`. Do not add, modify, or delete legacy `overlay/`
     content during a sync run, and do not keep an unrecorded source change.
   - A patch conflict is repair work, not a terminal result. Record the first
     failed attempt as `PENDING_ANALYSIS`, quarantine only that patch, and keep
     applying independent patches. If a later patch depends on the failed one,
     mark that dependency chain affected instead of pretending it was tested;
     continue all unaffected work.
   - Do not decide from textual applicability alone. Run the complete
     best-effort baseline before concentrating on one patch so runtime results
     can show whether the component upgrade absorbed the old behavior, changed
     the contract, or still needs an NPU adaptation. A tree missing a pending
     patch is diagnostic and cannot be release evidence.
   - When a target component baseline changes, semantically audit every patch
     for that target, including patches that still apply cleanly. `git apply`
     success proves only textual compatibility. Record a still-needed,
     behavior-verified patch as `KEEP_VERIFIED`; remove a redundant patch as
     `UPSTREAM_ABSORBED` only after checking equivalent target source and a
     narrow test without the patch.
   - After the baseline, compare the old hunk intent with the target source and
     runtime evidence, then classify it as `KEEP_VERIFIED`, `REWRITE`, `SPLIT`,
     `UPSTREAM_ABSORBED`, or `DROP_UNSAFE`. Persist evidence and verification in
     `.upstream-sync/patch-decisions.json`, update the manifest and retry from a
     clean target tree. Only a missing external resource may stop this loop.

3. Plan conversion:
   - Compare the previous and target upstream SHAs.
   - Identify changed tests that actually enter upstream GPU CI; do not infer
     this from filenames alone.
   - Write `.upstream-sync/conversion-plan.json` using the contract in
     `references/contracts.md`.

4. Convert:
   - Invoke `gpu-to-npu-test-converter` only for `NEW`, `STALE`, or explicitly
     requested cases.
   - Preserve assertions and behavior contracts.
   - Update each case in `conversion-plan.json`; static success is
     `STATIC_READY`, never `PASS`.

5. Validate and repair:
   - Invoke `npu-test-autofix` with the slime repository as `--repo-root` and
     test roots relative to it, for example `--repo-root slime --test-root tests`.
   - Perform asset inventory before execution. If inventory or path validation
     leaves any required asset unresolved, ask the user for HF weights,
     Megatron checkpoints, and dataset locations and validate the response as
     described in the preflight gate above. Do not silently skip this question.
     Then run a complete baseline, keep issue records, and stop after five
     rounds at most.
   - Do not iterate a single failing case before the complete baseline has
     finished. Record all first-round outcomes first, then repair by shared
     root cause within the bounded rounds.
   - During a full baseline, inspect the active case at least every three
     minutes. If its log proves a child-job failure while Ray remains running,
     preserve the log, record the false-exit risk, stop only that job, and
     continue the queue rather than waiting for the generic timeout.
   - Treat ten minutes without a new log record as a suspected hang, even when
     the generic per-case timeout has not expired. Inspect Ray and NPU state;
     when they confirm no meaningful progress, stop only that case, mark it
     NG, and record the log path, last-progress timestamp, job id, and reason
     in the issue ledger.
   - While a full baseline is active, send the user a status update at least
     every three minutes: completed/total count, active case, and the
     log/Ray/NPU inspection conclusion. Report a confirmed failure or hang
     immediately rather than waiting for the next interval.
   - Write `.upstream-sync/validation-result.json`. Skipped, missing-asset,
     timeout, and unexecuted cases are not PASS.
   - Continue through materialization, conversion, compilation, dependency,
     launcher, product, and NPU adaptation failures. Diagnose and repair them
     within the bounded issue-ledger rounds; do not stop merely because the
     first attempt or clean replay fails.
   - A failed or quarantined patch may make a subset of tests unrunnable. Mark
     those cases NG with the responsible patch/dependency chain, then continue
     the rest of the dynamically discovered queue. Never convert an affected
     unexecuted case into PASS and never abandon unaffected cases.
   - For every OOM, require the `npu-test-autofix` OOM playbook phase
     classification before tuning. Match the earliest failing phase and
     operator, then use a focused rerun to test the smallest hypothesis before
     returning the case to the full queue.

6. Persist:
   - Regenerate the smallest affected patches for all durable changes, including
     NPU tests and every other new file, and update manifest hashes and metadata.
   - Keep the historical `overlay/` byte-for-byte unchanged. It is a migration
     compatibility layer, not a destination for new work.
   - Leave logs, caches, checkpoints, `.npu-test-autofix*` state, and other
     runtime artifacts out of patch content.

7. Replay:
   - Keep the validated tree intact.
   - Run `scripts/replay_verify.py` to create a second clean tree at the exact
     target SHA, reapply final patches and the read-only legacy overlay, and
     compare complete content manifests.
   - Require `replay_verified=true`. A tracked, untracked, symlink, mode, or
     content mismatch blocks release.

8. Report and gate:
   - Include dedicated upstream-change and test-change sections sourced from
     `change-assessment.json`, plus a component matrix sourced from the final
     doctor inventory. Show base/target Slime SHAs, commit range, feature impact,
     dependency declaration changes, test/CI changes, and every component base
     commit/patched HEAD or binary build identity listed in Preflight.
   - Add a `dev 分支集成` section for the future slime-ascend dev handoff. Record
     the outer base branch/SHA, target Slime SHA, patch manifest hash and apply
     order, dependency base/patched commits, installer changes, clean replay
     status, unresolved issues and the exact verification commands. Do not call
     the handoff publishable unless the normal release gates pass.
   - Generate a Chinese audit report and schema-valid `result.json`.
   - Validate the result with `scripts/validate_contracts.py result`.
   - In full mode, publish only when every discovered case executed and passed,
     every patch decision is resolved and verified, dependency migration and
     replay succeeded, and the worktree scope gate passed.
   - In trial mode, never publish or update the success marker.
   - Make branch and success-tag push atomic. Update the local success cache
     only after remote success is confirmed.

9. Preserve repair knowledge:
   - Keep raw case evidence in `.npu-test-autofix/issues/ISSUE-*.md`.
   - After a successful repair, add only its reusable signature, exclusions,
     minimal fix, contract boundary, and verification level to the owning child
     skill. Migration differences belong in `gpu-to-npu-test-converter`;
     runtime and OOM recipes belong in `npu-test-autofix`.
   - Record case status and durable patch location in the sync reference
     while preserving historical run reports unchanged.
   - Mark user-reported manual success `USER_VERIFIED`; upgrade to
     `LOG_VERIFIED` only when a preserved log proves complete behavior and a
     clean exit. Never generalize from a failure signature alone.

## Safety

- Do not push from either child skill; publishing belongs to this skill.
- Do not download assets, guess paths, weaken assertions, or call unexecuted
  work PASS.
- Do not infer that assets are unavailable from unset environment variables or
  failed local discovery alone. Ask the user for their locations and wait for a
  response before final missing-asset classification, except in an explicitly
  requested non-interactive run.
- Do not terminate shared processes without user authorization.
- Preserve failed workspaces and audit artifacts by default.
- Stop early only for a proven unavailable model, dataset, NPU allocation, disk
  capacity, required external service, credential, or other resource that
  cannot be created or repaired in scope. Record the exact resource and all
  affected cases. Code, patch, workflow, dependency, and test failures are not
  resource blockers and must be iterated.
- Use a dedicated, replaceable NPU runner for autonomous full-access agents.
- All user-facing reports and summaries must be Chinese. Keep commands, paths,
  identifiers, logs, enum values, and machine-readable keys unchanged when
  translation would break parsing.
