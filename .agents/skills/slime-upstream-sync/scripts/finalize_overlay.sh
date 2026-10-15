#!/usr/bin/env bash
set -euo pipefail

: "${SLIME_SYNC_WORKSPACE:?}"
: "${SLIME_SYNC_SUBMODULE:?}"
: "${SLIME_SYNC_UPSTREAM_SHA:?}"

overlay_dir="${SLIME_SYNC_WORKSPACE}/overlay"
patch_dir="${SLIME_SYNC_WORKSPACE}/patches"
failure_file="${SLIME_SYNC_WORKSPACE}/.upstream-sync/materialize-failures.txt"
overlay_before="${SLIME_SYNC_WORKSPACE}/.upstream-sync/replay/legacy-overlay-before.json"
overlay_after="${SLIME_SYNC_WORKSPACE}/.upstream-sync/replay/legacy-overlay-after.json"
replay_dir="${SLIME_SYNC_WORKSPACE}/.upstream-sync/replay"
skill_scripts="${SLIME_SYNC_WORKSPACE}/.agents/skills/slime-upstream-sync/scripts"
result_path="${SLIME_SYNC_WORKSPACE}/.upstream-sync/result.json"

if [[ -s "${failure_file}" ]]; then
  echo "仍存在未解决的补丁物化失败，拒绝收尾: ${failure_file}" >&2
  exit 1
fi

# All durable changes, including new NPU tests, must be patches. The historical
# overlay is read-only and any mutation is a persistence contract violation.
if [[ ! -f "${overlay_before}" ]]; then
  echo "缺少 legacy overlay 基线清单，拒绝收尾: ${overlay_before}" >&2
  exit 1
fi
python3 "${skill_scripts}/tree_manifest.py" create \
  --root "${overlay_dir}" \
  --output "${overlay_after}" \
  --no-default-excludes >/dev/null
python3 "${skill_scripts}/tree_manifest.py" compare \
  --validated "${overlay_before}" \
  --replay "${overlay_after}"
rm -f "${overlay_before}" "${overlay_after}"

python3 "${skill_scripts}/replay_verify.py" \
  --validated-repo "${SLIME_SYNC_SUBMODULE}" \
  --target-sha "${SLIME_SYNC_UPSTREAM_SHA}" \
  --patches-root "${patch_dir}" \
  --manifest "${patch_dir}/MANIFEST.toml" \
  --overlay "${overlay_dir}" \
  --work-root "${SLIME_SYNC_WORKSPACE}/.upstream-sync/replay-work" \
  --output-dir "${replay_dir}"

python3 "${skill_scripts}/mark_replay.py" \
  --result "${result_path}" \
  --replay-result "${replay_dir}/replay-result.json"

# The validated tree remains intact until replay succeeds. Only then clean the
# disposable submodule so the outer repository records the exact target SHA.
git -C "${SLIME_SYNC_SUBMODULE}" reset --hard HEAD
git -C "${SLIME_SYNC_SUBMODULE}" clean -fd
rm -rf "${SLIME_SYNC_WORKSPACE}/.upstream-sync/replay-work"
rm -f \
  "${SLIME_SYNC_WORKSPACE}/.upstream-sync/materialize-failures.txt" \
  "${SLIME_SYNC_WORKSPACE}/.upstream-sync/materialized-files.txt"

git -C "${SLIME_SYNC_WORKSPACE}" add \
  "${SLIME_SYNC_WORKSPACE}/overlay" \
  "${SLIME_SYNC_WORKSPACE}/patches" \
  "${SLIME_SYNC_WORKSPACE}/.upstream-sync/audit-report.zh-CN.md" \
  "${SLIME_SYNC_WORKSPACE}/.upstream-sync/result.json" \
  "${SLIME_SYNC_WORKSPACE}/.upstream-sync/replay"
