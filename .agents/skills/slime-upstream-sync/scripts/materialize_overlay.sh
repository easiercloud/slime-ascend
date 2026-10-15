#!/usr/bin/env bash
set -euo pipefail

: "${SLIME_SYNC_WORKSPACE:?}"
: "${SLIME_SYNC_SUBMODULE:?}"

patch_dir="${SLIME_SYNC_WORKSPACE}/patches"
overlay_dir="${SLIME_SYNC_WORKSPACE}/overlay"
patch_manifest="${patch_dir}/MANIFEST.toml"
materialized_manifest="${SLIME_SYNC_WORKSPACE}/.upstream-sync/materialized-files.txt"
failure_file="${SLIME_SYNC_WORKSPACE}/.upstream-sync/materialize-failures.txt"
manifest_tool="${SLIME_SYNC_WORKSPACE}/.agents/skills/slime-upstream-sync/scripts/patch_manifest.py"
tree_manifest_tool="${SLIME_SYNC_WORKSPACE}/.agents/skills/slime-upstream-sync/scripts/tree_manifest.py"
overlay_before="${SLIME_SYNC_WORKSPACE}/.upstream-sync/replay/legacy-overlay-before.json"

for required in "${patch_dir}" "${overlay_dir}" "${patch_manifest}" "${manifest_tool}" "${tree_manifest_tool}"; do
  if [[ ! -e "${required}" ]]; then
    echo "缺少同步输入: ${required}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${materialized_manifest}")"
: >"${materialized_manifest}"
: >"${failure_file}"

# The existing overlay is a read-only compatibility layer. Snapshot every
# entry, including historical runtime artifacts, so finalize rejects mutation.
python3 "${tree_manifest_tool}" create \
  --root "${overlay_dir}" \
  --output "${overlay_before}" \
  --no-default-excludes >/dev/null

patch_list="$(mktemp)"
trap 'rm -f "${patch_list}"' EXIT
python3 "${manifest_tool}" validate \
  --manifest "${patch_manifest}" \
  --patches-root "${patch_dir}" >"${patch_list}"

while IFS=$'\t' read -r category identifier patch target; do
  [[ -n "${patch}" ]] || continue
  # External dependency patches are authoritative inventory entries, but are
  # applied to their own repositories by scripts/quick_install.sh.
  [[ "${target}" == "slime" ]] || continue
  if git -C "${SLIME_SYNC_SUBMODULE}" apply --check "${patch}"; then
    git -C "${SLIME_SYNC_SUBMODULE}" apply "${patch}"
  else
    printf '%s\t%s\t%s\n' "${category}" "${identifier}" "${patch#${SLIME_SYNC_WORKSPACE}/}" >>"${failure_file}"
  fi
done <"${patch_list}"

(
  cd "${overlay_dir}"
  find . \( -type f -o -type l \) -print | sort
) >"${materialized_manifest}"

cp -a "${overlay_dir}/." "${SLIME_SYNC_SUBMODULE}/"

if [[ -s "${failure_file}" ]]; then
  echo "部分补丁未能应用；已记录并交由代理诊断: ${failure_file}" >&2
fi
