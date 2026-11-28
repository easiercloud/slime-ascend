#!/usr/bin/env bash
# Run with the original image Python, never activate this venv globally.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/install-fsdp-environment.sh /absolute/FSDPTurbo /absolute/new-venv" >&2
  exit 2
fi
provider=$(realpath "$1")
venv=$(realpath -m "$2")
revision=262f5ba4ae266d62b01c7e8a2e2bc46d46d1a0c3
git -C "$provider" merge-base --is-ancestor "$revision" HEAD || {
  echo "Use FSDPTurbo $revision with the Slime A2 accelerator backport" >&2
  exit 1
}
[[ $(git -C "$provider" diff --name-only "$revision" HEAD) == "fsdp_turbo/utils/device.py" ]] || {
  echo "Apply the accelerator backport in patches/third_party/fsdp-turbo before installation" >&2
  exit 1
}
[[ -z $(git -C "$provider" status --porcelain) ]] || { echo "FSDPTurbo checkout must be clean" >&2; exit 1; }
[[ ! -e "$venv" ]] || { echo "Refusing to overwrite an existing environment: $venv" >&2; exit 1; }
"${BASE_PYTHON:-python}" -m venv --system-site-packages "$venv"
PIP_CONSTRAINT= "$venv/bin/python" -m pip install -r "$repo_root/requirements/fsdp2.txt"
PIP_CONSTRAINT= "$venv/bin/python" -m pip install --no-deps --no-build-isolation -e "$provider"
"$venv/bin/python" "$repo_root/scripts/fsdp2/check_pinned_runtime.py"
echo "Training Python: $venv/bin/python (keep the driver/SGLang on the original Python)"
