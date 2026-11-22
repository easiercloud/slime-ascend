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
revision=0383285e5a7e75d269624354fcc958bd6dcda739
[[ $(git -C "$provider" rev-parse HEAD) == "$revision" ]] || { echo "FSDPTurbo must be at $revision" >&2; exit 1; }
[[ -z $(git -C "$provider" status --porcelain) ]] || { echo "FSDPTurbo checkout must be clean" >&2; exit 1; }
[[ ! -e "$venv" ]] || { echo "Refusing to overwrite an existing environment: $venv" >&2; exit 1; }
"${BASE_PYTHON:-python}" -m venv --system-site-packages "$venv"
PIP_CONSTRAINT= "$venv/bin/python" -m pip install -r "$repo_root/requirements/fsdp2.txt"
PIP_CONSTRAINT= "$venv/bin/python" -m pip install --no-deps --no-build-isolation -e "$provider"
echo "Training Python: $venv/bin/python (keep the driver/SGLang on the original Python)"
