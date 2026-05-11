#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_EXE="${CONDA_EXE:-/vision/u/$(whoami)/miniconda3/bin/conda}"

cd "$OPENPI_ROOT"
if [ ! -x "$CONDA_EXE" ]; then
    echo "conda executable not found: $CONDA_EXE" >&2
    exit 1
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate behavior
source "$OPENPI_ROOT/.venv/bin/activate"

CONFIG_NAME=${1:-pi05_s2rg_droid_real}
if [ $# -gt 0 ]; then
    shift
fi

EXP_NAME=${EXP_NAME:-${CONFIG_NAME}_sc_lowprio}
HF_DOWNLOAD_RETRY_SECONDS=${HF_DOWNLOAD_RETRY_SECONDS:-300}
HF_DOWNLOAD_MAX_WORKERS=${HF_DOWNLOAD_MAX_WORKERS:-4}
OPENPI_HF_OFFLINE_AFTER_DOWNLOAD=${OPENPI_HF_OFFLINE_AFTER_DOWNLOAD:-1}
OPENPI_HF_TOKEN_JSON=${OPENPI_HF_TOKEN_JSON:-$HOME/Documents/credentials/hf_token.json}

if [ -z "${OPENPI_HF_TOKEN:-}" ]; then
    if [ ! -f "$OPENPI_HF_TOKEN_JSON" ]; then
        echo "OPENPI_HF_TOKEN_JSON does not exist: $OPENPI_HF_TOKEN_JSON" >&2
        exit 1
    fi
    OPENPI_HF_TOKEN="$(python - "$OPENPI_HF_TOKEN_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
token = data.get("read")
if not token:
    raise SystemExit("Missing 'read' token in HF token JSON")
print(token, end="")
PY
)"
fi

echo "Current time: $(date)"
echo "OPENPI_ROOT=$OPENPI_ROOT"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "HF_DOWNLOAD_RETRY_SECONDS=$HF_DOWNLOAD_RETRY_SECONDS"
echo "HF_DOWNLOAD_MAX_WORKERS=$HF_DOWNLOAD_MAX_WORKERS"
echo "OPENPI_HF_OFFLINE_AFTER_DOWNLOAD=$OPENPI_HF_OFFLINE_AFTER_DOWNLOAD"
if [ -n "${OPENPI_HF_TOKEN:-}" ]; then
    echo "OPENPI_HF_TOKEN is set; authenticated HF downloads will be used."
else
    echo "OPENPI_HF_TOKEN is not set; downloads will be unauthenticated."
fi
echo "Training args: $*"

DATASETS_FILE="$(mktemp)"
DOWNLOAD_LOG="$(mktemp)"
cleanup() {
    rm -f "$DATASETS_FILE" "$DOWNLOAD_LOG"
}
trap cleanup EXIT

python - "$CONFIG_NAME" > "$DATASETS_FILE" <<'PY'
import pathlib
import sys

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets import lerobot_dataset
from openpi.training import config as _config

config = _config.get_config(sys.argv[1])
data_config = config.data.create(config.assets_dirs, config.model)

if data_config.repo_id is None:
    raise SystemExit("Config does not use a LeRobot repo_id; nothing to pre-download.")

revision = lerobot_dataset.CODEBASE_VERSION
dataset_root = pathlib.Path(data_config.dataset_root) if data_config.dataset_root is not None else None

repo_ids = data_config.repo_id if isinstance(data_config.repo_id, list) else [data_config.repo_id]
is_multi = isinstance(data_config.repo_id, list)

for repo_id in repo_ids:
    if is_multi:
        local_dir = (dataset_root if dataset_root is not None else HF_LEROBOT_HOME) / repo_id
    else:
        local_dir = dataset_root if dataset_root is not None else HF_LEROBOT_HOME / repo_id
    print(f"{repo_id}\t{revision}\t{local_dir}")
PY

download_dataset() {
    local repo_id=$1
    local revision=$2
    local local_dir=$3
    local attempt=1
    local token_args=()

    if [ -n "${OPENPI_HF_TOKEN:-}" ]; then
        token_args=(--token "$OPENPI_HF_TOKEN")
    fi

    mkdir -p "$local_dir"
    while true; do
        echo "Downloading/checking dataset repo=$repo_id revision=$revision local_dir=$local_dir attempt=$attempt"
        set +e
        hf download "$repo_id" \
            --repo-type dataset \
            --revision "$revision" \
            --local-dir "$local_dir" \
            --max-workers "$HF_DOWNLOAD_MAX_WORKERS" \
            "${token_args[@]}" \
            2>&1 | tee "$DOWNLOAD_LOG"
        status=${PIPESTATUS[0]}
        set -e

        if [ "$status" -eq 0 ]; then
            echo "Dataset ready: $repo_id -> $local_dir"
            return 0
        fi

        if grep -Eiq '(^|[^0-9])429([^0-9]|$)|Too Many Requests|rate.?limit|RateLimit' "$DOWNLOAD_LOG"; then
            echo "HF rate limit while downloading $repo_id. Sleeping ${HF_DOWNLOAD_RETRY_SECONDS}s before retry."
            sleep "$HF_DOWNLOAD_RETRY_SECONDS"
            attempt=$((attempt + 1))
            continue
        fi

        echo "Dataset download failed for $repo_id with a non-rate-limit error."
        return "$status"
    done
}

while IFS=$'\t' read -r repo_id revision local_dir; do
    [ -n "$repo_id" ] || continue
    download_dataset "$repo_id" "$revision" "$local_dir"
done < "$DATASETS_FILE"

if [ "$OPENPI_HF_OFFLINE_AFTER_DOWNLOAD" = "1" ]; then
    export HF_HUB_OFFLINE=1
    echo "HF_HUB_OFFLINE=1 for training; datasets must be fully present locally."
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9} \
uv run scripts/b1k/train_b1k.py "$CONFIG_NAME" \
    --exp_name="$EXP_NAME" \
    --resume \
    --batch_size=64 \
    "$@"

echo "Job finished."
