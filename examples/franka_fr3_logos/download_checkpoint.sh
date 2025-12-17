#!/bin/bash

# Usage: ./download_checkpoint.sh <experiment_name> <checkpoint_name>
# Example: ./download_checkpoint.sh xvla_multi_task_1 last

if [ "$#" -ne 2 ]; then
    echo "Error: Expected 2 arguments"
    echo "Usage: $0 <experiment_name> <checkpoint_name>"
    echo "Example: $0 xvla_multi_task_1 last"
    exit 1
fi

EXPERIMENT_NAME=$1
CHECKPOINT_NAME=$2

# Remote server details
REMOTE_USER="prath4"
REMOTE_HOST="129.219.30.59"
REMOTE_BASE_PATH="/mount/scratch3/prabin/outputs/train"

# Local paths (relative to repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCAL_BASE_PATH="$REPO_ROOT/outputs/train"

# Full paths
REMOTE_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_PATH}/${EXPERIMENT_NAME}/checkpoints/${CHECKPOINT_NAME}"
LOCAL_PATH="${LOCAL_BASE_PATH}/${EXPERIMENT_NAME}/checkpoints"

# Create local directory structure
echo "Creating directory: ${LOCAL_PATH}"
mkdir -p "${LOCAL_PATH}"

# Download using scp
echo "Downloading from: ${REMOTE_PATH}"
echo "Downloading to: ${LOCAL_PATH}"
scp -r "${REMOTE_PATH}" "${LOCAL_PATH}"

if [ $? -eq 0 ]; then
    echo "Download completed successfully!"
    echo "Files saved to: ${LOCAL_PATH}/${CHECKPOINT_NAME}"
else
    echo "Download failed!"
    exit 1
fi
