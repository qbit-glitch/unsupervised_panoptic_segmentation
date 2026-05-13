#!/usr/bin/env bash
# Upload DepthPro tau=0.20 pseudo-labels to HF Hub
# Run this LOCALLY (Mac) before downloading on A6000

set -euo pipefail

REPO_ID="qbit-glitch/cups-pseudo-labels-depthpro-tau020"
LOCAL_DIR="/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020"

echo "=== Uploading DepthPro tau=0.20 labels to HF Hub ==="
echo "Source: $LOCAL_DIR"
echo "Target: $REPO_ID"
echo "Files: $(ls "$LOCAL_DIR" | wc -l)"

python3 -c "
from huggingface_hub import HfApi, create_repo
import os

repo_id = '$REPO_ID'
local_dir = '$LOCAL_DIR'

api = HfApi()

# Create repo if needed
try:
    api.repo_info(repo_id, repo_type='dataset')
    print(f'Repo {repo_id} already exists')
except Exception:
    print(f'Creating dataset repo: {repo_id}')
    create_repo(repo_id, repo_type='dataset', private=False)

# Upload entire folder
print('Uploading... (this may take a few minutes)')
api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type='dataset',
    commit_message='Add DepthPro tau=0.20 pseudo-labels (k=80, 2975 images)',
)
print('Upload complete!')
"
