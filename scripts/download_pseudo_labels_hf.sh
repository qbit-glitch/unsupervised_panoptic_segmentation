#!/bin/bash
# Download CUPS pseudo-labels from HuggingFace Hub for AnyDesk machine
# Usage: bash download_pseudo_labels_hf.sh [TARGET_DIR]

set -euo pipefail

# Default to standard cityscapes location on AnyDesk
TARGET_DIR="${1:-/home/cvpr_ug_5/umesh/datasets/cityscapes}"
REPO_ID="qbit-glitch/cityscapes-cups-pseudo-labels-v1"
FILENAME="cups_pseudo_labels_dcfa_simcf_abc.tar.gz"
URL="https://huggingface.co/datasets/$REPO_ID/resolve/main/$FILENAME"

echo "=== Downloading pseudo-labels from HuggingFace Hub ==="
echo "URL: $URL"
echo "Target: $TARGET_DIR"
echo ""

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Download if not already present
if [ -f "$FILENAME" ]; then
    echo "Tarball already exists. Skipping download."
else
    # Primary: wget (no auth needed for public datasets)
    if command -v wget &> /dev/null; then
        echo "Using wget..."
        wget "$URL"
    # Fallback: curl
    elif command -v curl &> /dev/null; then
        echo "Using curl..."
        curl -L -o "$FILENAME" "$URL"
    else
        echo "ERROR: No download tool found (wget or curl). Please install one."
        exit 1
    fi
fi

echo ""
echo "=== Extracting tarball ==="
tar xzf "$FILENAME"
rm "$FILENAME"

echo ""
echo "=== Done ==="
echo "Contents:"
ls -la "$TARGET_DIR/" | head -10
