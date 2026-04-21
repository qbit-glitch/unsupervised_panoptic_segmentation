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

# Remove any partial/corrupted file from previous interrupted run
if [ -f "$FILENAME" ]; then
    echo "Existing tarball found. Checking if it is a valid archive..."
    if ! tar tzf "$FILENAME" >/dev/null 2>&1; then
        echo "  → Corrupted or incomplete. Re-downloading..."
        rm -f "$FILENAME"
    else
        echo "  → Valid archive. Skipping download."
    fi
fi

# Download if not already present
if [ ! -f "$FILENAME" ]; then
    # Primary: wget (no auth needed for public datasets)
    if command -v wget &> /dev/null; then
        echo "Using wget..."
        wget -O "$FILENAME" "$URL"
    # Fallback: curl with -f to fail on HTTP errors
    elif command -v curl &> /dev/null; then
        echo "Using curl..."
        curl -f -L -o "$FILENAME" "$URL"
    else
        echo "ERROR: No download tool found (wget or curl). Please install one." >&2
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
ls -la "$TARGET_DIR/"
