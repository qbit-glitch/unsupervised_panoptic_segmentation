#!/bin/bash
# Download CUPS pseudo-labels from HuggingFace Hub for AnyDesk machine
# Usage: bash download_pseudo_labels_hf.sh [TARGET_DIR]

set -euo pipefail

TARGET_DIR="${1:-./cityscapes_pseudo_labels}"
REPO_ID="qbit-glitch/cityscapes-cups-pseudo-labels-v1"
FILENAME="cups_pseudo_labels_dcfa_simcf_abc.tar.gz"
URL="https://huggingface.co/datasets/$REPO_ID/resolve/main/$FILENAME"

echo "=== Downloading pseudo-labels from HuggingFace Hub ==="
echo "URL: $URL"
echo "Target: $TARGET_DIR"
echo ""

mkdir -p "$TARGET_DIR"

# Primary: wget (no auth needed for public datasets)
if command -v wget &> /dev/null; then
    echo "Using wget..."
    wget -O "$TARGET_DIR/$FILENAME" "$URL"
# Fallback: curl
elif command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -L -o "$TARGET_DIR/$FILENAME" "$URL"
# Fallback: hf (new HuggingFace CLI)
elif command -v hf &> /dev/null; then
    echo "Using hf..."
    hf download "$REPO_ID" "$FILENAME" --local-dir "$TARGET_DIR" --repo-type dataset
else
    echo "ERROR: No download tool found (wget, curl, or hf). Please install one."
    exit 1
fi

echo ""
echo "=== Extracting tarball ==="
cd "$TARGET_DIR"
tar xzf "$FILENAME"
rm "$FILENAME"

echo ""
echo "=== Done ==="
echo "Pseudo-labels available at: $TARGET_DIR/cups_pseudo_labels_dcfa_simcf_abc/"
ls -la "$TARGET_DIR/cups_pseudo_labels_dcfa_simcf_abc/" | head -5
