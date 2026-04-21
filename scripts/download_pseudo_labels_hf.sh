#!/bin/bash
# Download CUPS pseudo-labels from HuggingFace Hub for AnyDesk machine
# Usage: bash download_pseudo_labels_hf.sh [TARGET_DIR]

set -euo pipefail

TARGET_DIR="${1:-./cityscapes_pseudo_labels}"
REPO_ID="qbit-glitch/cityscapes-cups-pseudo-labels-v1"
FILENAME="cups_pseudo_labels_dcfa_simcf_abc.tar.gz"

echo "=== Downloading pseudo-labels from HuggingFace Hub ==="
echo "Repo: $REPO_ID"
echo "Target: $TARGET_DIR"
echo ""

mkdir -p "$TARGET_DIR"

# Download using huggingface-cli (assumes HF token is configured)
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download "$REPO_ID" \
        --local-dir "$TARGET_DIR" \
        --repo-type dataset \
        --include "$FILENAME"
else
    echo "huggingface-cli not found. Trying wget fallback..."
    # Public dataset URL fallback
    wget -O "$TARGET_DIR/$FILENAME" \
        "https://huggingface.co/datasets/$REPO_ID/resolve/main/$FILENAME"
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
