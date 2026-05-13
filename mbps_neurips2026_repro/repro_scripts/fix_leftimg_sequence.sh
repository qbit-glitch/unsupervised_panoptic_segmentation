#!/bin/bash
# Fix: CUPS expects leftImg8bit_sequence/ but we only have leftImg8bit/
# Creates a symlink so CUPS can find the images.
CITYSCAPES_ROOT="$HOME/umesh/datasets/cityscapes"

if [ -d "$CITYSCAPES_ROOT/leftImg8bit_sequence" ]; then
    echo "leftImg8bit_sequence already exists — nothing to do."
    exit 0
fi

if [ ! -d "$CITYSCAPES_ROOT/leftImg8bit" ]; then
    echo "ERROR: $CITYSCAPES_ROOT/leftImg8bit not found!"
    exit 1
fi

ln -s "$CITYSCAPES_ROOT/leftImg8bit" "$CITYSCAPES_ROOT/leftImg8bit_sequence"
echo "Created symlink: leftImg8bit_sequence -> leftImg8bit"
ls -la "$CITYSCAPES_ROOT/leftImg8bit_sequence"
