#!/bin/bash
# Extract temporal neighbor frames from leftImg8bit_sequence zip and run CUPS pipeline.
# Run on remote GPU machine after download completes.
#
# Usage: bash scripts/extract_and_run_cups_pipeline.sh

set -euo pipefail

export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}
PYTHON=/home/santosh/anaconda3/envs/cups/bin/python
DATA_ROOT=/home/santosh/datasets/cityscapes
ZIP_FILE=/home/santosh/datasets/leftImg8bit_sequence_trainvaltest.zip
SEQ_DIR=${DATA_ROOT}/leftImg8bit_sequence/train
OUTPUT_DIR=${DATA_ROOT}/cups_pseudo_labels_pipeline
PROJECT=/home/santosh/mbps_panoptic_segmentation

echo "=== Step 1: Extract temporal neighbor frames from zip ==="
echo "This extracts frames 000018 and 000020 for each clip (needed for optical flow)"

# First, remove old key-frame-only sequence data
rm -rf ${SEQ_DIR}
mkdir -p ${SEQ_DIR}

# Extract frame 000018 and 000020 (temporal neighbors of key frame 000019)
# Also extract key frame 000019 (it's already there but let's be complete)
echo "Extracting frames 000018, 000019, 000020 from all train cities..."
unzip -o "${ZIP_FILE}" \
    "leftImg8bit_sequence/train/*/*_000018_leftImg8bit.png" \
    "leftImg8bit_sequence/train/*/*_000019_leftImg8bit.png" \
    "leftImg8bit_sequence/train/*/*_000020_leftImg8bit.png" \
    -d "${DATA_ROOT}/" 2>&1 | tail -5

echo ""
echo "Extracted files per city:"
for city in ${SEQ_DIR}/*/; do
    city_name=$(basename "$city")
    count=$(ls "$city" | wc -l)
    echo "  ${city_name}: ${count} files"
done
TOTAL=$(find ${SEQ_DIR} -name "*.png" | wc -l)
echo "Total extracted: ${TOTAL} files (expected ~8925 = 2975 * 3)"

echo ""
echo "=== Step 2: Delete zip to free space ==="
echo "Zip size: $(du -sh ${ZIP_FILE} | cut -f1)"
rm -f "${ZIP_FILE}"
echo "Zip deleted. Free space: $(df -h /home/santosh/ | tail -1 | awk '{print $4}')"

echo ""
echo "=== Step 3: Run CUPS pseudo-label generation pipeline ==="
echo "Using: DepthG + RAFT-SMURF + SF2SE3 (full CUPS pipeline)"
echo "Output: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

cd ${PROJECT}/refs/cups

$PYTHON -u ${PROJECT}/mbps_pytorch/gen_cups_pseudo_labels_remote.py \
    --data_root "${DATA_ROOT}" \
    --depthg_ckpt "${DATA_ROOT}/depthg.ckpt" \
    --output_dir "${OUTPUT_DIR}" \
    --gpu 0 \
    --num_workers 4

echo ""
echo "=== Step 4: Verify output ==="
PNG_COUNT=$(ls ${OUTPUT_DIR}/*.png 2>/dev/null | wc -l)
PT_COUNT=$(ls ${OUTPUT_DIR}/*.pt 2>/dev/null | wc -l)
echo "Generated: ${PNG_COUNT} PNGs (expected 5950), ${PT_COUNT} .pt files (expected 1)"
echo "Output size: $(du -sh ${OUTPUT_DIR} | cut -f1)"
echo "DONE"
