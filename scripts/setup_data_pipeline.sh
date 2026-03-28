#!/bin/bash
# Full data pipeline: download datasets, precompute depth, create TFRecords,
# convert DINO weights, and upload everything to GCS.
#
# Run this on a single TPU VM (e.g. mbps-v4-0) BEFORE launching training.
# The VM needs internet access for downloads and GCS access for uploads.
#
# Usage:
#   bash scripts/setup_data_pipeline.sh              # Full pipeline
#   bash scripts/setup_data_pipeline.sh cityscapes    # Cityscapes only
#   bash scripts/setup_data_pipeline.sh coco          # COCO only
#   bash scripts/setup_data_pipeline.sh weights       # DINO weights only
#   bash scripts/setup_data_pipeline.sh verify        # Verify GCS contents

set -euo pipefail

PROJECT_DIR="${HOME}/mbps_panoptic_segmentation"
DATA_DIR="${PROJECT_DIR}/data"
BUCKET="gs://mbps-panoptic"
DATASET="${1:-all}"

cd "${PROJECT_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# 1. DINO ViT-S/8 weights
# ---------------------------------------------------------------------------
setup_dino_weights() {
    log "=== DINO ViT-S/8 Weight Conversion ==="

    local DINO_DIR="${DATA_DIR}/dino"
    mkdir -p "${DINO_DIR}"

    # Check if already on GCS
    if gsutil -q stat "${BUCKET}/weights/dino_vits8_flax.npz" 2>/dev/null; then
        log "DINO Flax weights already on GCS, skipping."
        return 0
    fi

    # Download PyTorch checkpoint
    if [ ! -f "${DINO_DIR}/dino_deitsmall8_pretrain.pth" ]; then
        log "Downloading DINO ViT-S/8 PyTorch checkpoint..."
        wget -q --show-progress -O "${DINO_DIR}/dino_deitsmall8_pretrain.pth" \
            "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
    fi

    # Convert to Flax
    log "Converting PyTorch → Flax..."
    python3 scripts/convert_dino_weights.py \
        --input "${DINO_DIR}/dino_deitsmall8_pretrain.pth" \
        --output "${DINO_DIR}/dino_vits8_flax.npz"

    # Upload to GCS
    log "Uploading DINO weights to GCS..."
    gsutil cp "${DINO_DIR}/dino_vits8_flax.npz" "${BUCKET}/weights/dino_vits8_flax.npz"

    log "DINO weights ready."
}

# ---------------------------------------------------------------------------
# 2. Cityscapes dataset
# ---------------------------------------------------------------------------
setup_cityscapes() {
    log "=== Cityscapes Dataset ==="

    local CS_DIR="${DATA_DIR}/cityscapes"
    mkdir -p "${CS_DIR}"

    # Check if TFRecords already on GCS
    local train_count
    train_count=$(gsutil ls "${BUCKET}/datasets/cityscapes/tfrecords/train/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    if [ "${train_count}" -gt 0 ]; then
        log "Cityscapes TFRecords already on GCS (${train_count} shards), skipping."
        return 0
    fi

    # Download Cityscapes (requires CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD env vars)
    if [ ! -d "${CS_DIR}/leftImg8bit" ]; then
        if [ -z "${CITYSCAPES_USERNAME:-}" ] || [ -z "${CITYSCAPES_PASSWORD:-}" ]; then
            log "ERROR: Set CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD env vars."
            log "  Register at: https://www.cityscapes-dataset.com/register/"
            log "  Then: export CITYSCAPES_USERNAME=you@email.com"
            log "        export CITYSCAPES_PASSWORD=yourpassword"
            return 1
        fi

        log "Downloading Cityscapes images..."
        wget --keep-session-cookies --header="Cookie: " \
            --post-data "username=${CITYSCAPES_USERNAME}&password=${CITYSCAPES_PASSWORD}&submit=Login" \
            -O /dev/null "https://www.cityscapes-dataset.com/login/" 2>/dev/null

        for pkg in leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip; do
            log "  Downloading ${pkg}..."
            wget -q --show-progress -O "${CS_DIR}/${pkg}" \
                "https://www.cityscapes-dataset.com/file-handling/?packageID=$(echo ${pkg} | sed 's/_trainvaltest.zip//')" || {
                log "  Cityscapes auto-download failed. Please download manually:"
                log "    1. Go to https://www.cityscapes-dataset.com/downloads/"
                log "    2. Download leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip"
                log "    3. Place them in ${CS_DIR}/"
                log "    4. Re-run this script."
                return 1
            }
        done

        log "Extracting Cityscapes..."
        cd "${CS_DIR}"
        unzip -q leftImg8bit_trainvaltest.zip
        unzip -q gtFine_trainvaltest.zip
        cd "${PROJECT_DIR}"
    fi
    log "Cityscapes images ready at ${CS_DIR}"

    # Precompute depth maps
    if [ ! -d "${CS_DIR}/depth_zoedepth" ] || [ "$(find ${CS_DIR}/depth_zoedepth -name '*.npy' 2>/dev/null | wc -l)" -lt 100 ]; then
        log "Precomputing depth maps (ZoeDepth)..."
        python3 scripts/precompute_depth.py \
            --data_dir "${CS_DIR}/leftImg8bit" \
            --output_dir "${CS_DIR}/depth_zoedepth" \
            --dataset cityscapes \
            --image_size 512 1024
    fi
    log "Cityscapes depth maps ready."

    # Upload depth maps to GCS
    log "Uploading depth maps to GCS..."
    gsutil -m rsync -r "${CS_DIR}/depth_zoedepth/" \
        "${BUCKET}/datasets/cityscapes/depth_zoedepth/"

    # Create TFRecords
    log "Creating Cityscapes TFRecords..."
    local TFRECORD_DIR="${CS_DIR}/tfrecords"
    mkdir -p "${TFRECORD_DIR}/train" "${TFRECORD_DIR}/val"

    python3 scripts/create_tfrecords.py \
        --config configs/cityscapes_gcs.yaml \
        --output_dir "${TFRECORD_DIR}/train" \
        --split train

    python3 scripts/create_tfrecords.py \
        --config configs/cityscapes_gcs.yaml \
        --output_dir "${TFRECORD_DIR}/val" \
        --split val

    # Upload TFRecords to GCS
    log "Uploading Cityscapes TFRecords to GCS..."
    gsutil -m rsync -r "${TFRECORD_DIR}/train/" \
        "${BUCKET}/datasets/cityscapes/tfrecords/train/"
    gsutil -m rsync -r "${TFRECORD_DIR}/val/" \
        "${BUCKET}/datasets/cityscapes/tfrecords/val/"

    # Upload raw images for evaluation/visualization
    log "Uploading Cityscapes raw images to GCS..."
    gsutil -m rsync -r "${CS_DIR}/leftImg8bit/" \
        "${BUCKET}/datasets/cityscapes/leftImg8bit/"
    gsutil -m rsync -r "${CS_DIR}/gtFine/" \
        "${BUCKET}/datasets/cityscapes/gtFine/"

    log "Cityscapes fully uploaded to GCS."
}

# ---------------------------------------------------------------------------
# 3. COCO-Stuff-27 dataset
# ---------------------------------------------------------------------------
setup_coco() {
    log "=== COCO-Stuff-27 Dataset ==="

    local COCO_DIR="${DATA_DIR}/coco"
    mkdir -p "${COCO_DIR}"

    # Check if TFRecords already on GCS
    local train_count
    train_count=$(gsutil ls "${BUCKET}/datasets/coco/tfrecords/train/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    if [ "${train_count}" -gt 0 ]; then
        log "COCO TFRecords already on GCS (${train_count} shards), skipping."
        return 0
    fi

    # Download COCO images
    if [ ! -d "${COCO_DIR}/images/train2017" ]; then
        log "Downloading COCO 2017 train images..."
        wget -q --show-progress -O "${COCO_DIR}/train2017.zip" \
            "http://images.cocodataset.org/zips/train2017.zip"
        cd "${COCO_DIR}" && unzip -q train2017.zip && rm train2017.zip && cd "${PROJECT_DIR}"
    fi

    if [ ! -d "${COCO_DIR}/images/val2017" ]; then
        log "Downloading COCO 2017 val images..."
        wget -q --show-progress -O "${COCO_DIR}/val2017.zip" \
            "http://images.cocodataset.org/zips/val2017.zip"
        cd "${COCO_DIR}" && unzip -q val2017.zip && rm val2017.zip && cd "${PROJECT_DIR}"
        # Move into images/ subdirectory if needed
        if [ ! -d "${COCO_DIR}/images" ]; then
            mkdir -p "${COCO_DIR}/images"
            mv "${COCO_DIR}/train2017" "${COCO_DIR}/images/"
            mv "${COCO_DIR}/val2017" "${COCO_DIR}/images/"
        fi
    fi

    # Download COCO-Stuff annotations (27-class pixel maps)
    if [ ! -d "${COCO_DIR}/annotations" ]; then
        log "Downloading COCO-Stuff pixel annotations..."
        wget -q --show-progress -O "${COCO_DIR}/stuffthingmaps_trainval2017.zip" \
            "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip"
        cd "${COCO_DIR}" && unzip -q stuffthingmaps_trainval2017.zip && rm stuffthingmaps_trainval2017.zip && cd "${PROJECT_DIR}"
    fi
    log "COCO images and annotations ready at ${COCO_DIR}"

    # Precompute depth maps
    if [ ! -d "${COCO_DIR}/depth_zoedepth" ] || [ "$(find ${COCO_DIR}/depth_zoedepth -name '*.npy' 2>/dev/null | wc -l)" -lt 100 ]; then
        log "Precomputing depth maps (ZoeDepth)..."
        python3 scripts/precompute_depth.py \
            --data_dir "${COCO_DIR}/images" \
            --output_dir "${COCO_DIR}/depth_zoedepth" \
            --dataset coco_stuff27 \
            --image_size 512 512
    fi
    log "COCO depth maps ready."

    # Upload depth maps to GCS
    log "Uploading depth maps to GCS..."
    gsutil -m rsync -r "${COCO_DIR}/depth_zoedepth/" \
        "${BUCKET}/datasets/coco/depth_zoedepth/"

    # Create TFRecords
    log "Creating COCO TFRecords..."
    local TFRECORD_DIR="${COCO_DIR}/tfrecords"
    mkdir -p "${TFRECORD_DIR}/train" "${TFRECORD_DIR}/val"

    python3 scripts/create_tfrecords.py \
        --config configs/coco_stuff27_gcs.yaml \
        --output_dir "${TFRECORD_DIR}/train" \
        --split train

    python3 scripts/create_tfrecords.py \
        --config configs/coco_stuff27_gcs.yaml \
        --output_dir "${TFRECORD_DIR}/val" \
        --split val

    # Upload TFRecords to GCS
    log "Uploading COCO TFRecords to GCS..."
    gsutil -m rsync -r "${TFRECORD_DIR}/train/" \
        "${BUCKET}/datasets/coco/tfrecords/train/"
    gsutil -m rsync -r "${TFRECORD_DIR}/val/" \
        "${BUCKET}/datasets/coco/tfrecords/val/"

    # Upload raw images
    log "Uploading COCO raw images to GCS..."
    gsutil -m rsync -r "${COCO_DIR}/images/" \
        "${BUCKET}/datasets/coco/images/"

    log "COCO-Stuff-27 fully uploaded to GCS."
}

# ---------------------------------------------------------------------------
# 4. Verify GCS contents
# ---------------------------------------------------------------------------
verify_gcs() {
    log "=== Verifying GCS Contents ==="

    echo ""
    echo "--- Weights ---"
    gsutil ls -l "${BUCKET}/weights/" 2>/dev/null || echo "  (empty)"

    echo ""
    echo "--- Cityscapes ---"
    local cs_train cs_val cs_depth
    cs_train=$(gsutil ls "${BUCKET}/datasets/cityscapes/tfrecords/train/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    cs_val=$(gsutil ls "${BUCKET}/datasets/cityscapes/tfrecords/val/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    cs_depth=$(gsutil ls "${BUCKET}/datasets/cityscapes/depth_zoedepth/**/*.npy" 2>/dev/null | wc -l || echo 0)
    echo "  Train TFRecords: ${cs_train} shards"
    echo "  Val TFRecords:   ${cs_val} shards"
    echo "  Depth maps:      ${cs_depth} files"

    echo ""
    echo "--- COCO-Stuff-27 ---"
    local coco_train coco_val coco_depth
    coco_train=$(gsutil ls "${BUCKET}/datasets/coco/tfrecords/train/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    coco_val=$(gsutil ls "${BUCKET}/datasets/coco/tfrecords/val/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    coco_depth=$(gsutil ls "${BUCKET}/datasets/coco/depth_zoedepth/**/*.npy" 2>/dev/null | wc -l || echo 0)
    echo "  Train TFRecords: ${coco_train} shards"
    echo "  Val TFRecords:   ${coco_val} shards"
    echo "  Depth maps:      ${coco_depth} files"

    echo ""
    echo "--- Checkpoints ---"
    gsutil ls "${BUCKET}/checkpoints/" 2>/dev/null || echo "  (empty)"

    echo ""
    echo "--- Total Bucket Size ---"
    gsutil du -sh "${BUCKET}" 2>/dev/null || echo "  (unknown)"

    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "MBPS Data Pipeline Setup"
log "Dataset: ${DATASET}"
log "Bucket:  ${BUCKET}"
echo ""

case "${DATASET}" in
    all)
        setup_dino_weights
        setup_cityscapes
        setup_coco
        verify_gcs
        ;;
    cityscapes)
        setup_cityscapes
        ;;
    coco)
        setup_coco
        ;;
    weights)
        setup_dino_weights
        ;;
    verify)
        verify_gcs
        ;;
    *)
        echo "Usage: $0 {all|cityscapes|coco|weights|verify}"
        exit 1
        ;;
esac

log "Data pipeline complete."
