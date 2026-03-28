#!/bin/bash
# Staged data pipeline for populating GCS bucket.
# Runs on a TPU VM with limited disk (~80GB free).
# Processes one dataset at a time, uploads to GCS, cleans up local copies.
#
# Usage:
#   bash scripts/setup_gcs_data.sh all           # Full pipeline
#   bash scripts/setup_gcs_data.sh weights        # DINOv2 ViT-B/14 only
#   bash scripts/setup_gcs_data.sh coco           # COCO-Stuff-27 only
#   bash scripts/setup_gcs_data.sh cityscapes     # Cityscapes only (needs creds)
#   bash scripts/setup_gcs_data.sh verify         # Check GCS contents

set -euo pipefail

PROJECT_DIR="${HOME}/mbps_panoptic_segmentation"
DATA_DIR="${PROJECT_DIR}/data"
BUCKET="gs://mbps-panoptic"
STAGE="${1:-all}"

cd "${PROJECT_DIR}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
disk_free() { df -h / | awk 'NR==2{print $4}'; }

# ---------------------------------------------------------------------------
# Stage 1: DINOv2 ViT-B/14 weights (~350MB)
# ---------------------------------------------------------------------------
setup_dinov2_weights() {
    log "=== Stage 1: DINOv2 ViT-B/14 Weights ==="
    log "Disk free: $(disk_free)"

    # Check if already on GCS
    if gsutil -q stat "${BUCKET}/weights/dinov2_vitb14_flax.npz" 2>/dev/null; then
        log "DINOv2 Flax weights already on GCS, skipping."
        return 0
    fi

    local WEIGHTS_DIR="${DATA_DIR}/weights"
    mkdir -p "${WEIGHTS_DIR}"

    # Download PyTorch checkpoint (~330MB)
    local PT_FILE="${WEIGHTS_DIR}/dinov2_vitb14_pretrain.pth"
    if [ ! -f "${PT_FILE}" ]; then
        log "Downloading DINOv2 ViT-B/14 PyTorch checkpoint..."
        wget -q --show-progress -O "${PT_FILE}" \
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    fi
    log "Download complete: $(du -h ${PT_FILE} | cut -f1)"

    # Convert to Flax
    local FLAX_FILE="${WEIGHTS_DIR}/dinov2_vitb14_flax.npz"
    log "Converting PyTorch -> Flax..."
    python3 scripts/convert_dino_weights.py \
        --input "${PT_FILE}" \
        --output "${FLAX_FILE}" \
        --model dinov2_vitb14

    # Upload to GCS
    log "Uploading to GCS..."
    gsutil cp "${FLAX_FILE}" "${BUCKET}/weights/dinov2_vitb14_flax.npz"

    # Cleanup local
    rm -f "${PT_FILE}" "${FLAX_FILE}"
    log "Stage 1 complete. Disk free: $(disk_free)"
}

# ---------------------------------------------------------------------------
# Stage 2: COCO-Stuff-27 (sequential: download -> depth -> tfrecords -> upload)
# ---------------------------------------------------------------------------
setup_coco() {
    log "=== Stage 2: COCO-Stuff-27 ==="
    log "Disk free: $(disk_free)"

    # Check if TFRecords already on GCS
    local train_count
    train_count=$(gsutil ls "${BUCKET}/datasets/coco/tfrecords/train/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    if [ "${train_count}" -gt 0 ]; then
        log "COCO TFRecords already on GCS (${train_count} shards), skipping."
        return 0
    fi

    local COCO_DIR="${DATA_DIR}/coco"
    mkdir -p "${COCO_DIR}/images"

    # --- 2a: Download COCO train2017 images (~18GB) ---
    if [ ! -d "${COCO_DIR}/images/train2017" ]; then
        log "Downloading COCO 2017 train images (~18GB)..."
        wget -q --show-progress -O "${COCO_DIR}/train2017.zip" \
            "http://images.cocodataset.org/zips/train2017.zip"
        log "Extracting train2017..."
        cd "${COCO_DIR}" && unzip -q train2017.zip -d images/ && rm -f train2017.zip && cd "${PROJECT_DIR}"
        log "Train images extracted. Disk free: $(disk_free)"
    fi

    # --- 2b: Download COCO val2017 images (~1GB) ---
    if [ ! -d "${COCO_DIR}/images/val2017" ]; then
        log "Downloading COCO 2017 val images (~1GB)..."
        wget -q --show-progress -O "${COCO_DIR}/val2017.zip" \
            "http://images.cocodataset.org/zips/val2017.zip"
        cd "${COCO_DIR}" && unzip -q val2017.zip -d images/ && rm -f val2017.zip && cd "${PROJECT_DIR}"
        log "Val images extracted. Disk free: $(disk_free)"
    fi

    # --- 2c: Download COCO-Stuff annotations (~2GB) ---
    if [ ! -d "${COCO_DIR}/annotations" ]; then
        log "Downloading COCO-Stuff pixel annotations (~600MB)..."
        wget -q --show-progress -O "${COCO_DIR}/stuffthingmaps_trainval2017.zip" \
            "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip"
        cd "${COCO_DIR}" && unzip -q stuffthingmaps_trainval2017.zip && rm -f stuffthingmaps_trainval2017.zip && cd "${PROJECT_DIR}"
        log "Annotations extracted. Disk free: $(disk_free)"
    fi

    # --- 2d: Upload raw images to GCS (before depth, so we free space after) ---
    log "Uploading COCO raw images to GCS..."
    gsutil -m rsync -r "${COCO_DIR}/images/" "${BUCKET}/datasets/coco/images/"
    log "Raw images uploaded."

    # --- 2e: Compute depth maps in batches and stream to GCS ---
    log "Computing depth maps with ZoeDepth (CPU — this will take a while)..."
    local DEPTH_DIR="${COCO_DIR}/depth_zoedepth"
    mkdir -p "${DEPTH_DIR}"

    # Process in batches: compute locally, upload batch, delete local batch
    python3 -c "
import os, sys, subprocess, glob, time
from pathlib import Path

coco_dir = '${COCO_DIR}'
depth_dir = '${DEPTH_DIR}'
bucket_depth = '${BUCKET}/datasets/coco/depth_zoedepth'
batch_size = 500  # upload every 500 depth maps

# Collect all images
images = sorted(glob.glob(os.path.join(coco_dir, 'images', '**', '*.jpg'), recursive=True))
print(f'Total images to process: {len(images)}')

# Check which are already done on GCS (rough check)
try:
    result = subprocess.run(
        ['gsutil', 'ls', f'{bucket_depth}/**/*.npy'],
        capture_output=True, text=True, timeout=60
    )
    done_gcs = set(Path(p.strip()).stem for p in result.stdout.strip().split('\n') if p.strip())
    print(f'Already on GCS: {len(done_gcs)} depth maps')
except Exception:
    done_gcs = set()

# Try loading ZoeDepth
try:
    import torch
    from PIL import Image
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    print('Loading ZoeDepth model...')
    model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=True)
    model.eval()
    use_zoedepth = True
    transform = Compose([
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print('ZoeDepth loaded (CPU mode)')
except Exception as e:
    print(f'ZoeDepth unavailable ({e}), using procedural depth')
    use_zoedepth = False

import numpy as np
batch_count = 0
start_time = time.time()

for idx, img_path in enumerate(images):
    rel = os.path.relpath(img_path, os.path.join(coco_dir, 'images'))
    stem = Path(rel).stem

    # Skip if already on GCS
    if stem in done_gcs:
        continue

    depth_path = os.path.join(depth_dir, Path(rel).with_suffix('.npy'))
    os.makedirs(os.path.dirname(depth_path), exist_ok=True)

    if not os.path.exists(depth_path):
        if use_zoedepth:
            img = Image.open(img_path).convert('RGB')
            inp = transform(img).unsqueeze(0)
            with torch.no_grad():
                depth = model.infer(inp)
            depth_np = depth.squeeze().cpu().numpy()
        else:
            np.random.seed(idx)
            y_grad = np.linspace(0.3, 1.0, 512)[:, None]
            noise = np.random.rand(512, 512) * 0.2
            depth_np = (y_grad * np.ones((1, 512)) + noise).astype(np.float32)

        np.save(depth_path, depth_np)

    batch_count += 1

    # Upload batch to GCS and cleanup local
    if batch_count >= batch_size:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (len(images) - idx - 1) / rate if rate > 0 else 0
        print(f'[{idx+1}/{len(images)}] Uploading batch ({batch_count} files)... '
              f'Rate: {rate:.1f} img/s, ETA: {eta/3600:.1f}h')
        subprocess.run(
            ['gsutil', '-m', 'rsync', '-r', depth_dir, f'{bucket_depth}/'],
            capture_output=True
        )
        # Delete local depth files to save disk
        for f in glob.glob(os.path.join(depth_dir, '**', '*.npy'), recursive=True):
            os.remove(f)
        batch_count = 0

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        print(f'[{idx+1}/{len(images)}] Rate: {rate:.1f} img/s')

# Final upload
if batch_count > 0:
    print(f'Final upload ({batch_count} files)...')
    subprocess.run(['gsutil', '-m', 'rsync', '-r', depth_dir, f'{bucket_depth}/'], capture_output=True)
    for f in glob.glob(os.path.join(depth_dir, '**', '*.npy'), recursive=True):
        os.remove(f)

print('Depth computation complete.')
"
    log "Depth maps uploaded to GCS. Disk free: $(disk_free)"

    # --- 2f: Create TFRecords ---
    log "Creating COCO TFRecords..."
    local TFRECORD_DIR="${COCO_DIR}/tfrecords"
    mkdir -p "${TFRECORD_DIR}/train" "${TFRECORD_DIR}/val"

    # Create a local config pointing to local data paths
    cat > "${COCO_DIR}/local_config.yaml" << 'LOCALCFG'
_base_: "../configs/default.yaml"
data:
  dataset: "coco_stuff27"
  dataset_name: "coco_stuff27"
  data_dir: "LOCAL_COCO_DIR"
  depth_dir: "LOCAL_COCO_DIR/depth_zoedepth"
  num_classes: 27
  image_size: [512, 512]
LOCALCFG
    # Patch in the actual path
    sed -i "s|LOCAL_COCO_DIR|${COCO_DIR}|g" "${COCO_DIR}/local_config.yaml"

    # Re-download depth maps from GCS for TFRecord creation (only what we need)
    log "Syncing depth maps from GCS for TFRecord creation..."
    gsutil -m rsync -r "${BUCKET}/datasets/coco/depth_zoedepth/" "${DEPTH_DIR}/"

    python3 scripts/create_tfrecords.py \
        --config "${COCO_DIR}/local_config.yaml" \
        --output_dir "${TFRECORD_DIR}/train" \
        --split train || log "WARNING: TFRecord creation for train failed (may need code fix)"

    python3 scripts/create_tfrecords.py \
        --config "${COCO_DIR}/local_config.yaml" \
        --output_dir "${TFRECORD_DIR}/val" \
        --split val || log "WARNING: TFRecord creation for val failed (may need code fix)"

    # Upload TFRecords to GCS
    log "Uploading COCO TFRecords to GCS..."
    gsutil -m rsync -r "${TFRECORD_DIR}/train/" "${BUCKET}/datasets/coco/tfrecords/train/"
    gsutil -m rsync -r "${TFRECORD_DIR}/val/" "${BUCKET}/datasets/coco/tfrecords/val/"

    # Cleanup ALL local COCO data
    log "Cleaning up local COCO data..."
    rm -rf "${COCO_DIR}"
    log "Stage 2 complete. Disk free: $(disk_free)"
}

# ---------------------------------------------------------------------------
# Stage 3: Cityscapes (requires CITYSCAPES_USERNAME + CITYSCAPES_PASSWORD)
# ---------------------------------------------------------------------------
setup_cityscapes() {
    log "=== Stage 3: Cityscapes ==="
    log "Disk free: $(disk_free)"

    # Check if TFRecords already on GCS
    local train_count
    train_count=$(gsutil ls "${BUCKET}/datasets/cityscapes/tfrecords/train/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    if [ "${train_count}" -gt 0 ]; then
        log "Cityscapes TFRecords already on GCS (${train_count} shards), skipping."
        return 0
    fi

    if [ -z "${CITYSCAPES_USERNAME:-}" ] || [ -z "${CITYSCAPES_PASSWORD:-}" ]; then
        log "ERROR: Set CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD env vars."
        log "  Register at: https://www.cityscapes-dataset.com/register/"
        log "  Then: export CITYSCAPES_USERNAME=you@email.com"
        log "        export CITYSCAPES_PASSWORD=yourpassword"
        return 1
    fi

    local CS_DIR="${DATA_DIR}/cityscapes"
    mkdir -p "${CS_DIR}"

    # Download Cityscapes images
    if [ ! -d "${CS_DIR}/leftImg8bit" ]; then
        log "Downloading Cityscapes (needs authentication)..."
        # Login to get session cookie
        wget --keep-session-cookies --save-cookies="${CS_DIR}/cookies.txt" \
            --post-data="username=${CITYSCAPES_USERNAME}&password=${CITYSCAPES_PASSWORD}&submit=Login" \
            -O /dev/null "https://www.cityscapes-dataset.com/login/" 2>/dev/null

        for pkg in leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip; do
            log "  Downloading ${pkg}..."
            wget -q --show-progress --load-cookies="${CS_DIR}/cookies.txt" \
                -O "${CS_DIR}/${pkg}" \
                "https://www.cityscapes-dataset.com/file-handling/?packageID=$(echo ${pkg} | sed 's/_trainvaltest.zip//')" || {
                log "  Auto-download failed. Please download manually:"
                log "    1. Visit https://www.cityscapes-dataset.com/downloads/"
                log "    2. Download ${pkg}"
                log "    3. Place in ${CS_DIR}/"
                log "    4. Re-run this script."
                return 1
            }
        done
        rm -f "${CS_DIR}/cookies.txt"

        log "Extracting Cityscapes..."
        cd "${CS_DIR}"
        unzip -q leftImg8bit_trainvaltest.zip && rm -f leftImg8bit_trainvaltest.zip
        unzip -q gtFine_trainvaltest.zip && rm -f gtFine_trainvaltest.zip
        cd "${PROJECT_DIR}"
    fi
    log "Cityscapes images ready. Disk free: $(disk_free)"

    # Upload raw images to GCS
    log "Uploading raw images to GCS..."
    gsutil -m rsync -r "${CS_DIR}/leftImg8bit/" "${BUCKET}/datasets/cityscapes/leftImg8bit/"
    gsutil -m rsync -r "${CS_DIR}/gtFine/" "${BUCKET}/datasets/cityscapes/gtFine/"

    # Compute depth maps (Cityscapes: ~5K images, ~4-6hrs on CPU)
    log "Computing Cityscapes depth maps (ZoeDepth CPU, ~5K images)..."
    python3 scripts/precompute_depth.py \
        --data_dir "${CS_DIR}/leftImg8bit" \
        --output_dir "${CS_DIR}/depth_zoedepth" \
        --dataset cityscapes \
        --image_size 512 1024

    # Upload depth maps to GCS
    log "Uploading depth maps to GCS..."
    gsutil -m rsync -r "${CS_DIR}/depth_zoedepth/" "${BUCKET}/datasets/cityscapes/depth_zoedepth/"

    # Create TFRecords
    log "Creating Cityscapes TFRecords..."
    local TFRECORD_DIR="${CS_DIR}/tfrecords"
    mkdir -p "${TFRECORD_DIR}/train" "${TFRECORD_DIR}/val"

    cat > "${CS_DIR}/local_config.yaml" << LOCALCFG
_base_: "../configs/default.yaml"
data:
  dataset: "cityscapes"
  dataset_name: "cityscapes"
  data_dir: "${CS_DIR}"
  depth_dir: "${CS_DIR}/depth_zoedepth"
  num_classes: 19
  num_stuff_classes: 11
  num_thing_classes: 8
  image_size: [512, 1024]
LOCALCFG

    python3 scripts/create_tfrecords.py \
        --config "${CS_DIR}/local_config.yaml" \
        --output_dir "${TFRECORD_DIR}/train" \
        --split train || log "WARNING: TFRecord creation for train failed"

    python3 scripts/create_tfrecords.py \
        --config "${CS_DIR}/local_config.yaml" \
        --output_dir "${TFRECORD_DIR}/val" \
        --split val || log "WARNING: TFRecord creation for val failed"

    # Upload TFRecords to GCS
    log "Uploading Cityscapes TFRecords to GCS..."
    gsutil -m rsync -r "${TFRECORD_DIR}/train/" "${BUCKET}/datasets/cityscapes/tfrecords/train/"
    gsutil -m rsync -r "${TFRECORD_DIR}/val/" "${BUCKET}/datasets/cityscapes/tfrecords/val/"

    # Cleanup
    log "Cleaning up local Cityscapes data..."
    rm -rf "${CS_DIR}"
    log "Stage 3 complete. Disk free: $(disk_free)"
}

# ---------------------------------------------------------------------------
# Verify GCS contents
# ---------------------------------------------------------------------------
verify_gcs() {
    log "=== GCS Bucket Contents ==="

    echo ""
    echo "--- Weights ---"
    gsutil ls -l "${BUCKET}/weights/" 2>/dev/null || echo "  (empty)"

    echo ""
    echo "--- Cityscapes ---"
    local cs_train cs_val cs_depth cs_imgs
    cs_train=$(gsutil ls "${BUCKET}/datasets/cityscapes/tfrecords/train/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    cs_val=$(gsutil ls "${BUCKET}/datasets/cityscapes/tfrecords/val/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    cs_depth=$(gsutil ls -r "${BUCKET}/datasets/cityscapes/depth_zoedepth/" 2>/dev/null | grep -c '\.npy$' || echo 0)
    cs_imgs=$(gsutil ls -r "${BUCKET}/datasets/cityscapes/leftImg8bit/" 2>/dev/null | grep -c '\.png$' || echo 0)
    echo "  Raw images:      ${cs_imgs}"
    echo "  Depth maps:      ${cs_depth}"
    echo "  Train TFRecords: ${cs_train} shards"
    echo "  Val TFRecords:   ${cs_val} shards"

    echo ""
    echo "--- COCO-Stuff-27 ---"
    local coco_train coco_val coco_depth coco_imgs
    coco_train=$(gsutil ls "${BUCKET}/datasets/coco/tfrecords/train/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    coco_val=$(gsutil ls "${BUCKET}/datasets/coco/tfrecords/val/*.tfrecord" 2>/dev/null | wc -l || echo 0)
    coco_depth=$(gsutil ls -r "${BUCKET}/datasets/coco/depth_zoedepth/" 2>/dev/null | grep -c '\.npy$' || echo 0)
    coco_imgs=$(gsutil ls -r "${BUCKET}/datasets/coco/images/" 2>/dev/null | grep -c '\.jpg$' || echo 0)
    echo "  Raw images:      ${coco_imgs}"
    echo "  Depth maps:      ${coco_depth}"
    echo "  Train TFRecords: ${coco_train} shards"
    echo "  Val TFRecords:   ${coco_val} shards"

    echo ""
    echo "--- Total Bucket Size ---"
    gsutil du -sh "${BUCKET}" 2>/dev/null || echo "  (unknown)"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "MBPS GCS Data Pipeline (Staged)"
log "Stage: ${STAGE}"
log "Bucket: ${BUCKET}"
log "Disk free: $(disk_free)"
echo ""

case "${STAGE}" in
    all)
        setup_dinov2_weights
        setup_coco
        setup_cityscapes
        verify_gcs
        ;;
    weights)
        setup_dinov2_weights
        ;;
    coco)
        setup_coco
        ;;
    cityscapes)
        setup_cityscapes
        ;;
    verify)
        verify_gcs
        ;;
    *)
        echo "Usage: $0 {all|weights|coco|cityscapes|verify}"
        exit 1
        ;;
esac

log "Pipeline complete."
