#!/bin/bash
# Regenerate pseudo-labels on A6000 using santosh's k-means centroids.
#
# ROOT CAUSE: A6000 had independently-generated centroids → different k=80 cluster
# assignments → different .pt distributions → different thing/stuff split (12 things
# vs santosh's 15, only 2 overlap). This caused PQ plateau at 22%.
#
# FIX: Use santosh's centroids (committed to repo at weights/kmeans_centroids_k80_santosh.npz)
# to regenerate semantic labels + CUPS pseudo-labels on A6000. Same centroids + same images
# = same cluster assignments = same thing/stuff split = PQ should reach 28+.
#
# Run on A6000:
#   nohup bash scripts/regen_pseudolabels_a6000.sh > ~/regen_pseudolabels.log 2>&1 &
set -euo pipefail

PROJ_ROOT="$HOME/umesh/unsupervised_panoptic_segmentation"
CS_ROOT="$HOME/umesh/datasets/cityscapes"

echo "=== Step 0: Pull latest code ==="
cd "$PROJ_ROOT"
git fetch origin main
git checkout origin/main -- weights/kmeans_centroids_k80_santosh.npz scripts/regen_pseudolabels_a6000.sh mbps_pytorch/convert_to_cups_format.py

echo ""
echo "=== Step 1: Replace centroids with santosh's ==="
CENTROIDS_SRC="$PROJ_ROOT/weights/kmeans_centroids_k80_santosh.npz"
CENTROIDS_DST="$CS_ROOT/pseudo_semantic_raw_k80/kmeans_centroids.npz"

if [ ! -f "$CENTROIDS_SRC" ]; then
    echo "ERROR: Santosh centroids not found at $CENTROIDS_SRC"
    exit 1
fi

# Backup old centroids
if [ -f "$CENTROIDS_DST" ]; then
    cp "$CENTROIDS_DST" "${CENTROIDS_DST}.old"
    echo "Backed up old centroids to ${CENTROIDS_DST}.old"
fi
cp "$CENTROIDS_SRC" "$CENTROIDS_DST"
echo "Installed santosh centroids: $(md5sum "$CENTROIDS_DST")"
echo "Expected MD5: a0cf51613fcbdc14af5294a9588bfcf6"

echo ""
echo "=== Step 2: Regenerate k=80 semantic labels from new centroids ==="
echo "This re-assigns every pixel to the nearest centroid using DINOv2 features."
echo "Requires: DINOv2 features already extracted, or the extract script."

# Check if semantic labels need regenerating
# The semantic labels in pseudo_semantic_raw_k80/{train,val}/ are argmin assignments
# from DINOv2 features to centroids. Different centroids = different assignments.
SEM_DIR="$CS_ROOT/pseudo_semantic_raw_k80"
if [ -d "$SEM_DIR/train" ]; then
    OLD_COUNT=$(find "$SEM_DIR/train" -name "*.png" | wc -l)
    echo "Existing semantic labels: $OLD_COUNT files"
    echo "Backing up to pseudo_semantic_raw_k80_old/"
    mv "$SEM_DIR" "${SEM_DIR}_old"
    mkdir -p "$SEM_DIR/train" "$SEM_DIR/val"
    # Copy centroids to new dir
    cp "$CENTROIDS_SRC" "$SEM_DIR/kmeans_centroids.npz"
fi

# Regenerate semantic labels using the new centroids
# This requires the k-means assignment script
if [ -f "$PROJ_ROOT/mbps_pytorch/assign_kmeans_labels.py" ]; then
    echo "Running k-means assignment with santosh centroids..."
    python -u "$PROJ_ROOT/mbps_pytorch/assign_kmeans_labels.py" \
        --cityscapes_root "$CS_ROOT" \
        --centroids_path "$SEM_DIR/kmeans_centroids.npz" \
        --output_dir "$SEM_DIR" \
        --num_classes 80
else
    echo ""
    echo "WARNING: assign_kmeans_labels.py not found."
    echo "You need to regenerate semantic labels from the new centroids."
    echo "The old labels used DIFFERENT centroids and must NOT be reused."
    echo ""
    echo "If you have DINOv2 features cached, run the assignment script."
    echo "Otherwise, restore old labels and just copy the full 120MB pseudo-labels from santosh."
    echo ""
    echo "Quick alternative: if the old semantic labels were generated from features"
    echo "that are STILL on disk, you can re-run the clustering assignment step only."
    # Restore old labels since we can't regenerate
    if [ -d "${SEM_DIR}_old" ]; then
        rm -rf "$SEM_DIR"
        mv "${SEM_DIR}_old" "$SEM_DIR"
        cp "$CENTROIDS_SRC" "$SEM_DIR/kmeans_centroids.npz"
        echo "Restored old semantic labels (WARNING: these use different centroids!)"
    fi
    echo ""
    echo "IMPORTANT: If semantic labels don't match the centroids, the pseudo-labels"
    echo "will still be wrong. You MUST regenerate semantic labels from the new centroids."
    exit 1
fi

echo ""
echo "=== Step 3: Regenerate CUPS pseudo-labels ==="
PSEUDO_DIR="$CS_ROOT/cups_pseudo_labels_depthpro_tau020"

# Backup old pseudo-labels
if [ -d "$PSEUDO_DIR" ]; then
    mv "$PSEUDO_DIR" "${PSEUDO_DIR}_old_wrong_split"
    echo "Backed up old pseudo-labels"
fi

cd "$PROJ_ROOT"

# Generate for train split
echo "--- Train split ---"
python -u mbps_pytorch/convert_to_cups_format.py \
    --cityscapes_root "$CS_ROOT" \
    --semantic_subdir pseudo_semantic_raw_k80 \
    --output_subdir cups_pseudo_labels_depthpro_tau020 \
    --split train \
    --num_classes 80 \
    --depth_cc_instances \
    --centroids_path "$SEM_DIR/kmeans_centroids.npz" \
    --depth_subdir depth_depthpro \
    --grad_threshold 0.20 \
    --depth_blur_sigma 0.0 \
    --dilation_iters 3 \
    --min_instance_area 1000

echo ""
echo "=== Step 4: Verify ==="
OUT_DIR="$CS_ROOT/cups_pseudo_labels_depthpro_tau020"
SEM_OUT=$(find "$OUT_DIR" -name "*_semantic.png" 2>/dev/null | wc -l)
INST_OUT=$(find "$OUT_DIR" -name "*_instance.png" 2>/dev/null | wc -l)
PT_OUT=$(find "$OUT_DIR" -name "*.pt" 2>/dev/null | wc -l)
VAL_COUNT=$(ls "$OUT_DIR" 2>/dev/null | grep -E '^(frankfurt|lindau|munster)_' | wc -l)
echo "Generated: $SEM_OUT semantic, $INST_OUT instance, $PT_OUT .pt files"
echo "Val city contamination: $VAL_COUNT (should be 0)"

if [ "$SEM_OUT" -lt 2975 ] || [ "$INST_OUT" -lt 2975 ]; then
    echo "WARNING: Expected >= 2975 training files."
fi

echo ""
echo "=== Done ==="
echo "Now restart CUPS training:"
echo "  cd ~/umesh/unsupervised_panoptic_segmentation/refs/cups"
echo "  nohup python -u train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_anydesk.yaml > ~/cups_fixed_centroids.log 2>&1 &"
echo ""
echo "Check the log for: Thing classes: (16, 14, 73, 3, 29, 46, 38, 75, 37, 11, 65, 32, 15, 62, 45)"
echo "If you see 15 thing classes matching santosh, the fix worked."
