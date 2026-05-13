#!/bin/bash
# Full pipeline: Generate DA3-trained V3/DCFA adapter k=80 pseudo-labels +
# convert to CUPS format using Depth Anything V3 depth-guided instances.
#
# This is the depth-model replacement ablation for:
#   V3/DCFA semantics + DepthPro tau=0.20
#
# It keeps centroids fixed but retrains/applies the semantic adapter with DA3:
#   DINOv2/CAUSE-TR 90D codes + DA3 sinusoidal depth -> DCFA residual adapter
#   -> existing k=80 centroids
#
# It also runs the matching SIMCF-ABC pass:
#   cups_pseudo_labels_adapter_V3_da3dcfa_tau020
#     -> cups_pseudo_labels_dcfa_da3_simcf_abc
#
# Target: local MPS/CPU preprocessing on the MacBook.
set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation}"
CS_ROOT="${CS_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PYTHON="${PYTHON:-/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python}"

ADAPTER="${ADAPTER:-$PROJ_ROOT/results/depth_adapter/V3_da3_dd16_h384_l2/best.pt}"
CENTROIDS="${CENTROIDS:-$CS_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz}"

CODES_SUBDIR="${CODES_SUBDIR:-cause_codes_90d_da3}"
SEM_SUBDIR="${SEM_SUBDIR:-pseudo_semantic_adapter_V3_da3_k80}"
DEPTH_SUBDIR="${DEPTH_SUBDIR:-depth_dav3}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-cups_pseudo_labels_adapter_V3_da3dcfa_tau020}"
SIMCF_OUTPUT_SUBDIR="${SIMCF_OUTPUT_SUBDIR:-cups_pseudo_labels_dcfa_da3_simcf_abc}"
RUN_SIMCF="${RUN_SIMCF:-1}"

# Hold this fixed to the DepthPro+DCFA methodology for a clean depth-model
# substitution ablation.
GRAD_THRESHOLD="${GRAD_THRESHOLD:-0.20}"
MIN_INSTANCE_AREA="${MIN_INSTANCE_AREA:-1000}"
DEPTH_BLUR_SIGMA="${DEPTH_BLUR_SIGMA:-0.0}"
DILATION_ITERS="${DILATION_ITERS:-3}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.85}"
SIGMA_THRESHOLD="${SIGMA_THRESHOLD:-3.0}"

echo "=== V3/DCFA Adapter + DA3 CUPS Label Pipeline ==="
echo "Project:       $PROJ_ROOT"
echo "Cityscapes:    $CS_ROOT"
echo "Semantics:     $SEM_SUBDIR"
echo "Code cache:    $CODES_SUBDIR"
echo "Depth:         $DEPTH_SUBDIR"
echo "Output:        $OUTPUT_SUBDIR"
echo "SIMCF output:  $SIMCF_OUTPUT_SUBDIR"
echo "tau:           $GRAD_THRESHOLD"
echo "min area:      $MIN_INSTANCE_AREA"
echo "sigma:         $DEPTH_BLUR_SIGMA"
echo "dilation:      $DILATION_ITERS"
echo "run SIMCF:     $RUN_SIMCF"
echo ""

cd "$PROJ_ROOT"

count_files() {
    local root="$1"
    local pattern="$2"
    if [ ! -d "$root" ]; then
        echo 0
        return
    fi
    find "$root" -name "$pattern" | wc -l | tr -d ' '
}

echo "--- Step 0: Verify prerequisites ---"

if [ ! -d "$CS_ROOT/$CODES_SUBDIR/train" ]; then
    echo "ERROR: DA3 CAUSE-code cache not found at $CS_ROOT/$CODES_SUBDIR/train"
    exit 1
fi
echo "  CAUSE train codes: $(find "$CS_ROOT/$CODES_SUBDIR/train" -name "*_codes.npy" | wc -l | tr -d ' ')"
echo "  DA3 cached train depth: $(find "$CS_ROOT/$CODES_SUBDIR/train" -name "*_depth.npy" | wc -l | tr -d ' ')"

if [ ! -d "$CS_ROOT/$DEPTH_SUBDIR/train" ] || [ ! -d "$CS_ROOT/$DEPTH_SUBDIR/val" ]; then
    echo "ERROR: DA3 depth maps not found at $CS_ROOT/$DEPTH_SUBDIR/{train,val}"
    exit 1
fi
echo "  DA3 train depth:   $(find "$CS_ROOT/$DEPTH_SUBDIR/train" -name "*.npy" | wc -l | tr -d ' ')"
echo "  DA3 val depth:     $(find "$CS_ROOT/$DEPTH_SUBDIR/val" -name "*.npy" | wc -l | tr -d ' ')"

if [ ! -f "$ADAPTER" ]; then
    echo "ERROR: V3/DCFA adapter checkpoint not found at $ADAPTER"
    exit 1
fi
echo "  Adapter:           $ADAPTER"

if [ ! -f "$CENTROIDS" ]; then
    echo "ERROR: V3 k=80 centroids not found at $CENTROIDS"
    exit 1
fi
echo "  Centroids:         $CENTROIDS"

echo ""
echo "--- Step 1: Generate/load V3/DCFA k=80 raw cluster semantics ---"
for SPLIT in train; do
    EXPECTED=2975
    COUNT=$(count_files "$CS_ROOT/$SEM_SUBDIR/$SPLIT" "*.png")
    if [ "$COUNT" -ge "$EXPECTED" ]; then
        echo "  $SPLIT: using existing $COUNT semantic PNGs"
        continue
    fi

    echo "  $SPLIT: generating semantics into $CS_ROOT/$SEM_SUBDIR/$SPLIT"
    "$PYTHON" -u mbps_pytorch/generate_depth_overclustered_semantics.py \
        --cityscapes_root "$CS_ROOT" \
        --split "$SPLIT" \
        --adapter_checkpoint "$ADAPTER" \
        --codes_subdir "$CODES_SUBDIR" \
        --depth_subdir "$DEPTH_SUBDIR" \
        --variant sinusoidal \
        --alpha 0.1 \
        --k 80 \
        --load_centroids "$CENTROIDS" \
        --output_subdir "$SEM_SUBDIR" \
        --skip_crf \
        --raw_clusters
done

echo ""
echo "--- Step 2: Convert to CUPS format with DA3 depth-guided CC ---"
for SPLIT in train; do
    echo "  $SPLIT split"
    "$PYTHON" -u mbps_pytorch/convert_to_cups_format.py \
        --cityscapes_root "$CS_ROOT" \
        --semantic_subdir "$SEM_SUBDIR" \
        --output_subdir "$OUTPUT_SUBDIR" \
        --split "$SPLIT" \
        --num_classes 80 \
        --depth_cc_instances \
        --centroids_path "$CENTROIDS" \
        --depth_subdir "$DEPTH_SUBDIR" \
        --grad_threshold "$GRAD_THRESHOLD" \
        --depth_blur_sigma "$DEPTH_BLUR_SIGMA" \
        --dilation_iters "$DILATION_ITERS" \
        --min_instance_area "$MIN_INSTANCE_AREA"
done

echo ""
echo "--- Step 3: Verify DA3 CUPS pseudo-labels ---"
OUT_DIR="$CS_ROOT/$OUTPUT_SUBDIR"
SEM_OUT=$(count_files "$OUT_DIR" "*_semantic.png")
INST_OUT=$(count_files "$OUT_DIR" "*_instance.png")
PT_OUT=$(count_files "$OUT_DIR" "*.pt")
echo "  Semantic PNGs: $SEM_OUT"
echo "  Instance PNGs: $INST_OUT"
echo "  Distribution .pt: $PT_OUT"

if [ "$SEM_OUT" -lt 2975 ] || [ "$INST_OUT" -lt 2975 ] || [ "$PT_OUT" -lt 2975 ]; then
    echo "ERROR: Expected >= 2975 train files per type."
    exit 1
fi

if [ "$RUN_SIMCF" = "1" ]; then
    echo ""
    echo "--- Step 4: Apply SIMCF-ABC with DA3 depth stats ---"
    "$PYTHON" -u scripts/refine_simcf.py \
        --input_dir "$OUT_DIR" \
        --output_dir "$CS_ROOT/$SIMCF_OUTPUT_SUBDIR" \
        --centroids_path "$CENTROIDS" \
        --cityscapes_root "$CS_ROOT" \
        --steps A,B,C \
        --features_subdir dinov3_features \
        --depth_subdir "$DEPTH_SUBDIR" \
        --sim_threshold "$SIM_THRESHOLD" \
        --sigma_threshold "$SIGMA_THRESHOLD" \
        --num_clusters 80

    echo ""
    echo "--- Step 5: Verify DA3 + DCFA + SIMCF-ABC labels ---"
    SIMCF_OUT="$CS_ROOT/$SIMCF_OUTPUT_SUBDIR"
    SIMCF_SEM=$(count_files "$SIMCF_OUT" "*_semantic.png")
    SIMCF_INST=$(count_files "$SIMCF_OUT" "*_instance.png")
    SIMCF_PT=$(count_files "$SIMCF_OUT" "*.pt")
    echo "  Semantic PNGs: $SIMCF_SEM"
    echo "  Instance PNGs: $SIMCF_INST"
    echo "  Distribution .pt: $SIMCF_PT"

    if [ "$SIMCF_SEM" -lt 2975 ] || [ "$SIMCF_INST" -lt 2975 ] || [ "$SIMCF_PT" -lt 2975 ]; then
        echo "ERROR: Expected >= 2975 SIMCF files per type."
        exit 1
    fi
fi

echo ""
echo "=== Pipeline complete ==="
echo "CUPS pseudo-labels: $OUT_DIR"
if [ "$RUN_SIMCF" = "1" ]; then
    echo "SIMCF-ABC labels:  $CS_ROOT/$SIMCF_OUTPUT_SUBDIR"
fi
echo ""
echo "Suggested val quality check:"
echo "$PYTHON -u mbps_pytorch/evaluate_panoptic_combined.py \\"
echo "  --sem_dir \"$CS_ROOT/$SEM_SUBDIR/val\" \\"
echo "  --cityscapes_root \"$CS_ROOT\" \\"
echo "  --depth_subdir \"$DEPTH_SUBDIR\" \\"
echo "  --tau \"$GRAD_THRESHOLD\" --min_area \"$MIN_INSTANCE_AREA\" \\"
echo "  --sigma \"$DEPTH_BLUR_SIGMA\" --dilation \"$DILATION_ITERS\""
