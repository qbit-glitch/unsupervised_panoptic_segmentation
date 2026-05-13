#!/bin/bash
# DCFA v2 Full Pipeline: Pre-extraction → Ablation Sweep → Evaluation
#
# Runs everything end-to-end on MPS (M4 Pro 48GB).
# Total time: ~6-8 hours (45min extraction + 5-6h training + eval)
#
# Usage:
#   nohup bash scripts/run_dcfa_v2_full_pipeline.sh \
#       > logs/dcfa_v2_pipeline.log 2>&1 &
#   tail -f logs/dcfa_v2_pipeline.log
set -euo pipefail

CS_ROOT="${CS_ROOT:-/Users/qbit-glitch/Desktop/datasets/cityscapes}"
PROJ_ROOT="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
CENTROIDS="$CS_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz"
BASE_OUT="$PROJ_ROOT/results/depth_adapter/DCFA_v2"
COMMON="--cityscapes_root $CS_ROOT --epochs 20 --lambda_preserve 20.0 --lr 1e-3 --seed 42 --depth_dim 16 --batch_size 8"

cd "$PROJ_ROOT"
mkdir -p logs

echo "================================================================"
echo "  DCFA v2 Full Pipeline"
echo "  Started: $(date)"
echo "  Cityscapes: $CS_ROOT"
echo "================================================================"
echo ""

# ─── Phase 0: Pre-extraction ─────────────────────────────────────────────────

echo ">>> Phase 0a: Surface normals extraction"
NORMALS_COUNT=$(find "$CS_ROOT/cause_codes_90d" -name "*_normals.npy" 2>/dev/null | wc -l | tr -d ' ')
if [ "$NORMALS_COUNT" -ge 3475 ]; then
    echo "  Already extracted ($NORMALS_COUNT files). Skipping."
else
    echo "  Extracting normals for train+val..."
    python3 -u scripts/extract_surface_normals.py --cityscapes_root "$CS_ROOT"
    echo "  Done: $(find "$CS_ROOT/cause_codes_90d" -name "*_normals.npy" | wc -l | tr -d ' ') files"
fi
echo ""

echo ">>> Phase 0b: DINOv2 768D feature extraction"
DINO_COUNT=$(find "$CS_ROOT/cause_codes_90d" -name "*_dino768.npy" 2>/dev/null | wc -l | tr -d ' ')
if [ "$DINO_COUNT" -ge 3475 ]; then
    echo "  Already extracted ($DINO_COUNT files). Skipping."
else
    echo "  Extracting 768D features for train+val (~45 min on MPS)..."
    python3 -u scripts/extract_cause_codes.py \
        --cityscapes_root "$CS_ROOT" --save_dino768
    echo "  Done: $(find "$CS_ROOT/cause_codes_90d" -name "*_dino768.npy" | wc -l | tr -d ' ') files"
fi
echo ""

# ─── Verify all prerequisites ────────────────────────────────────────────────

echo ">>> Verifying prerequisites..."
CODES_VAL=$(find "$CS_ROOT/cause_codes_90d/val" -name "*_codes.npy" | wc -l | tr -d ' ')
CODES_TRAIN=$(find "$CS_ROOT/cause_codes_90d/train" -name "*_codes.npy" | wc -l | tr -d ' ')
NORMALS_VAL=$(find "$CS_ROOT/cause_codes_90d/val" -name "*_normals.npy" | wc -l | tr -d ' ')
DINO_VAL=$(find "$CS_ROOT/cause_codes_90d/val" -name "*_dino768.npy" | wc -l | tr -d ' ')
echo "  90D codes:  train=$CODES_TRAIN, val=$CODES_VAL"
echo "  Normals:    val=$NORMALS_VAL"
echo "  DINOv2 768D: val=$DINO_VAL"

if [ "$CODES_VAL" -lt 500 ] || [ "$NORMALS_VAL" -lt 500 ] || [ "$DINO_VAL" -lt 500 ]; then
    echo "ERROR: Missing prerequisite data. Aborting."
    exit 1
fi
echo "  All prerequisites OK"
echo ""

# ─── Helper function ─────────────────────────────────────────────────────────

run_experiment() {
    local name="$1"
    shift
    local out_dir="$BASE_OUT/${name}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$name] Starting at $(date '+%H:%M:%S')"
    echo "  Output: ${out_dir}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    local t0=$(date +%s)
    python3 -u mbps_pytorch/train_depth_adapter.py \
        $COMMON --output_dir "$out_dir" "$@"
    local t1=$(date +%s)
    echo "  [$name] Done in $((t1 - t0))s"
}

# ─── Phase 1: Individual Ablations ───────────────────────────────────────────

echo "================================================================"
echo "  Phase 1: Individual Ablations (8 experiments)"
echo "  Started: $(date)"
echo "================================================================"

# Group A: Input Enrichment
run_experiment "A1_cross_attn" \
    --adapter_type cross_attn --use_dino768

run_experiment "A2_normals" \
    --adapter_type v3 --use_normals --hidden_dim 384 --num_layers 2

run_experiment "A3_gradients" \
    --adapter_type v3 --use_gradients --hidden_dim 384 --num_layers 2

# Group B: Architecture Enhancement
run_experiment "B1_film" \
    --adapter_type film --hidden_dim 384 --num_layers 2

run_experiment "B2_deep" \
    --adapter_type deep --hidden_dim 384

run_experiment "B3_window_attn" \
    --adapter_type window_attn --hidden_dim 384 --num_layers 2

# Group C: Loss Improvements
run_experiment "C1_contrastive" \
    --adapter_type v3 --hidden_dim 384 --num_layers 2 \
    --lambda_cluster 0.1 --centroids_path "$CENTROIDS"

run_experiment "C2_cross_image" \
    --adapter_type v3 --hidden_dim 384 --num_layers 2 \
    --cross_image_mining

echo ""
echo "================================================================"
echo "  Phase 1 complete. Starting DCFA-X..."
echo "================================================================"

# ─── Phase 2: DCFA-X Combined Architecture ───────────────────────────────────

run_experiment "X_combined" \
    --adapter_type x --use_dino768 --use_normals \
    --lambda_cluster 0.1 --centroids_path "$CENTROIDS"

echo ""
echo "================================================================"
echo "  DCFA v2 Full Pipeline COMPLETE"
echo "  Finished: $(date)"
echo "  Results: $BASE_OUT/"
echo "================================================================"
echo ""
echo "Next: Evaluate each checkpoint at k=80 with:"
echo "  python3 mbps_pytorch/generate_depth_overclustered_semantics.py \\"
echo "    --cityscapes_root $CS_ROOT --k 80 --variant sinusoidal --alpha 0.1 \\"
echo "    --adapter_checkpoint \$BASE_OUT/{name}/best.pt --skip_crf"
