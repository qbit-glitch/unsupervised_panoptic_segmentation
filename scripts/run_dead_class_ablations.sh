#!/bin/bash
# Master driver for dead-class clustering ablation (Stage-0 level).
# Baseline: ViT-L/16 + spherical k-means k=100 (PQ=20.88, mIoU=39.14)
# 7 dead classes: wall, fence, traffic light, rider, truck, train, motorcycle
#
# Usage:
#   bash scripts/run_dead_class_ablations.sh [ablation_number]
#   bash scripts/run_dead_class_ablations.sh 5   # run only PCL+Sinkhorn
#   bash scripts/run_dead_class_ablations.sh all  # run all sequentially

set -euo pipefail

CS_ROOT="/Users/qbit-glitch/Desktop/datasets/cityscapes"
PROJECT_ROOT="/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation"
RESULTS_DIR="${PROJECT_ROOT}/results/clustering_ablation"
FEAT_SUBDIR="dinov3_features_vitl16"
K=100
SEED=42

mkdir -p "${RESULTS_DIR}"

run_eval() {
    local pred_dir="$1"
    local output_json="$2"
    echo "  Evaluating: ${pred_dir}"
    python3 "${PROJECT_ROOT}/scripts/evaluate_pseudolabel_quality.py" \
        --pred_dir "${pred_dir}/val" \
        --gt_dir "${CS_ROOT}/gtFine/val" \
        --split val \
        --output "${output_json}" \
        --num_classes 19
    echo "  Results saved to: ${output_json}"
}

ablation_5_pcl_sinkhorn() {
    echo "=== Ablation 5: PCL + Balanced Sinkhorn ==="
    local OUT_DIR="${CS_ROOT}/pseudo_semantic_raw_dinov3_k${K}_pcl_sinkhorn_vitl16"
    python3 "${PROJECT_ROOT}/mbps_pytorch/generate_pcl_sinkhorn_ablation.py" \
        --cityscapes_root "${CS_ROOT}" \
        --feat_subdir "${FEAT_SUBDIR}" \
        --k ${K} --seed ${SEED} \
        --em_iterations 10 \
        --contrastive_temp 0.2 \
        --contrastive_lr 1e-3 \
        --contrastive_epochs 3 \
        --sinkhorn_temp 0.1 \
        --sinkhorn_iters 5 \
        --device auto
    run_eval "${OUT_DIR}" "${RESULTS_DIR}/eval_pcl_sinkhorn_vitl16_k${K}.json"
}

ablation_2_recursive_dsc() {
    echo "=== Ablation 2: Recursive Deep Spectral Clustering ==="
    local OUT_DIR="${CS_ROOT}/pseudo_semantic_raw_dinov3_k${K}_recursive_dsc_vitl16"
    python3 "${PROJECT_ROOT}/mbps_pytorch/generate_recursive_dsc_ablation.py" \
        --cityscapes_root "${CS_ROOT}" \
        --feat_subdir "${FEAT_SUBDIR}" \
        --k ${K} --seed ${SEED}
    run_eval "${OUT_DIR}" "${RESULTS_DIR}/eval_recursive_dsc_vitl16_k${K}.json"
}

ablation_1a_shift_avg() {
    echo "=== Ablation 1a: Shift-Average Feature Upsampling ==="
    echo "  Step 1: Extract shift-average features (train)..."
    python3 "${PROJECT_ROOT}/mbps_pytorch/extract_shift_avg_features.py" \
        --cityscapes_root "${CS_ROOT}" \
        --split train --device auto
    echo "  Step 2: Extract shift-average features (val)..."
    python3 "${PROJECT_ROOT}/mbps_pytorch/extract_shift_avg_features.py" \
        --cityscapes_root "${CS_ROOT}" \
        --split val --device auto
    echo "  Step 3: Cluster shift-avg features with spherical k-means..."
    python3 "${PROJECT_ROOT}/mbps_pytorch/generate_clustering_ablation.py" \
        --cityscapes_root "${CS_ROOT}" \
        --feat_subdir dinov3_features_shiftavg_vitl16 \
        --method spherical_kmeans \
        --k ${K} --seed ${SEED} \
        --output_suffix vitl16_shiftavg
    local OUT_DIR="${CS_ROOT}/pseudo_semantic_raw_dinov3_k${K}_spherical_kmeans_vitl16_shiftavg"
    run_eval "${OUT_DIR}" "${RESULTS_DIR}/eval_shift_avg_vitl16_k${K}.json"
}

ablation_4_ppap() {
    echo "=== Ablation 4: PPAP (Progressive Proxy Anchor Propagation) ==="
    local OUT_DIR="${CS_ROOT}/pseudo_semantic_raw_dinov3_k${K}_ppap_vitl16"
    python3 "${PROJECT_ROOT}/mbps_pytorch/generate_ppap_ablation.py" \
        --cityscapes_root "${CS_ROOT}" \
        --feat_subdir "${FEAT_SUBDIR}" \
        --k ${K} --seed ${SEED} \
        --epochs 20 \
        --lr_proxy 1e-3 \
        --temperature 0.1 \
        --device auto
    run_eval "${OUT_DIR}" "${RESULTS_DIR}/eval_ppap_vitl16_k${K}.json"
}

ablation_3_diffcut() {
    echo "=== Ablation 3: DiffCut (Diffusion features + NCut) ==="
    local OUT_DIR="${CS_ROOT}/pseudo_semantic_raw_dinov3_k${K}_diffcut_vitl16"
    python3 "${PROJECT_ROOT}/mbps_pytorch/generate_diffcut_cityscapes_ablation.py" \
        --cityscapes_root "${CS_ROOT}" \
        --k ${K} --seed ${SEED} \
        --device auto
    run_eval "${OUT_DIR}" "${RESULTS_DIR}/eval_diffcut_vitl16_k${K}.json"
}

ablation_6_sdcluster() {
    echo "=== Ablation 6: SDCluster (Prototype constraint + semantic consistency) ==="
    local OUT_DIR="${CS_ROOT}/pseudo_semantic_raw_dinov3_k${K}_sdcluster_vitl16"
    python3 "${PROJECT_ROOT}/mbps_pytorch/generate_sdcluster_ablation.py" \
        --cityscapes_root "${CS_ROOT}" \
        --feat_subdir "${FEAT_SUBDIR}" \
        --k ${K} --seed ${SEED} \
        --epochs 30 \
        --dead_threshold 0.005 \
        --reinit_strategy furthest_point \
        --device auto
    run_eval "${OUT_DIR}" "${RESULTS_DIR}/eval_sdcluster_vitl16_k${K}.json"
}

TARGET="${1:-all}"

case "${TARGET}" in
    5)   ablation_5_pcl_sinkhorn ;;
    2)   ablation_2_recursive_dsc ;;
    1|1a) ablation_1a_shift_avg ;;
    4)   ablation_4_ppap ;;
    3)   ablation_3_diffcut ;;
    6)   ablation_6_sdcluster ;;
    all)
        ablation_5_pcl_sinkhorn
        ablation_2_recursive_dsc
        ablation_1a_shift_avg
        ablation_4_ppap
        ablation_6_sdcluster
        ablation_3_diffcut
        echo "=== ALL ABLATIONS COMPLETE ==="
        ;;
    *)
        echo "Usage: $0 {1|1a|2|3|4|5|6|all}"
        exit 1
        ;;
esac
