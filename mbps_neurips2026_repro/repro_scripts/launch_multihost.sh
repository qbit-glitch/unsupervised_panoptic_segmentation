#!/bin/bash
# Launch multi-host distributed training across 4 TPU VMs.
#
# All VMs cooperate on a single training run using JAX multi-host pmap.
# Gradients are averaged across all 16 TPU chips (4 VMs x 4 chips).
# Only process 0 (coordinator) saves checkpoints and logs to W&B.
#
# Usage:
#   bash scripts/launch_multihost.sh                        # defaults
#   bash scripts/launch_multihost.sh cityscapes_gcs.yaml    # custom config
#   bash scripts/launch_multihost.sh cityscapes_gcs.yaml cityscapes_full_multihost 42
#
# Prerequisites:
#   - All 4 VMs are READY and have code + deps installed
#   - All VMs are in the same zone/VPC (us-central2-b)

set -euo pipefail

CONFIG="${1:-cityscapes_gcs.yaml}"
EXPERIMENT="${2:-cityscapes_full_multihost}"
SEED="${3:-42}"
RESUME="${4:-}"

PROJECT_ID="unsupervised-panoptic-segment"
ZONE="us-central2-b"
COORD_PORT="1234"

# VM names (coordinator = first VM)
VMS=("panoptic-tpu-mbps" "panoptic-tpu-v4" "panoptic-tpu-depthg" "panoptic-tpu-cuts3d")
NUM_PROCESSES=${#VMS[@]}
COORDINATOR_VM="${VMS[0]}"

echo "=========================================="
echo " MBPS Multi-Host Training Launch"
echo "=========================================="
echo "  Config:      ${CONFIG}"
echo "  Experiment:  ${EXPERIMENT}"
echo "  Seed:        ${SEED}"
echo "  VMs:         ${VMS[*]}"
echo "  Processes:   ${NUM_PROCESSES}"
echo "  Coordinator: ${COORDINATOR_VM}"
if [ -n "${RESUME}" ]; then
    echo "  Resume from: ${RESUME}"
fi
echo ""

# Get coordinator internal IP
echo "Getting coordinator internal IP..."
COORDINATOR_IP=$(gcloud compute tpus tpu-vm describe "${COORDINATOR_VM}" \
    --zone="${ZONE}" --project="${PROJECT_ID}" \
    --format="value(networkEndpoints[0].ipAddress)")

if [ -z "${COORDINATOR_IP}" ]; then
    echo "ERROR: Could not get internal IP for ${COORDINATOR_VM}"
    exit 1
fi
echo "  Coordinator IP: ${COORDINATOR_IP}:${COORD_PORT}"
echo ""

# Build resume flag
RESUME_FLAG=""
if [ -n "${RESUME}" ]; then
    RESUME_FLAG="--resume ${RESUME}"
fi

# Launch on all VMs in parallel
for i in "${!VMS[@]}"; do
    VM="${VMS[$i]}"
    echo "Launching process ${i} on ${VM}..."

    CMD="cd ~/mbps_panoptic_segmentation && mkdir -p logs && "
    CMD+="nohup python3 scripts/train.py "
    CMD+="--config configs/${CONFIG} "
    CMD+="--seed ${SEED} "
    CMD+="--vm_name ${VM} "
    CMD+="--experiment ${EXPERIMENT} "
    CMD+="--coordinator_address ${COORDINATOR_IP}:${COORD_PORT} "
    CMD+="--num_processes ${NUM_PROCESSES} "
    CMD+="--process_id ${i} "
    CMD+="${RESUME_FLAG} "
    CMD+="> logs/${EXPERIMENT}_p${i}.log 2>&1 &"

    gcloud compute tpus tpu-vm ssh "${VM}" \
        --zone="${ZONE}" --project="${PROJECT_ID}" \
        --command="${CMD}" \
        --strict-host-key-checking=no &
done

# Wait for all SSH commands to finish
wait

echo ""
echo "=========================================="
echo " All ${NUM_PROCESSES} processes launched!"
echo "=========================================="
echo ""
echo "Monitor progress:"
for i in "${!VMS[@]}"; do
    echo "  gcloud compute tpus tpu-vm ssh ${VMS[$i]} --zone=${ZONE} --project=${PROJECT_ID} \\"
    echo "    --command='tail -50 ~/mbps_panoptic_segmentation/logs/${EXPERIMENT}_p${i}.log'"
done
echo ""
echo "Check checkpoints (only process 0 saves):"
echo "  gsutil ls gs://mbps-panoptic/checkpoints/${EXPERIMENT}/${COORDINATOR_VM}/"
echo ""
echo "Live tail coordinator log:"
echo "  gcloud compute tpus tpu-vm ssh ${COORDINATOR_VM} --zone=${ZONE} --project=${PROJECT_ID} \\"
echo "    --command='tail -f ~/mbps_panoptic_segmentation/logs/${EXPERIMENT}_p0.log'"
