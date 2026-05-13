#!/bin/bash
# One-time GCS bucket setup for MBPS project.
#
# Usage:
#   bash scripts/setup_gcs_bucket.sh
#
# Prerequisites:
#   - gcloud CLI authenticated (gcloud auth login)
#   - Project set: gcloud config set project unsupervised-panoptic-segment

set -euo pipefail

PROJECT="unsupervised-panoptic-segment"
BUCKET="mbps-panoptic"
LOCATION="us-central2"

echo "=== MBPS GCS Bucket Setup ==="
echo "Project:  ${PROJECT}"
echo "Bucket:   gs://${BUCKET}"
echo "Location: ${LOCATION}"
echo ""

# 1. Create bucket (nearline for cost savings, same region as TPUs)
if gsutil ls -b "gs://${BUCKET}" 2>/dev/null; then
    echo "Bucket gs://${BUCKET} already exists, skipping creation."
else
    echo "Creating bucket gs://${BUCKET} in ${LOCATION}..."
    gsutil mb -p "${PROJECT}" -l "${LOCATION}" -c standard "gs://${BUCKET}"
fi

# 2. Create directory structure
echo "Creating directory structure..."
for dir in \
    datasets/cityscapes/tfrecords/train \
    datasets/cityscapes/tfrecords/val \
    datasets/cityscapes/depth_zoedepth \
    datasets/coco/tfrecords/train \
    datasets/coco/tfrecords/val \
    datasets/coco/depth_zoedepth \
    checkpoints \
    results \
    weights \
    logs; do
    # Touch a placeholder to create the "directory"
    echo "" | gsutil -q cp - "gs://${BUCKET}/${dir}/.keep"
done
echo "Directory structure created."

# 3. Set lifecycle policy: auto-delete checkpoints older than 30 days
echo "Setting 30-day lifecycle on checkpoints/..."
cat > /tmp/mbps_lifecycle.json << 'LIFECYCLE_EOF'
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 30,
        "matchesPrefix": ["checkpoints/"]
      }
    }
  ]
}
LIFECYCLE_EOF
gsutil lifecycle set /tmp/mbps_lifecycle.json "gs://${BUCKET}"
rm /tmp/mbps_lifecycle.json
echo "Lifecycle policy set."

# 4. Grant storage access to compute service account
echo "Granting storage access to TPU VMs..."
SA=$(gcloud iam service-accounts list \
    --project="${PROJECT}" \
    --filter="displayName:Compute Engine default" \
    --format="value(email)" 2>/dev/null || true)

if [ -n "${SA}" ]; then
    gsutil iam ch "serviceAccount:${SA}:objectAdmin" "gs://${BUCKET}"
    echo "Granted objectAdmin to ${SA}"
else
    echo "WARNING: Could not find default compute service account."
    echo "You may need to manually grant access:"
    echo "  gsutil iam ch serviceAccount:<SA_EMAIL>:objectAdmin gs://${BUCKET}"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload DINO weights:"
echo "     gsutil cp dino_vits8_flax.npz gs://${BUCKET}/weights/"
echo ""
echo "  2. Upload TFRecords (from a VM with data):"
echo "     gsutil -m rsync -r /data/cityscapes/tfrecords/ gs://${BUCKET}/datasets/cityscapes/tfrecords/"
echo "     gsutil -m rsync -r /data/coco/tfrecords/ gs://${BUCKET}/datasets/coco/tfrecords/"
echo ""
echo "  3. Verify:"
echo "     gsutil ls gs://${BUCKET}/datasets/"
echo ""
echo "Estimated monthly cost: ~\$1.70 (85GB standard storage)"
