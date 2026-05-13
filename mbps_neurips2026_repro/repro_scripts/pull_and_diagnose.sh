#!/bin/bash
# Pull latest main and run DINOv3 weight diagnostic
cd ~/umesh/unsupervised_panoptic_segmentation
git fetch origin main
git merge origin/main --no-edit
python scripts/diagnose_dinov3_weights.py
