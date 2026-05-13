#!/bin/bash
# MMGD-Cut Cityscapes panoptic evaluation
# Runs 4 configs: baseline, R6 (multiscale), R4 (NAMR), R6+R4

PYTHON=/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python
SCRIPT=$(dirname "$(dirname "$0")")/mbps_pytorch/mmgd_cut_cityscapes.py
CS_ROOT=/Users/qbit-glitch/Desktop/datasets/cityscapes
LOG=$(dirname "$(dirname "$0")")/logs/mmgd_cityscapes.log
OUT_JSON=$CS_ROOT/mmgd_cityscapes_results.json

cd "$(dirname "$(dirname "$0")")"

echo "=== MMGD-Cut Cityscapes Panoptic Evaluation ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG

# --- Config 1: Baseline (K=80, no post-processing) ---
echo "" | tee -a $LOG
echo "=== C1: Baseline (K=80, no post-proc) ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
$PYTHON $SCRIPT \
    --cs_root $CS_ROOT \
    --device mps \
    --K 80 \
    --out_json $OUT_JSON \
    2>&1 | tee -a $LOG
echo "Done C1: $(date)" | tee -a $LOG

# --- Config 2: R6 Multiscale ---
echo "" | tee -a $LOG
echo "=== C2: R6 Multiscale (coarse 32x64 + fine 64x128) ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
$PYTHON $SCRIPT \
    --cs_root $CS_ROOT \
    --device mps \
    --K 80 \
    --multiscale \
    --out_json $OUT_JSON \
    2>&1 | tee -a $LOG
echo "Done C2: $(date)" | tee -a $LOG

# --- Config 3: R4 NAMR ---
echo "" | tee -a $LOG
echo "=== C3: R4 NAMR post-processing ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
$PYTHON $SCRIPT \
    --cs_root $CS_ROOT \
    --device mps \
    --K 80 \
    --namr \
    --out_json $OUT_JSON \
    2>&1 | tee -a $LOG
echo "Done C3: $(date)" | tee -a $LOG

# --- Config 4: R6 + R4 combined ---
echo "" | tee -a $LOG
echo "=== C4: R6+R4 combined ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
$PYTHON $SCRIPT \
    --cs_root $CS_ROOT \
    --device mps \
    --K 80 \
    --multiscale \
    --namr \
    --out_json $OUT_JSON \
    2>&1 | tee -a $LOG
echo "Done C4: $(date)" | tee -a $LOG

echo "" | tee -a $LOG
echo "=== All configs complete ===" | tee -a $LOG
echo "Results: $OUT_JSON" | tee -a $LOG
