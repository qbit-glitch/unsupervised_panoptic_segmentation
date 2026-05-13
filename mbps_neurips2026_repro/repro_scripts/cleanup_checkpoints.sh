#!/bin/bash
# Delete all checkpoints EXCEPT the best 3 (by pq_val) and last.ckpt.
#
# Run on A6000:
#   cd ~/umesh/unsupervised_panoptic_segmentation
#   bash scripts/cleanup_checkpoints.sh
#
# Add --dry-run to see what would be deleted without deleting:
#   bash scripts/cleanup_checkpoints.sh --dry-run
set -euo pipefail

EXP_DIR="$HOME/umesh/experiments"
DRY_RUN=false

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE (nothing will be deleted) ==="
    echo ""
fi

echo "=== Checkpoint Cleanup ==="
echo "Experiment dir: $EXP_DIR"
echo ""

# Find all experiment subdirectories
for exp in "$EXP_DIR"/cups_dinov3_*; do
    if [ ! -d "$exp" ]; then
        continue
    fi

    echo "--- Experiment: $(basename "$exp") ---"

    # Find all .ckpt files recursively
    CKPT_FILES=$(find "$exp" -name "*.ckpt" -type f 2>/dev/null | sort)
    TOTAL=$(echo "$CKPT_FILES" | grep -c . || true)

    if [ "$TOTAL" -eq 0 ]; then
        echo "  No checkpoints found."
        continue
    fi

    echo "  Total checkpoints: $TOTAL"

    # Separate into categories
    BEST_CKPTS=$(echo "$CKPT_FILES" | grep "best_pq_" || true)
    LAST_CKPT=$(echo "$CKPT_FILES" | grep "last.ckpt" || true)
    REGULAR_CKPTS=$(echo "$CKPT_FILES" | grep "ups_checkpoint_" || true)

    BEST_COUNT=$(echo "$BEST_CKPTS" | grep -c . || true)
    LAST_COUNT=$(echo "$LAST_CKPT" | grep -c . || true)
    REGULAR_COUNT=$(echo "$REGULAR_CKPTS" | grep -c . || true)

    echo "  Best PQ checkpoints: $BEST_COUNT (KEEP)"
    echo "  Last checkpoint: $LAST_COUNT (KEEP)"
    echo "  Regular checkpoints: $REGULAR_COUNT (DELETE)"

    # Show what we're keeping
    if [ -n "$BEST_CKPTS" ]; then
        echo "  Keeping:"
        echo "$BEST_CKPTS" | while read -r f; do
            SIZE=$(du -sh "$f" | cut -f1)
            echo "    $SIZE  $(basename "$f")"
        done
    fi
    if [ -n "$LAST_CKPT" ]; then
        echo "$LAST_CKPT" | while read -r f; do
            SIZE=$(du -sh "$f" | cut -f1)
            echo "    $SIZE  $(basename "$f") (last)"
        done
    fi

    # Calculate space to free
    if [ -n "$REGULAR_CKPTS" ]; then
        SPACE=$(echo "$REGULAR_CKPTS" | xargs du -ch 2>/dev/null | tail -1 | cut -f1)
        echo "  Space to free: $SPACE"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] Would delete $REGULAR_COUNT checkpoints ($SPACE)"
        else
            echo "  Deleting $REGULAR_COUNT regular checkpoints..."
            echo "$REGULAR_CKPTS" | xargs rm -f
            echo "  Done. Freed $SPACE"
        fi
    else
        echo "  Nothing to delete."
    fi

    echo ""
done

# Also check for wandb cache
WANDB_DIR="$EXP_DIR/wandb"
if [ -d "$WANDB_DIR" ]; then
    WANDB_SIZE=$(du -sh "$WANDB_DIR" 2>/dev/null | cut -f1)
    echo "--- W&B cache: $WANDB_SIZE at $WANDB_DIR ---"
    echo "  (delete manually if needed: rm -rf $WANDB_DIR)"
fi

echo ""
echo "=== Disk usage after cleanup ==="
df -h "$EXP_DIR" | head -2
echo ""
du -sh "$EXP_DIR"/* 2>/dev/null | sort -rh | head -10
