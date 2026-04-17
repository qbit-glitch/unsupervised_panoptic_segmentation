#!/bin/bash
# Investigate what was wrong with the A6000 pseudo-labels.
#
# Run on A6000:
#   cd ~/umesh/unsupervised_panoptic_segmentation
#   bash scripts/investigate_a6000_pseudo_labels.sh
#
# This compares:
#   OLD (wrong): cups_pseudo_labels_depthpro_tau020_old_wrong_split/
#   NEW (fixed): cups_pseudo_labels_depthpro_tau020/
#
# The old dir was renamed by download_pseudolabels_a6000.sh when we swapped in
# santosh's labels. If you already deleted it, Steps 2-5 will be skipped.
set -euo pipefail

CS_ROOT="$HOME/umesh/datasets/cityscapes"
OLD_DIR="$CS_ROOT/cups_pseudo_labels_depthpro_tau020_old_wrong_split"
NEW_DIR="$CS_ROOT/cups_pseudo_labels_depthpro_tau020"
PROJECT="$HOME/umesh/unsupervised_panoptic_segmentation"

echo "=============================================="
echo "  A6000 Pseudo-Label Investigation"
echo "=============================================="
echo ""

# -------------------------------------------------------------------
# Step 1: Verify the NEW (fixed) labels are in place and correct
# -------------------------------------------------------------------
echo "=== Step 1: Verify NEW (fixed) labels ==="
if [ -d "$NEW_DIR" ]; then
    INST=$(ls "$NEW_DIR"/*_instance.png 2>/dev/null | wc -l)
    SEM=$(ls "$NEW_DIR"/*_semantic.png 2>/dev/null | wc -l)
    PT=$(ls "$NEW_DIR"/*.pt 2>/dev/null | wc -l)
    echo "NEW dir: $INST instance, $SEM semantic, $PT .pt files"
    echo "Expected: 2975 each"

    # Check for val contamination
    VAL_COUNT=$(ls "$NEW_DIR" | grep -cE '^(frankfurt|lindau|munster)_' || true)
    echo "Val contamination: $VAL_COUNT (should be 0)"

    # File size fingerprint
    FIRST="$NEW_DIR/aachen_000000_000019_leftImg8bit_instance.png"
    if [ -f "$FIRST" ]; then
        SIZE=$(stat -c%s "$FIRST" 2>/dev/null || stat -f%z "$FIRST")
        echo "First file size: $SIZE bytes (santosh reference: 6120)"
    fi
else
    echo "ERROR: NEW dir does not exist: $NEW_DIR"
    echo "Run scripts/download_pseudolabels_a6000.sh first!"
    exit 1
fi

echo ""

# -------------------------------------------------------------------
# Step 2: Check if OLD (wrong) labels still exist
# -------------------------------------------------------------------
echo "=== Step 2: Check OLD (wrong) labels ==="
if [ -d "$OLD_DIR" ]; then
    OLD_INST=$(ls "$OLD_DIR"/*_instance.png 2>/dev/null | wc -l)
    OLD_SEM=$(ls "$OLD_DIR"/*_semantic.png 2>/dev/null | wc -l)
    OLD_PT=$(ls "$OLD_DIR"/*.pt 2>/dev/null | wc -l)
    echo "OLD dir: $OLD_INST instance, $OLD_SEM semantic, $OLD_PT .pt files"
    echo "(Can compare with NEW dir)"
    HAS_OLD=true
else
    echo "OLD dir not found (already deleted). Skipping comparison steps."
    echo "Only Steps 1 and 6 (diagnose NEW labels) will run."
    HAS_OLD=false
fi

echo ""

# -------------------------------------------------------------------
# Step 3: Compare file counts
# -------------------------------------------------------------------
if [ "$HAS_OLD" = true ]; then
    echo "=== Step 3: File count comparison ==="
    echo "OLD: $OLD_INST instance, $OLD_SEM semantic, $OLD_PT .pt"
    echo "NEW: $INST instance, $SEM semantic, $PT .pt"
    if [ "$OLD_INST" != "$INST" ]; then
        echo "MISMATCH: Different number of instance files!"
        echo "  Missing in OLD: $(diff <(ls "$NEW_DIR"/*_instance.png | xargs -n1 basename | sort) <(ls "$OLD_DIR"/*_instance.png | xargs -n1 basename | sort) | grep "^< " | wc -l)"
        echo "  Extra in OLD:   $(diff <(ls "$NEW_DIR"/*_instance.png | xargs -n1 basename | sort) <(ls "$OLD_DIR"/*_instance.png | xargs -n1 basename | sort) | grep "^> " | wc -l)"
    else
        echo "OK: Same number of files."
    fi
    echo ""

    # -------------------------------------------------------------------
    # Step 4: Compare file sizes (are the images identical?)
    # -------------------------------------------------------------------
    echo "=== Step 4: File content comparison (first 10 instance files) ==="
    DIFF_COUNT=0
    SAME_COUNT=0
    for f in $(ls "$NEW_DIR"/*_instance.png | head -10 | xargs -n1 basename); do
        NEW_SIZE=$(stat -c%s "$NEW_DIR/$f" 2>/dev/null || stat -f%z "$NEW_DIR/$f")
        if [ -f "$OLD_DIR/$f" ]; then
            OLD_SIZE=$(stat -c%s "$OLD_DIR/$f" 2>/dev/null || stat -f%z "$OLD_DIR/$f")
            if [ "$NEW_SIZE" = "$OLD_SIZE" ]; then
                # Same size — check if content is identical
                if cmp -s "$NEW_DIR/$f" "$OLD_DIR/$f"; then
                    SAME_COUNT=$((SAME_COUNT + 1))
                else
                    echo "  CONTENT DIFFERS (same size): $f ($NEW_SIZE bytes)"
                    DIFF_COUNT=$((DIFF_COUNT + 1))
                fi
            else
                echo "  SIZE DIFFERS: $f (new=$NEW_SIZE, old=$OLD_SIZE)"
                DIFF_COUNT=$((DIFF_COUNT + 1))
            fi
        else
            echo "  MISSING in OLD: $f"
            DIFF_COUNT=$((DIFF_COUNT + 1))
        fi
    done
    echo "Result: $SAME_COUNT identical, $DIFF_COUNT different (out of 10)"
    echo ""

    # Also check semantic files
    echo "=== Step 4b: Semantic file comparison (first 10) ==="
    SEM_DIFF=0
    SEM_SAME=0
    for f in $(ls "$NEW_DIR"/*_semantic.png | head -10 | xargs -n1 basename); do
        if [ -f "$OLD_DIR/$f" ]; then
            if cmp -s "$NEW_DIR/$f" "$OLD_DIR/$f"; then
                SEM_SAME=$((SEM_SAME + 1))
            else
                echo "  SEMANTIC DIFFERS: $f"
                SEM_DIFF=$((SEM_DIFF + 1))
            fi
        else
            echo "  MISSING in OLD: $f"
            SEM_DIFF=$((SEM_DIFF + 1))
        fi
    done
    echo "Semantic: $SEM_SAME identical, $SEM_DIFF different (out of 10)"
    echo ""

    # And .pt files
    echo "=== Step 4c: .pt file comparison (first 10) ==="
    PT_DIFF=0
    PT_SAME=0
    for f in $(ls "$NEW_DIR"/*.pt | head -10 | xargs -n1 basename); do
        if [ -f "$OLD_DIR/$f" ]; then
            if cmp -s "$NEW_DIR/$f" "$OLD_DIR/$f"; then
                PT_SAME=$((PT_SAME + 1))
            else
                echo "  .PT DIFFERS: $f"
                PT_DIFF=$((PT_DIFF + 1))
            fi
        else
            echo "  MISSING in OLD: $f"
            PT_DIFF=$((PT_DIFF + 1))
        fi
    done
    echo ".pt files: $PT_SAME identical, $PT_DIFF different (out of 10)"
    echo ""
fi

# -------------------------------------------------------------------
# Step 5: Run full diagnostic on BOTH directories (Python)
# -------------------------------------------------------------------
echo "=== Step 5: Full Python diagnostics ==="
cd "$PROJECT"

echo "--- Running on NEW (fixed) labels ---"
python scripts/diagnose_pseudo_labels.py \
    --pseudo_dir "$NEW_DIR" \
    --cityscapes_root "$CS_ROOT" \
    --output /tmp/a6000_new_diag.json

if [ "$HAS_OLD" = true ]; then
    echo ""
    echo "--- Running on OLD (wrong) labels ---"
    python scripts/diagnose_pseudo_labels.py \
        --pseudo_dir "$OLD_DIR" \
        --cityscapes_root "$CS_ROOT" \
        --output /tmp/a6000_old_diag.json
fi

echo ""

# -------------------------------------------------------------------
# Step 6: Compare thing/stuff splits side by side
# -------------------------------------------------------------------
echo "=== Step 6: Thing/stuff split comparison ==="
python3 -c "
import json

# Load NEW diagnostics
with open('/tmp/a6000_new_diag.json') as f:
    new = json.load(f)

print('=== NEW (fixed, from santosh) ===')
ts = new.get('thing_stuff_split', {})
print(f'  Things ({ts.get(\"num_things\", \"?\")}): {ts.get(\"things_classes\", [])}')
print(f'  Stuff  ({ts.get(\"num_stuff\", \"?\")}): {len(ts.get(\"stuff_classes\", []))} classes')
print(f'  Files:  {new[\"file_counts\"]}')

import os
if os.path.exists('/tmp/a6000_old_diag.json'):
    with open('/tmp/a6000_old_diag.json') as f:
        old = json.load(f)

    print()
    print('=== OLD (wrong, locally generated) ===')
    ts_old = old.get('thing_stuff_split', {})
    print(f'  Things ({ts_old.get(\"num_things\", \"?\")}): {ts_old.get(\"things_classes\", [])}')
    print(f'  Stuff  ({ts_old.get(\"num_stuff\", \"?\")}): {len(ts_old.get(\"stuff_classes\", []))} classes')
    print(f'  Files:  {old[\"file_counts\"]}')

    # Compute overlap
    new_things = set(ts.get('things_classes', []))
    old_things = set(ts_old.get('things_classes', []))
    overlap = new_things & old_things
    only_new = new_things - old_things
    only_old = old_things - new_things

    print()
    print('=== COMPARISON ===')
    print(f'  Thing class overlap: {len(overlap)} / {len(new_things)} (santosh has {len(new_things)}, A6000 had {len(old_things)})')
    print(f'  Overlap classes:     {sorted(overlap)}')
    print(f'  Only in santosh:     {sorted(only_new)}')
    print(f'  Only in old A6000:   {sorted(only_old)}')
    print()

    # Instance stats
    new_inst = new.get('instance_stats', {})
    old_inst = old.get('instance_stats', {})
    print(f'  Instance avg/img: NEW={new_inst.get(\"avg_instances_per_image\", \"?\")}, OLD={old_inst.get(\"avg_instances_per_image\", \"?\")}')
    print(f'  Empty maps:       NEW={new_inst.get(\"empty_pct\", \"?\")}%, OLD={old_inst.get(\"empty_pct\", \"?\")}%')

    # Semantic distribution
    new_sem = new.get('semantic_distribution', {})
    old_sem = old.get('semantic_distribution', {})
    print(f'  Unique sem classes: NEW={new_sem.get(\"num_unique_classes\", \"?\")}, OLD={old_sem.get(\"num_unique_classes\", \"?\")}')
else:
    print()
    print('OLD diagnostics not available (old labels deleted).')

# Compare with santosh reference
print()
print('=== vs SANTOSH REFERENCE ===')
SANTOSH_THINGS = [3, 11, 14, 15, 16, 29, 32, 37, 38, 45, 46, 62, 65, 73, 75]
new_things = ts.get('things_classes', [])
if new_things == SANTOSH_THINGS:
    print('  NEW labels MATCH santosh exactly. GOOD.')
else:
    print(f'  WARNING: NEW labels do NOT match santosh!')
    print(f'  Santosh: {SANTOSH_THINGS}')
    print(f'  NEW:     {new_things}')
"

echo ""
echo "=============================================="
echo "  Investigation complete."
echo "  Diagnostic JSONs saved to /tmp/"
echo "    /tmp/a6000_new_diag.json"
if [ "$HAS_OLD" = true ]; then
    echo "    /tmp/a6000_old_diag.json"
fi
echo "=============================================="
