#!/usr/bin/env bash
# E2 Conv-DoRA setup + launch for Anydesk A6000
# Usage:
#   bash scripts/setup_e2_dora_a6000.sh check      # pre-flight checks
#   bash scripts/setup_e2_dora_a6000.sh download    # download tau=0.20 labels from HF
#   bash scripts/setup_e2_dora_a6000.sh patch       # apply empty_cache validation fix
#   bash scripts/setup_e2_dora_a6000.sh train       # launch Stage-2 training
#   bash scripts/setup_e2_dora_a6000.sh status      # check training progress

set -euo pipefail

PROJECT_ROOT="$HOME/umesh/unsupervised_panoptic_segmentation"
CUPS_ROOT="$PROJECT_ROOT/refs/cups"
DATASET_ROOT="$HOME/umesh/datasets/cityscapes"
PSEUDO_DIR="$DATASET_ROOT/cups_pseudo_labels_depthpro_tau020"
WEIGHTS="$PROJECT_ROOT/weights/dinov3_vitb16_official.pth"
CONFIG="configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_a6000.yaml"
LOG_DIR="$HOME/umesh/experiments/logs"
VENV="$HOME/umesh/ups_env/bin/activate"

mkdir -p "$LOG_DIR"

check() {
    echo "=== E2 Conv-DoRA A6000 Pre-flight ==="

    # 1. Python env
    echo "[1/6] Python environment:"
    if [ -f "$VENV" ]; then
        source "$VENV"
        echo "  venv: $VENV"
        python3 -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    else
        echo "  FAIL: venv not found at $VENV"
    fi

    # 2. GPU
    echo "[2/6] GPU:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    echo "  GPU processes: $procs"

    # 3. DINOv3 weights
    echo "[3/6] DINOv3 weights:"
    if [ -f "$WEIGHTS" ]; then
        echo "  FOUND: $WEIGHTS"
    else
        echo "  MISSING: $WEIGHTS"
        ls "$PROJECT_ROOT/weights/" 2>/dev/null || echo "  No weights dir"
    fi

    # 4. Pseudo-labels
    echo "[4/6] DepthPro tau=0.20 pseudo-labels:"
    if [ -d "$PSEUDO_DIR" ]; then
        n_files=$(ls "$PSEUDO_DIR" | wc -l)
        n_sem=$(ls "$PSEUDO_DIR"/*_semantic.png 2>/dev/null | wc -l)
        n_inst=$(ls "$PSEUDO_DIR"/*_instance.png 2>/dev/null | wc -l)
        n_pt=$(ls "$PSEUDO_DIR"/*.pt 2>/dev/null | wc -l)
        echo "  FOUND: $n_files files ($n_sem semantic, $n_inst instance, $n_pt .pt)"
        if [ "$n_files" -eq 8925 ]; then
            echo "  OK: All 8925 files present"
        else
            echo "  WARN: Expected 8925 files, got $n_files. Run 'download' step."
        fi
    else
        echo "  MISSING: Run 'bash scripts/setup_e2_dora_a6000.sh download' first"
    fi

    # 5. Config
    echo "[5/6] Config:"
    if [ -f "$CUPS_ROOT/$CONFIG" ]; then
        echo "  FOUND: $CUPS_ROOT/$CONFIG"
        echo "  BATCH_SIZE: $(grep BATCH_SIZE "$CUPS_ROOT/$CONFIG" | head -1 | awk '{print $2}')"
        echo "  ACCUMULATE: $(grep ACCUMULATE "$CUPS_ROOT/$CONFIG" | awk '{print $2}')"
        echo "  PRECISION: $(grep PRECISION "$CUPS_ROOT/$CONFIG" | awk '{print $2}')"
    else
        echo "  MISSING: $CUPS_ROOT/$CONFIG"
    fi

    # 6. Conv-DoRA code
    echo "[6/6] Conv-DoRA code:"
    if grep -q "ConvDoRALinear" "$CUPS_ROOT/cups/model/lora.py" 2>/dev/null; then
        echo "  FOUND: ConvDoRALinear in lora.py"
    else
        echo "  MISSING: Conv-DoRA not in lora.py — git pull needed"
    fi

    # Check empty_cache patch
    if grep -q "on_validation_epoch_start" "$CUPS_ROOT/cups/pl_model_pseudo.py" 2>/dev/null; then
        echo "  empty_cache patch: APPLIED"
    else
        echo "  empty_cache patch: NOT APPLIED — run 'patch' step"
    fi

    echo ""
    echo "=== Check complete ==="
}

download() {
    echo "=== Downloading DepthPro tau=0.20 pseudo-labels from HF Hub ==="
    source "$VENV"

    mkdir -p "$PSEUDO_DIR"

    python3 -c "
from huggingface_hub import snapshot_download
import os, shutil

repo_id = 'qbit-glitch/cityscapes-pseudo-labels'
target = '$PSEUDO_DIR'
tmp_dir = '/tmp/hf_pseudo_labels'

print(f'Downloading from {repo_id}...')
snapshot_download(
    repo_id=repo_id,
    repo_type='dataset',
    local_dir=tmp_dir,
    allow_patterns='cups_pseudo_labels_depthpro_tau020/*',
)

# Move files from subfolder to target
src = os.path.join(tmp_dir, 'cups_pseudo_labels_depthpro_tau020')
os.makedirs(target, exist_ok=True)
for f in os.listdir(src):
    shutil.move(os.path.join(src, f), os.path.join(target, f))

n_files = len([f for f in os.listdir(target) if not f.startswith('.')])
print(f'Done. {n_files} files in {target}')
"
    echo ""
    echo "Verifying download..."
    ls "$PSEUDO_DIR" | wc -l
}

patch() {
    echo "=== Applying empty_cache validation patch ==="
    source "$VENV"

    python3 -c "
filepath = '$CUPS_ROOT/cups/pl_model_pseudo.py'
with open(filepath, 'r') as f:
    content = f.read()

insert_before = '    def validation_step(self, batch'
insert_code = '''    def on_validation_epoch_start(self) -> None:
        \"\"\"Clear CUDA cache before validation to prevent OOM from memory fragmentation.\"\"\"
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

'''

if 'on_validation_epoch_start' not in content:
    # Find the exact line
    idx = content.find(insert_before)
    if idx == -1:
        print('ERROR: Could not find validation_step in pl_model_pseudo.py')
        exit(1)
    content = content[:idx] + insert_code + content[idx:]
    with open(filepath, 'w') as f:
        f.write(content)
    print('Patched: added on_validation_epoch_start with empty_cache')
else:
    print('Already patched')
"
}

train() {
    echo "=== Launching E2 Conv-DoRA Stage-2 on A6000 ==="
    source "$VENV"

    LOGFILE="$LOG_DIR/e2_depthpro_dora_a6000_$(date +%Y%m%d_%H%M%S).log"

    cd "$CUPS_ROOT"

    nohup python -u train.py \
        --experiment_config_file "$CONFIG" \
        --disable_wandb \
        > "$LOGFILE" 2>&1 &

    PID=$!
    echo "PID: $PID"
    echo "Log: $LOGFILE"
    echo "Config: bs=8 x accum=2 = effective 16, fp16-mixed"
    echo ""
    echo "Monitor with: tail -f $LOGFILE"
    echo "Kill with:    kill $PID"
}

status() {
    echo "=== E2 Conv-DoRA A6000 Status ==="
    pids=$(pgrep -f "train.*depthpro_e2_dora" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Training RUNNING (PIDs: $pids)"
    else
        echo "No training process found"
    fi

    LATEST=$(ls -t "$LOG_DIR"/e2_depthpro_dora_a6000_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "Latest log: $LATEST"
        echo "--- Last 20 lines ---"
        tail -20 "$LATEST"
        echo ""
        echo "--- Validation results ---"
        grep -E "^[0-9]+\.[0-9]+;" "$LATEST" | while IFS=';' read pq sq rq pqt sqt rqt pqs sqs rqs acc miou; do
            printf "PQ=%.2f%% PQ_things=%.2f%% PQ_stuff=%.2f%% mIoU=%.2f%%\n" \
                "$(echo "$pq * 100" | bc)" \
                "$(echo "$pqt * 100" | bc)" \
                "$(echo "$pqs * 100" | bc)" \
                "$(echo "$miou * 100" | bc)"
        done
    fi
}

case "${1:-help}" in
    check)    check ;;
    download) download ;;
    patch)    patch ;;
    train)    train ;;
    status)   status ;;
    *)
        echo "Usage: $0 {check|download|patch|train|status}"
        echo ""
        echo "  check     — verify GPU, weights, pseudo-labels, config"
        echo "  download  — download tau=0.20 labels from HF Hub"
        echo "  patch     — apply empty_cache validation fix"
        echo "  train     — launch Stage-2 training (bs=8, ~38GB VRAM)"
        echo "  status    — check training progress and metrics"
        ;;
esac
