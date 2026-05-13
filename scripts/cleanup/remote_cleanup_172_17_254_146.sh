#!/usr/bin/env bash
# Remote cleanup for santosh@172.17.254.146 (1.7T/1.8T used, 31G free).
# Safety: every entry is gated by an ENABLE flag. Default = all OFF.
# Each enabled entry prints `du -sh` before deleting so you can sanity-check the log.
# Run:
#   scp this script to remote, then:  bash remote_cleanup_172_17_254_146.sh
# Or pipe over ssh:
#   ssh santosh@172.17.254.146 'bash -s' < remote_cleanup_172_17_254_146.sh
set -u  # NOT -e: keep going if a path is already gone.

# ============================================================
# TIER FLAGS — set to 1 to enable. ALL OFF BY DEFAULT.
# ============================================================
TIER_A_SUPERSEDED_RUNS=0   # ~222 GB. Old Stage-2/Stage-3 + pip/torch caches
TIER_B_CITYSCAPES=0        # ~38 GB. rightImg8bit + cups_pseudo_labels_pipeline
TIER_C_PRUNE_CKPTS=0       # variable. Prune intermediate ckpts in current best runs (LISTS only, no delete)
TIER_D_HF_HUB=0            # ~170 GB. Unrelated LLM models in HF cache. CONFIRM with co-users.
DRY_RUN=1                  # If 1, only echo what would be removed. Set to 0 to actually rm.
# ============================================================

note() { printf '\n=== %s ===\n' "$*"; }
size() { du -sh -- "$1" 2>/dev/null || echo "(missing) $1"; }
zap()  {
  local p="$1"
  if [[ ! -e "$p" ]]; then echo "  (skip, not present) $p"; return; fi
  echo "  $(du -sh -- "$p" 2>/dev/null)"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [DRY] would: rm -rf -- $p"
  else
    rm -rf -- "$p"
    echo "  [DONE] removed $p"
  fi
}

note "BEFORE"
df -h /home / 2>/dev/null | head -5

# ---------- TIER A ----------
if [[ "$TIER_A_SUPERSEDED_RUNS" == "1" ]]; then
  note "TIER A — superseded experiment runs + regenerable caches"
  zap "$HOME/cups/experiments/experiments/e1_cups_official_dinov3_vitb_8k_stage2_gpu0"
  zap "$HOME/cups/experiments/experiments/e2_clean_conv_dora_r4"
  zap "$HOME/cups/experiments/experiments/e2_depthpro_conv_dora_r4_1gpu"
  zap "$HOME/cups/experiments/experiments/e2_depthpro_conv_dora_r4_2gpu"
  zap "$HOME/cups/experiments/experiments/e2_dinov3_vitb_k80_conv_dora_r4"
  zap "$HOME/cups/experiments/experiments/cups_stage3_local"
  zap "$HOME/cups/experiments/experiments/cups_dinov3_vitb_k80_local_test"
  zap "$HOME/experiments/stage3_ablation_a1_a2"
  zap "$HOME/experiments/stage2_dcfa_da3_simcf_abc"
  zap "$HOME/.cache/pip"
  zap "$HOME/.cache/torch"
fi

# ---------- TIER B ----------
if [[ "$TIER_B_CITYSCAPES" == "1" ]]; then
  note "TIER B — Cityscapes data not used by monocular training"
  zap "$HOME/datasets/cityscapes/rightImg8bit"
  zap "$HOME/datasets/cityscapes/cups_pseudo_labels_pipeline"
  # The next two are tiny but redundant; uncomment if you want them gone:
  # zap "$HOME/datasets/cityscapes/cups_pseudo_labels"
  # zap "$HOME/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020"
fi

# ---------- TIER C ----------
if [[ "$TIER_C_PRUNE_CKPTS" == "1" ]]; then
  note "TIER C — list intermediate checkpoints in current best runs (NO DELETE)"
  for d in \
    "$HOME/cups/experiments/experiments/dinov3_vitb_seed44_stage2" \
    "$HOME/cups/experiments/experiments/dcfa_simcf_abc_seed44_stage2" \
    "$HOME/experiments/stage2_dcfa_simcf_abc" \
    "$HOME/experiments/stage2_dcfa_simcf_abc_dora_r32" \
    "$HOME/experiments/stage3_dcfa_simcf_abc" \
    "$HOME/experiments/stage3_dcfa_simcf_abc_dora_r32"
  do
    echo
    echo "--- $d ---"
    [[ -d "$d" ]] || { echo "(missing)"; continue; }
    find "$d" -maxdepth 4 -type f \( -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) \
      -printf '%s\t%p\n' 2>/dev/null | sort -nr | head -30 \
      | awk '{ printf "  %8.2f GB  %s\n", $1/1024/1024/1024, $2 }'
  done
  echo
  echo "Tier C is INSPECTION-ONLY. Tell me which checkpoints to keep and I'll add explicit zap lines."
fi

# ---------- TIER D ----------
if [[ "$TIER_D_HF_HUB" == "1" ]]; then
  note "TIER D — HuggingFace hub models unrelated to MBPS (CONFIRM CO-USERS FIRST)"
  zap "$HOME/.cache/huggingface/hub/models--google--flan-t5-xxl"
  zap "$HOME/.cache/huggingface/hub/models--EleutherAI--gpt-neox-20b"
  zap "$HOME/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-13b"
  zap "$HOME/.cache/huggingface/hub/models--EleutherAI--gpt-j-6B"
  zap "$HOME/.cache/huggingface/hub/models--Salesforce--blip2-opt-2.7b"
  zap "$HOME/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b"
  zap "$HOME/.cache/huggingface/hub/models--microsoft--Phi-3-vision-128k-instruct"
  zap "$HOME/.cache/huggingface/hub/models--microsoft--Phi-3.5-vision-instruct"
  # Models likely USED somewhere in MBPS — DO NOT enable in Tier D:
  #   models--apple--DepthPro-hf
  #   models--depth-anything--Depth-Anything-V2-Large-hf
  #   models--openai--clip-vit-large-patch14-336   (used by some baselines)
fi

note "AFTER"
df -h /home / 2>/dev/null | head -5

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "DRY_RUN=1 — nothing was deleted. Set DRY_RUN=0 and re-run to actually delete."
fi
